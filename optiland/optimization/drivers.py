"""Optimization Drivers

Two driver families (D1):

* ``SteppedDriver``: owns the loop; calls ``optimizer.step()`` once per
  accepted iteration.  Used for torch and (Phase 2) native LM.
* ``ManagedDriver``: delegates the loop to an existing SciPy adapter.
  Threads observers through SciPy's ``callback=`` argument.

Both drivers return ``OptimizationResult`` and fire the same observer hooks.

Kramer Harrison, 2026
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.optimization.state import OptimizationResult, OptimizationState

if TYPE_CHECKING:
    from optiland.optimization.constraints.base import ConstraintStrategy
    from optiland.optimization.control.base import StepController
    from optiland.optimization.evaluators.base import Evaluator
    from optiland.optimization.native.base import SteppedOptimizer
    from optiland.optimization.observers.base import Observer
    from optiland.optimization.stopping.base import StoppingCriterion


def _to_float(v: Any) -> float:
    if hasattr(v, "item"):
        return float(v.item())
    return float(v)


def _build_result(
    state: OptimizationState,
    initial_value: float,
    wall_time: float,
    backend: str,
    method: str,
    history: list | None,
    resolved_from: str | None = None,
) -> OptimizationResult:
    final_value = _to_float(state.value)
    denom = max(abs(initial_value), 1e-12)
    improvement = (initial_value - final_value) / denom * 100.0
    import contextlib

    x_final = state.x
    with contextlib.suppress(Exception):
        x_final = be.to_numpy(state.x)
    return OptimizationResult(
        x=x_final,
        success=not (state.stop_reason or "").startswith("max_iter"),
        status=state.stop_reason or "completed",
        value=final_value,
        initial_value=initial_value,
        improvement_pct=improvement,
        iterations=state.iteration,
        n_value_evals=state.n_value_evals,
        n_jac_evals=state.n_jac_evals,
        wall_time_s=wall_time,
        history=history,
        backend=backend,
        method=method,
        resolved_from=resolved_from,
    )


# ---------------------------------------------------------------------------
# Stepped Driver
# ---------------------------------------------------------------------------


class SteppedDriver:
    """Runs a stepped optimizer (torch / Phase-2 native LM) until a stopping
    criterion fires or the optimizer signals intrinsic convergence.

    The loop is intentionally minimal (~20 lines of real logic).  Optimizers
    never log; observers never compute steps.
    """

    def run(
        self,
        optimizer: SteppedOptimizer,
        evaluator: Evaluator,
        x0: Any,
        *,
        controller: StepController,
        constraints: ConstraintStrategy | None,
        criteria: StoppingCriterion,
        observers: list[Observer],
        initial_value: float,
        method: str,
        resolved_from: str | None = None,
    ) -> OptimizationResult:
        t0 = time.monotonic()

        state = optimizer.initialize(
            evaluator, x0, controller=controller, constraints=constraints
        )
        criteria.reset(state)

        for obs in observers:
            obs.on_start(state)

        while True:
            if optimizer.converged(state):
                state.stop_reason = "intrinsic_convergence"
                state.converged = True
                break
            stop, reason = criteria.should_stop(state)
            if stop or state.converged:
                state.stop_reason = reason or state.stop_reason
                break
            state = optimizer.step(state)
            for obs in observers:
                obs.on_step(state)

        history = _collect_history(observers)
        result = _build_result(
            state,
            initial_value,
            time.monotonic() - t0,
            evaluator.backend,
            method,
            history,
            resolved_from=resolved_from,
        )
        for obs in observers:
            obs.on_end(state, result)

        return result


# ---------------------------------------------------------------------------
# Managed Driver
# ---------------------------------------------------------------------------


class ManagedDriver:
    """Delegates the optimization loop to a SciPy adapter.

    Threads an observer/criterion callback through the SciPy ``callback=``
    argument.  Fires ``on_start`` / ``on_step`` / ``on_end`` on all observers.

    Early stop is requested via ``StopIteration`` in the callback (modern
    SciPy convention, supported by ``minimize`` and some global solvers).
    """

    def run(
        self,
        adapter: Any,
        evaluator: Evaluator,
        *,
        criteria: StoppingCriterion,
        observers: list[Observer],
        initial_value: float,
        method: str,
        method_options: dict,
        resolved_from: str | None = None,
    ) -> OptimizationResult:
        t0 = time.monotonic()
        problem = evaluator.problem

        iteration_counter = [0]
        n_value_evals = [0]

        # Initial state for on_start
        x0 = evaluator.read_x()
        init_state = OptimizationState(
            x=x0,
            value=initial_value,
            iteration=0,
            n_value_evals=0,
        )
        criteria.reset(init_state)
        for obs in observers:
            obs.on_start(init_state)

        final_scipy_result: Any = None

        def _callback(xk: Any) -> None:
            iteration_counter[0] += 1
            n_value_evals[0] += 1
            try:
                val = float(be.to_numpy(problem.sum_squared()))
            except Exception:  # noqa: BLE001
                val = float("nan")
            state = OptimizationState(
                x=xk,
                value=val,
                iteration=iteration_counter[0],
                n_value_evals=n_value_evals[0],
            )
            for obs in observers:
                obs.on_step(state)
            stop, reason = criteria.should_stop(state)
            if stop or state.converged:
                raise StopIteration(reason or "criteria")

        try:
            final_scipy_result = adapter.run(callback=_callback, **method_options)
        except StopIteration:
            pass
        except Exception:
            raise

        # Final state after the adapter has written back result.x
        x_final = evaluator.read_x()
        try:
            final_val = float(be.to_numpy(problem.sum_squared()))
        except Exception:  # noqa: BLE001
            final_val = initial_value

        stop_reason = "completed"
        if final_scipy_result is not None:
            msg = getattr(final_scipy_result, "message", None)
            if msg:
                stop_reason = str(msg)

        # Extract actual iterations and evaluations from SciPy's result if available
        nfev_scipy = n_value_evals[0]
        njev_scipy = 0
        iterations_scipy = iteration_counter[0]

        if final_scipy_result is not None:
            nfev_scipy = getattr(final_scipy_result, "nfev", nfev_scipy)
            njev_scipy = getattr(final_scipy_result, "njev", njev_scipy)
            iterations_scipy = getattr(final_scipy_result, "nit", iterations_scipy)
            is_ls = method == "least_squares"
            if (iterations_scipy is None or iterations_scipy == 0) and is_ls:
                has_j = njev_scipy is not None and njev_scipy > 0
                iterations_scipy = njev_scipy if has_j else nfev_scipy

        # Ensure they are integers and not None
        iterations_scipy = int(iterations_scipy) if iterations_scipy is not None else 0
        nfev_scipy = int(nfev_scipy) if nfev_scipy is not None else 0
        njev_scipy = int(njev_scipy) if njev_scipy is not None else 0

        final_state = OptimizationState(
            x=x_final,
            value=final_val,
            iteration=iterations_scipy,
            n_value_evals=nfev_scipy,
            n_jac_evals=njev_scipy,
            stop_reason=stop_reason,
        )

        history = _collect_history(observers)
        result = _build_result(
            final_state,
            initial_value,
            time.monotonic() - t0,
            evaluator.backend,
            method,
            history,
            resolved_from=resolved_from,
        )
        if final_scipy_result is not None:
            result_success = getattr(final_scipy_result, "success", True)
        else:
            result_success = True
        object.__setattr__(result, "success", bool(result_success))

        for obs in observers:
            obs.on_end(final_state, result)

        return result


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _collect_history(observers: list[Observer]) -> list | None:
    from optiland.optimization.observers.history import HistoryObserver

    for obs in observers:
        if isinstance(obs, HistoryObserver):
            return list(obs.records)
    return None
