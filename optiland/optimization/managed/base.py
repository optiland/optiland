"""Shared plumbing of the managed (SciPy) cores.

:class:`ProblemEvaluation` evaluates the problem for SciPy under the failure
policy; :class:`RunMonitor` drives observers and stopping criteria from a SciPy
callback. Private: reached through ``minimize()``.

Kramer Harrison, 2026
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

import optiland.backend as be
from optiland import _epoch
from optiland.optimization.failure import (
    ConfigurationError,
    OptimizationFailure,
    describe_rows,
    failing_rows,
    normalize_failure_mode,
    penalty_merit,
    penalty_residual,
)
from optiland.optimization.observers.cancel import with_cancel
from optiland.optimization.observers.history import find_history
from optiland.optimization.state import OptimizationResult, OptimizationState

if TYPE_CHECKING:
    from optiland.optimization.observers.base import Observer
    from optiland.optimization.observers.cancel import CancelToken
    from optiland.optimization.problem import OptimizationProblem
    from optiland.optimization.stopping.base import StoppingCriterion


class ProblemEvaluation:
    """Residuals and merit of a problem at ``x``, for SciPy.

    Residuals are ``effective_weight * delta`` (decision V7). SciPy cannot
    consume ``NaN``: a failed evaluation becomes a scale-aware penalty (the
    PR1.1 rule), or raises :class:`OptimizationFailure` with
    ``on_failure="raise"``.

    Args:
        problem: The optimization problem.
        on_failure: ``"reject"`` (penalty) or ``"raise"``.

    Attributes:
        nfev: Evaluations so far.
        best_x: The variable vector with the lowest finite merit, or None.
        best_merit: That merit.
    """

    def __init__(self, problem: OptimizationProblem, on_failure: str = "reject"):
        if any(isinstance(var.value, str) for var in problem.variables):
            raise ConfigurationError(
                "Glass/material variables are not supported here; use GlassExpert."
            )
        self.problem = problem
        self.on_failure = normalize_failure_mode(on_failure)
        self.nfev = 0
        self.best_x: np.ndarray | None = None
        self.best_merit = math.inf
        self._optics = tuple({id(v.optic): v.optic for v in problem.variables}.values())
        self._n_rows = max(len(problem.operands), 1)
        self._ref_norm = 0.0
        self._recent: dict[bytes, float] = {}

    def read_x(self) -> np.ndarray:
        """Current scaled variable values."""
        return np.array([float(be.to_numpy(v.value)) for v in self.problem.variables])

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper bounds of the scaled variables (``±inf`` if unset)."""
        n = len(self.problem.variables)
        lower, upper = np.full(n, -np.inf), np.full(n, np.inf)
        for i, var in enumerate(self.problem.variables):
            lo, hi = var.bounds
            if lo is not None:
                lower[i] = float(be.to_numpy(lo))
            if hi is not None:
                upper[i] = float(be.to_numpy(hi))
        return lower, upper

    def write_x(self, x: Any) -> None:
        """Write ``x`` into the variables and update the optics."""
        for var, value in zip(self.problem.variables, x, strict=True):
            var.update(float(value))
        self.problem.update_optics()

    def _raw(self, x: np.ndarray) -> np.ndarray | None:
        """Residual vector at ``x``; None when the whole evaluation failed."""
        self.nfev += 1
        try:
            self.write_x(x)
            with _epoch.evaluation_epoch(self._optics):
                return np.asarray(
                    be.to_numpy(self.problem.residual_vector()), dtype=float
                ).reshape(-1)
        except _epoch.EpochViolationError:
            raise  # a contract breach is a bug, not a bad design
        except Exception as exc:
            if self.on_failure == "raise":
                raise OptimizationFailure(
                    "Design evaluation failed (ray-trace error)."
                ) from exc
            return None

    def _record(self, x: np.ndarray, merit: float) -> None:
        self._ref_norm = math.sqrt(merit)
        self._recent[np.asarray(x, dtype=float).tobytes()] = merit
        if len(self._recent) > 8:
            self._recent.pop(next(iter(self._recent)))
        if merit < self.best_merit:
            self.best_merit = merit
            self.best_x = np.array(x, dtype=float)

    def check_start(self, x: np.ndarray) -> float:
        """Evaluate the starting point; refuse an untraceable design.

        Returns:
            The starting merit.

        Raises:
            ConfigurationError: If any residual at ``x`` is non-finite.
        """
        r = self._raw(x)
        rows = failing_rows(r) if r is not None else np.arange(self._n_rows)
        if r is None or rows.size:
            raise ConfigurationError(
                "Cannot optimize from an untraceable design: the starting point "
                "has non-finite residuals for operand(s) "
                f"{describe_rows(self.problem, rows)}."
            )
        merit = float(r @ r)
        self._record(x, merit)
        return merit

    def merit(self, x: Any) -> float:
        """Scalar merit ``sum(r ** 2)`` at ``x`` (penalty on failure)."""
        x = np.asarray(x, dtype=float)
        r = self._raw(x)
        if r is None or not np.all(np.isfinite(r)):
            if self.on_failure == "raise":
                raise OptimizationFailure("Non-finite merit.")
            return penalty_merit(self._ref_norm)
        merit = float(r @ r)
        self._record(x, merit)
        return merit

    def residuals(self, x: Any) -> np.ndarray:
        """Residual vector at ``x``; failed rows become a penalty."""
        x = np.asarray(x, dtype=float)
        r = self._raw(x)
        if r is None:
            return np.full(self._n_rows, penalty_residual(self._ref_norm, self._n_rows))
        finite = np.isfinite(r)
        if finite.all():
            self._record(x, float(r @ r))
            return r
        if self.on_failure == "raise":
            raise OptimizationFailure("Non-finite residuals.")
        return np.where(finite, r, penalty_residual(self._ref_norm, r.size))

    def merit_at(self, x: Any) -> float:
        """Merit already computed at ``x``, else the best merit so far."""
        key = np.asarray(x, dtype=float).tobytes()
        return self._recent.get(key, self.best_merit)


class RunMonitor:
    """Drives observers and stopping criteria of a managed run.

    Args:
        observers: Observers of the run.
        stop: Extra stopping criterion.
        cancel_token: Cancels the run at the next callback.
    """

    def __init__(
        self,
        observers: list[Observer] | None,
        stop: StoppingCriterion | None,
        cancel_token: CancelToken | None,
    ) -> None:
        self.observers = list(observers or [])
        self.criterion = with_cancel(stop, cancel_token)
        self.stop_reason: str | None = None
        self.state: OptimizationState | None = None

    def start(self, x: np.ndarray, value: float, n_evals: int) -> None:
        """Create the state and fire ``on_start``."""
        self.state = OptimizationState(x=x.copy(), value=value, n_evals=n_evals)
        for observer in self.observers:
            observer.on_start(self.state)
        if self.criterion is not None:
            self.criterion.reset(self.state)

    def step(self, x: np.ndarray, value: float, n_evals: int) -> None:
        """Record an accepted step; raise ``StopIteration`` to end the run."""
        state = self.state
        previous = state.x
        state.step_norm = float(np.linalg.norm(np.asarray(x) - previous))
        state.x, state.value = np.array(x, dtype=float), float(value)
        state.iteration += 1
        state.n_evals = n_evals
        try:
            for observer in self.observers:
                observer.on_step(state)
        except StopIteration:
            self.stop_reason = "stopped by an observer"
            raise
        if self.criterion is not None:
            stopped, reason = self.criterion.should_stop(state)
            if stopped:
                self.stop_reason = reason or "stopped by a stopping criterion"
                raise StopIteration(self.stop_reason)

    def finish(self, result: OptimizationResult) -> OptimizationResult:
        """Attach history, fire ``on_end`` and return the result."""
        state = self.state
        state.x, state.value = np.array(result.x, dtype=float), float(result.fun)
        state.n_evals, state.stop_reason = result.n_evals, result.stop_reason
        result.history = find_history(self.observers)
        for observer in self.observers:
            observer.on_end(state, result)
        return result
