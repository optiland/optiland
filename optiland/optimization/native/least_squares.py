"""Native Levenberg-Marquardt and Gauss-Newton optimizers.

Both work with any backend via the Evaluator protocol:

- **numpy path**: ``FiniteDiffEvaluator`` + ``LevenbergController``
- **torch path**: ``AutogradEvaluator`` + ``LevenbergController``

``LevenbergMarquardt`` delegates the full inner λ-search to the step
controller; one call to ``step()`` = one *accepted* step.

``GaussNewton`` is the undamped special case (λ → 0): it solves
``(JᵀJ)Δ = −Jᵀr`` each step without a trust-region loop, and relies on
external stopping criteria (e.g. ``StepStall``) for termination.

Kramer Harrison, 2026
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from optiland.optimization.state import Capability, OptimizationState

from .base import SteppedOptimizer

if TYPE_CHECKING:
    from optiland.optimization.constraints.base import ConstraintStrategy
    from optiland.optimization.control.base import StepController
    from optiland.optimization.evaluators.base import Evaluator


_CAPS: frozenset[Capability] = frozenset(
    {
        Capability.OBSERVE,
        Capability.STEP_CONTROL,
        Capability.MUTATE_TERMINATION,
        Capability.PER_STEP_CONSTRAINTS,
        Capability.KKT_CONSTRAINTS,
        Capability.BOUNDS,
    }
)

# Module-level guard: emit the weight-consistency notice at most once per process.
_weight_notice_emitted: bool = False


class LevenbergMarquardt(SteppedOptimizer):
    """Native Levenberg-Marquardt (damped least-squares) optimizer.

    Works with both numpy (``FiniteDiffEvaluator``) and torch
    (``AutogradEvaluator``) backends via the ``Evaluator`` protocol.

    The step controller owns the damping strategy and the inner λ-search.
    The default controller — ``LevenbergController`` with Moré diagonal
    damping — is wired by ``api.minimize()`` when ``method="dls"`` or
    ``method="lm"``.

    Args:
        emit_weight_notice: Emit a one-time ``UserWarning`` on the first run
            where ``effective_weight ≠ operand.weight``.
    """

    capabilities: frozenset[Capability] = _CAPS
    requires_backend: str | None = None

    def __init__(self, emit_weight_notice: bool = True) -> None:
        self._emit_weight_notice = emit_weight_notice
        self._evaluator: Any = None
        self._controller: Any = None
        self._constraints: Any = None

    # ------------------------------------------------------------------
    # SteppedOptimizer interface
    # ------------------------------------------------------------------

    def initialize(
        self,
        evaluator: Evaluator,
        x0: Any,
        *,
        controller: StepController,
        constraints: ConstraintStrategy | None,
    ) -> OptimizationState:
        """Capture collaborators, evaluate the initial point, seed λ.

        Args:
            evaluator: Evaluator providing residuals / jacobian / value.
            x0: Initial parameter vector (may be overridden by
                ``evaluator.read_x()``).
            controller: Step controller (typically ``LevenbergController``).
            constraints: Active constraint strategy, or None.

        Returns:
            An ``OptimizationState`` at the initial point.
        """
        self._evaluator = evaluator
        self._controller = controller
        self._constraints = constraints

        # Hand the constrained KKT controller its dependencies (evaluator +
        # the problem's hard-constraint manager) so it can assemble the KKT
        # system each step.
        if hasattr(controller, "bind"):
            controller.bind(evaluator, evaluator.problem.constraints)

        import optiland.backend as be

        x0_raw = evaluator.read_x()
        x0_np = np.asarray(be.to_numpy(x0_raw), dtype=float)
        _maybe_emit_weight_notice(evaluator, self._emit_weight_notice)

        r0 = evaluator.residuals(x0_np)
        r0_np = np.asarray(be.to_numpy(r0), dtype=float)
        val0 = float(np.dot(r0_np, r0_np))

        state = OptimizationState(
            x=x0_np,
            value=val0,
            residuals=r0,
            iteration=0,
            n_value_evals=1,
        )
        controller.reset(state)
        return state

    def step(self, state: OptimizationState) -> OptimizationState:
        """Execute one accepted LM step (inner λ-search is hidden in the controller).

        Computes the Jacobian once, delegates the full accept/reject λ-search
        to the controller, applies bounds, and updates state in place.

        Args:
            state: Current mutable state.

        Returns:
            The mutated ``state`` after one accepted move.
        """
        import optiland.backend as be
        from optiland.optimization.control.base import StepInfo

        x = np.asarray(state.x, dtype=float)

        # One residual + Jacobian computation per accepted step.  The known
        # baseline ``r`` is threaded into ``jacobian`` so the FD evaluator does
        # not re-evaluate it internally (§3.8).
        r = self._evaluator.residuals(x)
        J = self._evaluator.jacobian(x, r0=r)  # restores system to x
        state.n_jac_evals += 1

        info = StepInfo(
            direction=None,
            residuals=r,
            jacobian=J,
            value_fn=self._evaluator.value,
        )

        outcome = self._controller.transform(None, info, state)

        n_trials: int = outcome.info.get("n_trials", 0)
        state.n_trial_evals += n_trials
        state.n_value_evals += n_trials

        if not outcome.accepted:
            # λ too large; restore system to current x and signal intrinsic convergence
            self._evaluator.write_x(x)
            state.converged = True
            state.stop_reason = "lm_lambda_max: all trials rejected"
            return state

        dx = np.asarray(be.to_numpy(outcome.delta_x), dtype=float)
        x_raw = x + dx
        x_new = x_raw

        # Apply box bounds (or other apply_to_step constraint) if present
        if self._constraints is not None and hasattr(
            self._constraints, "apply_to_step"
        ):
            x_new = np.asarray(
                be.to_numpy(self._constraints.apply_to_step(x_new, state)), dtype=float
            )

        # Effective step (post-constraint) drives the step-norm stopping test.
        state.step_norm = float(np.linalg.norm(x_new - x))
        state.x = x_new
        state.residuals = r
        state.jacobian = J
        state.iteration += 1

        if not np.array_equal(x_new, x_raw):
            # Bounds / apply_to_step altered the step: the controller's last
            # value_fn call left the optic at the *unconstrained* trial point,
            # so re-sync the live optic and the merit to the accepted x_new.
            self._evaluator.write_x(x_new)
            state.value = self._evaluator.value(x_new)
            state.n_value_evals += 1
        else:
            # System is already at x_new from the last value_fn call in the
            # controller; reuse the stored trial value to avoid an extra eval.
            state.value = outcome.info.get("trial_value", self._evaluator.value(x_new))
        return state

    def converged(self, state: OptimizationState) -> bool:
        """Return True when the λ-search exhausts all trials (λ blow-up).

        External stopping criteria (MaxIter, CostTolerance, …) are handled
        by the driver; this signals *intrinsic* LM convergence only.
        """
        return bool(state.converged)


class GaussNewton(SteppedOptimizer):
    """Undamped Gauss-Newton optimizer (λ → 0 special case of LM).

    Solves ``(JᵀJ)Δ = −Jᵀr`` each step and applies the result without a
    trust-region loop.  Terminates via external criteria (``StepStall``,
    ``CostTolerance``) or when ``JᵀJ`` becomes singular (intrinsic).

    For ill-conditioned or underdetermined problems, prefer
    ``LevenbergMarquardt``.
    """

    capabilities: frozenset[Capability] = _CAPS
    requires_backend: str | None = None

    def __init__(self, min_diag: float = 1e-8, fallback_lambda: float = 1e-2) -> None:
        """Args:
        min_diag: Floor on ``diag(JᵀJ)`` used by the damped fallback step.
        fallback_lambda: Damping factor for the LM fallback step taken when
            the line search cannot find a decrease.
        """
        from optiland.optimization.control.line_search import LineSearchController

        self._evaluator: Any = None
        self._constraints: Any = None
        self._min_diag = min_diag
        self._fallback_lambda = fallback_lambda
        self._line_search = LineSearchController()

    def initialize(
        self,
        evaluator: Evaluator,
        x0: Any,
        *,
        controller: StepController,  # noqa: ARG002  (unused by GN)
        constraints: ConstraintStrategy | None,
    ) -> OptimizationState:
        """Capture evaluator/constraints and return the initial state.

        Args:
            evaluator: Evaluator providing residuals / jacobian / value.
            x0: Unused; initial x is read from ``evaluator.read_x()``.
            controller: Unused by Gauss-Newton (no damping loop).
            constraints: Active constraint strategy, or None.

        Returns:
            An ``OptimizationState`` at the initial point.
        """
        import optiland.backend as be

        self._evaluator = evaluator
        self._constraints = constraints

        x0_np = np.asarray(be.to_numpy(evaluator.read_x()), dtype=float)
        r0 = np.asarray(be.to_numpy(evaluator.residuals(x0_np)), dtype=float)
        val0 = float(np.dot(r0, r0))

        return OptimizationState(
            x=x0_np,
            value=val0,
            residuals=r0,
            iteration=0,
            n_value_evals=1,
        )

    def step(self, state: OptimizationState) -> OptimizationState:
        """Execute one Gauss-Newton step with an Armijo descent guard.

        The undamped GN direction is computed first.  If the full step does not
        decrease the merit, an Armijo backtracking line search is run; if that
        also fails to find a decrease, a single damped (LM) fallback step is
        taken.  All arrays are routed through ``be.to_numpy`` so the linear
        algebra is correct on CUDA/torch backends.

        Args:
            state: Current mutable state.

        Returns:
            The mutated ``state`` after one GN move, or with
            ``converged=True`` if ``JᵀJ`` is singular.
        """
        import optiland.backend as be
        from optiland.optimization.control.base import StepInfo

        x = np.asarray(be.to_numpy(state.x), dtype=float)

        r = np.asarray(be.to_numpy(self._evaluator.residuals(x)), dtype=float)
        J = np.asarray(be.to_numpy(self._evaluator.jacobian(x, r0=r)), dtype=float)
        state.n_jac_evals += 1

        JTJ: np.ndarray = J.T @ J
        JTr: np.ndarray = J.T @ r

        try:
            dx = np.linalg.solve(JTJ, -JTr)
        except np.linalg.LinAlgError:
            self._evaluator.write_x(x)
            state.converged = True
            state.stop_reason = "gauss_newton_singular: JᵀJ is singular"
            return state

        f0 = _to_float(state.value)
        x_new = x + dx
        f_new = self._evaluator.value(x_new)
        n_eval = 1

        # Descent guard: the undamped GN step is only safe near a solution.
        if (not math.isfinite(f_new)) or f_new > f0:
            # ∇(Σrᵢ²) = 2·Jᵀr — the slope used by the Armijo test.
            info = StepInfo(
                direction=dx,
                gradient=2.0 * JTr,
                value_fn=self._evaluator.value,
            )
            outcome = self._line_search.transform(dx, info, state)
            n_eval += outcome.info.get("n_trials", 0)
            if outcome.accepted:
                dx = np.asarray(be.to_numpy(outcome.delta_x), dtype=float)
                x_new = x + dx
                f_new = outcome.info.get("trial_value", self._evaluator.value(x_new))
            else:
                # Persistent increase: fall back to a damped (LM) step.
                dx = self._damped_step(JTJ, JTr)
                x_new = x + dx
                f_new = self._evaluator.value(x_new)
                n_eval += 1

        x_raw = x_new
        if self._constraints is not None and hasattr(
            self._constraints, "apply_to_step"
        ):
            x_new = np.asarray(
                be.to_numpy(self._constraints.apply_to_step(x_new, state)), dtype=float
            )

        state.n_value_evals += n_eval
        state.n_trial_evals += n_eval
        state.step_norm = float(np.linalg.norm(x_new - x))
        state.x = x_new
        state.residuals = r
        state.jacobian = J
        state.iteration += 1

        if not np.array_equal(x_new, x_raw):
            # Constraint altered the accepted point: re-sync optic + merit.
            self._evaluator.write_x(x_new)
            state.value = self._evaluator.value(x_new)
            state.n_value_evals += 1
        else:
            state.value = f_new
        return state

    def _damped_step(self, JTJ: np.ndarray, JTr: np.ndarray) -> np.ndarray:
        """Solve a Moré-damped LM system as a Gauss-Newton fallback."""
        d = np.maximum(np.abs(np.diag(JTJ)), self._min_diag)
        A = JTJ + self._fallback_lambda * np.diag(d)
        try:
            return np.linalg.solve(A, -JTr)
        except np.linalg.LinAlgError:
            return np.zeros_like(JTr)

    def converged(self, state: OptimizationState) -> bool:
        return bool(state.converged)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_float(v: Any) -> float:
    """Coerce a backend scalar / tensor to a Python float."""
    if hasattr(v, "item"):
        try:
            return float(v.item())
        except Exception:  # noqa: BLE001
            pass
    import optiland.backend as be

    return float(be.to_numpy(v))


def _maybe_emit_weight_notice(evaluator: Any, enabled: bool) -> None:
    """Emit a one-time notice when non-trivial field/wavelength weights exist."""
    global _weight_notice_emitted
    if not enabled or _weight_notice_emitted:
        return
    try:
        problem = evaluator.problem
        for op in problem.operands:
            eff = getattr(op, "effective_weight", None)
            w = getattr(op, "weight", 1.0)
            if eff is not None and not math.isclose(float(eff), float(w), rel_tol=1e-9):
                warnings.warn(
                    "Optimization problem has non-unity field/wavelength weights "
                    "(effective_weight ≠ operand.weight).  The native LM optimizer "
                    "minimizes Σ(sqrt(effective_weight)·δ)² which exactly equals "
                    "sum_squared() (correctness fix).  Results may differ from "
                    "pre-Phase-1b runs of LeastSquares on multi-field / "
                    "multi-wavelength systems.",
                    UserWarning,
                    stacklevel=4,
                )
                _weight_notice_emitted = True
                break
    except Exception:  # noqa: BLE001
        pass
