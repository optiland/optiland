"""Finite-Difference Evaluator (numpy path)

Computes gradients and Jacobians by looping ``n+1`` forward-difference
evaluations (one baseline + one perturbation per variable). Each single
evaluation still goes through BatchedRayEvaluator, so the per-evaluation cost
is already operand-batched.

The torch autograd evaluator is the fast path; this evaluator is
the honest CPU path. It is not slow in practice for the small/dense variable
counts of classical lens design.

Non-finite evaluations propagate as NaN — they are *not* replaced with
1e10. The native LM controller treats NaN as a rejected step.

Kramer Harrison, 2026
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

import optiland.backend as be
from optiland.optimization.failure import (
    OptimizationFailure,
    normalize_failure_mode,
    penalty_merit,
    penalty_residual,
)
from optiland.optimization.state import EvalCapability

if TYPE_CHECKING:
    from optiland.optimization.problem import OptimizationProblem

_PROVIDES = frozenset(
    {
        EvalCapability.VALUE,
        EvalCapability.RESIDUALS,
        EvalCapability.GRADIENT,
        EvalCapability.JACOBIAN,
    }
)


class FiniteDiffEvaluator:
    """Evaluator for the numpy backend using forward finite differences.

    Args:
        problem: The optimization problem to wrap.
        rel_step: Relative perturbation size (default 1e-5).
        abs_step: Absolute floor for perturbation size (default 1e-8).
        scheme: ``"forward"`` (default) or ``"central"`` differences.
        on_failure: Non-finite / trace-failure policy (default ``"reject"``).
            ``"reject"`` returns a non-finite merit / ``NaN`` residual so the
            stepped controller rejects the trial; ``"raise"`` raises
            :class:`~optiland.optimization.errors.OptimizationFailure`;
            ``"penalty"`` returns a finite, scale-proportional penalty.
    """

    def __init__(
        self,
        problem: OptimizationProblem,
        rel_step: float = 1e-5,
        abs_step: float = 1e-8,
        scheme: str = "forward",
        on_failure: str = "reject",
    ):
        assert not any(isinstance(var.value, str) for var in problem.variables), (
            "Glass/material variables are not supported by FiniteDiffEvaluator. "
            "Use GlassExpert directly."
        )
        self.problem = problem
        self.backend: str = "numpy"
        self.n_vars: int = len(list(problem.variables))
        self.provides: frozenset[EvalCapability] = _PROVIDES
        self._rel = rel_step
        self._abs = abs_step
        self._scheme = scheme
        self._on_failure = normalize_failure_mode(on_failure)
        # Reference merit for the scale-proportional penalty (last finite value).
        self._ref_merit: float = 1.0
        # Last known residual length, for sizing a penalty residual vector.
        self._last_m: int = max(len(list(problem.operands)), 1)

    # ------------------------------------------------------------------
    # Core read / write
    # ------------------------------------------------------------------

    def read_x(self) -> np.ndarray:
        """Return current scaled variable values as a numpy array."""
        return np.array(
            [float(be.to_numpy(var.value)) for var in self.problem.variables]
        )

    def write_x(self, x: Any) -> None:
        """Write ``x`` into variables and update optics."""
        self.problem.set_variable_vector(x)

    # ------------------------------------------------------------------
    # Evaluation primitives
    # ------------------------------------------------------------------

    def value(self, x: Any) -> float:
        """Evaluate merit at ``x``; returns a Python float.

        Honors the configured ``on_failure`` policy: a ray-trace exception or
        a non-finite merit becomes ``NaN`` (``"reject"``), an
        :class:`OptimizationFailure` (``"raise"``), or a finite penalty
        (``"penalty"``).
        """
        self.write_x(x)
        try:
            v = self.problem.sum_squared()
            result = float(be.to_numpy(v))
        except Exception as exc:  # noqa: BLE001
            return self._handle_value_failure(exc)
        if not math.isfinite(result):
            return self._handle_value_failure(None)
        self._ref_merit = result
        return result

    def residuals(self, x: Any) -> np.ndarray:
        """Return ``weighted_residuals()`` at ``x`` as a numpy array.

        Honors the configured ``on_failure`` policy (see :meth:`value`).
        """
        self.write_x(x)
        try:
            r = self.problem.weighted_residuals()
            r_np = np.asarray(be.to_numpy(r), dtype=float)
        except Exception as exc:  # noqa: BLE001
            return self._handle_residual_failure(exc)
        if not np.all(np.isfinite(r_np)):
            return self._handle_residual_failure(None)
        self._last_m = r_np.shape[0] if r_np.ndim else self._last_m
        return r_np

    # ------------------------------------------------------------------
    # Failure policy
    # ------------------------------------------------------------------

    def _handle_value_failure(self, exc: Exception | None) -> float:
        if self._on_failure == "raise":
            raise OptimizationFailure(
                "Merit evaluation failed (non-finite or ray-trace error)."
            ) from exc
        if self._on_failure == "penalty":
            return penalty_merit(self._ref_merit)
        return float("nan")

    def _handle_residual_failure(self, exc: Exception | None) -> np.ndarray:
        if self._on_failure == "raise":
            raise OptimizationFailure(
                "Residual evaluation failed (non-finite or ray-trace error)."
            ) from exc
        m = max(self._last_m, 1)
        if self._on_failure == "penalty":
            return np.full(m, penalty_residual(self._ref_merit, m))
        return np.full(m, np.nan)

    # ------------------------------------------------------------------
    # Finite-difference gradient / Jacobian
    # ------------------------------------------------------------------

    def _h(self, xi: float) -> float:
        return self._rel * abs(xi) + self._abs

    def gradient(self, x: Any) -> np.ndarray:
        """Forward (or central) finite-difference gradient of the merit.

        Loops ``n+1`` (forward) or ``2n`` (central) evaluations.
        """
        x = np.asarray(x, dtype=float)
        if self._scheme == "central":
            return self._gradient_central(x)
        return self._gradient_forward(x)

    def _gradient_forward(self, x: np.ndarray) -> np.ndarray:
        f0 = self.value(x)
        grad = np.empty(self.n_vars)
        for i in range(self.n_vars):
            hi = self._h(x[i])
            xp = x.copy()
            xp[i] += hi
            grad[i] = (self.value(xp) - f0) / hi
        self.write_x(x)  # restore
        return grad

    def _gradient_central(self, x: np.ndarray) -> np.ndarray:
        grad = np.empty(self.n_vars)
        for i in range(self.n_vars):
            hi = self._h(x[i])
            xp, xm = x.copy(), x.copy()
            xp[i] += hi
            xm[i] -= hi
            grad[i] = (self.value(xp) - self.value(xm)) / (2 * hi)
        self.write_x(x)
        return grad

    def jacobian(self, x: Any, r0: Any = None) -> np.ndarray:
        """Forward finite-difference Jacobian of ``weighted_residuals()``.

        Args:
            x: Point at which to evaluate the Jacobian.
            r0: Pre-computed baseline residual ``residuals(x)``.  When provided
                (e.g. by ``LevenbergMarquardt.step``), the baseline evaluation
                is skipped, saving one full residual evaluation per accepted
                step (§3.8).

        Returns:
            An ``(m, n)`` numpy array.
        """
        x = np.asarray(x, dtype=float)
        r0 = self.residuals(x) if r0 is None else np.asarray(r0, dtype=float)
        m = r0.shape[0]
        J = np.empty((m, self.n_vars))
        for i in range(self.n_vars):
            hi = self._h(x[i])
            xp = x.copy()
            xp[i] += hi
            ri = self.residuals(xp)
            J[:, i] = (ri - r0) / hi
        self.write_x(x)
        return J
