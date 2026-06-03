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

from typing import TYPE_CHECKING, Any

import numpy as np

import optiland.backend as be
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
    """

    def __init__(
        self,
        problem: OptimizationProblem,
        rel_step: float = 1e-5,
        abs_step: float = 1e-8,
        scheme: str = "forward",
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
        """Evaluate merit at ``x``; returns a Python float."""
        self.write_x(x)
        v = self.problem.sum_squared()
        result = float(be.to_numpy(v))
        return result

    def residuals(self, x: Any) -> np.ndarray:
        """Return ``weighted_residuals()`` at ``x`` as a numpy array."""
        self.write_x(x)
        r = self.problem.weighted_residuals()
        return be.to_numpy(r)

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

    def jacobian(self, x: Any) -> np.ndarray:
        """Forward finite-difference Jacobian of ``weighted_residuals()``.

        Returns an ``(m, n)`` numpy array.
        """
        x = np.asarray(x, dtype=float)
        r0 = self.residuals(x)
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
