"""ScipyLeastSquaresAdapter — private core for scipy.optimize.least_squares.

Calls scipy directly without going through the deprecated ``LeastSquares``
class — no DeprecationWarning is triggered from the facade path.

Note: ``scipy.optimize.least_squares`` has no ``callback`` argument, so
``on_step`` observers are not fired during the run.  ``on_start`` and
``on_end`` still fire normally.

Kramer Harrison, 2026
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.optimization.failure import (
    OptimizationFailure,
    normalize_failure_mode,
    penalty_residual,
)
from optiland.optimization.state import Capability

if TYPE_CHECKING:
    from optiland.optimization.problem import OptimizationProblem

_CAPS = frozenset(
    {
        Capability.OBSERVE,
        Capability.SCIPY_CONSTRAINTS,
        Capability.BOUNDS,
    }
)


class ScipyLeastSquaresAdapter:
    """Private-core adapter for ``scipy.optimize.least_squares``.

    Calls scipy directly — does not instantiate any deprecated legacy class.

    Args:
        problem: The optimization problem.
        method_choice: SciPy least_squares method (``"lm"``, ``"trf"``,
            ``"dogbox"``).  Default ``"lm"``.
    """

    capabilities: frozenset[Capability] = _CAPS
    requires_backend: str | None = None
    method_name: str = "least_squares"

    def __init__(
        self,
        problem: OptimizationProblem,
        method_choice: str = "lm",
        on_failure: str = "penalty",
    ):
        self.problem = problem
        self.method_choice = method_choice
        self._on_failure = normalize_failure_mode(on_failure)
        # Reference merit for the scale-proportional penalty (last finite SSR).
        self._ref_merit: float = 1.0
        self._last_m: int = max(len(list(problem.operands)), 1)

    def run(
        self,
        *,
        callback: Any = None,  # noqa: ARG002 — least_squares has no callback
        maxiter: int | None = None,
        tol: float = 1e-3,
        on_failure: str | None = None,
        **_: Any,
    ) -> Any:
        """Run scipy.optimize.least_squares and write back optimal params."""
        if on_failure is not None:
            self._on_failure = normalize_failure_mode(on_failure)
        from scipy import optimize

        x0_numpy = be.to_numpy([var.value for var in self.problem.variables])
        bounds_list = [var.bounds for var in self.problem.variables]
        lower = be.to_numpy(
            [b[0] if b[0] is not None else -be.inf for b in bounds_list]
        )
        upper = be.to_numpy([b[1] if b[1] is not None else be.inf for b in bounds_list])

        method_choice = self.method_choice
        num_res = len(self.problem.operands)
        num_vars = len(x0_numpy)

        if method_choice == "lm":
            if num_res < num_vars:
                method_choice = "trf"
            elif be.any(lower != -be.inf) or be.any(upper != be.inf):
                pass  # lm ignores bounds; proceed with warning suppressed

        if method_choice == "lm":
            actual_bounds = (-be.inf, be.inf)
        else:
            actual_bounds = (lower, upper)
            eps = be.finfo(be.float64).eps
            for i in range(x0_numpy.shape[0]):
                if lower[i] != -be.inf and x0_numpy[i] <= lower[i]:
                    x0_numpy[i] = lower[i] + eps
                if upper[i] != be.inf and x0_numpy[i] >= upper[i]:
                    x0_numpy[i] = upper[i] - eps

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = optimize.least_squares(
                self._residuals,
                x0_numpy,
                method=method_choice,
                bounds=actual_bounds,
                max_nfev=maxiter,
                verbose=0,
                ftol=tol,
            )

        for i, var in enumerate(self.problem.variables):
            var.update(result.x[i])
        self.problem.update_optics()
        return result

    def _residuals(self, x: Any) -> Any:
        """Canonical least-squares residual ``sqrt(effective_weight)·delta``.

        Uses :meth:`OptimizationProblem.weighted_residuals` so that
        ``sum(residuals**2) == sum_squared()`` exactly — the same merit
        minimized by the native LM / Gauss-Newton path and the scipy-local
        path.  This replaces the previous ``op.fun() = weight·delta`` form,
        which silently dropped field/wavelength weights and produced a
        *different* objective on multi-weight problems (§3.2).
        """
        import numpy as np

        for i, var in enumerate(self.problem.variables):
            var.update(x[i])
        self.problem.update_optics()
        try:
            r = self.problem.weighted_residuals()
            r_np = np.asarray(be.to_numpy(r), dtype=float)
        except Exception as exc:  # noqa: BLE001
            return self._residual_failure(exc)
        if not np.all(np.isfinite(r_np)):
            return self._residual_failure(None)
        self._ref_merit = float(np.dot(r_np, r_np))
        self._last_m = r_np.shape[0] if r_np.ndim else self._last_m
        return r_np

    def _residual_failure(self, exc: Exception | None) -> Any:
        """Map a non-finite / failed residual to the active failure policy.

        ``scipy.optimize.least_squares`` cannot consume ``NaN``, so any policy
        other than ``"raise"`` resolves to a finite, scale-proportional
        penalty (a multiple of the last finite SSR), never a hardcoded
        ``1e10``.
        """
        import numpy as np

        if self._on_failure == "raise":
            raise OptimizationFailure(
                "Residual evaluation failed (non-finite or ray-trace error)."
            ) from exc
        m = max(self._last_m, 1)
        return np.full(m, penalty_residual(self._ref_merit, m))
