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

    def __init__(self, problem: OptimizationProblem, method_choice: str = "lm"):
        self.problem = problem
        self.method_choice = method_choice

    def run(
        self,
        *,
        callback: Any = None,  # noqa: ARG002 — least_squares has no callback
        maxiter: int | None = None,
        tol: float = 1e-3,
        **_: Any,
    ) -> Any:
        """Run scipy.optimize.least_squares and write back optimal params."""
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
        for i, var in enumerate(self.problem.variables):
            var.update(x[i])
        self.problem.update_optics()
        try:
            res = be.array([op.fun() for op in self.problem.operands])
            if be.any(be.isnan(res)):
                n = len(self.problem.operands)
                err = be.sqrt(1e10 / n if n > 0 else 1e10)
                return be.to_numpy(be.full(n, err))
            return be.to_numpy(res)
        except Exception:  # noqa: BLE001
            n = len(self.problem.operands)
            err = be.sqrt(1e10 / n if n > 0 else 1e10)
            return be.to_numpy(be.full(n, err))
