"""ScipyLocalAdapter — wraps OptimizerGeneric (scipy.optimize.minimize).

Delegates to the existing ``OptimizerGeneric.optimize()`` and threads the
observer callback through its ``callback=`` argument.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from optiland.optimization.state import Capability

if TYPE_CHECKING:
    from optiland.optimization.problem import OptimizationProblem

_CAPS = frozenset(
    {
        Capability.OBSERVE,
        Capability.MUTATE_TERMINATION,
        Capability.SCIPY_CONSTRAINTS,
        Capability.BOUNDS,
    }
)

# SciPy methods routed through OptimizerGeneric
_LOCAL_SCIPY_METHODS = {
    "l-bfgs-b",
    "bfgs",
    "slsqp",
    "trust-constr",
    "cobyla",
    "nelder-mead",
    "cg",
    "newton-cg",
    "tnc",
    "powell",
}


class ScipyLocalAdapter:
    """Adapter wrapping ``OptimizerGeneric`` behind the managed adapter interface.

    Args:
        problem: The optimization problem.
        method: SciPy ``minimize`` method string (default None = auto-select).
    """

    capabilities: frozenset[Capability] = _CAPS
    requires_backend: str | None = None
    method_name: str = "scipy_local"

    def __init__(self, problem: OptimizationProblem, method: str | None = None):
        from optiland.optimization.optimizer.scipy.base import OptimizerGeneric

        self.problem = problem
        self.method = method
        self._inner = OptimizerGeneric(problem)

    def run(  # noqa: E501
        self, *, callback: Any = None, maxiter: int = 1000, tol: float = 1e-3, **_: Any
    ) -> Any:
        """Delegate to ``OptimizerGeneric.optimize``."""
        return self._inner.optimize(
            method=self.method,
            maxiter=maxiter,
            tol=tol,
            callback=callback,
            disp=False,  # display handled by ConsoleObserver
        )
