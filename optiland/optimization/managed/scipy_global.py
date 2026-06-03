"""ScipyGlobalAdapter — private core for SciPy global optimizers.

Calls scipy directly without going through deprecated legacy classes —
no DeprecationWarning is triggered from the facade path.

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
        Capability.MUTATE_TERMINATION,
        Capability.SCIPY_CONSTRAINTS,
        Capability.BOUNDS,
    }
)

_GLOBAL_METHODS = {
    "differential_evolution",
    "dual_annealing",
    "shgo",
    "basin_hopping",
}


class ScipyGlobalAdapter:
    """Private-core adapter for SciPy global optimizers.

    Calls scipy directly — does not instantiate any deprecated legacy class.

    Args:
        problem: The optimization problem.
        algorithm: One of ``"differential_evolution"``, ``"dual_annealing"``,
            ``"shgo"``, ``"basin_hopping"``.
    """

    capabilities: frozenset[Capability] = _CAPS
    requires_backend: str | None = None

    def __init__(self, problem: OptimizationProblem, algorithm: str):
        algorithm = algorithm.lower()
        if algorithm not in _GLOBAL_METHODS:
            raise ValueError(f"Unknown global algorithm: {algorithm!r}")
        self.algorithm = algorithm
        self.method_name = algorithm
        self.problem = problem

    def run(self, *, callback: Any = None, maxiter: int = 1000, **kwargs: Any) -> Any:
        """Run the appropriate scipy global optimizer."""
        alg = self.algorithm
        if alg == "differential_evolution":
            return self._run_differential_evolution(
                maxiter=maxiter, callback=callback, **kwargs
            )
        if alg == "dual_annealing":
            return self._run_dual_annealing(maxiter=maxiter, callback=callback)
        if alg == "shgo":
            return self._run_shgo(callback=callback, **kwargs)
        if alg == "basin_hopping":
            return self._run_basin_hopping(niter=maxiter, callback=callback)
        raise ValueError(f"Unknown algorithm: {alg!r}")  # pragma: no cover

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fun(self, x: Any) -> float:
        for i, var in enumerate(self.problem.variables):
            var.update(be.array(x[i]))
        self.problem.update_optics()
        try:
            rss = self.problem.sum_squared()
            if be.isnan(rss):
                return 1e10
            return be.to_numpy(rss).item()
        except (ValueError, Exception):  # noqa: BLE001
            return 1e10

    def _writeback(self, x_opt: Any) -> None:
        for i, var in enumerate(self.problem.variables):
            var.update(x_opt[i])
        self.problem.update_optics()

    def _run_differential_evolution(
        self, maxiter: int, callback: Any, workers: int = -1, **_: Any
    ) -> Any:
        from scipy import optimize

        x0 = be.to_numpy([var.value for var in self.problem.variables])
        bounds = tuple([var.bounds for var in self.problem.variables])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = optimize.differential_evolution(
                self._fun,
                bounds=bounds,
                maxiter=maxiter,
                callback=callback,
                workers=workers,
                x0=x0,
            )
        self._writeback(result.x)
        return result

    def _run_dual_annealing(self, maxiter: int, callback: Any) -> Any:
        from scipy import optimize

        x0 = be.to_numpy([var.value for var in self.problem.variables])
        bounds = tuple([var.bounds for var in self.problem.variables])
        if any(None in bound for bound in bounds):
            raise ValueError("Dual annealing requires all variables have bounds.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = optimize.dual_annealing(
                self._fun, bounds=bounds, maxiter=maxiter, x0=x0, callback=callback
            )
        self._writeback(result.x)
        return result

    def _run_shgo(self, workers: int = 1, callback: Any = None, **kwargs: Any) -> Any:
        from scipy import optimize

        bounds = tuple([var.bounds for var in self.problem.variables])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = optimize.shgo(
                self._fun, bounds=bounds, workers=workers, callback=callback, **kwargs
            )
        self._writeback(result.x)
        return result

    def _run_basin_hopping(self, niter: int, callback: Any) -> Any:
        from scipy import optimize

        x0 = be.to_numpy([var.value for var in self.problem.variables])
        bounds = tuple([var.bounds for var in self.problem.variables])
        if not all(x is None for pair in bounds for x in pair):
            raise ValueError("Basin-hopping does not accept bounds.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = optimize.basinhopping(
                self._fun, x0=x0, niter=niter, callback=callback
            )
        self._writeback(result.x)
        return result
