"""ScipyGlobalAdapter — wraps DE / DualAnnealing / SHGO / BasinHopping.

Routes to the appropriate existing global-optimizer class based on the
``algorithm`` constructor argument.

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

_GLOBAL_METHODS = {
    "differential_evolution",
    "dual_annealing",
    "shgo",
    "basin_hopping",
}


class ScipyGlobalAdapter:
    """Adapter wrapping a SciPy global optimizer.

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
        self._inner = self._create_inner(problem, algorithm)

    @staticmethod
    def _create_inner(problem: Any, algorithm: str) -> Any:
        if algorithm == "differential_evolution":
            from optiland.optimization.optimizer.scipy.differential_evolution import (
                DifferentialEvolution,
            )

            return DifferentialEvolution(problem)
        if algorithm == "dual_annealing":
            from optiland.optimization.optimizer.scipy.dual_annealing import (
                DualAnnealing,
            )

            return DualAnnealing(problem)
        if algorithm == "shgo":
            from optiland.optimization.optimizer.scipy.shgo import SHGO

            return SHGO(problem)
        if algorithm == "basin_hopping":
            from optiland.optimization.optimizer.scipy.basin_hopping import BasinHopping

            return BasinHopping(problem)
        raise ValueError(f"Unknown algorithm: {algorithm!r}")  # pragma: no cover

    def run(self, *, callback: Any = None, maxiter: int = 1000, **kwargs: Any) -> Any:
        """Delegate to the appropriate global optimizer's ``optimize``."""
        alg = self.algorithm
        if alg == "differential_evolution":
            workers = kwargs.get("workers", -1)
            return self._inner.optimize(
                maxiter=maxiter, callback=callback, workers=workers
            )
        if alg == "dual_annealing":
            return self._inner.optimize(maxiter=maxiter, callback=callback)
        if alg == "shgo":
            workers = kwargs.get("workers", 1)
            return self._inner.optimize(workers=workers, callback=callback)
        if alg == "basin_hopping":
            return self._inner.optimize(niter=maxiter, callback=callback)
        raise ValueError(f"Unknown algorithm: {alg!r}")  # pragma: no cover
