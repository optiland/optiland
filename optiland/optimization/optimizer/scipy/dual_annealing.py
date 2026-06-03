from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import optiland.backend as be
from scipy import optimize

from .base import OptimizerGeneric

if TYPE_CHECKING:
    from ...problem import OptimizationProblem


class DualAnnealing(OptimizerGeneric):
    """DualAnnealing is an optimizer that uses the dual annealing algorithm
    to find the minimum of an optimization problem.

    Args:
        problem (OptimizationProblem): The optimization problem to be solved.

    Methods:
        optimize(maxiter=1000, disp=True): Runs the dual annealing algorithm
            to optimize the problem and returns the result.

    """

    def __init__(self, problem: OptimizationProblem):
        warnings.warn(
            "DualAnnealing is deprecated; use "
            "optiland.optimization.minimize(problem, method='dual_annealing'). "
            "Removal in v0.7.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(problem)

    def optimize(self, maxiter=1000, disp=True, callback=None):
        """Runs the dual annealing algorithm to optimize the problem.

        Args:
            maxiter (int): Maximum number of iterations.
            disp (bool): Whether to display the optimization process.
            callback (callable): A callable called after each iteration.

        Returns:
            result: The result of the optimization.

        """
        # Get initial values in backend format
        x0_backend = [var.value for var in self.problem.variables]
        self._x.append(x0_backend)  # Store backend values
        # Convert x0 to NumPy for SciPy
        x0_numpy = be.to_numpy(x0_backend)
        bounds = tuple([var.bounds for var in self.problem.variables])
        if any(None in bound for bound in bounds):
            raise ValueError("Dual annealing requires all variables have bounds.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = optimize.dual_annealing(
                self._fun,
                bounds=bounds,
                maxiter=maxiter,
                x0=x0_numpy,
                callback=callback,
            )

        for idvar, var in enumerate(self.problem.variables):
            var.update(result.x[idvar])

        self.problem.update_optics()

        return result
