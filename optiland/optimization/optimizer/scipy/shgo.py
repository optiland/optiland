from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from scipy import optimize

from .base import OptimizerGeneric

if TYPE_CHECKING:
    from ...problem import OptimizationProblem


class SHGO(OptimizerGeneric):
    """Simplicity Homology Global Optimization (SHGO).

    Args:
        problem (OptimizationProblem): The optimization problem to be solved.

    Methods:
        optimize(workers=-1, *args, **kwargs): Runs the SHGO algorithm.

    """

    def __init__(self, problem: OptimizationProblem):
        """Initializes a new instance of the SHGO class.

        Args:
            problem (OptimizationProblem): The optimization problem to be
                solved.

        .. deprecated::
            Use ``optiland.optimization.minimize(problem, method='shgo')``
            instead.  Removal in v0.7.0.
        """
        warnings.warn(
            "SHGO is deprecated; use "
            "optiland.optimization.minimize(problem, method='shgo'). "
            "Removal in v0.7.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(problem)

    def optimize(self, workers=-1, callback=None, *args, **kwargs):
        """Runs the SHGO algorithm. Note that the SHGO algorithm accepts the same
        arguments as the scipy.optimize.shgo function.

        Args:
            workers (int): Number of parallel workers to use. Set to -1 to use
                all available CPU processors. Default is -1.
            callback (callable): A callable called after each iteration.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            result (OptimizeResult): The optimization result.

        Raises:
            ValueError: If any variable in the problem does not have bounds.

        """
        x0 = [var.value for var in self.problem.variables]
        self._x.append(x0)
        bounds = tuple([var.bounds for var in self.problem.variables])
        if any(None in bound for bound in bounds):
            raise ValueError("SHGO requires all variables have bounds.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            result = optimize.shgo(
                self._fun,
                bounds=bounds,
                workers=workers,
                callback=callback,
                **kwargs,
            )

        for idvar, var in enumerate(self.problem.variables):
            var.update(result.x[idvar])

        self.problem.update_optics()

        return result
