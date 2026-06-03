from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import optiland.backend as be
from scipy import optimize

from .base import OptimizerGeneric

if TYPE_CHECKING:
    from ...problem import OptimizationProblem


class BasinHopping(OptimizerGeneric):
    """Basin-hopping optimizer for solving optimization problems.

    Args:
        problem (OptimizationProblem): The optimization problem to be solved.

    Methods:
        optimize(maxiter=1000, disp=True, workers=-1): Runs the basin-hopping
            optimization algorithm.

    """

    def __init__(self, problem: OptimizationProblem):
        """Initializes a new instance of the BasinHopping class.

        Args:
            problem (OptimizationProblem): The optimization problem to be
                solved.

        .. deprecated::
            Use ``optiland.optimization.minimize(problem,
            method='basin_hopping')`` instead.  Removal in v0.7.0.
        """
        warnings.warn(
            "BasinHopping is deprecated; use "
            "optiland.optimization.minimize(problem, method='basin_hopping'). "
            "Removal in v0.7.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(problem)

    def optimize(self, niter=100, callback=None, *args, **kwargs):
        """Runs the basin-hopping algorithm. Note that the basin-hopping
        algorithm accepts the same arguments as the
        scipy.optimize.basinhopping function.

        Args:
            niter (int): Number of iterations to perform. Default is 100.
            callback (callable): A callable called after each iteration.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            result (OptimizeResult): The optimization result.

        Raises:
            ValueError: If any variable in the problem does not have bounds.

        """
        # Get initial values in backend format
        x0_backend = [var.value for var in self.problem.variables]
        self._x.append(x0_backend)  # Store backend values
        # Convert x0 to NumPy for SciPy
        x0_numpy = be.to_numpy(x0_backend)
        bounds = tuple([var.bounds for var in self.problem.variables])
        if not all(x is None for pair in bounds for x in pair):
            raise ValueError("Basin-hopping does not accept bounds.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            result = optimize.basinhopping(
                self._fun,
                x0=x0_numpy,
                niter=niter,
                callback=callback,
                **kwargs,
            )

        for idvar, var in enumerate(self.problem.variables):
            var.update(result.x[idvar])

        self.problem.update_optics()

        return result
