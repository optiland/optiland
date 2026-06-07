from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import optiland.backend as be
from scipy import optimize

from ..live_plotter import LiveOptimizationPlotter
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

        """
        super().__init__(problem)

    def optimize(
        self,
        niter: int = 100,
        plot: bool = False,
        callback: Any = None,
        *args,
        **kwargs,
    ):
        """Runs the basin-hopping algorithm. Note that the basin-hopping
        algorithm accepts the same arguments as the
        scipy.optimize.basinhopping function.

        Args:
            niter (int): Number of iterations to perform. Default is 100.
            plot: If True, update live plots during optimization.
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

        plot: bool = kwargs.get("plot", False)
        live_plotter: LiveOptimizationPlotter | None = None
        if plot:
            live_plotter = LiveOptimizationPlotter(self)
            live_plotter.initialize()

        def _wrapped_callback(*args: Any, **kwargs: Any) -> None:
            if callback is not None:
                callback(*args, **kwargs)

            if live_plotter is not None:
                live_plotter.update()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            result = optimize.basinhopping(
                self._fun,
                x0=x0_numpy,
                niter=niter,
                callback=_wrapped_callback,
                **kwargs,
            )

        # The last function evaluation is not necessarily the lowest.
        # Update all lens variables to their optimized values
        for idvar, var in enumerate(self.problem.variables):
            var.update(result.x[idvar])
        self.problem.update_optics()

        if live_plotter is not None:
            live_plotter.update()
            live_plotter.finalize()

        return result
