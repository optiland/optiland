from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import optiland.backend as be
from scipy import optimize

from ..live_plotter import LiveOptimizationPlotter
from .base import OptimizerGeneric

if TYPE_CHECKING:
    from ...problem import OptimizationProblem


class DifferentialEvolution(OptimizerGeneric):
    """Differential Evolution optimizer for solving optimization problems.

    Args:
        problem (OptimizationProblem): The optimization problem to be solved.

    Methods:
        optimize(maxiter=1000, disp=True, workers=-1): Runs the differential
            evolution optimization algorithm.

    """

    def __init__(self, problem: OptimizationProblem):
        """Initializes a new instance of the DifferentialEvolution class.

        Args:
            problem (OptimizationProblem): The optimization problem to be
                solved.

        """
        super().__init__(problem)

    def optimize(
        self,
        maxiter: int = 1000,
        disp: bool = True,
        plot: bool = False,
        workers: int = -1,
        callback: Any = None,
    ):
        """Runs the differential evolution optimization algorithm.

        Args:
            maxiter (int): Maximum number of iterations.
            disp (bool): Set to True to display status messages.
            plot: If True, update live plots during optimization.
            workers (int): Number of parallel workers to use. Set to -1 to use
                all available processors.
            callback (callable): A callable called after each iteration.

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
        if any(None in bound for bound in bounds):
            raise ValueError(
                "Differential evolution requires all variables have bounds.",
            )

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

            updating = "deferred" if workers == -1 else "immediate"

            result = optimize.differential_evolution(
                self._fun,
                bounds=bounds,
                maxiter=maxiter,
                x0=x0_numpy,
                disp=disp,
                updating=updating,
                workers=workers,
                callback=_wrapped_callback,
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
