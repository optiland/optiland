"""Optiland Scipy Optimization Module

This module contains classes for various optimization algorithms that can be
used to solve optimization problems defined in the OptimizationProblem class.
This module provides a generic optimizer class and several specific optimizers
that utilize different algorithms from the SciPy library.

Kramer Harrison, 2024
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import optiland.backend as be
from scipy import optimize

from ..base import BaseOptimizer
from ..live_plotter import LiveOptimizationPlotter

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...problem import OptimizationProblem


class OptimizerGeneric(BaseOptimizer):
    """Generic optimizer class for solving optimization problems.

    Args:
        problem (OptimizationProblem): The optimization problem to be solved.

    Attributes:
        problem (OptimizationProblem): The optimization problem to be solved.
        _x (list): List to store the values of the variables during
            optimization.

    Methods:
        optimize(maxiter=1000, disp=True, tol=1e-3): Optimize the problem
            using the specified parameters.
        undo(): Undo the last optimization step.
        _fun(x): Internal function to evaluate the objective function.

    """

    def __init__(self, problem: OptimizationProblem):
        super().__init__(problem)
        self._x = []

        assert not any(isinstance(var.value, str) for var in self.problem.variables), (
            "Glass material(s) have been declared as variable(s). "
            "Please use GlassExpert or remove them."
        )

        if self.problem.initial_value == 0.0:
            self.problem.initial_value = self.problem.sum_squared()

    def optimize(
        self,
        method: str | None = None,
        maxiter: int = 1000,
        disp: bool = True,
        tol: float = 1e-3,
        callback: Callable | None = None,
        plot: bool = False,
    ) -> optimize.OptimizeResult:
        """Optimize the problem using the specified parameters.

        Args:
            method (str, optional): The optimization method to use. Default is
                chosen to be one of BFGS, L-BFGS-B, SLSQP, depending on
                whether contraints or bounds given. Follows
                scipy.optimize.minimize method.
            maxiter (int, optional): Maximum number of iterations.
                Default is 1000.
            disp (bool, optional): Whether to display optimization information.
                Default is True.
            tol (float, optional): Tolerance for convergence. Default is 1e-3.
            callback (callable): A callable called after each iteration.
            plot: If True, update live plots during optimization.

        Returns:
            result (OptimizeResult): The optimization result.

        """
        if method == "Default":
            method = None

        x0 = [var.value for var in self.problem.variables]
        self._x.append(x0)
        x0 = be.to_numpy(x0)
        bounds = tuple([var.bounds for var in self.problem.variables])
        options = {"maxiter": maxiter, "disp": disp}

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
            result = optimize.minimize(
                self._fun,
                x0,
                method=method,
                bounds=bounds,
                options=options,
                tol=tol,
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

    def undo(self):
        """Undo the last optimization step."""
        if len(self._x) > 0:
            x0 = self._x[-1]
            for idvar, var in enumerate(self.problem.variables):
                var.update(x0[idvar])
            self._x.pop(-1)

    def _fun(self, x) -> float:
        """Internal function to evaluate the objective function.

        Args:
            x (array-like): The values of the variables.

        Returns:
            rss (float): The residual sum of squares.

        """
        # Update all variables to their new values
        for idvar, var in enumerate(self.problem.variables):
            var.update(be.array(x[idvar]))

        # Update optics (e.g., pickups and solves)
        self.problem.update_optics()

        # Compute merit function value
        try:
            rss = self.problem.sum_squared()
            if be.isnan(rss):
                return 1e10
            # --- Convert result back to float for SciPy ---
            return be.to_numpy(rss).item()
        except ValueError:
            return 1e10
