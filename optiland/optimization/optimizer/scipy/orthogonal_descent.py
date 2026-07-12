"""Orthogonal Descent Optimizer Module

This module contains the OrthogonalDescent class, which implements a coordinate
descent optimization algorithm. The algorithm sequentially optimizes each
variable while holding others fixed.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.optimization.optimizer.base import BaseOptimizer
from scipy.optimize import minimize_scalar

from ..live_plotter import LiveOptimizationPlotter

if TYPE_CHECKING:
    from optiland.optimization.problem import OptimizationProblem


class OrthogonalDescent(BaseOptimizer):
    """
    Orthogonal Descent (Coordinate Descent) optimizer.

    This optimizer minimizes the objective function by sequentially optimizing
    each variable one at a time. It is useful when derivatives are not available
    or unreliable.
    """

    def __init__(self, problem: OptimizationProblem):
        super().__init__(problem)

    def optimize(self, max_iter=100, tol=1e-4, plot=False):
        """
        Run the orthogonal descent optimization.

        Args:
            max_iter (int): Maximum number of full cycles through all variables.
            tol (float): Tolerance for convergence (relative change in cost function).
            plot: If True, update live plots during optimization.
        """

        live_plotter: LiveOptimizationPlotter | None = None
        if plot:
            live_plotter = LiveOptimizationPlotter(self)
            live_plotter.initialize()

        self.problem.initial_value = self.problem.rss().item()

        current_value = self.problem.initial_value

        for _i in range(max_iter):
            prev_value = current_value

            for _, generic_var in enumerate(self.problem.variables):
                self._optimize_variable(generic_var)

            if live_plotter is not None:
                live_plotter.update()

            current_value = self.problem.rss().item()

            relative_change = abs(prev_value - current_value) / (prev_value + 1e-10)

            if relative_change < tol:
                break

        if live_plotter is not None:
            live_plotter.update()
            live_plotter.finalize()

    def _optimize_variable(self, generic_var):
        """
        Optimizes a single variable using line search.

        Args:
            generic_var: The GenericVariable instance to optimize.
        """
        val_start = generic_var.value.item()

        # Calculate initial cost
        f_start = self.problem.rss().item()

        limit = 1e12  # Soft limit for unbounded variables
        # ``value`` and ``bounds`` are both expressed in the variable's scaled space.
        bounds = generic_var.bounds
        low = bounds[0] if bounds is not None and bounds[0] is not None else -limit
        high = bounds[1] if bounds is not None and bounds[1] is not None else limit

        def objective_func(x):
            # Enforce bounds manually for 'brent' method
            if x < low or x > high:
                return 1e20

            try:
                generic_var.update(x)
                self.problem.update_optics()
                return self.problem.rss().item()
            except Exception:
                return 1e20

        # Define initial bracket based on magnitude
        # Use a relative step size, but keep it within reasonable limits
        step = max(abs(val_start) * 0.05, 0.1)
        bracket = (val_start - step, val_start + step)

        # Use 'brent' method which allows specifying a starting bracket.
        res = minimize_scalar(objective_func, bracket=bracket, method="brent", tol=1e-5)

        # Update variable only if we found a better solution
        if res.fun < f_start:
            generic_var.update(res.x)
        else:
            generic_var.update(val_start)

        self.problem.update_optics()
