"""Optiland Glass Expert Optimization Module

This module provides a class for defining and solving
optimization problems that include categorical variables
in the form of lens materials.
For each glass variable, the goal is to choose the glass
that yields the lowest merit function value.
The glasses to choose from are specified in the form
of a list of strings:
glasses = ['N-BK7', 'S-BSM22', 'LF5G19', ...]

The optimizer performs categorical optimization on
glass variables encoded into their (n_d, V_d) properties.
It systematically searches for better-performing glasses by
exploring nearest neighbors in the material space,
followed by continuous local refinement of lens parameters.

drpaprika, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from optiland.materials import (
    downsample_glass_map,
    get_nd_vd,
    get_neighbour_glasses,
    plot_glass_map,
)

from ..live_plotter import LiveOptimizationPlotter
from .base import OptimizerGeneric

if TYPE_CHECKING:
    from ...problem import OptimizationProblem
    from ...variable import Variable


class GlassExpert(OptimizerGeneric):
    """
    Greedy nearest-neighbour glass substitution strategy
    inspired by CODE V's Glass Expert.

    This optimizer performs categorical optimization on glass variables
    defined by their (n_d, V_d) properties. It systematically searches
    for better-performing glasses by exploring nearest neighbors
    in the material space, followed by continuous local refinement
    of lens parameters.

    Overview of the algorithm:

    - Treat each glass choice as a discrete (categorical) variable
      and perform a fixed number of greedy passes.
    - First pass, for each variable:
        - Perform a broad search over the entire glass catalogue.
    - Second pass, for each variable:
        - Retrieve the `num_neighbours` nearest materials in (n_d, V_d) space
        - Perform a focused search around the current glass choices.
    - For every candidate glass:
        - Substitute it into the design.
        - Run a continuous local optimization to evaluate performance.
        - If the objective improves, keep the substitution; otherwise, roll back.
    - Finally, perform one more local-only optimization
      over all continuous variables to polish the solution.
    """

    def __init__(self, problem: OptimizationProblem):
        self.problem = problem
        self._x = []
        self._state = []
        self.verbose = True
        self.plot_glass_map = False
        self.opt_params = dict()

        self._nd_vd_cache: dict[str, tuple[float, float]] = {}

        if self.problem.initial_value == 0.0:
            self.problem.initial_value = self.problem.sum_squared()

    def vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _get_nd_vd(self, glasses: list[str]) -> dict[str, tuple[float, float]]:
        """
        Return (n_d, V_d) for all requested glasses.

        Already computed values are reused from memory.
        Only unknown glasses are fetched from disk via get_nd_vd().
        """
        new_glasses = [g for g in glasses if g not in self._nd_vd_cache]
        if new_glasses:
            fetched = {g: get_nd_vd(g) for g in new_glasses}
            self._nd_vd_cache.update(fetched)
        return {g: self._nd_vd_cache[g] for g in glasses}

    def _save_state(self):
        # Store current values of all problem variables for later restoration
        self._state = [var.value for var in self.problem.variables]

    def _restore_state(self, old_state=None):
        # Restore variables to a previous state and update the optical system
        state = old_state or self._state
        for variable, val in zip(self.problem.variables, state, strict=False):
            variable.update(val)
        self.problem.update_optics()

    def global_exploration(
        self,
        glass_variables: list[Variable],
        pool_size: int,
    ) -> None:
        """
        Perform a broad search over the entire glass catalogue.

        For each glass variable, downsample the material map
        to `pool_size` using K-Means clustering in the (n_d, V_d) space,
        and evaluate all glasses. This step ensures diverse
        initial sampling before focusing on nearest neighbours.

        Args:
            glass_variables (list[Variable]): Glass variables to optimize.
            pool_size (int): Number of materials to keep after downsampling.
        """

        if glass_variables:
            self.vprint(f"\n{'-' * 70}\nGlobal exploration\n")

        for variable in glass_variables:
            self.vprint(f"Selecting {variable}:")

            # Retreive (nd, vd) information for all glasses
            glass_dict = self._get_nd_vd(variable.glass_selection)

            # Downsample to promote diversity and limit search size
            glass_dict = downsample_glass_map(glass_dict, num_glasses_to_keep=pool_size)

            # Plot the glass selection
            if self.plot_glass_map:
                plot_glass_map(
                    glass_selection=variable.glass_selection,
                    highlights=glass_dict.keys(),
                    title=f"Map of global exploration space\n{variable}",
                )

            # Evaluate each candidate material within the downsampled pool
            self.explore_glasses(
                glass_variables=glass_variables,
                current_glass_variable=variable,
                glasses=glass_dict.keys(),
            )

    def local_exploration(
        self,
        glass_variables: list[Variable],
        num_neighbours: int,
    ) -> None:
        """
        Perform a focused search around the current glass choices.

        For each glass variable, identify its nearest `num_neighbours`
        in (n_d, V_d) space, then run `explore_glass_list` on those candidates.
        This refines selections made during global exploration.

        Args:
            glass_variables (list[Variable]): Glass variables to refine.
            num_neighbours (int): Number of nearest neighbours to consider.
        """

        if glass_variables:
            self.vprint(f"\n{'-' * 70}\nLocal exploration\n")

        for variable in glass_variables:
            self.vprint(f"Selecting {variable}:")

            # Retreive (nd, vd) information for all glasses
            glass_dict = self._get_nd_vd(variable.glass_selection)

            # Identify top candidates based on material proximity in (n_d, V_d) space
            neighbours = get_neighbour_glasses(
                glass=variable.value,
                glass_dict=glass_dict,
                num_neighbours=num_neighbours,
                plot=self.plot_glass_map,
            )

            # Evaluate each neighbouring material
            self.explore_glasses(
                glass_variables=glass_variables,
                current_glass_variable=variable,
                glasses=neighbours,
            )

    def explore_glasses(
        self,
        glass_variables: list[Variable],
        current_glass_variable: Variable,
        glasses: list,
    ) -> None:
        """
        Test a list of candidate glasses by performing local optimizations.

        Saves the initial state, then for each candidate:
          - Updates the glass variable
          - Runs a continuous optimization to measure objective
          - Restores the previous state before the next trial
        After all trials, restores the best-performing configuration.
        """
        # Save state before testing new glasses
        self._save_state()
        current_glass = current_glass_variable.value
        best_glass = current_glass
        best_error = self.problem.sum_squared()
        best_error_init = best_error
        state_init = [var.value for var in (*self.problem.variables, *glass_variables)]
        best_state = state_init

        # Optimize locally on each neighbour glass
        for neighbour in glasses:
            self.vprint(
                f"\tTrying {neighbour:<8} as {current_glass_variable}. ", end=""
            )
            current_glass_variable.update(neighbour)
            self.problem.update_optics()
            result = self.optimize(**self.opt_params)
            error = result.fun
            fmt = (
                ".0f"
                if error >= 100
                else ".1f"
                if error >= 10
                else ".2f"
                if error >= 1e-2
                else ".2e"
            )
            self.vprint(f"Error function value: {format(error, fmt)}")
            if error < best_error:
                best_error = error
                best_glass = neighbour
                best_state = [
                    var.value for var in (*self.problem.variables, *glass_variables)
                ]

            # Always restore to original state before next candidate
            self._restore_state()

        # Restore to the best state
        current_glass_variable.update(best_glass)
        self._restore_state(best_state)

        if best_glass != current_glass and best_error < best_error_init:
            self.vprint(f"\t-> Selected {best_glass} as {current_glass_variable}.")
            self.vprint(f"\tNew combination: {[var.value for var in glass_variables]}")
        else:
            self.vprint(f"\tNo better glass found, keeping {current_glass}.")
            self.vprint(f"\tCombination: {[var.value for var in glass_variables]}")

        self.vprint(
            f"\tBest error function value: {best_error:.2f} "
            f"({(best_error / best_error_init - 1) * 100:.0f}%).\n"
        )

        # Ensure optics reflect the final selection
        self.problem.update_optics()

    def run(
        self,
        num_neighbours: int = 7,
        maxiter: int = 1000,
        tol: float = 1e-3,
        disp: bool = True,
        plot: bool = False,
        callback: Any = None,
        verbose: bool = True,
        plot_glass_map=False,
    ):
        """
        Execute the full glass optimization workflow.

        This includes:
          1. Global exploration of diverse materials.
          2. Local refinement around current selections.
          3. Final continuous optimization on remaining variables.

        This method performs a greedy, cyclic optimization over all
        glass variables by iteratively substituting each one with its
        neighbors in refractive index (n_d) and Abbe  number (V_d) space.
        For each neighbor, a local optimization is performed over the
        continuous variables only to evaluate merit.
        If a substitution improves the objective function, it is retained.
        After completing the greedy passes, a final local optimization
        is performed over all continuous variables only.

        Args:
        num_neighbours (int, optional): Number of nearest neighbors to try
            for each categorical variable during each pass.
            Default is 7.
        maxiter (int, optional): Maximum number of iterations for each
            local optimization run.
            Default is 1000.
        tol (float, optional): Tolerance for convergence in local optimization.
            Default is 1e-3.
        disp (bool, optional): Whether to display internal messages from the optimizer.
            Default is True.
        plot: If True, update live plots during optimization.
        callback (callable, optional): Optional function called at each
            iteration of the optimizer.
        verbose (bool, optional): Whether to print informative messages
            from this method.
            Default is True.
        plot_glass_map (bool, optional): Wheter to plot the glass selection
            on a (nd, vd) glass map.
            Default is False.

        Returns:
            result (OptimizeResult): Result of the final local optimization
                containing fields such as:
                - x: optimized continuous variable values
                - fun: final objective function value
                - nfev: number of function evaluations, etc.
        """

        self.verbose = verbose
        self.plot_glass_map = plot_glass_map

        live_plotter: LiveOptimizationPlotter | None = None
        if plot:
            live_plotter = LiveOptimizationPlotter(self)
            live_plotter.initialize()

        def _wrapped_callback(*args: Any, **kwargs: Any) -> None:
            if callback is not None:
                callback(*args, **kwargs)

            if live_plotter is not None:
                live_plotter.update()

        # Local optimizer params
        self.opt_params = dict(
            maxiter=maxiter, tol=tol, disp=disp, callback=_wrapped_callback
        )

        # Identify variables
        continuous_variables = [
            var for var in self.problem.variables if isinstance(var.value, float)
        ]
        glass_variables = [
            var for var in self.problem.variables if isinstance(var.value, str)
        ]

        if not glass_variables:
            self.vprint("No glass variables — skipping GlassExpert.")
        else:
            self.vprint(
                f"Initial glasses combination: {[var.value for var in glass_variables]}"
            )

        # Remove the categorical variables for the local optimization
        self.problem.variables = continuous_variables

        # Broad search across the glass catalogue
        self.global_exploration(
            glass_variables=glass_variables,
            pool_size=num_neighbours,
        )

        # Focused search around chosen materials
        self.local_exploration(
            glass_variables=glass_variables,
            num_neighbours=num_neighbours,
        )

        # Final local-only optimization (on continuous variables only)
        result = self.optimize(**self.opt_params)

        # Update all continuous variables to final values
        for var, val in zip(self.problem.variables, result.x, strict=False):
            var.update(val)
        self.problem.update_optics()

        if live_plotter is not None:
            live_plotter.update()
            live_plotter.finalize()

        return result
