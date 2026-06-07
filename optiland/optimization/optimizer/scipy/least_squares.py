from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import optiland.backend as be
from scipy import optimize

from ..live_plotter import LiveOptimizationPlotter
from .base import OptimizerGeneric

if TYPE_CHECKING:
    from ...problem import OptimizationProblem


class LeastSquares(OptimizerGeneric):
    def __init__(self, problem: OptimizationProblem):
        super().__init__(problem)

    def _compute_residuals_vector(self, x_numpy_variables_from_scipy):
        """
        Internal function to update variables and compute the vector of residuals.
        'x_numpy_variables_from_scipy' contains the current values of the optimization
        variables as provided by the SciPy optimizer. These are typically scaled values
        if variable scaling is active.
        """

        for i, optiland_var_wrapper in enumerate(self.problem.variables):
            optiland_var_wrapper.update(x_numpy_variables_from_scipy[i])

        self.problem.update_optics()

        try:
            residuals_backend_array = be.array(
                [op.fun() for op in self.problem.operands]
            )

            # Handle cases where ray tracing might fail and produce NaNs
            if be.any(be.isnan(residuals_backend_array)):
                num_operands = len(self.problem.operands)
                # Return a vector of large constant values to penalize this region
                # The magnitude should be large enough to indicate a poor solution.
                error_value = be.sqrt(1e10 / num_operands if num_operands > 0 else 1e10)
                return be.to_numpy(be.full(num_operands, error_value))

            return be.to_numpy(
                residuals_backend_array
            )  # Convert to NumPy array for SciPy

        except Exception:
            # Catch any other exceptions during optical calculation
            # (e.g., critical ray failure)
            # This is a general fallback; more specific error handling
            # might be beneficial.
            num_operands = len(self.problem.operands)
            error_value = be.sqrt(1e10 / num_operands if num_operands > 0 else 1e10)
            # Return a vector of large constant values
            return be.to_numpy(be.full(num_operands, error_value))

    def optimize(
        self,
        maxiter: int | None = None,
        disp: bool = False,
        plot: bool = False,
        tol: float = 1e-3,
        callback: Any = None,
        method_choice: str = "lm",  # Default to 'lm' for DLS
    ):
        """
        Optimize the problem using a SciPy least squares method.

        Args:
            max_nfev (int, optional): Maximum number of function evaluations (NFEV).
                                      SciPy's least_squares uses max_nfev.
            disp (bool, optional): Whether to display optimization progress.
            plot: If True, update live plots during optimization.
            tol (float, optional): Tolerance for termination (ftol - tolerance for the
                                   change in the sum of squares). Defaults to 1e-3.
            method_choice (str, optional): Method for scipy.optimize.least_squares.
                                         'lm': Levenberg-Marquardt (DLS,
                                         does not support bounds).
                                         'trf': Trust Region Reflective
                                         (supports bounds).
                                         'dogbox': Dogleg algorithm
                                         (supports bounds).
                                         Defaults to 'lm'.
        """

        x0_scaled_values = [var.value for var in self.problem.variables]
        self._x.append(list(x0_scaled_values))
        x0_numpy = be.to_numpy(x0_scaled_values)

        current_bounds_scaled = tuple([var.bounds for var in self.problem.variables])
        lower_bounds_np = be.to_numpy(
            [b[0] if b[0] is not None else -be.inf for b in current_bounds_scaled]
        )
        upper_bounds_np = be.to_numpy(
            [b[1] if b[1] is not None else be.inf for b in current_bounds_scaled]
        )

        num_residuals = len(self.problem.operands)
        num_variables = len(x0_numpy)
        original_method_choice = method_choice  # Store for warning message

        # Validate and adjust method_choice
        if method_choice == "lm":
            if num_residuals < num_variables:
                print(
                    f"Warning: Method 'lm' (Levenberg-Marquardt) "
                    f"chosen, but number of residuals ({num_residuals}) is less "
                    f"than number of variables ({num_variables}). "
                    "This is not supported by 'lm'. Switching to 'trf' method."
                )
                method_choice = "trf"
            elif be.any(lower_bounds_np != -be.inf) or be.any(
                upper_bounds_np != be.inf
            ):
                # This warning is for 'lm' when bounds are present but m >= n
                print(
                    "Warning: Method 'lm' (Levenberg-Marquardt) chosen, "
                    "but variable bounds are set. "
                    "SciPy's 'lm' method does not support bounds; bounds will "
                    "be ignored."
                )
        elif method_choice not in ["trf", "dogbox"]:
            print(
                f"Warning: Unknown method_choice '{original_method_choice}'. "
                "Defaulting to 'trf' method."
            )
            method_choice = "trf"

        # Determine actual_bounds_for_scipy and adjust x0 if needed based on
        # the *final* method_choice
        if (
            method_choice == "lm"
        ):  # 'lm' was originally chosen and conditions for switching were not met
            actual_bounds_for_scipy = (-be.inf, be.inf)
        else:
            # method_choice is 'trf' or 'dogbox' (either originally or
            # after adjustment)
            actual_bounds_for_scipy = (lower_bounds_np, upper_bounds_np)
            # Ensure x0 is strictly within bounds
            eps = be.finfo(be.float64).eps
            for i in range(x0_numpy.shape[0]):
                if lower_bounds_np[i] != -be.inf and x0_numpy[i] <= lower_bounds_np[i]:
                    x0_numpy[i] = lower_bounds_np[i] + eps
                if upper_bounds_np[i] != be.inf and x0_numpy[i] >= upper_bounds_np[i]:
                    x0_numpy[i] = upper_bounds_np[i] - eps
                # Clip if adjustment pushed it out of the other bound
                # (e.g. narrow bound range)
                if x0_numpy[i] > upper_bounds_np[i] and upper_bounds_np[i] != be.inf:
                    x0_numpy[i] = upper_bounds_np[i]
                if x0_numpy[i] < lower_bounds_np[i] and lower_bounds_np[i] != -be.inf:
                    x0_numpy[i] = lower_bounds_np[i]

        scipy_verbose_level = 1 if disp else 0

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
            result = optimize.least_squares(
                self._compute_residuals_vector,
                x0_numpy,
                method=method_choice,
                bounds=actual_bounds_for_scipy,
                max_nfev=maxiter,
                verbose=scipy_verbose_level,
                ftol=tol,
                callback=_wrapped_callback,
            )

        for i, optiland_var_wrapper in enumerate(self.problem.variables):
            optiland_var_wrapper.update(result.x[i])
        self.problem.update_optics()  # Final update to the optical system

        if live_plotter is not None:
            live_plotter.update()
            live_plotter.finalize()

        return result
