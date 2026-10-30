"""SciPy least-squares optimizer wrapper."""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

import optiland.backend as be
from optiland import _epoch
from scipy import optimize

from ..live_plotter import LiveOptimizationPlotter
from .base import OptimizerGeneric

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from ...problem import OptimizationProblem


class LeastSquares(OptimizerGeneric):
    """Optimizer using SciPy's least-squares solver."""

    def __init__(self, problem: OptimizationProblem):
        super().__init__(problem)
        self._f_last = 0.0
        self._best_x: np.ndarray | None = None
        self._best_f = float("inf")
        self._last_finite = True

    def _penalty_vector(self, num_rows: int) -> np.ndarray:
        """Scale-aware stand-in for rows that could not be evaluated."""
        penalty = 10.0 * max(self._f_last, 1.0) / math.sqrt(max(num_rows, 1))
        return np.full(num_rows, penalty)

    def _compute_residuals_vector(self, x_numpy_variables_from_scipy):
        """
        Internal function to update variables and compute the vector of residuals.
        'x_numpy_variables_from_scipy' contains the current values of the optimization
        variables as provided by the SciPy optimizer. These are typically scaled values
        if variable scaling is active.

        The residuals are read inside one evaluation epoch, so per-design work
        such as material indices is computed once per evaluation.

        Residuals are ``effective_weight * delta`` per operand. SciPy cannot accept
        non-finite values, so non-finite rows are replaced by a penalty scaled to the
        last finite residual norm; a failed call yields a full penalty vector.
        """
        num_operands = len(self.problem.operands)

        try:
            for i, optiland_var_wrapper in enumerate(self.problem.variables):
                optiland_var_wrapper.update(x_numpy_variables_from_scipy[i])

            self.problem.update_optics()
            # Optic state is frozen from here until the residuals are read
            optics = {var.optic for var in self.problem.variables}
            with _epoch.evaluation_epoch(optics):
                residuals = np.asarray(
                    be.to_numpy(self.problem.residual_vector()), dtype=float
                ).reshape(-1)
        except _epoch.EpochViolationError:
            raise  # debug guard: a contract breach is a bug, not a bad design
        except Exception:
            # e.g. critical ray failure: no usable rows at all
            self._last_finite = False
            return self._penalty_vector(num_operands)

        finite = np.isfinite(residuals)
        self._last_finite = bool(finite.all())
        if self._last_finite:
            norm = float(np.linalg.norm(residuals))
            self._f_last = norm
            if norm < self._best_f:
                self._best_f = norm
                self._best_x = np.array(x_numpy_variables_from_scipy, dtype=float)
            return residuals

        # Keep finite rows so the Jacobian stays informative
        return np.where(finite, residuals, self._penalty_vector(residuals.size))

    def optimize(
        self,
        maxiter: int | None = None,
        disp: bool = False,
        plot: bool = False,
        tol: float = 1e-3,
        callback: Any = None,
        method_choice: str = "lm",  # Default to 'lm' for DLS
        x_scale: ArrayLike | float | Literal["jac"] | None = None,
    ) -> optimize.OptimizeResult:
        """
        Optimize the problem using a SciPy least squares method.

        Args:
            maxiter (int, optional): Maximum number of function evaluations.
                SciPy's least_squares uses max_nfev.
            disp (bool, optional): Whether to display optimization progress.
            plot (bool, optional): If True, update live plots during optimization.
            tol (float, optional): Tolerance for termination (ftol - tolerance for the
                                   change in the sum of squares). Defaults to 1e-3.
            callback (callable, optional): Called after each optimization iteration.
            method_choice (str, optional): Method for scipy.optimize.least_squares.
                                         'dls': Optiland's native damped least
                                         squares (experimental; supports
                                         bounds; maxiter counts residual
                                         evaluations including Jacobian
                                         columns).
                                         'lm': Levenberg-Marquardt (DLS,
                                         does not support bounds).
                                         'trf': Trust Region Reflective
                                         (supports bounds).
                                         'dogbox': Dogleg algorithm
                                         (supports bounds).
                                         Defaults to 'lm'.
            x_scale (array-like, float, "jac", or None, optional): Characteristic
                scale of each optimizer-space variable, after any Optiland variable
                scaling. If set to "jac", SciPy updates the scale using inverse
                Jacobian column norms. If None, the argument is omitted so the
                installed SciPy version uses its existing default.

        Returns:
            OptimizeResult: The SciPy optimization result.
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

        if method_choice == "dls":
            return self._optimize_dls(maxiter, disp, plot, tol, callback, x_scale)

        # Validate and adjust method_choice
        if method_choice == "lm":
            if num_residuals < num_variables:
                warnings.warn(
                    f"Method 'lm' (Levenberg-Marquardt) "
                    f"chosen, but number of residuals ({num_residuals}) is less "
                    f"than number of variables ({num_variables}). "
                    "This is not supported by 'lm'. Switching to 'trf' method.",
                    stacklevel=2,
                )
                method_choice = "trf"
            elif be.any(lower_bounds_np != -be.inf) or be.any(
                upper_bounds_np != be.inf
            ):
                # This warning is for 'lm' when bounds are present but m >= n
                warnings.warn(
                    "Method 'lm' (Levenberg-Marquardt) chosen, "
                    "but variable bounds are set. "
                    "SciPy's 'lm' method does not support bounds; bounds will "
                    "be ignored.",
                    stacklevel=2,
                )
        elif method_choice not in ["trf", "dogbox"]:
            warnings.warn(
                f"Unknown method_choice '{original_method_choice}'. "
                "Defaulting to 'trf' method.",
                stacklevel=2,
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

        # Seed F_last and the best finite point from x0
        self._f_last = 0.0
        self._best_x = None
        self._best_f = float("inf")
        self._compute_residuals_vector(x0_numpy)

        live_plotter: LiveOptimizationPlotter | None = None
        if plot:
            live_plotter = LiveOptimizationPlotter(self)
            live_plotter.initialize()

        def _wrapped_callback(*args: Any, **kwargs: Any) -> None:
            if callback is not None:
                callback(*args, **kwargs)

            if live_plotter is not None:
                live_plotter.update()

        least_squares_kwargs: dict[str, Any] = {}
        if x_scale is not None:
            least_squares_kwargs["x_scale"] = x_scale

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
                **least_squares_kwargs,
            )

        x_final = result.x
        self._compute_residuals_vector(x_final)
        if not self._last_finite and self._best_x is not None:
            x_final = self._best_x
        for i, optiland_var_wrapper in enumerate(self.problem.variables):
            optiland_var_wrapper.update(x_final[i])
        self.problem.update_optics()  # Final update to the optical system

        if live_plotter is not None:
            live_plotter.update()
            live_plotter.finalize()

        return result

    def _optimize_dls(
        self,
        maxiter: int | None,
        disp: bool,
        plot: bool,
        tol: float,
        callback: Any,
        x_scale: ArrayLike | float | Literal["jac"] | None,
    ) -> optimize.OptimizeResult:
        """Run the native damped least-squares core (``method_choice="dls"``).

        Residuals are ``effective_weight * delta`` from a finite-difference
        evaluator with partial-trace Jacobians; variable bounds are honored.
        ``maxiter`` caps residual evaluations, Jacobian columns included.
        """
        from optiland.optimization.evaluators import FiniteDiffEvaluator
        from optiland.optimization.native import dls
        from optiland.optimization.native.least_squares import _callback_wants_result

        evaluator = FiniteDiffEvaluator(self.problem)
        wants_result = callback is not None and _callback_wants_result(callback)

        live_plotter: LiveOptimizationPlotter | None = None
        if plot:
            live_plotter = LiveOptimizationPlotter(self)
            live_plotter.initialize()

        def _wrapped_callback(intermediate_result: optimize.OptimizeResult) -> None:
            if callback is not None:
                if wants_result:
                    callback(intermediate_result=intermediate_result)
                else:
                    callback(intermediate_result.x)
            if live_plotter is not None:
                live_plotter.update()

        initial_cost = 0.5 * float(be.to_numpy(self.problem.sum_squared()))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = dls(
                evaluator,
                max_nfev=maxiter,
                ftol=tol,
                x_scale=DLS_X_SCALE if x_scale is None else x_scale,
                jacobian_update=DLS_JACOBIAN_UPDATE,
                callback=_wrapped_callback,
            )

        # dls leaves the optics at result.x
        if disp:
            print(
                f"{result.message}\nFunction evaluations {result.nfev}, initial "
                f"cost {initial_cost:.4e}, final cost {result.cost:.4e}, "
                f"first-order optimality {result.optimality:.2e}."
            )
        if live_plotter is not None:
            live_plotter.update()
            live_plotter.finalize()
        return result


# Jacobian update of method_choice="dls". "exact" is the default because gated
# Broyden ("auto") regresses on the benchmarks: both modes reach every
# certified analytic/NIST minimum and stay within 1% of SciPy "trf" on the
# fast lens set (tests/optimization/validation/), but on the freeform
# benchmark "auto" ends at merit 4.49 versus 2.08 for "exact" after the same
# 100n residual evaluations.
DLS_JACOBIAN_UPDATE = "exact"

# Damping scale of method_choice="dls" when x_scale is None: fixed unit scale,
# SciPy "trf"'s default. Moré scaling (x_scale="jac") stalls on the Double
# Gauss benchmark at ~2x trf's merit, as SciPy "trf" itself does with "jac".
DLS_X_SCALE: float | str = 1.0
