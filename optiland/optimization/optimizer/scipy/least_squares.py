"""SciPy least-squares optimizer wrapper."""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

import optiland.backend as be
import scipy
from optiland.optimization.evaluators import FiniteDiffEvaluator
from optiland.optimization.failure import reject_hard_constraints
from scipy import optimize

from ..live_plotter import LiveOptimizationPlotter
from .base import OptimizerGeneric

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from ...evaluators import AimPolicy
    from ...problem import OptimizationProblem

_SCIPY_VERSION = tuple(int(part) for part in scipy.__version__.split(".")[:2])

# Before SciPy 1.16, 'lm' counted its own Jacobian columns toward max_nfev.
_LM_COUNTS_FD_COLUMNS = _SCIPY_VERSION < (1, 16)


class LeastSquares(OptimizerGeneric):
    """Optimizer using SciPy's least-squares solver.

    SciPy's ``"lm"``, ``"trf"`` and ``"dogbox"`` methods get their Jacobians
    from Optiland's finite-difference evaluator, which reuses the residuals
    already computed at ``x``, resumes eligible columns from trace snapshots
    and can freeze ray aiming (``aim_policy="frozen"``).
    """

    def __init__(self, problem: OptimizationProblem):
        super().__init__(problem)
        self._f_last = 0.0
        self._best_x: np.ndarray | None = None
        self._best_f = float("inf")
        self._last_finite = True
        self._evaluator: _SciPyFiniteDiffEvaluator | None = None

    @property
    def _fd_evaluator(self) -> _SciPyFiniteDiffEvaluator:
        """Evaluator behind the residuals and Jacobians (exact aim by default)."""
        if self._evaluator is None:
            self._evaluator = _SciPyFiniteDiffEvaluator(self.problem)
        return self._evaluator

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

        The finite-difference evaluator reads the residuals inside one
        evaluation epoch, so per-design work such as material indices is
        computed once per evaluation. It remembers the evaluation as the
        baseline of a Jacobian at the same point.

        Residuals are ``effective_weight * delta`` per operand. SciPy cannot accept
        non-finite values, so non-finite rows are replaced by a penalty scaled to the
        last finite residual norm; a failed call (all rows ``NaN``, e.g. a
        critical ray failure) yields a full penalty vector.
        """
        residuals = self._fd_evaluator.residuals(x_numpy_variables_from_scipy)

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

    def _compute_jacobian(self, x_numpy_variables_from_scipy: np.ndarray) -> np.ndarray:
        """Finite-difference Jacobian of the residuals, SciPy's ``jac``.

        SciPy asks for the Jacobian at the point it evaluated last, which the
        evaluator reuses as its baseline. Entries that cannot be differenced
        (failed rows on both sides of ``x``) are 0.
        """
        return self._fd_evaluator.jacobian(x_numpy_variables_from_scipy)

    def optimize(
        self,
        maxiter: int | None = None,
        disp: bool = False,
        plot: bool = False,
        tol: float = 1e-3,
        callback: Any = None,
        method_choice: str = "lm",  # Default to 'lm' for DLS
        x_scale: ArrayLike | float | Literal["jac"] | None = None,
        aim_policy: str | AimPolicy = "exact",
    ) -> optimize.OptimizeResult:
        """
        Optimize the problem using a SciPy least squares method.

        Args:
            maxiter (int, optional): Maximum number of function evaluations.
                SciPy's least_squares uses max_nfev. Jacobian columns are not
                counted, except by 'lm' with SciPy older than 1.16, whose
                budget becomes ``max_nfev = ceil(maxiter / (n + 1))`` for
                ``n`` variables: as many iterations as its own 'lm'
                differences allowed.
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
                                         columns). The only choice that
                                         enforces hard constraints.
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
            aim_policy (str or AimPolicy, optional): Ray aiming during the
                finite-difference Jacobian columns. 'exact' (default) re-aims
                every column. 'frozen' reuses the aim solutions at ``x``:
                residuals and merit stay exact, and only the Jacobian ignores
                how a variable moves the aim. Much faster with iterative ray
                aiming. See :mod:`optiland.optimization.evaluators.aim`.

        Returns:
            OptimizeResult: The SciPy optimization result.

        Raises:
            ConfigurationError: If the problem has hard constraints and
                ``method_choice`` is not ``"dls"``.
        """
        if method_choice != "dls":
            reject_hard_constraints(self.problem, f"LeastSquares({method_choice!r})")

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

        num_residuals = self.problem.num_residuals
        num_variables = len(x0_numpy)
        original_method_choice = method_choice  # Store for warning message

        if method_choice == "dls":
            return self._optimize_dls(
                maxiter, disp, plot, tol, callback, x_scale, aim_policy
            )

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
        # 'lm' ignores bounds, so its Jacobian steps ignore them too
        self._evaluator = _SciPyFiniteDiffEvaluator(
            self.problem, aim_policy=aim_policy, bounded=method_choice != "lm"
        )
        max_nfev = maxiter
        if method_choice == "lm" and maxiter is not None and _LM_COUNTS_FD_COLUMNS:
            # Keep the iteration budget of SciPy's own 'lm' differences
            max_nfev = max(math.ceil(maxiter / (num_variables + 1)), 1)

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
                jac=self._compute_jacobian,
                method=method_choice,
                bounds=actual_bounds_for_scipy,
                max_nfev=max_nfev,
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
        aim_policy: str | AimPolicy,
    ) -> optimize.OptimizeResult:
        """Run the native damped least-squares core (``method_choice="dls"``).

        Residuals are ``effective_weight * delta`` from a finite-difference
        evaluator with partial-trace Jacobians; variable bounds and hard
        constraints are honored.
        ``maxiter`` caps residual evaluations, Jacobian columns included.
        """
        from optiland.optimization.native import dls
        from optiland.optimization.observers.callback import CallbackObserver

        evaluator = FiniteDiffEvaluator(self.problem, aim_policy=aim_policy)

        live_plotter: LiveOptimizationPlotter | None = None
        if plot:
            live_plotter = LiveOptimizationPlotter(self)
            live_plotter.initialize()

        # The user callback and the live plot are observers of the run.
        observers: list[Any] = []
        if callback is not None:
            observers.append(CallbackObserver(callback))
        if live_plotter is not None:
            observers.append(_PlotObserver(live_plotter))

        initial_cost = 0.5 * float(be.to_numpy(self.problem.sum_squared()))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = dls(
                evaluator,
                max_nfev=maxiter,
                ftol=tol,
                x_scale=DLS_X_SCALE if x_scale is None else x_scale,
                jacobian_update=DLS_JACOBIAN_UPDATE,
                observers=observers,
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


class _SciPyFiniteDiffEvaluator(FiniteDiffEvaluator):
    """Finite-difference evaluator with the steps of SciPy's '2-point' scheme.

    The columns are the ones SciPy's own differences would compute, so a run
    follows the same path (up to round-off), while each column reuses the
    baseline evaluation, resumes eligible traces and follows the aim policy.
    The step is ``h = sqrt(eps) * max(1, |x|)`` (the evaluator's default with
    ``typical_x = 1``), taken away from zero as SciPy does and flipped toward
    the interior at a bound.

    Args:
        problem: The optimization problem.
        aim_policy: Ray aiming during Jacobian columns, as for
            :class:`FiniteDiffEvaluator`.
        bounded: Whether the steps stay inside the variable bounds. False for
            'lm', which ignores bounds, so an iterate outside them is
            differenced with ordinary steps.
    """

    def __init__(
        self,
        problem: OptimizationProblem,
        *,
        aim_policy: str | AimPolicy = "exact",
        bounded: bool = True,
    ) -> None:
        super().__init__(problem, aim_policy=aim_policy)
        self._bounded = bounded

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Bounds the Jacobian steps honor (``±inf`` when unbounded)."""
        if self._bounded:
            return super().bounds()
        return np.full(self.n_vars, -np.inf), np.full(self.n_vars, np.inf)

    @staticmethod
    def _interior_step(
        xj: float, h: float, lo: float, hi: float, fits_up: bool, fits_down: bool
    ) -> float:
        """Step of size ``h`` away from zero, flipped toward the interior."""
        if xj < 0.0 and fits_down:
            return -h
        return FiniteDiffEvaluator._interior_step(xj, h, lo, hi, fits_up, fits_down)


class _PlotObserver:
    """Refreshes a live plot after every accepted step."""

    def __init__(self, plotter: LiveOptimizationPlotter) -> None:
        self._plotter = plotter

    def on_start(self, state: Any) -> None:
        pass

    def on_step(self, state: Any) -> None:
        self._plotter.update()

    def on_end(self, state: Any, result: Any) -> None:
        pass


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
