"""SciPy ``least_squares`` behind ``minimize(method="least_squares")``.

Private core: calls ``scipy.optimize.least_squares`` directly on the residuals
``effective_weight * delta`` (decision V7), with the scale-aware failure
penalty of PR1.1. SciPy offers no per-iteration callback here, so observers
receive ``on_start`` and ``on_end`` only; a cancel token is honored at the
next residual evaluation.

Kramer Harrison, 2026
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy import optimize

from optiland.optimization.failure import ConfigurationError
from optiland.optimization.managed.base import ProblemEvaluation, RunMonitor
from optiland.optimization.state import OptimizationResult

if TYPE_CHECKING:
    from optiland.optimization.observers.cancel import CancelToken
    from optiland.optimization.problem import OptimizationProblem

#: ``method_choice`` values of ``scipy.optimize.least_squares``.
LEAST_SQUARES_METHODS: tuple[str, ...] = ("lm", "trf", "dogbox")


class _Cancelled(Exception):
    """Raised inside the residual function to abandon a cancelled run."""


def run_scipy_least_squares(
    problem: OptimizationProblem,
    monitor: RunMonitor,
    *,
    maxiter: int | None,
    tol: float,
    on_failure: str,
    method_choice: str = "lm",
    cancel_token: CancelToken | None = None,
) -> OptimizationResult:
    """Minimize ``sum(r ** 2)`` with ``scipy.optimize.least_squares``.

    ``"lm"`` is replaced by ``"trf"`` when there are fewer residuals than
    variables or any variable is bounded, since ``"lm"`` supports neither.

    Args:
        problem: The optimization problem.
        monitor: Drives the observers (``on_start``/``on_end`` only).
        maxiter: SciPy ``max_nfev`` (residual evaluations), or None.
        tol: SciPy ``ftol``.
        on_failure: ``"reject"`` or ``"raise"``.
        method_choice: One of :data:`LEAST_SQUARES_METHODS`.
        cancel_token: Cancels the run at the next residual evaluation.

    Returns:
        The result; the problem's optics are left at the best finite point.

    Raises:
        ConfigurationError: If ``method_choice`` is unknown.
    """
    if method_choice not in LEAST_SQUARES_METHODS:
        raise ConfigurationError(
            f"Unknown method_choice {method_choice!r}; expected one of "
            f"{list(LEAST_SQUARES_METHODS)}."
        )
    evaluation = ProblemEvaluation(problem, on_failure)
    x0 = evaluation.read_x()
    initial = evaluation.check_start(x0)
    lower, upper = evaluation.bounds()
    bounded = bool(np.any(np.isfinite(lower)) or np.any(np.isfinite(upper)))
    if method_choice == "lm" and (bounded or len(problem.operands) < x0.size):
        method_choice = "trf"
    if method_choice == "lm":
        bounds = (-np.inf, np.inf)
    else:
        bounds = (lower, upper)
        # SciPy requires a start strictly inside the box
        eps = np.finfo(float).eps
        x0 = np.clip(
            x0, lower + eps * np.isfinite(lower), upper - eps * np.isfinite(upper)
        )
    monitor.start(x0, initial, evaluation.nfev)

    def residuals(x: np.ndarray) -> np.ndarray:
        if cancel_token is not None and cancel_token.cancelled:
            raise _Cancelled
        return evaluation.residuals(x)

    cancelled = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            res = optimize.least_squares(
                residuals,
                x0,
                method=method_choice,
                bounds=bounds,
                max_nfev=maxiter,
                ftol=tol,
            )
            x = np.asarray(res.x, dtype=float)
            status, message, success = int(res.status), str(res.message), res.success
        except _Cancelled:
            cancelled = True
            x, status, message, success = evaluation.best_x, -2, "cancelled", False

    # The last evaluation is not necessarily the best one
    merit = evaluation.merit(x)
    if evaluation.best_x is not None and merit > evaluation.best_merit:
        x, merit = evaluation.best_x, evaluation.best_merit
    evaluation.write_x(x)
    if cancelled:
        message = "cancelled by user"
    result = OptimizationResult(
        x=x,
        fun=float(merit),
        success=bool(success),
        status=status,
        message=message,
        nfev=evaluation.nfev,
        nit=0,
        stop_reason=message,
        n_evals=evaluation.nfev,
        history=None,
        method="least_squares",
    )
    return monitor.finish(result)
