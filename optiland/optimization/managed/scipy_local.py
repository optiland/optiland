"""SciPy local minimizers behind ``minimize()``.

Private core: calls ``scipy.optimize.minimize`` directly on the scalar merit
``sum(r ** 2)``, with the scale-aware failure penalty of PR1.1. Observers and
stopping criteria run from the SciPy callback.

Kramer Harrison, 2026
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import optimize

from optiland.optimization.managed.base import ProblemEvaluation, RunMonitor
from optiland.optimization.state import OptimizationResult

if TYPE_CHECKING:
    from optiland.optimization.problem import OptimizationProblem

#: SciPy ``minimize`` methods reachable through ``minimize()``.
SCIPY_LOCAL_METHODS: frozenset[str] = frozenset(
    {
        "l-bfgs-b",
        "bfgs",
        "slsqp",
        "trust-constr",
        "cobyla",
        "nelder-mead",
        "cg",
        "newton-cg",
        "tnc",
        "powell",
    }
)


def run_scipy_local(
    problem: OptimizationProblem,
    method: str,
    monitor: RunMonitor,
    *,
    maxiter: int | None,
    tol: float,
    on_failure: str,
) -> OptimizationResult:
    """Minimize the merit with a SciPy local method.

    Args:
        problem: The optimization problem.
        method: A member of :data:`SCIPY_LOCAL_METHODS`.
        monitor: Drives the observers and stopping criteria.
        maxiter: SciPy ``maxiter`` (iterations), or None for its default.
        tol: SciPy ``tol``.
        on_failure: ``"reject"`` or ``"raise"``.

    Returns:
        The result; the problem's optics are left at ``result.x``.
    """
    evaluation = ProblemEvaluation(problem, on_failure)
    x0 = evaluation.read_x()
    initial = evaluation.check_start(x0)
    bounds = tuple(var.bounds for var in problem.variables)
    monitor.start(x0, initial, evaluation.nfev)

    def callback(xk: Any, *_: Any) -> None:
        monitor.step(xk, evaluation.merit_at(xk), evaluation.nfev)

    options = {} if maxiter is None else {"maxiter": int(maxiter)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=optimize.OptimizeWarning)
        res = optimize.minimize(
            evaluation.merit,
            x0,
            method=method,
            bounds=bounds,
            options=options,
            tol=tol,
            callback=callback,
        )

    x = np.asarray(res.x, dtype=float)
    evaluation.write_x(x)
    message = monitor.stop_reason or str(res.message)
    result = OptimizationResult(
        x=x,
        fun=float(res.fun),
        success=bool(res.success) or monitor.stop_reason is not None,
        status=int(res.status),
        message=message,
        nfev=evaluation.nfev,
        nit=int(getattr(res, "nit", monitor.state.iteration)),
        stop_reason=message,
        n_evals=evaluation.nfev,
        history=None,
        method=method,
    )
    return monitor.finish(result)
