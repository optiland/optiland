"""Measurement harness for the optimization benchmarks.

Metrics per (problem, method): single merit-evaluation time, forward-FD
Jacobian time, time to converge, iterations, residual evaluations, trace
calls, final merit and crash/NaN count. Counts come from
:mod:`optiland._instrument`; timings are wall-clock minima and are meant for
local before/after comparison only, never for CI assertions (decision V15).

Kramer Harrison, 2026
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland import _instrument
from optiland.optimization import LeastSquares, OptimizationProblem

DEFAULT_MAXITER = 100

if TYPE_CHECKING:
    from collections.abc import Callable


def _run_scipy(method_choice: str) -> Callable[[OptimizationProblem, int], object]:
    """Build a runner for the legacy ``LeastSquares`` optimizer."""

    def run(problem: OptimizationProblem, maxiter: int):
        return LeastSquares(problem).optimize(
            maxiter=maxiter, method_choice=method_choice
        )

    return run


# Later PRs register their methods here (e.g. "dls").
METHODS: dict[str, Callable[[OptimizationProblem, int], object]] = {
    "scipy_lm": _run_scipy("lm"),
    "scipy_trf": _run_scipy("trf"),
}


@dataclass
class BenchmarkResult:
    """One (problem, method) measurement."""

    problem: str
    method: str
    num_variables: int
    num_operands: int
    eval_time_s: float
    jacobian_time_s: float
    converge_time_s: float | None
    iterations: int | None
    residual_evals: int | None
    trace_calls: int | None
    initial_merit: float
    final_merit: float | None
    crashes: int
    nan_count: int

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict."""
        return asdict(self)


def time_evaluation(problem: OptimizationProblem, repeats: int = 5) -> float:
    """Minimum wall time of one ``sum_squared`` call."""
    problem.sum_squared()  # warm caches
    best = math.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        problem.sum_squared()
        best = min(best, time.perf_counter() - t0)
    return best


def time_fd_jacobian(problem: OptimizationProblem, repeats: int = 2) -> float:
    """Minimum wall time of a forward-difference Jacobian of the residuals."""
    optimizer = LeastSquares(problem)
    x0 = be.to_numpy([var.value for var in problem.variables]).astype(float)
    step = 1e-8
    best = math.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        f0 = optimizer._compute_residuals_vector(x0)
        for j in range(len(x0)):
            x = x0.copy()
            x[j] += step * max(1.0, abs(x0[j]))
            optimizer._compute_residuals_vector(x)
        best = min(best, time.perf_counter() - t0)
        _ = f0
    optimizer._compute_residuals_vector(x0)  # restore the starting design
    return best


def run_benchmark(
    problem_name: str,
    factory: Callable[[], tuple[object, OptimizationProblem]],
    method_name: str,
    maxiter: int = DEFAULT_MAXITER,
) -> BenchmarkResult:
    """Measure one problem with one method on freshly built systems."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, problem = factory()
        eval_time = time_evaluation(problem)
        jac_time = time_fd_jacobian(problem)
        initial = float(be.to_numpy(problem.sum_squared()))

        # Fresh system so the convergence run starts from the initial design.
        _, problem = factory()
        crashes = 0
        nan_count = 0
        result = None
        t0 = time.perf_counter()
        with _instrument.counting() as counts:
            try:
                result = METHODS[method_name](problem, maxiter)
            except Exception:
                crashes = 1
        elapsed = time.perf_counter() - t0
        trace_calls = counts["trace"] + counts["trace_generic"]

        final = None
        if not crashes:
            final = float(be.to_numpy(problem.sum_squared()))
            if not math.isfinite(final):
                nan_count = 1

    return BenchmarkResult(
        problem=problem_name,
        method=method_name,
        num_variables=len(problem.variables.variables),
        num_operands=len(problem.operands.operands),
        eval_time_s=eval_time,
        jacobian_time_s=jac_time,
        converge_time_s=None if crashes else elapsed,
        iterations=_attr_int(result, "nit") or _attr_int(result, "njev"),
        residual_evals=_attr_int(result, "nfev"),
        trace_calls=None if crashes else trace_calls,
        initial_merit=initial,
        final_merit=final,
        crashes=crashes,
        nan_count=nan_count,
    )


def _attr_int(result: object, name: str) -> int | None:
    """Integer attribute of a solver result, or ``None`` when absent."""
    value = getattr(result, name, None)
    if value is None or not np.isfinite(value):
        return None
    return int(value)


def to_markdown(results: list[BenchmarkResult]) -> str:
    """Render results as a markdown table for pasting into a PR description."""

    def fmt(value, spec=".4g"):
        return "-" if value is None else format(value, spec)

    header = (
        "| problem | method | vars | ops | eval ms | FD jac s | converge s | "
        "iters | resid evals | traces | final merit | crash | NaN |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    rows = [
        f"| {r.problem} | {r.method} | {r.num_variables} | {r.num_operands} | "
        f"{r.eval_time_s * 1e3:.1f} | {r.jacobian_time_s:.2f} | "
        f"{fmt(r.converge_time_s, '.2f')} | {fmt(r.iterations, 'd')} | "
        f"{fmt(r.residual_evals, 'd')} | {fmt(r.trace_calls, 'd')} | "
        f"{fmt(r.final_merit)} | {r.crashes} | {r.nan_count} |"
        for r in results
    ]
    return "\n".join([header, *rows])
