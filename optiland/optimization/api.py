"""Facade: ``optiland.optimization.minimize()``

Experimental (opt v2 M5, public until M10). One entry point over the native
damped least squares and the SciPy local solvers. The legacy optimizer classes
are unchanged.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from optiland.optimization.failure import ConfigurationError, normalize_failure_mode
from optiland.optimization.managed import (
    SCIPY_LOCAL_METHODS,
    TORCH_METHODS,
    run_scipy_least_squares,
    run_scipy_local,
    run_torch_first_order,
)
from optiland.optimization.managed.base import RunMonitor
from optiland.optimization.observers.console import ConsoleObserver
from optiland.optimization.state import OptimizationResult

if TYPE_CHECKING:
    from optiland.optimization.observers.base import Observer
    from optiland.optimization.observers.cancel import CancelToken
    from optiland.optimization.problem import OptimizationProblem
    from optiland.optimization.stopping.base import StoppingCriterion

_ALL_METHODS = (
    frozenset({"auto", "dls", "least_squares"}) | SCIPY_LOCAL_METHODS | TORCH_METHODS
)

#: Options accepted by each method; anything else raises ConfigurationError.
_DLS_OPTIONS = frozenset(
    {"scheme", "rel_step", "aim_policy", "jacobian_update", "on_failure"}
)
_SCIPY_OPTIONS = frozenset({"on_failure"})
_LEAST_SQUARES_OPTIONS = frozenset({"on_failure", "method_choice"})
_TORCH_OPTIONS = frozenset({"lr", "gamma"})


def minimize(
    problem: OptimizationProblem,
    method: str = "auto",
    *,
    maxiter: int | None = None,
    tol: float = 1e-3,
    stop: StoppingCriterion | None = None,
    observers: list[Observer] | None = None,
    cancel_token: CancelToken | None = None,
    disp: bool = False,
    **options: Any,
) -> OptimizationResult:
    """Minimize an optimization problem. *Experimental.*

    The optic is **mutated in place**: on return it holds the best accepted
    design, ``result.x``. Copy the optic first to keep the starting design.

    Args:
        problem: The problem to solve.
        method: ``"auto"`` (default) picks ``"dls"`` when all variables are
            continuous, else ``"l-bfgs-b"``. ``"dls"`` is Optiland's native
            damped least squares. SciPy local solvers by name (``"l-bfgs-b"``,
            ``"bfgs"``, ``"slsqp"``, ``"trust-constr"``, ``"cobyla"``,
            ``"nelder-mead"``, ``"cg"``, ``"newton-cg"``, ``"tnc"``,
            ``"powell"``) minimize the merit ``sum(r ** 2)``;
            ``"least_squares"`` is ``scipy.optimize.least_squares`` on the
            residuals. ``"adam"``/``"sgd"`` are torch first-order methods
            (torch backend only). Global SciPy methods are only available
            through the legacy classes.
        maxiter: Iteration budget. For ``"dls"`` and ``"least_squares"`` it
            counts residual evaluations, Jacobian columns included; for the
            SciPy local solvers it is SciPy's ``maxiter``; for ``"adam"``/``"sgd"``
            it is the number of steps (default 100). None uses the method's
            default.
        tol: Convergence tolerance: ``ftol`` for ``"dls"`` and
            ``"least_squares"``, SciPy ``tol`` for the local solvers. Ignored by
            ``"adam"``/``"sgd"`` (fixed step count).
        stop: Extra stopping criterion, checked after every accepted step on
            top of the method's own tests. Not checked by ``"least_squares"``.
        observers: Observers notified of the run.
        cancel_token: Cancels the run cleanly once set from any thread.
        disp: Attach a :class:`~optiland.optimization.observers.ConsoleObserver`.
        **options: Method-specific options. ``"dls"``: ``scheme``
            (``"forward"``/``"central"``), ``rel_step``, ``aim_policy``
            (``"exact"``/``"frozen"``), ``jacobian_update``
            (``"exact"``/``"auto"``/``"broyden"``), ``on_failure``
            (``"reject"``/``"raise"``). SciPy local solvers: ``on_failure``.
            ``"adam"``/``"sgd"``: ``lr``, ``gamma``.
            ``"least_squares"``: ``on_failure`` and ``method_choice``
            (``"lm"``/``"trf"``/``"dogbox"``).

    Returns:
        An :class:`~optiland.optimization.state.OptimizationResult`, a SciPy
        ``OptimizeResult`` with ``x`` (scaled variables), ``fun`` (final
        merit), ``success``, ``message``, ``nfev``, ``nit``, plus
        ``stop_reason``, ``n_evals``, ``history`` (None without a
        ``HistoryObserver``) and ``method`` (the resolved method).

    Raises:
        ConfigurationError: For an unknown method or option, a problem
            without variables, or a starting design with non-finite residuals.
    """
    resolved = _resolve_method(method, problem)
    _validate_options(resolved, options)
    if not problem.variables:
        raise ConfigurationError("The problem has no variables.")
    if maxiter is not None and maxiter < 1:
        raise ConfigurationError(f"maxiter must be positive, got {maxiter}.")
    if not tol > 0:
        raise ConfigurationError(f"tol must be positive, got {tol}.")

    observer_list = list(observers or [])
    if disp:
        observer_list.append(ConsoleObserver())
    if problem.initial_value == 0.0:
        problem.initial_value = problem.sum_squared()

    if resolved == "dls":
        return _run_dls(
            problem, resolved, maxiter, tol, stop, observer_list, cancel_token, options
        )
    monitor = RunMonitor(observer_list, stop, cancel_token)
    if resolved in TORCH_METHODS:
        return run_torch_first_order(
            problem,
            resolved,
            monitor,
            maxiter=maxiter,
            lr=options.get("lr", 1e-2),
            gamma=options.get("gamma", 0.99),
        )
    on_failure = normalize_failure_mode(options.get("on_failure"))
    if resolved == "least_squares":
        return run_scipy_least_squares(
            problem,
            monitor,
            maxiter=maxiter,
            tol=tol,
            on_failure=on_failure,
            method_choice=options.get("method_choice", "lm"),
            cancel_token=cancel_token,
        )
    return run_scipy_local(
        problem, resolved, monitor, maxiter=maxiter, tol=tol, on_failure=on_failure
    )


def _resolve_method(method: str, problem: OptimizationProblem) -> str:
    """Map ``method`` to a concrete method name, or raise."""
    name = str(method).lower()
    if name not in _ALL_METHODS:
        raise ConfigurationError(
            f"Unknown method {method!r}; expected one of {sorted(_ALL_METHODS)}. "
            "Global SciPy methods, GlassExpert, OrthogonalDescent and "
            "ParticleSwarm are only available through their own classes."
        )
    if name == "auto":
        continuous = not any(isinstance(var.value, str) for var in problem.variables)
        return "dls" if continuous else "l-bfgs-b"
    return name


def _validate_options(method: str, options: dict[str, Any]) -> None:
    """Raise ConfigurationError for an option the method does not take."""
    if method == "dls":
        allowed = _DLS_OPTIONS
    elif method == "least_squares":
        allowed = _LEAST_SQUARES_OPTIONS
    elif method in TORCH_METHODS:
        allowed = _TORCH_OPTIONS
    else:
        allowed = _SCIPY_OPTIONS
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ConfigurationError(
            f"Unknown option(s) {unknown} for method {method!r}; "
            f"valid options: {sorted(allowed)}."
        )


def _run_dls(
    problem: OptimizationProblem,
    method: str,
    maxiter: int | None,
    tol: float,
    stop: StoppingCriterion | None,
    observers: list[Observer],
    cancel_token: CancelToken | None,
    options: dict[str, Any],
) -> OptimizationResult:
    """Run the native DLS core and shape its result for ``minimize()``."""
    from optiland.optimization.evaluators import FiniteDiffEvaluator
    from optiland.optimization.native import dls

    evaluator = FiniteDiffEvaluator(
        problem,
        **{
            key: options[key]
            for key in ("scheme", "rel_step", "aim_policy", "on_failure")
            if key in options
        },
    )
    raw = dls(
        evaluator,
        max_nfev=maxiter,
        ftol=tol,
        jacobian_update=options.get("jacobian_update", "exact"),
        stop=stop,
        observers=observers,
        cancel_token=cancel_token,
    )
    result = OptimizationResult(raw)
    result.update(fun=float(raw.fun @ raw.fun), residuals=raw.fun, method=method)
    return result


__all__ = ["minimize"]
