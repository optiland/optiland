"""Public facade: ``optiland.optimization.minimize()``

The **single** public wiring point for the optimization subpackage.
``OptimizationProblem`` stays a pure merit/data container — no solver
coupling lives there.

Method routing (``method`` arg → optimizer family):

- ``"dls"`` / ``"lm"`` → native LevenbergMarquardt, stepped, both backends
- ``"gauss_newton"`` → native GaussNewton, stepped, both backends
- ``"adam"`` / ``"sgd"`` → TorchOptimizer, stepped, torch only
- ``"l-bfgs-b"`` / ``"bfgs"`` / ``"slsqp"`` / ``"trust-constr"`` / etc.
  → ScipyLocalAdapter, managed, numpy
- ``"least_squares"`` → ScipyLeastSquaresAdapter, managed, numpy
- ``"differential_evolution"`` / ``"dual_annealing"`` / ``"shgo"``
  / ``"basin_hopping"`` → ScipyGlobalAdapter, managed, numpy

``method="auto"`` selects ``"adam"`` under torch, ``"dls"`` for all-equality
numpy problems (m ≥ n), or ``"l-bfgs-b"`` otherwise (R3-Q4).

Special optimizers (D10): ``GlassExpert``, ``OrthogonalDescent``, and
``ParticleSwarm`` are **not** accessible via this facade.  Use them directly.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.optimization.errors import ConfigurationError
from optiland.optimization.state import OptimizationResult  # noqa: TC001

if TYPE_CHECKING:
    from optiland.optimization.constraints.base import ConstraintStrategy
    from optiland.optimization.control.base import StepController
    from optiland.optimization.observers.base import Observer
    from optiland.optimization.observers.cancel import CancelToken
    from optiland.optimization.problem import OptimizationProblem
    from optiland.optimization.stopping.base import StoppingCriterion

# ---------------------------------------------------------------------------
# Method routing tables
# ---------------------------------------------------------------------------

_TORCH_METHODS = frozenset({"adam", "sgd"})
_SCIPY_LOCAL_METHODS = frozenset(
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
_SCIPY_LS_METHODS = frozenset({"least_squares"})
_SCIPY_GLOBAL_METHODS = frozenset(
    {"differential_evolution", "dual_annealing", "shgo", "basin_hopping"}
)
_NATIVE_STEPPED_METHODS = frozenset({"dls", "lm", "gauss_newton"})

_ALL_METHODS = (
    _TORCH_METHODS
    | _SCIPY_LOCAL_METHODS
    | _SCIPY_LS_METHODS
    | _SCIPY_GLOBAL_METHODS
    | _NATIVE_STEPPED_METHODS
    | {"auto"}
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def minimize(
    problem: OptimizationProblem,
    method: str = "auto",
    *,
    constraints: ConstraintStrategy | list | None = None,
    bounds: bool = True,
    controller: StepController | None = None,
    stop: StoppingCriterion | None = None,
    observers: list[Observer] | None = None,
    cancel_token: CancelToken | None = None,
    maxiter: int = 1000,
    tol: float = 1e-3,
    disp: bool = True,
    **method_options: Any,
) -> OptimizationResult:
    """Run an optimization and return an ``OptimizationResult``.

    Args:
        problem: The ``OptimizationProblem`` to solve.
        method: Optimizer selection string or ``"auto"`` (see module docstring).
        constraints: A ``ConstraintStrategy``, a list of them, or None.
            Inequality operands (min_val/max_val) are already expressed in the
            merit function and need not be passed here.
        bounds: If True (default), honor ``var.bounds`` via
            ``BoxBoundsStrategy`` for stepped optimizers.
        controller: Step controller (stepped family only).  Defaults to
            ``IdentityController`` for torch; ``LevenbergController`` for
            native LM (Phase 2).
        stop: Composite stopping criterion.  Defaults to
            ``MaxIter(maxiter) | CostTolerance(tol)``.
        observers: Additional observers to attach (e.g. ``HistoryObserver``).
        cancel_token: ``CancelToken`` for GUI/thread-based early stop.
        maxiter: Maximum iterations (used in default stopping criterion).
        tol: Convergence tolerance (used in default stopping criterion).
        disp: If True, attach a ``ConsoleObserver`` (default True).
        **method_options: Method-specific keyword arguments forwarded to the
            optimizer (e.g. ``lr``, ``gamma``, ``n_steps`` for torch;
            ``method_choice`` for ``least_squares``).

    Returns:
        An ``OptimizationResult`` with rich fields plus ``.fun`` / ``.x`` /
        ``.success`` / ``.message`` for SciPy-compatible duck-typing.

    Raises:
        ConfigurationError: If the method/backend/strategy combination is
            invalid (checked before any evaluation).

    Note:
        ``GlassExpert``, ``OrthogonalDescent``, and ``ParticleSwarm`` are not
        accessible via this facade (D10).  Use those classes directly.
    """
    backend = be.get_backend()

    # ---- Resolve method ------------------------------------------------
    resolved = _resolve_method(method, problem, backend)

    # ---- Validate backend / capability conflicts ----------------------
    _validate(resolved, backend, constraints, controller)

    # ---- Build evaluator -----------------------------------------------
    evaluator = _build_evaluator(problem, backend, resolved)

    # ---- Build stopping criterion --------------------------------------
    from optiland.optimization.stopping.criteria import CostTolerance, MaxIter

    if stop is None:
        stop = MaxIter(maxiter) | CostTolerance(tol)

    # ---- Build observer list ------------------------------------------
    all_observers: list[Observer] = list(observers or [])
    if disp:
        from optiland.optimization.observers.logging import ConsoleObserver

        all_observers.append(ConsoleObserver())
    if cancel_token is not None:
        from optiland.optimization.observers.cancel import CancelObserver

        all_observers.append(CancelObserver(cancel_token))

    # ---- Prepare constraints ------------------------------------------
    effective_constraints = _build_constraints(constraints, bounds, evaluator)

    # ---- Record initial merit ----------------------------------------
    if problem.initial_value == 0.0:
        try:
            from optiland.optimization.evaluators.finite_difference import (
                FiniteDiffEvaluator,
            )

            if isinstance(evaluator, FiniteDiffEvaluator):
                problem.initial_value = evaluator.value(evaluator.read_x())
            else:
                problem.initial_value = float(be.to_numpy(problem.sum_squared()))
        except Exception:  # noqa: BLE001
            problem.initial_value = float(be.to_numpy(problem.sum_squared()))

    initial_value = float(be.to_numpy(problem.sum_squared()))

    # ---- Dispatch to driver -------------------------------------------
    if resolved in _NATIVE_STEPPED_METHODS:
        return _run_stepped_native(
            resolved,
            problem,
            evaluator,
            controller,
            effective_constraints,
            stop,
            all_observers,
            initial_value,
            method_options,
        )
    if resolved in _TORCH_METHODS:
        return _run_stepped_torch(
            resolved,
            problem,
            evaluator,
            controller,
            effective_constraints,
            stop,
            all_observers,
            initial_value,
            method_options,
        )
    return _run_managed(
        resolved,
        problem,
        evaluator,
        effective_constraints,
        stop,
        all_observers,
        initial_value,
        method_options,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_method(method: str, problem: OptimizationProblem, backend: str) -> str:
    method = method.lower()
    if method == "auto":
        if backend == "torch":
            return "adam"
        # numpy: native DLS for all-equality and m >= n (R3-Q4)
        ops = list(problem.operands)
        n_vars = len(list(problem.variables))
        all_equality = all(op.target is not None for op in ops)
        if all_equality and len(ops) >= n_vars:
            return "dls"
        return "l-bfgs-b"
    if method not in _ALL_METHODS:
        raise ConfigurationError(
            f"Unknown method {method!r}. "
            f"Valid options: {sorted(_ALL_METHODS)}. "
            "Note: GlassExpert, OrthogonalDescent, ParticleSwarm are not "
            "accessible via minimize() — use them directly."
        )
    return method


def _validate(method: str, backend: str, constraints: Any, controller: Any) -> None:
    if method in _TORCH_METHODS and backend != "torch":
        raise ConfigurationError(
            f"Method {method!r} requires the torch backend. "
            "Call be.set_backend('torch') before calling minimize()."
        )
    # NullSpaceStrategy requires a stepped optimizer (PER_STEP_CONSTRAINTS).
    # Reject it for managed methods early to give an actionable message.
    _managed_methods = _SCIPY_LOCAL_METHODS | _SCIPY_LS_METHODS | _SCIPY_GLOBAL_METHODS
    if method in _managed_methods and constraints is not None:
        from optiland.optimization.constraints.null_space import NullSpaceStrategy

        candidates = constraints if isinstance(constraints, list) else [constraints]
        if any(isinstance(c, NullSpaceStrategy) for c in candidates):
            raise ConfigurationError(
                "NullSpaceStrategy requires a stepped optimizer (e.g. 'dls', 'lm', "
                "'gauss_newton') that declares the PER_STEP_CONSTRAINTS capability. "
                f"Method {method!r} is a managed (SciPy) optimizer and does not "
                "support per-step constraint projection."
            )


def _build_evaluator(problem: OptimizationProblem, backend: str, method: str) -> Any:
    if backend == "torch" and method in (_TORCH_METHODS | _NATIVE_STEPPED_METHODS):
        from optiland.optimization.evaluators.autograd import AutogradEvaluator

        return AutogradEvaluator(problem)
    from optiland.optimization.evaluators.finite_difference import FiniteDiffEvaluator

    return FiniteDiffEvaluator(problem)


def _build_constraints(
    constraints: Any,
    bounds: bool,
    evaluator: Any,
) -> Any:
    from optiland.optimization.constraints.base import CompositeStrategy
    from optiland.optimization.constraints.bounds import BoxBoundsStrategy

    strategies: list[Any] = []

    if bounds:
        box = BoxBoundsStrategy()
        box.prepare(evaluator, evaluator.problem.variables)
        strategies.append(box)

    if constraints is not None:
        if isinstance(constraints, list):
            strategies.extend(constraints)
        else:
            strategies.append(constraints)

    if not strategies:
        return None
    if len(strategies) == 1:
        return strategies[0]
    return CompositeStrategy(strategies)


def _build_controller(method: str, controller: Any) -> Any:
    if controller is not None:
        return controller
    if method in ("dls", "lm"):
        from optiland.optimization.control.levenberg import LevenbergController

        return LevenbergController()
    from optiland.optimization.control.identity import IdentityController

    return IdentityController()


def _run_stepped_native(
    method: str,
    problem: OptimizationProblem,
    evaluator: Any,
    controller: Any,
    constraints: Any,
    stop: Any,
    observers: list,
    initial_value: float,
    method_options: dict,
) -> OptimizationResult:
    from optiland.optimization.drivers import SteppedDriver
    from optiland.optimization.native.least_squares import (
        GaussNewton,
        LevenbergMarquardt,
    )

    optimizer = LevenbergMarquardt() if method in ("dls", "lm") else GaussNewton()

    ctrl = _build_controller(method, controller)

    driver = SteppedDriver()
    x0 = evaluator.read_x()
    return driver.run(
        optimizer,
        evaluator,
        x0,
        controller=ctrl,
        constraints=constraints,
        criteria=stop,
        observers=observers,
        initial_value=initial_value,
        method=method,
    )


def _run_stepped_torch(
    method: str,
    problem: OptimizationProblem,
    evaluator: Any,
    controller: Any,
    constraints: Any,
    stop: Any,
    observers: list,
    initial_value: float,
    method_options: dict,
) -> OptimizationResult:
    from optiland.optimization.drivers import SteppedDriver
    from optiland.optimization.native.torch_opt import TorchOptimizer

    lr = method_options.pop("lr", 1e-2)
    gamma = method_options.pop("gamma", 0.99)

    optimizer = TorchOptimizer(optimizer_name=method, lr=lr, gamma=gamma)
    ctrl = _build_controller(method, controller)

    driver = SteppedDriver()
    x0 = evaluator.read_x()
    return driver.run(
        optimizer,
        evaluator,
        x0,
        controller=ctrl,
        constraints=constraints,
        criteria=stop,
        observers=observers,
        initial_value=initial_value,
        method=method,
    )


def _run_managed(
    method: str,
    problem: OptimizationProblem,
    evaluator: Any,
    constraints: Any,
    stop: Any,
    observers: list,
    initial_value: float,
    method_options: dict,
) -> OptimizationResult:
    from optiland.optimization.drivers import ManagedDriver

    adapter = _build_managed_adapter(method, problem, method_options)

    driver = ManagedDriver()
    return driver.run(
        adapter,
        evaluator,
        criteria=stop,
        observers=observers,
        initial_value=initial_value,
        method=method,
        method_options=method_options,
    )


def _build_managed_adapter(
    method: str, problem: OptimizationProblem, method_options: dict
) -> Any:
    if method in _SCIPY_LOCAL_METHODS:
        from optiland.optimization.managed.scipy_local import ScipyLocalAdapter

        return ScipyLocalAdapter(problem, method=method)

    if method in _SCIPY_LS_METHODS:
        method_choice = method_options.pop("method_choice", "lm")
        from optiland.optimization.managed.scipy_least_squares import (
            ScipyLeastSquaresAdapter,
        )

        return ScipyLeastSquaresAdapter(problem, method_choice=method_choice)

    if method in _SCIPY_GLOBAL_METHODS:
        from optiland.optimization.managed.scipy_global import ScipyGlobalAdapter

        return ScipyGlobalAdapter(problem, algorithm=method)

    raise ConfigurationError(  # pragma: no cover
        f"No adapter available for method {method!r}"
    )
