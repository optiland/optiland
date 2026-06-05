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
numpy problems (m ≥ n), or ``"l-bfgs-b"`` otherwise.

Special optimizers: ``GlassExpert``, ``OrthogonalDescent``, and
``ParticleSwarm`` are **not** accessible via this facade.  Use them directly.

Kramer Harrison, 2026
"""

from __future__ import annotations

import warnings
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

_MANAGED_METHODS = _SCIPY_LOCAL_METHODS | _SCIPY_LS_METHODS | _SCIPY_GLOBAL_METHODS


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
            native LM.
        stop: Composite stopping criterion.  Defaults to
            ``MaxIter(maxiter) | CostTolerance(tol)``.
        observers: Additional observers to attach (e.g. ``HistoryObserver``).
        cancel_token: ``CancelToken`` for GUI/thread-based early stop.
        maxiter: Maximum iterations (used in default stopping criterion).
        tol: Convergence tolerance (used in default stopping criterion).
            Controls ``CostTolerance`` (relative merit change) for
            DLS/LM/scipy-LS/scipy-local methods; ``GradNormTolerance`` for
            first-order torch methods (adam/sgd).  The active criterion and
            threshold are recorded in ``result.status``.
        disp: If True, attach a ``ConsoleObserver`` (default True).  When
            ``method="auto"``, the resolved method is also printed at start.
        **method_options: Method-specific keyword arguments forwarded to the
            optimizer (e.g. ``lr``, ``gamma`` for torch; ``method_choice`` for
            ``least_squares``).  Cross-cutting options:

            * ``on_failure`` ∈ ``{"reject"(default), "raise", "penalty"}`` —
              policy for ray-trace failures / non-finite merits (§3.5).
            * ``scheme`` ∈ ``{"forward"(default), "central"}``, ``rel_step``,
              ``abs_step`` — finite-difference options for the numpy path
              (§4.6).
            * ``linear_solver`` ∈ ``{"normal"(default), "qr"}`` — KKT linear
              solve for hard-constrained ``dls``/``lm`` runs (§4.3).

            Hard equality/inequality constraints are read automatically from
            ``problem.constraints`` (declared via
            :meth:`OptimizationProblem.add_constraint`) and solved with a KKT
            active-set step; they require ``method="dls"`` or ``"lm"``.

    Returns:
        An ``OptimizationResult`` with rich fields plus ``.fun`` / ``.x`` /
        ``.success`` / ``.message`` for SciPy-compatible duck-typing.
        ``result.method`` holds the resolved method string.
        ``result.resolved_from`` is ``"auto"`` when auto-selection was used,
        ``None`` when an explicit method was passed.

    Raises:
        ConfigurationError: If the method/backend/strategy combination is
            impossible (checked before any evaluation).  Inapplicable-but-
            harmless options raise ``UserWarning`` and are silently ignored.

    Side effects:
        ``minimize()`` mutates ``problem``'s optic **in place** — on return
        the optic holds the optimized design.  ``result.x`` is a copy of the
        final parameter vector.  To preserve the starting design, snapshot it
        yourself before calling.

    Note:
        ``GlassExpert``, ``OrthogonalDescent``, and ``ParticleSwarm`` are not
        accessible via this facade.  Use those classes directly.

    Choosing a method:
        **Core rule:** does Optiland run the loop, or do you?  If Optiland →
        ``minimize()``.  If you own the loop (custom loss, batched data, optic
        as a layer) → ``optiland.ml.OpticalSystemModule``.

        Families and when to pick them:

        * ``"dls"`` / ``"lm"`` — native Levenberg-Marquardt damped least
          squares.  Best for classical lens design (CODE V / Zemax style),
          equality targets, m ≥ n, both numpy and torch.  Default when
          ``"auto"`` resolves on a qualifying numpy problem.  ``tol`` controls
          ``CostTolerance`` (relative merit change).
        * ``"gauss_newton"`` — undamped Gauss-Newton; use when close to a
          solution (no λ damping overhead).
        * ``"adam"`` / ``"sgd"`` — torch first-order, GPU-ready.  ``"auto"``
          resolves here under torch.  ``tol`` controls ``GradNormTolerance``.
        * ``"l-bfgs-b"`` / ``"slsqp"`` / ``"trust-constr"`` / … — managed
          SciPy local solvers.  Good for bound/inequality-dominated numpy
          problems.  ``"auto"`` falls back here for numpy without m ≥ n.
        * ``"least_squares"`` — managed SciPy ``scipy.optimize.least_squares``.
          Residual-based; use when you want trust-region / dogbox methods.
        * ``"differential_evolution"`` / ``"dual_annealing"`` / … — managed
          SciPy global solvers.  No gradient required; ignores bounds kwarg.
        * For discrete glass search or derivative-free global: use
          ``GlassExpert`` / ``ParticleSwarm`` / ``OrthogonalDescent`` directly
          (not accessible via this facade).
    """
    backend = be.get_backend()

    # ---- Resolve method ------------------------------------------------
    resolved, resolved_from = _resolve_method(method, problem, backend)

    # ---- Validate backend / capability conflicts ----------------------
    _validate(resolved, backend, constraints, controller)
    _validate_hard_constraints(resolved, problem)

    # ---- Announce auto-resolution when disp=True ----------------------
    if disp and resolved_from == "auto":
        print(f"auto → {resolved}")

    # ---- Failure policy (D16, §3.5) -----------------------------------
    from optiland.optimization.failure import normalize_failure_mode

    on_failure = normalize_failure_mode(method_options.pop("on_failure", None))

    # ---- Finite-difference scheme exposure (D17, §4.6) ----------------
    fd_options = {
        "scheme": method_options.pop("scheme", "forward"),
        "rel_step": method_options.pop("rel_step", 1e-5),
        "abs_step": method_options.pop("abs_step", 1e-8),
    }
    # ---- Build evaluator -----------------------------------------------
    evaluator = _build_evaluator(
        problem, backend, resolved, on_failure=on_failure, fd_options=fd_options
    )

    # ---- Build stopping criterion --------------------------------------
    from optiland.optimization.stopping.criteria import (
        CostTolerance,
        GradNormTolerance,
        MaxIter,
    )

    if stop is None:
        if resolved in _TORCH_METHODS:
            stop = MaxIter(maxiter) | GradNormTolerance(tol)
        else:
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
            resolved_from=resolved_from,
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
            resolved_from=resolved_from,
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
        on_failure=on_failure,
        resolved_from=resolved_from,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_method(
    method: str, problem: OptimizationProblem, backend: str
) -> tuple[str, str | None]:
    """Return ``(resolved_method, resolved_from)``."""
    method = method.lower()
    if method == "auto":
        # Hard constraints (add_constraint) require the KKT-capable LM path.
        has_hard = (
            getattr(problem, "constraints", None) is not None
            and len(problem.constraints) > 0
        )
        if has_hard:
            return "dls", "auto"
        if backend == "torch":
            return "adam", "auto"
        ops = list(problem.operands)
        n_vars = len(list(problem.variables))
        all_equality = all(op.target is not None for op in ops)
        if all_equality and len(ops) >= n_vars:
            return "dls", "auto"
        return "l-bfgs-b", "auto"
    if method not in _ALL_METHODS:
        raise ConfigurationError(
            f"Unknown method {method!r}. "
            f"Valid options: {sorted(_ALL_METHODS)}. "
            "Note: GlassExpert, OrthogonalDescent, ParticleSwarm are not "
            "accessible via minimize() — use them directly."
        )
    return method, None


def _validate(method: str, backend: str, constraints: Any, controller: Any) -> None:
    """Tiered validation: hard ConfigurationError or UserWarning.

    Hard errors (impossible requests):
    - torch-only method under numpy backend.
    - A per-step-projection constraint strategy with a managed (SciPy) method.

    Warn-and-ignore (inapplicable but harmless):
    - ``controller=`` passed to a managed method.
    - Global SciPy methods (bounds are ignored by the solver).
    """
    # --- Hard errors ---
    if method in _TORCH_METHODS and backend != "torch":
        raise ConfigurationError(
            f"Method {method!r} requires the torch backend. "
            "Call be.set_backend('torch') before calling minimize()."
        )
    if method in _MANAGED_METHODS and constraints is not None:
        # A managed SciPy method cannot honor an arbitrary per-step projection
        # strategy (one that mutates the step via ``apply_to_step`` but cannot
        # be expressed to SciPy via ``to_scipy``).  Hard equality/inequality
        # constraints should instead be declared on the problem
        # (``add_constraint``) and solved with a KKT-capable stepped method.
        candidates = constraints if isinstance(constraints, list) else [constraints]
        for c in candidates:
            has_per_step = hasattr(c, "apply_to_step")
            scipy_native = hasattr(c, "to_scipy")
            if has_per_step and not scipy_native:
                raise ConfigurationError(
                    "A per-step constraint strategy requires a stepped optimizer "
                    "(e.g. 'dls', 'lm', 'gauss_newton'). "
                    f"Method {method!r} is a managed (SciPy) optimizer. Express the "
                    "constraint via SciPy (bounds / NonlinearConstraint) or declare "
                    "it on the problem with add_constraint() and use a stepped "
                    "method."
                )

    # --- Warn-and-ignore ---
    if method in _MANAGED_METHODS and controller is not None:
        warnings.warn(
            f"controller= is ignored for managed method {method!r}. "
            "Step controllers only apply to stepped optimizers (dls, lm, adam, etc.).",
            UserWarning,
            stacklevel=4,
        )
    if method in _SCIPY_GLOBAL_METHODS:
        # Global solvers typically ignore bounds; don't hard-error, just inform
        warnings.warn(
            f"Method {method!r} is a global solver; variable bounds may be "
            "ignored by the underlying SciPy algorithm.",
            UserWarning,
            stacklevel=4,
        )


def _validate_hard_constraints(method: str, problem: OptimizationProblem) -> None:
    """Hard constraints (``add_constraint``) require a KKT-capable method.

    The native stepped solvers (``dls`` / ``lm``) declare ``KKT_CONSTRAINTS``
    and route the constraints through :class:`KKTController`.  Passing hard
    constraints to any other method raises with guidance (D13).
    """
    constraints = getattr(problem, "constraints", None)
    if constraints is None or len(constraints) == 0:
        return
    if method not in ("dls", "lm"):
        raise ConfigurationError(
            f"Problem declares {len(constraints)} hard constraint(s) via "
            "add_constraint(), which require a KKT-capable stepped method "
            f"('dls' or 'lm'). Method {method!r} does not support hard "
            "equality/inequality constraints. Either switch to method='dls', or "
            "express the requirement as a soft penalty operand (add_operand "
            "with min_val/max_val)."
        )


def _build_evaluator(
    problem: OptimizationProblem,
    backend: str,
    method: str,
    on_failure: str = "reject",
    fd_options: dict | None = None,
) -> Any:
    if backend == "torch" and method in (_TORCH_METHODS | _NATIVE_STEPPED_METHODS):
        from optiland.optimization.evaluators.autograd import AutogradEvaluator

        return AutogradEvaluator(problem, on_failure=on_failure)
    from optiland.optimization.evaluators.finite_difference import FiniteDiffEvaluator

    fd_options = fd_options or {}
    return FiniteDiffEvaluator(
        problem,
        rel_step=fd_options.get("rel_step", 1e-5),
        abs_step=fd_options.get("abs_step", 1e-8),
        scheme=fd_options.get("scheme", "forward"),
        on_failure=on_failure,
    )


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


def _build_controller(
    method: str,
    controller: Any,
    problem: Any = None,
    linear_solver: str = "normal",
) -> Any:
    if controller is not None:
        return controller
    if method in ("dls", "lm"):
        # Route to the KKT active-set controller when the problem declares hard
        # equality / inequality constraints; otherwise the plain Levenberg
        # controller (Nielsen gain-ratio damping).
        has_hard = (
            problem is not None
            and getattr(problem, "constraints", None) is not None
            and len(problem.constraints) > 0
        )
        if has_hard:
            from optiland.optimization.control.kkt import KKTController

            return KKTController(linear_solver=linear_solver)
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
    resolved_from: str | None = None,
) -> OptimizationResult:
    from optiland.optimization.drivers import SteppedDriver
    from optiland.optimization.native.least_squares import (
        GaussNewton,
        LevenbergMarquardt,
    )

    optimizer = LevenbergMarquardt() if method in ("dls", "lm") else GaussNewton()

    linear_solver = method_options.pop("linear_solver", "normal")
    ctrl = _build_controller(method, controller, problem, linear_solver=linear_solver)

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
        resolved_from=resolved_from,
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
    resolved_from: str | None = None,
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
        resolved_from=resolved_from,
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
    on_failure: str = "penalty",
    resolved_from: str | None = None,
) -> OptimizationResult:
    from optiland.optimization.drivers import ManagedDriver

    adapter = _build_managed_adapter(method, problem, method_options, on_failure)

    driver = ManagedDriver()
    return driver.run(
        adapter,
        evaluator,
        criteria=stop,
        observers=observers,
        initial_value=initial_value,
        method=method,
        method_options=method_options,
        resolved_from=resolved_from,
    )


def _build_managed_adapter(
    method: str,
    problem: OptimizationProblem,
    method_options: dict,
    on_failure: str = "penalty",
) -> Any:
    if method in _SCIPY_LOCAL_METHODS:
        from optiland.optimization.managed.scipy_local import ScipyLocalAdapter

        return ScipyLocalAdapter(problem, method=method, on_failure=on_failure)

    if method in _SCIPY_LS_METHODS:
        method_choice = method_options.pop("method_choice", "lm")
        from optiland.optimization.managed.scipy_least_squares import (
            ScipyLeastSquaresAdapter,
        )

        return ScipyLeastSquaresAdapter(
            problem, method_choice=method_choice, on_failure=on_failure
        )

    if method in _SCIPY_GLOBAL_METHODS:
        from optiland.optimization.managed.scipy_global import ScipyGlobalAdapter

        return ScipyGlobalAdapter(problem, algorithm=method)

    raise ConfigurationError(  # pragma: no cover
        f"No adapter available for method {method!r}"
    )
