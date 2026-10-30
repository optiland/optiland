"""Native Damped Least Squares

The native Levenberg-Marquardt (DLS) core of opt v2 (PR4.2). Private for now:
it is reached through ``LeastSquares.optimize(method_choice="dls")``.

Per outer iteration ``k``:

1. ``r = residuals(x)`` and ``J = jacobian(x, r0=r)``, or a Broyden update.
2. Column scaling ``D`` (monotone): Moré (``x_scale="jac"``) or fixed
   relative weights ``1 / x_scale``.
3. ``g = J^T r`` and one SVD of ``J D^-1`` per outer iteration. ``J^T J``
   is never formed: it squares the condition number.
4. Inner damping loop with no Jacobian work: the damped step
   ``(J^T J + lambda D^2) dx = -g`` from the SVD, with ``lambda`` set by a
   trust radius (MINPACK / SciPy ``trf`` rules, see
   :class:`~optiland.optimization.control.levenberg.LevenbergController`),
   trial ``r_t = residuals(x + dx)``. A trial is accepted when it lowers the
   merit; a non-finite trial is rejected and shrinks the radius.
5. Bounds and hard constraints (PR7.3): with either present,
   :class:`~optiland.optimization.control.kkt.KKTController` takes over the
   step. A bound is a linear inequality row, so the active set holds it and a
   trial point stays inside the box. Trials are then accepted on the exact
   penalty merit, and a run that ends infeasible reports ``success=False``
   after damped restoration, with a per-row report.
6. Gated Broyden (``jacobian_update``): ``"exact"`` recomputes ``J`` after
   every accepted step. ``"broyden"`` applies the rank-one update
   ``J += ((dr - J dx) dx^T) / (dx^T dx)`` after every accepted step.
   ``"auto"`` updates only after a very successful step (``rho >= 0.75``) and
   at most twice in a row. In both Broyden modes, a rejected trial while ``J``
   is not exact recomputes ``J`` at ``x`` before the radius shrinks.
7. Stopping: relative merit decrease below ``ftol`` over an accepted step
   (SciPy ``ftol`` semantics), scaled gradient below ``gtol``, step below
   ``xtol``, the residual-evaluation budget ``max_nfev`` (Jacobian columns
   included, decision V17), or a collapsed trust region (``success=False``).
   On top of these, a user ``stop`` criterion and ``cancel_token`` are
   checked after every accepted step (experimental).
8. Observers (experimental) get ``on_start``, ``on_step`` after every
   accepted step and ``on_end``. The state's gradient and Jacobian are those
   of the last Jacobian evaluation: at ``on_step`` they lag ``x`` by one
   accepted step, because the run may end before the next Jacobian is paid for.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

import optiland.backend as be
from optiland.optimization.constraint.constraint import row_violations
from optiland.optimization.control.kkt import KKTController
from optiland.optimization.control.levenberg import LevenbergController
from optiland.optimization.failure import (
    ConfigurationError,
    describe_rows,
    failing_rows,
)
from optiland.optimization.observers.callback import CallbackObserver
from optiland.optimization.observers.cancel import with_cancel
from optiland.optimization.observers.history import find_history
from optiland.optimization.state import OptimizationResult, OptimizationState

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike

    from optiland.optimization.evaluators.base import Evaluator
    from optiland.optimization.observers.base import Observer
    from optiland.optimization.observers.cancel import CancelToken
    from optiland.optimization.stopping.base import StoppingCriterion

#: Accepted values of ``jacobian_update``.
JACOBIAN_UPDATES: tuple[str, ...] = ("auto", "exact", "broyden")

# Broyden gate of "auto": gain ratio and consecutive-update limit.
BROYDEN_MIN_RHO = 0.75
BROYDEN_MAX_CONSECUTIVE = 2

# Gain ratio above which a small merit decrease counts as converged (SciPy).
FTOL_MIN_RHO = 0.25

# Gain ratio below which a trial that raised the violation gets a correction.
CORRECTION_MAX_RHO = 0.75

# Restoration: steps per attempt, attempts per run, and backtracking fractions.
MAX_RESTORATION_STEPS = 8
MAX_RESTORATIONS = 3
RESTORATION_FRACTIONS = (1.0, 0.5, 0.25, 0.125)

STATUS_MESSAGES = {
    -5: "The hard constraints could not be satisfied.",
    -3: "The trust region collapsed: no step reduces the merit.",
    -4: "Stopped by a stopping criterion.",
    -2: "Stopped by the callback.",
    0: "The maximum number of residual evaluations is exceeded.",
    1: "`gtol` termination condition is satisfied.",
    2: "`ftol` termination condition is satisfied.",
    3: "`xtol` termination condition is satisfied.",
    4: "Both `ftol` and `xtol` termination conditions are satisfied.",
}


def _np(value: Any) -> np.ndarray:
    return np.asarray(be.to_numpy(value), dtype=float)


def dls(
    evaluator: Evaluator,
    x0: ArrayLike | None = None,
    *,
    bounds: tuple[ArrayLike, ArrayLike] | None = None,
    max_nfev: int | None = None,
    ftol: float = 1e-8,
    xtol: float = 1e-8,
    gtol: float = 1e-8,
    jacobian_update: str = "exact",
    x_scale: ArrayLike | float | str = 1.0,
    callback: Callable | None = None,
    stop: StoppingCriterion | None = None,
    observers: list[Observer] | None = None,
    cancel_token: CancelToken | None = None,
) -> OptimizationResult:
    """Minimize ``sum(r(x) ** 2)`` by damped least squares.

    Args:
        evaluator: Supplies ``residuals``, ``jacobian(x, r0)``, ``write_x``
            and ``n_vars``; its ``nfev`` counter (if any) drives ``max_nfev``.
            One with ``n_constraints > 0`` also supplies the hard-constraint
            rows (see
            :class:`~optiland.optimization.evaluators.SupportsConstraints`).
        x0: Starting point. Defaults to ``evaluator.read_x()``.
        bounds: ``(lower, upper)`` arrays; ``±inf`` for no bound. Defaults to
            ``evaluator.bounds()`` when available, else unbounded.
        max_nfev: Maximum residual evaluations, Jacobian columns included.
            Defaults to ``100 * n``.
        ftol: Stop when an accepted step lowers the merit by less than
            ``ftol * F``.
        xtol: Stop when ``||dx|| <= xtol * (xtol + ||x||)``.
        gtol: Stop when ``max_j |g_j| / (||J_:j|| ||r||) <= gtol``.
        jacobian_update: ``"exact"`` (default), ``"auto"`` (gated Broyden)
            or ``"broyden"``.
        x_scale: Characteristic variable scale, as in SciPy. A scalar or
            array (default 1.0, SciPy ``trf``'s default) fixes the scaling
            ``D = 1 / x_scale``; ``"jac"`` uses the Moré scaling from the
            Jacobian's column norms.
        callback: Called after every accepted step, as ``callback(x)`` or,
            when its only parameter is ``intermediate_result``, with an
            ``OptimizeResult``. Raising ``StopIteration`` stops the run.
            Equivalent to attaching a
            :class:`~optiland.optimization.observers.callback.CallbackObserver`.
        stop: Extra stopping criterion, checked after every accepted step on
            top of ``ftol``/``xtol``/``gtol``/``max_nfev`` (experimental).
        observers: Observers notified of the run (experimental).
        cancel_token: Stops the run cleanly at the next accepted step once
            cancelled (experimental).

    Returns:
        An :class:`~optiland.optimization.state.OptimizationResult` (a
        ``scipy.optimize.OptimizeResult``) with ``x``, ``fun`` (residuals),
        ``cost`` (``0.5 * sum(r ** 2)``, SciPy's convention), ``jac``,
        ``grad``, ``optimality``, ``nfev``, ``njev``, ``nit``, ``status``,
        ``message``, ``success``, ``stop_reason``, ``n_evals``, ``history``
        (None without a ``HistoryObserver``), ``multipliers``, ``active_set``,
        ``max_constraint_violation`` and ``constraint_report``. The evaluator's
        optic is left at ``x``. A run that ends with a violated hard
        constraint has ``success=False``.

    Raises:
        ConfigurationError: If ``jacobian_update`` is unknown, or the
            starting point has non-finite residuals.
    """
    if jacobian_update not in JACOBIAN_UPDATES:
        raise ConfigurationError(
            f"Unknown jacobian_update {jacobian_update!r}; expected one of "
            f"{list(JACOBIAN_UPDATES)}."
        )
    n = int(evaluator.n_vars)
    x = _np(evaluator.read_x() if x0 is None else x0).reshape(-1).copy()
    if bounds is None and hasattr(evaluator, "bounds"):
        bounds = evaluator.bounds()
    lower, upper = (
        (np.full(n, -np.inf), np.full(n, np.inf))
        if bounds is None
        else (_np(bounds[0]).reshape(-1), _np(bounds[1]).reshape(-1))
    )
    if max_nfev is None:
        max_nfev = 100 * max(n, 1)
    observer_list = list(observers or [])
    if callback is not None:
        observer_list.append(CallbackObserver(callback))
    criterion = with_cancel(stop, cancel_token)

    counter = _Counter(evaluator)
    scaling: dict[str, Any] = {}
    if isinstance(x_scale, str):
        if x_scale != "jac":
            raise ConfigurationError(
                f"Unknown x_scale {x_scale!r}; use 'jac' or numbers."
            )
    else:
        scale = np.broadcast_to(np.asarray(x_scale, dtype=float), (n,))
        if not np.all(np.isfinite(scale) & (scale > 0)):
            raise ConfigurationError("x_scale must be positive and finite.")
        scaling = {"scaling": "fixed", "weights": 1.0 / scale}
    if counter.n_constraints or np.any(np.isfinite(lower) | np.isfinite(upper)):
        controller = KKTController(*counter.constraint_rows(), lower, upper, **scaling)
    else:
        controller = LevenbergController(**scaling)

    x = np.clip(x, lower, upper)
    r = counter.residuals(x)
    s = counter.constraints(x)
    rows = failing_rows(np.concatenate([r, s]))
    if rows.size:
        problem = getattr(evaluator, "problem", None)
        names = describe_rows(problem, rows) if problem is not None else str(list(rows))
        raise ConfigurationError(
            "Cannot optimize from an untraceable design: the starting point "
            f"has non-finite residuals for operand(s) {names}."
        )
    F = float(r @ r)
    J = counter.jacobian(x, r)
    A = counter.constraint_jacobian(x)
    fresh = True
    consecutive_broyden = 0
    nit = 0
    status: int | None = None
    restorations = 0

    def build(J: np.ndarray) -> np.ndarray:
        controller.rescale(J)
        controller.linearize(x, J, r, s, A)
        return J.T @ r

    g = build(J)
    controller.init_radius(x)
    state = OptimizationState(
        x=x.copy(),
        value=F,
        residuals=r.copy(),
        gradient=g.copy(),
        jacobian=J,
        n_evals=counter.nfev,
        grad_norm=float(np.linalg.norm(g)),
    )
    stop_reason: str | None = None
    for observer in observer_list:
        observer.on_start(state)
    if criterion is not None:
        criterion.reset(state)
    while status is None:
        # Only an exact Jacobian may certify a stationary point.
        if controller.feasible(s) and (
            F == 0.0
            or (fresh and _scaled_gradient(J, r, controller.stationarity(g)) <= gtol)
        ):
            status = 1
            break

        # Inner damping loop: no Jacobian work except a stale-J refresh.
        while True:
            if counter.nfev >= max_nfev:
                status = 0
                break
            dx = controller.solve(J, r)
            # The rows already hold every bound: this clip only trims round-off.
            x_t = np.clip(x + dx, lower, upper)
            dx = x_t - x
            if not np.any(dx):
                status = 3  # the bounds or round-off block every move
                break
            predicted = controller.model_reduction(dx, J, r)
            r_t = counter.residuals(x_t)
            s_t = counter.constraints(x_t)
            finite = np.all(np.isfinite(r_t)) and np.all(np.isfinite(s_t))
            F_t = float(r_t @ r_t) if finite else np.inf
            phi = controller.merit(F, s)
            phi_t = controller.merit(F_t, s_t)
            if not np.isfinite(F_t):
                rho = -np.inf
            elif predicted > 0:
                rho = (phi - phi_t) / predicted
            else:
                rho = 1.0 if phi_t < phi else -np.inf
            if np.isfinite(F_t) and rho < CORRECTION_MAX_RHO:
                dx_c = controller.second_order_step(dx, s_t)
                if dx_c is not None:
                    x_c = np.clip(x + dx_c, lower, upper)
                    r_c = counter.residuals(x_c)
                    s_c = counter.constraints(x_c)
                    finite = np.all(np.isfinite(r_c)) and np.all(np.isfinite(s_c))
                    F_c = float(r_c @ r_c) if finite else np.inf
                    phi = controller.merit(F, s)  # the correction may raise sigma
                    phi_t = controller.merit(F_t, s_t)
                    phi_c = controller.merit(F_c, s_c)
                    if phi_c < phi_t:
                        x_t, dx, r_t, s_t, F_t = x_c, x_c - x, r_c, s_c, F_c
                        phi_t = phi_c
                    predicted = controller.model_reduction(dx, J, r)
                    rho = (phi - phi_t) / predicted if predicted > 0 else -np.inf
            accepted = phi_t < phi
            if not accepted and not fresh:
                # A stale Broyden J may be the reason: refresh before shrinking.
                J = counter.jacobian(x, r)
                A = counter.constraint_jacobian(x)
                fresh, consecutive_broyden = True, 0
                g = build(J)
                continue
            if accepted and predicted <= 0:
                rho = 1.0
            controller.update(rho, dx)
            if accepted:
                break
            if controller.collapsed(x):
                status = -3
                break
        if status is not None:
            # A blocked or collapsed step does not end an infeasible run yet.
            stuck = status in (3, -3) and not controller.feasible(s)
            if stuck and restorations < MAX_RESTORATIONS and counter.nfev < max_nfev:
                restorations += 1
                restored = _restore(
                    counter, controller, x, r, s, lower, upper, max_nfev
                )
                if restored is not None:
                    x, r, s = restored
                    F = float(r @ r)
                    J = counter.jacobian(x, r)
                    A = counter.constraint_jacobian(x)
                    fresh, consecutive_broyden = True, 0
                    g = build(J)
                    controller.init_radius(x)
                    status = None
                    continue
            break

        # Accepted step.
        nit += 1
        dF = phi - phi_t
        # SciPy ftol semantics: a small decrease counts only on a good model fit
        small_f = dF < ftol * phi and rho > FTOL_MIN_RHO
        small_x = np.linalg.norm(dx) <= xtol * (xtol + np.linalg.norm(x))
        dr = r_t - r
        ds = s_t - s
        step_norm = float(np.linalg.norm(dx))
        x, r, s, F = x_t, r_t, s_t, F_t

        state.x, state.value, state.residuals = x.copy(), F, r.copy()
        state.iteration, state.n_evals, state.step_norm = nit, counter.nfev, step_norm
        try:
            for observer in observer_list:
                observer.on_step(state)
        except StopIteration:
            status = -2
            break
        if criterion is not None:
            stopped, stop_reason = criterion.should_stop(state)
            if stopped:
                status = -4
                break

        if small_f or small_x:
            # Converged but for the rows: restore them instead of stepping on.
            infeasible = not controller.feasible(s)
            if infeasible and restorations < MAX_RESTORATIONS:
                restorations += 1
                restored = _restore(
                    counter, controller, x, r, s, lower, upper, max_nfev
                )
                if restored is not None:
                    x, r, s = restored
                    F = float(r @ r)
                    infeasible = False
            if not infeasible:
                status = 4 if (small_f and small_x) else (2 if small_f else 3)
                break
        if counter.nfev >= max_nfev:
            status = 0
            break

        use_broyden = jacobian_update == "broyden" or (
            jacobian_update == "auto"
            and rho >= BROYDEN_MIN_RHO
            and consecutive_broyden < BROYDEN_MAX_CONSECUTIVE
        )
        if use_broyden:
            J = J + np.outer(dr - J @ dx, dx) / float(dx @ dx)
            A = A + np.outer(ds - A @ dx, dx) / float(dx @ dx)
            fresh = False
            consecutive_broyden += 1
        else:
            J = counter.jacobian(x, r)
            A = counter.constraint_jacobian(x)
            fresh, consecutive_broyden = True, 0
        g = build(J)
        state.gradient, state.jacobian = g.copy(), J
        state.grad_norm = float(np.linalg.norm(g))

    evaluator.write_x(x)
    g = J.T @ r
    if status > 0 and not controller.feasible(s):
        status = -5
    diagnostics = controller.diagnostics(s)
    message = STATUS_MESSAGES[status]
    if status == -5:
        worst = max(diagnostics["constraint_report"], key=lambda row: row["violation"])
        message = (
            f"{message[:-1]}: largest violation {worst['violation']:.3e} "
            f"at {worst['label']}."
        )
    if status == -4 and stop_reason:
        message = f"{message[:-1]}: {stop_reason}."
    state.x, state.value, state.residuals = x.copy(), F, r.copy()
    state.gradient, state.jacobian = g, J
    state.grad_norm = float(np.linalg.norm(g))
    state.n_evals, state.stop_reason = counter.nfev, message
    result = OptimizationResult(
        x=x,
        fun=r,
        cost=0.5 * F,
        jac=J,
        grad=g,
        optimality=float(np.max(np.abs(g))) if g.size else 0.0,
        active_mask=np.where(x <= lower, -1, np.where(x >= upper, 1, 0)),
        nfev=counter.nfev,
        njev=counter.njev,
        nit=nit,
        status=status,
        message=message,
        success=status > 0,
        stop_reason=message,
        n_evals=counter.nfev,
        history=find_history(observer_list),
        **diagnostics,
    )
    for observer in observer_list:
        observer.on_end(state, result)
    return result


def _scaled_gradient(J: np.ndarray, r: np.ndarray, g: np.ndarray) -> float:
    """MINPACK ``gtol`` measure: the largest cosine between ``r`` and a column."""
    rnorm = float(np.linalg.norm(r))
    if rnorm == 0.0:
        return 0.0
    norms = np.linalg.norm(J, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.where(norms > 0, np.abs(g) / (norms * rnorm), 0.0)
    return float(np.max(cos)) if cos.size else 0.0


def _restore(
    counter: _Counter,
    controller: KKTController,
    x: np.ndarray,
    r: np.ndarray,
    s: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    max_nfev: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Damped feasibility restoration: lower the violation, ignoring the merit.

    Each step is a backtracked correction of the violated rows.

    Returns:
        The feasible ``(x, r, s)``, or None if the rows could not be met.
    """
    equality = controller.equality

    def violation(values: np.ndarray) -> float:
        return float(np.sum(row_violations(equality, values)))

    for _ in range(MAX_RESTORATION_STEPS):
        if counter.nfev >= max_nfev:
            return None
        J = counter.jacobian(x, r)
        controller.linearize(x, J, r, s, counter.constraint_jacobian(x))
        dx = controller.restoration_step()
        for fraction in RESTORATION_FRACTIONS:
            x_t = np.clip(x + fraction * dx, lower, upper)
            r_t = counter.residuals(x_t)
            s_t = counter.constraints(x_t)
            finite = np.all(np.isfinite(r_t)) and np.all(np.isfinite(s_t))
            if finite and violation(s_t) < violation(s):
                x, r, s = x_t, r_t, s_t
                break
        else:
            return None
        if controller.feasible(s):
            return x, r, s
    return None


class _Counter:
    """Residual/Jacobian calls with evaluation counts.

    Uses the evaluator's own ``nfev``/``njev`` when it keeps them (finite
    differences count every column); otherwise counts one evaluation per
    residual call and ``n`` per Jacobian, matching forward differences.
    Constraint rows share those evaluations and are not counted again.
    """

    def __init__(self, evaluator: Evaluator) -> None:
        self.evaluator = evaluator
        self._own = hasattr(evaluator, "nfev") and hasattr(evaluator, "njev")
        self._start = (evaluator.nfev, evaluator.njev) if self._own else (0, 0)
        self._nfev = 0
        self._njev = 0
        self.n_constraints = int(getattr(evaluator, "n_constraints", 0))

    def constraint_rows(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Equality mask, tolerances and labels of the constraint rows."""
        if not self.n_constraints:
            return np.zeros(0, dtype=bool), np.zeros(0), []
        return self.evaluator.constraint_rows()

    def constraints(self, x: np.ndarray) -> np.ndarray:
        if not self.n_constraints:
            return np.zeros(0)
        return _np(self.evaluator.constraints(x)).reshape(-1)

    def constraint_jacobian(self, x: np.ndarray) -> np.ndarray:
        n = int(self.evaluator.n_vars)
        if not self.n_constraints:
            return np.zeros((0, n))
        A = _np(self.evaluator.constraint_jacobian(x)).reshape(self.n_constraints, n)
        return np.where(np.isfinite(A), A, 0.0)

    @property
    def nfev(self) -> int:
        if self._own:
            return self.evaluator.nfev - self._start[0]
        return self._nfev

    @property
    def njev(self) -> int:
        if self._own:
            return self.evaluator.njev - self._start[1]
        return self._njev

    def residuals(self, x: np.ndarray) -> np.ndarray:
        self._nfev += 1
        return _np(self.evaluator.residuals(x)).reshape(-1)

    def jacobian(self, x: np.ndarray, r: np.ndarray) -> np.ndarray:
        self._njev += 1
        self._nfev += int(self.evaluator.n_vars)
        n = int(self.evaluator.n_vars)
        J = _np(self.evaluator.jacobian(x, r0=r)).reshape(r.size, n)
        # Entries an evaluator could not difference carry no information.
        return np.where(np.isfinite(J), J, 0.0)
