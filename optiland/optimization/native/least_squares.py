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
5. Bounds: variables at a bound that the gradient pushes outward are held,
   and the trial point is projected to the box *before* it is evaluated, so
   the optic and the residuals always belong to the projected point.
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

Kramer Harrison, 2026
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import OptimizeResult

import optiland.backend as be
from optiland.optimization.control.levenberg import LevenbergController
from optiland.optimization.failure import (
    ConfigurationError,
    describe_rows,
    failing_rows,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike

    from optiland.optimization.evaluators.base import Evaluator

#: Accepted values of ``jacobian_update``.
JACOBIAN_UPDATES: tuple[str, ...] = ("auto", "exact", "broyden")

# Broyden gate of "auto": gain ratio and consecutive-update limit.
BROYDEN_MIN_RHO = 0.75
BROYDEN_MAX_CONSECUTIVE = 2

# Gain ratio above which a small merit decrease counts as converged (SciPy).
FTOL_MIN_RHO = 0.25

STATUS_MESSAGES = {
    -3: "The trust region collapsed: no step reduces the merit.",
    -2: "Stopped by the callback.",
    0: "The maximum number of residual evaluations is exceeded.",
    1: "`gtol` termination condition is satisfied.",
    2: "`ftol` termination condition is satisfied.",
    3: "`xtol` termination condition is satisfied.",
    4: "Both `ftol` and `xtol` termination conditions are satisfied.",
}


def _np(value: Any) -> np.ndarray:
    return np.asarray(be.to_numpy(value), dtype=float)


def _callback_wants_result(callback: Callable) -> bool:
    """SciPy convention: a sole parameter named ``intermediate_result``."""
    try:
        params = list(inspect.signature(callback).parameters)
    except (TypeError, ValueError):
        return False
    return params == ["intermediate_result"]


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
) -> OptimizeResult:
    """Minimize ``sum(r(x) ** 2)`` by damped least squares.

    Args:
        evaluator: Supplies ``residuals``, ``jacobian(x, r0)``, ``write_x``
            and ``n_vars``; its ``nfev`` counter (if any) drives ``max_nfev``.
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

    Returns:
        A ``scipy.optimize.OptimizeResult`` with ``x``, ``fun`` (residuals),
        ``cost`` (``0.5 * sum(r ** 2)``, SciPy's convention), ``jac``,
        ``grad``, ``optimality``, ``nfev``, ``njev``, ``nit``, ``status``,
        ``message`` and ``success``. The evaluator's optic is left at ``x``.

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
    wants_result = callback is not None and _callback_wants_result(callback)

    counter = _Counter(evaluator)
    if isinstance(x_scale, str):
        if x_scale != "jac":
            raise ConfigurationError(
                f"Unknown x_scale {x_scale!r}; use 'jac' or numbers."
            )
        controller = LevenbergController()
    else:
        scale = np.broadcast_to(np.asarray(x_scale, dtype=float), (n,))
        if not np.all(np.isfinite(scale) & (scale > 0)):
            raise ConfigurationError("x_scale must be positive and finite.")
        controller = LevenbergController(scaling="fixed", weights=1.0 / scale)

    x = np.clip(x, lower, upper)
    r = counter.residuals(x)
    rows = failing_rows(r)
    if rows.size:
        problem = getattr(evaluator, "problem", None)
        names = describe_rows(problem, rows) if problem is not None else str(list(rows))
        raise ConfigurationError(
            "Cannot optimize from an untraceable design: the starting point "
            f"has non-finite residuals for operand(s) {names}."
        )
    F = float(r @ r)
    J = counter.jacobian(x, r)
    fresh = True
    consecutive_broyden = 0
    nit = 0
    status: int | None = None

    def build(J: np.ndarray) -> np.ndarray:
        controller.rescale(J)
        return J.T @ r

    g = build(J)
    controller.init_radius(x)
    while status is None:
        # Only an exact Jacobian may certify a stationary point.
        if F == 0.0 or (fresh and _scaled_gradient(J, r, g) <= gtol):
            status = 1
            break

        # Inner damping loop: no Jacobian work except a stale-J refresh.
        while True:
            if counter.nfev >= max_nfev:
                status = 0
                break
            # Variables at a bound that the gradient pushes outward stay put.
            held = ((x <= lower) & (g > 0)) | ((x >= upper) & (g < 0))
            dx = controller.solve(J, r, free=~held)
            x_t = np.clip(x + dx, lower, upper)
            dx = x_t - x
            if not np.any(dx):
                status = 3  # the bounds or round-off block every move
                break
            predicted = controller.model_reduction(dx, J, r)
            r_t = counter.residuals(x_t)
            F_t = float(r_t @ r_t) if np.all(np.isfinite(r_t)) else np.inf
            accepted = F_t < F
            if not accepted and not fresh:
                # A stale Broyden J may be the reason: refresh before shrinking.
                J = counter.jacobian(x, r)
                fresh, consecutive_broyden = True, 0
                g = build(J)
                continue
            if not np.isfinite(F_t):
                rho = -np.inf
            elif predicted > 0:
                rho = (F - F_t) / predicted
            else:
                rho = 1.0 if accepted else -np.inf
            controller.update(rho, dx)
            if accepted:
                break
            if controller.collapsed(x):
                status = -3
                break
        if status is not None:
            break

        # Accepted step.
        nit += 1
        dF = F - F_t
        # SciPy ftol semantics: a small decrease counts only on a good model fit
        small_f = dF < ftol * F and rho > FTOL_MIN_RHO
        small_x = np.linalg.norm(dx) <= xtol * (xtol + np.linalg.norm(x))
        dr = r_t - r
        x, r, F = x_t, r_t, F_t

        if callback is not None:
            try:
                if wants_result:
                    callback(
                        intermediate_result=OptimizeResult(
                            x=x.copy(), fun=r.copy(), cost=0.5 * F, nit=nit
                        )
                    )
                else:
                    callback(x.copy())
            except StopIteration:
                status = -2
                break

        if small_f or small_x:
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
            fresh = False
            consecutive_broyden += 1
        else:
            J = counter.jacobian(x, r)
            fresh, consecutive_broyden = True, 0
        g = build(J)

    evaluator.write_x(x)
    g = J.T @ r
    return OptimizeResult(
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
        message=STATUS_MESSAGES[status],
        success=status > 0,
    )


def _scaled_gradient(J: np.ndarray, r: np.ndarray, g: np.ndarray) -> float:
    """MINPACK ``gtol`` measure: the largest cosine between ``r`` and a column."""
    rnorm = float(np.linalg.norm(r))
    if rnorm == 0.0:
        return 0.0
    norms = np.linalg.norm(J, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.where(norms > 0, np.abs(g) / (norms * rnorm), 0.0)
    return float(np.max(cos)) if cos.size else 0.0


class _Counter:
    """Residual/Jacobian calls with evaluation counts.

    Uses the evaluator's own ``nfev``/``njev`` when it keeps them (finite
    differences count every column); otherwise counts one evaluation per
    residual call and ``n`` per Jacobian, matching forward differences.
    """

    def __init__(self, evaluator: Evaluator) -> None:
        self.evaluator = evaluator
        self._own = hasattr(evaluator, "nfev") and hasattr(evaluator, "njev")
        self._start = (evaluator.nfev, evaluator.njev) if self._own else (0, 0)
        self._nfev = 0
        self._njev = 0

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
        J = _np(self.evaluator.jacobian(x, r0=r)).reshape(r.size, -1)
        # Entries an evaluator could not difference carry no information.
        return np.where(np.isfinite(J), J, 0.0)
