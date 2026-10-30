"""KKT Active-Set Control

:class:`KKTController` extends the trust-region damping of
:class:`~optiland.optimization.control.levenberg.LevenbergController` to hard
constraints and variable bounds (opt v2 PR7.3). All of them are inequality or
equality rows ``s(x)`` with Jacobian ``A``: a bound is the linear row
``lower - x <= 0`` or ``x - upper <= 0``. Each trial step solves the QP::

    min_dx  1/2 dx^T H dx + g^T dx,  H = J^T J + lambda D^2,  g = J^T r
    s.t.    A_i dx  = -s_i   (equalities)
            A_i dx <= -s_i   (inequalities)

If the Levenberg step already satisfies every linearized row, it is the
solution and ``J^T J`` is never formed. Otherwise:

* ``H`` is factored by Cholesky once per ``lambda``. A failed factorization
  raises ``lambda``.
* With ``H^-1`` applied once to the whole of ``A``, the Schur complement
  ``P = A H^-1 A^T`` holds every working set as a submatrix. The QP is solved
  in its dual (Goldfarb-Idnani): start from the equalities, add the most
  violated row, and release a row when its multiplier would change sign. The
  working set is exact, not a heuristic, and needs no feasible start.
* ``lambda`` is the smallest one (within a factor) whose step fits the radius
  ``||D dx|| <= Delta``. ``lambda`` moves through a bracket, not a secular
  Newton iteration, because the active set may change along the way.
* Damped feasibility restoration: if the minimum-norm correction of the
  violated rows does not fit the radius, those rows only ask for the fraction
  ``alpha`` of the correction that does. If the rows are inconsistent, the step
  minimizes the violation instead, and :meth:`restoration_step` lets the
  optimizer reduce the violation alone.

Steps are accepted on the exact penalty merit
``phi = F + sigma * ||violation||_1``, with ``sigma`` above the largest
multiplier. That merit can refuse a good step along a curved row (the Maratos
effect), so a poor trial gets one second-order correction: the same QP with
the rows shifted by the curvature the trial revealed. ``J^T J`` omits the
curvature of the rows, ``mu * d^2 s / dx^2``, so the rate along a curved active
row is linear. Multipliers are reported for the merit ``F = sum(r ** 2)``, so
that ``2 J^T r + A^T mu = 0`` at a solution, with ``mu >= 0`` on inequalities.
Row units are those of ``s``.

The Cholesky factor goes through ``be.linalg``. Everything else runs in numpy
on the CPU, as in the Levenberg controller.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular

import optiland.backend as be
from optiland.optimization.constraint.constraint import row_violations
from optiland.optimization.constraint.manager import row_report
from optiland.optimization.control.levenberg import LevenbergController

# Lower end of lambda, relative to the largest diagonal of the scaled J^T J.
_LAM_FLOOR = 1e-12
_LAM_RAISE = 10.0
_MAX_FACTOR_TRIES = 40
_MAX_EXPAND = 24
_MAX_BISECT = 6

# A step this close to the radius needs no smaller lambda.
_RADIUS_FILL = 0.75
# Fraction of the radius the damped correction of violated rows may use.
_NORMAL_FILL = 0.8

# Row margin for the dual QP, relative to the largest right-hand side.
_QP_RTOL = 1e-12
# Largest slack a consistent set of equality rows may leave.
_EQUALITY_SLACK = 1e-9
# A row whose Schur curvature is below this fraction of its own is dependent.
_DEPENDENT_RTOL = 1e-11

# Rows closer than this to their limit may carry a multiplier.
_NEAR_ACTIVE = 1e-8

_SIGMA_FLOOR = 1.0
_SIGMA_MARGIN = 1.1

# Relative Tikhonov term of the minimum-violation step.
_VIOLATION_DAMPING = 1e-8


def _cholesky(H: np.ndarray) -> np.ndarray:
    """Lower Cholesky factor of ``H`` through the active backend."""
    return np.asarray(be.to_numpy(be.linalg.cholesky(be.array(H))), dtype=float)


def _solve_small(S: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve a working-set block, by least squares if it is singular."""
    if S.size == 0:
        return np.zeros(0)
    try:
        return np.linalg.solve(S, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(S, b, rcond=None)[0]


def solve_dual_qp(
    P: np.ndarray, w0: np.ndarray, equality: np.ndarray, tol: float
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Dual active-set solve of ``w = w0 + P mu``, ``mu`` free on equalities.

    Finds ``mu`` with ``w = 0`` on equality rows and, on inequality rows,
    ``w >= 0``, ``mu >= 0`` and ``mu * w = 0``. This is the Goldfarb-Idnani
    method written on the Schur complement ``P``: raise the multiplier of the
    most violated row until it is satisfied, releasing any working row whose
    multiplier reaches zero first.

    Args:
        P: Positive semidefinite Schur complement, shape ``(q, q)``.
        w0: The slacks at ``mu = 0``, shape ``(q,)``.
        equality: Boolean mask of the equality rows.
        tol: Slack margin.

    Returns:
        ``(mu, active, solved)``. ``solved`` is False when the rows are
        inconsistent or the change limit is reached.
    """
    q = w0.size
    mu = np.zeros(q)
    active = equality.copy()
    rows = np.flatnonzero(equality)
    if rows.size:
        mu[rows] = np.linalg.lstsq(P[np.ix_(rows, rows)], -w0[rows], rcond=None)[0]
        if np.any(np.abs((w0 + P @ mu)[rows]) > max(tol, _EQUALITY_SLACK)):
            return mu, active, False

    for _ in range(10 * (q + 1)):
        w = w0 + P @ mu
        waiting = np.flatnonzero(~active)
        if waiting.size == 0:
            return mu, active, True
        p = waiting[np.argmin(w[waiting])]
        if w[p] >= -tol:
            return mu, active, True

        # Raise mu_p; the working rows stay at their slack through z.
        for _ in range(q + 1):
            working = np.flatnonzero(active)
            z = _solve_small(P[np.ix_(working, working)], -P[working, p])
            curvature = float(P[p, p] + P[p, working] @ z)
            dependent = curvature <= _DEPENDENT_RTOL * max(float(P[p, p]), tol)
            full = np.inf if dependent else -float(w[p]) / curvature
            blocking = (z < 0) & ~equality[working]
            release = np.inf
            if np.any(blocking):
                ratios = mu[working][blocking] / -z[blocking]
                k = int(np.argmin(ratios))
                release = float(ratios[k])
            t = min(full, release)
            if not np.isfinite(t):
                return mu, active, False
            mu[working] += t * z
            mu[p] += t
            if full <= release:
                active[p] = True
                break
            out = working[blocking][k]
            active[out] = False
            mu[out] = 0.0
            w = w0 + P @ mu
        else:
            return mu, active, False
    return mu, active, False


@dataclass
class _Factored:
    """Everything that depends on one ``lambda`` but not on the row limits."""

    u: np.ndarray  # H^-1 g
    W: np.ndarray  # H^-1 A^T
    P: np.ndarray  # A H^-1 A^T
    Au: np.ndarray  # A H^-1 g


@dataclass
class _Step:
    """A solved QP."""

    dx: np.ndarray
    mu: np.ndarray
    active: np.ndarray
    size: float


class KKTController(LevenbergController):
    """Trust-region active-set steps under hard constraints and bounds.

    Args:
        equality: Boolean mask of the equality rows among the hard rows.
        tols: Feasibility tolerance of each hard row.
        labels: Label of each hard row.
        lower: Lower variable bounds, ``-inf`` for none.
        upper: Upper variable bounds, ``inf`` for none.
        **levenberg: Scaling arguments of :class:`LevenbergController`.

    Attributes:
        sigma: Penalty weight of the merit, non-decreasing over a run.
        factorizations: Cholesky factorizations so far.
        lam_kkt: Damping of the last QP step.
    """

    def __init__(
        self,
        equality: np.ndarray,
        tols: np.ndarray,
        labels: list[str],
        lower: np.ndarray,
        upper: np.ndarray,
        **levenberg,
    ) -> None:
        super().__init__(**levenberg)
        self.equality = np.asarray(equality, dtype=bool)
        self.tols = np.asarray(tols, dtype=float)
        self.labels = list(labels)
        self.sigma = _SIGMA_FLOOR
        self.factorizations = 0
        self.lam_kkt = 0.0

        n = lower.size
        self._lower, self._upper = lower, upper
        self._lo = np.flatnonzero(np.isfinite(lower))
        self._hi = np.flatnonzero(np.isfinite(upper))
        self._n_hard = self.equality.size
        identity = np.eye(n)
        self._bound_A = np.vstack([-identity[self._lo], identity[self._hi]])
        self._bound_labels = [f"x[{j}] >= {lower[j]:g}" for j in self._lo] + [
            f"x[{j}] <= {upper[j]:g}" for j in self._hi
        ]
        self._eq_all = np.concatenate(
            [self.equality, np.zeros(self._bound_A.shape[0], dtype=bool)]
        )
        self._A = np.zeros((0, n))
        self._s = np.zeros(0)
        self._J = np.zeros((0, n))
        self._r = np.zeros(0)
        self._g = np.zeros(n)
        self._hess: np.ndarray | None = None
        self._systems: dict[float, _Factored] = {}
        self._floor: float | None = None
        self._warm = 0.0
        self._mu = np.zeros(self._eq_all.size)
        self._active = self._eq_all.copy()
        self._normal: np.ndarray | None = None
        self._last_limits = np.zeros(0)

    # ------------------------------------------------------------------
    # Linearization
    # ------------------------------------------------------------------

    def linearize(
        self,
        x: np.ndarray,
        J: np.ndarray,
        r: np.ndarray,
        s: np.ndarray,
        A: np.ndarray,
    ) -> None:
        """Set the model of the next trial steps (no evaluations)."""
        self._J, self._r, self._g = J, r, J.T @ r
        self._A = np.vstack([A, self._bound_A])
        self._s = np.concatenate(
            [
                s,
                self._lower[self._lo] - x[self._lo],
                x[self._hi] - self._upper[self._hi],
            ]
        )
        self._hess = None
        self._systems = {}
        self._floor = None
        self._normal = None
        self._last_limits = self._limits(1.0)

    def feasible(self, s: np.ndarray) -> bool:
        """Whether the hard rows ``s`` are within their tolerances."""
        return bool(np.all(row_violations(self.equality, s) <= self.tols))

    def merit(self, F: float, s: np.ndarray) -> float:
        """Exact penalty merit ``F + sigma * ||violation||_1``."""
        if not np.isfinite(F):
            return np.inf
        return F + self.sigma * self._violation(s)

    def _violation(self, s: np.ndarray) -> float:
        return float(np.sum(row_violations(self.equality, s)))

    def stationarity(self, g: np.ndarray) -> np.ndarray:
        """Lagrangian gradient ``g + A^T mu`` at the linearization point.

        Multipliers come from the QP at the smallest ``lambda``. A point with
        no row near its limit needs no factorization.
        """
        near = self._eq_all | (self._s >= -_NEAR_ACTIVE)
        if not np.any(near):
            return g
        step = self._qp(self._lam_floor(), self._limits(1.0))
        if step is None:
            return g
        self._mu, self._active = step.mu, step.active
        return g + self._A.T @ step.mu

    # ------------------------------------------------------------------
    # Trial step
    # ------------------------------------------------------------------

    def solve(
        self,
        J: np.ndarray,
        r: np.ndarray,
        free: np.ndarray | None = None,  # noqa: ARG002
    ) -> np.ndarray:
        """Active-set step with ``||D dx|| <= Delta`` at the current radius.

        Args:
            J: The Jacobian given to :meth:`linearize`.
            r: The residuals given to :meth:`linearize`.
            free: Accepted for parity; rows hold variables here.

        Returns:
            The step ``dx``.
        """
        step = super().solve(J, r)
        if self._holds(step):
            self._mu = np.zeros(self._s.size)
            self._active = self._eq_all.copy()
            self._last_limits = self._limits(1.0)
            return step

        alpha = self._feasibility_fraction()
        if alpha is None:
            return self._limited(self._minimum_violation())
        limits = self._limits(alpha)
        self._last_limits = limits
        solved = self._fit_radius(limits)
        if solved is None:
            return self._limited(alpha * self._normal)
        self._mu, self._active = solved.mu, solved.active
        self._raise_sigma(solved)
        return solved.dx

    def second_order_step(self, dx: np.ndarray, s_t: np.ndarray) -> np.ndarray | None:
        """Trial step with the rows shifted by the curvature ``dx`` revealed.

        Args:
            dx: The trial step.
            s_t: The hard rows at ``x + dx``.

        Returns:
            The corrected step, or None if ``dx`` did not raise the violation.
        """
        s = self._s[: self._n_hard]
        if self._violation(s_t) <= self._violation(s):
            return None
        curvature = s_t - s - self._A[: self._n_hard] @ dx
        limits = self._last_limits.copy()
        limits[: self._n_hard] -= curvature
        solved = self._fit_radius(limits)
        if solved is None:
            return None
        self._mu, self._active = solved.mu, solved.active
        self._raise_sigma(solved)
        return solved.dx

    def _holds(self, dx: np.ndarray) -> bool:
        """Whether ``dx`` satisfies every linearized row."""
        predicted = self._s + self._A @ dx
        margin = _QP_RTOL * max(1.0, float(np.max(np.abs(self._s), initial=0.0)))
        return bool(np.all(row_violations(self._eq_all, predicted) <= margin))

    def _limits(self, alpha: float) -> np.ndarray:
        """Right-hand sides ``A dx <= b``; violated rows relax to ``alpha``."""
        violated = self._eq_all | (self._s > 0)
        return np.where(violated, -alpha * self._s, -self._s)

    def _feasibility_fraction(self) -> float | None:
        """Fraction of the violated rows' correction that fits the radius.

        Returns:
            ``alpha`` in ``(0, 1]``, or None when the linearized rows are
            inconsistent.
        """
        self._normal = None
        d2 = self.d**2
        if not (np.any(self._eq_all) or np.any(self._s > 0)):
            self._normal = np.zeros(self.d.size)
            return 1.0
        scaled = self._A / d2
        b = -self._s
        tol = self._qp_tol(b)
        mu, _, solved = solve_dual_qp(self._A @ scaled.T, b, self._eq_all, tol)
        if not solved:
            return None
        self._normal = -(scaled.T @ mu)
        size = float(np.linalg.norm(self.d * self._normal))
        if size <= _NORMAL_FILL * self.radius:
            return 1.0
        return _NORMAL_FILL * self.radius / size

    def _fit_radius(self, limits: np.ndarray) -> _Step | None:
        """Smallest ``lambda`` (within a bracket) whose step fits the radius."""
        lam = self._lam_floor()
        step = self._qp(lam, limits)
        if step is None:
            return None
        if step.size <= self.radius:
            self.lam_kkt = lam
            return step

        low, high = lam, max(self._warm, _LAM_RAISE * lam)
        fitted = None
        for _ in range(_MAX_EXPAND):
            trial = self._qp(high, limits)
            if trial is not None and trial.size <= self.radius:
                fitted = trial
                break
            low, high = high, _LAM_RAISE * high
        if fitted is None:
            return None
        for _ in range(_MAX_BISECT):
            if fitted.size >= _RADIUS_FILL * self.radius:
                break
            middle = float(np.sqrt(low * high))
            trial = self._qp(middle, limits)
            if trial is not None and trial.size <= self.radius:
                high, fitted = middle, trial
            else:
                low = middle
        self._warm = self.lam_kkt = high
        return fitted

    def _qp(self, lam: float, limits: np.ndarray) -> _Step | None:
        """The QP at ``lam``, or None if it cannot be solved."""
        system = self._factored(lam)
        if system is None:
            return None
        mu, active, solved = solve_dual_qp(
            system.P, limits + system.Au, self._eq_all, self._qp_tol(limits)
        )
        if not solved:
            return None
        dx = -(system.u + system.W @ mu)
        return _Step(dx, mu, active, float(np.linalg.norm(self.d * dx)))

    @staticmethod
    def _qp_tol(limits: np.ndarray) -> float:
        return _QP_RTOL * max(1.0, float(np.max(np.abs(limits), initial=0.0)))

    def _hessian(self) -> np.ndarray:
        if self._hess is None:
            self._hess = self._J.T @ self._J
        return self._hess

    def _factored(self, lam: float) -> _Factored | None:
        """One Cholesky per ``lambda``, then ``H^-1`` of ``g`` and ``A^T``."""
        system = self._systems.get(lam)
        if system is not None:
            return system
        self.factorizations += 1
        H = self._hessian() + lam * np.diag(self.d**2)
        try:
            L = _cholesky(H)
        except be.linalg.LinAlgError:
            return None
        rhs = np.column_stack([self._g, self._A.T])
        y = solve_triangular(L, rhs, lower=True, check_finite=False)
        solved = solve_triangular(L.T, y, lower=False, check_finite=False)
        u, W = solved[:, 0], solved[:, 1:]
        system = _Factored(u, W, self._A @ W, self._A @ u)
        self._systems[lam] = system
        return system

    def _lam_floor(self) -> float:
        """Smallest ``lambda``: raised until ``H`` factors."""
        if self._floor is None:
            ratio = float(np.max(np.sum(self._J**2, axis=0) / self.d**2, initial=0.0))
            lam = _LAM_FLOOR * (ratio if ratio > 0 else 1.0)
            for _ in range(_MAX_FACTOR_TRIES):
                if self._factored(lam) is not None:
                    break
                lam *= _LAM_RAISE
            self._floor = lam
        return self._floor

    def _raise_sigma(self, step: _Step) -> None:
        """Keep ``sigma`` exact and the predicted penalty reduction positive."""
        hard = step.mu[: self._n_hard]
        largest = 2.0 * float(np.max(np.abs(hard), initial=0.0))
        self.sigma = max(self.sigma, _SIGMA_MARGIN * largest)
        gain = self._violation_gain(step.dx)
        reduction = super().model_reduction(step.dx, self._J, self._r)
        if reduction < 0 and gain > 0:
            self.sigma = max(self.sigma, -2.0 * reduction / gain)

    def _violation_gain(self, dx: np.ndarray) -> float:
        s = self._s[: self._n_hard]
        return self._violation(s) - self._violation(s + self._A[: s.size] @ dx)

    def model_reduction(self, dx: np.ndarray, J: np.ndarray, r: np.ndarray) -> float:
        """Reduction of the penalty merit predicted for ``dx``."""
        reduction = LevenbergController.model_reduction(dx, J, r)
        return reduction + self.sigma * self._violation_gain(dx)

    # ------------------------------------------------------------------
    # Restoration
    # ------------------------------------------------------------------

    def restoration_step(self) -> np.ndarray:
        """Step that reduces the violation alone, ignoring the merit.

        The minimum-norm correction of the violated rows, or the minimum
        violation when they are inconsistent, limited to the bounds.
        """
        radius, self.radius = self.radius, np.inf
        try:
            if self._feasibility_fraction() is None:
                return self._limited(self._minimum_violation())
            return self._limited(self._normal)
        finally:
            self.radius = radius

    def _minimum_violation(self) -> np.ndarray:
        """Damped least-squares step on the violated rows."""
        violated = np.flatnonzero(self._eq_all | (self._s > 0))
        if violated.size == 0:
            return np.zeros(self.d.size)
        A = self._A[violated]
        scaled = A / self.d**2
        S = A @ scaled.T
        S += (
            _VIOLATION_DAMPING
            * max(np.trace(S) / S.shape[0], 1e-12)
            * np.eye(S.shape[0])
        )
        return -(scaled.T @ _solve_small(S, self._s[violated]))

    def _limited(self, dx: np.ndarray) -> np.ndarray:
        """Shorten ``dx`` to the radius and to the bound rows."""
        size = float(np.linalg.norm(self.d * dx))
        if size > self.radius:
            dx = dx * (self.radius / size)
        rate = self._A[self._n_hard :] @ dx
        rows = rate > 0
        if np.any(rows):
            room = -self._s[self._n_hard :][rows] / rate[rows]
            dx = dx * min(1.0, float(np.min(room)))
        return dx

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def diagnostics(self, s: np.ndarray) -> dict:
        """Result fields: multipliers, active set, violation, report."""
        mu = 2.0 * self._mu
        bound = np.arange(self._s.size) >= self._n_hard
        multipliers = {label: float(mu[i]) for i, label in enumerate(self.labels)}
        names = self.labels + self._bound_labels
        for i in np.flatnonzero(bound & self._active):
            multipliers[names[i]] = float(mu[i])
        return {
            "multipliers": multipliers,
            "active_set": [names[i] for i in np.flatnonzero(self._active)],
            "max_constraint_violation": float(
                np.max(row_violations(self.equality, s), initial=0.0)
            ),
            "constraint_report": row_report(self.labels, self.equality, self.tols, s),
        }
