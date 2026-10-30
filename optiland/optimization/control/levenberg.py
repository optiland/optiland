"""Levenberg-Marquardt Damping Control

:class:`LevenbergController` owns the damped least-squares linear algebra and
the damping state of the native DLS core (opt v2 PR4.2). With the merit
``F = sum(r ** 2)``, ``g = J^T r`` and ``A = J^T J``, each trial step solves::

    (A + lambda D^2) dx = -g

through one SVD of the scaled Jacobian ``J D^-1`` per outer iteration (as
SciPy ``trf`` does), not through ``A``: forming ``A`` squares the condition
number, and lens and NIST Jacobians with columns ten decades apart then lose
their small modes to round-off. The damping ``lambda`` is set by a
trust region, as in MINPACK's ``lmder`` and SciPy's ``trf``: the controller
keeps a radius ``Delta`` and picks the smallest ``lambda >= 0`` whose step
satisfies ``||D dx|| <= Delta`` (Moré-Hebden Newton iteration on the secular
equation, safeguarded by a bracket). The gain ratio
``rho = (F - F_trial) / predicted`` updates the radius:

* ``rho < 1/4`` (or a non-finite trial): ``Delta = ||D dx|| / 4``.
* ``rho > 3/4`` with the step on the boundary: ``Delta *= 2``.

Radius control lets ``lambda`` move by many decades in one step, which a
multiplicative ``lambda`` schedule (Nielsen) cannot: on lens problems whose
Jacobian columns span ten decades (asphere coefficients next to radii) that
schedule stalls. ``D`` is either the Moré column scaling
``D_j = max(||J_:j||, D_prev_j, eps)`` with
``eps = max(d_min, d_rel_min * max_j ||J_:j||)``, monotone over the run
(SciPy ``x_scale="jac"``), or fixed weights ``1 / x_scale``.

Linear algebra runs in numpy on the CPU for every backend: lens problems have
few variables, and GPU solver latency would dominate.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np

_EPS = float(np.finfo(float).eps)

#: Accepted values of ``scaling``.
SCALINGS: tuple[str, ...] = ("more", "fixed")

# Radius update thresholds and factors (MINPACK / SciPy trf).
_SHRINK_BELOW = 0.25
_GROW_ABOVE = 0.75
_SHRINK_FACTOR = 0.25
_GROW_FACTOR = 2.0
_ON_BOUNDARY = 0.95

# Secular-equation iteration: relative tolerance on ||D dx|| and iterations.
_RADIUS_RTOL = 0.01
_MAX_LAMBDA_ITER = 10


class LevenbergController:
    """Trust-region damping state and step solver of Levenberg-Marquardt.

    Args:
        scaling: ``"more"`` (default) for the per-column Moré scaling, or
            ``"fixed"`` for fixed weights ``D = weights``.
        weights: Weights of the ``"fixed"`` scaling, ``1 / x_scale`` in SciPy
            terms. Defaults to ones.
        d_min: Absolute floor of the Moré scaling. Defaults to 1e-12.
        d_rel_min: Floor of the Moré scaling relative to the largest column
            norm. A variable the residuals barely depend on has a column of
            finite-difference noise; without a floor its damping vanishes and
            the step along it explodes. Defaults to 1e-3.

    Attributes:
        d: Current scaling ``D`` (None until the first :meth:`rescale`).
        radius: Current trust radius ``Delta`` (None until
            :meth:`init_radius`).
        lam: Damping of the last solved step.
        on_boundary: Whether the last solved step reached the radius.
    """

    def __init__(
        self,
        scaling: str = "more",
        weights: np.ndarray | None = None,
        d_min: float = 1e-12,
        d_rel_min: float = 1e-3,
    ) -> None:
        if scaling not in SCALINGS:
            raise ValueError(f"scaling must be one of {SCALINGS}, got {scaling!r}.")
        self.scaling = scaling
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        self.d_min = float(d_min)
        self.d_rel_min = float(d_rel_min)
        self.d: np.ndarray | None = None
        self.radius: float | None = None
        self.lam = 0.0
        self.on_boundary = False
        self._svd_key: tuple | None = None
        self._svd_cache: tuple | None = None

    def rescale(self, J: np.ndarray) -> np.ndarray:
        """Update the scaling from the Jacobian (monotone for Moré)."""
        norms = np.linalg.norm(J, axis=0)
        if self.scaling == "fixed":
            self.d = np.ones_like(norms) if self.weights is None else self.weights
            return self.d
        largest = float(norms.max()) if norms.size else 0.0
        d = np.maximum(norms, max(self.d_min, self.d_rel_min * largest))
        self.d = d if self.d is None else np.maximum(d, self.d)
        return self.d

    def init_radius(self, x: np.ndarray) -> float:
        """Initial radius ``||D x||``, or 1 at the origin (SciPy convention)."""
        size = float(np.linalg.norm(self.d * x))
        self.radius = size if size > 0 else 1.0
        return self.radius

    def solve(
        self, J: np.ndarray, r: np.ndarray, free: np.ndarray | None = None
    ) -> np.ndarray:
        """Damped step ``dx`` with ``||D dx|| <= Delta`` at the current radius.

        Args:
            J: The Jacobian, shape ``(m, n)``.
            r: The residuals, shape ``(m,)``.
            free: Optional boolean mask of the variables to solve for; the
                others (held at an active bound) get a zero step.

        Returns:
            The step ``dx``; :attr:`lam` holds its damping.
        """
        if free is None or np.all(free):
            return self._trust_region_step(J, r, self.d)
        dx = np.zeros(J.shape[1])
        if np.any(free):
            dx[free] = self._trust_region_step(J[:, free], r, self.d[free])
        return dx

    def _trust_region_step(
        self, J: np.ndarray, r: np.ndarray, d: np.ndarray
    ) -> np.ndarray:
        """Solve ``||D dx(lambda)|| = Delta`` on the SVD of ``J D^-1``.

        With ``J D^-1 = U diag(s) V^T`` and ``u = U^T r``, the scaled step is
        ``D dx = -V z`` with ``z_i = s_i u_i / (s_i^2 + lambda)``, so each trial
        ``lambda`` costs O(n^2). Follows SciPy's ``solve_lsq_trust_region``:
        the Gauss-Newton step only for full column rank, otherwise the
        boundary solution, scaled onto the boundary exactly.
        """
        delta = self.radius
        s, V, u = self._svd(J, r, d)
        c = s * u
        # Gauss-Newton only for full column rank (SciPy trf): with fewer
        # residuals than variables the minimum-norm step is too aggressive.
        m, n = J.shape
        full_rank = m >= n and s.size > 0 and s[-1] > _EPS * m * s[0]
        if full_rank:
            z = u / s
            if np.linalg.norm(z) <= delta:
                self.lam, self.on_boundary = 0.0, False
                return -(V @ z) / d

        # ||D dx|| decreases in lambda; ||D^-1 g|| / Delta overshoots the root.
        lower, upper = 0.0, float(np.linalg.norm(c)) / delta
        if upper == 0.0:  # zero gradient: nothing to gain
            self.lam, self.on_boundary = 0.0, False
            return np.zeros(J.shape[1])
        lam = self.lam if 0.0 < self.lam < upper else 1e-3 * upper
        for _ in range(_MAX_LAMBDA_ITER):
            if not lower <= lam <= upper:
                lam = max(1e-3 * upper, np.sqrt(lower * upper))
            den = s**2 + lam
            z = c / den
            size = float(np.linalg.norm(z))
            if size < delta:
                upper = lam
            # Newton step on 1/||D dx|| - 1/Delta (Moré 1978).
            slope = float(np.sum(z**2 / den))
            if not slope > 0:
                break
            ratio = (size - delta) * size / slope
            lower = max(lower, lam + ratio)  # Newton from below never overshoots
            lam = lam + ratio * size / delta
            if abs(size - delta) < _RADIUS_RTOL * delta:
                break
        z = c / (s**2 + lam)
        size = float(np.linalg.norm(z))
        if size > 0:
            z = z * (delta / size)  # land exactly on the boundary
        self.lam, self.on_boundary = lam, True
        return -(V @ z) / d

    def _svd(self, J: np.ndarray, r: np.ndarray, d: np.ndarray) -> tuple:
        """Thin SVD of ``J D^-1`` cached per ``(J, D)``: ``s, V, U^T r``."""
        key = (id(J), d.tobytes())
        if self._svd_key != key:
            U, s, Vt = np.linalg.svd(J / d, full_matrices=False)
            self._svd_cache = (U, s, Vt.T, J)  # keep J alive for its id
            self._svd_key = key
        U, s, V, _ = self._svd_cache
        return s, V, U.T @ r

    @staticmethod
    def model_reduction(dx: np.ndarray, J: np.ndarray, r: np.ndarray) -> float:
        """Reduction of ``F`` predicted by the linear model for a step ``dx``.

        ``||r||^2 - ||r + J dx||^2``, in the units of ``F`` (no 1/2), computed
        without forming ``J^T J``.
        """
        Jdx = J @ dx
        return float(-(2.0 * (r @ Jdx) + Jdx @ Jdx))

    def update(self, rho: float, dx: np.ndarray) -> None:
        """Update the radius after a trial with gain ratio ``rho``.

        Args:
            rho: Gain ratio; pass ``-inf`` for a non-finite trial.
            dx: The trial step.
        """
        size = float(np.linalg.norm(self.d * dx))
        if rho < _SHRINK_BELOW:
            self.radius = _SHRINK_FACTOR * size
        elif rho > _GROW_ABOVE and size > _ON_BOUNDARY * self.radius:
            self.radius *= _GROW_FACTOR

    def collapsed(self, x: np.ndarray) -> bool:
        """Whether the radius shrank below round-off of ``||D x||``."""
        return self.radius <= _EPS * max(1.0, float(np.linalg.norm(self.d * x)))
