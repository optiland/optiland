"""Levenberg-Marquardt Damping Control

:class:`LevenbergController` owns the damped least-squares linear algebra and
the damping state of the native DLS core (opt v2 PR4.2). With the merit
``F = sum(r ** 2)``, ``g = J^T r`` and ``A = J^T J`` formed once per outer
iteration, each trial step solves::

    (A + lambda D^2) dx = -g

by Cholesky (``scipy.linalg.cho_factor``). The damping ``lambda`` is set by a
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
import scipy.linalg

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
_RADIUS_RTOL = 0.1
_MAX_LAMBDA_ITER = 30


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
        self, A: np.ndarray, g: np.ndarray, free: np.ndarray | None = None
    ) -> np.ndarray:
        """Damped step ``dx`` with ``||D dx|| <= Delta`` at the current radius.

        Args:
            A: ``J^T J``.
            g: ``J^T r``.
            free: Optional boolean mask of the variables to solve for; the
                others (held at an active bound) get a zero step.

        Returns:
            The step ``dx``; :attr:`lam` holds its damping.
        """
        if free is not None and not np.all(free):
            dx = np.zeros_like(g)
            if np.any(free):
                d_full = self.d
                self.d = d_full[free]
                try:
                    dx[free] = self.solve(A[np.ix_(free, free)], g[free])
                finally:
                    self.d = d_full
            return dx
        return self._trust_region_step(A, g)

    def _damped(self, A: np.ndarray, g: np.ndarray, lam: float) -> tuple:
        """Cholesky solve at ``lam``: ``(dx, factor)`` or None if not SPD."""
        M = A + lam * np.diag(self.d**2)
        try:
            factor = scipy.linalg.cho_factor(M, check_finite=False)
        except (np.linalg.LinAlgError, ValueError):
            return None
        dx = scipy.linalg.cho_solve(factor, -g, check_finite=False)
        return (dx, factor) if np.all(np.isfinite(dx)) else None

    def _trust_region_step(self, A: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Moré-Hebden iteration for ``||D dx(lambda)|| = Delta``."""
        delta = self.radius
        d = self.d
        # Gauss-Newton step when it is defined and inside the region.
        gn = self._damped(A, g, 0.0)
        if gn is not None and np.linalg.norm(d * gn[0]) <= (1 + _RADIUS_RTOL) * delta:
            self.lam, self.on_boundary = 0.0, False
            return gn[0]

        # Bracket: phi(lam) = ||D dx|| - Delta decreases in lam; the steepest
        # descent bound ||D^-1 g|| / Delta overshoots it.
        upper = float(np.linalg.norm(g / d)) / delta
        lower = 0.0
        lam = min(max(self.lam, _EPS * upper), upper) if upper > 0 else 0.0
        best = None
        for _ in range(_MAX_LAMBDA_ITER):
            if not lower < lam < upper or lam <= 0:
                lam = max(np.sqrt(max(lower, _EPS * upper) * upper), _EPS * upper)
            solved = self._damped(A, g, lam)
            if solved is None:  # not positive definite yet: more damping
                lower = lam
                lam = 2.0 * lam
                continue
            dx, factor = solved
            q = d * dx
            norm_q = float(np.linalg.norm(q))
            best = (dx, lam)
            if abs(norm_q - delta) <= _RADIUS_RTOL * delta:
                break
            if norm_q > delta:
                lower = lam
            else:
                upper = lam
            # Newton step on 1/||D dx|| - 1/Delta (Moré 1978).
            w = scipy.linalg.solve_triangular(
                factor[0], d * q, trans="T", lower=factor[1], check_finite=False
            )
            norm_w2 = float(w @ w)
            if norm_w2 <= 0:
                lam = 0.5 * (lower + upper)
                continue
            lam = lam + (norm_q / delta - 1.0) * norm_q**2 / norm_w2
        if best is None:  # pathological A: least squares at the upper bound
            lam = max(upper, 1.0)
            M = A + lam * np.diag(d**2)
            best = (np.linalg.lstsq(M, -g, rcond=None)[0], lam)
        dx, self.lam = best
        self.on_boundary = True
        return dx

    @staticmethod
    def model_reduction(dx: np.ndarray, g: np.ndarray, A: np.ndarray) -> float:
        """Reduction of ``F`` predicted by the linear model for a step ``dx``.

        ``||r||^2 - ||r + J dx||^2``, in the units of ``F`` (no 1/2).
        """
        return float(-(2.0 * g @ dx + dx @ A @ dx))

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
