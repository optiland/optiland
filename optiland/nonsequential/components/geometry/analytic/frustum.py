"""Cylindrical frustum geometry for Non-Sequential Raytracing.

CylindricalFrustumGeometry -- a truncated cone (frustum) defined by front
and back radii and axial positions. Used as the edge surface of a lens.

All operations in LOCAL coordinates.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np

import optiland.backend as be
from optiland.nonsequential import _tol
from optiland.nonsequential._utils import as_float, as_param
from optiland.nonsequential.components.geometry.base import AABB, AnalyticGeometry

# Degenerate-quadratic multiple (k = 8):
# |a| <= k * ulp(max(|b|, |c|)) is treated as "a is not usefully nonzero",
# not a bare 1e-14.
_DEGENERACY_K = 8


class CylindricalFrustumGeometry(AnalyticGeometry):
    """Lateral surface of a truncated cone (frustum) along the local z-axis.

    The frustum connects a circle of radius ``r_front`` at ``z = z_front``
    to a circle of radius ``r_back`` at ``z = z_back``.  Only the lateral
    (barrel) surface is modelled -- the two end-caps are handled by separate
    geometry objects (e.g. ConicGeometry for lens faces).

    Attributes:
        r_front: Radius at the front rim [mm].
        r_back: Radius at the back rim [mm].
        z_front: Axial position of the front rim in local frame [mm].
        z_back: Axial position of the back rim in local frame [mm].
    """

    def __init__(
        self,
        r_front: float,
        r_back: float,
        z_front: float,
        z_back: float,
    ) -> None:
        """Initialize CylindricalFrustumGeometry.

        Args:
            r_front: Radius at the front rim [mm].
            r_back: Radius at the back rim [mm].
            z_front: Axial z-position of the front rim [mm].
            z_back: Axial z-position of the back rim [mm].
        """
        self.r_front = as_param(r_front)
        self.r_back = as_param(r_back)
        self.z_front = as_param(z_front)
        self.z_back = as_param(z_back)

    def ray_intersect(
        self, origins: np.ndarray, directions: np.ndarray, eps: float | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Intersect rays with the frustum lateral surface.

        The frustum surface satisfies::

            x^2 + y^2 = r(z)^2

        where ``r(z) = r_front + slope*(z - z_front)`` and
        ``slope = (r_back - r_front)/(z_back - z_front)``.

        Substituting the ray parametric equations yields a quadratic in t.
        Both roots are tested; the smallest positive root within the axial
        range [z_front, z_back] is returned.

        Args:
            origins: Ray origins in local frame, shape (N, 3) [mm].
            directions: Ray directions in local frame, shape (N, 3).

        Returns:
            (t, normals, hit_mask, n_geom) all in local frame. n_geom points
            radially outward from the frustum axis -- this geometry is only
            ever used for edge/barrel surfaces with the same material on
            both sides, so it never needs to satisfy the ``material_back``
            sidedness contract in :meth:`ComponentGeometry.ray_intersect`.
        """
        ox, oy, oz = origins[:, 0], origins[:, 1], origins[:, 2]
        dx, dy, dz = directions[:, 0], directions[:, 1], directions[:, 2]

        h = self.z_back - self.z_front  # axial height [mm]
        # Construction-time check, evaluated once (not per ray): is this
        # frustum's axial extent itself indistinguishable from zero? Detached
        # float64 (z_front/z_back are geometry parameters, not ray data), k
        # ulps of the extent's own magnitude with a 1 mm floor.
        h_val = as_float(h)
        h_scale = max(abs(as_float(self.z_front)), abs(as_float(self.z_back)), 1.0)
        if abs(h_val) < _DEGENERACY_K * np.spacing(h_scale):
            # Degenerate frustum (zero height) -- no lateral surface to hit
            N = origins.shape[0]
            return (
                be.ones(N) * be.inf,
                be.zeros((N, 3)),
                be.zeros(N, dtype=bool),
                be.zeros((N, 3)),
            )

        slope = (self.r_back - self.r_front) / h

        rz = self.r_front + slope * (oz - self.z_front)
        rv = slope * dz

        a = dx * dx + dy * dy - rv * rv
        b = 2.0 * (ox * dx + oy * dy - rz * rv)
        c = ox * ox + oy * oy - rz * rz

        disc = b * b - 4.0 * a * c
        disc_safe = be.maximum(disc, 0.0)
        sqrt_disc = be.sqrt(disc_safe)

        if eps is None:
            eps = _tol.accept_t_min(be.abs(origins).max())
        inf_val = be.ones_like(a) * be.inf

        # Linear fallback when |a| is very small (ray nearly parallel to
        # axis): a degenerate quadratic, k ulps of the other coefficients'
        # scale rather than a bare 1e-14. b_small uses the same scale so the
        # two thresholds agree.
        coeff_scale = be.maximum(be.abs(b), be.abs(c))
        degeneracy_floor = _DEGENERACY_K * _tol.ulp(coeff_scale)
        a_small = be.abs(a) < degeneracy_floor
        b_small = be.abs(b) < degeneracy_floor

        # Guarded reciprocals: mask the INPUT into a well-conditioned value
        # before dividing, rather than add an epsilon to the denominator
        # (see optiland.nonsequential._tol.tiny_for). The additive-epsilon form
        # this replaced, "1.0 / (2.0 * a + 1e-30)", is the exact NaN source
        # measured on this geometry in float32: its
        # backward pass forms eps**2 = 1e-60, which underflows to zero in
        # float32, turning a `where`'s discarded zero-cotangent branch into
        # 0 * inf = NaN. Masking the input to 1.0 needs no epsilon at all --
        # the discarded branch evaluates a perfectly ordinary reciprocal.
        b_safe = be.where(b_small, be.ones_like(b), b)
        t_lin = be.where(b_small, inf_val, -c / b_safe)

        a_safe = be.where(a_small, be.ones_like(a), a)
        inv2a = be.where(a_small, 0.0, 1.0 / (2.0 * a_safe))
        t1 = (-b - sqrt_disc) * inv2a
        t2 = (-b + sqrt_disc) * inv2a

        def _valid(t: np.ndarray) -> np.ndarray:
            """True where t > eps, within axial limits, and disc >= 0."""
            z_hit = oz + t * dz
            in_z = (z_hit >= self.z_front - eps) & (z_hit <= self.z_back + eps)
            return (disc >= 0.0) & (t > eps) & in_z

        valid1 = _valid(t1)
        valid2 = _valid(t2)
        valid_lin = _valid(t_lin)

        # Pick the smallest valid t
        t_best = be.ones_like(a) * be.inf
        t_best = be.where(valid2, t2, t_best)
        t_best = be.where(valid1, t1, t_best)
        t_best = be.where(a_small & valid_lin, t_lin, t_best)

        hit_mask = be.isfinite(t_best)

        # Compute normals at hit points (clamp t for miss rays to avoid NaN)
        t_nrm = be.where(hit_mask, t_best, be.zeros_like(t_best))
        hx = ox + t_nrm * dx
        hy = oy + t_nrm * dy
        hz = oz + t_nrm * dz
        rz_hit = self.r_front + slope * (hz - self.z_front)

        # Gradient of f(x,y,z) = x^2 + y^2 - r(z)^2:
        #   (2x, 2y, -2*r(z)*slope)
        nx = hx
        ny = hy
        nz = -rz_hit * slope * be.ones_like(hx)
        n_len = be.sqrt(nx * nx + ny * ny + nz * nz + _tol.tiny_for(nx))
        n_geom = be.stack([nx / n_len, ny / n_len, nz / n_len], axis=1)

        # Flip to face incoming ray
        dot = (directions * n_geom).sum(axis=1, keepdims=True)
        normals = be.where(dot > 0, -n_geom, n_geom)

        t_out = be.where(hit_mask, t_best, inf_val)
        return t_out, normals, hit_mask, n_geom

    def bounding_box(self, transform: tuple[np.ndarray, np.ndarray]) -> AABB:
        """Return AABB of the frustum in global coordinates.

        Args:
            transform: (translation, rotation_matrix).

        Returns:
            AABB in global frame.
        """
        t_vec = np.array(transform[0], dtype=float)
        R = np.array(transform[1], dtype=float)
        r_max = max(as_float(self.r_front), as_float(self.r_back))
        z_f = as_float(self.z_front)
        z_b = as_float(self.z_back)

        corners_local = np.array(
            [
                [-r_max, -r_max, z_f],
                [-r_max, r_max, z_f],
                [r_max, -r_max, z_f],
                [r_max, r_max, z_f],
                [-r_max, -r_max, z_b],
                [-r_max, r_max, z_b],
                [r_max, -r_max, z_b],
                [r_max, r_max, z_b],
            ],
            dtype=float,
        )
        corners_global = corners_local @ R.T + t_vec
        return AABB(corners_global.min(axis=0), corners_global.max(axis=0))
