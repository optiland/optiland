"""Cylindrical frustum geometry for Non-Sequential Raytracing.

CylindricalFrustumGeometry -- a truncated cone (frustum) defined by front
and back radii and axial positions. Used as the edge surface of a lens.

All operations in LOCAL coordinates.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np

from optiland.nonsequential.components.geometry.base import AABB, AnalyticGeometry


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
        self.r_front = float(r_front)
        self.r_back = float(r_back)
        self.z_front = float(z_front)
        self.z_back = float(z_back)

    def ray_intersect(
        self, origins: np.ndarray, directions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Intersect rays with the frustum lateral surface.

        The frustum surface satisfies:
            x^2 + y^2 = r(z)^2
        where r(z) = r_front + slope*(z - z_front),
        slope = (r_back - r_front)/(z_back - z_front).

        Substituting the ray parametric equations yields a quadratic in t.
        Both roots are tested; the smallest positive root within the axial
        range [z_front, z_back] is returned.

        Args:
            origins: Ray origins in local frame, shape (N, 3) [mm].
            directions: Ray directions in local frame, shape (N, 3).

        Returns:
            (t, normals, hit_mask) all in local frame.
        """
        xp = _get_xp(origins)

        ox, oy, oz = origins[:, 0], origins[:, 1], origins[:, 2]
        dx, dy, dz = directions[:, 0], directions[:, 1], directions[:, 2]

        h = self.z_back - self.z_front  # axial height [mm]
        if abs(h) < 1e-15:
            # Degenerate frustum (zero height) -- no lateral surface to hit
            N = origins.shape[0]
            return (
                xp.full(N, xp.inf),
                xp.zeros((N, 3)),
                xp.zeros(N, dtype=bool),
            )

        slope = (self.r_back - self.r_front) / h

        # r at the ray origin along z: rz = r_front + slope*(oz - z_front)
        rz = self.r_front + slope * (oz - self.z_front)
        # rate of change of r with t: rv = slope * dz
        rv = slope * dz

        # Quadratic coefficients: (dx^2 + dy^2 - rv^2)*t^2 + 2*(ox*dx + oy*dy - rz*rv)*t
        #                          + (ox^2 + oy^2 - rz^2) = 0
        a = dx * dx + dy * dy - rv * rv
        b = 2.0 * (ox * dx + oy * dy - rz * rv)
        c = ox * ox + oy * oy - rz * rz

        disc = b * b - 4.0 * a * c
        disc_safe = xp.maximum(disc, 0.0)
        sqrt_disc = xp.sqrt(disc_safe)

        eps = 1e-9
        inf_val = xp.full(a.shape, xp.inf)

        # Linear fallback when |a| is very small (ray nearly parallel to axis)
        a_small = xp.abs(a) < 1e-14
        t_lin = xp.where(xp.abs(b) > 1e-14, -c / (b + 1e-30), inf_val)

        inv2a = xp.where(a_small, 0.0, 1.0 / (2.0 * a + 1e-30))
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
        t_best = xp.full(a.shape, xp.inf)
        t_best = xp.where(valid2, t2, t_best)
        t_best = xp.where(valid1, t1, t_best)
        # For near-linear case, use linear solution
        t_best = xp.where(a_small & valid_lin, t_lin, t_best)

        hit_mask = xp.isfinite(t_best)

        # Compute normals at hit points (clamp t for miss rays to avoid NaN)
        t_nrm = xp.where(hit_mask, t_best, xp.zeros_like(t_best))
        hx = ox + t_nrm * dx
        hy = oy + t_nrm * dy
        hz = oz + t_nrm * dz
        rz_hit = self.r_front + slope * (hz - self.z_front)

        # Gradient of f(x,y,z) = x^2 + y^2 - r(z)^2:
        #   (2x, 2y, -2*r(z)*slope)
        nx = hx
        ny = hy
        nz = -rz_hit * slope * xp.ones_like(hx)
        n_len = xp.sqrt(nx * nx + ny * ny + nz * nz + 1e-30)
        normals = xp.stack([nx / n_len, ny / n_len, nz / n_len], axis=1)

        # Flip to face incoming ray
        dot = (directions * normals).sum(axis=1, keepdims=True)
        normals = xp.where(dot > 0, -normals, normals)

        t_out = xp.where(hit_mask, t_best, inf_val)
        return t_out, normals, hit_mask

    def bounding_box(self, transform: tuple[np.ndarray, np.ndarray]) -> AABB:
        """Return AABB of the frustum in global coordinates.

        Args:
            transform: (translation, rotation_matrix).

        Returns:
            AABB in global frame.
        """
        t_vec = np.array(transform[0], dtype=float)
        R = np.array(transform[1], dtype=float)
        r_max = max(self.r_front, self.r_back)

        corners_local = np.array(
            [
                [-r_max, -r_max, self.z_front],
                [-r_max, r_max, self.z_front],
                [r_max, -r_max, self.z_front],
                [r_max, r_max, self.z_front],
                [-r_max, -r_max, self.z_back],
                [-r_max, r_max, self.z_back],
                [r_max, -r_max, self.z_back],
                [r_max, r_max, self.z_back],
            ],
            dtype=float,
        )
        corners_global = corners_local @ R.T + t_vec
        return AABB(corners_global.min(axis=0), corners_global.max(axis=0))


def _get_xp(arr: np.ndarray):
    try:
        import cupy  # type: ignore[import]

        if isinstance(arr, cupy.ndarray):
            return cupy
    except ImportError:
        pass
    return np
