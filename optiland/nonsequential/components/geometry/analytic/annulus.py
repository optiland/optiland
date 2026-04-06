"""Annular plane geometry for Non-Sequential Raytracing.

AnnularPlaneGeometry -- a flat ring (annulus) at a fixed axial offset, with
inner and outer radii.  Used as the rim surface of a lens whose front and
back apertures differ.

All operations in LOCAL coordinates.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np

from optiland.nonsequential.components.geometry.base import AABB, AnalyticGeometry


class AnnularPlaneGeometry(AnalyticGeometry):
    """Flat annular ring at ``z = z_offset`` in the local frame.

    The annulus extends from ``inner_radius`` to ``outer_radius`` in the
    radial direction.  Rays that hit the z = z_offset plane inside the
    annular band register a hit; rays outside or on the inner hole do not.

    Attributes:
        inner_radius: Inner radius of the annulus [mm].
        outer_radius: Outer radius of the annulus [mm].
        z_offset: Axial position of the plane in local frame [mm].
    """

    def __init__(
        self,
        inner_radius: float,
        outer_radius: float,
        z_offset: float = 0.0,
    ) -> None:
        """Initialize AnnularPlaneGeometry.

        Args:
            inner_radius: Inner (hole) radius [mm].
            outer_radius: Outer (rim) radius [mm].
            z_offset: Axial z-position of the plane [mm].
        """
        self.inner_radius = float(inner_radius)
        self.outer_radius = float(outer_radius)
        self.z_offset = float(z_offset)

    def ray_intersect(
        self, origins: np.ndarray, directions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Intersect rays with the annular plane.

        Uses the standard plane intersection: t = (z_offset - oz) / dz,
        then checks that the hit radius is within [inner_radius, outer_radius].

        Args:
            origins: Ray origins in local frame, shape (N, 3) [mm].
            directions: Ray directions in local frame, shape (N, 3).

        Returns:
            (t, normals, hit_mask) all in local frame.
        """
        xp = _get_xp(origins)
        N = origins.shape[0]

        oz = origins[:, 2]
        dz = directions[:, 2]

        eps = 1e-9
        # Avoid division by zero for rays parallel to the plane
        t = xp.where(
            xp.abs(dz) > eps,
            (self.z_offset - oz) / (dz + 1e-30),
            xp.full(N, xp.inf),
        )

        # Compute hit position; for parallel rays t=inf so hx/hy may be
        # inf or nan -- this is intentional: those values fail the r^2 check.
        with np.errstate(invalid="ignore"):
            hx = origins[:, 0] + t * directions[:, 0]
            hy = origins[:, 1] + t * directions[:, 1]
            r2 = hx * hx + hy * hy

        hit_mask = (
            (t > eps) & (r2 >= self.inner_radius**2) & (r2 <= self.outer_radius**2)
        )

        t_out = xp.where(hit_mask, t, xp.full(N, xp.inf))

        # Normal is (0, 0, +/-1) -- flip to face incoming ray
        nz_sign = xp.where(dz > 0, -1.0, 1.0)
        normals = xp.stack([xp.zeros(N), xp.zeros(N), nz_sign * xp.ones(N)], axis=1)

        return t_out, normals, hit_mask

    def bounding_box(self, transform: tuple[np.ndarray, np.ndarray]) -> AABB:
        """Return AABB of the annulus in global coordinates.

        Args:
            transform: (translation, rotation_matrix).

        Returns:
            AABB in global frame.
        """
        t_vec = np.array(transform[0], dtype=float)
        R = np.array(transform[1], dtype=float)
        r = self.outer_radius

        corners_local = np.array(
            [
                [-r, -r, self.z_offset],
                [-r, r, self.z_offset],
                [r, -r, self.z_offset],
                [r, r, self.z_offset],
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
