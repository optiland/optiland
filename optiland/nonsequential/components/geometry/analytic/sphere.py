"""Sphere geometry for Non-Sequential Raytracing.

Full sphere or spherical cap. Operations in LOCAL coordinates.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np

from optiland.nonsequential.components.geometry.base import AABB, AnalyticGeometry


class SphereGeometry(AnalyticGeometry):
    """Full sphere centred at the local origin.

    The sphere equation: x^2 + y^2 + z^2 = radius^2.

    Attributes:
        radius: Sphere radius [mm].
        aperture_radius: Optional aperture limit [mm]. Only the part of the
            sphere within this transverse radius is considered.
    """

    def __init__(self, radius: float, aperture_radius: float | None = None) -> None:
        """Initialize SphereGeometry.

        Args:
            radius: Sphere radius [mm].
            aperture_radius: Optional transverse aperture limit [mm]. Points
                outside this radius are not considered valid hits.
        """
        self.radius = float(radius)
        self.aperture_radius = (
            float(aperture_radius) if aperture_radius is not None else None
        )

    def ray_intersect(
        self, origins: np.ndarray, directions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Intersect rays with the sphere.

        Uses the analytic quadratic solution. Returns the nearest positive hit.

        Args:
            origins: Ray origins in local frame, shape (N, 3) [mm].
            directions: Ray directions in local frame, shape (N, 3), unit.

        Returns:
            (t, normals, hit_mask).
        """
        xp = _get_xp(origins)
        ox, oy, oz = origins[:, 0], origins[:, 1], origins[:, 2]
        dx, dy, dz = directions[:, 0], directions[:, 1], directions[:, 2]

        # Quadratic: |o + t*d|^2 = R^2
        # a*t^2 + b*t + c = 0
        # a = |d|^2 = 1 (unit directions)
        b = 2.0 * (ox * dx + oy * dy + oz * dz)
        c = ox**2 + oy**2 + oz**2 - self.radius**2

        discriminant = b**2 - 4.0 * c
        disc_ok = discriminant >= 0.0
        sqrt_disc = xp.where(
            disc_ok, (xp.maximum(discriminant, 0.0)) ** 0.5, xp.zeros_like(discriminant)
        )

        t1 = xp.where(disc_ok, (-b - sqrt_disc) / 2.0, xp.inf)
        t2 = xp.where(disc_ok, (-b + sqrt_disc) / 2.0, xp.inf)

        # Choose nearest positive t
        eps = 1e-9
        use_t1 = disc_ok & (t1 > eps)
        use_t2 = disc_ok & (~use_t1) & (t2 > eps)
        t = xp.where(use_t1, t1, xp.where(use_t2, t2, xp.inf))

        # Compute hit position and normal
        hx = ox + t * dx
        hy = oy + t * dy
        hz = oz + t * dz

        # Normal: outward from sphere centre, flipped to face incoming ray
        nx = xp.where(t < xp.inf, hx / self.radius, xp.zeros_like(hx))
        ny = xp.where(t < xp.inf, hy / self.radius, xp.zeros_like(hy))
        nz = xp.where(t < xp.inf, hz / self.radius, xp.zeros_like(hz))

        # Flip to face incoming ray
        dot = dx * nx + dy * ny + dz * nz
        flip = xp.where(dot > 0, -1.0, 1.0)
        normals = xp.stack([nx * flip, ny * flip, nz * flip], axis=1)

        hit_mask = t < xp.inf

        # Aperture check
        if self.aperture_radius is not None:
            r_transverse = (hx**2 + hy**2) ** 0.5
            in_aperture = r_transverse <= self.aperture_radius
            hit_mask = hit_mask & in_aperture
            t = xp.where(hit_mask, t, xp.inf)

        return t, normals, hit_mask

    def bounding_box(self, transform: tuple[np.ndarray, np.ndarray]) -> AABB:
        """Return AABB for the sphere in global coordinates.

        Args:
            transform: (translation, rotation_matrix).

        Returns:
            AABB in global frame.
        """
        t = np.array(transform[0], dtype=float)
        r = self.radius
        return AABB(t - r, t + r)


def _get_xp(arr: np.ndarray):
    try:
        import cupy  # type: ignore[import]

        if isinstance(arr, cupy.ndarray):
            return cupy
    except ImportError:
        pass
    return np
