"""Sphere geometry for Non-Sequential Raytracing.

Full sphere or spherical cap. Operations in LOCAL coordinates.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np

import optiland.backend as be
from optiland.nonsequential._utils import as_float, as_param
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
        self.radius = as_param(radius)
        self.aperture_radius = (
            as_param(aperture_radius) if aperture_radius is not None else None
        )

    def ray_intersect(
        self, origins: np.ndarray, directions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Intersect rays with the sphere.

        Uses the analytic quadratic solution. Returns the nearest positive hit.

        Args:
            origins: Ray origins in local frame, shape (N, 3) [mm].
            directions: Ray directions in local frame, shape (N, 3), unit.

        Returns:
            (t, normals, hit_mask, n_geom). n_geom points *inward*, toward
            the sphere centre -- the ``material_back`` side by contract
            (see :meth:`ComponentGeometry.ray_intersect`): a standalone
            sphere used as a refractive surface (e.g. a ball lens) should
            be built with the exterior medium as ``material_front`` and the
            interior as ``material_back``, matching the convention every
            compound builder (``Lens``, ``Doublet``) already uses.
        """
        ox, oy, oz = origins[:, 0], origins[:, 1], origins[:, 2]
        dx, dy, dz = directions[:, 0], directions[:, 1], directions[:, 2]

        # Quadratic: |o + t*d|^2 = R^2
        b = 2.0 * (ox * dx + oy * dy + oz * dz)
        c = ox**2 + oy**2 + oz**2 - self.radius**2

        discriminant = b**2 - 4.0 * c
        disc_ok = discriminant >= 0.0
        # Clamp the radicand to a small positive epsilon (not 0). sqrt has an
        # infinite derivative at 0; combined with be.where this produces a
        # 0 * inf = NaN in the backward pass even though the forward is masked.
        sqrt_disc = be.where(
            disc_ok, be.maximum(discriminant, 1e-12) ** 0.5, be.zeros_like(discriminant)
        )

        inf_arr = be.ones_like(discriminant) * be.inf
        t1 = be.where(disc_ok, (-b - sqrt_disc) / 2.0, inf_arr)
        t2 = be.where(disc_ok, (-b + sqrt_disc) / 2.0, inf_arr)

        # Choose nearest positive t
        eps = 1e-9
        use_t1 = disc_ok & (t1 > eps)
        use_t2 = disc_ok & (~use_t1) & (t2 > eps)
        t = be.where(use_t1, t1, be.where(use_t2, t2, inf_arr))

        # Compute hit position and normal
        hx = ox + t * dx
        hy = oy + t * dy
        hz = oz + t * dz

        # n_geom: inward, toward the sphere centre, fixed regardless of ray
        # direction (see the material_back contract in the docstring above).
        nx = be.where(t < be.inf, hx / self.radius, be.zeros_like(hx))
        ny = be.where(t < be.inf, hy / self.radius, be.zeros_like(hy))
        nz = be.where(t < be.inf, hz / self.radius, be.zeros_like(hz))
        n_geom = be.stack([-nx, -ny, -nz], axis=1)

        # Flip to face incoming ray
        dot = dx * nx + dy * ny + dz * nz
        flip = be.where(dot > 0, -1.0, 1.0)
        normals = be.stack([nx * flip, ny * flip, nz * flip], axis=1)

        hit_mask = t < be.inf

        # Aperture check
        if self.aperture_radius is not None:
            r_transverse = (hx**2 + hy**2) ** 0.5
            in_aperture = r_transverse <= self.aperture_radius
            hit_mask = hit_mask & in_aperture
            t = be.where(hit_mask, t, inf_arr)

        return t, normals, hit_mask, n_geom

    def bounding_box(self, transform: tuple[np.ndarray, np.ndarray]) -> AABB:
        """Return AABB for the sphere in global coordinates.

        Args:
            transform: (translation, rotation_matrix).

        Returns:
            AABB in global frame.
        """
        t = np.array(transform[0], dtype=float)
        r = as_float(self.radius)
        return AABB(t - r, t + r)
