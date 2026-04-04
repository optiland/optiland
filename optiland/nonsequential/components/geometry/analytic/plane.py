"""Planar geometry primitives for Non-Sequential Raytracing.

PlaneGeometry (infinite flat plane) and FinitePlaneGeometry (rectangular
or circular bounded plane). All operations in LOCAL coordinates.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np

from optiland.nonsequential.components.geometry.base import AABB, AnalyticGeometry


class PlaneGeometry(AnalyticGeometry):
    """Infinite flat plane at z=0 in local coordinates, normal along +z.

    The plane equation in local frame is: z = 0.
    """

    def ray_intersect(
        self, origins: np.ndarray, directions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Intersect rays with the infinite plane z=0.

        Args:
            origins: Ray origins in local frame, shape (N, 3).
            directions: Ray directions in local frame, shape (N, 3).

        Returns:
            (t, normals, hit_mask).
        """
        xp = _get_xp(origins)
        dz = directions[:, 2]
        oz = origins[:, 2]

        # t = -oz / dz  (plane z=0)
        valid = xp.abs(dz) > 1e-12
        t = xp.where(valid, -oz / xp.where(valid, dz, xp.ones_like(dz)), xp.inf)
        hit_mask = valid & (t > 1e-9)
        t = xp.where(hit_mask, t, xp.inf)

        # Normal: +z or -z depending on ray direction
        normals = xp.zeros_like(origins)
        normals[:, 2] = xp.where(dz < 0, 1.0, -1.0)

        return t, normals, hit_mask

    def bounding_box(self, transform: tuple[np.ndarray, np.ndarray]) -> AABB:
        """Return infinite AABB (plane is unbounded).

        Args:
            transform: (translation, rotation_matrix).

        Returns:
            Infinite AABB.
        """
        inf = float("inf")
        return AABB(np.array([-inf, -inf, -inf]), np.array([inf, inf, inf]))


class FinitePlaneGeometry(AnalyticGeometry):
    """Finite planar surface at z=0 in local coordinates.

    Supports rectangular (width × height) or circular (aperture_radius)
    apertures. If aperture_radius is set, it takes precedence.

    Attributes:
        width: Rectangular half-width [mm] along local x. Used when
            aperture_radius is None.
        height: Rectangular half-height [mm] along local y. Used when
            aperture_radius is None.
        aperture_radius: Circular aperture radius [mm]. If set, the active
            region is a disk of this radius.
    """

    def __init__(
        self,
        width: float = 10.0,
        height: float = 10.0,
        aperture_radius: float | None = None,
    ) -> None:
        """Initialize FinitePlaneGeometry.

        Args:
            width: Full width [mm] along local x axis.
            height: Full height [mm] along local y axis.
            aperture_radius: If set, circular aperture of this radius [mm].
        """
        self.width = float(width)
        self.height = float(height)
        self.aperture_radius = (
            float(aperture_radius) if aperture_radius is not None else None
        )

    def ray_intersect(
        self, origins: np.ndarray, directions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Intersect rays with the finite plane.

        Args:
            origins: Ray origins in local frame, shape (N, 3).
            directions: Ray directions in local frame, shape (N, 3).

        Returns:
            (t, normals, hit_mask).
        """
        xp = _get_xp(origins)
        dz = directions[:, 2]
        oz = origins[:, 2]

        plane_valid = xp.abs(dz) > 1e-12
        t = xp.where(
            plane_valid, -oz / xp.where(plane_valid, dz, xp.ones_like(dz)), xp.inf
        )
        t = xp.where(plane_valid & (t > 1e-9), t, xp.inf)

        # Hit position in local frame — clamp t so inf*0 doesn't produce NaN
        safe_t = xp.where(xp.isfinite(t), t, xp.zeros_like(t))
        hx = origins[:, 0] + safe_t * directions[:, 0]
        hy = origins[:, 1] + safe_t * directions[:, 1]

        # Aperture check
        if self.aperture_radius is not None:
            in_aperture = (hx**2 + hy**2) <= self.aperture_radius**2
        else:
            half_w = self.width / 2.0
            half_h = self.height / 2.0
            in_aperture = (xp.abs(hx) <= half_w) & (xp.abs(hy) <= half_h)

        hit_mask = plane_valid & (t < xp.inf) & in_aperture

        t = xp.where(hit_mask, t, xp.inf)
        normals = xp.zeros_like(origins)
        normals[:, 2] = xp.where(dz < 0, 1.0, -1.0)

        return t, normals, hit_mask

    def bounding_box(self, transform: tuple[np.ndarray, np.ndarray]) -> AABB:
        """Return AABB for this finite plane in global coordinates.

        Args:
            transform: (translation, rotation_matrix).

        Returns:
            AABB in global frame.
        """
        translation, rot = transform
        t = np.array(translation, dtype=float)
        R = np.array(rot, dtype=float)

        if self.aperture_radius is not None:
            r = self.aperture_radius
            corners_local = np.array(
                [
                    [-r, -r, 0],
                    [-r, r, 0],
                    [r, -r, 0],
                    [r, r, 0],
                ],
                dtype=float,
            )
        else:
            hw = self.width / 2.0
            hh = self.height / 2.0
            corners_local = np.array(
                [
                    [-hw, -hh, 0],
                    [-hw, hh, 0],
                    [hw, -hh, 0],
                    [hw, hh, 0],
                ],
                dtype=float,
            )

        # local→global: g = R @ l + t  (column convention: R transforms col vectors)
        # For row vectors: g_row = l_row @ R^T + t
        corners_global = corners_local @ R.T + t
        return AABB(corners_global.min(axis=0), corners_global.max(axis=0))


def _get_xp(arr: np.ndarray):
    """Return numpy or cupy depending on the array type."""
    try:
        import cupy  # type: ignore[import]

        if isinstance(arr, cupy.ndarray):
            return cupy
    except ImportError:
        pass
    return np
