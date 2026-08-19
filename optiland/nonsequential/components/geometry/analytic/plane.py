"""Planar geometry primitives for Non-Sequential Raytracing.

PlaneGeometry (infinite flat plane) and FinitePlaneGeometry (rectangular
or circular bounded plane). All operations in LOCAL coordinates.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np

import optiland.backend as be
from optiland.nonsequential._utils import as_float, as_param
from optiland.nonsequential.components.geometry.base import AABB, AnalyticGeometry


class PlaneGeometry(AnalyticGeometry):
    """Infinite flat plane at z=0 in local coordinates, normal along +z.

    The plane equation in local frame is: z = 0.
    """

    def ray_intersect(
        self, origins: np.ndarray, directions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Intersect rays with the infinite plane z=0.

        Args:
            origins: Ray origins in local frame, shape (N, 3).
            directions: Ray directions in local frame, shape (N, 3).

        Returns:
            (t, normals, hit_mask, n_geom). n_geom is the fixed local +z
            (the ``material_back`` side by contract; see
            :meth:`ComponentGeometry.ray_intersect`).
        """
        dz = directions[:, 2]
        oz = origins[:, 2]

        # t = -oz / dz  (plane z=0)
        valid = be.abs(dz) > 1e-12
        safe_dz = be.where(valid, dz, be.ones_like(dz))
        t = be.where(valid, -oz / safe_dz, be.ones_like(oz) * be.inf)
        hit_mask = valid & (t > 1e-9)
        t = be.where(hit_mask, t, be.ones_like(t) * be.inf)

        # n_geom: fixed +z, independent of ray direction.
        # Normal: +z or -z depending on ray direction
        # Built with stack (not in-place assignment): an in-place write into
        # a leaf tensor raises under be.grad_mode.enable().
        n_geom = be.stack(
            [be.zeros_like(dz), be.zeros_like(dz), be.ones_like(dz)], axis=1
        )
        nz = be.where(dz < 0, be.ones_like(dz), -be.ones_like(dz))
        normals = be.stack([be.zeros_like(dz), be.zeros_like(dz), nz], axis=1)

        return t, normals, hit_mask, n_geom

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

    Supports rectangular (width x height) or circular (aperture_radius)
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
        self.width = as_param(width)
        self.height = as_param(height)
        self.aperture_radius = (
            as_param(aperture_radius) if aperture_radius is not None else None
        )

    def ray_intersect(
        self, origins: np.ndarray, directions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Intersect rays with the finite plane.

        Args:
            origins: Ray origins in local frame, shape (N, 3).
            directions: Ray directions in local frame, shape (N, 3).

        Returns:
            (t, normals, hit_mask, n_geom). n_geom is the fixed local +z
            (the ``material_back`` side by contract; see
            :meth:`ComponentGeometry.ray_intersect`).
        """
        dz = directions[:, 2]
        oz = origins[:, 2]

        inf_arr = be.ones_like(oz) * be.inf
        plane_valid = be.abs(dz) > 1e-12
        t = be.where(
            plane_valid, -oz / be.where(plane_valid, dz, be.ones_like(dz)), inf_arr
        )
        t = be.where(plane_valid & (t > 1e-9), t, inf_arr)

        # Hit position in local frame
        safe_t = be.where(be.isfinite(t), t, be.zeros_like(t))
        hx = origins[:, 0] + safe_t * directions[:, 0]
        hy = origins[:, 1] + safe_t * directions[:, 1]

        # Aperture check
        if self.aperture_radius is not None:
            in_aperture = (hx**2 + hy**2) <= self.aperture_radius**2
        else:
            half_w = self.width / 2.0
            half_h = self.height / 2.0
            in_aperture = (be.abs(hx) <= half_w) & (be.abs(hy) <= half_h)

        hit_mask = plane_valid & (t < be.inf) & in_aperture

        t = be.where(hit_mask, t, inf_arr)
        n_geom = be.stack(
            [be.zeros_like(dz), be.zeros_like(dz), be.ones_like(dz)], axis=1
        )
        # Built with stack (not in-place assignment): an in-place write into
        # a leaf tensor raises under be.grad_mode.enable().
        nz = be.where(dz < 0, be.ones_like(dz), -be.ones_like(dz))
        normals = be.stack([be.zeros_like(dz), be.zeros_like(dz), nz], axis=1)

        return t, normals, hit_mask, n_geom

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
            r = as_float(self.aperture_radius)
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
            hw = as_float(self.width) / 2.0
            hh = as_float(self.height) / 2.0
            corners_local = np.array(
                [
                    [-hw, -hh, 0],
                    [-hw, hh, 0],
                    [hw, -hh, 0],
                    [hw, hh, 0],
                ],
                dtype=float,
            )

        corners_global = corners_local @ R.T + t
        return AABB(corners_global.min(axis=0), corners_global.max(axis=0))
