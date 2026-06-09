"""Conic geometry for Non-Sequential Raytracing.

ConicGeometry (paraboloid, ellipsoid, hyperboloid) and the ParaboloidGeometry
convenience subclass. All operations in LOCAL coordinates.

The conic surface is defined as:
    z = r^2 / (R * (1 + sqrt(1 - (1+K) * r^2 / R^2)))
where r^2 = x^2 + y^2, R is the radius of curvature, and K is the conic constant.

Special cases:
    K = 0  -> sphere
    K = -1 -> paraboloid
    K < -1 -> hyperboloid
    -1 < K < 0 -> prolate ellipsoid
    K > 0  -> oblate ellipsoid

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np

import optiland.backend as be
from optiland.nonsequential.components.geometry.base import AABB, AnalyticGeometry


class ConicGeometry(AnalyticGeometry):
    """Conic section surface (z = f(r)) centred on the local z-axis.

    The surface vertex is at the local origin. The aperture is circular with
    radius aperture_radius.

    Attributes:
        radius: Radius of curvature at the vertex [mm]. Positive = centre of
            curvature on +z side.
        conic: Conic constant K.
        aperture_radius: Semi-aperture radius [mm].
    """

    def __init__(self, radius: float, conic: float, aperture_radius: float) -> None:
        """Initialize ConicGeometry.

        Args:
            radius: Vertex radius of curvature [mm].
            conic: Conic constant K.
            aperture_radius: Aperture semi-diameter [mm].
        """
        self.radius = float(radius)
        self.conic = float(conic)
        self.aperture_radius = float(aperture_radius)

    def _sag(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute conic surface sag z(x, y).

        Args:
            x: x-coordinates, shape (N,).
            y: y-coordinates, shape (N,).

        Returns:
            Sag values z, shape (N,).
        """
        r2 = x**2 + y**2
        R = self.radius
        K = self.conic
        under_root = 1.0 - (1.0 + K) * r2 / (R * R)
        safe_root = be.maximum(under_root, 0.0)
        return r2 / (R * (1.0 + safe_root**0.5))

    def _normal_local(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute outward surface normals in local frame at (x, y, z(x,y)).

        The gradient of f(x,y,z) = z - sag(x,y) is (df/dx, df/dy, 1).

        Args:
            x: Hit x-coordinates, shape (N,).
            y: Hit y-coordinates, shape (N,).

        Returns:
            Outward normals (not yet normalized), shape (N, 3).
        """
        r2 = x**2 + y**2
        R = self.radius
        K = self.conic
        under_root = be.maximum(1.0 - (1.0 + K) * r2 / (R * R), 0.0)
        sqrt_term = under_root**0.5

        # dz/dx, dz/dy from implicit differentiation
        factor = 1.0 / (R * (1.0 + sqrt_term)) + (1.0 + K) * r2 / (
            2.0 * R**3 * under_root * (1.0 + sqrt_term) ** 2 + 1e-30
        )
        # Gradient of (z - sag): (-dsag/dx, -dsag/dy, 1)
        gx = -x * 2.0 * factor
        gy = -y * 2.0 * factor
        gz = be.ones_like(x)
        norms = be.stack([gx, gy, gz], axis=1)
        return norms

    def ray_intersect(
        self, origins: np.ndarray, directions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Intersect rays with the conic surface.

        Uses iterative Newton-Raphson to solve r(t) for the intersection.

        Args:
            origins: Ray origins in local frame, shape (N, 3) [mm].
            directions: Ray directions in local frame, shape (N, 3).

        Returns:
            (t, normals, hit_mask).
        """
        ox, oy, oz = origins[:, 0], origins[:, 1], origins[:, 2]
        dx, dy, dz = directions[:, 0], directions[:, 1], directions[:, 2]

        # Initial guess: plane intersection at z=0
        t = be.where(be.abs(dz) > 1e-12, -oz / (dz + 1e-30), be.zeros_like(oz))

        # Newton-Raphson: f(t) = oz + t*dz - sag(ox + t*dx, oy + t*dy) = 0
        num_iters = 10
        eps = 1e-9
        for _ in range(num_iters):
            px = ox + t * dx
            py = oy + t * dy
            pz = oz + t * dz
            sag = self._sag(px, py)
            f = pz - sag

            # df/dt = dz - d(sag)/dt
            n_raw = self._normal_local(px, py)  # (-dsag/dx, -dsag/dy, 1)
            dsag_dx = -n_raw[:, 0]
            dsag_dy = -n_raw[:, 1]
            df_dt = dz - dsag_dx * dx - dsag_dy * dy

            step = be.where(be.abs(df_dt) > 1e-15, f / df_dt, be.zeros_like(f))
            t = t - step

        # Validate hit
        px = ox + t * dx
        py = oy + t * dy
        pz = oz + t * dz
        sag_final = self._sag(px, py)
        residual = be.abs(pz - sag_final)

        in_aperture = (px**2 + py**2) <= self.aperture_radius**2
        hit_mask = (t > eps) & (residual < 1e-4) & in_aperture

        inf_arr = be.ones_like(t) * be.inf
        t_out = be.where(hit_mask, t, inf_arr)

        # Compute normals at hit points
        n_raw = self._normal_local(px, py)
        n_len = (n_raw * n_raw).sum(axis=1, keepdims=True) ** 0.5
        normals = n_raw / (n_len + 1e-30)

        # Flip to face incoming ray
        dot = (directions * normals).sum(axis=1, keepdims=True)
        normals = be.where(dot > 0, -normals, normals)

        return t_out, normals, hit_mask

    def bounding_box(self, transform: tuple[np.ndarray, np.ndarray]) -> AABB:
        """Return AABB in global coordinates.

        Args:
            transform: (translation, rotation_matrix).

        Returns:
            AABB in global frame.
        """
        t_vec = np.array(transform[0], dtype=float)
        R = np.array(transform[1], dtype=float)
        r = self.aperture_radius
        # Max sag (at edge of aperture)
        x_edge = np.array([r])
        y_edge = np.array([0.0])
        sag_edge = float(self._sag(x_edge, y_edge)[0])
        z_max = max(0.0, sag_edge)
        z_min = min(0.0, sag_edge)

        corners_local = np.array(
            [
                [-r, -r, z_min],
                [-r, r, z_min],
                [r, -r, z_min],
                [r, r, z_min],
                [-r, -r, z_max],
                [-r, r, z_max],
                [r, -r, z_max],
                [r, r, z_max],
            ],
            dtype=float,
        )
        corners_global = corners_local @ R.T + t_vec
        return AABB(corners_global.min(axis=0), corners_global.max(axis=0))


class ParaboloidGeometry(ConicGeometry):
    """Convenience subclass for paraboloid (conic constant K = -1).

    Attributes:
        radius: Radius of curvature [mm].
        aperture_radius: Semi-aperture [mm].
    """

    def __init__(self, radius: float, aperture_radius: float) -> None:
        """Initialize ParaboloidGeometry.

        Args:
            radius: Vertex radius of curvature [mm].
            aperture_radius: Aperture semi-diameter [mm].
        """
        super().__init__(radius=radius, conic=-1.0, aperture_radius=aperture_radius)
