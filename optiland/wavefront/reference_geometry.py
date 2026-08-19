"""Reference Geometry Module

This module defines the geometry for the reference surface used in wavefront
analysis. It supports both spherical (focal) and planar (afocal) references.

Kramer Harrison, 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import optiland.backend as be

if TYPE_CHECKING:
    from optiland._types import BEArrayT, RealRaysT


class ReferenceGeometry(ABC):
    """Abstract base class for reference geometries."""

    @abstractmethod
    def path_length(self, rays: RealRaysT, n_medium: float) -> BEArrayT:
        """Calculates optical path length from ray positions to the reference.

        Args:
            rays: The rays at the image surface (containing x, y, z, L, M, N).
            n_medium: The refractive index of the medium.

        Returns:
            The optical path length correction.
        """
        pass

    @property
    @abstractmethod
    def radius(self) -> float:
        """The radius of the reference geometry (inf for plane)."""
        pass


class SphericalReference(ReferenceGeometry):
    """Spherical reference geometry (for focal systems).

    Args:
        center: (x, y, z) coordinates of the sphere center.
        radius: Radius of the sphere.
    """

    def __init__(self, center: tuple[float, float, float], radius: float):
        self.center = center
        self._radius = radius

    def path_length(self, rays: RealRaysT, n_medium: float) -> BEArrayT:
        xc, yc, zc = self.center
        xr, yr, zr = rays.x, rays.y, rays.z
        L, M, N = -rays.L, -rays.M, -rays.N
        R = self._radius

        a = L**2 + M**2 + N**2
        b = 2 * (L * (xr - xc) + M * (yr - yc) + N * (zr - zc))
        c = (
            xr**2
            + yr**2
            + zr**2
            - 2 * (xr * xc + yr * yc + zr * zc)
            + xc**2
            + yc**2
            + zc**2
            - R**2
        )
        d = b**2 - 4 * a * c
        d = be.where(d < 0, 0, d)

        t1 = (-b - be.sqrt(d)) / (2 * a)
        t2 = (-b + be.sqrt(d)) / (2 * a)
        t = be.where(t1 < 0, t2, t1)

        return n_medium * t

    @property
    def radius(self) -> float:
        return self._radius


class PlanarReference(ReferenceGeometry):
    """Planar reference geometry (for afocal systems).

    Args:
        point: (x, y, z) point on the plane.
        normal: (nx, ny, nz) normal vector of the plane.
    """

    def __init__(
        self, point: tuple[float, float, float], normal: tuple[float, float, float]
    ):
        self.point = point
        self.normal = normal

    def path_length(self, rays: RealRaysT, n_medium: float) -> BEArrayT:
        # Intersection of line P = P0 + t*D with plane (P - PlanePt) . Normal = 0
        # (P0 + t*D - PlanePt) . Normal = 0
        # t * (D . Normal) + (P0 - PlanePt) . Normal = 0
        # t = - ((P0 - PlanePt) . Normal) / (D . Normal)

        # We trace backwards from image plane
        L, M, N = -rays.L, -rays.M, -rays.N
        xr, yr, zr = rays.x, rays.y, rays.z
        px, py, pz = self.point
        nx, ny, nz = self.normal

        num = (xr - px) * nx + (yr - py) * ny + (zr - pz) * nz
        den = L * nx + M * ny + N * nz

        # Avoid division by zero
        den = be.where(be.abs(den) < 1e-12, 1e-12, den)

        t = -num / den

        return n_medium * t

    @property
    def radius(self) -> float:
        return float("inf")
