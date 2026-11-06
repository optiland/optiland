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

    def _relative_path_length(self, rays: RealRaysT, n_medium: float) -> BEArrayT:
        """Return a stable path correction for relative wavefront calculations.

        A geometry may remove a common optical-path constant, provided it uses
        the same constant for every ray and every call on that geometry.
        Chief referencing and piston removal both cancel this constant.
        This relationship assumes unit direction cosines, as supplied by
        wavefront tracing; implementations may normalize their input directions.
        Use ``path_length`` instead for physical intersection coordinates.
        """
        return self.path_length(rays, n_medium)

    @property
    @abstractmethod
    def radius(self) -> float:
        """The radius of the reference geometry (inf for plane)."""
        pass


class SphericalReference(ReferenceGeometry):
    """Spherical reference geometry (for focal systems).

    Args:
        center: (x, y, z) coordinates of the sphere center.
        radius: Finite sphere radius. Either sign describes the same sphere;
            intersections use its magnitude. The stored value retains its sign.

    Note:
        Positions, center, refractive index and directions must be finite, and
        direction vectors must have nonzero norms. Zero vectors and infinite
        radii are unsupported. For physical optical distances, ``path_length``
        requires unit direction cosines. ``_relative_path_length`` normalizes
        directions internally without modifying the rays.
    """

    def __init__(self, center: tuple[float, float, float], radius: float):
        self.center = center
        self._radius = radius

    def path_length(self, rays: RealRaysT, n_medium: float) -> BEArrayT:
        """Return ``n_medium * t`` for the backward ray ``position - t * direction``.

        The supplied direction vector is used unchanged. Thus ``t`` is a
        geometric distance only for unit directions; scaling a direction by a
        positive factor divides ``t`` by that factor. Radius sign has no effect.
        Choose the first nonnegative root, or the larger root if both are
        negative. A negative discriminant is clamped to zero.
        """
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

    def _relative_path_length(self, rays: RealRaysT, n_medium: float) -> BEArrayT:
        """Return normalized optical distance minus ``n_medium * abs(radius)``.

        In exact arithmetic this equals ``path_length`` evaluated on unit
        directions, minus the common optical radius. It is not generally
        ``path_length(rays, n_medium) - n_medium * abs(radius)`` for non-unit
        input directions: those are normalized here, including at zero radius.

        For a unit forward direction u and image displacement v from the
        sphere center, the backward intersection roots are h +/- sqrt(R**2 - q),
        where R = abs(radius), h = u.v and q = |u cross v|**2. Root selection
        and discriminant clamping follow ``path_length``.
        Rationalizing sqrt(R**2 - q) - R retains small path differences even
        when the exit-pupil radius is many orders larger than the system.
        Normalize the direction so roundoff in |u| cannot be amplified by R.
        """
        R = abs(self._radius)

        x, y, z = (
            rays.x - self.center[0],
            rays.y - self.center[1],
            rays.z - self.center[2],
        )
        norm = be.sqrt(rays.L**2 + rays.M**2 + rays.N**2)
        L, M, N = rays.L / norm, rays.M / norm, rays.N / norm
        h = L * x + M * y + N * z
        if R == 0:
            # Both roots coincide at h after clamping the discriminant -q.
            return n_medium * h
        q_over_R = (
            (M * z - N * y) ** 2 + (N * x - L * z) ** 2 + (L * y - M * x) ** 2
        ) / R
        root = be.sqrt(be.maximum(1 - q_over_R / R, 0))
        # Retain path_length's clamped discriminant and first-forward-root
        # convention for rays outside the reference sphere as well.
        offset = -be.minimum(q_over_R, R) / (1 + root)
        t_minus_R = be.where(h >= R * root, h - R * (1 + root), h + offset)
        return n_medium * t_minus_R

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
