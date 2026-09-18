"""Reference Geometry Module

This module defines the geometry for the reference surface used in wavefront
analysis. It supports both spherical (focal) and planar (afocal) references.

Kramer Harrison, 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import isfinite
from numbers import Real
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
    """Planar reference geometry for afocal systems.

    The plane is defined by exactly three finite scalar coordinates and a
    finite, exactly nonzero three-component normal. The normal is neither
    normalized nor tested against an epsilon, so rescaling or reversing it
    leaves the represented plane unchanged. Extremely small accepted normals
    remain subject to ordinary floating-point underflow during later arithmetic.
    Validation uses each component's own precision, retaining the original real
    scalars or zero-dimensional arrays for computation and differentiation.

    Args:
        point: (x, y, z) point on the plane.
        normal: (nx, ny, nz) normal vector of the plane.

    Raises:
        ValueError: If either vector does not contain exactly three finite
            scalar components, or if the normal is exactly zero.
    """

    def __init__(
        self,
        point: tuple[float | BEArrayT, float | BEArrayT, float | BEArrayT],
        normal: tuple[float | BEArrayT, float | BEArrayT, float | BEArrayT],
    ) -> None:
        vectors = []
        for name, components in (("point", point), ("normal", normal)):
            try:
                values = tuple(components)
            except TypeError as exc:
                raise ValueError(
                    f"Plane {name} must contain exactly three finite scalar components."
                ) from exc

            if len(values) != 3:
                raise ValueError(
                    f"Plane {name} must contain exactly three finite scalar components."
                )

            for component in values:
                try:
                    if isinstance(component, Real):
                        # Backend scalar conversion could narrow a finite Python
                        # float to the configured working precision.
                        valid = isfinite(component)
                    elif isinstance(component, be.ndarray) and component.shape == ():
                        # Explicit dtype also preserves NumPy scalar arrays when
                        # checking them with the Torch backend. Store no conversion.
                        scalar = be.asarray(component, dtype=component.dtype)
                        # Reject unsupported nonnumeric dtypes before real(),
                        # which can unwrap a NumPy object scalar to a Python value.
                        valid = be.all(be.isfinite(scalar)) and (
                            scalar.dtype == be.real(scalar).dtype
                        )
                    else:
                        valid = False
                except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
                    raise ValueError(
                        f"Plane {name} must contain exactly three finite scalar "
                        "components."
                    ) from exc
                if not valid:
                    raise ValueError(
                        f"Plane {name} must contain exactly three finite scalar "
                        "components."
                    )

            vectors.append(values)

        self.point, self.normal = vectors
        if all(bool(component == 0) for component in self.normal):
            raise ValueError("Plane normal must be nonzero.")

    def path_length(self, rays: RealRaysT, n_medium: float) -> BEArrayT:
        """Return signed optical distance to the plane along reversed rays.

        Every finite nonzero ray-plane denominator is used exactly, without
        normalization or an epsilon band. A finite parallel ray returns zero
        when its origin is exactly coplanar and NaN otherwise. Nonfinite
        numerators or denominators also return NaN. The medium index is assumed
        finite; signed geometric intersections are scaled by it.

        Masking protects discarded exact-parallel/nonfinite-input lanes only
        when evaluated arithmetic and derivatives are representable. Overflow
        in finite-input lanes, even if subsequently discarded by a caller, can
        still contaminate shared-medium gradients.

        Args:
            rays: Rays whose positions and forward direction cosines define
                the reverse intersection lines.
            n_medium: Refractive index used to convert geometric distance to
                optical path length.

        Returns:
            Signed optical path lengths preserving the ray array shape and
            following normal backend dtype-promotion rules for the ray, plane,
            and medium operands.
        """
        L, M, N = -rays.L, -rays.M, -rays.N
        xr, yr, zr = rays.x, rays.y, rays.z
        px, py, pz = self.point
        nx, ny, nz = self.normal

        num = (xr - px) * nx + (yr - py) * ny + (zr - pz) * nz
        den = L * nx + M * ny + N * nz

        finite = be.isfinite(num) & be.isfinite(den)
        unique = finite & (den != 0)
        coplanar = finite & (den == 0) & (num == 0)

        # Both where branches may be evaluated, so make division safe first.
        # Scalar branches adopt the computed operands' dtype rather than the
        # backend's configured default precision.
        safe_num = be.where(unique, num, 0.0)
        safe_den = be.where(unique, den, 1.0)
        path = n_medium * (-safe_num / safe_den)

        # Insert NaN last so masked lanes do not poison n_medium's gradient.
        # This does not protect against overflow in evaluated finite arithmetic.
        return be.where(unique | coplanar, path, be.nan)

    @property
    def radius(self) -> float:
        return float("inf")
