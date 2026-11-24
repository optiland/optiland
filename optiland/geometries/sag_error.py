"""Sag Error Geometry

The sag error geometry adds a Zernike sag error to any other geometry, its
base. It models a form error of a manufactured surface, such as the
irregularity of a tolerance analysis. The sag is

z(x, y) = base.sag(x, y) + sum_j c_j Z_j(x / R, y / R)

where Z_j is the j-th Fringe (University of Arizona) Zernike polynomial,
which has no normalization constant, so that c_j is the amplitude of a term
in lens units, and R is the normalization radius. The coefficients are in
Fringe order, starting at Z1 (piston): Z5 is rho^2 cos(2 phi) and Z9 is
6 rho^4 - 6 rho^2 + 1. The polynomials extend beyond R without a check.

The geometry shares the coordinate system of its base and never changes it.
The intersection is found by Newton-Raphson iteration, seeded by the
intersection with the base.

Kramer Harrison, 2026
"""

from __future__ import annotations

import inspect
from typing import Any

import optiland.backend as be
from optiland.geometries.base import BaseGeometry
from optiland.geometries.newton_raphson import NewtonRaphsonGeometry
from optiland.geometries.standard import StandardGeometry, _is_radius_infinite
from optiland.geometries.zernike import ZernikePolynomialGeometry
from optiland.zernike import ZernikeFringe

__all__ = ["SagErrorGeometry"]


class _Points:
    """The points at which ``surface_normal(rays)`` of a base is evaluated."""

    def __init__(self, x: Any, y: Any) -> None:
        self.x = x
        self.y = y


class _BaseIntersectionSeed(StandardGeometry):
    """Seeds the Newton-Raphson iteration with the intersection of the base.

    ``NewtonRaphsonGeometry`` seeds its iteration with the distance of the
    next class in the method resolution order, usually the base conic. This
    class takes that place for ``SagErrorGeometry``.
    """

    def distance(self, rays, aperture=None):
        """Return the distance to the base geometry."""
        if self._base_accepts_aperture:
            return self.base.distance(rays, aperture=aperture)
        return self.base.distance(rays)


class SagErrorGeometry(NewtonRaphsonGeometry, _BaseIntersectionSeed):
    """A geometry with a Zernike sag error added to a base geometry.

    The base is not changed. Its radius, conic and coordinate system are the
    radius, conic and coordinate system of this geometry, so changes of
    these values (for example by an optimization variable) change the base.
    If all coefficients are zero, the sag, the normals and the distances are
    those of the base, with the same bits.

    The paraxial radius includes the rho^2 part of the rotationally
    symmetric terms; the non-rotational terms (for example astigmatism) have
    no mean curvature and are not seen by the paraxial trace.

    Args:
        base: The geometry to which the sag error is added.
        coefficients: The Fringe Zernike coefficients in lens units, from Z1
            (piston).
        norm_radius: The normalization radius R of the Zernike polynomials.
        tol: Tolerance of the Newton-Raphson iteration. Defaults to 1e-10.
        max_iter: Maximum iterations of the Newton-Raphson iteration.
            Defaults to 100.
    """

    def __init__(
        self,
        base: BaseGeometry,
        coefficients: Any,
        norm_radius: float,
        tol: float = 1e-10,
        max_iter: int = 100,
    ) -> None:
        if not norm_radius > 0:
            raise ValueError(f"norm_radius must be positive, got {norm_radius}.")
        self.base = base
        BaseGeometry.__init__(self, base.cs)
        self.tol = tol
        self.max_iter = max_iter
        parameters = inspect.signature(base.distance).parameters
        self._base_accepts_aperture = "aperture" in parameters or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        )
        # A plane geometry evaluates the Fringe Zernike sum and its slopes.
        self._zernike = ZernikePolynomialGeometry(
            base.cs,
            radius=be.inf,
            coefficients=coefficients,
            zernike_type="fringe",
            norm_radius=norm_radius,
        )
        self._is_zero = not bool(be.any(self.coefficients != 0))

    def __str__(self) -> str:
        return "Sag Error"

    @property
    def coefficients(self) -> Any:
        """The Fringe Zernike coefficients in lens units, from Z1."""
        return self._zernike.coefficients

    @property
    def norm_radius(self) -> float:
        """The normalization radius of the Zernike polynomials."""
        return self._zernike.norm_radius

    @property
    def cs(self):
        """The coordinate system, shared with the base."""
        return self.base.cs

    @cs.setter
    def cs(self, value) -> None:
        self.base.cs = value

    @property
    def radius(self):
        """The radius of the base, or infinity if the base has none."""
        return getattr(self.base, "radius", be.inf)

    @radius.setter
    def radius(self, value) -> None:
        self.base.set_radius(value)

    @property
    def k(self):
        """The conic constant of the base, or 0 if the base has none."""
        return getattr(self.base, "k", 0.0)

    @k.setter
    def k(self, value) -> None:
        self.base.k = value

    @property
    def is_symmetric(self) -> bool:
        """True if the base and all nonzero Zernike terms are rotationally
        symmetric."""
        if not getattr(self.base, "is_symmetric", False):
            return False
        indices = self._zernike.zernike.indices
        return all(
            m == 0 or not bool(c != 0)
            for (_, m), c in zip(indices, self.coefficients, strict=True)
        )

    @property
    def paraxial_radius(self):
        """The vertex radius, with the rho^2 part of the symmetric terms."""
        radius = getattr(self.base, "paraxial_radius", self.radius)
        if self._is_zero:
            return radius
        a2 = 0.0
        indices = self._zernike.zernike.indices
        for (n, m), c in zip(indices, self.coefficients, strict=True):
            if m == 0 and n >= 2:
                a2 = a2 + c * ZernikeFringe._radial_coefficients(n, m)[1]
        if not bool(a2 != 0):
            return radius
        base_curvature = 0.0 if _is_radius_infinite(radius) else 1.0 / radius
        # a2 rho^2 = (a2 / R^2) r^2 = (c / 2) r^2 for a curvature c.
        curvature = base_curvature + 2.0 * a2 / self.norm_radius**2
        return be.inf if not bool(curvature != 0) else 1.0 / curvature

    def set_radius(self, value: float) -> None:
        """Set the radius of the base."""
        self.base.set_radius(value)

    def scale(self, scale_factor: float) -> None:
        """Scale the base, the normalization radius and the coefficients."""
        self.base.scale(scale_factor)
        self._zernike.scale(scale_factor)

    def flip(self) -> None:
        """Flip the base and change the sign of the coefficients."""
        self.base.flip()
        self._zernike.coefficients = -self.coefficients

    def update_normalization(self, semi_aperture: float) -> None:
        """Update the normalization of the base; R does not change."""
        self.base.update_normalization(semi_aperture)

    def _zernike_tables(self, x: Any, y: Any):
        return self._zernike._point_tables(x, y)

    def _base_slopes(self, x: Any, y: Any) -> tuple[Any, Any]:
        base = self.base
        if hasattr(base, "_surface_normal"):
            nx, ny, nz = base._surface_normal(x, y)
        elif hasattr(base, "_normal_components"):
            nx, ny, nz = base._normal_components(x, y)
        else:
            nx, ny, nz = base.surface_normal(_Points(x, y))
        return -nx / nz, -ny / nz

    def sag(self, x=0, y=0):
        """Return the sag of the base plus the Zernike sag error."""
        if self._is_zero:
            return self.base.sag(x, y)
        zernike = self._zernike._sag_from_tables(self._zernike_tables(x, y))
        return self.base.sag(x, y) + zernike

    def _surface_normal(self, x, y):
        """Return the unit normal from the sum of the base and Zernike slopes."""
        if self._is_zero:
            return self.base.surface_normal(_Points(x, y))
        fx, fy = self._base_slopes(x, y)
        zx, zy = self._zernike._slopes_from_tables(self._zernike_tables(x, y))
        fx, fy = fx + zx, fy + zy
        norm = be.sqrt(fx**2 + fy**2 + 1)
        return fx / norm, fy / norm, -be.ones_like(fx) / norm

    def surface_normal(self, rays):
        """Return the surface normal at the ray positions."""
        if self._is_zero:
            return self.base.surface_normal(rays)
        return self._surface_normal(rays.x, rays.y)

    def distance(self, rays, aperture=None):
        """Return the distance to the surface along each ray.

        Newton-Raphson iteration from the intersection with the base, or the
        distance to the base if all coefficients are zero.
        """
        if self._is_zero:
            return _BaseIntersectionSeed.distance(self, rays, aperture)
        return super().distance(rays, aperture=aperture)

    def to_dict(self) -> dict:
        """Convert the geometry to a dictionary, with the base."""
        return {
            "type": self.__class__.__name__,
            "cs": self.cs.to_dict(),
            "base": self.base.to_dict(),
            "coefficients": [float(c) for c in be.to_numpy(self.coefficients)],
            "norm_radius": float(self.norm_radius),
            "tol": self.tol,
            "max_iter": self.max_iter,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SagErrorGeometry:
        """Create a sag error geometry from a dictionary.

        Args:
            data: The dictionary of ``to_dict``.

        Returns:
            SagErrorGeometry: The geometry, with a new base.
        """
        missing = {"base", "coefficients", "norm_radius"} - data.keys()
        if missing:
            raise ValueError(f"Missing required keys: {missing}")
        return cls(
            BaseGeometry.from_dict(data["base"]),
            be.array(data["coefficients"]),
            data["norm_radius"],
            tol=data.get("tol", 1e-10),
            max_iter=data.get("max_iter", 100),
        )
