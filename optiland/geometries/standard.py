"""Standard Geometry

The Standard geometry represents a surface defined by a sphere or conic in two
dimensions. The surface is defined as:

z = r^2 / (R * (1 + sqrt(1 - (1 + k) * r^2 / R^2)))

where
- r^2 = x^2 + y^2
- R is the radius of curvature
- k is the conic constant

Kramer Harrison, 2024
"""

from __future__ import annotations

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.base import BaseGeometry
from optiland.utils import machine_eps

# Multiplier on the machine epsilon of the ray coordinates when building the
# scale-aware forward floor for conic root selection. The floor must reject a
# ray's own starting point (an exact t = 0 plus its round-off halo) without
# ever rejecting a genuine propagation to a nearby surface. Matches the
# conservative multiplier used by the Newton-Raphson thresholds.
_FORWARD_EPS_MULTIPLIER = 32.0


def _is_radius_infinite(radius):
    """Checks if the given radius represents an infinite radius (a plane)."""
    is_inf_tensor = be.isinf(radius)
    if hasattr(is_inf_tensor, "ndim") and is_inf_tensor.ndim > 0:
        return bool(be.all(is_inf_tensor))
    return (
        bool(is_inf_tensor.item())
        if hasattr(is_inf_tensor, "item")
        else bool(is_inf_tensor)
    )


def _conic_intersection_distance(rays, radius, conic, aperture=None):
    """Propagation distance of rays to a conic, selecting the physical root.

    The line-conic intersection is the quadratic ``a t^2 + b t + c = 0``,
    solved in the numerically stable (Numerical Recipes / "citardauque") form
    rather than the textbook ``(-b +/- sqrt(d)) / (2a)``. For rays close to
    the optical axis (small L, M) and conics near a parabola (k = -1), ``a``
    is a tiny value dominated by floating-point noise; the textbook formula
    then subtracts two nearly-equal numbers (b and sqrt(d)) while dividing by
    that near-zero ``a``, amplifying the cancellation by orders of magnitude.
    The stable form avoids the cancellation entirely and reduces continuously
    to the linear solution as a -> 0.

    The quadric contains every crossing of the ray with the *infinite
    mathematical* conic: points the sag function does not describe (the far
    side of a sphere or ellipsoid, the detached second sheet of a k < -1
    hyperboloid) as well as crossings behind the ray. A root is therefore
    admissible only when it is

    - in front of the ray: ``t`` above a scale-aware positive floor, and
    - on the sheet the sag function describes: on the surface,
      ``sqrt(1 - (1 + k) r^2 / R^2) = 1 - (1 + k) z / R``, so the sag sheet
      is ``1 - (1 + k) z / R >= 0``.

    Among admissible roots, ones landing inside ``aperture`` (when given) are
    preferred -- for off-axis sections of a conic, e.g. off-axis parabolic
    mirrors, both roots can be genuine forward hits on the surface and only
    the aperture identifies the used region -- and the nearest root of the
    best tier is returned. Rays with no admissible root (a genuine miss, or a
    ray leaving the surface's neighborhood) fall back to the legacy
    vertex-nearest root, NaN when the discriminant is negative.

    All masking is per ray, and denominators and the radicand are guarded
    *before* dividing / taking the square root: a masked inf or NaN would
    still enter the autograd graph and backpropagate as NaN.

    Args:
        rays (RealRays): The rays, in the local frame of the surface.
        radius (be.ndarray or float): Radius of curvature of the conic.
        conic (be.ndarray or float): Conic constant of the conic.
        aperture (BaseAperture, optional): Physical aperture of the surface,
            used only to prefer intersections on the used region. When None,
            all admissible roots are equal candidates.

    Returns:
        be.ndarray: Propagation distance per ray.
    """
    if _is_radius_infinite(radius):
        # intersection with the plane z=0 is z0 + t*Nz = 0
        N_safe = be.where(be.abs(rays.N) > 1e-14, rays.N, 1e-14)
        return -rays.z / N_safe

    k = conic
    a = k * rays.N**2 + rays.L**2 + rays.M**2 + rays.N**2
    b = (
        2 * k * rays.N * rays.z
        + 2 * rays.L * rays.x
        + 2 * rays.M * rays.y
        - 2 * rays.N * radius
        + 2 * rays.N * rays.z
    )
    c = (
        k * rays.z**2
        - 2 * radius * rays.z
        + rays.x**2
        + rays.y**2
        + rays.z**2
    )

    d = b**2 - 4 * a * c
    d_ok = d >= 0

    eps = machine_eps(d)
    ones = be.ones_like(d)

    sqrt_d = be.where(d_ok, be.sqrt(be.maximum(d, eps)), be.zeros_like(d))
    sign_b = be.where(b >= 0, 1.0, -1.0)
    q = -0.5 * (b + sign_b * sqrt_d)
    a_ok = be.abs(a) > eps
    q_ok = be.abs(q) > eps
    t1 = q / be.where(a_ok, a, ones)
    t2 = c / be.where(q_ok, q, ones)
    solvable1 = d_ok & a_ok
    solvable2 = d_ok & q_ok

    x1 = rays.x + t1 * rays.L
    y1 = rays.y + t1 * rays.M
    z1 = rays.z + t1 * rays.N
    x2 = rays.x + t2 * rays.L
    y2 = rays.y + t2 * rays.M
    z2 = rays.z + t2 * rays.N

    # Scale-aware forward floor: the ray positions enter the quadratic
    # coefficients, so a would-be-zero root carries their round-off.
    position_scale = 1.0 + be.abs(rays.x) + be.abs(rays.y) + be.abs(rays.z)
    t_min = _FORWARD_EPS_MULTIPLIER * eps * position_scale

    sheet1 = 1 - (1 + k) * z1 / radius >= 0
    sheet2 = 1 - (1 + k) * z2 / radius >= 0

    valid1 = solvable1 & be.isfinite(t1) & (t1 > t_min) & sheet1
    valid2 = solvable2 & be.isfinite(t2) & (t2 > t_min) & sheet2

    if aperture is not None:
        pref1 = valid1 & aperture.contains(x1, y1)
        pref2 = valid2 & aperture.contains(x2, y2)
    else:
        pref1 = valid1
        pref2 = valid2

    # Nearest root of the best available tier: in-aperture, then any
    # admissible, then the legacy vertex-nearest fallback.
    first1 = t1 <= t2

    pick_pref1 = pref1 & (~pref2 | first1)
    t_pref = be.where(pick_pref1, t1, t2)
    have_pref = pref1 | pref2

    pick_valid1 = valid1 & (~valid2 | first1)
    t_valid = be.where(pick_valid1, t1, t2)
    have_valid = valid1 | valid2

    inf = be.full_like(d, be.inf)
    abs_z1 = be.where(solvable1, be.abs(z1), inf)
    abs_z2 = be.where(solvable2, be.abs(z2), inf)
    t_vertex = be.where(abs_z1 <= abs_z2, t1, t2)
    t_fallback = be.where(d_ok, t_vertex, be.full_like(d, be.nan))

    return be.where(have_pref, t_pref, be.where(have_valid, t_valid, t_fallback))


class StandardGeometry(BaseGeometry):
    """Represents a standard geometry with a given coordinate system, radius, and
    conic.

    Args:
        coordinate_system (CoordinateSystem): The coordinate system of the geometry.
        radius (float): The radius of curvature of the geometry.
        conic (float, optional): The conic constant of the geometry. Defaults to 0.0.

    Methods:
        sag(x=0, y=0): Calculates the surface sag of the geometry at the given
            coordinates.
        distance(rays): Finds the propagation distance to the geometry for the
            given rays.
        surface_normal(rays): Calculates the surface normal of the geometry at
            the given ray positions.

    """

    def __init__(self, coordinate_system, radius, conic=0.0):
        super().__init__(coordinate_system)
        self.radius = be.array(radius)
        self.k = be.array(conic)
        self.is_symmetric = True

    def __str__(self):
        return "Standard"

    def set_radius(self, value: float) -> None:
        """Set the radius of curvature.

        Args:
            value (float): The new radius of curvature.
        """
        self.radius = be.array(value)

    def flip(self):
        """Flip the geometry.

        Changes the sign of the radius of curvature.
        The conic constant remains unchanged.
        """
        self.radius = -self.radius

    def scale(self, scale_factor: float):
        """Scale the geometry parameters.

        Args:
            scale_factor (float): The factor by which to scale the geometry.
        """
        self.radius = self.radius * scale_factor

    def sag(self, x=0, y=0):
        """Calculate the surface sag of the geometry at the given coordinates.

        Args:
            x (float or be.ndarray, optional): The x-coordinate(s). Defaults to 0.
            y (float or be.ndarray, optional): The y-coordinate(s). Defaults to 0.

        Returns:
            be.ndarray or float: The sag value(s) at the given coordinates.

        """
        r2 = x**2 + y**2
        return r2 / (
            self.radius * (1 + be.sqrt(1 - (1 + self.k) * r2 / self.radius**2))
        )

    def distance(self, rays, aperture=None):
        """Find the propagation distance to the geometry for the given rays.

        Args:
            rays (RealRays): The rays for which to calculate the distance.
            aperture (BaseAperture, optional): The physical aperture of the
                surface this geometry belongs to. When given, intersections
                landing inside the aperture are preferred over intersections
                the aperture would clip. This matters for off-axis sections
                of a conic (e.g. off-axis parabolic mirrors), where both
                quadratic roots can be genuine forward hits on the surface
                and only the aperture identifies the used region.

        Returns:
            be.ndarray: An array of distances from each ray's current position
            to its intersection point with the geometry.

        """
        return _conic_intersection_distance(rays, self.radius, self.k, aperture)

    def _normal_components(self, x, y):
        """Compute the normalized surface normal at local (x, y) points on
        the surface.

        Args:
            x (be.ndarray): Local x-coordinate(s) on the surface.
            y (be.ndarray): Local y-coordinate(s) on the surface.

        Returns:
            tuple[be.ndarray, be.ndarray, be.ndarray]: The x, y, and z
            components of the surface normal vectors.
        """
        r2 = x**2 + y**2

        denom = self.radius * be.sqrt(1 - (1 + self.k) * r2 / self.radius**2)
        dfdx = x / denom
        dfdy = y / denom
        dfdz = -1

        mag = be.sqrt(dfdx**2 + dfdy**2 + dfdz**2)

        return dfdx / mag, dfdy / mag, dfdz / mag

    def surface_normal(self, rays):
        """Calculate the surface normal of the geometry at the given points.

        Args:
            rays (RealRays): The rays, positioned at the surface, for which to
                calculate the surface normals.

        Returns:
            tuple[be.ndarray, be.ndarray, be.ndarray]: The x, y, and z
            components of the surface normal vectors.

        """
        return self._normal_components(rays.x, rays.y)

    def to_dict(self):
        """Convert the geometry to a dictionary.

        Returns:
            dict: The dictionary representation of the geometry.

        """
        geometry_dict = super().to_dict()
        geometry_dict.update({"radius": float(self.radius), "conic": float(self.k)})
        return geometry_dict

    @classmethod
    def from_dict(cls, data):
        """Create a geometry from a dictionary.

        Args:
            data (dict): The dictionary representation of the geometry.

        Returns:
            StandardGeometry: An instance of StandardGeometry.

        """
        required_keys = {"cs", "radius"}
        if not required_keys.issubset(data):
            missing = required_keys - data.keys()
            raise ValueError(f"Missing required keys: {missing}")

        cs = CoordinateSystem.from_dict(data["cs"])

        return cls(cs, data["radius"], data.get("conic", 0.0))
