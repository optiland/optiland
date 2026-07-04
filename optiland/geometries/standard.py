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

import warnings

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.base import BaseGeometry


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

    def distance(self, rays):
        """Find the propagation distance to the geometry for the given rays.

        Args:
            rays (RealRays): The rays for which to calculate the distance.

        Returns:
            be.ndarray: An array of distances from each ray's current position
            to its intersection point with the geometry.

        """
        if _is_radius_infinite(self.radius):
            # intersection with the plane z=0 is z0 + t*Nz = 0
            N_safe = be.where(be.abs(rays.N) > 1e-14, rays.N, 1e-14)
            return -rays.z / N_safe
        a = self.k * rays.N**2 + rays.L**2 + rays.M**2 + rays.N**2
        b = (
            2 * self.k * rays.N * rays.z
            + 2 * rays.L * rays.x
            + 2 * rays.M * rays.y
            - 2 * rays.N * self.radius
            + 2 * rays.N * rays.z
        )
        c = (
            self.k * rays.z**2
            - 2 * self.radius * rays.z
            + rays.x**2
            + rays.y**2
            + rays.z**2
        )

        # discriminant
        d = b**2 - 4 * a * c

        # Two solutions for distance to conic, computed via the numerically
        # stable form (Numerical Recipes / "citardauque" formula) rather
        # than the textbook (-b +/- sqrt(d)) / (2a). For rays close to the
        # optical axis (small L, M) and conics near a parabola (k = -1),
        # "a" is a tiny value dominated by floating-point noise rather than
        # 0 exactly, so the a == 0 guard below never triggers in practice.
        # The textbook formula then subtracts two nearly-equal numbers
        # (b and sqrt(d), both ~ -2*N*R) in the numerator while dividing by
        # a near-zero "a", amplifying that cancellation error by orders of
        # magnitude. This form avoids the cancellation entirely and reduces
        # continuously to the a == 0 (linear) solution as a -> 0, so no
        # separate branch is needed for that case.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sign_b = be.where(b >= 0, 1.0, -1.0)
            q = -0.5 * (b + sign_b * be.sqrt(d))
            t1 = q / a
            t2 = c / q

        # find intersection points in z
        z1 = rays.z + t1 * rays.N
        z2 = rays.z + t2 * rays.N

        # take intersection closest to z = 0 (i.e., vertex of geometry)
        geom_is_1 = be.abs(z1) <= be.abs(z2)
        t_geom = be.where(geom_is_1, t1, t2)

        # "Closest to vertex" is also always the root a ray genuinely enters
        # from the object side, *except* for rays steep enough that the two
        # roots' proximity to the vertex no longer tracks which one is
        # physically in front (e.g. extreme wide-angle field rays against a
        # convex surface). That essentially never happens for a ray still
        # comfortably clear of grazing incidence (|N| comfortably away from
        # zero), which covers ordinary usage -- including systems where rays
        # travel in the -z direction throughout (N uniformly negative) -- so
        # skip the (otherwise unconditional, since which rays in a batch need
        # it can't be known without computing it) disambiguation below
        # entirely when every ray in this call clears that bar.
        if bool(be.all(be.abs(rays.N) > 1e-2)):
            return t_geom

        # Only the entry-side dot product with the local normal actually
        # distinguishes the two roots for the remaining (rare) rays: take
        # "closest to vertex" unless it fails that check and the other root
        # passes it, in which case take the other root instead. Uses the
        # unnormalized normal -- only its sign matters here, so the
        # sqrt(mag) normalization used by surface_normal() (needed for
        # actual refraction) is skipped.
        x1 = rays.x + t1 * rays.L
        y1 = rays.y + t1 * rays.M
        x2 = rays.x + t2 * rays.L
        y2 = rays.y + t2 * rays.M

        with be.errstate(invalid="ignore"):
            dot1 = self._unnormalized_entry_dot(x1, y1, rays.L, rays.M, rays.N)
            dot2 = self._unnormalized_entry_dot(x2, y2, rays.L, rays.M, rays.N)

        # The "-N" term in the dot product bakes in a forward-propagation
        # (+z) assumption; for systems where rays travel in -z overall, the
        # entry side is the opposite sign, so flip the comparison by the
        # ray's own propagation direction.
        sign_n = be.where(rays.N < 0, -1.0, 1.0)
        entry1 = dot1 * sign_n < 0
        entry2 = dot2 * sign_n < 0

        geom_valid = be.where(geom_is_1, entry1, entry2)
        other_valid = be.where(geom_is_1, entry2, entry1)
        other_t = be.where(geom_is_1, t2, t1)

        use_other = be.logical_and(be.logical_not(geom_valid), other_valid)
        return be.where(use_other, other_t, t_geom)

    def _unnormalized_entry_dot(self, x, y, L, M, N):
        """Sign of the incident-direction dot the local surface normal, at
        local (x, y) points on the surface.

        Args:
            x (be.ndarray): Local x-coordinate(s) on the surface.
            y (be.ndarray): Local y-coordinate(s) on the surface.
            L (be.ndarray): Incident direction cosine, x-component.
            M (be.ndarray): Incident direction cosine, y-component.
            N (be.ndarray): Incident direction cosine, z-component.

        Returns:
            be.ndarray: ``dot(incident, normal)`` up to a positive scale
            factor -- unnormalized, since only its sign is used.
        """
        r2 = x**2 + y**2
        denom = self.radius * be.sqrt(1 - (1 + self.k) * r2 / self.radius**2)
        return L * x / denom + M * y / denom - N

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
