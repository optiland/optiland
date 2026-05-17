"""Standard Grating Geometry

The Standard grating geometry represents a surface defined by a sphere or conic in two
dimensions. The surface is defined as:

z = r^2 / (R * (1 + sqrt(1 - (1 + k) * r^2 / R^2)))

where
- r^2 = x^2 + y^2
- R is the radius of curvature
- k is the conic constant

Kramer Harrison, 2024  Matteo Taccola 2025
"""

from __future__ import annotations

import warnings

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.base import BaseGeometry


class StandardGratingGeometry(BaseGeometry):
    """Represents a standard geometry with a given coordinate system, radius, and
    conic.

    Args:
        coordinate_system (CoordinateSystem): The coordinate system of the geometry.
        radius (float): The radius of curvature of the geometry.
        grating_order (int): The grating diffraction order
        grating_period (float): The grating period (units micrometers)
        groove_orientation_angle (float): The groove orientation angle (units radians)
        conic (float, optional): The conic constant of the geometry. Defaults to 0.0.

    Methods:
        sag(x=0, y=0): Calculates the surface sag of the geometry at the given
            coordinates.
        distance(rays): Finds the propagation distance to the geometry for the
            given rays.
        surface_normal(rays): Calculates the surface normal of the geometry at
            the given ray positions.
        grating_vector(rays): Compute the grating vector at the given ray positions.


    """

    def __init__(
        self,
        coordinate_system,
        radius,
        grating_order,
        grating_period,
        groove_orientation_angle,
        conic=0.0,
    ):
        super().__init__(coordinate_system)
        self.radius = be.array(radius)
        self.k = be.array(conic)
        self.grating_order = be.array(grating_order)
        self.grating_period = be.array(grating_period)
        self.groove_orientation_angle = be.array(groove_orientation_angle)
        self.is_symmetric = True

    def __str__(self):
        return "StandardGrating"

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
        self.grating_period = self.grating_period * scale_factor

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

    def _tangent(self, x=0, y=0, alfa=0):
        """Compute the unit vector that is lying on the tangent plane in (x,y,z) and
        tangent to the grating line

        Args:
            x (float or be.ndarray, optional): The x-coordinate(s). Defaults to 0.
            y (float or be.ndarray, optional): The y-coordinate(s). Defaults to 0.
            alfa (float): The groove orientation angle in radians. Defaults to 0.

        Returns:
            tuple[be.ndarray, be.ndarray, be.ndarray]: The x, y, and z
            components of the tangent vector.
        """
        dzdx = (
            (x + y * be.tan(alfa))
            * (
                2
                * self.radius**2
                * be.sqrt(
                    (self.radius**2 - (self.k + 1) * (x**2 + y**2)) / self.radius**2
                )
                * (
                    be.sqrt(
                        (self.radius**2 - (self.k + 1) * (x**2 + y**2)) / self.radius**2
                    )
                    + 1
                )
                + (self.k + 1) * (x**2 + y**2)
            )
            / (
                self.radius**3
                * be.sqrt(
                    (self.radius**2 - (self.k + 1) * (x**2 + y**2)) / self.radius**2
                )
                * (
                    be.sqrt(
                        (self.radius**2 - (self.k + 1) * (x**2 + y**2)) / self.radius**2
                    )
                    + 1
                )
                ** 2
            )
        )

        tx = be.ones_like(dzdx)
        ty = be.ones_like(dzdx) * be.tan(alfa)
        tz = dzdx

        # Normalize t
        norm_t = be.sqrt(tx**2 + ty**2 + tz**2)
        tx = tx / norm_t
        ty = ty / norm_t
        tz = tz / norm_t
        return tx, ty, tz

    def distance(self, rays):
        """Find the propagation distance to the geometry for the given rays.

        Args:
            rays (RealRays): The rays for which to calculate the distance.

        Returns:
            be.ndarray: An array of distances from each ray's current position
            to its intersection point with the geometry.

        """
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

        # two solutions for distance to conic
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t1 = (-b + be.sqrt(d)) / (2 * a)
            t2 = (-b - be.sqrt(d)) / (2 * a)

        # find intersection points in z
        z1 = rays.z + t1 * rays.N
        z2 = rays.z + t2 * rays.N

        # take intersection closest to z = 0 (i.e., vertex of geometry)
        t = be.where(be.abs(z1) <= be.abs(z2), t1, t2)

        # handle case when a = 0
        # Assumes b is not zero when a is zero, based on original logic.
        t = be.where(a == 0, -c / b, t)

        return t

    def surface_normal(self, rays):
        """Calculate the surface normal of the geometry at the given points.

        Args:
            rays (RealRays): The rays, positioned at the surface, for which to
                calculate the surface normals.

        Returns:
            tuple[be.ndarray, be.ndarray, be.ndarray]: The x, y, and z
            components of the surface normal vectors.

        """
        r2 = rays.x**2 + rays.y**2

        denom = self.radius * be.sqrt(1 - (1 + self.k) * r2 / self.radius**2)
        dfdx = rays.x / denom
        dfdy = rays.y / denom
        dfdz = -1

        mag = be.sqrt(dfdx**2 + dfdy**2 + dfdz**2)

        nx = dfdx / mag
        ny = dfdy / mag
        nz = dfdz / mag

        return nx, ny, nz

    def grating_vector(self, rays):
        """Calculate the grating normal vector of the geometry at the given points.

        Args:
            rays (RealRays): The rays, positioned at the surface, for which to
                calculate the surface normals.

        Returns:
            tuple[be.ndarray, be.ndarray, be.ndarray]: The x, y, and z
            components of the grating vector.

        """
        nx, ny, nz = self.surface_normal(rays)
        tx, ty, tz = self._tangent(rays.x, rays.y, self.groove_orientation_angle)
        fx = ny * tz - nz * ty
        fy = -nx * tz + nz * tx
        fz = nx * ty - ny * tx

        mag = be.sqrt(fx**2 + fy**2 + fz**2)
        fx = fx / mag
        fy = fy / mag
        fz = fz / mag

        return -fx, -fy, -fz

    def to_dict(self):
        """Convert the geometry to a dictionary.

        Returns:
            dict: The dictionary representation of the geometry.

        """
        geometry_dict = super().to_dict()
        geometry_dict.update(
            {
                "radius": float(self.radius),
                "conic": float(self.k),
                "order": float(self.grating_order),
                "period": float(self.grating_period),
                "angle": float(self.groove_orientation_angle),
            }
        )
        return geometry_dict

    @classmethod
    def from_dict(cls, data):
        """Create a geometry from a dictionary.

        Args:
            data (dict): The dictionary representation of the geometry.

        Returns:
            StandardGratingGeometry: An instance of StandardGratingGeometry.

        """
        required_keys = {"cs", "radius", "order", "period", "angle"}
        if not required_keys.issubset(data):
            missing = required_keys - data.keys()
            raise ValueError(f"Missing required keys: {missing}")

        cs = CoordinateSystem.from_dict(data["cs"])

        return cls(
            cs,
            data["radius"],
            data["order"],
            data["period"],
            data["angle"],
            data.get("conic", 0.0),
        )
