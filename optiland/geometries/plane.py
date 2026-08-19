"""Plane Geometry

The Plane geometry represents an infinite plane in two dimensions. The surface
is defined as an XY plane with z=0 for all points. Recall that surfaces are
always defined in the local coordinate system of the geometry.

Kramer Harrison, 2024
"""

from __future__ import annotations

import warnings

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.base import BaseGeometry


class Plane(BaseGeometry):
    """An infinite plane geometry.

    Args:
        cs (CoordinateSystem): The coordinate system of the plane geometry.

    """

    def __init__(self, coordinate_system):
        super().__init__(coordinate_system)
        self.radius = be.inf
        self.is_symmetric = True

    def __str__(self):
        return "Planar"

    def flip(self):
        """Flip the geometry.

        For a plane, this operation does nothing as its geometry is unchanged
        by flipping.
        """
        pass

    def scale(self, scale_factor: float):
        """Scale the geometry parameters.

        For a plane, this operation does nothing as its geometry is unchanged
        by scaling (radius remains infinite).

        Args:
            scale_factor (float): The factor by which to scale the geometry.
        """
        pass

    def sag(self, x=0, y=0):
        """Calculate the surface sag of the plane geometry.

        Args:
            x (float or be.ndarray, optional): The x-coordinate(s) of the point(s)
                on the plane. Defaults to 0.
            y (float or be.ndarray, optional): The y-coordinate(s) of the point(s)
                on the plane. Defaults to 0.

        Returns:
            be.ndarray or float: The surface sag of the plane at the given
            point(s), which is always 0.

        """
        if be.is_array_like(y):
            return be.zeros_like(y)
        return 0

    def distance(self, rays):
        """Find the propagation distance to the plane geometry.

        Args:
            rays (RealRays): The rays for which to calculate the distance to
                the plane.

        Returns:
            be.ndarray: An array of propagation distances from each ray's
            current position to the plane along the ray's direction.

        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = -rays.z / rays.N

        return t

    def surface_normal(self, rays):
        """Find the surface normal of the plane geometry at the given points.

        Args:
            rays (RealRays): The rays, positioned at the surface, for which to
                calculate the surface normal. This argument is used to determine
                the shape of the output arrays.

        Returns:
            tuple[be.ndarray, be.ndarray, be.ndarray]: A tuple containing three
            arrays (nx, ny, nz) representing the x, y, and z components of the
            surface normals. For a plane z=0 in local coordinates, this will be
            (0, 0, 1) for all points, broadcast to the shape of the input rays.

        """
        # The normal is always (0, 0, 1) in the local coordinate system of the plane.
        # We return arrays of the same shape as ray coordinates for consistency.
        zero_comp = be.zeros_like(rays.x)
        one_comp = be.ones_like(rays.x)
        return zero_comp, zero_comp, one_comp

    def to_dict(self):
        """Convert the plane geometry to a dictionary.

        Returns:
            dict: The dictionary representation of the plane geometry.

        """
        geometry_dict = super().to_dict()
        geometry_dict.update(
            {
                "radius": be.inf,
            },
        )
        return geometry_dict

    @classmethod
    def from_dict(cls, data):
        """Create a plane geometry from a dictionary.

        Args:
            data (dict): The dictionary representation of the plane geometry.

        Returns:
            Plane: An instance of the Plane geometry.

        """
        cs = CoordinateSystem.from_dict(data["cs"])
        return cls(cs)
