# ruff: noqa: B027
"""Base Geometry

The base geometry class for all geometries.
Geometries are used to define the shape of optical elements.

Kramer Harrison, 2024
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseGeometry(ABC):
    """Base geometry for all geometries.

    Args:
        cs (CoordinateSystem): The coordinate system of the geometry.

    """

    _registry = {}

    def __init__(self, coordinate_system):
        self.cs = coordinate_system

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses."""
        super().__init_subclass__(**kwargs)
        BaseGeometry._registry[cls.__name__] = cls

    def __str__(self):
        return f"{self.__class__.__name__}"  # pragma: no cover

    @abstractmethod
    def sag(self, x=0, y=0):
        """Calculate the surface sag of the geometry.

        Args:
            x (float or be.ndarray, optional): The x-coordinate(s). Defaults to 0.
            y (float or be.ndarray, optional): The y-coordinate(s). Defaults to 0.

        Returns:
            float or be.ndarray: The surface sag of the geometry at the given
            coordinates.

        """
        # pragma: no cover

    @abstractmethod
    def distance(self, rays):
        """Find the propagation distance to the geometry.

        Args:
            rays (RealRays): The rays for which to calculate the distance to
                the geometry.

        Returns:
            be.ndarray: An array of propagation distances from each ray's
            current position to the geometry surface along the ray's direction.

        """
        # pragma: no cover

    @abstractmethod
    def scale(self, scale_factor: float):
        """Scale the geometry parameters.

        Args:
            scale_factor (float): The factor by which to scale the geometry.
        """
        # pragma: no cover

    @abstractmethod
    def surface_normal(self, rays):
        """Find the surface normal of the geometry at the given ray positions.

        Args:
            rays (RealRays): The rays, positioned at the surface, for which to
                calculate the surface normal.

        Returns:
            tuple[be.ndarray, be.ndarray, be.ndarray]: A tuple containing three
            arrays (nx, ny, nz) representing the x, y, and z components of the
            surface normals at each ray's intersection point.

        """
        # pragma: no cover

    @abstractmethod
    def flip(self):
        """Flip the geometry.

        This method should modify the geometry parameters such that the
        curvature is inverted. For example, the radius of curvature should
        change sign. Conic constants and polynomial coefficients should not
        change. The coordinate system of the geometry should not be modified.
        """
        pass  # pragma: no cover

    def set_radius(self, value: float) -> None:
        """Set the radius of curvature.

        Args:
            value (float): The new radius of curvature.

        Raises:
            AttributeError: If the geometry type does not support setting a radius.
        """
        raise AttributeError(
            f"Geometry type '{self.__class__.__name__}' does not support set_radius()."
        )

    def update_normalization(self, semi_aperture: float) -> None:
        """Update the normalization attributes of the geometry based on
        its defined normalization_mode ('auto' or 'manual').
        Base geometry generally does not maintain a normalization radius.

        Args:
            semi_aperture (float): The current semi-aperture of the surface.
        """
        pass  # pragma: no cover

    def localize(self, rays):
        """Convert rays from the global coordinate system to the local
        coordinate system.

        Args:
            rays (RealRays): The rays to convert.

        """
        self.cs.localize(rays)

    def globalize(self, rays):
        """Convert rays from the local coordinate system to the global
        coordinate system.

        Args:
            rays (RealRays): The rays to convert.

        """
        self.cs.globalize(rays)

    def to_dict(self):
        """Convert the geometry to a dictionary.

        Returns:
            dict: The dictionary representation of the geometry.

        """
        return {"type": self.__class__.__name__, "cs": self.cs.to_dict()}

    @classmethod
    def from_dict(cls, data):
        """Create a geometry from a dictionary.

        Args:
            data (dict): A dictionary containing the geometry data, including
                its 'type' and coordinate system 'cs'.

        Returns:
            BaseGeometry: An instance of a specific geometry subclass created
            from the dictionary data.

        """
        geometry_type = data.get("type")
        if geometry_type not in cls._registry:
            raise ValueError(f"Unknown geometry type: {geometry_type}")

        # Delegate to the correct subclass's from_dict
        return cls._registry[geometry_type].from_dict(data)
