"""Elliptical Aperture Module

This module contains the EllipticalAperture class, which represents an
elliptical aperture that clips rays based on their position.

Kramer Harrison, 2025
"""

from __future__ import annotations

from optiland.physical_apertures.base import BaseAperture


class EllipticalAperture(BaseAperture):
    """Represents an elliptical aperture that clips rays based on their position.

    Attributes:
        a (float): The semi-major axis of the ellipse.
        b (float): The semi-minor axis of the ellipse.
        offset_x (float): The x-coordinate of the aperture's center.
        offset_y (float): The y-coordinate of the aperture's center.

    """

    def __init__(self, a, b, offset_x=0, offset_y=0):
        super().__init__()
        self.a = a
        self.b = b
        self.offset_x = offset_x
        self.offset_y = offset_y

    @property
    def extent(self):
        """Returns the extent of the aperture.

        Returns:
            tuple: The extent of the aperture in the x and y directions.

        """
        return (
            self.offset_x - self.a,
            self.offset_x + self.a,
            self.offset_y - self.b,
            self.offset_y + self.b,
        )

    def contains(self, x, y):
        """Checks if the given point is inside the aperture.

        Args:
            x (np.ndarray): The x-coordinate of the point.
            y (np.ndarray): The y-coordinate of the point.

        Returns:
            np.ndarray: Boolean array indicating if the point is inside the
                aperture

        """
        x = x - self.offset_x
        y = y - self.offset_y
        return (x**2 / self.a**2 + y**2 / self.b**2) <= 1

    def scale(self, scale_factor):
        """Scales the aperture by the given factor.

        Args:
            scale_factor (float): The factor by which to scale the aperture.

        """
        self.a = self.a * scale_factor
        self.b = self.b * scale_factor
        self.offset_x = self.offset_x * scale_factor
        self.offset_y = self.offset_y * scale_factor

    def to_dict(self):
        """Convert the aperture to a dictionary.

        Returns:
            dict: The dictionary representation of the aperture.

        """
        aperture_dict = super().to_dict()
        aperture_dict["a"] = self.a
        aperture_dict["b"] = self.b
        aperture_dict["offset_x"] = self.offset_x
        aperture_dict["offset_y"] = self.offset_y
        return aperture_dict

    @classmethod
    def from_dict(cls, data):
        """Create an aperture from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the aperture.

        Returns:
            EllipticalAperture: The aperture object.

        """
        return cls(data["a"], data["b"], data["offset_x"], data["offset_y"])
