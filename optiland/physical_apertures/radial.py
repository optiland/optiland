"""Radial Aperture Module

This module contains the RadialAperture class, which represents a radial
aperture that clips rays based on their distance from the origin.

Kramer Harrison, 2025
"""

from __future__ import annotations

from numbers import Number

from optiland.physical_apertures.base import BaseAperture


def configure_aperture(aperture):
    """Normalize a raw aperture spec (None, number, or BaseAperture)."""
    if aperture is None:
        return None
    elif isinstance(aperture, Number):
        radius = aperture / 2
        return RadialAperture(r_max=radius)
    elif isinstance(aperture, BaseAperture):
        return aperture
    else:
        raise ValueError(
            f"Invalid `aperture` provided: {aperture}. Must be scalar "
            f"or of type `BaseAperture`."
        )


class RadialAperture(BaseAperture):
    """Represents a radial aperture that clips rays based on their distance from
    the origin.

    Attributes:
        r_max (float): The maximum radius allowed for the rays.
        r_min (float): The minimum radius allowed for the rays. Defaults to 0.

    """

    def __init__(self, r_max: float, r_min: float = 0):
        super().__init__()
        self.r_max = r_max
        self.r_min = r_min

    @property
    def extent(self):
        """Returns the extent of the aperture.

        Returns:
            tuple: The extent of the aperture in the x and y directions.

        """
        return -self.r_max, self.r_max, -self.r_max, self.r_max

    def contains(self, x, y):
        """Checks if the given point is inside the aperture.

        Args:
            x (np.ndarray): The x-coordinate of the point.
            y (np.ndarray): The y-coordinate of the point.

        Returns:
            np.ndarray: Boolean array indicating if the point is inside the
                aperture

        """
        radius2 = x**2 + y**2
        inside = (radius2 <= self.r_max**2) & (radius2 >= self.r_min**2)
        return inside

    def scale(self, scale_factor):
        """Scales the aperture by the given factor.

        Args:
            scale_factor (float): The factor by which to scale the aperture.

        """
        self.r_max = self.r_max * scale_factor
        self.r_min = self.r_min * scale_factor

    def to_dict(self):
        """Convert the aperture to a dictionary.

        Returns:
            dict: The dictionary representation of the aperture.

        """
        aperture_dict = super().to_dict()
        aperture_dict["r_max"] = self.r_max
        aperture_dict["r_min"] = self.r_min
        return aperture_dict

    @classmethod
    def from_dict(cls, data):
        """Create an aperture from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the aperture.

        Returns:
            RadialAperture: The aperture object.

        """
        return cls(data["r_max"], data["r_min"])
