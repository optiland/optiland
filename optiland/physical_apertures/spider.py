"""Spider Aperture Module

This module contains the SpiderAperture class, which represents the vanes of a
spider: straight arms of equal width that run outward from a common centre and
block the rays that fall on them.
"""

from __future__ import annotations

import math

import optiland.backend as be
from optiland.physical_apertures.base import BaseAperture


class SpiderAperture(BaseAperture):
    """Represents a spider that blocks the rays falling on its arms.

    The arms are half-lines of equal width from the centre, equally spaced in
    angle, the first along +x. A ray is blocked when it lies on an arm's side
    of the centre and within half the width of the arm's centreline; every
    other ray passes, with no outer limit.

    Attributes:
        width (float): The full width of each arm.
        num_arms (int): The number of arms.
        offset_x (float): The x-coordinate of the spider's centre.
        offset_y (float): The y-coordinate of the spider's centre.

    """

    def __init__(self, width, num_arms, offset_x=0, offset_y=0):
        super().__init__()
        if int(num_arms) < 1:
            raise ValueError("A spider needs at least one arm.")
        self.width = width
        self.num_arms = int(num_arms)
        self.offset_x = offset_x
        self.offset_y = offset_y

    @property
    def extent(self):
        """Returns the extent of the aperture.

        Returns:
            tuple: The extent of the aperture in the x and y directions. The
                arms have no outer limit.

        """
        return -be.inf, be.inf, -be.inf, be.inf

    def contains(self, x, y):
        """Checks if the given point is inside the aperture.

        Args:
            x (np.ndarray): The x-coordinate of the point.
            y (np.ndarray): The y-coordinate of the point.

        Returns:
            np.ndarray: Boolean array indicating if the point is inside the
                aperture, that is, not on an arm.

        """
        u = x - self.offset_x
        v = y - self.offset_y
        half_width = self.width / 2
        blocked = be.zeros_like(u) != 0
        for k in range(self.num_arms):
            angle = 2 * math.pi * k / self.num_arms
            cos, sin = math.cos(angle), math.sin(angle)
            along = u * cos + v * sin
            across = -u * sin + v * cos
            blocked = blocked | ((along >= 0) & (be.abs(across) <= half_width))
        return ~blocked

    def scale(self, scale_factor):
        """Scales the aperture by the given factor.

        Args:
            scale_factor (float): The factor by which to scale the aperture.

        """
        self.width = self.width * scale_factor
        self.offset_x = self.offset_x * scale_factor
        self.offset_y = self.offset_y * scale_factor

    def to_dict(self):
        """Convert the aperture to a dictionary.

        Returns:
            dict: The dictionary representation of the aperture.

        """
        aperture_dict = super().to_dict()
        aperture_dict["width"] = self.width
        aperture_dict["num_arms"] = self.num_arms
        aperture_dict["offset_x"] = self.offset_x
        aperture_dict["offset_y"] = self.offset_y
        return aperture_dict

    @classmethod
    def from_dict(cls, data):
        """Create an aperture from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the aperture.

        Returns:
            SpiderAperture: The aperture object.

        """
        return cls(
            data["width"],
            data["num_arms"],
            data.get("offset_x", 0),
            data.get("offset_y", 0),
        )
