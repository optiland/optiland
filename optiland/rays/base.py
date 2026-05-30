"""Base Rays

This module contains the base class for rays defined in a 3D space.

Kramer Harrison, 2024
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import optiland.backend as be

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from optiland.surfaces.standard_surface import Surface


class BaseRays(abc.ABC):
    """Base class for rays in a 3D space.

    Attributes:
        x: x-coordinate of the ray.
        y: y-coordinate of the ray.
        z: z-coordinate of the ray.

    """

    @abc.abstractmethod
    def trace_on_surface(self, surface: Surface) -> BaseRays:
        """Dispatch tracing to the appropriate surface method.

        Args:
            surface (Surface): The surface to trace through.

        Returns:
            BaseRays: The traced rays.

        """

    @abc.abstractmethod
    def record_on_surface(self, surface: Surface) -> None:
        """Dispatch recording to the appropriate surface method.

        Args:
            surface (Surface): The surface to record onto.

        """

    def translate(self, dx: ArrayLike, dy: ArrayLike, dz: ArrayLike):
        """Shifts the rays in the x, y, and z directions.

        Args:
            dx: The amount to shift the rays in the x direction.
            dy: The amount to shift the rays in the y direction.
            dz: The amount to shift the rays in the z direction.

        """
        dx = be.array(dx)
        dy = be.array(dy)
        dz = be.array(dz)
        self.x = self.x + dx
        self.y = self.y + dy
        self.z = self.z + dz
