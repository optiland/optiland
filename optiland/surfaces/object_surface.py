"""Object Surface

This module contains the ObjectSurface class, which represents an object
surface in an optical system.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.geometries import BaseGeometry
from optiland.materials import BaseMaterial
from optiland.surfaces.standard_surface import Surface

if TYPE_CHECKING:
    from optiland.rays import BaseRays, ParaxialRays, RealRays


class ObjectSurface(Surface):
    """Represents an object surface in an optical system.

    Args:
        geometry (Geometry): The geometry of the surface.
        material_post (Material): The material of the surface after
            interaction.
        comment (str, optional): A comment for the surface. Defaults
            to ''.

    Attributes:
        is_infinite (bool): Indicates whether the surface is infinitely
            far away.

    """

    def __init__(self, geometry, material_post, comment=""):
        super().__init__(
            geometry=geometry,
            previous_surface=None,
            material_post=material_post,
            is_stop=False,
            aperture=None,
            comment=comment,
        )

    @property
    def is_infinite(self):
        """Returns True if the surface is infinitely far away, False otherwise."""
        return be.isinf(self.geometry.cs.z)

    def set_aperture(self):
        """Sets the aperture of the surface."""

    def trace(self, rays: BaseRays) -> BaseRays:
        """Traces the given rays through the surface.

        Args:
            rays (BaseRays): The rays to be traced.

        Returns:
            BaseRays: The traced rays.

        """
        self.reset()
        rays.trace_on_surface(self)
        rays.record_on_surface(self)
        return rays

    def _trace_paraxial(self, rays: ParaxialRays) -> ParaxialRays:
        """No-op physics kernel for the object surface (paraxial).

        Args:
            rays (ParaxialRays): The paraxial rays to be traced.

        Returns:
            ParaxialRays: The rays (unchanged).

        """
        return rays

    def _trace_real(self, rays: RealRays) -> RealRays:
        """No-op physics kernel for the object surface (real rays).

        Args:
            rays (RealRays): The real rays to be traced.

        Returns:
            RealRays: The rays (unchanged).

        """
        return rays

    def _interact(self, rays):
        """Interacts the given rays with the surface.

        Args:
            rays (Rays): The rays to be interacted.

        Returns:
            RealRays: The interacted rays.

        """
        return rays

    def to_dict(self):
        """Returns a dictionary representation of the surface."""
        return {
            "type": self.__class__.__name__,
            "geometry": self.geometry.to_dict(),
            "material_post": self.material_post.to_dict(),
            "comment": self.comment,
        }

    @classmethod
    def _from_dict(cls, data):
        """Creates a surface from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the surface.

        Returns:
            Surface: The surface.

        """
        geometry = BaseGeometry.from_dict(data["geometry"])
        material_post = BaseMaterial.from_dict(data["material_post"])
        comment = data.get("comment", "")
        return cls(geometry, material_post, comment=comment)

    @property
    def material_pre(self):
        return self.material_post
