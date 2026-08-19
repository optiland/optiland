"""Image Surface

This module contains the ImageSurface class, which represents an image surface
in an optical system.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.surfaces.standard_surface import Surface

if TYPE_CHECKING:
    from optiland.geometries import BaseGeometry
    from optiland.materials import BaseMaterial
    from optiland.physical_apertures import BaseAperture
    from optiland.rays import ParaxialRays


class ImageSurface(Surface):
    """Represents an image surface in an optical system.

    Args:
        geometry (BaseGeometry): The geometry of the surface.
        material_pre (BaseMaterial): The material before the surface.
        aperture (BaseAperture, optional): The aperture of the surface.
            Defaults to None.

    """

    def __init__(
        self,
        previous_surface: Surface | None,
        geometry: BaseGeometry,
        material_post: BaseMaterial,
        aperture: BaseAperture = None,
    ):
        super().__init__(
            previous_surface=previous_surface,
            geometry=geometry,
            material_post=material_post,
            is_stop=False,
            aperture=aperture,
        )

    def _trace_paraxial(self, rays: ParaxialRays) -> ParaxialRays:
        """Paraxial physics kernel for the image surface.

        Args:
            rays (ParaxialRays): The paraxial rays to be traced.

        Returns:
            ParaxialRays: The traced paraxial rays.

        """
        t = -rays.z
        rays.propagate(t)
        return rays

    def _interact(self, rays):
        """Interacts rays with the surface.

        Args:
            rays: The rays to be interacted with the surface.

        Returns:
            RealRays: The modified rays after interaction with the surface.

        """
        return rays
