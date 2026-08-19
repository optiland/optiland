"""Interaction model for standard refraction and reflection

This module implements the RefractiveReflectiveModel class, which handles
ray interactions with surfaces through refraction and reflection.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.interactions.base import BaseInteractionModel

if TYPE_CHECKING:
    from optiland.rays import ParaxialRays, RealRays


class RefractiveReflectiveModel(BaseInteractionModel):
    """Interaction model for standard refraction and reflection."""

    interaction_type = "refractive_reflective"

    def to_dict(self):
        """Returns a dictionary representation of the model."""
        return super().to_dict()

    def flip(self):
        """Flip the interaction model."""
        pass

    def interact_real_rays(self, rays: RealRays) -> RealRays:
        """Interact with real rays, causing refraction or reflection.

        Args:
            rays (RealRays): The incoming real rays.

        Returns:
            RealRays: The outgoing real rays.
        """
        # find surface normals
        nx, ny, nz = self.geometry.surface_normal(rays)

        # Interact with surface (refract or reflect)
        if self.is_reflective:
            rays.reflect(nx, ny, nz)
        else:
            n1 = self.material_pre.n(rays.w)
            n2 = self.material_post.n(rays.w)
            rays.refract(nx, ny, nz, n1, n2)

        # Apply coating and BSDF
        rays = self._apply_coating_and_bsdf(rays, nx, ny, nz)

        return rays

    def interact_paraxial_rays(self, rays: ParaxialRays) -> ParaxialRays:
        """Interact with paraxial rays, causing refraction or reflection.

        Args:
            rays (ParaxialRays): The incoming paraxial rays.

        Returns:
            ParaxialRays: The outgoing paraxial rays.
        """
        if self.is_reflective:
            # reflect (derived from paraxial equations when n'=-n)
            rays.u = -rays.u - 2 * rays.y / self.geometry.radius
        else:
            # surface power
            n1 = self.material_pre.n(rays.w)
            n2 = self.material_post.n(rays.w)
            power = (n2 - n1) / self.geometry.radius

            # refract
            rays.u = 1 / n2 * (n1 * rays.u - rays.y * power)

        return rays
