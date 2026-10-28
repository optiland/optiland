"""Interaction model for a thin lens

This module implements the ThinLensInteractionModel class, which handles
ray interactions with a thin lens surface.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.interactions.base import BaseInteractionModel

if TYPE_CHECKING:
    # pragma: no cover
    from optiland.coatings import BaseCoating
    from optiland.scatter import BaseBSDF
    from optiland.surfaces import Surface


class ThinLensInteractionModel(BaseInteractionModel):
    """Interaction model for a thin lens."""

    interaction_type = "thin_lens"

    def __init__(
        self,
        parent_surface: Surface | None,
        focal_length: float,
        is_reflective: bool,
        coating: BaseCoating | None = None,
        bsdf: BaseBSDF | None = None,
    ):
        super().__init__(
            parent_surface=parent_surface,
            is_reflective=is_reflective,
            coating=coating,
            bsdf=bsdf,
        )
        self.f = be.array(focal_length)

    def to_dict(self):
        """Returns a dictionary representation of the thin lens model."""
        data = super().to_dict()
        data["focal_length"] = self.f.item()
        return data

    def flip(self):
        """Flip the interaction model."""
        pass

    def interact_real_rays(self, rays):
        """Interacts the rays with the surface by either reflecting or refracting

        Note that phase is added assuming a thin lens as a phase
        transformation. A cosine correction is applied for rays propagating
        off-axis. This correction is equivalent to the ray z direction cosine.

        Args:
            rays: The rays.

        Returns:
            RealRays: The refracted rays.

        """
        # A zero-power lens adds no phase; inf - inf would destroy the OPD.
        if not be.isinf(self.f):
            h2 = rays.x**2 + rays.y**2
            rays.opd = rays.opd + self.f - be.copysign(be.sqrt(h2 + self.f**2), self.f)

        n1 = self.material_pre.n(rays.w)
        n2 = n1 if self.is_reflective else self.material_post.n(rays.w)
        # Capture this surface's incident direction, not a previous bend's.
        rays.normalize()
        rays.L0, rays.M0, rays.N0 = (
            be.copy(component) for component in (rays.L, rays.M, rays.N)
        )
        L, M, N = [component / be.abs(rays.N) for component in (rays.L, rays.M, rays.N)]
        if not be.isinf(self.f):
            if self.is_reflective:
                f1 = f2 = -self.f * be.copysign(be.ones_like(rays.N), rays.N)
            else:
                f = self.f * be.copysign(be.ones_like(rays.N), rays.N)
                f1 = f * n1
                f2 = f * n2
            L = L * f1 - rays.x
            M = M * f1 - rays.y
            N = be.where(rays.N > 0, f2, -f2)
            if self.f < 0:
                L = -L
                M = -M
                N = -N

        else:
            N *= n2 / n1

        if self.is_reflective:
            N = -N
        rays.L = L
        rays.M = M
        rays.N = N

        rays.normalize()
        # Transport the field once, from the incident to the outgoing direction.
        # Scatter must also act on the outgoing ray rather than be overwritten.
        return self._apply_coating_and_bsdf(rays, nx=0, ny=0, nz=1)

    def interact_paraxial_rays(self, rays):
        """Traces paraxial rays through the surface.

        Args:
            ParaxialRays: The paraxial rays to be traced.

        """
        n1 = self.material_pre.n(rays.w)
        if self.is_reflective:
            # reflect (derived from paraxial equations when n'=-n)
            rays.u = rays.y / (self.f * n1) - rays.u

        else:
            # surface power
            n2 = self.material_post.n(rays.w)

            # refract
            rays.u = 1 / n2 * (n1 * rays.u - rays.y / self.f)

        return rays
