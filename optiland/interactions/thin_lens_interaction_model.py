"""Interaction model for a thin lens

This module implements the ThinLensInteractionModel class, which handles
ray interactions with a thin lens surface.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.interactions.base import BaseInteractionModel
from optiland.rays.polarized_rays import PolarizedRays

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
        # add optical path length - workaround for now
        # TODO: develop more robust method
        rays.opd = rays.opd - (rays.x**2 + rays.y**2) / (2 * self.f)

        n1 = self.material_pre.n(rays.w)
        n2 = n1 if self.is_reflective else self.material_post.n(rays.w)
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

        # only normalize if required
        if self.bsdf or self.coating or isinstance(rays, PolarizedRays):
            rays.normalize()

        # if there is a surface scatter model, modify ray properties
        if self.bsdf:
            rays = self.bsdf.scatter(rays, nx=0, ny=0, nz=1)

        # if there is a coating, modify ray properties
        if self.coating:
            rays = self.coating.interact(
                rays,
                reflect=self.is_reflective,
                nx=0,
                ny=0,
                nz=1,
            )
        else:
            # update polarization matrices, if PolarizedRays
            rays.update()

        if self.is_reflective:
            N = -N
        rays.L = L
        rays.M = M
        rays.N = N

        rays.normalize()

        return rays

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
