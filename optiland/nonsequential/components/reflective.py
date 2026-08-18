"""Reflective component for Non-Sequential Raytracing.

Mirrors and baffles with coating. Specular reflection (or BSDF scatter).

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.nonsequential.components.base import BaseComponent
from optiland.nonsequential.materials.nsq_material import VACUUM

if TYPE_CHECKING:
    import numpy as np

    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.bsdf.base import BaseBSDF
    from optiland.nonsequential.components.geometry.base import ComponentGeometry
    from optiland.nonsequential.materials.nsq_material import NSQMaterial
    from optiland.nonsequential.ray_bundle import NSQRayBundle


class ReflectiveComponent(BaseComponent):
    """Purely reflective optical element (mirror, baffle).

    Reflects rays specularly (or via BSDF). Does not transmit.

    Attributes:
        cs: Coordinate system.
        geometry: Surface geometry.
        bsdf: Optional BSDF for scatter. None = specular mirror.
        name: Optional label.
    """

    def __init__(
        self,
        cs: CoordinateSystem,
        geometry: ComponentGeometry,
        bsdf: BaseBSDF | None = None,
        material_front: NSQMaterial = VACUUM,
        name: str = "",
        scatter_fraction: float = 1.0,
    ) -> None:
        """Initialize ReflectiveComponent.

        Args:
            cs: Coordinate system.
            geometry: Surface geometry.
            bsdf: Optional BSDF scatter model. None = perfect mirror.
            material_front: Medium on the front side (default: vacuum).
            name: Optional label.
            scatter_fraction: Probability that a hit ray is routed through
                ``bsdf`` rather than specularly reflected.
        """
        super().__init__(
            cs,
            geometry,
            material_front,
            material_front,
            bsdf,
            name,
            scatter_fraction=scatter_fraction,
        )

    def interact(
        self,
        rays: NSQRayBundle,
        t: np.ndarray,
        normals: np.ndarray,
        hit_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        """Apply specular (or BSDF) reflection at hit points (in-place).

        Args:
            rays: Ray bundle updated in-place.
            t: Hit distances [mm], shape (N,).
            normals: Surface normals in global frame, shape (N, 3).
            hit_mask: True for rays hitting this component, shape (N,).
            rng: Random number generator.
        """
        # Missed rays carry t = inf; zero it before the position update so the
        # masked-out be.where branch cannot backpropagate 0 * inf = NaN into
        # the ray directions.
        t = be.where(hit_mask, t, be.zeros_like(t))

        # Advance to hit point
        rays.x = be.where(hit_mask, rays.x + t * rays.L, rays.x)
        rays.y = be.where(hit_mask, rays.y + t * rays.M, rays.y)
        rays.z = be.where(hit_mask, rays.z + t * rays.N, rays.z)

        dirs = be.stack([rays.L, rays.M, rays.N], axis=1)

        # Specular reflection: d - 2*(d.n)*n. Always computed, because with a
        # scatter_fraction below 1 it is the fallback for rays that do not
        # enter the BSDF lobe.
        raw_dot = (dirs * normals).sum(axis=1, keepdims=True)
        reflected = dirs - 2.0 * raw_dot * normals
        norms_r = (reflected * reflected).sum(axis=1, keepdims=True) ** 0.5
        reflected = reflected / (norms_r + 1e-30)
        hit_col = hit_mask[:, None]
        new_dirs = be.where(hit_col, reflected, dirs)

        if self.bsdf is not None:
            # Compute BSDF for all N rays; where-select only scattering rays
            bsdf_dirs, bsdf_weights = self.bsdf.sample(
                rays.num_rays,
                dirs,
                normals,
                rays.wavelength,
                rng,
            )
            # Route only a scatter_fraction of the hit rays into the lobe; the
            # rest reflect specularly. The branch is drawn from a detached
            # probability, matching the Fresnel split.
            scatters = hit_mask & be.array(
                rng.random(rays.num_rays) < self.scatter_fraction
            )
            new_dirs = be.where(scatters[:, None], bsdf_dirs, new_dirs)
            rays.flux = rays.flux * be.where(
                scatters, bsdf_weights, be.ones_like(bsdf_weights)
            )

        rays.L = new_dirs[:, 0]
        rays.M = new_dirs[:, 1]
        rays.N = new_dirs[:, 2]

        # n_current unchanged (reflection stays in same medium)
        rays.bounce = be.where(hit_mask, rays.bounce + 1, rays.bounce)
