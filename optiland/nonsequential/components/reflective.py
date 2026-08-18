"""Reflective component for Non-Sequential Raytracing.

Mirrors and baffles with coating. Specular reflection (or BSDF scatter).

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.backend.utils import to_numpy
from optiland.nonsequential.components.base import BaseComponent
from optiland.nonsequential.components.coating_support import (
    reject_polarized_coating,
    resolve_reflectance,
)
from optiland.nonsequential.materials.nsq_material import VACUUM
from optiland.nonsequential.rng import EventSlot

if TYPE_CHECKING:
    from collections.abc import Callable

    from optiland.coatings import BaseCoating
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.bsdf.base import BaseBSDF
    from optiland.nonsequential.components.geometry.base import ComponentGeometry
    from optiland.nonsequential.ir.bsdf_ir import BsdfIR
    from optiland.nonsequential.ir.scene_ir import SamplingPolicy
    from optiland.nonsequential.materials.nsq_material import NSQMaterial
    from optiland.nonsequential.ray_bundle import NSQRayBundle
    from optiland.nonsequential.rng import NSQRng


class ReflectiveComponent(BaseComponent):
    """Purely reflective optical element (mirror, baffle).

    Reflects rays specularly (or via BSDF). Does not transmit.

    Attributes:
        cs: Coordinate system.
        geometry: Surface geometry.
        reflectance: Constant, callable(wavelength_um), or unpolarized
            BaseCoating giving the fraction of flux reflected.
        bsdf: Optional BSDF for scatter. None = specular mirror.
        name: Optional label.
    """

    def __init__(
        self,
        cs: CoordinateSystem,
        geometry: ComponentGeometry,
        reflectance: float | Callable[[be.ndarray], be.ndarray] | BaseCoating,
        bsdf: BaseBSDF | None = None,
        material_front: NSQMaterial = VACUUM,
        name: str = "",
        scatter_fraction: float = 1.0,
    ) -> None:
        """Initialize ReflectiveComponent.

        Args:
            cs: Coordinate system.
            geometry: Surface geometry.
            reflectance: Fraction of incident flux reflected. Required --
                unlike ``bsdf``, there is no "perfect mirror" default:
                a mirror built without saying how reflective it is would
                otherwise silently reflect 100% of incoming flux. Accepts a
                constant in [0, 1], a ``callable(wavelength_um) ->
                reflectance``, or an unpolarized ``optiland.coatings
                .BaseCoating`` (e.g. ``SimpleCoating``); a
                ``BaseCoatingPolarized`` instance raises
                ``NotImplementedError`` at construction.
            bsdf: Optional BSDF scatter model. None = perfect mirror.
            material_front: Medium on the front side (default: vacuum).
            name: Optional label.
            scatter_fraction: Probability that a hit ray is routed through
                ``bsdf`` rather than specularly reflected.
        """
        reject_polarized_coating(reflectance, surface_name=name)
        self.reflectance = reflectance
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
        rng: NSQRng,
        bsdf_ir: BsdfIR,
        n_geom: np.ndarray,
        sampling: SamplingPolicy | None = None,
        forced_branch: str | None = None,
    ) -> None:
        """Apply specular (or BSDF) reflection at hit points (in-place).

        Args:
            rays: Ray bundle updated in-place.
            t: Hit distances [mm], shape (N,).
            normals: Surface normals in global frame, shape (N, 3).
            hit_mask: True for rays hitting this component, shape (N,).
            rng: Keyed PCG32 RNG. Draws are keyed by this ray's own id and
                its bounce count as of this interaction.
            bsdf_ir: This surface's lowered BSDF descriptor. Whether the
                scatter branch below runs at all is decided from
                ``bsdf_ir.kind != "none"`` (verified by the caller to match
                ``self.bsdf``), not from ``self.bsdf is not None``.
            n_geom: Unused -- a mirror never transmits, so it never needs
                to determine which medium a ray is entering.
            sampling: Unused -- a mirror has no Fresnel reflect/transmit
                branch to importance-bias; its reflectance is
                applied as a deterministic flux weight, not a stochastic
                draw.
            forced_branch: Unused -- bounded splitting only applies
                to ``RefractiveComponent``'s Fresnel branch.
        """
        # Captured before any mutation below (including the bounce
        # increment at the end of this method) changes rays.bounce.
        ray_id_key = to_numpy(rays.ray_id)
        bounce_key = to_numpy(rays.bounce)

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

        # D-3: reflectance is mandatory, applied to every hit ray regardless
        # of whether it also scatters through a BSDF below.
        R = resolve_reflectance(self.reflectance, rays.wavelength)
        rays.flux = rays.flux * be.where(hit_mask, R, be.ones_like(R))

        if bsdf_ir.kind != "none":
            # Compute BSDF for all N rays; where-select only scattering rays.
            # A mirror has no far side to transmit into (material_front ==
            # material_back), so the lobe's reflect/transmit side is unused
            # here -- unlike RefractiveComponent, there is no second medium
            # for a "transmitted" scattered ray to have entered.
            bsdf_dirs, bsdf_weights, _bsdf_transmitted = self.bsdf.sample(
                rays.num_rays,
                dirs,
                normals,
                rays.wavelength,
                rng,
                ray_id_key,
                bounce_key,
            )
            # Route only a scatter_fraction of the hit rays into the lobe; the
            # rest reflect specularly. The branch is drawn from a detached
            # probability, matching the Fresnel split -- and, like it,
            # carries a compensating attached weight so
            # d(flux)/d(scatter_fraction) is correct rather than silently
            # zero. See RefractiveComponent.interact for the epsilon-clamp
            # rationale.
            sf_det = float(np.clip(to_numpy(self.scatter_fraction), 1e-6, 1.0 - 1e-6))
            u_scatter = rng.uniform(ray_id_key, bounce_key, EventSlot.SCATTER_BRANCH)
            scatters_np = to_numpy(hit_mask).astype(bool) & (u_scatter < sf_det)
            scatters = be.array(scatters_np)

            sf = self.scatter_fraction
            weight_scatter_branch = sf / sf_det
            weight_nonscatter_branch = (1.0 - sf) / (1.0 - sf_det)
            sf_gate = be.where(
                scatters, weight_scatter_branch, weight_nonscatter_branch
            )
            rays.flux = rays.flux * be.where(hit_mask, sf_gate, be.ones_like(sf_gate))

            new_dirs = be.where(scatters[:, None], bsdf_dirs, new_dirs)
            rays.flux = rays.flux * be.where(
                scatters, bsdf_weights, be.ones_like(bsdf_weights)
            )

        rays.L = new_dirs[:, 0]
        rays.M = new_dirs[:, 1]
        rays.N = new_dirs[:, 2]

        # n_current unchanged (reflection stays in same medium)
        rays.bounce = be.where(hit_mask, rays.bounce + 1, rays.bounce)
