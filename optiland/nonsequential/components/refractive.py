"""Refractive component for Non-Sequential Raytracing.

Transmits and reflects (lenses, prisms, windows). Uses detached-sample /
attached-weight Fresnel splitting for differentiable Monte Carlo.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.backend.utils import to_numpy
from optiland.nonsequential.components.base import BaseComponent

if TYPE_CHECKING:
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.bsdf.base import BaseBSDF
    from optiland.nonsequential.components.geometry.base import ComponentGeometry
    from optiland.nonsequential.materials.nsq_material import NSQMaterial
    from optiland.nonsequential.ray_bundle import NSQRayBundle


class RefractiveComponent(BaseComponent):
    """Refractive optical element (lens, prism, window).

    At each interface, Fresnel splitting uses the detached-sample /
    attached-weight scheme: the branch decision (reflect vs transmit) is
    drawn from a detached probability, while the throughput weight carries
    the attached reflectance so gradients flow through material parameters.

    Attributes:
        cs: Coordinate system.
        geometry: Surface geometry.
        material_front: Medium on the front (normal-facing) side.
        material_back: Medium on the back side.
        bsdf: Optional BSDF for scatter. None = specular.
        name: Optional label.
    """

    def __init__(
        self,
        cs: CoordinateSystem,
        geometry: ComponentGeometry,
        material_front: NSQMaterial,
        material_back: NSQMaterial,
        bsdf: BaseBSDF | None = None,
        name: str = "",
        scatter_fraction: float = 1.0,
    ) -> None:
        """Initialize RefractiveComponent.

        Args:
            cs: Coordinate system.
            geometry: Surface geometry.
            material_front: Front-side medium.
            material_back: Back-side medium.
            bsdf: Optional BSDF scatter model.
            name: Optional label.
            scatter_fraction: Probability that a hit ray is routed through
                ``bsdf`` rather than refracted.
        """
        super().__init__(
            cs,
            geometry,
            material_front,
            material_back,
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
        """Apply Fresnel refraction/reflection at hit points (in-place).

        Uses detached-sample / attached-weight Fresnel: the reflect/transmit
        branch decision is drawn from a detached probability so stochastic
        choices do not block gradients; the throughput weight multiplier
        carries the attached reflectance so ∂flux/∂R is non-zero.

        Args:
            rays: Ray bundle updated in-place.
            t: Hit distances [mm], shape (N,).
            normals: Surface normals in global frame, shape (N, 3).
            hit_mask: True for rays hitting this component, shape (N,).
            rng: Random number generator (used for detached sampling only).
        """
        # Missed rays carry t = inf; zero it for the differentiable position
        # update so the masked-out be.where branch cannot inject a
        # 0 * inf = NaN into the backward pass.
        t = be.where(hit_mask, t, be.zeros_like(t))

        # Advance hit rays to intersection point
        rays.x = be.where(hit_mask, rays.x + t * rays.L, rays.x)
        rays.y = be.where(hit_mask, rays.y + t * rays.M, rays.y)
        rays.z = be.where(hit_mask, rays.z + t * rays.N, rays.z)

        dirs = be.stack([rays.L, rays.M, rays.N], axis=1)
        wl = rays.wavelength  # µm

        # Determine n1 and n2 for each ray (based on side of the surface)
        dot = (dirs * normals).sum(axis=1)  # signed cos_theta
        cos_theta_i = be.abs(dot)

        n1 = rays.n_current  # shape (N,)

        # Evaluate n2 at each wavelength -- attached (differentiable)
        n2_front = self.material_front.n(wl)
        n2_back = self.material_back.n(wl)

        # dot < 0 -> ray opposes normal -> came from front side
        from_front = dot < 0
        n2 = be.where(from_front, n2_back, n2_front)

        # Fresnel reflectance (unpolarized, attached)
        n_ratio = n1 / (n2 + 1e-30)
        sin2_t = n_ratio**2 * (1.0 - cos_theta_i**2)
        tir = sin2_t > 1.0
        # Epsilon-clamp the radicand (not 0): sqrt's infinite derivative at 0
        # combined with be.where yields a 0 * inf = NaN gradient at the TIR
        # boundary. Forward value changes by <= 1e-6.
        cos_theta_t = be.where(
            tir,
            be.zeros_like(sin2_t),
            be.maximum(1.0 - sin2_t, 1e-12) ** 0.5,
        )

        rs = (n1 * cos_theta_i - n2 * cos_theta_t) / (
            n1 * cos_theta_i + n2 * cos_theta_t + 1e-30
        )
        rp = (n2 * cos_theta_i - n1 * cos_theta_t) / (
            n2 * cos_theta_i + n1 * cos_theta_t + 1e-30
        )
        R_fresnel = be.where(tir, be.ones_like(rs), 0.5 * (rs**2 + rp**2))

        # --- Detached-sample / attached-weight ---
        # Sampling decision uses detached R so the branch doesn't block grad.
        R_np = to_numpy(R_fresnel).astype(np.float64)
        u = rng.random(rays.num_rays)
        do_reflect_np = (u < R_np) | to_numpy(tir).astype(bool)
        do_reflect = be.array(do_reflect_np)

        # Throughput weight: forward value is 1.0; carries ∂R.
        # For TIR rays weight stays 1.0 (full reflection is deterministic).
        R_det = be.array(R_np)  # detached copy used as denominator
        weight_reflect = R_fresnel / (R_det + 1e-30)
        weight_transmit = (1.0 - R_fresnel) / (1.0 - R_det + 1e-30)
        weight = be.where(do_reflect, weight_reflect, weight_transmit)
        # TIR: weight is exactly 1
        weight = be.where(tir, be.ones_like(weight), weight)

        # Apply weight to flux for hit rays
        rays.flux = rays.flux * be.where(hit_mask, weight, be.ones_like(weight))

        # Compute reflected direction: d - 2*(d.n)*n
        raw_dot = (dirs * normals).sum(axis=1, keepdims=True)
        reflected = dirs - 2.0 * raw_dot * normals
        norms_r = (reflected * reflected).sum(axis=1, keepdims=True) ** 0.5
        reflected = reflected / (norms_r + 1e-30)

        # Compute refracted direction (Snell's law, vector form)
        n_facing = be.where(raw_dot < 0, normals, -normals)
        cos_i_pos = be.abs(raw_dot)
        n_ratio_col = n_ratio[:, None]
        cos_t_col = cos_theta_t[:, None]
        refracted = (
            n_ratio_col * dirs + (n_ratio_col * cos_i_pos - cos_t_col) * n_facing
        )
        norms_t = (refracted * refracted).sum(axis=1, keepdims=True) ** 0.5
        refracted = refracted / (norms_t + 1e-30)

        # Select direction based on branch decision
        do_reflect_col = do_reflect[:, None]
        new_d = be.where(do_reflect_col, reflected, refracted)

        # Apply only to hit rays
        hit_col = hit_mask[:, None]
        rays.L = be.where(hit_col[:, 0], new_d[:, 0], rays.L)
        rays.M = be.where(hit_col[:, 0], new_d[:, 1], rays.M)
        rays.N = be.where(hit_col[:, 0], new_d[:, 2], rays.N)

        # Update n_current: stays n1 on reflect, becomes n2 on refract
        rays.n_current = be.where(
            hit_mask, be.where(do_reflect, n1, n2), rays.n_current
        )

        # Update bounce count
        rays.bounce = be.where(hit_mask, rays.bounce + 1, rays.bounce)

        # Apply BSDF scatter if present (compute for all rays, use where to select)
        if self.bsdf is not None:
            # Compute BSDF for all N rays; where-select only hit rays
            bsdf_dirs, bsdf_weights = self.bsdf.sample(
                rays.num_rays,
                be.stack([rays.L, rays.M, rays.N], axis=1),
                normals,
                rays.wavelength,
                rng,
            )
            # Route only a scatter_fraction of the hit rays through the BSDF;
            # the rest keep the refracted direction computed above. The branch
            # is drawn from a detached probability, matching the Fresnel
            # split, so the surface can be partially diffusing.
            scatters = hit_mask & be.array(
                rng.random(rays.num_rays) < self.scatter_fraction
            )
            scatter_col = scatters[:, None]
            cur_dirs = be.stack([rays.L, rays.M, rays.N], axis=1)
            new_dirs = be.where(scatter_col, bsdf_dirs, cur_dirs)
            rays.L = new_dirs[:, 0]
            rays.M = new_dirs[:, 1]
            rays.N = new_dirs[:, 2]
            bsdf_gate = be.where(scatters, bsdf_weights, be.ones_like(bsdf_weights))
            rays.flux = rays.flux * bsdf_gate
