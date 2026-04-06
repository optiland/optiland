"""Refractive component for Non-Sequential Raytracing.

Transmits and reflects (lenses, prisms, windows). Uses Russian-roulette
Fresnel splitting.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optiland.nonsequential.components.base import BaseComponent, _get_xp

if TYPE_CHECKING:
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.bsdf.base import BaseBSDF
    from optiland.nonsequential.components.geometry.base import ComponentGeometry
    from optiland.nonsequential.materials.nsq_material import NSQMaterial
    from optiland.nonsequential.ray_bundle import NSQRayBundle


class RefractiveComponent(BaseComponent):
    """Refractive optical element (lens, prism, window).

    At each interface, Russian-roulette Fresnel splitting decides whether
    the ray reflects or refracts. TIR is handled as full reflection.

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
    ) -> None:
        """Initialize RefractiveComponent.

        Args:
            cs: Coordinate system.
            geometry: Surface geometry.
            material_front: Front-side medium.
            material_back: Back-side medium.
            bsdf: Optional BSDF scatter model.
            name: Optional label.
        """
        super().__init__(cs, geometry, material_front, material_back, bsdf, name)

    def interact(
        self,
        rays: NSQRayBundle,
        t: np.ndarray,
        normals: np.ndarray,
        hit_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        """Apply Fresnel refraction/reflection at hit points (in-place).

        For each hit ray:
        1. Advance position to hit point.
        2. Determine n1 (current medium) and n2 (next medium).
        3. Compute Fresnel reflectance R.
        4. Russian roulette: reflect if u < R, else refract.
        5. Handle TIR as full reflection.
        6. Update n_current, bounce, flux.

        Args:
            rays: Ray bundle updated in-place.
            t: Hit distances [mm], shape (N,).
            normals: Surface normals in global frame, shape (N, 3).
            hit_mask: True for rays hitting this component, shape (N,).
            rng: Random number generator.
        """
        xp = _get_xp(rays.x)

        # Advance hit rays to intersection point
        rays.x = xp.where(hit_mask, rays.x + t * rays.dx, rays.x)
        rays.y = xp.where(hit_mask, rays.y + t * rays.dy, rays.y)
        rays.z = xp.where(hit_mask, rays.z + t * rays.dz, rays.z)

        dirs = xp.stack([rays.dx, rays.dy, rays.dz], axis=1)
        wl = rays.wavelength  # nm

        # Determine n1 and n2 for each ray (based on side of the surface)
        dot = (dirs * normals).sum(axis=1)  # cos_theta_i (signed)
        # Normal points toward incoming ray (guaranteed by geometry), so dot <= 0
        # If dot > 0, the normal was not flipped correctly. Handle both cases.
        cos_theta_i = xp.abs(dot)

        n1 = rays.n_current  # shape (N,)

        # Evaluate n2 at each wavelength
        n2_front = self.material_front.n(wl)
        n2_back = self.material_back.n(wl)

        # Determine which side the ray is coming from
        # dot < 0 -> ray opposes normal -> came from front
        from_front = dot < 0
        if xp is not np:
            n2 = xp.where(from_front, n2_back, n2_front)
        else:
            n2 = np.where(from_front, n2_back, n2_front)

        # Fresnel reflectance (unpolarized)
        n_ratio = n1 / (n2 + 1e-30)
        sin2_t = n_ratio**2 * (1.0 - cos_theta_i**2)
        tir = sin2_t > 1.0
        cos_theta_t = xp.where(
            tir, xp.zeros_like(sin2_t), (xp.maximum(1.0 - sin2_t, 0.0)) ** 0.5
        )

        rs = (n1 * cos_theta_i - n2 * cos_theta_t) / (
            n1 * cos_theta_i + n2 * cos_theta_t + 1e-30
        )
        rp = (n2 * cos_theta_i - n1 * cos_theta_t) / (
            n2 * cos_theta_i + n1 * cos_theta_t + 1e-30
        )
        R_fresnel = xp.where(tir, xp.ones_like(rs), 0.5 * (rs**2 + rp**2))

        # Russian roulette
        u_np = rng.random(rays.num_rays)
        if xp is not np:
            u = xp.array(u_np, dtype=R_fresnel.dtype)
        else:
            u = u_np.astype(R_fresnel.dtype)

        do_reflect = (u < R_fresnel) | tir

        # Compute reflected direction: d - 2*(d.n)*n
        # Note: dot = cos_theta_i with sign; use actual dot product
        raw_dot = (dirs * normals).sum(axis=1, keepdims=True)
        reflected = dirs - 2.0 * raw_dot * normals
        norms_r = (reflected * reflected).sum(axis=1, keepdims=True) ** 0.5
        reflected = reflected / (norms_r + 1e-30)

        # Compute refracted direction (Snell's law, vector form)
        # d_refracted = n_ratio * d + (n_ratio * cos_i - cos_t) * n_facing
        # n_facing = -sign(dot)*normal (faces incoming ray)
        n_facing = xp.where(raw_dot < 0, normals, -normals)
        cos_i_pos = xp.abs(raw_dot)  # (N, 1)
        n_ratio_col = n_ratio[:, None]
        cos_t_col = cos_theta_t[:, None]
        refracted = (
            n_ratio_col * dirs + (n_ratio_col * cos_i_pos - cos_t_col) * n_facing
        )
        norms_t = (refracted * refracted).sum(axis=1, keepdims=True) ** 0.5
        refracted = refracted / (norms_t + 1e-30)

        # Update directions
        do_reflect_col = do_reflect[:, None]
        new_d = xp.where(do_reflect_col, reflected, refracted)

        # Apply only to hit rays
        hit_col = hit_mask[:, None]
        rays.dx = xp.where(hit_col[:, 0], new_d[:, 0], rays.dx)
        rays.dy = xp.where(hit_col[:, 0], new_d[:, 1], rays.dy)
        rays.dz = xp.where(hit_col[:, 0], new_d[:, 2], rays.dz)

        # Update n_current: stays n1 on reflect, becomes n2 on refract
        rays.n_current = xp.where(
            hit_mask, xp.where(do_reflect, n1, n2), rays.n_current
        )

        # Update bounce count
        rays.bounce = xp.where(hit_mask, rays.bounce + 1, rays.bounce)

        # Apply BSDF scatter if present
        if self.bsdf is not None:
            hit_indices = np.where(_to_numpy(xp, hit_mask))[0]
            if len(hit_indices) > 0:
                dirs_hit = xp.stack([rays.dx, rays.dy, rays.dz], axis=1)[hit_indices]
                normals_hit = normals[hit_indices]
                wl_hit = rays.wavelength[hit_indices]
                scattered, weights = self.bsdf.sample(
                    len(hit_indices), dirs_hit, normals_hit, wl_hit, rng
                )
                # Update directions for hit rays
                all_dx = xp.array(rays.dx)
                all_dy = xp.array(rays.dy)
                all_dz = xp.array(rays.dz)
                all_dx[hit_indices] = scattered[:, 0]
                all_dy[hit_indices] = scattered[:, 1]
                all_dz[hit_indices] = scattered[:, 2]
                rays.dx = all_dx
                rays.dy = all_dy
                rays.dz = all_dz
                flux_arr = xp.array(rays.flux)
                flux_arr[hit_indices] *= weights
                rays.flux = flux_arr


def _to_numpy(xp, arr):
    if xp is not np:
        return xp.asnumpy(arr)
    return arr
