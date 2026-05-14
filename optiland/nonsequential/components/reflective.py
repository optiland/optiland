"""Reflective component for Non-Sequential Raytracing.

Mirrors and baffles with coating. Specular reflection (or BSDF scatter).

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optiland.nonsequential._utils import to_numpy
from optiland.nonsequential.components.base import BaseComponent, _get_xp
from optiland.nonsequential.materials.nsq_material import VACUUM

if TYPE_CHECKING:
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
    ) -> None:
        """Initialize ReflectiveComponent.

        Args:
            cs: Coordinate system.
            geometry: Surface geometry.
            bsdf: Optional BSDF scatter model. None = perfect mirror.
            material_front: Medium on the front side (default: vacuum).
            name: Optional label.
        """
        super().__init__(cs, geometry, material_front, material_front, bsdf, name)

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
        xp = _get_xp(rays.x)

        # Advance to hit point
        rays.x = xp.where(hit_mask, rays.x + t * rays.L, rays.x)
        rays.y = xp.where(hit_mask, rays.y + t * rays.M, rays.y)
        rays.z = xp.where(hit_mask, rays.z + t * rays.N, rays.z)

        dirs = xp.stack([rays.L, rays.M, rays.N], axis=1)

        if self.bsdf is not None:
            # BSDF scatter
            hit_np = np.where(to_numpy(hit_mask))[0]
            if len(hit_np) > 0:
                dirs_hit = dirs[hit_np]
                normals_hit = normals[hit_np]
                wl_hit = rays.wavelength[hit_np]
                scattered, weights = self.bsdf.sample(
                    len(hit_np), dirs_hit, normals_hit, wl_hit, rng
                )
                all_d = dirs.copy() if xp is np else xp.array(dirs)
                all_d[hit_np] = scattered
                flux_arr = xp.array(rays.flux)
                flux_arr[hit_np] *= weights
                rays.flux = flux_arr
            else:
                all_d = dirs
        else:
            # Specular reflection: d - 2*(d.n)*n
            raw_dot = (dirs * normals).sum(axis=1, keepdims=True)
            reflected = dirs - 2.0 * raw_dot * normals
            norms_r = (reflected * reflected).sum(axis=1, keepdims=True) ** 0.5
            reflected = reflected / (norms_r + 1e-30)
            hit_col = hit_mask[:, None]
            all_d = xp.where(hit_col, reflected, dirs)

        rays.L = all_d[:, 0]
        rays.M = all_d[:, 1]
        rays.N = all_d[:, 2]

        # n_current unchanged (reflection stays in same medium)
        rays.bounce = xp.where(hit_mask, rays.bounce + 1, rays.bounce)
