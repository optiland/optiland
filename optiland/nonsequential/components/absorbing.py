"""Absorbing component for Non-Sequential Raytracing.

Terminates rays on contact (light traps, baffles, aperture stops).

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
    from optiland.nonsequential.components.geometry.base import ComponentGeometry
    from optiland.nonsequential.ir.bsdf_ir import BsdfIR
    from optiland.nonsequential.ir.scene_ir import SamplingPolicy
    from optiland.nonsequential.materials.nsq_material import NSQMaterial
    from optiland.nonsequential.ray_bundle import NSQRayBundle
    from optiland.nonsequential.rng import NSQRng


class AbsorbingComponent(BaseComponent):
    """Absorbing surface that terminates all rays on contact.

    Use for light traps, aperture stops, and baffles.

    Attributes:
        cs: Coordinate system.
        geometry: Surface geometry.
        name: Optional label.
    """

    def __init__(
        self,
        cs: CoordinateSystem,
        geometry: ComponentGeometry,
        material_front: NSQMaterial = VACUUM,
        name: str = "",
    ) -> None:
        """Initialize AbsorbingComponent.

        Args:
            cs: Coordinate system.
            geometry: Surface geometry.
            material_front: Surrounding medium (default: vacuum).
            name: Optional label.
        """
        super().__init__(cs, geometry, material_front, material_front, None, name)
        self._absorbed_count: int = 0
        self._absorbed_flux: float = 0.0

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
        """Kill all rays that hit this component (in-place).

        Args:
            rays: Ray bundle updated in-place.
            t: Hit distances [mm], shape (N,).
            normals: Surface normals in global frame, shape (N, 3).
            hit_mask: True for rays hitting this component, shape (N,).
            rng: Keyed PCG32 RNG (unused).
            bsdf_ir: Unused -- absorbing surfaces never scatter.
            n_geom: Unused -- absorbing surfaces never determine sidedness.
            sampling: Unused -- an absorber has no stochastic branch (D2,
                PR11).
            forced_branch: Unused -- bounded splitting only applies
                to ``RefractiveComponent``.
        """
        # Missed rays carry t = inf; zero it before the position update so the
        # masked-out be.where branch cannot backpropagate 0 * inf = NaN into
        # the ray directions.
        t = be.where(hit_mask, t, be.zeros_like(t))

        # Advance to hit point before killing
        rays.x = be.where(hit_mask, rays.x + t * rays.L, rays.x)
        rays.y = be.where(hit_mask, rays.y + t * rays.M, rays.y)
        rays.z = be.where(hit_mask, rays.z + t * rays.N, rays.z)

        # Count absorbed rays (must be alive when they hit)
        hit_alive = hit_mask & rays.alive
        self._absorbed_count += int(hit_alive.sum())
        self._absorbed_flux += float(rays.flux[hit_alive].sum())

        # Terminate rays
        rays.alive = rays.alive & ~hit_mask
        rays.flux = be.where(hit_mask, be.zeros_like(rays.flux), rays.flux)
        rays.bounce = be.where(hit_mask, rays.bounce + 1, rays.bounce)

    def reset_stats(self) -> None:
        """Reset per-simulation absorbed ray and flux counters."""
        self._absorbed_count = 0
        self._absorbed_flux = 0.0
