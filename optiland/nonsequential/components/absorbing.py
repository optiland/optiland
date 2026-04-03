"""Absorbing component for Non-Sequential Raytracing.

Terminates rays on contact (light traps, baffles, aperture stops).

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.nonsequential.components.base import BaseComponent, _get_xp
from optiland.nonsequential.materials.nsq_material import VACUUM

if TYPE_CHECKING:
    import numpy as np

    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.components.geometry.base import ComponentGeometry
    from optiland.nonsequential.materials.nsq_material import NSQMaterial
    from optiland.nonsequential.ray_bundle import NSQRayBundle


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

    def interact(
        self,
        rays: NSQRayBundle,
        t: np.ndarray,
        normals: np.ndarray,
        hit_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        """Kill all rays that hit this component (in-place).

        Args:
            rays: Ray bundle updated in-place.
            t: Hit distances [mm], shape (N,).
            normals: Surface normals in global frame, shape (N, 3).
            hit_mask: True for rays hitting this component, shape (N,).
            rng: Random number generator (unused).
        """
        xp = _get_xp(rays.x)

        # Advance to hit point before killing
        rays.x = xp.where(hit_mask, rays.x + t * rays.dx, rays.x)
        rays.y = xp.where(hit_mask, rays.y + t * rays.dy, rays.y)
        rays.z = xp.where(hit_mask, rays.z + t * rays.dz, rays.z)

        # Terminate rays
        rays.alive = rays.alive & ~hit_mask
        rays.flux = xp.where(hit_mask, xp.zeros_like(rays.flux), rays.flux)
        rays.bounce = xp.where(hit_mask, rays.bounce + 1, rays.bounce)
