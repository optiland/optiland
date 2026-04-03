"""Base detector for Non-Sequential Raytracing.

Kramer Harrison, 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from optiland.nonsequential.components.base import _get_transform, _get_xp

if TYPE_CHECKING:
    import numpy as np

    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.components.geometry.base import AABB, ComponentGeometry
    from optiland.nonsequential.ray_bundle import NSQRayBundle


class BaseDetector(ABC):
    """Abstract base class for detectors in the NSQ scene.

    Detectors record ray data at a surface. They intersect rays (via their
    geometry) and accumulate hit data across simulation batches.

    Attributes:
        cs: Coordinate system defining detector position and orientation.
        geometry: Surface geometry that defines the detector area.
        name: Optional human-readable label.
    """

    def __init__(
        self,
        cs: CoordinateSystem,
        geometry: ComponentGeometry,
        name: str = "",
    ) -> None:
        """Initialize BaseDetector.

        Args:
            cs: Coordinate system.
            geometry: Surface geometry.
            name: Optional label.
        """
        self.cs = cs
        self.geometry = geometry
        self.name = name

    def intersect(
        self, rays: NSQRayBundle
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find ray intersections with this detector surface.

        Args:
            rays: Ray bundle in global coordinates.

        Returns:
            Tuple (t, normals, hit_mask) in global frame.
        """
        xp = _get_xp(rays.x)
        translation, rot = _get_transform(self.cs)

        positions_g = xp.stack([rays.x, rays.y, rays.z], axis=1)
        directions_g = xp.stack([rays.dx, rays.dy, rays.dz], axis=1)

        t_np = xp.array(translation, dtype=positions_g.dtype)
        R_np = xp.array(rot, dtype=positions_g.dtype)

        positions_l = (positions_g - t_np) @ R_np
        directions_l = directions_g @ R_np

        t_hit, normals_l, hit_mask = self.geometry.ray_intersect(
            positions_l, directions_l
        )
        t_hit = xp.where(rays.alive, t_hit, xp.full_like(t_hit, xp.inf))
        hit_mask = hit_mask & rays.alive

        normals_g = normals_l @ R_np.T
        return t_hit, normals_g, hit_mask

    @abstractmethod
    def record(self, rays: NSQRayBundle, t: np.ndarray, hit_mask: np.ndarray) -> None:
        """Accumulate ray data for rays that hit this detector.

        Args:
            rays: Current ray bundle. Positions have NOT yet been advanced
                to the hit point; use t to compute hit positions.
            t: Hit distances [mm], shape (N,).
            hit_mask: Boolean mask of rays hitting this detector, shape (N,).
        """

    @abstractmethod
    def get_result(self):
        """Return the accumulated result object.

        Returns:
            A result object (IrradianceMap, FarFieldPattern, etc.).
        """

    def reset(self) -> None:
        """Clear accumulated data for reuse in a new simulation."""

    @property
    def bounding_box(self) -> AABB:
        """AABB of this detector in global coordinates."""
        transform = _get_transform(self.cs)
        return self.geometry.bounding_box(transform)
