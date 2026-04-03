"""Base component for Non-Sequential Raytracing.

Kramer Harrison, 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.bsdf.base import BaseBSDF
    from optiland.nonsequential.components.geometry.base import AABB, ComponentGeometry
    from optiland.nonsequential.materials.nsq_material import NSQMaterial
    from optiland.nonsequential.ray_bundle import NSQRayBundle


class BaseComponent(ABC):
    """Abstract base class for all non-sequential optical components.

    Components define the geometry and optical interaction (reflection,
    refraction, absorption) for a surface in the NSQ scene.

    Attributes:
        cs: Coordinate system defining position and orientation in global frame.
        geometry: Shape of the component surface.
        material_front: Medium on the front side (normal-facing side).
        material_back: Medium on the back side.
        bsdf: Optional scatter model. None means specular-only.
        name: Optional human-readable label.
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
        """Initialize BaseComponent.

        Args:
            cs: Coordinate system for this component.
            geometry: Surface geometry.
            material_front: Medium on the front (normal-facing) side.
            material_back: Medium on the back side.
            bsdf: Optional BSDF scatter model.
            name: Optional label for this component.
        """
        self.cs = cs
        self.geometry = geometry
        self.material_front = material_front
        self.material_back = material_back
        self.bsdf = bsdf
        self.name = name

    def intersect(
        self, rays: NSQRayBundle
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find the nearest intersection of alive rays with this component.

        Transforms rays to local frame, delegates to geometry, then
        transforms normals back to global frame.

        Args:
            rays: The ray bundle in global coordinates.

        Returns:
            Tuple (t, normals, hit_mask) in global frame:
                - t: Per-ray distances [mm], shape (N,). inf if no hit.
                - normals: Surface normals in global frame, shape (N, 3).
                - hit_mask: Boolean hit mask, shape (N,).
        """
        xp = _get_xp(rays.x)
        translation, rot = _get_transform(self.cs)
        # R transforms column vectors local→global, so:
        # local = R^T @ (global - t), i.e., for row vectors: (global - t) @ R

        # Global ray data as (N, 3) arrays
        positions_g = xp.stack([rays.x, rays.y, rays.z], axis=1)
        directions_g = xp.stack([rays.dx, rays.dy, rays.dz], axis=1)

        t_np = xp.array(translation, dtype=positions_g.dtype)
        R_np = xp.array(rot, dtype=positions_g.dtype)

        # Transform to local frame
        positions_l = (positions_g - t_np) @ R_np
        directions_l = directions_g @ R_np

        # Only intersect alive rays — mask others with inf
        t_hit, normals_l, hit_mask = self.geometry.ray_intersect(
            positions_l, directions_l
        )

        # Dead rays can't hit
        t_hit = xp.where(rays.alive, t_hit, xp.full_like(t_hit, xp.inf))
        hit_mask = hit_mask & rays.alive

        # Transform normals back to global: n_global_row = n_local_row @ R^T
        normals_g = normals_l @ R_np.T

        return t_hit, normals_g, hit_mask

    @abstractmethod
    def interact(
        self,
        rays: NSQRayBundle,
        t: np.ndarray,
        normals: np.ndarray,
        hit_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        """Apply optical interaction at hit points (in-place).

        Updates ray positions, directions, flux, n_current, bounce, and
        alive status for rays that hit this component.

        Args:
            rays: Ray bundle to update in-place.
            t: Hit distances [mm], shape (N,).
            normals: Surface normals in global frame, shape (N, 3).
            hit_mask: True for rays that hit this component, shape (N,).
            rng: Random number generator for stochastic interactions.
        """

    @property
    def bounding_box(self) -> AABB:
        """Axis-aligned bounding box in global coordinates.

        Returns:
            AABB for this component.
        """
        transform = _get_transform(self.cs)
        return self.geometry.bounding_box(transform)


def _get_transform(cs: CoordinateSystem) -> tuple[np.ndarray, np.ndarray]:
    """Extract (translation, rotation_matrix) from a CoordinateSystem.

    Returns plain numpy float64 arrays regardless of the current
    optiland backend.

    Args:
        cs: The coordinate system.

    Returns:
        Tuple (translation [mm], rotation_matrix) as numpy float64 arrays.
        rotation_matrix is (3, 3) and transforms column vectors local→global.
    """
    from optiland.backend.utils import to_numpy  # noqa: PLC0415

    t_be, R_be = cs.get_effective_transform()
    translation = to_numpy(t_be).astype(np.float64)
    rotation = to_numpy(R_be).astype(np.float64)
    return translation, rotation


def _get_xp(arr: np.ndarray):
    try:
        import cupy  # type: ignore[import]

        if isinstance(arr, cupy.ndarray):
            return cupy
    except ImportError:
        pass
    return np
