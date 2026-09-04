"""Base detector for Non-Sequential Raytracing.

Kramer Harrison, 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.nonsequential.components.base import _get_transform

if TYPE_CHECKING:
    import numpy as np

    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.components.geometry.base import AABB, ComponentGeometry
    from optiland.nonsequential.ray_bundle import NSQRayBundle


class BaseDetector(ABC):
    """Abstract base class for detectors in the NSQ scene.

    Detectors record ray data at a surface. They intersect rays (via their
    geometry) and accumulate hit data across simulation batches.

    Detectors are **absorbing by default**: a ray that reaches a detector is
    recorded and then terminated. Setting ``absorb=False`` records the ray
    but lets it continue unchanged, so a detector can be tilted into a
    converging beam to sample it mid-system without terminating it.
    Several detectors may share one scene; among the detectors a ray would
    still reach, only the nearest one sees it, so stacking *absorbing*
    detectors down a beam records the beam at the nearest plane and nothing
    beyond it -- use ``absorb=False`` (or trace one scene per plane) to
    profile a beam at several planes.

    Attributes:
        cs: Coordinate system defining detector position and orientation.
        geometry: Surface geometry that defines the detector area.
        name: Optional human-readable label.
        absorb: Whether a hit terminates the ray. False => the ray is
            recorded and passes through unaffected.
    """

    def __init__(
        self,
        cs: CoordinateSystem,
        geometry: ComponentGeometry,
        name: str = "",
        absorb: bool = True,
    ) -> None:
        """Initialize BaseDetector.

        Args:
            cs: Coordinate system.
            geometry: Surface geometry.
            name: Optional label.
            absorb: Whether a hit terminates the ray (default True).
                False makes the detector transmissive: the hit is recorded
                and the ray continues with its direction unchanged.
        """
        self.cs = cs
        self.geometry = geometry
        self.name = name
        self.absorb = bool(absorb)

    def intersect(
        self, rays: NSQRayBundle
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find ray intersections with this detector surface.

        Args:
            rays: Ray bundle in global coordinates.

        Returns:
            Tuple (t, normals, hit_mask) in global frame.
        """
        translation, rot = _get_transform(self.cs)

        positions_g = be.stack([rays.x, rays.y, rays.z], axis=1)
        directions_g = be.stack([rays.L, rays.M, rays.N], axis=1)

        t_arr = be.array(translation)
        R_arr = be.array(rot)

        positions_l = (positions_g - t_arr) @ R_arr
        directions_l = directions_g @ R_arr

        t_hit, normals_l, hit_mask, _n_geom_l = self.geometry.ray_intersect(
            positions_l, directions_l
        )

        # Geometry may return numpy arrays even in torch-backend mode (geometry
        # internals are numpy-based). Convert to the current backend format so
        # that be.where dispatches correctly in both NumPy and Torch paths.
        t_hit = be.array(t_hit)
        normals_l = be.array(normals_l)
        hit_mask = be.array(hit_mask)

        # T_EPSILON guard: prevent self-intersection
        T_EPSILON = 1e-9
        t_hit = be.where(t_hit > T_EPSILON, t_hit, be.full_like(t_hit, be.inf))
        hit_mask = hit_mask & (t_hit > T_EPSILON)

        alive_be = be.array(rays.alive)
        t_hit = be.where(alive_be, t_hit, be.full_like(t_hit, be.inf))
        hit_mask = hit_mask & alive_be

        normals_g = normals_l @ R_arr.T
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

    @abstractmethod
    def reset(self) -> None:
        """Clear accumulated data for reuse in a new simulation."""

    @property
    def bounding_box(self) -> AABB:
        """AABB of this detector in global coordinates."""
        transform = _get_transform(self.cs)
        return self.geometry.bounding_box(transform)
