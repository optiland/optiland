"""Base geometry classes for Non-Sequential Components.

Kramer Harrison, 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class AABB:
    """Axis-aligned bounding box in global coordinates.

    Attributes:
        min_corner: Minimum (x, y, z) corner [mm], shape (3,).
        max_corner: Maximum (x, y, z) corner [mm], shape (3,).
    """

    min_corner: np.ndarray
    max_corner: np.ndarray

    @property
    def xmin(self) -> float:
        """Minimum x-extent [mm]."""
        return float(self.min_corner[0])

    @property
    def xmax(self) -> float:
        """Maximum x-extent [mm]."""
        return float(self.max_corner[0])

    @property
    def ymin(self) -> float:
        """Minimum y-extent [mm]."""
        return float(self.min_corner[1])

    @property
    def ymax(self) -> float:
        """Maximum y-extent [mm]."""
        return float(self.max_corner[1])

    @property
    def zmin(self) -> float:
        """Minimum z-extent [mm]."""
        return float(self.min_corner[2])

    @property
    def zmax(self) -> float:
        """Maximum z-extent [mm]."""
        return float(self.max_corner[2])

    def intersects_ray(self, origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
        """Test ray-AABB intersection (slab method).

        Args:
            origins: Ray origins, shape (N, 3).
            directions: Ray directions (unit vectors), shape (N, 3).

        Returns:
            Boolean mask of rays that intersect the AABB, shape (N,).
        """
        inv_d = np.where(
            np.abs(directions) > 1e-15, 1.0 / directions, np.sign(directions) * 1e15
        )
        t_min = (self.min_corner - origins) * inv_d
        t_max = (self.max_corner - origins) * inv_d
        t_enter = np.minimum(t_min, t_max).max(axis=1)
        t_exit = np.maximum(t_min, t_max).min(axis=1)
        return (t_exit >= t_enter) & (t_exit > 0.0)

    @staticmethod
    def union(boxes: list[AABB]) -> AABB:
        """Return the AABB that contains all given boxes.

        Args:
            boxes: List of AABB instances.

        Returns:
            The enclosing AABB.
        """
        if not boxes:
            inf = float("inf")
            return AABB(
                np.array([-inf, -inf, -inf]),
                np.array([inf, inf, inf]),
            )
        mins = np.stack([b.min_corner for b in boxes])
        maxs = np.stack([b.max_corner for b in boxes])
        return AABB(mins.min(axis=0), maxs.max(axis=0))


class ComponentGeometry(ABC):
    """Abstract base for component geometry.

    Subclasses implement `ray_intersect()` for their specific shape.
    Geometry operates in the component's LOCAL coordinate frame.
    """

    @abstractmethod
    def ray_intersect(
        self, origins: np.ndarray, directions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Find ray intersections with this geometry in local coordinates.

        Args:
            origins: Ray origins in local frame, shape (N, 3) [mm].
            directions: Ray directions in local frame, shape (N, 3), unit vectors.

        Returns:
            A tuple (t, normals, hit_mask, n_geom) where:
                - t: Distance to nearest hit, shape (N,). inf if no hit.
                - normals: Hit surface normals in local frame, shape (N, 3).
                    Normals point toward the ray origin side (outward) --
                    used for shading/reflection/refraction math.
                - hit_mask: True where ray actually hits, shape (N,) bool.
                - n_geom: The same surface normal *before* the "flip to face
                    the incoming ray" step, shape (N, 3). Fixed per surface
                    point, independent of which side the ray approached
                    from (D-1, D11 4.7): every geometry orients this so it
                    points from the ``material_front`` side toward the
                    ``material_back`` side -- the contract
                    ``RefractiveComponent`` relies on to determine which
                    material a ray is entering without comparing refractive
                    index values. Components that never need sidedness
                    (reflective, absorbing) ignore it.
        """

    @abstractmethod
    def bounding_box(self, transform: tuple[np.ndarray, np.ndarray]) -> AABB:
        """Return axis-aligned bounding box in global coordinates.

        Args:
            transform: Tuple (translation, rotation_matrix) defining the
                local-to-global transformation. rotation_matrix is (3, 3),
                transforming column vectors local->global.

        Returns:
            AABB in global coordinates.
        """


class AnalyticGeometry(ComponentGeometry, ABC):
    """ABC for analytic geometry primitives.

    Analytic geometries implement closed-form intersection formulas,
    enabling pure-GPU computation with no BVH traversal.
    """
