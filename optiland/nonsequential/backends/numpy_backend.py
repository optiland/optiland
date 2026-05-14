"""NumPy CPU backend for Non-Sequential Raytracing.

Owns the full Monte Carlo simulation loop.  CupyBackend inherits this loop
and overrides intersect_scene() to run on the GPU.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optiland.nonsequential._utils import (
    to_numpy as _to_numpy,  # noqa: F401 backward compat
)
from optiland.nonsequential.backends.array_backend import ArrayBackend

if TYPE_CHECKING:
    from optiland.nonsequential.components.base import BaseComponent
    from optiland.nonsequential.ray_bundle import NSQRayBundle


class NumpyBackend(ArrayBackend):
    """CPU backend using NumPy for all array operations.

    This is the default fallback backend.  All ray data remains in host
    (CPU) memory throughout the simulation.  The full Monte Carlo trace loop
    lives here so that ``CupyBackend`` can inherit it and substitute only the
    device-specific :meth:`intersect_scene` implementation.

    Attributes:
        rng: NumPy random generator.
        seed: RNG seed stored for internal use.
    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialize NumpyBackend.

        Args:
            seed: Optional random seed for reproducibility.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def intersect_scene(
        self,
        rays: NSQRayBundle,
        components: list[BaseComponent],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find nearest intersection of each ray with all scene components.

        Iterates over components, calls each component's intersect(), and
        takes the argmin over distances.

        Args:
            rays: Current ray bundle (NumPy arrays).
            components: List of scene components.

        Returns:
            ``(t_min, hit_normals, component_indices)``.
        """
        N = rays.num_rays
        t_min = np.full(N, np.inf)
        hit_normals = np.zeros((N, 3))
        comp_indices = np.full(N, -1, dtype=np.int32)

        for i, comp in enumerate(components):
            t_c, normals_c, hit_c = comp.intersect(rays)
            better = hit_c & (t_c < t_min)
            t_min = np.where(better, t_c, t_min)
            hit_normals = np.where(better[:, None], normals_c, hit_normals)
            comp_indices = np.where(better, i, comp_indices)

        return t_min, hit_normals, comp_indices

    def random_uniform(self, shape: tuple[int, ...]) -> np.ndarray:
        """Generate uniform random numbers using the internal RNG.

        Args:
            shape: Shape of the output array.

        Returns:
            NumPy array of uniform random numbers in [0, 1).
        """
        return self.rng.random(shape)


def _get_detector_names(scene) -> list[str]:
    """Extract registry names for detectors, falling back to index-based names.

    Args:
        scene: NSQScene instance.

    Returns:
        Ordered list of detector names.
    """
    try:
        return list(scene.detector_registry._registry.keys())
    except AttributeError:
        return []
