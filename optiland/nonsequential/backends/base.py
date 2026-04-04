"""TracerBackend ABC for Non-Sequential Raytracing.

Kramer Harrison, 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from optiland.nonsequential.components.base import BaseComponent
    from optiland.nonsequential.ray_bundle import NSQRayBundle


class TracerBackend(ABC):
    """Abstract base class for NSQ tracer backends.

    A TracerBackend encapsulates device-specific operations: scene
    intersection and random number generation. Subclasses can target
    CPU (NumPy), GPU (CuPy), or hardware ray-tracing (OptiX).

    New backends (e.g., from the `optiland-rt` package) implement this
    ABC and are passed to `NSQTracer.trace(backend=...)`.
    """

    @abstractmethod
    def intersect_scene(
        self,
        rays: NSQRayBundle,
        components: list[BaseComponent],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find the nearest intersection of each ray with the scene components.

        Args:
            rays: Current ray bundle.
            components: List of scene components to test.

        Returns:
            Tuple (t_min, hit_normals, component_indices) where:
                - t_min: Per-ray nearest hit distance [mm], shape (N,).
                    inf for rays with no hit.
                - hit_normals: Surface normals at nearest hit, shape (N, 3),
                    in global frame.
                - component_indices: Index of nearest-hit component in
                    `components` list, shape (N,). -1 if no hit.
        """

    @abstractmethod
    def random_uniform(self, shape: tuple[int, ...]) -> np.ndarray:
        """Generate uniform random numbers [0, 1) on the backend device.

        Args:
            shape: Shape of the output array.

        Returns:
            Uniform random array on the backend device.
        """
