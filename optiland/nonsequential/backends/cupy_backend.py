"""CuPy GPU backend for Non-Sequential Raytracing.

Inherits the full Monte Carlo loop from NumpyBackend and overrides only the
device-specific operations (intersection and random number generation).

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.nonsequential.backends.array_backend import ArrayBackend

if TYPE_CHECKING:
    import numpy as np

    from optiland.nonsequential.components.base import BaseComponent
    from optiland.nonsequential.ray_bundle import NSQRayBundle


class CupyBackend(ArrayBackend):
    """GPU backend using CuPy for analytic geometry intersection.

    All ray data lives in device (GPU) memory.  Analytic geometry
    intersection is performed entirely on-device without CPU round-trips.
    Mesh geometry (trimesh BVH) requires CPU transfer per intersection step.

    The full Monte Carlo trace loop is inherited from
    :class:`~optiland.nonsequential.backends.array_backend.ArrayBackend`.
    Only :meth:`intersect_scene` and :meth:`random_uniform` are overridden
    for GPU execution.

    Requires: ``pip install cupy-cudaXX`` matching your CUDA version.

    Attributes:
        _cp: The ``cupy`` module.
        rng: CuPy random generator.
    """

    def __init__(self, seed: int | None = None, device_id: int = 0) -> None:
        """Initialize CupyBackend.

        Args:
            seed: Optional random seed for reproducibility.
            device_id: CUDA device index to use (default: 0).

        Raises:
            ImportError: If CuPy is not installed.
        """
        try:
            import cupy as cp  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "CuPy is required for the GPU backend. "
                "Install with: pip install cupy-cudaXX"
            ) from e
        self._cp = cp
        cp.cuda.Device(device_id).use()
        # Store seed for NumpyBackend; override rng with CuPy generator
        self.seed = seed
        self.rng = cp.random.default_rng(seed)

    def intersect_scene(
        self,
        rays: NSQRayBundle,
        components: list[BaseComponent],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find nearest intersection on the GPU.

        Analytic components compute intersections entirely on-device.
        Mesh components require a CPU transfer for BVH traversal.

        Args:
            rays: Current ray bundle (CuPy arrays).
            components: List of scene components.

        Returns:
            ``(t_min, hit_normals, component_indices)`` as CuPy arrays.
        """
        cp = self._cp
        num_rays = rays.num_rays
        t_min = cp.full(num_rays, cp.inf, dtype=cp.float64)
        hit_normals = cp.zeros((num_rays, 3), dtype=cp.float64)
        comp_indices = cp.full(num_rays, -1, dtype=cp.int32)

        for i, comp in enumerate(components):
            t_c, normals_c, hit_c = comp.intersect(rays)
            better = hit_c & (t_c < t_min)
            t_min = cp.where(better, t_c, t_min)
            hit_normals = cp.where(better[:, None], normals_c, hit_normals)
            comp_indices = cp.where(better, i, comp_indices)

        return t_min, hit_normals, comp_indices

    def random_uniform(self, shape: tuple[int, ...]) -> np.ndarray:
        """Generate uniform random numbers on the GPU.

        Args:
            shape: Shape of the output array.

        Returns:
            CuPy array of uniform random numbers in [0, 1).
        """
        return self.rng.random(shape)

    def to_device(self, arr: np.ndarray):
        """Transfer a NumPy array to the GPU.

        Args:
            arr: NumPy array to transfer.

        Returns:
            CuPy array on the GPU device.
        """
        return self._cp.array(arr)

    def to_host(self, arr) -> np.ndarray:
        """Transfer a CuPy array to CPU memory.

        Args:
            arr: CuPy array to transfer.

        Returns:
            NumPy array on the CPU.
        """
        return self._cp.asnumpy(arr)
