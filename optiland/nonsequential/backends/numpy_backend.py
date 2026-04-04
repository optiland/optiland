"""NumPy CPU backend for Non-Sequential Raytracing.

Owns the full Monte Carlo simulation loop.  CupyBackend inherits this loop
and overrides intersect_scene() to run on the GPU.

Kramer Harrison, 2026
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from optiland.nonsequential.backends.base import TracerBackend

if TYPE_CHECKING:
    from optiland.nonsequential.components.base import BaseComponent
    from optiland.nonsequential.ray_bundle import NSQRayBundle
    from optiland.nonsequential.scene import NSQScene
    from optiland.nonsequential.tracer import SimulationResult


class NumpyBackend(TracerBackend):
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

    # ------------------------------------------------------------------
    # TracerBackend.trace — full Monte Carlo loop
    # ------------------------------------------------------------------

    def trace(
        self,
        scene: NSQScene,
        n_rays: int,
        max_bounces: int = 200,
        min_flux_fraction: float = 1e-6,
        batch_size: int = 1_000_000,
        seed: int | None = None,
    ) -> SimulationResult:
        """Run the full Monte Carlo simulation.

        The simulation loop is vectorised over the batch dimension; there are
        no per-ray Python loops inside the inner loop.

        Args:
            scene: The NSQScene to simulate.
            n_rays: Total rays to launch.
            max_bounces: Maximum surface hits per ray.
            min_flux_fraction: Kill threshold relative to per-ray initial flux.
            batch_size: Rays per processing batch.
            seed: RNG seed (overrides the seed passed at construction when
                not None).

        Returns:
            SimulationResult.
        """
        from optiland.nonsequential.tracer import SimulationResult  # noqa: PLC0415

        rng_seed = seed if seed is not None else self.seed
        rng = np.random.default_rng(rng_seed)

        # Reset all detectors
        for det in scene.detectors:
            det.reset()

        t_start = time.perf_counter()

        sources = scene.sources
        total_flux_in = sum(s.total_flux for s in sources)
        n_rays_total = int(n_rays)
        n_rays_remaining = n_rays_total
        n_rays_absorbed = 0
        n_rays_escaped = 0

        flux_per_ray = total_flux_in / n_rays_total if n_rays_total > 0 else 1.0
        min_flux = min_flux_fraction * flux_per_ray

        source = sources[0]  # MVP: single source

        while n_rays_remaining > 0:
            batch = min(batch_size, n_rays_remaining)
            rays = source.generate(batch, rng)

            while rays.n_alive > 0:
                # Component intersections
                t_min, hit_normals, comp_idx = self.intersect_scene(
                    rays, scene.surfaces
                )

                # Detector intersections
                det_t_min, det_normals, det_idx = self._intersect_detectors(
                    rays, scene.detectors
                )

                # Nearest hit: component vs detector
                comp_closer = t_min <= det_t_min
                any_comp_hit = comp_idx >= 0
                any_det_hit = det_idx >= 0

                det_first = any_det_hit & (~comp_closer | ~any_comp_hit)
                comp_first = any_comp_hit & (~det_first)

                # Record detector hits
                for di, det in enumerate(scene.detectors):
                    mask_di = det_first & (det_idx == di)
                    if mask_di.any():
                        det.record(rays, det_t_min, mask_di)

                # Advance and kill detector-hit rays
                if det_first.any():
                    rays.x = np.where(det_first, rays.x + det_t_min * rays.dx, rays.x)
                    rays.y = np.where(det_first, rays.y + det_t_min * rays.dy, rays.y)
                    rays.z = np.where(det_first, rays.z + det_t_min * rays.dz, rays.z)
                    rays.alive = rays.alive & ~det_first
                    rays.bounce = np.where(det_first, rays.bounce + 1, rays.bounce)

                # Apply component interactions
                for ci, comp in enumerate(scene.surfaces):
                    mask_ci = comp_first & (comp_idx == ci)
                    if mask_ci.any():
                        comp.interact(rays, t_min, hit_normals, mask_ci, rng)

                # Kill rays with no hit (escaped)
                no_hit = ~any_comp_hit & ~any_det_hit
                escaped_now = no_hit & rays.alive
                n_rays_escaped += int(escaped_now.sum())
                rays.alive = rays.alive & ~no_hit

                # Kill rays below flux threshold or exceeding max bounces
                alive_flux = rays.flux >= min_flux
                alive_bounce = rays.bounce < max_bounces
                rays.alive = rays.alive & alive_flux & alive_bounce

                # Compact when >50% rays are dead
                if rays.n_alive > 0 and rays.n_alive < rays.n_rays // 2:
                    rays = rays.compact()

            n_rays_absorbed += batch - int(rays.n_rays)
            n_rays_remaining -= batch

        t_end = time.perf_counter()

        # Collect detector results — use registry names when available
        detector_results: dict[str, object] = {}
        total_flux_detected = 0.0
        det_names = _get_detector_names(scene)
        for i, det in enumerate(scene.detectors):
            name = det_names[i] if i < len(det_names) else (det.name or f"detector_{i}")
            result = det.get_result()
            detector_results[name] = result
            if hasattr(result, "total_flux"):
                total_flux_detected += result.total_flux

        flux_err = (
            abs(total_flux_in - total_flux_detected) / total_flux_in
            if total_flux_in > 0
            else 0.0
        )

        return SimulationResult(
            detectors=detector_results,
            n_rays_total=n_rays_total,
            n_rays_absorbed=n_rays_absorbed,
            n_rays_escaped=n_rays_escaped,
            total_flux_in=total_flux_in,
            total_flux_detected=total_flux_detected,
            flux_conservation_error=flux_err,
            trace_time_sec=t_end - t_start,
        )

    # ------------------------------------------------------------------
    # intersect_scene — used internally by trace()
    # ------------------------------------------------------------------

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
        N = rays.n_rays
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

    # ------------------------------------------------------------------
    # Private: detector intersection (not scene-component-specific)
    # ------------------------------------------------------------------

    def _intersect_detectors(
        self,
        rays: NSQRayBundle,
        detectors: list,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find nearest detector intersection for each ray.

        Args:
            rays: Current ray bundle.
            detectors: List of BaseDetector objects.

        Returns:
            ``(t_min, hit_normals, detector_indices)`` where indices are −1
            for rays with no detector hit.
        """
        N = rays.n_rays
        t_min = np.full(N, np.inf)
        hit_normals = np.zeros((N, 3))
        det_indices = np.full(N, -1, dtype=np.int32)

        for i, det in enumerate(detectors):
            t_d, normals_d, hit_d = det.intersect(rays)
            t_d_np = _to_numpy(t_d)
            normals_d_np = _to_numpy(normals_d)
            hit_d_np = _to_numpy(hit_d).astype(bool)

            better = hit_d_np & (t_d_np < t_min)
            t_min = np.where(better, t_d_np, t_min)
            hit_normals = np.where(better[:, None], normals_d_np, hit_normals)
            det_indices = np.where(better, i, det_indices)

        return t_min, hit_normals, det_indices


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


def _to_numpy(arr: np.ndarray) -> np.ndarray:
    """Convert a CuPy or NumPy array to NumPy."""
    try:
        import cupy  # type: ignore[import]

        if isinstance(arr, cupy.ndarray):
            return cupy.asnumpy(arr)
    except ImportError:
        pass
    return np.asarray(arr)
