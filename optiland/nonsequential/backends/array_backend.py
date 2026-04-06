"""ArrayBackend -- base class for array-based tracing backends.

Provides the shared Monte Carlo trace loop for Numpy, Cupy, and other
similar array-based backends. Extracts paths bounce-by-bounce when
requested for visualization.

Kramer Harrison, 2026
"""

from __future__ import annotations

import time
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from optiland.nonsequential.backends.base import TracerBackend
from optiland.nonsequential.ray_bundle import _get_xp

if TYPE_CHECKING:
    from optiland.nonsequential.ray_bundle import NSQRayBundle
    from optiland.nonsequential.scene import NSQScene
    from optiland.nonsequential.tracer import SimulationResult


class ArrayBackend(TracerBackend):
    """Abstract base class for array-based tracing backends.

    Inherits from TracerBackend and defines the concrete Monte Carlo
    while-loop logic shared by Numpy, Cupy, Torch, and Jax backends.
    Subclasses only need to provide intersect_scene() and random_uniform().
    """

    @abstractmethod
    def random_uniform(self, shape: tuple[int, ...]) -> np.ndarray:
        """Generate uniform random numbers on the backend device."""

    def trace(
        self,
        scene: NSQScene,
        num_rays: int,
        max_bounces: int = 200,
        min_flux_fraction: float = 1e-6,
        batch_size: int = 1_000_000,
        seed: int | None = None,
        record_paths: bool = False,
    ) -> SimulationResult:
        """Run the full Monte Carlo simulation using the array backend.

        Args:
            scene: The NSQScene to simulate.
            num_rays: Total rays to launch.
            max_bounces: Maximum surface hits per ray.
            min_flux_fraction: Kill threshold relative to per-ray initial flux.
            batch_size: Rays per processing batch.
            seed: RNG seed for reproducibility.
            record_paths: If True, tracks node points of bouncing rays.

        Returns:
            SimulationResult containing detectors and optional paths.
        """
        from optiland.nonsequential.tracer import SimulationResult  # noqa: PLC0415

        # Reset all detectors
        for det in scene.detectors:
            det.reset()

        t_start = time.perf_counter()

        sources = scene.sources
        total_flux_in = sum(s.total_flux for s in sources)
        num_rays_total = int(num_rays)
        num_rays_remaining = num_rays_total
        num_rays_absorbed = 0
        num_rays_escaped = 0

        flux_per_ray = total_flux_in / num_rays_total if num_rays_total > 0 else 1.0
        min_flux = min_flux_fraction * flux_per_ray

        source = sources[0]  # MVP: single source

        # Storage for ray paths
        recorded_paths: dict[str, list[np.ndarray]] | None = None
        if record_paths:
            recorded_paths = {"x": [], "y": [], "z": []}

        while num_rays_remaining > 0:
            batch = min(batch_size, num_rays_remaining)
            rays = source.generate(batch, self.rng)
            xp = _get_xp(rays.x)

            if record_paths:
                recorded_paths["x"].append(self._to_numpy(rays.x).copy())
                recorded_paths["y"].append(self._to_numpy(rays.y).copy())
                recorded_paths["z"].append(self._to_numpy(rays.z).copy())

            while rays.num_rays_alive > 0:
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
                    rays.x = xp.where(det_first, rays.x + det_t_min * rays.dx, rays.x)
                    rays.y = xp.where(det_first, rays.y + det_t_min * rays.dy, rays.y)
                    rays.z = xp.where(det_first, rays.z + det_t_min * rays.dz, rays.z)
                    rays.alive = rays.alive & ~det_first
                    rays.bounce = xp.where(det_first, rays.bounce + 1, rays.bounce)

                # Apply component interactions
                for ci, comp in enumerate(scene.surfaces):
                    mask_ci = comp_first & (comp_idx == ci)
                    if mask_ci.any():
                        comp.interact(rays, t_min, hit_normals, mask_ci, self.rng)

                # Kill rays with no hit (escaped)
                no_hit = ~any_comp_hit & ~any_det_hit
                escaped_now = no_hit & rays.alive
                num_rays_escaped += int(escaped_now.sum())

                if record_paths and escaped_now.any():
                    bounding_scale = self._estimate_bounding_scale(scene)
                    # Extend escaped rays linearly
                    rays.x = xp.where(
                        escaped_now, rays.x + bounding_scale * rays.dx, rays.x
                    )
                    rays.y = xp.where(
                        escaped_now, rays.y + bounding_scale * rays.dy, rays.y
                    )
                    rays.z = xp.where(
                        escaped_now, rays.z + bounding_scale * rays.dz, rays.z
                    )

                rays.alive = rays.alive & ~no_hit

                # Kill rays below flux threshold or exceeding max bounces
                alive_flux = rays.flux >= min_flux
                alive_bounce = rays.bounce < max_bounces
                rays.alive = rays.alive & alive_flux & alive_bounce

                if record_paths:
                    recorded_paths["x"].append(self._to_numpy(rays.x).copy())
                    recorded_paths["y"].append(self._to_numpy(rays.y).copy())
                    recorded_paths["z"].append(self._to_numpy(rays.z).copy())
                else:
                    # Compact when >50% rays are dead, only if NOT recording paths!
                    if (
                        rays.num_rays_alive > 0
                        and rays.num_rays_alive < rays.num_rays // 2
                    ):
                        rays = rays.compact()

            num_rays_absorbed += batch - int(rays.num_rays)
            num_rays_remaining -= batch

        t_end = time.perf_counter()

        # Collect detector results
        detector_results: dict[str, object] = {}
        total_flux_detected = 0.0
        from optiland.nonsequential.backends.numpy_backend import _get_detector_names

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
            num_rays_total=num_rays_total,
            num_rays_absorbed=num_rays_absorbed,
            num_rays_escaped=num_rays_escaped,
            total_flux_in=total_flux_in,
            total_flux_detected=total_flux_detected,
            flux_conservation_error=flux_err,
            trace_time_sec=t_end - t_start,
            ray_paths=recorded_paths,
        )

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
            ``(t_min, hit_normals, detector_indices)``
        """
        xp = _get_xp(rays.x)
        N = rays.num_rays
        t_min = xp.full(N, xp.inf, dtype=xp.float64)
        hit_normals = xp.zeros((N, 3), dtype=xp.float64)
        det_indices = xp.full(N, -1, dtype=xp.int32)

        for i, det in enumerate(detectors):
            t_d, normals_d, hit_d = det.intersect(rays)

            if xp.__name__ != np.__name__:
                from optiland.nonsequential.backends.numpy_backend import _to_numpy

                t_d_np = _to_numpy(t_d)
                normals_d_np = _to_numpy(normals_d)
                hit_d_np = _to_numpy(hit_d).astype(bool)

                # convert to cupy
                t_d_xp = xp.asarray(t_d_np)
                normals_d_xp = xp.asarray(normals_d_np)
                hit_d_xp = xp.asarray(hit_d_np)
            else:
                t_d_xp = t_d
                normals_d_xp = normals_d
                hit_d_xp = hit_d.astype(bool)

            better = hit_d_xp & (t_d_xp < t_min)
            t_min = xp.where(better, t_d_xp, t_min)
            hit_normals = xp.where(better[:, None], normals_d_xp, hit_normals)
            det_indices = xp.where(better, i, det_indices)

        return t_min, hit_normals, det_indices

    def _estimate_bounding_scale(self, scene: NSQScene) -> float:
        """Estimate a reasonable length to extend escaped rays."""
        try:
            boxes = [comp.bounding_box for comp in scene.surfaces]
            if not boxes:
                return 100.0

            import numpy as fallback_np

            xmin = min(b.xmin for b in boxes)
            xmax = max(b.xmax for b in boxes)
            ymin = min(b.ymin for b in boxes)
            ymax = max(b.ymax for b in boxes)
            zmin = min(b.zmin for b in boxes)
            zmax = max(b.zmax for b in boxes)

            extent = fallback_np.sqrt(
                (xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2
            )
            return float(extent) if extent > 1.0 else 100.0
        except Exception:
            return 100.0

    def _to_numpy(self, arr):
        """Convert array to numpy."""
        try:
            import cupy

            if isinstance(arr, cupy.ndarray):
                return cupy.asnumpy(arr)
        except ImportError:
            pass
        return arr
