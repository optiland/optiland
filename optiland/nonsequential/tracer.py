"""NSQ Tracer — core simulation loop for Non-Sequential Raytracing.

Implements the Monte Carlo photon-tracing loop described in the spec.
No blocking operations (disk I/O, print, per-ray Python loops) are
performed inside the trace loop.

Kramer Harrison, 2026
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from optiland.nonsequential.backends.numpy_backend import NumpyBackend

if TYPE_CHECKING:
    from optiland.nonsequential.backends.base import TracerBackend
    from optiland.nonsequential.ray_bundle import NSQRayBundle
    from optiland.nonsequential.scene import NSQScene


@dataclass
class SimulationResult:
    """Top-level result returned by NSQTracer.trace().

    Attributes:
        detectors: Per-detector result objects, keyed by detector name
            (or 'detector_N' if unnamed).
        n_rays_total: Total number of rays launched.
        n_rays_absorbed: Rays terminated by absorbing components.
        n_rays_escaped: Rays that ran out of bounces or fell below flux threshold.
        total_flux_in: Total flux launched by all sources [W].
        total_flux_detected: Total flux recorded on all detectors [W].
        flux_conservation_error: |flux_in - (detected + absorbed + escaped)| / flux_in.
        trace_time_sec: Wall-clock time for the trace [s].
        detector_results: Alias for detectors (same object).
    """

    detectors: dict[str, object] = field(default_factory=dict)
    n_rays_total: int = 0
    n_rays_absorbed: int = 0
    n_rays_escaped: int = 0
    total_flux_in: float = 0.0
    total_flux_detected: float = 0.0
    flux_conservation_error: float = 0.0
    trace_time_sec: float = 0.0


class NSQTracer:
    """Non-sequential Monte Carlo ray tracer.

    Traces batches of rays from sources through components and detectors
    using Russian-roulette Fresnel splitting. The simulation loop is fully
    vectorized over N rays — no per-ray Python loops inside the inner loop.

    Attributes:
        scene: The NSQScene to trace.
        n_rays: Total number of rays to launch.
        max_bounces: Maximum surface interactions per ray.
        min_flux_fraction: Kill rays below this fraction of initial flux.
        batch_size: Number of rays per batch (tune for GPU memory).
        polarization: If True, track Stokes parameters (not yet implemented).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        scene: NSQScene,
        n_rays: int,
        max_bounces: int = 200,
        min_flux_fraction: float = 1e-6,
        batch_size: int = 1_000_000,
        polarization: bool = False,
        seed: int | None = None,
    ) -> None:
        """Initialize NSQTracer.

        Args:
            scene: The NSQ scene to trace.
            n_rays: Total number of rays to launch across all batches.
            max_bounces: Maximum surface hits per ray before termination.
            min_flux_fraction: Rays with flux below
                min_flux_fraction * initial_flux_per_ray are killed.
            batch_size: Number of rays per simulation batch.
            polarization: If True, allocate Stokes arrays (not yet functional).
            seed: Random seed for reproducibility.
        """
        self.scene = scene
        self.n_rays = int(n_rays)
        self.max_bounces = int(max_bounces)
        self.min_flux_fraction = float(min_flux_fraction)
        self.batch_size = int(batch_size)
        self.polarization = polarization
        self.seed = seed

    def trace(self, backend: TracerBackend | None = None) -> SimulationResult:
        """Run the simulation and return results.

        Follows the simulation loop from the spec:
        1. Validate scene.
        2. Initialize RNG.
        3. Process rays in batches.
        4. Aggregate detector results.

        Args:
            backend: TracerBackend to use. Defaults to NumpyBackend.

        Returns:
            SimulationResult with detector data and statistics.
        """
        self.scene.validate()

        if backend is None:
            backend = NumpyBackend(seed=self.seed)

        rng = np.random.default_rng(self.seed)

        # Reset all detectors for a clean run
        for det in self.scene.detectors:
            det.reset()

        t_start = time.perf_counter()

        # Gather all sources (combine flux if multiple sources)
        sources = self.scene.sources
        total_flux_in = sum(s.total_flux for s in sources)
        n_rays_total = self.n_rays
        n_rays_remaining = n_rays_total
        n_rays_absorbed = 0
        n_rays_escaped = 0

        # Flux threshold per ray
        flux_per_ray = total_flux_in / n_rays_total if n_rays_total > 0 else 1.0
        min_flux = self.min_flux_fraction * flux_per_ray

        source = sources[0]  # MVP: single source

        while n_rays_remaining > 0:
            batch = min(self.batch_size, n_rays_remaining)

            # Generate ray batch from source
            rays = source.generate(batch, rng)

            # Inner loop: bounce until all rays dead
            while rays.n_alive > 0:
                # --- Component intersection ---
                t_min, hit_normals, comp_idx = backend.intersect_scene(
                    rays, self.scene.components
                )

                # --- Detector intersection ---
                det_t_min, det_normals, det_idx = self._intersect_detectors(rays)

                # --- Find globally nearest hit (component or detector) ---
                # Determine which is closer: component or detector
                comp_closer = t_min <= det_t_min
                any_comp_hit = comp_idx >= 0
                any_det_hit = det_idx >= 0

                # Rays that hit a detector closer than any component
                det_first = any_det_hit & (~comp_closer | ~any_comp_hit)
                comp_first = any_comp_hit & (~det_first)

                # --- Record detector hits (before advancing rays) ---
                for di, det in enumerate(self.scene.detectors):
                    mask_di = det_first & (det_idx == di)
                    if mask_di.any():
                        det.record(rays, det_t_min, mask_di)

                # --- Advance detector-hit rays (kill them: detectors absorb) ---
                if det_first.any():
                    rays.x = np.where(det_first, rays.x + det_t_min * rays.dx, rays.x)
                    rays.y = np.where(det_first, rays.y + det_t_min * rays.dy, rays.y)
                    rays.z = np.where(det_first, rays.z + det_t_min * rays.dz, rays.z)
                    rays.alive = rays.alive & ~det_first
                    rays.bounce = np.where(det_first, rays.bounce + 1, rays.bounce)

                # --- Apply component interactions ---
                for ci, comp in enumerate(self.scene.components):
                    mask_ci = comp_first & (comp_idx == ci)
                    if mask_ci.any():
                        comp.interact(rays, t_min, hit_normals, mask_ci, rng)

                # --- Kill rays that escape (no hit at all) ---
                no_hit = ~any_comp_hit & ~any_det_hit
                escaped_now = no_hit & rays.alive
                n_rays_escaped += int(escaped_now.sum())
                rays.alive = rays.alive & ~no_hit

                # --- Kill rays below flux threshold or exceeding max bounces ---
                flux_ok = rays.flux >= min_flux
                bounce_ok = rays.bounce < self.max_bounces
                rays.alive = rays.alive & flux_ok & bounce_ok

                # --- Compact ray array when >50% rays are dead ---
                if rays.n_alive > 0 and rays.n_alive < rays.n_rays // 2:
                    rays = rays.compact()

            # Count absorbed rays (those killed by absorbing components)
            # Approximation: rays not alive and not detected and not escaped
            n_rays_absorbed += batch - int(rays.n_rays)  # after compaction

            n_rays_remaining -= batch

        t_end = time.perf_counter()

        # Collect detector results
        detector_results: dict[str, object] = {}
        total_flux_detected = 0.0
        for i, det in enumerate(self.scene.detectors):
            name = det.name if det.name else f"detector_{i}"
            result = det.get_result()
            detector_results[name] = result
            # Sum detected flux
            if hasattr(result, "total_flux"):
                total_flux_detected += result.total_flux

        # Flux conservation
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

    def _intersect_detectors(
        self, rays: NSQRayBundle
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find nearest detector intersection for each ray.

        Args:
            rays: Current ray bundle.

        Returns:
            (t_min, hit_normals, detector_indices) where detector_indices
            is -1 for rays with no detector hit.
        """
        N = rays.n_rays
        t_min = np.full(N, np.inf)
        hit_normals = np.zeros((N, 3))
        det_indices = np.full(N, -1, dtype=np.int32)

        for i, det in enumerate(self.scene.detectors):
            t_d, normals_d, hit_d = det.intersect(rays)
            # Ensure numpy arrays (in case CuPy backend was used for components)
            t_d_np = _to_numpy_if_needed(t_d)
            normals_d_np = _to_numpy_if_needed(normals_d)
            hit_d_np = _to_numpy_if_needed(hit_d).astype(bool)

            better = hit_d_np & (t_d_np < t_min)
            t_min = np.where(better, t_d_np, t_min)
            hit_normals = np.where(better[:, None], normals_d_np, hit_normals)
            det_indices = np.where(better, i, det_indices)

        return t_min, hit_normals, det_indices


def _to_numpy_if_needed(arr):
    """Convert cupy array to numpy if needed, else return as-is."""
    try:
        import cupy  # noqa: PLC0415  # type: ignore[import]

        if isinstance(arr, cupy.ndarray):
            return cupy.asnumpy(arr)
    except ImportError:
        pass
    return np.asarray(arr)
