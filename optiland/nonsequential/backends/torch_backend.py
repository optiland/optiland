"""TorchBackend -- differentiable PyTorch backend for Non-Sequential Raytracing.

Implements a fixed-depth wavefront megakernel loop with autograd support.
Compaction is disabled to keep fixed-shape tensors for the autograd graph.

Memory scaling: O(num_rays x max_depth) activations when gradient_mode is
"autograd". The recommended envelope is ~1e5 rays at depth 16 on a single
GPU.

Kramer Harrison, 2026
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Literal

import numpy as np

import optiland.backend as be
from optiland.backend.utils import to_numpy
from optiland.nonsequential.backends.base import TracerBackend

if TYPE_CHECKING:
    from optiland.nonsequential.components.base import BaseComponent
    from optiland.nonsequential.ray_bundle import NSQRayBundle
    from optiland.nonsequential.scene import NSQScene
    from optiland.nonsequential.tracer import SimulationResult


class TorchBackend(TracerBackend):
    """Differentiable PyTorch backend for NSQ raytracing.

    Uses ``optiland.backend`` (configured to torch) for all computation.
    The fixed-depth wavefront loop lets PyTorch build an autograd graph
    through the entire trace so that ``result.detectors[name].data.backward()``
    propagates gradients to scene parameters.

    Compaction is disabled: dead rays (``alive=False``) carry zero throughput
    and participate in all operations as no-ops; the tensor shape stays fixed
    across bounces so the graph remains clean.

    Gradient strategy is "autograd" (naive attached graph) in v1. A pluggable
    ``gradient_mode`` seam is provided for future Path Replay Backpropagation.

    Attributes:
        seed: RNG seed.
        gradient_mode: Gradient strategy (currently only "autograd").
        rng: NumPy RNG for detached sampling decisions.
    """

    def __init__(
        self,
        seed: int | None = None,
        gradient_mode: Literal["autograd"] = "autograd",
    ) -> None:
        """Initialize TorchBackend.

        Args:
            seed: Optional random seed for reproducibility.
            gradient_mode: Gradient computation strategy. Currently only
                ``"autograd"`` is supported; "prb" is the planned follow-up.
        """
        self.seed = seed
        self.gradient_mode = gradient_mode
        # Detached sampling uses numpy RNG (sampling decisions are detached)
        self.rng = np.random.default_rng(seed)

    def intersect_scene(
        self,
        rays: NSQRayBundle,
        components: list[BaseComponent],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find nearest intersection of each ray with all scene components.

        t_min and hit_normals stay in the torch graph (attached to geometry
        parameters). comp_indices are numpy ints (no grad needed).

        Args:
            rays: Current ray bundle.
            components: List of scene components.

        Returns:
            ``(t_min, hit_normals, component_indices)``.
        """
        N = rays.num_rays
        t_min = be.ones(N) * be.inf
        hit_normals = be.zeros((N, 3))
        comp_indices = np.full(N, -1, dtype=np.int32)

        for i, comp in enumerate(components):
            t_c, normals_c, hit_c = comp.intersect(rays)
            hit_c_np = to_numpy(hit_c).astype(bool)
            t_c_np = to_numpy(t_c)
            better_np = hit_c_np & (t_c_np < to_numpy(t_min))
            # Update t_min and hit_normals via be.where (stays in graph)
            better = be.array(better_np)
            t_min = be.where(better, t_c, t_min)
            hit_normals = be.where(better[:, None], normals_c, hit_normals)
            comp_indices = np.where(better_np, i, comp_indices)

        return t_min, hit_normals, comp_indices

    def random_uniform(self, shape: tuple[int, ...]) -> np.ndarray:
        """Generate uniform random numbers for detached sampling.

        Args:
            shape: Shape of the output array.

        Returns:
            NumPy array of uniform random numbers in [0, 1).
        """
        return self.rng.random(shape)

    def trace(
        self,
        scene: NSQScene,
        num_rays: int,
        max_depth: int = 16,
        min_flux_fraction: float = 1e-6,
        batch_size: int = 1_000_000,
        seed: int | None = None,
        record_paths: bool = False,
    ) -> SimulationResult:
        """Run the differentiable fixed-depth trace.

        Args:
            scene: NSQScene to simulate.
            num_rays: Total rays to launch.
            max_depth: Fixed number of bounces. Rays exceeding this are
                depth-killed. Memory scales O(num_rays × max_depth).
            min_flux_fraction: Kill threshold relative to per-ray initial flux.
            batch_size: Rays per processing batch (forward pass only).
            seed: RNG seed override (overrides constructor seed if provided).
            record_paths: If True, records per-ray event log (numpy, detached).

        Returns:
            SimulationResult with differentiable detector ``data`` tensors.
        """
        from optiland.nonsequential.components.absorbing import (  # noqa: PLC0415
            AbsorbingComponent,
        )
        from optiland.nonsequential.tracer import (  # noqa: PLC0415, I001
            SimulationResult,
        )

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Reset detectors and absorber stats
        for det in scene.detectors:
            det.reset()
        for comp in scene.surfaces:
            if isinstance(comp, AbsorbingComponent):
                comp.reset_stats()

        t_start = time.perf_counter()

        sources = scene.sources
        total_flux_in = sum(s.total_flux for s in sources)
        num_rays_total = int(num_rays)

        flux_per_ray = total_flux_in / num_rays_total if num_rays_total > 0 else 1.0
        min_flux = min_flux_fraction * flux_per_ray

        num_rays_absorbed = 0
        num_rays_escaped = 0
        num_rays_flux_killed = 0
        num_rays_depth_killed = 0
        total_flux_escaped = 0.0
        total_flux_lost = 0.0

        # Distribute ray budget across sources proportional to flux
        rays_per_source: list[int] = []
        if len(sources) == 1:
            rays_per_source = [num_rays_total]
        else:
            remaining = num_rays_total
            for i, src in enumerate(sources):
                if i == len(sources) - 1:
                    n = remaining
                else:
                    n = max(1, round(num_rays_total * src.total_flux / total_flux_in))
                    remaining -= n
                rays_per_source.append(n)

        # Per-ray event log (numpy, detached -- for visualization only)
        event_log: list[dict] | None = [] if record_paths else None
        _next_ray_id: list[int] = [0]

        def _log_birth(rays: NSQRayBundle, source_name: str) -> None:
            if event_log is None:
                return
            n = rays.num_rays
            ids = np.arange(_next_ray_id[0], _next_ray_id[0] + n, dtype=np.int64)
            rays.ray_id = ids
            _next_ray_id[0] += n
            for k in range(n):
                event_log.append(
                    {
                        "ray_id": int(ids[k]),
                        "event_type": "birth",
                        "x": float(to_numpy(rays.x)[k]),
                        "y": float(to_numpy(rays.y)[k]),
                        "z": float(to_numpy(rays.z)[k]),
                        "L": float(to_numpy(rays.L)[k]),
                        "M": float(to_numpy(rays.M)[k]),
                        "N": float(to_numpy(rays.N)[k]),
                        "flux": float(to_numpy(rays.flux)[k]),
                        "wavelength": float(to_numpy(rays.wavelength)[k]),
                        "bounce": int(to_numpy(rays.bounce)[k]),
                        "component_name": source_name,
                    }
                )

        # Main trace loop (no compaction -- fixed-shape for autograd)
        for source_idx, (source, source_num_rays) in enumerate(
            zip(sources, rays_per_source, strict=False)
        ):
            source_name = getattr(source, "name", f"source_{source_idx}")
            source_remaining = source_num_rays

            while source_remaining > 0:
                batch = min(batch_size, source_remaining)
                rays = source.generate(batch, self.rng)
                _log_birth(rays, source_name)

                # Fixed-depth loop (no compaction)
                for _depth in range(max_depth):
                    alive_np = to_numpy(rays.alive).astype(bool)
                    if not alive_np.any():
                        break

                    # Component intersections
                    t_min, hit_normals, comp_idx = self.intersect_scene(
                        rays, scene.surfaces
                    )

                    # Detector intersections (numpy, no grad needed for dispatch)
                    det_t_min_np, det_normals_np, det_idx_np = (
                        self._intersect_detectors_np(rays, scene.detectors)
                    )
                    t_min_np = to_numpy(t_min)

                    # Nearest hit: component vs detector
                    comp_closer_np = t_min_np <= det_t_min_np
                    any_comp_hit_np = comp_idx >= 0
                    any_det_hit_np = det_idx_np >= 0

                    det_first_np = any_det_hit_np & (~comp_closer_np | ~any_comp_hit_np)
                    comp_first_np = any_comp_hit_np & (~det_first_np)

                    det_first = be.array(det_first_np)
                    det_t_min = be.array(det_t_min_np)

                    # Record detector hits
                    for di, det in enumerate(scene.detectors):
                        mask_di_np = det_first_np & (det_idx_np == di)
                        if mask_di_np.any():
                            mask_di = be.array(mask_di_np)
                            det.record(rays, det_t_min, mask_di)

                    # Advance and kill detector-hit rays
                    if det_first_np.any():
                        dx_det = det_t_min * rays.L
                        dy_det = det_t_min * rays.M
                        dz_det = det_t_min * rays.N
                        rays.x = be.where(det_first, rays.x + dx_det, rays.x)
                        rays.y = be.where(det_first, rays.y + dy_det, rays.y)
                        rays.z = be.where(det_first, rays.z + dz_det, rays.z)
                        rays.alive = rays.alive & ~det_first
                        rays.bounce = be.where(det_first, rays.bounce + 1, rays.bounce)

                    # Apply component interactions
                    for ci, comp in enumerate(scene.surfaces):
                        mask_ci_np = comp_first_np & (comp_idx == ci)
                        if mask_ci_np.any():
                            mask_ci = be.array(mask_ci_np)
                            comp.interact(rays, t_min, hit_normals, mask_ci, self.rng)

                    # Kill escaped rays
                    no_hit_np = ~any_comp_hit_np & ~any_det_hit_np
                    escaped_np = no_hit_np & alive_np
                    if escaped_np.any():
                        num_rays_escaped += int(escaped_np.sum())
                        total_flux_escaped += float(
                            to_numpy(rays.flux)[escaped_np].sum()
                        )
                        escaped = be.array(escaped_np)
                        bs = self._estimate_bounding_scale(scene)
                        rays.x = be.where(escaped, rays.x + bs * rays.L, rays.x)
                        rays.y = be.where(escaped, rays.y + bs * rays.M, rays.y)
                        rays.z = be.where(escaped, rays.z + bs * rays.N, rays.z)
                    rays.alive = rays.alive & ~be.array(no_hit_np)

                    # Kill by flux threshold or depth
                    alive_np_now = to_numpy(rays.alive).astype(bool)
                    flux_np = to_numpy(rays.flux)
                    bounce_np = to_numpy(rays.bounce)

                    alive_flux_np = flux_np >= min_flux
                    alive_depth_np = bounce_np < max_depth

                    newly_flux_killed = alive_np_now & ~alive_flux_np
                    newly_depth_killed = alive_np_now & alive_flux_np & ~alive_depth_np

                    if newly_flux_killed.any():
                        num_rays_flux_killed += int(newly_flux_killed.sum())
                        total_flux_lost += float(flux_np[newly_flux_killed].sum())
                    if newly_depth_killed.any():
                        num_rays_depth_killed += int(newly_depth_killed.sum())
                        total_flux_lost += float(flux_np[newly_depth_killed].sum())

                    kill_np = ~alive_flux_np | ~alive_depth_np
                    rays.alive = rays.alive & ~be.array(kill_np)

                source_remaining -= batch

        t_end = time.perf_counter()

        # Collect absorbed stats
        total_flux_absorbed = sum(
            c._absorbed_flux
            for c in scene.surfaces
            if isinstance(c, AbsorbingComponent)
        )
        for comp in scene.surfaces:
            if isinstance(comp, AbsorbingComponent):
                num_rays_absorbed += comp._absorbed_count

        # Collect detector results
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
            abs(
                total_flux_in
                - total_flux_detected
                - total_flux_absorbed
                - total_flux_escaped
            )
            / total_flux_in
            if total_flux_in > 0
            else 0.0
        )

        # Build ray_paths from event log (numpy structured array, detached)
        from optiland.nonsequential.backends.array_backend import (  # noqa: PLC0415
            _EVENT_DTYPE,
        )

        ray_paths = None
        if event_log is not None and event_log:
            arr = np.zeros(len(event_log), dtype=_EVENT_DTYPE)
            for k, ev in enumerate(event_log):
                for fname in _EVENT_DTYPE.names:
                    arr[k][fname] = ev[fname]
            ray_paths = {"events": arr}

        return SimulationResult(
            detectors=detector_results,
            num_rays_total=num_rays_total,
            num_rays_absorbed=num_rays_absorbed,
            num_rays_escaped=num_rays_escaped,
            num_rays_flux_killed=num_rays_flux_killed,
            num_rays_depth_killed=num_rays_depth_killed,
            total_flux_in=total_flux_in,
            total_flux_detected=total_flux_detected,
            total_flux_absorbed=total_flux_absorbed,
            total_flux_escaped=total_flux_escaped,
            total_flux_lost=total_flux_lost,
            flux_conservation_error=flux_err,
            trace_time_sec=t_end - t_start,
            ray_paths=ray_paths,
        )

    def _intersect_detectors_np(
        self,
        rays: NSQRayBundle,
        detectors: list,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find nearest detector intersection, returning numpy arrays.

        Detector dispatch indices don't need gradients.

        Returns:
            ``(t_min_np, hit_normals_np, detector_indices_np)``
        """
        N = rays.num_rays
        t_min = np.full(N, np.inf, dtype=np.float64)
        hit_normals = np.zeros((N, 3), dtype=np.float64)
        det_indices = np.full(N, -1, dtype=np.int32)

        for i, det in enumerate(detectors):
            t_d, normals_d, hit_d = det.intersect(rays)
            t_d_np = to_numpy(t_d).astype(np.float64)
            normals_d_np = to_numpy(normals_d).astype(np.float64)
            hit_d_np = to_numpy(hit_d).astype(bool)
            better = hit_d_np & (t_d_np < t_min)
            t_min = np.where(better, t_d_np, t_min)
            hit_normals = np.where(better[:, None], normals_d_np, hit_normals)
            det_indices = np.where(better, i, det_indices)

        return t_min, hit_normals, det_indices

    def _estimate_bounding_scale(self, scene: NSQScene) -> float:
        """Estimate a reasonable length to extend escaped rays."""
        try:
            boxes = [comp.bounding_box for comp in scene.surfaces]
            if not boxes:
                return 100.0
            xmin = min(b.xmin for b in boxes)
            xmax = max(b.xmax for b in boxes)
            ymin = min(b.ymin for b in boxes)
            ymax = max(b.ymax for b in boxes)
            zmin = min(b.zmin for b in boxes)
            zmax = max(b.zmax for b in boxes)
            extent = float(
                np.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2)
            )
            return extent if extent > 1.0 else 100.0
        except Exception:
            return 100.0


def _get_detector_names(scene: object) -> list[str]:
    """Extract registry names for detectors."""
    try:
        return list(scene.detector_registry._registry.keys())  # type: ignore[attr-defined]
    except AttributeError:
        return []
