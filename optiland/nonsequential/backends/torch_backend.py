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
from optiland.nonsequential._utils import DEFAULT_BATCH_SIZE
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
        batch_size: int = DEFAULT_BATCH_SIZE,
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
            batch_size: Rays per processing batch (forward pass only). Does not
                change the result, only the speed; see ``DEFAULT_BATCH_SIZE``.
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
        # Float-cast for stats / kill-threshold; source.generate() uses the
        # raw total_flux (may be a torch Tensor for autograd).
        total_flux_in = sum(float(s.total_flux) for s in sources)
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
                    # float(): total_flux may be a tensor under autograd
                    n = max(
                        1,
                        round(num_rays_total * float(src.total_flux) / total_flux_in),
                    )
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
                # source.generate() spreads the source's whole total_flux over
                # the rays it is asked for, so a batched source would re-emit
                # the full flux once per batch. Rescale to this batch's share
                # of the source's ray budget. A no-op when batch == the budget.
                if batch != source_num_rays:
                    rays.flux = rays.flux * (batch / source_num_rays)
                # Ensure all physics arrays are torch tensors.  Sources
                # produce numpy arrays by default; NumPy 2.0 disallows
                # mixed numpy/torch arithmetic, so we promote upfront.
                rays = self._ensure_torch_bundle(rays)
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

                    # Detector intersections. Dispatch is numpy; t stays attached.
                    det_t_min, _det_normals, det_idx_np = self._intersect_detectors(
                        rays, scene.detectors
                    )
                    det_t_min_np = to_numpy(det_t_min)
                    t_min_np = to_numpy(t_min)

                    # Nearest hit: component vs detector
                    comp_closer_np = t_min_np <= det_t_min_np
                    any_comp_hit_np = comp_idx >= 0
                    any_det_hit_np = det_idx_np >= 0

                    det_first_np = any_det_hit_np & (~comp_closer_np | ~any_comp_hit_np)
                    comp_first_np = any_comp_hit_np & (~det_first_np)

                    det_first = be.array(det_first_np)
                    # Rays that hit no detector carry t = inf. Zero those before
                    # any multiplication: the discarded branch of be.where still
                    # backpropagates 0 * inf = NaN into the ray directions.
                    det_t_safe = be.where(
                        det_first, det_t_min, be.zeros_like(det_t_min)
                    )

                    # Record detector hits
                    for di, det in enumerate(scene.detectors):
                        mask_di_np = det_first_np & (det_idx_np == di)
                        if mask_di_np.any():
                            mask_di = be.array(mask_di_np)
                            det.record(rays, det_t_safe, mask_di)

                    # Advance and kill detector-hit rays
                    if det_first_np.any():
                        dx_det = det_t_safe * rays.L
                        dy_det = det_t_safe * rays.M
                        dz_det = det_t_safe * rays.N
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

        # Every launched watt ends up detected, absorbed, escaped, or killed
        # by the flux/depth cutoffs. Omitting total_flux_lost makes the metric
        # report a large error for any scene that depth-kills rays, which is
        # exactly the stray-light case this diagnostic exists to serve.
        flux_err = (
            abs(
                total_flux_in
                - total_flux_detected
                - total_flux_absorbed
                - total_flux_escaped
                - total_flux_lost
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

    def _intersect_detectors(
        self,
        rays: NSQRayBundle,
        detectors: list,
    ) -> tuple[object, object, np.ndarray]:
        """Find nearest detector intersection, keeping ``t`` in the graph.

        Only the *dispatch* (which detector, and whether it beats the nearest
        component) is decided in NumPy — that choice is a discrete visibility
        event with no gradient anyway.  The returned ``t_min`` stays attached
        to the autograd graph, because the splatted landing position is
        ``origin + t * direction``: detaching ``t`` drops the
        ``direction * dt/dtheta`` term from every spatial loss.  That term is
        negligible at normal incidence but reaches tens of percent for fast
        systems and tilted detectors.

        Args:
            rays: Current ray bundle.
            detectors: Scene detectors.

        Returns:
            ``(t_min, hit_normals, detector_indices)`` where ``t_min`` and
            ``hit_normals`` are backend arrays and the indices are NumPy.
        """
        N = rays.num_rays
        t_min = be.ones(N) * be.inf
        t_min_np = np.full(N, np.inf, dtype=np.float64)
        hit_normals = be.zeros((N, 3))
        det_indices = np.full(N, -1, dtype=np.int32)

        for i, det in enumerate(detectors):
            t_d, normals_d, hit_d = det.intersect(rays)
            t_d_np = to_numpy(t_d).astype(np.float64)
            hit_d_np = to_numpy(hit_d).astype(bool)
            better_np = hit_d_np & (t_d_np < t_min_np)
            better = be.array(better_np)

            t_min = be.where(better, t_d, t_min)
            hit_normals = be.where(better[:, None], normals_d, hit_normals)
            t_min_np = np.where(better_np, t_d_np, t_min_np)
            det_indices = np.where(better_np, i, det_indices)

        return t_min, hit_normals, det_indices

    def _ensure_torch_bundle(self, rays: NSQRayBundle) -> NSQRayBundle:
        """Convert all NSQRayBundle float arrays to torch tensors.

        Sources produce numpy arrays by default.  NumPy 2.0 disallows mixed
        numpy/torch arithmetic, so we promote every field to the current
        backend format at batch start.  Gradient-carrying fields (flux)
        are left untouched if already a Tensor.

        Args:
            rays: Ray bundle from source.generate().

        Returns:
            Same ray bundle with all physics arrays as torch Tensors.
        """
        import torch as _torch  # noqa: PLC0415

        def _to_float(x: object) -> _torch.Tensor:
            if isinstance(x, _torch.Tensor):
                return x
            return be.array(x)

        def _to_bool(x: object) -> _torch.Tensor:
            if isinstance(x, _torch.Tensor) and x.dtype == _torch.bool:
                return x
            arr = to_numpy(x) if isinstance(x, _torch.Tensor) else x
            return _torch.from_numpy(np.asarray(arr, dtype=bool).copy())

        rays.x = _to_float(rays.x)
        rays.y = _to_float(rays.y)
        rays.z = _to_float(rays.z)
        rays.L = _to_float(rays.L)
        rays.M = _to_float(rays.M)
        rays.N = _to_float(rays.N)
        rays.flux = _to_float(rays.flux)
        rays.wavelength = _to_float(rays.wavelength)
        rays.n_current = _to_float(rays.n_current)
        rays.alive = _to_bool(rays.alive)
        # bounce: keep as int32 tensor (used for depth comparisons)
        if not isinstance(rays.bounce, _torch.Tensor):
            rays.bounce = _torch.from_numpy(
                np.asarray(rays.bounce, dtype=np.int32).copy()
            )
        return rays

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
