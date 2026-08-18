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
import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np

import optiland.backend as be
from optiland.backend.utils import to_numpy
from optiland.nonsequential._utils import (
    DEFAULT_BATCH_SIZE,
    distribute_ray_budget,
    estimate_bounding_scale,
    get_detector_names,
)
from optiland.nonsequential.backends.base import TracerBackend
from optiland.nonsequential.detectors.dispatch import (
    detector_absorb_mask,
    intersect_detectors,
)
from optiland.nonsequential.diagnostics import build_diagnostics
from optiland.nonsequential.ir.interpreter import apply_primitive_interactions
from optiland.nonsequential.ir.lower import lower
from optiland.nonsequential.path_recording import PathRecorder
from optiland.nonsequential.rng import NSQRng
from optiland.nonsequential.sampling import russian_roulette

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
        rng: Keyed PCG32 RNG for detached sampling decisions (see
            :mod:`optiland.nonsequential.rng`).
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
        # Detached sampling uses a keyed RNG (sampling decisions are detached)
        self.rng = NSQRng(seed)

    def intersect_scene(
        self,
        rays: NSQRayBundle,
        components: list[BaseComponent],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Find nearest intersection of each ray with all scene components.

        t_min and hit_normals stay in the torch graph (attached to geometry
        parameters). comp_indices are numpy ints (no grad needed). n_geom is
        purely geometric (never a function of a differentiable material
        parameter), but is still built via be.where to stay consistent with
        the rest of this method and to support a differentiable geometry
        (radius, conic, ...) tilting n_geom itself.

        Args:
            rays: Current ray bundle.
            components: List of scene components.

        Returns:
            ``(t_min, hit_normals, component_indices, hit_n_geom)``.
        """
        N = rays.num_rays
        t_min = be.ones(N) * be.inf
        hit_normals = be.zeros((N, 3))
        hit_n_geom = be.zeros((N, 3))
        comp_indices = np.full(N, -1, dtype=np.int32)

        for i, comp in enumerate(components):
            t_c, normals_c, hit_c, n_geom_c = comp.intersect(rays)
            hit_c_np = to_numpy(hit_c).astype(bool)
            t_c_np = to_numpy(t_c)
            better_np = hit_c_np & (t_c_np < to_numpy(t_min))
            # Update t_min and hit_normals via be.where (stays in graph)
            better = be.array(better_np)
            t_min = be.where(better, t_c, t_min)
            hit_normals = be.where(better[:, None], normals_c, hit_normals)
            hit_n_geom = be.where(better[:, None], n_geom_c, hit_n_geom)
            comp_indices = np.where(better_np, i, comp_indices)

        return t_min, hit_normals, comp_indices, hit_n_geom

    def trace(
        self,
        scene: NSQScene,
        num_rays: int,
        max_depth: int = 16,
        min_flux_fraction: float = 1e-6,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seed: int | None = None,
        record_paths: bool | int = False,
    ) -> SimulationResult:
        """Run the differentiable fixed-depth trace.

        Args:
            scene: NSQScene to simulate.
            num_rays: Total rays to launch.
            max_depth: Fixed number of bounces. Rays exceeding this are
                depth-killed. Memory scales O(num_rays × max_depth).
            min_flux_fraction: Russian-roulette threshold, relative to
                per-ray initial flux -- combined with the scene's
                ``sampling_policy.rr_start_flux`` (the larger of the two
                wins). Below threshold, rays are killed with an unbiased
                probability and survivors' flux is boosted accordingly,
                rather than truncated outright.
            batch_size: Rays per processing batch (forward pass only). Does not
                change the result, only the speed; see ``DEFAULT_BATCH_SIZE``.
            seed: RNG seed override (overrides constructor seed if provided).
            record_paths: ``False`` records nothing, ``True`` records every
                ray's path (numpy, detached), and a positive ``int`` records
                an approximately that-many-ray subset selected
                deterministically by ``ray_id`` hash -- see
                :mod:`optiland.nonsequential.path_recording`.

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
            self.rng = NSQRng(seed)

        # Reset detectors and absorber stats
        for det in scene.detectors:
            det.reset()
        for comp in scene.surfaces:
            if isinstance(comp, AbsorbingComponent):
                comp.reset_stats()

        # The per-bounce interaction loop below is driven by this IR, not by
        # iterating scene.surfaces and branching on Python class identity.
        ir = lower(scene, strict=False)

        # Bounded splitting is the NumPy forward engine only: it
        # grows the live ray bundle, which conflicts with the fixed tensor
        # shapes this backend's autograd graph requires. Never silently
        # ignored (D-14-class failure mode) -- warn and fall back to
        # importance-biased single-branch sampling, which this backend
        # always uses regardless of split_depth.
        if ir.sampling.split_depth > 0:
            warnings.warn(
                f"TorchBackend does not support bounded splitting "
                f"(sampling_policy.split_depth={ir.sampling.split_depth}); "
                "fixed tensor shapes are required for the autograd graph. "
                "Falling back to importance-biased single-branch sampling "
                "(split_depth is ignored). Use NumpyBackend for bounded "
                "splitting.",
                stacklevel=2,
            )

        t_start = time.perf_counter()

        sources = scene.sources
        # Float-cast for stats / kill-threshold; source.generate() uses the
        # raw total_flux (may be a torch Tensor for autograd).
        total_flux_in = sum(float(s.total_flux) for s in sources)
        num_rays_total = int(num_rays)

        flux_per_ray = total_flux_in / num_rays_total if num_rays_total > 0 else 1.0

        num_rays_absorbed = 0
        num_rays_escaped = 0
        num_rays_flux_killed = 0
        num_rays_depth_killed = 0
        total_flux_escaped = 0.0
        total_flux_bulk_absorbed = 0.0
        # Tracked separately for Diagnostics -- see the
        # matching comment in ArrayBackend.trace().
        total_flux_depth_killed = 0.0
        total_flux_rr_killed = 0.0
        hit_component_ids: set[int] = set()
        total_medium_stack_underflows = 0

        # Distribute ray budget across sources proportional to flux
        rays_per_source = distribute_ray_budget(
            num_rays_total, [float(s.total_flux) for s in sources]
        )

        # Vectorised columnar path recording: PathRecorder
        # replaces the old birth-only per-event Python dict closure with
        # preallocated array writes, and adds hit/death recording (the
        # array backend already had both; this backend previously recorded
        # only birth events -- an existing parity gap this PR also closes)
        # plus the record_paths: int subset contract.
        path_recorder = PathRecorder(record_paths, num_rays_total, self.rng.seed)
        _next_ray_id: list[int] = [0]

        # Main trace loop (no compaction -- fixed-shape for autograd)
        for source_idx, (source, source_num_rays) in enumerate(
            zip(sources, rays_per_source, strict=False)
        ):
            source_name = getattr(source, "name", f"source_{source_idx}")
            source_remaining = source_num_rays

            while source_remaining > 0:
                batch = min(batch_size, source_remaining)
                ray_id = np.arange(
                    _next_ray_id[0], _next_ray_id[0] + batch, dtype=np.int64
                )
                _next_ray_id[0] += batch
                rays = source.generate(ray_id, self.rng)
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
                path_recorder.log_birth(rays, source_name)

                # Fixed-depth loop (no compaction)
                for _depth in range(max_depth):
                    alive_np = to_numpy(rays.alive).astype(bool)
                    if not alive_np.any():
                        break

                    # Component intersections
                    t_min, hit_normals, comp_idx, hit_n_geom = self.intersect_scene(
                        rays, scene.surfaces
                    )

                    # Detector intersections. Dispatch is numpy; t stays
                    # attached (shared with ArrayBackend/NumpyBackend; D-10).
                    det_t_min, _det_normals, det_idx_np = intersect_detectors(
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

                    # unreached_geometry: cheap running set of
                    # every primitive that was ever the nearest hit.
                    if comp_first_np.any():
                        hit_component_ids.update(
                            np.unique(comp_idx[comp_first_np]).tolist()
                        )

                    det_first = be.array(det_first_np)
                    # Rays that hit no detector carry t = inf. Zero those before
                    # any multiplication: the discarded branch of be.where still
                    # backpropagates 0 * inf = NaN into the ray directions.
                    det_t_safe = be.where(
                        det_first, det_t_min, be.zeros_like(det_t_min)
                    )

                    # Beer-Lambert bulk absorption: attenuate flux over
                    # the segment each ray just travelled through its
                    # *current* medium (rays.k_current, set at its last
                    # crossing or its source's ambient medium) before this
                    # bounce's nearest hit -- component or detector,
                    # whichever is closer. Applied before interact()/detector
                    # recording touch flux or k_current so both see the
                    # already-attenuated value; k_current itself is only
                    # updated afterwards, by RefractiveComponent.interact(),
                    # for the medium the ray is now entering.
                    hit_first_np = comp_first_np | det_first_np
                    if hit_first_np.any():
                        hit_first = be.array(hit_first_np)
                        comp_first = be.array(comp_first_np)
                        comp_t_safe = be.where(comp_first, t_min, be.zeros_like(t_min))
                        hit_t = be.where(comp_first, comp_t_safe, det_t_safe)
                        alpha = 4.0 * be.pi * rays.k_current / rays.wavelength
                        # hit_t is in mm; alpha is in 1/um -> convert to um.
                        transmittance = be.exp(-alpha * hit_t * 1e3)
                        flux_before = rays.flux
                        rays.flux = flux_before * be.where(
                            hit_first, transmittance, be.ones_like(rays.flux)
                        )
                        total_flux_bulk_absorbed += float(
                            to_numpy(flux_before - rays.flux).sum()
                        )

                    # Record detector hits
                    for di, det in enumerate(scene.detectors):
                        mask_di_np = det_first_np & (det_idx_np == di)
                        if mask_di_np.any():
                            mask_di = be.array(mask_di_np)
                            det_name = getattr(det, "name", f"detector_{di}")
                            path_recorder.log_hits(
                                rays, mask_di_np, det_name, t_offset=det_t_safe
                            )
                            det.record(rays, det_t_safe, mask_di)

                    # Advance detector-hit rays. Absorbing detectors
                    # terminate the ray; absorb=False detectors are
                    # transmissive: the hit is recorded (above) and the ray
                    # continues on its unchanged direction.
                    if det_first_np.any():
                        dx_det = det_t_safe * rays.L
                        dy_det = det_t_safe * rays.M
                        dz_det = det_t_safe * rays.N
                        rays.x = be.where(det_first, rays.x + dx_det, rays.x)
                        rays.y = be.where(det_first, rays.y + dy_det, rays.y)
                        rays.z = be.where(det_first, rays.z + dz_det, rays.z)
                        rays.bounce = be.where(det_first, rays.bounce + 1, rays.bounce)

                        absorb_np = detector_absorb_mask(det_idx_np, scene.detectors)
                        kill_np = det_first_np & absorb_np
                        rays.alive = rays.alive & ~be.array(kill_np)

                    # Apply component interactions, dispatched from the IR
                    # (ir.primitives[i].component_kind / .bsdf.kind) rather
                    # than by iterating scene.surfaces and checking isinstance.
                    apply_primitive_interactions(
                        rays,
                        ir,
                        scene.surfaces,
                        t_min,
                        hit_normals,
                        hit_n_geom,
                        comp_idx,
                        comp_first_np,
                        self.rng,
                        log_hit_fn=path_recorder.log_hits,
                    )

                    # D1: flush this bounce's medium-stack underflow counts
                    # (see RefractiveComponent.interact) into the running
                    # total, then reset -- counted exactly once per bounce.
                    total_medium_stack_underflows += int(
                        rays.medium_stack_underflows.sum()
                    )
                    rays.medium_stack_underflows[:] = 0

                    # Kill escaped rays
                    no_hit_np = ~any_comp_hit_np & ~any_det_hit_np
                    escaped_np = no_hit_np & alive_np
                    if escaped_np.any():
                        num_rays_escaped += int(escaped_np.sum())
                        total_flux_escaped += float(
                            to_numpy(rays.flux)[escaped_np].sum()
                        )
                        path_recorder.log_deaths(rays, escaped_np, "escaped")
                        escaped = be.array(escaped_np)
                        bs = estimate_bounding_scale(scene)
                        rays.x = be.where(escaped, rays.x + bs * rays.L, rays.x)
                        rays.y = be.where(escaped, rays.y + bs * rays.M, rays.y)
                        rays.z = be.where(escaped, rays.z + bs * rays.N, rays.z)
                    rays.alive = rays.alive & ~be.array(no_hit_np)

                    # Depth truncation: hard kill (inherent, reported bias --
                    # unlike the flux threshold below, not replaced by
                    # roulette; there is no unbiased way to "continue" a ray
                    # past a hard bounce-count cap).
                    alive_np_now = to_numpy(rays.alive).astype(bool)
                    bounce_np = to_numpy(rays.bounce)
                    alive_depth_np = bounce_np < max_depth
                    newly_depth_killed = alive_np_now & ~alive_depth_np
                    if newly_depth_killed.any():
                        num_rays_depth_killed += int(newly_depth_killed.sum())
                        total_flux_depth_killed += float(
                            to_numpy(rays.flux)[newly_depth_killed].sum()
                        )
                        path_recorder.log_deaths(
                            rays, newly_depth_killed, "depth_killed"
                        )
                    rays.alive = rays.alive & be.array(alive_depth_np)

                    # Russian roulette replaces the old biased hard
                    # kill below min_flux: unbiased stochastic termination
                    # (kill with probability p, boost survivors by
                    # 1/(1-p)), so total_flux_lost reports a genuine
                    # diagnostic (~0 for a well-configured scene) instead of
                    # an expected bookkeeping entry. Same mechanism as the
                    # NumPy backend (optiland.nonsequential.sampling) -- no
                    # shape change is needed here, since a killed ray simply
                    # gets alive=False like any other kill on this
                    # fixed-shape backend.
                    rr_threshold_fraction = max(
                        min_flux_fraction, ir.sampling.rr_start_flux
                    )
                    flux_before_rr = rays.flux
                    rays.flux, rays.alive, rr_killed_np = russian_roulette(
                        rays.flux,
                        rays.alive,
                        rr_threshold_fraction,
                        flux_per_ray,
                        self.rng,
                        to_numpy(rays.ray_id),
                        to_numpy(rays.bounce),
                    )
                    if rr_killed_np.any():
                        num_rays_flux_killed += int(rr_killed_np.sum())
                        total_flux_rr_killed += float(
                            to_numpy(flux_before_rr)[rr_killed_np].sum()
                        )
                        path_recorder.log_deaths(rays, rr_killed_np, "flux_killed")

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
        det_names = get_detector_names(scene)
        for i, det in enumerate(scene.detectors):
            name = det_names[i] if i < len(det_names) else (det.name or f"detector_{i}")
            result = det.get_result()
            detector_results[name] = result
            if hasattr(result, "total_flux"):
                # IrradianceMap.total_flux may be an attached backend array;
                # SimulationResult's aggregate stays a plain float.
                total_flux_detected += float(to_numpy(result.total_flux))

        total_flux_lost = total_flux_depth_killed + total_flux_rr_killed

        # Every launched watt ends up detected, absorbed, escaped, or killed
        # by the flux/depth cutoffs. Omitting total_flux_lost makes the metric
        # report a large error for any scene that depth-kills rays, which is
        # exactly the stray-light case this diagnostic exists to serve.
        flux_err = (
            abs(
                total_flux_in
                - total_flux_detected
                - total_flux_absorbed
                - total_flux_bulk_absorbed
                - total_flux_escaped
                - total_flux_lost
            )
            / total_flux_in
            if total_flux_in > 0
            else 0.0
        )

        # Vectorised: single conversion of the columnar buffers
        # to the structured-array format (numpy, detached), done once here
        # rather than incrementally per event.
        ray_paths = path_recorder.finalize()

        # split_budget_saturated is always False here: bounded splitting
        # is the NumPy forward engine only.
        diagnostics = build_diagnostics(
            scene,
            hit_component_ids,
            num_rays_total,
            total_flux_in,
            total_flux_depth_killed,
            total_flux_rr_killed,
            flux_err,
            False,
            detector_results,
            medium_stack_underflows=total_medium_stack_underflows,
        )

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
            total_flux_bulk_absorbed=total_flux_bulk_absorbed,
            total_flux_escaped=total_flux_escaped,
            total_flux_lost=total_flux_lost,
            flux_conservation_error=flux_err,
            trace_time_sec=t_end - t_start,
            ray_paths=ray_paths,
            diagnostics=diagnostics,
        )

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
        rays.k_current = _to_float(rays.k_current)
        rays.alive = _to_bool(rays.alive)
        # bounce: keep as int32 tensor (used for depth comparisons)
        if not isinstance(rays.bounce, _torch.Tensor):
            rays.bounce = _torch.from_numpy(
                np.asarray(rays.bounce, dtype=np.int32).copy()
            )
        return rays
