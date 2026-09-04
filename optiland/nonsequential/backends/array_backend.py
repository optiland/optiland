"""ArrayBackend -- base class for array-based tracing backends.

Provides the shared Monte Carlo trace loop for NumPy and Torch backends.

Kramer Harrison, 2026
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

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
from optiland.nonsequential.path_recording import (  # noqa: F401
    _EVENT_DTYPE,
    PathRecorder,
)
from optiland.nonsequential.ray_bundle import NSQRayBundle
from optiland.nonsequential.rng import EventSlot
from optiland.nonsequential.sampling import russian_roulette

if TYPE_CHECKING:
    from optiland.nonsequential.scene import NSQScene
    from optiland.nonsequential.tracer import SimulationResult


# Floor on the split-budget culling survival probability, mirroring
# sampling._RR_SURVIVE_FLOOR: bounds the worst-case flux boost so a nearly
# -saturated budget cannot produce an arbitrarily large boosted flux.
_BUDGET_CULL_SURVIVE_FLOOR = 0.02


def _cull_to_budget(
    spawned: NSQRayBundle, headroom: int, rng
) -> tuple[NSQRayBundle, np.ndarray, np.ndarray]:
    """Russian-roulette a spawned batch down to ``headroom`` rays.

    Bounded splitting (``ir.sampling.split_depth > 0``) caps live rays at
    ``split_budget * batch_size``; a batch of spawned transmit children that
    would exceed the remaining headroom is culled by roulette rather than
    truncated, so the excess is an unbiased kill + boost (like
    :func:`optiland.nonsequential.sampling.russian_roulette`) instead of a
    silent flux-truncation bias.

    Args:
        spawned: The spawned-ray batch to cull (already smaller-than
            -headroom batches must not be passed in -- callers check
            ``spawned.num_rays > headroom`` first).
        headroom: Number of additional live rays the budget can still admit.
            May be 0 (budget already saturated).
        rng: Keyed PCG32 RNG.

    Returns:
        ``(kept, culled_flux, culled_mask)``: the surviving (boosted-flux)
        subset as a new bundle, the pre-cull flux of the rays that were
        killed (for ``total_flux_lost`` bookkeeping), and the NumPy bool
        mask of which input rows were culled.
    """
    n = spawned.num_rays
    keep_prob = max(headroom / n, _BUDGET_CULL_SURVIVE_FLOOR) if n > 0 else 1.0
    ray_id_np = to_numpy(spawned.ray_id)
    bounce_np = to_numpy(spawned.bounce)
    u = rng.uniform(ray_id_np, bounce_np, EventSlot.RR, offset=1)
    keep_np = u < keep_prob
    culled_np = ~keep_np

    flux_np = to_numpy(spawned.flux)
    culled_flux = flux_np[culled_np].copy()

    idx = np.where(keep_np)[0]
    kept = spawned.select(idx)
    kept.flux = kept.flux / keep_prob  # unbiased boost
    return kept, culled_flux, culled_np


class ArrayBackend(TracerBackend):
    """Abstract base class for array-based tracing backends."""

    def _maybe_compact(self, rays: NSQRayBundle) -> NSQRayBundle:
        """Post-bounce hook: optionally compact dead rays from the bundle.

        Default is a no-op (TorchBackend keeps fixed-shape tensors for the
        autograd graph). NumpyBackend overrides this to call rays.compact(),
        removing dead rays before each subsequent intersection test.

        Args:
            rays: Current ray bundle.

        Returns:
            Possibly compacted ray bundle.
        """
        return rays

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
        """Run the full Monte Carlo simulation.

        Args:
            scene: The NSQScene to simulate.
            num_rays: Total rays to launch.
            max_depth: Maximum surface hits per ray.
            min_flux_fraction: Russian-roulette threshold, relative to
                per-ray initial flux -- combined with the scene's
                ``sampling_policy.rr_start_flux`` (the larger of the two
                wins). Below threshold, rays are killed with an unbiased
                probability and survivors' flux is boosted accordingly,
                rather than truncated outright.
            batch_size: Rays per processing batch. Does not change the result,
                only the speed; see ``DEFAULT_BATCH_SIZE``.
            seed: RNG seed for reproducibility.
            record_paths: ``False`` (default) records nothing. ``True``
                records every ray's full path -- fine for small traces, but
                O(rays x bounces) memory for large ones. A positive ``int``
                records an approximately that-many-ray subset, selected by a
                PCG32 hash of ``ray_id`` so the trace stays
                full-size and cheap while a bounded, deterministic sample is
                available for visualization/diagnosis -- e.g.
                ``scene.trace(num_rays=10_000_000, record_paths=1_000)``.

        Returns:
            SimulationResult.
        """
        from optiland.nonsequential.components.absorbing import (
            AbsorbingComponent,  # noqa: PLC0415
        )
        from optiland.nonsequential.rng import NSQRng  # noqa: PLC0415
        from optiland.nonsequential.tracer import (
            SimulationResult,  # noqa: PLC0415, I001
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
        # Tracked separately for Diagnostics: depth truncation
        # is an inherent, reported bias, while RR/split-budget culling is
        # unbiased in expectation -- conflating them into one total_flux_lost
        # would hide which mechanism a large loss actually came from.
        total_flux_depth_killed = 0.0
        total_flux_rr_killed = 0.0
        hit_component_ids: set[int] = set()
        split_budget_saturated = False
        total_medium_stack_underflows = 0

        # Distribute ray budget across sources proportional to flux
        rays_per_source = distribute_ray_budget(
            num_rays_total, [float(s.total_flux) for s in sources]
        )

        # Vectorised columnar path recording: PathRecorder
        # replaces the old per-event Python dict + repeated to_numpy()
        # closures with preallocated array writes, and implements the
        # record_paths: int subset contract.
        path_recorder = PathRecorder(record_paths, num_rays_total, self.rng.seed)

        _next_ray_id: list[int] = [0]

        def _alloc_ray_ids(n: int) -> np.ndarray:
            """Allocate ``n`` fresh ray ids for bounded splitting.

            Shares the same monotonic counter as source-birth ray ids, so
            a spawned ray's id never collides with any other ray's -- its
            PCG32 stream (keyed by ray_id) is therefore independent.
            """
            start = _next_ray_id[0]
            _next_ray_id[0] += n
            return np.arange(start, start + n, dtype=np.int64)

        # Main trace loop
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

                path_recorder.log_birth(rays, source_name)

                while rays.num_rays_alive > 0:
                    # Component intersections
                    t_min, hit_normals, comp_idx, hit_n_geom = self.intersect_scene(
                        rays, scene.surfaces
                    )

                    # Detector intersections (shared with TorchBackend; D-10)
                    det_t_min, det_normals, det_idx = intersect_detectors(
                        rays, scene.detectors
                    )

                    # Nearest hit: component vs detector
                    comp_closer = t_min <= det_t_min
                    any_comp_hit = comp_idx >= 0
                    any_det_hit = det_idx >= 0

                    det_first = any_det_hit & (~comp_closer | ~any_comp_hit)
                    comp_first = any_comp_hit & (~det_first)

                    # unreached_geometry: cheap running set of
                    # every primitive that was ever the nearest hit.
                    if comp_first.any():
                        hit_component_ids.update(
                            np.unique(comp_idx[comp_first]).tolist()
                        )

                    # Rays that reach no detector carry t = inf. Zero those
                    # before multiplying by a direction: inf * 0 is NaN, which
                    # the be.where below discards but not before NumPy warns.
                    det_t_safe = np.where(det_first, det_t_min, 0.0)

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
                    hit_first = comp_first | det_first
                    if hit_first.any():
                        comp_t_safe = be.where(
                            be.array(comp_first), t_min, be.zeros_like(t_min)
                        )
                        hit_t = be.where(
                            be.array(comp_first), comp_t_safe, be.array(det_t_safe)
                        )
                        alpha = 4.0 * be.pi * rays.k_current / rays.wavelength
                        # hit_t is in mm; alpha is in 1/um -> convert to um.
                        transmittance = be.exp(-alpha * hit_t * 1e3)
                        flux_before = rays.flux
                        rays.flux = flux_before * be.where(
                            be.array(hit_first), transmittance, be.ones_like(rays.flux)
                        )
                        total_flux_bulk_absorbed += float(
                            to_numpy(flux_before - rays.flux).sum()
                        )

                    # Record detector hits
                    for di, det in enumerate(scene.detectors):
                        mask_di = det_first & (det_idx == di)
                        if mask_di.any():
                            det_name = getattr(det, "name", f"detector_{di}")
                            path_recorder.log_hits(
                                rays, mask_di, det_name, t_offset=det_t_safe
                            )
                            det.record(rays, det_t_safe, mask_di)

                    # Advance detector-hit rays. Absorbing detectors
                    # terminate the ray; absorb=False detectors are
                    # transmissive: the hit is recorded (above) and the ray
                    # continues on its unchanged direction.
                    if det_first.any():
                        dx = det_t_safe * rays.L
                        dy = det_t_safe * rays.M
                        dz = det_t_safe * rays.N
                        rays.x = be.where(det_first, rays.x + dx, rays.x)
                        rays.y = be.where(det_first, rays.y + dy, rays.y)
                        rays.z = be.where(det_first, rays.z + dz, rays.z)
                        rays.bounce = be.where(det_first, rays.bounce + 1, rays.bounce)

                        absorb_np = detector_absorb_mask(det_idx, scene.detectors)
                        kill_np = np.asarray(det_first) & absorb_np
                        rays.alive = rays.alive & ~be.array(kill_np)

                    # Apply component interactions, dispatched from the IR
                    # (ir.primitives[i].component_kind / .bsdf.kind) rather
                    # than by iterating scene.surfaces and checking isinstance.
                    # ray_id_allocator enables bounded splitting (D2, PR11,
                    # NumPy forward engine only): a hit ray below
                    # ir.sampling.split_depth spawns both Fresnel children
                    # instead of drawing one, and the transmit child comes
                    # back as spawned (merged into `rays` below, after this
                    # bounce's own kill checks -- see the merge comment).
                    spawned = apply_primitive_interactions(
                        rays,
                        ir,
                        scene.surfaces,
                        t_min,
                        hit_normals,
                        hit_n_geom,
                        comp_idx,
                        comp_first,
                        self.rng,
                        log_hit_fn=path_recorder.log_hits,
                        ray_id_allocator=_alloc_ray_ids,
                    )

                    # Kill rays with no hit (escaped)
                    no_hit = ~any_comp_hit & ~any_det_hit
                    escaped_now = no_hit & rays.alive
                    if escaped_now.any():
                        num_rays_escaped += int(escaped_now.sum())
                        total_flux_escaped += float(
                            to_numpy(rays.flux[escaped_now]).sum()
                        )
                        path_recorder.log_deaths(rays, escaped_now, "escaped")
                        bounding_scale = estimate_bounding_scale(scene)
                        ex = bounding_scale * rays.L
                        ey = bounding_scale * rays.M
                        ez = bounding_scale * rays.N
                        rays.x = be.where(escaped_now, rays.x + ex, rays.x)
                        rays.y = be.where(escaped_now, rays.y + ey, rays.y)
                        rays.z = be.where(escaped_now, rays.z + ez, rays.z)
                    rays.alive = rays.alive & ~no_hit

                    # Depth truncation: hard kill. Inherent, reported bias
                    # (unlike the old flux truncation below, this is not
                    # replaced by roulette -- there is no unbiased way to
                    # "continue" a ray past a hard bounce-count cap).
                    alive_depth = rays.bounce < max_depth
                    newly_depth_killed = rays.alive & ~alive_depth
                    if newly_depth_killed.any():
                        num_rays_depth_killed += int(newly_depth_killed.sum())
                        total_flux_depth_killed += float(
                            to_numpy(rays.flux[newly_depth_killed]).sum()
                        )
                        path_recorder.log_deaths(
                            rays, newly_depth_killed, "depth_killed"
                        )
                    rays.alive = rays.alive & alive_depth

                    # Russian roulette replaces the old biased hard
                    # kill below min_flux: unbiased stochastic termination
                    # of low-flux rays (kill with probability p, boost
                    # survivors by 1/(1-p)), so total_flux_lost now reports
                    # a genuine diagnostic -- ~0 for a well-configured scene
                    # -- rather than an expected bookkeeping entry.
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

                    # Merge bounded-splitting spawned transmit
                    # children into the live bundle, now that this bounce's
                    # own escape/depth/RR kill checks (all sized to the
                    # pre-spawn ray count) are done. Spawned rays start
                    # fresh at the next while-loop iteration's
                    # intersect_scene call, same as any other live ray.
                    if spawned is not None and spawned.num_rays > 0:
                        budget = int(ir.sampling.split_budget * batch_size)
                        headroom = max(0, budget - rays.num_rays_alive)
                        if spawned.num_rays > headroom:
                            split_budget_saturated = True
                            spawned, culled_flux_np, culled_np = _cull_to_budget(
                                spawned, headroom, self.rng
                            )
                            if culled_np.any():
                                num_rays_flux_killed += int(culled_np.sum())
                                total_flux_rr_killed += float(culled_flux_np.sum())
                        if spawned.num_rays > 0:
                            rays = NSQRayBundle.concat([rays, spawned])

                    # D1: flush this bounce's medium-stack underflow counts
                    # (see RefractiveComponent.interact) into the running
                    # total, then reset so they are counted exactly once
                    # regardless of subsequent compaction/concat.
                    total_medium_stack_underflows += int(
                        rays.medium_stack_underflows.sum()
                    )
                    rays.medium_stack_underflows[:] = 0

                    rays = self._maybe_compact(rays)
                    if rays.num_rays == 0:
                        break

                source_remaining -= batch

        t_end = time.perf_counter()

        # Collect absorbed stats from AbsorbingComponents
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
        # to the structured-array format, done once here rather than
        # incrementally per event.
        ray_paths = path_recorder.finalize()

        diagnostics = build_diagnostics(
            scene,
            hit_component_ids,
            num_rays_total,
            total_flux_in,
            total_flux_depth_killed,
            total_flux_rr_killed,
            flux_err,
            split_budget_saturated,
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

    def _to_numpy(self, arr: object) -> np.ndarray:
        """Backward-compatible alias."""
        return to_numpy(arr)
