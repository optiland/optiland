"""ArrayBackend -- base class for array-based tracing backends.

Provides the shared Monte Carlo trace loop for NumPy and Torch backends.

Kramer Harrison, 2026
"""

from __future__ import annotations

import time
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.backend.utils import to_numpy
from optiland.nonsequential._utils import DEFAULT_BATCH_SIZE
from optiland.nonsequential.backends.base import TracerBackend

if TYPE_CHECKING:
    from optiland.nonsequential.ray_bundle import NSQRayBundle
    from optiland.nonsequential.scene import NSQScene
    from optiland.nonsequential.tracer import SimulationResult

# Dtype for per-ray event records (always numpy structured array for display)
_EVENT_DTYPE = np.dtype(
    [
        ("ray_id", np.int64),
        ("event_type", "U10"),
        ("x", np.float64),
        ("y", np.float64),
        ("z", np.float64),
        ("L", np.float64),
        ("M", np.float64),
        ("N", np.float64),
        ("flux", np.float64),
        ("wavelength", np.float64),
        ("bounce", np.int32),
        ("component_name", "U64"),
    ]
)


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

    @abstractmethod
    def random_uniform(self, shape: tuple[int, ...]) -> np.ndarray:
        """Generate uniform random numbers on the backend device."""

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
        """Run the full Monte Carlo simulation.

        Args:
            scene: The NSQScene to simulate.
            num_rays: Total rays to launch.
            max_depth: Maximum surface hits per ray.
            min_flux_fraction: Kill threshold relative to per-ray initial flux.
            batch_size: Rays per processing batch. Does not change the result,
                only the speed; see ``DEFAULT_BATCH_SIZE``.
            seed: RNG seed for reproducibility.
            record_paths: If True, records per-ray event log.

        Returns:
            SimulationResult.
        """
        from optiland.nonsequential.components.absorbing import (
            AbsorbingComponent,  # noqa: PLC0415
        )
        from optiland.nonsequential.tracer import (
            SimulationResult,  # noqa: PLC0415, I001
        )

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

        # Per-ray event log (numpy-only, for display/visualization)
        event_log: list[dict] | None = [] if record_paths else None
        _next_ray_id: list[int] = [0]

        def _log_birth(rays: NSQRayBundle, source_name: str) -> None:
            if event_log is None:
                return
            n = rays.num_rays
            ids = np.arange(_next_ray_id[0], _next_ray_id[0] + n, dtype=np.int64)
            rays.ray_id = ids
            _next_ray_id[0] += n
            x_np = to_numpy(rays.x)
            y_np = to_numpy(rays.y)
            z_np = to_numpy(rays.z)
            L_np = to_numpy(rays.L)
            M_np = to_numpy(rays.M)
            N_np = to_numpy(rays.N)
            flux_np = to_numpy(rays.flux)
            wl_np = to_numpy(rays.wavelength)
            bounce_np = to_numpy(rays.bounce)
            for k in range(n):
                event_log.append(
                    {
                        "ray_id": int(ids[k]),
                        "event_type": "birth",
                        "x": float(x_np[k]),
                        "y": float(y_np[k]),
                        "z": float(z_np[k]),
                        "L": float(L_np[k]),
                        "M": float(M_np[k]),
                        "N": float(N_np[k]),
                        "flux": float(flux_np[k]),
                        "wavelength": float(wl_np[k]),
                        "bounce": int(bounce_np[k]),
                        "component_name": source_name,
                    }
                )

        def _log_hits(
            rays: NSQRayBundle,
            mask: np.ndarray,
            comp_name: str,
            t_offset: np.ndarray | None = None,
        ) -> None:
            if event_log is None or rays.ray_id is None:
                return
            mask_np = to_numpy(mask).astype(bool)
            idx = np.where(mask_np)[0]
            if len(idx) == 0:
                return
            ids = to_numpy(rays.ray_id)[idx]
            x_np = to_numpy(rays.x)[idx]
            y_np = to_numpy(rays.y)[idx]
            z_np = to_numpy(rays.z)[idx]
            L_np = to_numpy(rays.L)[idx]
            M_np = to_numpy(rays.M)[idx]
            N_np = to_numpy(rays.N)[idx]
            if t_offset is not None:
                t_np = to_numpy(t_offset)[idx]
                x_np = x_np + t_np * L_np
                y_np = y_np + t_np * M_np
                z_np = z_np + t_np * N_np
            flux_np = to_numpy(rays.flux)[idx]
            wl_np = to_numpy(rays.wavelength)[idx]
            bounce_np = to_numpy(rays.bounce)[idx]
            for k in range(len(idx)):
                event_log.append(
                    {
                        "ray_id": int(ids[k]),
                        "event_type": "hit",
                        "x": float(x_np[k]),
                        "y": float(y_np[k]),
                        "z": float(z_np[k]),
                        "L": float(L_np[k]),
                        "M": float(M_np[k]),
                        "N": float(N_np[k]),
                        "flux": float(flux_np[k]),
                        "wavelength": float(wl_np[k]),
                        "bounce": int(bounce_np[k]),
                        "component_name": comp_name,
                    }
                )

        def _log_deaths(rays: NSQRayBundle, mask: np.ndarray, cause: str) -> None:
            if event_log is None or rays.ray_id is None:
                return
            mask_np = to_numpy(mask).astype(bool)
            idx = np.where(mask_np)[0]
            if len(idx) == 0:
                return
            ids = to_numpy(rays.ray_id)[idx]
            x_np = to_numpy(rays.x)[idx]
            y_np = to_numpy(rays.y)[idx]
            z_np = to_numpy(rays.z)[idx]
            L_np = to_numpy(rays.L)[idx]
            M_np = to_numpy(rays.M)[idx]
            N_np = to_numpy(rays.N)[idx]
            flux_np = to_numpy(rays.flux)[idx]
            wl_np = to_numpy(rays.wavelength)[idx]
            bounce_np = to_numpy(rays.bounce)[idx]
            for k in range(len(idx)):
                event_log.append(
                    {
                        "ray_id": int(ids[k]),
                        "event_type": "death",
                        "x": float(x_np[k]),
                        "y": float(y_np[k]),
                        "z": float(z_np[k]),
                        "L": float(L_np[k]),
                        "M": float(M_np[k]),
                        "N": float(N_np[k]),
                        "flux": float(flux_np[k]),
                        "wavelength": float(wl_np[k]),
                        "bounce": int(bounce_np[k]),
                        "component_name": cause,
                    }
                )

        # Main trace loop
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

                _log_birth(rays, source_name)

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

                    # Rays that reach no detector carry t = inf. Zero those
                    # before multiplying by a direction: inf * 0 is NaN, which
                    # the be.where below discards but not before NumPy warns.
                    det_t_safe = np.where(det_first, det_t_min, 0.0)

                    # Record detector hits
                    for di, det in enumerate(scene.detectors):
                        mask_di = det_first & (det_idx == di)
                        if mask_di.any():
                            det_name = getattr(det, "name", f"detector_{di}")
                            _log_hits(rays, mask_di, det_name, t_offset=det_t_safe)
                            det.record(rays, det_t_safe, mask_di)

                    # Advance and kill detector-hit rays
                    if det_first.any():
                        dx = det_t_safe * rays.L
                        dy = det_t_safe * rays.M
                        dz = det_t_safe * rays.N
                        rays.x = be.where(det_first, rays.x + dx, rays.x)
                        rays.y = be.where(det_first, rays.y + dy, rays.y)
                        rays.z = be.where(det_first, rays.z + dz, rays.z)
                        rays.alive = rays.alive & ~det_first
                        rays.bounce = be.where(det_first, rays.bounce + 1, rays.bounce)

                    # Apply component interactions
                    for ci, comp in enumerate(scene.surfaces):
                        mask_ci = comp_first & (comp_idx == ci)
                        if mask_ci.any():
                            comp_name = getattr(comp, "name", f"comp_{ci}")
                            _log_hits(rays, mask_ci, comp_name, t_offset=t_min)
                            comp.interact(rays, t_min, hit_normals, mask_ci, self.rng)

                    # Kill rays with no hit (escaped)
                    no_hit = ~any_comp_hit & ~any_det_hit
                    escaped_now = no_hit & rays.alive
                    if escaped_now.any():
                        num_rays_escaped += int(escaped_now.sum())
                        total_flux_escaped += float(
                            to_numpy(rays.flux[escaped_now]).sum()
                        )
                        _log_deaths(rays, escaped_now, "escaped")
                        bounding_scale = self._estimate_bounding_scale(scene)
                        ex = bounding_scale * rays.L
                        ey = bounding_scale * rays.M
                        ez = bounding_scale * rays.N
                        rays.x = be.where(escaped_now, rays.x + ex, rays.x)
                        rays.y = be.where(escaped_now, rays.y + ey, rays.y)
                        rays.z = be.where(escaped_now, rays.z + ez, rays.z)
                    rays.alive = rays.alive & ~no_hit

                    # Kill rays below flux threshold or exceeding max depth
                    alive_flux = rays.flux >= min_flux
                    alive_depth = rays.bounce < max_depth

                    newly_flux_killed = rays.alive & ~alive_flux
                    newly_depth_killed = rays.alive & alive_flux & ~alive_depth

                    if newly_flux_killed.any():
                        num_rays_flux_killed += int(newly_flux_killed.sum())
                        total_flux_lost += float(
                            to_numpy(rays.flux[newly_flux_killed]).sum()
                        )
                        _log_deaths(rays, newly_flux_killed, "flux_killed")

                    if newly_depth_killed.any():
                        num_rays_depth_killed += int(newly_depth_killed.sum())
                        total_flux_lost += float(
                            to_numpy(rays.flux[newly_depth_killed]).sum()
                        )
                        _log_deaths(rays, newly_depth_killed, "depth_killed")

                    rays.alive = rays.alive & alive_flux & alive_depth

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

        # Build ray_paths from event log
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find nearest detector intersection for each ray.

        Returns:
            ``(t_min, hit_normals, detector_indices)``
        """
        N = rays.num_rays
        t_min = np.full(N, np.inf, dtype=np.float64)
        hit_normals = np.zeros((N, 3), dtype=np.float64)
        det_indices = np.full(N, -1, dtype=np.int32)

        for i, det in enumerate(detectors):
            t_d, normals_d, hit_d = det.intersect(rays)
            # Convert to numpy for the index-comparison logic (no grad needed)
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

            extent = np.sqrt(
                (xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2
            )
            return float(extent) if extent > 1.0 else 100.0
        except Exception:
            return 100.0

    def _to_numpy(self, arr: object) -> np.ndarray:
        """Backward-compatible alias."""
        return to_numpy(arr)


def _get_detector_names(scene: object) -> list[str]:
    """Extract registry names for detectors.

    Args:
        scene: NSQScene instance.

    Returns:
        Ordered list of detector names.
    """
    try:
        return list(scene.detector_registry._registry.keys())  # type: ignore[attr-defined]
    except AttributeError:
        return []
