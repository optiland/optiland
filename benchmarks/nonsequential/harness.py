"""Timing harness for the NSQ Monte Carlo tracer.

Runs a scene through a chosen backend and reports rays/sec, using
``SimulationResult.trace_time_sec`` (backend-measured wall-clock trace time,
excluding scene construction) as the timed quantity. This module is
import-safe without PyTorch installed -- the torch backend is only touched
when explicitly requested.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import optiland.backend as be

if TYPE_CHECKING:
    from optiland.nonsequential.scene import NSQScene

BackendName = Literal["numpy", "torch"]


@dataclass(frozen=True)
class BenchmarkRecord:
    """One timed trace() call.

    Attributes:
        axis: Which axis this record belongs to ("surfaces", "rays", or
            "depth").
        backend: "numpy" or "torch".
        num_surfaces: Number of components in the scene.
        num_rays: Rays launched.
        max_depth: Max bounces allowed.
        trace_time_sec: Backend-measured wall-clock trace time [s].
        rays_per_sec: ``num_rays / trace_time_sec``.
    """

    axis: str
    backend: BackendName
    num_surfaces: int
    num_rays: int
    max_depth: int
    trace_time_sec: float
    rays_per_sec: float


def _make_backend(name: BackendName, seed: int):
    """Instantiate a TracerBackend, activating the matching optiland backend.

    Args:
        name: "numpy" or "torch".
        seed: RNG seed forwarded to the backend.

    Returns:
        A NumpyBackend or TorchBackend instance.

    Raises:
        ImportError: If ``name == "torch"`` and PyTorch is not installed.
    """
    if name == "numpy":
        from optiland.nonsequential.backends.numpy_backend import (  # noqa: PLC0415
            NumpyBackend,
        )

        be.set_backend("numpy")
        return NumpyBackend(seed=seed)

    if name == "torch":
        from optiland.nonsequential.backends.torch_backend import (  # noqa: PLC0415
            TorchBackend,
        )

        be.set_backend("torch")
        return TorchBackend(seed=seed)

    raise ValueError(f"Unknown backend name: {name!r}. Expected 'numpy' or 'torch'.")


def run_once(
    axis: str,
    scene: NSQScene,
    num_rays: int,
    backend_name: BackendName,
    max_depth: int = 16,
    seed: int = 0,
) -> BenchmarkRecord:
    """Trace ``scene`` once on the given backend and record its throughput.

    Args:
        axis: Label for which sweep this call belongs to.
        scene: Scene to trace.
        num_rays: Rays to launch.
        backend_name: "numpy" or "torch".
        max_depth: Max bounces per ray.
        seed: RNG seed.

    Returns:
        BenchmarkRecord for this run.
    """
    original_backend = be.get_backend()
    try:
        tracer_backend = _make_backend(backend_name, seed)
        result = scene.trace(
            num_rays=num_rays,
            max_depth=max_depth,
            seed=seed,
            backend=tracer_backend,
        )
    finally:
        be.set_backend(original_backend)

    trace_time = max(result.trace_time_sec, 1e-12)
    return BenchmarkRecord(
        axis=axis,
        backend=backend_name,
        num_surfaces=len(scene.surfaces),
        num_rays=num_rays,
        max_depth=max_depth,
        trace_time_sec=trace_time,
        rays_per_sec=num_rays / trace_time,
    )


def sweep_surface_count(
    surface_counts: list[int],
    num_rays: int,
    backend_names: list[BackendName],
) -> list[BenchmarkRecord]:
    """Time :func:`scenes.surface_count_scene` across a range of sizes.

    Args:
        surface_counts: Decoy-mirror counts to try.
        num_rays: Rays launched per trace.
        backend_names: Backends to run, e.g. ``["numpy", "torch"]``.

    Returns:
        One BenchmarkRecord per (surface_count, backend) pair.
    """
    from benchmarks.nonsequential.scenes import surface_count_scene  # noqa: PLC0415

    records = []
    for n in surface_counts:
        scene = surface_count_scene(n)
        for backend_name in backend_names:
            records.append(run_once("surfaces", scene, num_rays, backend_name))
    return records


def sweep_ray_count(
    ray_counts: list[int],
    backend_names: list[BackendName],
) -> list[BenchmarkRecord]:
    """Time :func:`scenes.lens_scene` across a range of ray counts.

    Args:
        ray_counts: Ray counts to try.
        backend_names: Backends to run.

    Returns:
        One BenchmarkRecord per (ray_count, backend) pair.
    """
    from benchmarks.nonsequential.scenes import lens_scene  # noqa: PLC0415

    scene = lens_scene()
    records = []
    for n in ray_counts:
        for backend_name in backend_names:
            records.append(run_once("rays", scene, n, backend_name))
    return records


def sweep_depth(
    depths: list[int],
    num_rays: int,
    backend_names: list[BackendName],
) -> list[BenchmarkRecord]:
    """Time :func:`scenes.cavity_scene` across a range of ``max_depth``.

    Args:
        depths: ``max_depth`` values to try.
        num_rays: Rays launched per trace.
        backend_names: Backends to run.

    Returns:
        One BenchmarkRecord per (depth, backend) pair.
    """
    from benchmarks.nonsequential.scenes import cavity_scene  # noqa: PLC0415

    scene = cavity_scene()
    records = []
    for depth in depths:
        for backend_name in backend_names:
            records.append(
                run_once("depth", scene, num_rays, backend_name, max_depth=depth)
            )
    return records
