"""Sequential ray-tracing throughput benchmark behind the JOSS paper's Table 1.

This is a direct port of the manual procedure used to produce the numbers in
``paper/paper.md`` (originally run interactively in
``optiland/jupyter/JOSS_NumPy_vs_PyTorch_Speed_Check.ipynb``): trace a fixed
random distribution of rays through a Cooke triplet once per backend
configuration and time only the ``surface_group.trace()`` call. Ray
generation, backend setup, and any warm-up are excluded from the timed
region.

Each configuration is measured with a *single* trace call, matching how the
published table was produced -- this script does not average repeated runs.
Reviewers/readers have measured 4-60% run-to-run spread depending on machine
load and configuration, so treat single-run numbers as illustrative rather
than precise; pass ``--repeats`` to see that spread directly on your own
hardware.

Not run as part of the pytest suite or CI -- see ``pyproject.toml``
(``testpaths = ["tests"]``). Run directly::

    .venv/Scripts/python.exe -m benchmarks.sequential.cooke_triplet
    .venv/Scripts/python.exe -m benchmarks.sequential.cooke_triplet --repeats 5
    .venv/Scripts/python.exe -m benchmarks.sequential.cooke_triplet \
        --output results.json

Kramer Harrison, 2026
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import time

import optiland.backend as be
from optiland import distribution
from optiland.samples.objectives import CookeTriplet

# Matches the notebook: 10M rays for the float64 configurations, 20M for the
# float32 GPU configuration (float32 traces fast enough that 10M rays no
# longer dominates fixed per-call overhead).
DEFAULT_NUM_RAYS = 10_000_000
DEFAULT_NUM_RAYS_FLOAT32 = 20_000_000
DEFAULT_SEED = 0

CONFIGS = [
    # (label, backend, precision, device)
    ("numpy_cpu_f64", "numpy", None, None),
    ("torch_cpu_f64", "torch", "float64", "cpu"),
    ("torch_gpu_f64", "torch", "float64", "cuda"),
    ("torch_gpu_f32", "torch", "float32", "cuda"),
]


@dataclasses.dataclass(frozen=True)
class BenchmarkResult:
    """One backend configuration's measured throughput.

    Attributes:
        label: Configuration name, e.g. "torch_gpu_f32".
        num_rays: Rays traced.
        num_surfaces: Surfaces in the Cooke triplet (including object/image).
        trace_times_sec: One elapsed time per repeat, in seconds.
        rays_surfaces_per_sec: ``num_rays * num_surfaces / time`` per repeat.
    """

    label: str
    num_rays: int
    num_surfaces: int
    trace_times_sec: list[float]
    rays_surfaces_per_sec: list[float]


def _build_rays(num_rays: int, seed: int):
    """Build a Cooke triplet and a fixed random distribution of rays for it.

    Args:
        num_rays: Number of rays to generate across the pupil.
        seed: Seed for the pupil-sampling RNG, for reproducibility.

    Returns:
        A ``(lens, rays)`` tuple, ready for ``lens.surface_group.trace(rays)``.
    """
    lens = CookeTriplet()
    d = distribution.RandomDistribution(seed=seed)
    d.generate_points(num_rays)
    rays = lens.ray_tracer.ray_generator.generate_rays(
        Hx=0, Hy=0, Px=d.x, Py=d.y, wavelength=0.55
    )
    return lens, rays


def run_config(
    label: str,
    backend: str,
    precision: str | None,
    device: str | None,
    num_rays: int,
    seed: int,
    repeats: int,
) -> BenchmarkResult | None:
    """Time ``repeats`` trace() calls for one backend configuration.

    Args:
        label: Configuration name for reporting.
        backend: "numpy" or "torch".
        precision: "float32"/"float64" (torch only) or None (numpy).
        device: "cpu"/"cuda" (torch only) or None (numpy).
        num_rays: Rays to trace.
        seed: Pupil-sampling RNG seed (rays are regenerated per repeat so
            each repeat has identical inputs).
        repeats: Number of independent trace() calls to time.

    Returns:
        A BenchmarkResult, or None if this configuration is unavailable
        (e.g. no CUDA device).
    """
    original_backend = be.get_backend()
    try:
        be.set_backend(backend)
        if precision is not None:
            be.set_precision(precision)
        if device == "cuda":
            try:
                import torch  # noqa: PLC0415

                if not torch.cuda.is_available():
                    print(f"skipping {label}: CUDA not available", file=sys.stderr)
                    return None
            except ImportError:
                print(f"skipping {label}: PyTorch not installed", file=sys.stderr)
                return None
            be.set_device("cuda")
        elif device is not None:
            be.set_device(device)

        times = []
        num_surfaces = None
        for _ in range(repeats):
            lens, rays = _build_rays(num_rays, seed)
            start = time.perf_counter()
            lens.surfaces.trace(rays)
            times.append(time.perf_counter() - start)
            num_surfaces = len(lens.surfaces.surfaces)

            # Free this repeat's tensors before building the next one --
            # otherwise repeated float64 GPU traces exhaust an 8 GB card.
            del lens, rays
            if device == "cuda":
                import torch  # noqa: PLC0415

                torch.cuda.empty_cache()
        throughput = [num_rays * num_surfaces / t / 1e6 for t in times]
        return BenchmarkResult(
            label=label,
            num_rays=num_rays,
            num_surfaces=num_surfaces,
            trace_times_sec=times,
            rays_surfaces_per_sec=throughput,
        )
    finally:
        be.set_backend(original_backend)


def _print_table(results: list[BenchmarkResult]) -> None:
    """Print measured throughput as a plain-text table.

    Args:
        results: Results to print, in the order given.
    """
    header = f"{'config':<16}{'rays':>12}{'repeats':>9}{'M ray-surf/s (best)':>22}"
    print(header)
    print("-" * len(header))
    for r in results:
        best = max(r.rays_surfaces_per_sec)
        print(f"{r.label:<16}{r.num_rays:>12}{len(r.trace_times_sec):>9}{best:>22.2f}")
        if len(r.trace_times_sec) > 1:
            spread = statistics.pstdev(r.rays_surfaces_per_sec) / statistics.mean(
                r.rays_surfaces_per_sec
            )
            print(
                f"    all: {[round(v, 2) for v in r.rays_surfaces_per_sec]} "
                f"(mean={statistics.mean(r.rays_surfaces_per_sec):.2f}, "
                f"spread={spread:.1%})"
            )


def main(argv: list[str] | None = None) -> None:
    """Run all four Table 1 configurations and print/save the results.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-rays",
        type=int,
        default=DEFAULT_NUM_RAYS,
        help=f"Rays for the float64 configurations (default: {DEFAULT_NUM_RAYS:,}).",
    )
    parser.add_argument(
        "--num-rays-float32",
        type=int,
        default=DEFAULT_NUM_RAYS_FLOAT32,
        help=(
            "Rays for the torch_gpu_f32 configuration "
            f"(default: {DEFAULT_NUM_RAYS_FLOAT32:,})."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="trace() calls to time per configuration (default: 1, matching "
        "the originally published measurement).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output", type=str, default=None, help="Path to write JSON results."
    )
    args = parser.parse_args(argv)

    results: list[BenchmarkResult] = []
    for label, backend, precision, device in CONFIGS:
        num_rays = args.num_rays_float32 if precision == "float32" else args.num_rays
        result = run_config(
            label, backend, precision, device, num_rays, args.seed, args.repeats
        )
        if result is not None:
            results.append(result)

    _print_table(results)

    if results:
        baseline = results[0].rays_surfaces_per_sec[0]
        print("\nRelative speedup vs numpy_cpu_f64 (first repeat):")
        for r in results:
            print(f"  {r.label:<16}{r.rays_surfaces_per_sec[0] / baseline:>8.1f}x")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump([dataclasses.asdict(r) for r in results], f, indent=2)
        print(f"\nWrote {len(results)} records to {args.output}")


if __name__ == "__main__":
    main()
