"""CLI entry point for the NSQ performance benchmark harness.

Usage::

    .venv/Scripts/python.exe -m benchmarks.nonsequential.run
    .venv/Scripts/python.exe -m benchmarks.nonsequential.run --output results.json
    .venv/Scripts/python.exe -m benchmarks.nonsequential.run --backends numpy

Not run as part of the pytest suite -- see ``tests/nonsequential/validation/``
for the CI-gated correctness suite. This script only measures throughput and
prints/saves the results as a baseline for future acceleration work.

Kramer Harrison, 2026
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from benchmarks.nonsequential.harness import (
    BackendName,
    BenchmarkRecord,
    sweep_depth,
    sweep_ray_count,
    sweep_surface_count,
)

DEFAULT_SURFACE_COUNTS = [0, 1, 2, 5, 10, 20, 50]
DEFAULT_RAY_COUNTS = [1_000, 10_000, 100_000, 1_000_000]
DEFAULT_DEPTHS = [2, 4, 8, 16, 32]
DEFAULT_SWEEP_RAYS = 100_000


def _available_backends(requested: list[str]) -> list[BackendName]:
    """Filter ``requested`` down to backends importable in this environment.

    Args:
        requested: Backend names the caller asked for.

    Returns:
        Subset of ``requested`` that can actually be constructed. "torch" is
        dropped (with a warning to stderr) if PyTorch is not installed.
    """
    available: list[BackendName] = []
    for name in requested:
        if name == "torch":
            try:
                import torch  # noqa: F401, PLC0415
            except ImportError:
                print(
                    "warning: PyTorch not installed -- skipping torch backend",
                    file=sys.stderr,
                )
                continue
        available.append(name)  # type: ignore[arg-type]
    return available


def _print_table(records: list[BenchmarkRecord]) -> None:
    """Print benchmark records as a plain-text table.

    Args:
        records: Records to print, in the order given.
    """
    header = (
        f"{'axis':<10}{'backend':<8}{'surfaces':>10}{'rays':>12}"
        f"{'depth':>7}{'time_s':>10}{'rays/s':>14}"
    )
    print(header)
    print("-" * len(header))
    for r in records:
        print(
            f"{r.axis:<10}{r.backend:<8}{r.num_surfaces:>10}{r.num_rays:>12}"
            f"{r.max_depth:>7}{r.trace_time_sec:>10.4f}{r.rays_per_sec:>14,.0f}"
        )


def main(argv: list[str] | None = None) -> None:
    """Run all three sweeps and print/save the results.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["numpy", "torch"],
        choices=["numpy", "torch"],
        help="Backends to benchmark (default: numpy torch).",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to write JSON results."
    )
    parser.add_argument(
        "--surface-counts",
        nargs="+",
        type=int,
        default=DEFAULT_SURFACE_COUNTS,
    )
    parser.add_argument("--ray-counts", nargs="+", type=int, default=DEFAULT_RAY_COUNTS)
    parser.add_argument("--depths", nargs="+", type=int, default=DEFAULT_DEPTHS)
    parser.add_argument("--sweep-rays", type=int, default=DEFAULT_SWEEP_RAYS)
    args = parser.parse_args(argv)

    backends = _available_backends(args.backends)
    if not backends:
        print("No requested backend is available; nothing to run.", file=sys.stderr)
        sys.exit(1)

    records: list[BenchmarkRecord] = []
    records += sweep_surface_count(args.surface_counts, args.sweep_rays, backends)
    records += sweep_ray_count(args.ray_counts, backends)
    records += sweep_depth(args.depths, args.sweep_rays, backends)

    _print_table(records)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump([dataclasses.asdict(r) for r in records], f, indent=2)
        print(f"\nWrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
