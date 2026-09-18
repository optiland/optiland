"""Measure how many worker processes saturate the CPU and the GPU.

Each job traces the Cooke triplet ``--rays`` times ``--repeat`` in a worker
process configured via ``optiland.parallel`` (spawn context). Every job records
wall-clock start/end stamps, so the report separates pool start-up (interpreter
spawn, torch import, Metal compile) from steady-state throughput:

* ``wall_s``: total ``evaluate_parallel`` time including start-up;
* ``span_s``: first job start to last job end (the compute phase);
* ``jobs_per_s``: ``jobs / span_s`` — the number that answers "does one more
  worker still help?";
* ``job_s_mean``: mean in-job compute time (grows when workers contend).

Usage (from the project root)::

    PYTORCH_MPS_FAST_MATH=0 .venv/bin/python \
        Optiland-Metal/scripts/metal_saturation.py --config numpy \
        --workers 1 2 4 6 8 10 12 16 --jobs 64 --rays 100000 --repeat 10 \
        --out NOTES/saturation-numpy.json
    PYTORCH_MPS_FAST_MATH=0 .venv/bin/python \
        Optiland-Metal/scripts/metal_saturation.py --config mps-df64 \
        --workers 1 2 3 4 6 8 --jobs 16 --rays 100000 --repeat 3 \
        --out NOTES/saturation-mps-df64.json

The ``mps-df64-fused`` / ``mps-sf64-fused`` configurations (plan 8.5) set
``OPTILAND_METAL_FUSED_TRACE=require`` inside every worker, so a job that
silently lost the fused path raises instead of quietly running per-op.  The
Cooke triplet is eligible for the kernel end to end (every trace in the job is
a 1e5-ray GPU-resident bundle); structural refusals -- the paraxial traces
inside ray generation and every host-resident bundle -- never raise (plan 1.3).

``--batch BxN`` replaces the per-trace job with one ``trace_batch`` call per
job: ``B`` designs of the Cooke triplet (surface 5's radius perturbed) times
``N`` rays in one launch, which is the batched throughput the "GPU processes vs
throughput" curve is re-measured against::

    PYTORCH_MPS_FAST_MATH=0 .venv/bin/python \
        Optiland-Metal/scripts/metal_saturation.py --config mps-df64-fused \
        --batch 100x100000 --workers 1 2 3 --jobs 6 --repeat 1 \
        --out NOTES/saturation-mps-df64-batch.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from typing import Any

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

CONFIGS: dict[str, dict[str, Any]] = {
    "numpy": {"backend": "numpy"},
    "torch-cpu-f64": {"backend": "torch", "device": "cpu", "precision": "float64"},
    "mps-f32": {"backend": "torch", "device": "mps", "precision": "float32"},
    "mps-df64": {
        "backend": "torch",
        "device": "mps",
        "precision": "float64",
        "metal_mode": "df64",
    },
    "mps-sf64": {
        "backend": "torch",
        "device": "mps",
        "precision": "float64",
        "metal_mode": "sf64",
    },
    "mps-df64-fused": {
        "backend": "torch",
        "device": "mps",
        "precision": "float64",
        "metal_mode": "df64",
        "fused": "require",
    },
    "mps-sf64-fused": {
        "backend": "torch",
        "device": "mps",
        "precision": "float64",
        "metal_mode": "sf64",
        "fused": "require",
    },
}


def _rings(n: int) -> int:
    return max(1, int(round(((1 + 4 * (n - 1) / 3) ** 0.5 - 1) / 2)))


def _apply_fused(fused: str | None) -> None:
    """Set ``OPTILAND_METAL_FUSED_TRACE`` inside the worker process.

    The hook reads the switch per trace (plan 1.5), so setting it here covers
    every trace the job runs, whatever order the worker initialised things in.
    """
    if fused is not None:
        os.environ["OPTILAND_METAL_FUSED_TRACE"] = fused


def _sync() -> None:
    """Block until the GPU queue drained, so the stamps bracket real work."""
    import optiland.backend as be

    if be.get_backend() == "torch" and be.get_device() == "mps":
        import torch

        torch.mps.synchronize()


def _counters() -> dict[str, int]:
    """``be.metal_stats()`` as a plain dict (empty off the Metal backend)."""
    import optiland.backend as be

    try:
        return dict(be.metal_stats())
    except Exception:  # noqa: BLE001 - non-Metal backends have no counters
        return {}


def job(rays: int, repeat: int, fused: str | None = None) -> dict[str, Any]:
    """Trace the Cooke triplet ``repeat`` times; return timing stamps and a checksum."""
    import optiland.backend as be
    from optiland.backend.utils import to_numpy
    from optiland.samples.objectives import CookeTriplet

    _apply_fused(fused)
    before = _counters()
    start = time.time()
    lens = CookeTriplet()
    checksum = 0.0
    for _ in range(repeat):
        out = lens.trace(
            Hx=0.0,
            Hy=1.0,
            wavelength=0.55,
            num_rays=_rings(rays),
            distribution="hexapolar",
        )
        if be.get_backend() == "torch" and be.get_device() == "mps":
            import torch

            torch.mps.synchronize()
        checksum += float(to_numpy(out.x)[0])
    end = time.time()
    after = _counters()
    return {
        "start": start,
        "end": end,
        "seconds": end - start,
        "checksum": checksum,
        "pid": os.getpid(),
        "fused_traces": after.get("fused_trace:traces", 0)
        - before.get("fused_trace:traces", 0),
        "fused_skips": {
            k: v - before.get(k, 0)
            for k, v in after.items()
            if k.startswith("fused_trace_skip:") and v != before.get(k, 0)
        },
    }


def batch_job(
    designs: int, rays: int, repeat: int, fused: str | None = None
) -> dict[str, Any]:
    """Trace ``designs`` Cooke triplets x ``rays`` rays per ``trace_batch`` call.

    The designs differ in surface 5's radius, the same perturbation a
    tolerancing sweep applies; ``record="image"`` keeps the snapshot buffer to
    one row per design (plan 3.6).  Returned like :func:`job`, plus whether the
    kernel produced the rows.
    """
    from optiland.backend.utils import to_numpy
    from optiland.optimization.variable import Variable
    from optiland.raytrace.batch_trace import trace_batch
    from optiland.samples.objectives import CookeTriplet

    _apply_fused(fused)
    before = _counters()
    start = time.time()
    lens = CookeTriplet()
    variable = Variable(lens, "radius", surface_number=5)
    base = float(variable.value)
    span = abs(base) * 0.01 if base else 1e-2
    values = [
        [base + span * (2.0 * i / max(1, designs - 1) - 1.0)] for i in range(designs)
    ]
    checksum, fused_calls = 0.0, 0
    for _ in range(repeat):
        result = trace_batch(
            lens,
            [variable],
            values,
            Hx=0.0,
            Hy=1.0,
            wavelength=0.55,
            num_rays=_rings(rays),
            distribution="hexapolar",
            record="image",
        )
        _sync()
        fused_calls += int(result.fused)
        checksum += float(sum(to_numpy(result.x[:, 0, 0])))
    end = time.time()
    after = _counters()
    return {
        "start": start,
        "end": end,
        "seconds": end - start,
        "checksum": checksum,
        "pid": os.getpid(),
        "designs": designs,
        "fused_calls": fused_calls,
        "fused_traces": after.get("fused_trace:traces", 0)
        - before.get("fused_trace:traces", 0),
        "fused_skips": {
            k: v - before.get(k, 0)
            for k, v in after.items()
            if k.startswith("fused_trace_skip:") and v != before.get(k, 0)
        },
    }


def run(
    config: str,
    workers: int,
    jobs: int,
    rays: int,
    repeat: int,
    designs: int | None = None,
) -> dict[str, Any]:
    """One point of the saturation curve: ``jobs`` jobs over ``workers`` workers."""
    from optiland.parallel import WorkerConfig, evaluate_parallel

    cfg = CONFIGS[config]
    fused = cfg.get("fused")
    wc = WorkerConfig(
        device=cfg.get("device", "cpu"),
        precision=cfg.get("precision", "float64"),
        backend=cfg["backend"],
        metal_mode=cfg.get("metal_mode", "df64"),
        threads=1,
    )
    if fused is not None:
        # Workers are spawned, so they inherit this too; the job sets it again
        # itself, which is what makes the setting independent of the start
        # method.
        os.environ["OPTILAND_METAL_FUSED_TRACE"] = fused
    fn = batch_job if designs else job
    args = (designs, rays, repeat, fused) if designs else (rays, repeat, fused)
    t0 = time.time()
    results = evaluate_parallel(fn, [args] * jobs, workers=[wc] * workers)
    wall = time.time() - t0
    starts = [r["start"] for r in results]
    ends = [r["end"] for r in results]
    span = max(ends) - min(starts)
    job_s = [r["seconds"] for r in results]
    checksums = {round(r["checksum"], 9) for r in results}
    rec = {
        "config": config,
        "workers": workers,
        "jobs": jobs,
        "rays": rays,
        "repeat": repeat,
        "wall_s": round(wall, 3),
        "startup_s": round(min(starts) - t0, 3),
        "span_s": round(span, 3),
        "jobs_per_s": round(jobs / span, 3),
        "traces_per_s": round(jobs * repeat / span, 3),
        "job_s_mean": round(statistics.mean(job_s), 4),
        "job_s_min": round(min(job_s), 4),
        "job_s_max": round(max(job_s), 4),
        "distinct_pids": len({r["pid"] for r in results}),
        "checksums_identical": len(checksums) == 1,
        "fused_traces": sum(r.get("fused_traces", 0) for r in results),
        "fused_skips": {
            k: sum(r.get("fused_skips", {}).get(k, 0) for r in results)
            for k in sorted({k for r in results for k in r.get("fused_skips", {})})
        },
    }
    if designs:
        rec["designs"] = designs
        rec["fused_calls"] = sum(r.get("fused_calls", 0) for r in results)
        rec["traces_per_s"] = round(jobs * repeat * designs / span, 3)
    if fused is not None and rec["fused_traces"] == 0:
        raise RuntimeError(
            f"{config}: OPTILAND_METAL_FUSED_TRACE={fused} but no worker ran a "
            f"fused trace (counters {rec['fused_skips']}); the row is not a "
            "measurement of the fused path"
        )
    print(
        f"{config:12s} workers={workers:2d} jobs={jobs:3d}: "
        f"wall {wall:6.2f} s, startup {rec['startup_s']:5.2f} s, "
        f"span {span:6.2f} s, {rec['jobs_per_s']:6.3f} jobs/s, "
        f"job {rec['job_s_mean']:.3f} s "
        f"(min {rec['job_s_min']:.3f}, max {rec['job_s_max']:.3f}), "
        f"pids={rec['distinct_pids']}",
        flush=True,
    )
    return rec


def _parse_batch(spec: str) -> tuple[int, int]:
    """``"100x100000"`` -> ``(100, 100000)``."""
    b, sep, n = spec.lower().partition("x")
    if not sep or not b.strip().isdigit() or not n.strip().isdigit():
        raise argparse.ArgumentTypeError(f"--batch wants BxN, got {spec!r}")
    return int(b), int(n)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="numpy", choices=sorted(CONFIGS))
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--jobs", type=int, default=32)
    parser.add_argument("--rays", type=int, default=100000)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--batch",
        default=None,
        metavar="BxN",
        help="run one trace_batch(B designs, N rays) call per job instead of "
        "the per-trace job",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    designs, rays = None, args.rays
    if args.batch:
        designs, rays = _parse_batch(args.batch)
    if CONFIGS[args.config].get("fused") and designs is None and args.rays <= 256:
        parser.error(
            f"--rays {args.rays} is at or below the host-residency threshold, so "
            "no trace is a fused candidate; a fused configuration needs a "
            "GPU-resident bundle"
        )
    report = {
        "host": {
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0],
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "config": args.config,
        "fused": CONFIGS[args.config].get("fused"),
        "batch": args.batch,
        "results": [
            run(args.config, w, args.jobs, rays, args.repeat, designs)
            for w in args.workers
        ],
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
