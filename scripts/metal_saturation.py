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
}


def _rings(n: int) -> int:
    return max(1, int(round(((1 + 4 * (n - 1) / 3) ** 0.5 - 1) / 2)))


def job(rays: int, repeat: int) -> dict[str, float]:
    """Trace the Cooke triplet ``repeat`` times; return timing stamps and a checksum."""
    import optiland.backend as be
    from optiland.backend.utils import to_numpy
    from optiland.samples.objectives import CookeTriplet

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
    return {
        "start": start,
        "end": end,
        "seconds": end - start,
        "checksum": checksum,
        "pid": os.getpid(),
    }


def run(config: str, workers: int, jobs: int, rays: int, repeat: int) -> dict[str, Any]:
    from optiland.parallel import WorkerConfig, evaluate_parallel

    cfg = CONFIGS[config]
    wc = WorkerConfig(
        device=cfg.get("device", "cpu"),
        precision=cfg.get("precision", "float64"),
        backend=cfg["backend"],
        metal_mode=cfg.get("metal_mode", "df64"),
        threads=1,
    )
    t0 = time.time()
    results = evaluate_parallel(job, [(rays, repeat)] * jobs, workers=[wc] * workers)
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
    }
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="numpy", choices=sorted(CONFIGS))
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--jobs", type=int, default=32)
    parser.add_argument("--rays", type=int, default=100000)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    report = {
        "host": {
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0],
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "results": [
            run(args.config, w, args.jobs, args.rays, args.repeat) for w in args.workers
        ],
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
