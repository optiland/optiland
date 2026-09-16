"""Benchmark Optiland workloads across backends: numpy, torch-cpu, and Metal (mps).

Usage (from the project root)::

    PYTORCH_MPS_FAST_MATH=0 .venv/bin/python Optiland-Metal/scripts/metal_benchmark.py \
        --configs numpy torch-cpu-f64 mps-f32 mps-df64 mps-sf64 \
        --rays 1000 10000 100000 1000000 --repeats 5 --out NOTES/benchmarks.json

Every timed call ends with a device synchronization and a host read of one
result element, so GPU times include completion. Each measurement is preceded
by warm-up runs; the median of ``--repeats`` is reported. Correctness is
checked alongside timing: every configuration's results are compared with the
numpy float64 oracle and the maximum position difference is recorded.

Workloads:
* ``trace``: one full sequential trace of N random-pupil rays through a system.
* ``spot``: SpotDiagram RMS over the default fields (hexapolar rings scale with N).
* ``fd_jacobian``: finite-difference Jacobian of the RMS spot radius with respect to
  every curvature (2 traces per variable), i.e. one optimizer step's worth of work.
* ``parallel``: the ``fd_jacobian`` workload distributed over worker processes via
  ``optiland.parallel`` (spawn), for 1/2/4/8 workers of the configuration.
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

import numpy as np  # noqa: E402

CONFIGS: dict[str, dict[str, Any]] = {
    "numpy": {"backend": "numpy"},
    "torch-cpu-f64": {"backend": "torch", "device": "cpu", "precision": "float64"},
    "torch-cpu-f32": {"backend": "torch", "device": "cpu", "precision": "float32"},
    "mps-f32": {"backend": "torch", "device": "mps", "precision": "float32"},
    "mps-df64": {"backend": "torch", "device": "mps", "precision": "float64", "mode": "df64"},
    "mps-sf64": {"backend": "torch", "device": "mps", "precision": "float64", "mode": "sf64"},
}


def apply_config(name: str) -> None:
    """Select backend/device/precision for ``name`` in this process."""
    import optiland.backend as be

    cfg = CONFIGS[name]
    be.set_backend(cfg["backend"])
    if cfg["backend"] == "torch":
        be.set_device(cfg["device"])
        be.set_precision(cfg["precision"])
        if cfg["device"] == "mps" and cfg["precision"] == "float64":
            from optiland.backend.torch_backend import metal

            metal.set_mode(cfg.get("mode", "df64"))


def sync() -> None:
    """Wait for outstanding GPU work (no-op on CPU)."""
    import optiland.backend as be

    if be.get_backend() == "torch":
        import torch

        if be.get_device() == "mps":
            torch.mps.synchronize()


def systems() -> dict[str, Any]:
    """Sample systems: a spherical triplet, a wide-angle retrofocus, and an asphere."""
    from optiland.samples import objectives

    out = {"cooke": objectives.CookeTriplet, "reverse_telephoto": objectives.ReverseTelephoto}
    for name in ("HeliarLens", "TessarLens", "DoubleGauss"):
        if hasattr(objectives, name):
            out[name.lower()] = getattr(objectives, name)
            break
    try:
        from optiland.samples import simple

        for name in dir(simple):
            if "asph" in name.lower():
                out["asphere"] = getattr(simple, name)
                break
    except ImportError:  # pragma: no cover
        pass
    return out


def hexapolar_rings(n: int) -> int:
    """Number of hexapolar rings giving about ``n`` rays (1 + 3 r (r + 1) rays)."""
    return max(1, int(round((np.sqrt(1 + 4 * (n - 1) / 3) - 1) / 2)))


def trace_workload(optic: Any, n: int) -> Any:
    """Trace about ``n`` hexapolar pupil rays at the edge field (deterministic)."""
    return optic.trace(Hx=0.0, Hy=1.0, wavelength=0.55, num_rays=hexapolar_rings(n), distribution="hexapolar")


def spot_workload(optic: Any, n: int) -> np.ndarray:
    """RMS spot radius over all fields (hexapolar rings ~ sqrt(n/3))."""
    from optiland.analysis import SpotDiagram
    from optiland.backend.utils import to_numpy

    sd = SpotDiagram(optic, num_rings=max(3, hexapolar_rings(n)))
    return np.array([[float(to_numpy(v)) for v in row] for row in sd.rms_spot_radius()])


def fd_jacobian_workload(optic: Any, n: int, step: float = 1e-4) -> np.ndarray:
    """Central finite differences of the mean RMS spot radius w.r.t. every finite radius."""
    from optiland.backend.utils import to_numpy

    radii = to_numpy(optic.surface_group.radii).astype(float)
    idx = [i for i, r in enumerate(radii) if np.isfinite(r) and r != 0 and 0 < i < len(radii) - 1]

    def merit() -> float:
        return float(np.mean(spot_workload(optic, n)))

    grad = np.zeros(len(idx))
    for k, i in enumerate(idx):
        r0 = float(radii[i])
        for sgn in (+1, -1):
            optic.set_radius(r0 * (1 + sgn * step), i)
            v = merit()
            grad[k] += sgn * v
        optic.set_radius(r0, i)
        grad[k] /= 2 * r0 * step
    return grad


def timed(fn: Any, repeats: int, warmup: int = 2) -> tuple[float, list[float], Any]:
    """Return (median seconds, all samples, last result) with sync inside the timing."""
    result = None
    for _ in range(warmup):
        result = fn()
        sync()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        sync()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), samples, result


def rays_to_numpy(rays: Any) -> dict[str, np.ndarray]:
    from optiland.backend.utils import to_numpy

    return {k: to_numpy(getattr(rays, k)) for k in ("x", "y", "z", "L", "M", "N", "opd")}


def max_diff(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, float]:
    out = {}
    for k in a:
        m = np.isfinite(a[k]) & np.isfinite(b[k])
        out[k] = float(np.max(np.abs(a[k][m] - b[k][m]))) if m.any() else float("nan")
    out["nan_pattern_equal"] = float(all(np.array_equal(np.isnan(a[k]), np.isnan(b[k])) for k in a))
    return out


def _parallel_job(optic_dict: Any, n: int) -> list[float]:
    from optiland.optic import Optic

    return fd_jacobian_workload(Optic.from_dict(optic_dict), n).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", default=["numpy", "torch-cpu-f64", "mps-df64"])
    parser.add_argument("--systems", nargs="+", default=["cooke"])
    parser.add_argument("--rays", nargs="+", type=int, default=[1000, 10000, 100000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--workloads", nargs="+", default=["trace", "spot", "fd_jacobian"])
    parser.add_argument("--parallel-workers", nargs="*", type=int, default=[])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import optiland.backend as be

    report: dict[str, Any] = {
        "host": {"machine": platform.machine(), "macos": platform.mac_ver()[0], "python": platform.python_version()},
        "results": [],
    }
    try:
        import torch

        report["host"]["torch"] = torch.__version__
    except ImportError:  # pragma: no cover
        pass
    sysmap = systems()
    oracle: dict[tuple[str, str, int], Any] = {}

    for sys_name in args.systems:
        for n in args.rays:
            for wl in args.workloads:
                for cfg in args.configs:
                    apply_config(cfg)
                    optic = sysmap[sys_name]()
                    if wl == "trace":
                        med, samples, res = timed(lambda: trace_workload(optic, n), args.repeats)
                        payload = rays_to_numpy(res)
                    elif wl == "spot":
                        med, samples, res = timed(lambda: spot_workload(optic, n), args.repeats)
                        payload = {"rms": np.asarray(res)}
                    elif wl == "fd_jacobian":
                        med, samples, res = timed(lambda: fd_jacobian_workload(optic, n), max(1, args.repeats // 2))
                        payload = {"grad": np.asarray(res)}
                    else:
                        raise ValueError(wl)
                    key = (sys_name, wl, n)
                    if cfg == "numpy":
                        oracle[key] = payload
                    diff = max_diff(payload, oracle[key]) if key in oracle else {}
                    rec = {"system": sys_name, "workload": wl, "n": n, "config": cfg, "median_s": med, "samples_s": samples, "max_abs_diff_vs_numpy": diff}
                    if be.get_backend() == "torch" and be.get_device() == "mps" and be.get_precision() == 64:
                        from optiland.backend.torch_backend import metal

                        rec["metal_stats"] = metal.stats()
                        metal.reset_stats()
                    report["results"].append(rec)
                    print(f"{sys_name:18s} {wl:12s} n={n:<8d} {cfg:14s} median {med*1e3:9.2f} ms  maxΔ {max((v for k, v in diff.items() if k != 'nan_pattern_equal'), default=float('nan')):.2e}", flush=True)
                    be.set_backend("numpy")

    if args.parallel_workers:
        from optiland.parallel import WorkerConfig, evaluate_parallel

        for cfg in args.configs:
            c = CONFIGS[cfg]
            wc = WorkerConfig(device=c.get("device", "cpu"), precision=c.get("precision", "float64"), backend=c["backend"], metal_mode=c.get("mode", "df64"))
            for w in args.parallel_workers:
                jobs = [(sysmap[args.systems[0]]().to_dict(), args.rays[0]) for _ in range(8)]
                t0 = time.perf_counter()
                evaluate_parallel(_parallel_job, jobs, workers=[wc] * w)
                dt = time.perf_counter() - t0
                report["results"].append({"workload": "parallel_fd_jacobian_x8jobs", "config": cfg, "workers": w, "wall_s": dt})
                print(f"parallel {cfg:14s} workers={w}: 8 fd_jacobian jobs in {dt:.2f} s", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1, default=float)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
