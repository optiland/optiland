"""Benchmark Optiland workloads across backends: numpy, torch-cpu and Metal (mps).

Usage (from the fork root; ``$PROJECT_ROOT`` is the directory above it)::

    PYTORCH_MPS_FAST_MATH=0 ../.venv/bin/python scripts/metal_benchmark.py \
        --configs numpy torch-cpu-f64 mps-df64 mps-df64-fused mps-sf64-fused mps-sf64 \
        --systems cooke reverse_telephoto asphere --rays 1000 10000 100000 1000000 \
        --repeats 5 --workloads trace trace_group spot fd_jacobian fd_jacobian_fused \
        --out ../NOTES/benchmarks-fused-single.json

Every timed call ends with a device synchronization, so GPU times include
completion; each measurement is preceded by warm-ups and the median of
``--repeats`` is reported.

**Every timed row carries a correctness record** computed from the same result
object the timing produced (plan section 9): the per-quantity maximum absolute
difference against the NumPy float64 oracle, the NaN-pattern verdict and the
``i == 0``-mask verdict, judged by the external rule of plan 7.1.  A row whose
record does not pass is reported ``FAIL(reason=correctness)`` and is never
summarised as a speed-up; a row on a fused configuration during which any
``fused_trace_skip:*`` counter moved is ``FAIL(reason=<skip reason>)``, and one
that never reached the kernel at all is ``FAIL(reason=not_fused)``.  A fused
number that silently measured the per-op path is the one failure this harness
exists to prevent.

Workloads:

* ``trace``: one full sequential trace of N hexapolar pupil rays.
* ``trace_group``: ``optic.surfaces.trace(rays)`` alone, on a bundle generated
  once and cloned before each repeat -- the kernel's own workload, without ray
  generation or the trailing propagation.
* ``spot``: SpotDiagram RMS over the default fields.
* ``fd_jacobian``: the existing sequential finite-difference Jacobian of the
  mean RMS spot radius with respect to every finite radius (one optimizer step).
* ``fd_jacobian_fused``: the same stencil through ``batch_trace.fd_jacobian``,
  i.e. one launch for all 2n designs.
* ``trace_batch``: ``--batch BxN`` -- B perturbed designs x N rays in one
  launch, against a sequential NumPy loop and against eight NumPy worker
  processes.
* ``--group-sweep``: the ``trace_group`` workload repeated for each
  ``OPTILAND_METAL_FUSED_TRACE_GROUP`` threadgroup size.

Rendering: ``--md <file> --from-json a.json b.json ...`` writes the report
tables (single trace, batch, targets, threadgroup sweep, memory, counters,
documented limits) from JSONs produced by earlier runs and measures nothing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import numpy as np  # noqa: E402

#: The fork root (the directory that holds ``optiland/`` and ``scripts/``).
REPO = Path(__file__).resolve().parents[1]

#: Machine epsilon per Metal representation (``metal/tensor.py``), and float64
#: for every host backend.  The external rule of plan 7.1 is derived from it;
#: no tolerance in this file is chosen by hand.
MACHINE_EPS: dict[str, float] = {
    "df64": 2.0**-48,
    "sf64": 2.0**-53,
    "float64": float(np.finfo(np.float64).eps),
    "float32": float(np.finfo(np.float32).eps),
}

#: The wavelength every timed workload traces at (plan 9.2 writes it out as
#: ``wavelength=0.55``).  Exposed as ``--wavelength`` so a run can move it, but
#: the default is the plan's, which is also what the historical
#: ``NOTES/benchmarks-single.json`` rows used.
WAVELENGTH = 0.55

#: Plan 7.1's external rule: the absolute position floor in mm, and the eps
#: multiplier applied to the system scale.
EXTERNAL_POS_ABS = 1e-11
EXTERNAL_FACTOR = 64.0

CONFIGS: dict[str, dict[str, Any]] = {
    "numpy": {"backend": "numpy"},
    "numpy-8w": {"backend": "numpy", "workers": 8},
    "torch-cpu-f64": {"backend": "torch", "device": "cpu", "precision": "float64"},
    "torch-cpu-f32": {"backend": "torch", "device": "cpu", "precision": "float32"},
    "mps-f32": {"backend": "torch", "device": "mps", "precision": "float32"},
    "mps-df64": {
        "backend": "torch",
        "device": "mps",
        "precision": "float64",
        "mode": "df64",
        "fused": "0",
    },
    "mps-sf64": {
        "backend": "torch",
        "device": "mps",
        "precision": "float64",
        "mode": "sf64",
        "fused": "0",
    },
    "mps-df64-fused": {
        "backend": "torch",
        "device": "mps",
        "precision": "float64",
        "mode": "df64",
        "fused": "require",
    },
    "mps-sf64-fused": {
        "backend": "torch",
        "device": "mps",
        "precision": "float64",
        "mode": "sf64",
        "fused": "require",
    },
}

#: Workloads a worker-pool configuration can run (plan 9.1: "batch and fd rows").
POOL_WORKLOADS = ("trace_batch", "fd_jacobian_fused")


def is_fused(config: str) -> bool:
    """Whether ``config`` asks the hook to fuse."""
    return CONFIGS[config].get("fused") not in (None, "0")


def is_metal(config: str) -> bool:
    """Whether ``config`` is Metal float64 (df64 or sf64)."""
    cfg = CONFIGS[config]
    return cfg.get("device") == "mps" and cfg.get("precision") == "float64"


def eps_of(config: str) -> float:
    """The machine epsilon the external rule uses for ``config``."""
    cfg = CONFIGS[config]
    if is_metal(config):
        return MACHINE_EPS[cfg.get("mode", "df64")]
    return MACHINE_EPS[cfg.get("precision", "float64")]


def apply_config(name: str) -> None:
    """Select backend/device/precision/mode and the fused switch for ``name``.

    The fused switch is part of the configuration, so it is applied here and
    *removed* for configurations that do not name one: a stale
    ``OPTILAND_METAL_FUSED_TRACE`` left in the environment would make a
    "per-op" row measure the kernel.
    """
    import optiland.backend as be

    cfg = CONFIGS[name]
    fused = cfg.get("fused")
    if fused is None:
        os.environ.pop("OPTILAND_METAL_FUSED_TRACE", None)
    else:
        os.environ["OPTILAND_METAL_FUSED_TRACE"] = fused
    be.set_backend(cfg["backend"])
    if cfg["backend"] == "torch":
        be.set_device(cfg["device"])
        be.set_precision(cfg["precision"])
        # The gate refuses any bundle that participates in autograd (plan 1.2),
        # so a benchmark that left grad on would measure the per-op path under
        # a fused label.
        be.grad_mode.disable()
        if cfg["device"] == "mps" and cfg["precision"] == "float64":
            from optiland.backend.torch_backend import metal

            metal.set_mode(cfg.get("mode", "df64"))


@contextmanager
def no_grad():
    """``torch.no_grad()`` on the torch backend, a no-op elsewhere."""
    import optiland.backend as be

    if be.get_backend() == "torch":
        import torch

        with torch.no_grad():
            yield
        return
    yield


def sync() -> None:
    """Wait for outstanding GPU work (no-op on CPU)."""
    import optiland.backend as be

    if be.get_backend() == "torch":
        import torch

        if be.get_device() == "mps":
            torch.mps.synchronize()


def counters() -> dict[str, int]:
    """``metal.stats()`` as a plain dict (empty off the Metal float64 path)."""
    try:
        from optiland.backend.torch_backend import metal

        return dict(metal.stats())
    except Exception:  # noqa: BLE001 - non-Metal backends have no counters
        return {}


def reset_counters() -> None:
    """Zero the Metal counters, where they exist."""
    try:
        from optiland.backend.torch_backend import metal

        metal.reset_stats()
    except Exception:  # noqa: BLE001 - non-Metal backends have no counters
        pass


def counter_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    """Non-zero increments of the ``fused_trace``/``gpu`` counters."""
    keys = set(before) | set(after)
    return {
        k: after.get(k, 0) - before.get(k, 0)
        for k in sorted(keys)
        if after.get(k, 0) != before.get(k, 0)
    }


def peak_bytes() -> int:
    """``torch.mps.driver_allocated_memory()``, or 0 off the GPU."""
    try:
        import torch

        return int(torch.mps.driver_allocated_memory())
    except Exception:  # noqa: BLE001 - CPU backends have no driver allocator
        return 0


# ---------------------------------------------------------------------------
# Systems and bundles
# ---------------------------------------------------------------------------


def systems() -> dict[str, Any]:
    """The benchmark's systems, by the name ``--systems`` uses."""
    from optiland.samples import objectives, simple

    out: dict[str, Any] = {
        "cooke": objectives.CookeTriplet,
        "reverse_telephoto": objectives.ReverseTelephoto,
        "asphere": simple.AsphericSinglet,
    }
    for name in ("HeliarLens", "TessarLens", "DoubleGauss"):
        if hasattr(objectives, name):
            out[name.lower()] = getattr(objectives, name)
    return out


def hexapolar_rings(n: int) -> int:
    """Number of hexapolar rings giving about ``n`` rays (1 + 3 r (r + 1))."""
    return max(1, int(round((math.sqrt(1 + 4 * (n - 1) / 3) - 1) / 2)))


def hexapolar_count(rings: int) -> int:
    """How many rays ``rings`` hexapolar rings actually produce."""
    return 1 + 3 * rings * (rings + 1)


def clone_rays(rays: Any) -> Any:
    """A bundle holding the SAME encoded words as ``rays``.

    Component-level ``be.copy``: a ``to_numpy`` / ``be.array`` round trip
    re-encodes and shifts the df64 low word, which would mean the timed traces
    did not all start from the same numbers.
    """
    import optiland.backend as be
    from optiland.rays import RealRays

    out = RealRays(
        *(be.copy(getattr(rays, a)) for a in ("x", "y", "z", "L", "M", "N", "i", "w"))
    )
    out.opd = be.copy(rays.opd)
    return out


def launch_bundle(optic: Any, n: int, wavelength: float = WAVELENGTH) -> Any:
    """A hexapolar edge-field bundle of about ``n`` rays, through the generator.

    Built exactly as ``optic.trace`` would build it, so ``trace_group`` traces
    the same rays ``trace`` does.
    """
    import optiland.backend as be
    from optiland.distribution import create_distribution

    rings = hexapolar_rings(n)
    distribution = create_distribution("hexapolar")
    distribution.generate_points(rings)
    count = hexapolar_count(rings)
    generator = optic.ray_tracer.ray_generator
    return generator.generate_rays(
        be.array(np.zeros(count)),
        be.array(np.ones(count)),
        distribution.x,
        distribution.y,
        wavelength,
    )


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------


def trace_workload(optic: Any, n: int, wavelength: float = WAVELENGTH) -> Any:
    """Trace about ``n`` hexapolar pupil rays at the edge field (deterministic)."""
    return optic.trace(
        Hx=0.0,
        Hy=1.0,
        wavelength=wavelength,
        num_rays=hexapolar_rings(n),
        distribution="hexapolar",
    )


def trace_group_workload(optic: Any, rays: Any) -> Any:
    """``SurfaceGroup.trace`` alone, on a clone of the pre-built bundle."""
    return optic.surfaces.trace(clone_rays(rays))


def spot_workload(optic: Any, n: int) -> np.ndarray:
    """RMS spot radius over all fields (hexapolar rings ~ sqrt(n/3))."""
    from optiland.analysis import SpotDiagram
    from optiland.backend.utils import to_numpy

    sd = SpotDiagram(optic, num_rings=max(3, hexapolar_rings(n)))
    return np.array([[float(to_numpy(v)) for v in row] for row in sd.rms_spot_radius()])


#: The relative step of the historical sequential ``fd_jacobian`` workload.
FD_STEP_SEQ = 1e-4


def finite_radius_indices(optic: Any) -> list[int]:
    """The interior surfaces whose radius is finite and non-zero."""
    from optiland.backend.utils import to_numpy

    radii = np.asarray(to_numpy(optic.surfaces.radii), dtype=float)
    return [
        i
        for i, r in enumerate(radii)
        if np.isfinite(r) and r != 0 and 0 < i < len(radii) - 1
    ]


def fd_jacobian_workload(optic: Any, n: int, step: float = FD_STEP_SEQ) -> np.ndarray:
    """Central finite differences of the mean RMS spot radius w.r.t. every radius.

    The historical sequential workload, kept unchanged so its rows stay
    comparable with ``NOTES/benchmarks-fd.json``.
    """
    idx = finite_radius_indices(optic)
    from optiland.backend.utils import to_numpy

    radii = np.asarray(to_numpy(optic.surfaces.radii), dtype=float)

    def merit() -> float:
        return float(np.mean(spot_workload(optic, n)))

    grad = np.zeros(len(idx))
    for k, i in enumerate(idx):
        r0 = float(radii[i])
        for sgn in (+1, -1):
            optic.set_radius(r0 * (1 + sgn * step), i)
            grad[k] += sgn * merit()
        optic.set_radius(r0, i)
        grad[k] /= 2 * r0 * step
    return grad


def rms_merit(result: Any, b: int) -> Any:
    """RMS spot radius of design ``b``'s image row (``test_trace_fd.rms_merit``).

    Written on the ``(N,)`` plane of one design so the reduction sees the same
    shape whether the batch holds one design or ``2 n``.
    """
    import optiland.backend as be
    from optiland.raytrace import batch_trace as bt

    row = result.rows[max(result.rows)]
    x = bt._select(result.x, b, row)  # noqa: SLF001 - the documented accessor
    y = bt._select(result.y, b, row)  # noqa: SLF001
    r2 = (x - be.nanmean(x)) ** 2 + (y - be.nanmean(y)) ** 2
    return be.sqrt(be.nanmean(r2))


def fd_variables(optic: Any) -> list[Any]:
    """One ``radius`` variable per finite-radius interior surface."""
    from optiland.optimization.variable import Variable

    return [
        Variable(optic, "radius", surface_number=i)
        for i in finite_radius_indices(optic)
    ]


#: The scaled-unit step of the batched Jacobian.  ``RadiusVariable`` scales by
#: 1/100, so this is 1e-2 mm of physical radius.
FD_STEP = 1e-4
FD_STEP_SMALL = 1e-6


def fd_sequential_denominators(optic: Any, step: float = FD_STEP_SEQ) -> np.ndarray:
    """The ACTUAL denominators of :func:`fd_jacobian_workload`: ``2 r0 step``.

    A finite-difference row's representation noise is the merit's own noise
    divided by the stencil width, so the width is part of the correctness rule
    and is read from the same expression the workload divides by -- not
    guessed.
    """
    from optiland.backend.utils import to_numpy

    radii = np.asarray(to_numpy(optic.surfaces.radii), dtype=float)
    return np.array(
        [2.0 * abs(float(radii[i])) * step for i in finite_radius_indices(optic)],
        dtype=np.float64,
    )


def fd_bound(merit_scale: float, denominators: Any, eps: float) -> float:
    """The correctness bound of a finite-difference row.

    Both differences of a central pair carry the merit's own agreement bound
    with the NumPy oracle -- the external rule of plan 7.1 applied to a
    position-like quantity in mm -- so the derivative's bound is twice that,
    divided by the narrowest stencil actually used.  Nothing here is a chosen
    constant: ``EXTERNAL_POS_ABS`` and ``EXTERNAL_FACTOR`` are plan 7.1's, the
    denominators are the workload's own, and ``eps`` is the mode's.
    """
    merit_bound = max(EXTERNAL_POS_ABS, EXTERNAL_FACTOR * eps * abs(merit_scale))
    width = float(np.min(np.abs(np.asarray(denominators, dtype=np.float64))))
    return 2.0 * merit_bound / width if width else float("inf")


def fd_jacobian_fused_workload(
    optic: Any, n: int, wavelength: float = WAVELENGTH
) -> Any:
    """``batch_trace.fd_jacobian``: the whole stencil in one launch."""
    from optiland.raytrace import batch_trace as bt

    variables = fd_variables(optic)
    return bt.fd_jacobian(
        optic,
        variables,
        rms_merit,
        FD_STEP,
        Hx=0.0,
        Hy=1.0,
        wavelength=wavelength,
        num_rays=hexapolar_rings(n),
        distribution="hexapolar",
        record="image",
    )


# ---------------------------------------------------------------------------
# Correctness (plan 7.1 external rule, plan 9.2 "correctness per row")
# ---------------------------------------------------------------------------


def rays_to_numpy(rays: Any) -> dict[str, np.ndarray]:
    """The returned bundle's planes as float64 host arrays."""
    from optiland.backend.utils import to_numpy

    out = {
        k: np.asarray(to_numpy(getattr(rays, k)), dtype=np.float64)
        for k in ("x", "y", "z", "L", "M", "N", "opd")
    }
    out["i"] = np.asarray(to_numpy(rays.i), dtype=np.float64)
    return out


def image_row_to_numpy(optic: Any) -> dict[str, np.ndarray]:
    """The image surface's recorded row as float64 host arrays."""
    from optiland.backend.utils import to_numpy

    surface = optic.surfaces.surfaces[-1]
    out: dict[str, np.ndarray] = {}
    for attr, key in (
        ("x", "x"),
        ("y", "y"),
        ("z", "z"),
        ("L", "L"),
        ("M", "M"),
        ("N", "N"),
        ("opd", "opd"),
        ("intensity", "i"),
    ):
        value = getattr(surface, attr, None)
        if value is None:
            continue
        arr = np.asarray(to_numpy(value), dtype=np.float64)
        if arr.size:
            out[key] = arr
    return out


def path_scale(planes: dict[str, np.ndarray]) -> float:
    """``max|finite positions| + max|x, y|`` -- the scale of plan 7.1."""
    positions: list[float] = []
    transverse: list[float] = []
    for attr in ("x", "y", "z"):
        arr = planes.get(attr)
        if arr is None:
            continue
        finite = arr[np.isfinite(arr)]
        if finite.size:
            positions.append(float(np.max(np.abs(finite))))
            if attr in ("x", "y"):
                transverse.append(float(np.max(np.abs(finite))))
    return (max(positions) if positions else 0.0) + (
        max(transverse) if transverse else 0.0
    )


#: Which bound each plane is judged against (plan 7.1 external rule).
COS_PLANES = ("L", "M", "N")


def external_rule(
    got: dict[str, np.ndarray],
    ref: dict[str, np.ndarray],
    *,
    eps: float,
    label: str,
) -> dict[str, Any]:
    """Plan 7.1's external rule on one pair of plane dicts.

    Positions and OPD pass at ``max(1e-11 mm, 64 * eps * scale)``; direction
    cosines at ``64 * eps``; NaN patterns and ``i == 0`` patterns must be
    equal exactly.  The three benchmark systems carry no aperture (recorded in
    the provenance as ``apertures``), so there is no rim band and the
    intensity-mask comparison is the exact one plan 7.1 asks for when no finite
    aperture edge exists.
    """
    scale = path_scale(ref)
    pos_bound = max(EXTERNAL_POS_ABS, EXTERNAL_FACTOR * eps * scale)
    cos_bound = EXTERNAL_FACTOR * eps
    record: dict[str, Any] = {
        "label": label,
        "scale_mm": scale,
        "pos_bound": pos_bound,
        "cos_bound": cos_bound,
        "max_abs_diff": {},
        "nan_pattern_equal": True,
        "intensity_zero_equal": True,
        "pass": True,
        "worst": "",
    }
    worst_ratio = 0.0
    for key in sorted(set(got) & set(ref)):
        a, b = got[key], ref[key]
        if a.shape != b.shape:
            record["pass"] = False
            record["worst"] = f"{key}: shape {a.shape} vs {b.shape}"
            return record
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            record["nan_pattern_equal"] = False
            record["pass"] = False
        if key == "i":
            if not np.array_equal(a == 0.0, b == 0.0):
                record["intensity_zero_equal"] = False
                record["pass"] = False
            continue
        both = np.isfinite(a) & np.isfinite(b)
        delta = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
        record["max_abs_diff"][key] = delta
        bound = cos_bound if key in COS_PLANES else pos_bound
        ratio = delta / bound if bound else float("inf")
        if ratio > worst_ratio:
            worst_ratio = ratio
            record["worst"] = f"{key} {delta:.3e} / {bound:.3e}"
        if delta > bound:
            record["pass"] = False
    record["worst_ratio"] = worst_ratio
    return record


def array_record(
    got: np.ndarray, ref: np.ndarray, *, bound: float, label: str
) -> dict[str, Any]:
    """A correctness record for a plain array result (spot, Jacobian)."""
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    both = np.isfinite(got) & np.isfinite(ref)
    delta = float(np.max(np.abs(got[both] - ref[both]))) if both.any() else 0.0
    return {
        "label": label,
        "max_abs_diff": {label: delta},
        "bound": bound,
        "nan_pattern_equal": bool(np.array_equal(np.isnan(got), np.isnan(ref))),
        "pass": bool(delta <= bound and np.array_equal(np.isnan(got), np.isnan(ref))),
        "worst": f"{label} {delta:.3e} / {bound:.3e}",
        "worst_ratio": (delta / bound) if bound else float("inf"),
    }


def merge_records(*records: dict[str, Any]) -> dict[str, Any]:
    """Combine several correctness records into the one a row carries."""
    parts = [r for r in records if r]
    if not parts:
        return {"pass": True, "parts": []}
    return {
        "pass": all(bool(r["pass"]) for r in parts),
        "worst": max(parts, key=lambda r: r.get("worst_ratio", 0.0)).get("worst", ""),
        "worst_ratio": max(float(r.get("worst_ratio", 0.0)) for r in parts),
        "nan_pattern_equal": all(bool(r.get("nan_pattern_equal", True)) for r in parts),
        "parts": parts,
    }


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def timed(fn: Any, repeats: int, warmup: int = 2) -> tuple[float, list[float], Any]:
    """``(median seconds, samples, last result)`` with the sync inside the timing."""
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


@contextmanager
def kernel_timer():
    """Accumulate the seconds spent inside ``trace.launch_trace``.

    ``batch_trace`` resolves ``driver.launch_trace`` at call time, so wrapping
    the module attribute measures the kernel alone -- record compilation,
    launch-set generation and writeback stay outside it (plan 9.2:
    "kernel time by timing ``launch_trace`` alone").
    """
    try:
        from optiland.backend.torch_backend.metal import trace as driver
    except Exception:  # noqa: BLE001 - no Metal: nothing to time
        yield {"seconds": 0.0, "calls": 0}
        return
    box = {"seconds": 0.0, "calls": 0}
    original = driver.launch_trace

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        out = original(*args, **kwargs)
        sync()
        box["seconds"] += time.perf_counter() - t0
        box["calls"] += 1
        return out

    driver.launch_trace = wrapper
    try:
        yield box
    finally:
        driver.launch_trace = original


# ---------------------------------------------------------------------------
# The batched-design workload (plan 9.2)
# ---------------------------------------------------------------------------

#: The two design-set variants of plan 9.2: a post-stop variable (the launch
#: set is shared by every design) and a pre-stop one (one launch set each).
BATCH_VARIANTS = ("post", "pre")

#: ``(kind, surface_number)`` per variant, on the CookeTriplet (stop index 4).
BATCH_VARIABLE: dict[str, tuple[str, int]] = {
    "post": ("radius", 5),
    "pre": ("thickness", 2),
}

#: Relative perturbation of the nominal value, plan 9.2.
BATCH_REL = 0.02


def batch_design_set_applies(builder: Any, variants: Any) -> bool:
    """Whether plan 9.2's CookeTriplet design set exists on ``builder``'s system.

    The batch workload perturbs surface 5's radius (post-stop) and surface 2's
    thickness (pre-stop).  A shallower system has neither, so it is skipped with
    a message instead of raising an ``IndexError`` half way through a run.
    """
    import optiland.backend as be

    previous = be.get_backend()
    be.set_backend("numpy")
    try:
        surfaces = len(builder().surfaces.surfaces)
    finally:
        be.set_backend(previous)
    return all(BATCH_VARIABLE[v][1] < surfaces - 1 for v in variants)


def batch_values(optic: Any, variant: str, b: int) -> tuple[Any, np.ndarray]:
    """``(variable, values)`` for ``b`` designs of ``variant``.

    The perturbations are drawn in *physical* units from
    ``np.random.default_rng(0)`` and converted with ``var.variable.scale``,
    because ``Variable.update`` takes scaled units (plan 3.9 "Units").
    """
    from optiland.optimization.variable import Variable

    kind, surface = BATCH_VARIABLE[variant]
    var = Variable(optic, kind, surface_number=surface)
    physical = float(var.variable.inverse_scale(var.value))
    rng = np.random.default_rng(0)
    perturbed = physical * (1.0 + BATCH_REL * (2.0 * rng.random(b) - 1.0))
    values = np.array(
        [[float(var.variable.scale(float(p)))] for p in perturbed], dtype=np.float64
    )
    return var, values


def _batch_job(
    optic_dict: dict[str, Any],
    var_spec: tuple[str, int],
    values: list,
    n: int,
    wavelength: float = WAVELENGTH,
) -> list[list[float]]:
    """One worker's chunk of designs, traced sequentially on its own backend.

    Module-level so it pickles by reference under the ``spawn`` start method
    (``metal_saturation.py`` does the same).  Returns the image-row ``x``/``y``
    of every design, which is what the correctness record compares.
    """
    import optiland.backend as be
    from optiland.backend.utils import to_numpy
    from optiland.optic import Optic
    from optiland.optimization.variable import Variable

    optic = Optic.from_dict(optic_dict)
    kind, surface = var_spec
    var = Variable(optic, kind, surface_number=surface)
    rings = hexapolar_rings(n)
    out: list[list[float]] = []
    for row in values:
        var.update(float(row[0]))
        optic.updater.update()
        rays = optic.trace(
            Hx=0.0,
            Hy=1.0,
            wavelength=wavelength,
            num_rays=rings,
            distribution="hexapolar",
        )
        out.append(
            [
                float(np.nansum(np.asarray(to_numpy(rays.x), dtype=float))),
                float(np.nansum(np.asarray(to_numpy(rays.y), dtype=float))),
            ]
        )
    del be
    return out


def numpy_batch_reference(
    optic: Any, variant: str, b: int, n: int, wavelength: float = WAVELENGTH
) -> tuple[float, dict[str, np.ndarray]]:
    """The sequential NumPy loop: ``(wall seconds, image rows per design)``.

    ``var.update`` / ``optic.updater.update()`` / ``optic.trace`` is what a
    tolerancing sweep or an optimizer does today, one design at a time.
    """
    from optiland.backend.utils import to_numpy

    var, values = batch_values(optic, variant, b)
    original = var.value
    rings = hexapolar_rings(n)
    rows: dict[str, list[np.ndarray]] = {k: [] for k in ("x", "y", "z", "opd", "i")}
    t0 = time.perf_counter()
    try:
        for row in values:
            var.update(float(row[0]))
            optic.updater.update()
            rays = optic.trace(
                Hx=0.0,
                Hy=1.0,
                wavelength=wavelength,
                num_rays=rings,
                distribution="hexapolar",
            )
            image = optic.surfaces.surfaces[-1]
            for key, attr in (
                ("x", "x"),
                ("y", "y"),
                ("z", "z"),
                ("opd", "opd"),
                ("i", "intensity"),
            ):
                rows[key].append(
                    np.asarray(to_numpy(getattr(image, attr)), dtype=np.float64)
                )
            del rays
    finally:
        var.update(original)
        optic.updater.update()
    wall = time.perf_counter() - t0
    return wall, {k: np.stack(v) for k, v in rows.items()}


def run_trace_batch(
    config: str,
    builder: Any,
    variant: str,
    b: int,
    n: int,
    repeats: int,
    wavelength: float = WAVELENGTH,
) -> dict[str, Any]:
    """One ``trace_batch`` row: timing, counters, memory and the design rows."""
    from optiland.raytrace import batch_trace as bt

    optic = builder()
    var, values = batch_values(optic, variant, b)
    rings = hexapolar_rings(n)
    rays_per_design = hexapolar_count(rings)
    surfaces = len(optic.surfaces.surfaces)

    def call() -> Any:
        return bt.trace_batch(
            optic,
            [var],
            values,
            Hx=0.0,
            Hy=1.0,
            wavelength=wavelength,
            num_rays=rings,
            distribution="hexapolar",
            record="image",
        )

    reset_counters()
    before = counters()
    with kernel_timer() as kernel, no_grad():
        median, samples, result = timed(call, repeats, warmup=1)
        measured_peak = peak_bytes()
    deltas = counter_delta(before, counters())
    kernel_s = kernel["seconds"] / max(1, kernel["calls"])
    rows = result.to_numpy()
    row = result.rows[max(result.rows)]
    image = {
        k: np.asarray(rows[a], dtype=np.float64)[:, row, :]
        for k, a in (
            ("x", "x"),
            ("y", "y"),
            ("z", "z"),
            ("opd", "opd"),
            ("i", "intensity"),
        )
    }
    return {
        "median_s": median,
        "samples_s": samples,
        "image": image,
        "counters": deltas,
        "kernel_s": kernel_s,
        "peak_bytes": measured_peak,
        "fused": bool(result.fused),
        "launch_shared": bool(result.launch_shared),
        "B": b,
        "N": rays_per_design,
        "S": surfaces,
        "config": config,
        "variant": variant,
    }


def _noop_job() -> int:
    """A job that does nothing, for timing pool start-up (module-level: spawn)."""
    return 0


def pool_startup_seconds(workers: int) -> float:
    """Pool start-up alone: spawn, configure each worker, one round trip.

    One no-op job *per worker* rather than an empty job list:
    ``multiprocessing.Pool`` does not wait for its initializers, so an empty
    ``map`` would time the spawn call and not the backend configuration each
    worker has to finish before it can trace (plan 9.2 asks for the start-up
    that a real dispatch pays).
    """
    from optiland.parallel import WorkerConfig, evaluate_parallel

    wc = WorkerConfig(backend="numpy", threads=1)
    t0 = time.perf_counter()
    evaluate_parallel(_noop_job, [() for _ in range(workers)], workers=[wc] * workers)
    return time.perf_counter() - t0


def run_pool_batch(
    builder: Any,
    variant: str,
    b: int,
    n: int,
    workers: int,
    wavelength: float = WAVELENGTH,
) -> dict[str, Any]:
    """The same designs in ``workers`` chunks through ``evaluate_parallel``."""
    from optiland.parallel import WorkerConfig, evaluate_parallel

    optic = builder()
    var, values = batch_values(optic, variant, b)
    spec = BATCH_VARIABLE[variant]
    chunks = np.array_split(values, workers)
    jobs = [
        (optic.to_dict(), spec, [list(map(float, r)) for r in chunk], n, wavelength)
        for chunk in chunks
        if len(chunk)
    ]
    wc = WorkerConfig(backend="numpy", threads=1)
    t0 = time.perf_counter()
    evaluate_parallel(_batch_job, jobs, workers=[wc] * workers)
    wall = time.perf_counter() - t0
    del var
    return {"wall_s": wall, "workers": workers, "B": b}


# ---------------------------------------------------------------------------
# Provenance and status
# ---------------------------------------------------------------------------


def provenance() -> dict[str, Any]:
    """Host, toolchain, fork commit and mirror-table hash (plan 9.4)."""
    try:
        commit = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:  # pragma: no cover - git missing
        commit = ""
    try:
        from optiland.backend.torch_backend.metal import trace_mirror

        table = trace_mirror.table_hash()
    except Exception:  # noqa: BLE001 - provenance is reported, never fatal
        table = ""
    out: dict[str, Any] = {
        "host": platform.machine(),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "commit": commit,
        "mirror_table_hash": table,
        "gpu_cores": os.environ.get("OPTILAND_BENCH_GPU_CORES", ""),
    }
    try:
        import torch

        out["torch"] = torch.__version__
    except ImportError:  # pragma: no cover
        pass
    return out


def structural_reasons() -> frozenset[str]:
    """The six structural refusal reasons of plan 1.2, read from the gate.

    Structural refusals are the bundles that could never be kernel candidates
    -- a chief ray, a paraxial trace inside ``generate_rays``, a host-resident
    bundle -- and plan 1.3 says they are counted and never raised on.  Every
    realistic workload produces some (``SpotDiagram`` alone traces one
    paraxial ray per field), so a benchmark that failed a row for them would
    fail every row.  A *feature* refusal is the opposite: the bundle was a
    candidate and the kernel declined it, which is exactly what plan 9's
    ``FAIL(reason=...)`` rule exists to catch (``memory`` being its example).

    The set is imported rather than restated so it cannot drift from the gate.
    """
    try:
        from optiland.backend.torch_backend.metal.trace_adapters import (
            STRUCTURAL_REASONS,
        )

        return frozenset(reason.value for reason in STRUCTURAL_REASONS)
    except Exception:  # noqa: BLE001 - no Metal: no refusal can be counted
        return frozenset()


def row_status(config: str, correctness: dict[str, Any], deltas: dict[str, int]) -> str:
    """``ok`` or ``FAIL(reason=...)`` for one row (plan 9, opening paragraph)."""
    structural = structural_reasons()
    feature_skips = sorted(
        k.split(":", 1)[1]
        for k, v in deltas.items()
        if k.startswith("fused_trace_skip:")
        and v
        and k.split(":", 1)[1] not in structural
    )
    if is_fused(config):
        if feature_skips:
            return f"FAIL(reason={feature_skips[0]})"
        if deltas.get("fused_trace:traces", 0) <= 0:
            return "FAIL(reason=not_fused)"
        if deltas.get("fused_trace:late_fallback", 0):
            return "FAIL(reason=late_fallback)"
    if not correctness.get("pass", True):
        return "FAIL(reason=correctness)"
    return "ok"


# ---------------------------------------------------------------------------
# Targets (plan 9.3)
# ---------------------------------------------------------------------------

#: The predicted band of plan 9.3 T3, in surface-steps per second.
DF64_STEP_BAND = (3.9e8, 1.1e9)


def _median(records: list[dict], **match: Any) -> float | None:
    """The median seconds of the record matching ``match``, or None.

    Threadgroup-sweep rows are excluded (they carry ``group``), and when
    several ray counts match, the largest is taken -- a target that names no
    ``n`` means "at the workload's own size", not "whichever row came first".
    A row that is not ``ok`` yields None, so a failed row can never become a
    speed-up (plan section 9).
    """
    candidates = [
        rec
        for rec in records
        if not rec.get("group") and all(rec.get(k) == v for k, v in match.items())
    ]
    if not candidates:
        return None
    best = max(candidates, key=lambda rec: rec.get("n") or 0)
    if best.get("status", "ok") != "ok":
        return None
    return float(best["median_s"])


def _batch_rec(records: list[dict], **match: Any) -> dict | None:
    for rec in records:
        if rec.get("workload") != "trace_batch":
            continue
        if all(rec.get(k) == v for k, v in match.items()):
            return rec
    return None


def compute_targets(records: list[dict]) -> dict[str, Any]:
    """Plan 9.3's nine targets, each with its measured value and verdict."""
    targets: dict[str, Any] = {}

    fused = _median(
        records, workload="trace", config="mps-df64-fused", system="cooke", n=100000
    )
    reference = _median(
        records, workload="trace", config="numpy", system="cooke", n=100000
    )
    perop = _median(
        records, workload="trace", config="mps-df64", system="cooke", n=100000
    )
    targets["T1"] = {
        "what": "single fused trace beats one NumPy core at 1e5 rays",
        "value": {"fused_s": fused, "numpy_s": reference},
        "pass": bool(fused is not None and reference is not None and fused < reference),
    }
    ratio = (perop / fused) if (fused and perop) else None
    targets["T2"] = {
        "what": "fused vs per-op at 1e5 rays, >= 10x",
        "value": {"speedup": ratio, "perop_s": perop, "fused_s": fused},
        "pass": bool(ratio is not None and ratio >= 10.0),
    }

    def _steps(config: str) -> float | None:
        """``surface_steps_per_s`` of the plain (non-sweep) 1e6-ray row."""
        for rec in records:
            if (
                rec.get("workload") == "trace_group"
                and rec.get("config") == config
                and rec.get("n") == 1000000
                and not rec.get("group")
                and rec.get("status", "ok") == "ok"
            ):
                return rec.get("surface_steps_per_s")
        return None

    steps = _steps("mps-df64-fused")
    sf_steps = _steps("mps-sf64-fused")
    targets["T3"] = {
        "what": "kernel band at 1e6: df64 surface-steps/s in [3.9e8, 1.1e9]",
        "value": {
            "df64_steps_per_s": steps,
            "sf64_steps_per_s": sf_steps,
            "band": list(DF64_STEP_BAND),
        },
        "pass": bool(
            steps is not None and DF64_STEP_BAND[0] <= steps <= DF64_STEP_BAND[1]
        ),
        "reported_either_way": True,
    }
    for name, (b, n) in (
        ("T4", (1000, 1000)),
        ("T5", (100, 100000)),
        ("T6", (10000, 100)),
    ):
        gpu = _batch_rec(
            records, config="mps-df64-fused", variant="post", B=b, N_requested=n
        )
        pool = _batch_rec(
            records, config="numpy-8w", variant="post", B=b, N_requested=n
        )
        speed = None
        with_startup = None
        if gpu and pool and gpu.get("status", "ok") == "ok":
            gpu_rate = gpu["batch"]["design_traces_per_s"]
            pool_rate = pool["batch"]["design_traces_per_s"]
            speed = gpu_rate / pool_rate if pool_rate else None
            # The pool row's ``wall_s`` already carries its start-up, so the
            # "including start-up" ratio is that wall over the GPU's.
            gpu_wall = gpu["batch"]["wall_s"]
            with_startup = (pool["batch"]["wall_s"] / gpu_wall) if gpu_wall else None
        targets[name] = {
            "what": f"batched {b} x {n} vs eight NumPy workers, >= 3x",
            "value": {"speedup": speed, "speedup_with_pool_startup": with_startup},
            "pass": bool(speed is not None and speed >= 3.0),
        }
    failures = [
        f"{r.get('config')}/{r.get('workload')}/{r.get('system')}/"
        f"{r.get('n', r.get('B'))}: {r.get('status')}"
        for r in records
        if r.get("status", "ok") != "ok"
    ]
    targets["T7"] = {
        "what": "correctness everywhere: every summarised row passes",
        "value": {"failing_rows": failures},
        "pass": not failures,
    }
    memory = _batch_rec(
        records, config="mps-df64-fused", variant="post", B=100, N_requested=100000
    )
    pre = _batch_rec(
        records, config="mps-df64-fused", variant="pre", B=100, N_requested=100000
    )
    targets["T8"] = {
        "what": ("peak driver memory during 1e2 x 1e5, record=image, post < 1 GB"),
        "value": {
            "post_peak_bytes": memory["batch"]["peak_bytes"] if memory else None,
            "pre_peak_bytes": pre["batch"]["peak_bytes"] if pre else None,
        },
        "pass": bool(memory is not None and memory["batch"]["peak_bytes"] < 1024**3),
        "reported_either_way": True,
    }
    fused_fd = _median(
        records, workload="fd_jacobian_fused", config="mps-df64-fused", system="cooke"
    )
    numpy_fd = _median(records, workload="fd_jacobian", config="numpy", system="cooke")
    targets["T9"] = {
        "what": "fd_jacobian_fused vs the sequential NumPy fd_jacobian",
        "value": {
            "speedup": (numpy_fd / fused_fd) if (fused_fd and numpy_fd) else None,
            "fused_s": fused_fd,
            "numpy_s": numpy_fd,
        },
        "pass": bool(fused_fd is not None and numpy_fd is not None),
        "reported_either_way": True,
    }
    return targets


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

#: The memory table of plan 3.6: ``(label, predicted bytes, matcher)``.  The
#: matcher is the ``(B, N_requested, variant)`` of the batch row whose measured
#: peak fills the row in, or None for a case this harness does not measure (the
#: two single-trace cases need ``record="all"``, which no benchmark row uses).
MEMORY_TABLE: tuple[tuple[str, int, tuple[int, int, str] | None], ...] = (
    ("Cooke (S = 8), N = 1e6, all rows, final", 688 * 1024**2, None),
    ("UVProjectionLens (S ~ 45), N = 1e6, all rows, final", 3100 * 1024**2, None),
    (
        "batch 1e2 x 1e5, record=image, post (shared launch)",
        807 * 1024**2,
        (100, 100000, "post"),
    ),
    (
        "batch 1e2 x 1e5, record=image, pre (per-design launch)",
        1520 * 1024**2,
        (100, 100000, "pre"),
    ),
    ("batch 1e3 x 1e3, record=image", 152 * 1024**2, (1000, 1000, "post")),
    ("batch 1e4 x 1e2, record=image", 152 * 1024**2, (10000, 100, "post")),
)


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _ms(value: Any) -> str:
    return "-" if value is None else f"{float(value) * 1e3:.2f}"


def render_markdown(reports: list[dict], path: str) -> None:
    """Write the report tables of plan 9.4 from one or more benchmark JSONs."""
    records: list[dict] = []
    provenances: list[dict] = []
    for report in reports:
        records.extend(report.get("results", []))
        provenances.append(report.get("provenance", {}))
    targets = compute_targets(records)

    out: list[str] = ["# Fused trace benchmark", ""]
    prov = provenances[0] if provenances else {}
    out.append(
        f"Host `{prov.get('host', '')}` macOS {prov.get('macos', '')}, "
        f"torch {prov.get('torch', '')}, python {prov.get('python', '')}, "
        f"fork `{prov.get('commit', '')}`, mirror table "
        f"`{str(prov.get('mirror_table_hash', ''))[:16]}`."
    )
    out.append("")
    out.append(
        "Every row carries its own correctness record; a row whose record does "
        "not pass is marked `FAIL(...)` and is excluded from every speed-up "
        "below (plan section 9)."
    )

    single = [
        r
        for r in records
        if r.get("workload")
        in ("trace", "trace_group", "spot", "fd_jacobian", "fd_jacobian_fused")
        and not r.get("group")
    ]
    if single:
        out += ["", "## 1. Single-design workloads", ""]
        out.append(
            "| system | workload | N | config | median (ms) | max abs diff vs "
            "numpy | worst / bound | status |"
        )
        out.append("|---|---|---:|---|---:|---:|---|---|")
        for rec in single:
            worst = rec.get("correctness", {}).get("worst", "")
            diffs = []
            for part in rec.get("correctness", {}).get("parts", []):
                diffs += list(part.get("max_abs_diff", {}).values())
            out.append(
                f"| {rec.get('system', '')} | {rec['workload']} | "
                f"{rec.get('n', '')} | {rec['config']} | {_ms(rec.get('median_s'))} | "
                f"{_fmt(max(diffs) if diffs else None)} | {worst} | "
                f"{rec.get('status', 'ok')} |"
            )

    batch = [r for r in records if r.get("workload") == "trace_batch"]
    if batch:
        out += ["", "## 2. Batched designs", ""]
        out.append(
            "| config | variant | B | N | wall (s) | design-traces/s | rays/s | "
            "surface-steps/s | kernel (s) | peak (MiB) | shared launch | status |"
        )
        out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for rec in batch:
            b = rec["batch"]
            out.append(
                f"| {rec['config']} | {rec.get('variant', '')} | {b.get('B')} | "
                f"{b.get('N')} | {_fmt(b.get('wall_s'))} | "
                f"{_fmt(b.get('design_traces_per_s'))} | {_fmt(b.get('rays_per_s'))} | "
                f"{_fmt(b.get('surface_steps_per_s'))} | {_fmt(b.get('kernel_s'))} | "
                f"{_fmt((b.get('peak_bytes') or 0) / 1024**2)} | "
                f"{b.get('launch_shared', '-')} | {rec.get('status', 'ok')} |"
            )

    sweep = [
        r for r in records if r.get("workload") == "trace_group" and r.get("group")
    ]
    if sweep:
        out += ["", "## 3. Threadgroup sweep", ""]
        out.append(
            "| config | system | N | threadgroup | median (ms) | surface-steps/s |"
        )
        out.append("|---|---|---:|---:|---:|---:|")
        for rec in sweep:
            out.append(
                f"| {rec['config']} | {rec.get('system', '')} | {rec.get('n')} | "
                f"{rec['group']} | {_ms(rec.get('median_s'))} | "
                f"{_fmt(rec.get('surface_steps_per_s'))} |"
            )

    out += ["", "## 4. Targets (plan 9.3)", ""]
    out.append("| target | what | measured | verdict |")
    out.append("|---|---|---|---|")
    for name in sorted(targets):
        entry = targets[name]
        verdict = "PASS" if entry["pass"] else "FAIL"
        if entry.get("reported_either_way"):
            verdict += " (reported either way)"
        out.append(
            f"| {name} | {entry['what']} | "
            f"`{json.dumps(entry['value'], default=_fmt)}` | {verdict} |"
        )

    out += ["", "## 5. Predicted band vs measured (plan 9.3 T3)", ""]
    value = targets["T3"]["value"]
    out.append(
        f"df64 kernel throughput at 1e6 rays: measured "
        f"{_fmt(value.get('df64_steps_per_s'))} surface-steps/s against the "
        f"predicted band [{DF64_STEP_BAND[0]:.2g}, {DF64_STEP_BAND[1]:.2g}]; "
        f"sf64 {_fmt(value.get('sf64_steps_per_s'))}."
    )

    out += ["", "## 6. Memory (plan 3.6)", ""]
    out.append("| case | predicted | measured peak |")
    out.append("|---|---:|---:|")
    measured: dict[tuple[int, int, str], int] = {}
    for rec in batch:
        if not rec.get("config", "").endswith("fused"):
            continue
        key = (int(rec["B"]), int(rec["N_requested"]), str(rec.get("variant")))
        peak = rec["batch"].get("peak_bytes")
        if peak:
            measured[key] = max(measured.get(key, 0), int(peak))
    for label, predicted, matcher in MEMORY_TABLE:
        found = measured.get(matcher) if matcher else None
        note = "" if matcher else " (not measured by this harness)"
        out.append(
            f"| {label}{note} | {predicted / 1024**2:.0f} MiB | "
            f"{'-' if found is None else f'{found / 1024**2:.0f} MiB'} |"
        )

    totals: dict[str, int] = {}
    for rec in records:
        for key, value in (rec.get("counters") or {}).items():
            if key.startswith(("fused_trace", "gpu:fused_trace")):
                totals[key] = totals.get(key, 0) + int(value)
    out += ["", "## 7. `fused_trace:*` totals over the run", "", "```"]
    out += [f"{k}: {v}" for k, v in sorted(totals.items())] or ["(none)"]
    out += ["```"]

    failures = targets["T7"]["value"]["failing_rows"]
    out += ["", "## 8. Documented limits and failed rows", ""]
    if failures:
        out += [f"- {line}" for line in failures]
    else:
        out.append("- No row failed its correctness record or its counter check.")

    out += ["", "## 9. Summary", ""]
    passed = [n for n in sorted(targets) if targets[n]["pass"]]
    failed = [n for n in sorted(targets) if not targets[n]["pass"]]
    out.append(f"Targets met: {', '.join(passed) or 'none'}.")
    out.append(f"Targets not met: {', '.join(failed) or 'none'}.")
    out.append(
        "T1, T2, T4, T5 and T7 are the required ones (plan 9.3); T3, T6, T8 and "
        "T9 are reported either way."
    )
    Path(path).write_text("\n".join(out) + "\n")
    print("wrote", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_batch(spec: str) -> tuple[int, int]:
    """``"1000x1000"`` -> ``(1000, 1000)``."""
    b, _, n = spec.lower().partition("x")
    return int(b), int(n)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--configs", nargs="+", default=["numpy", "torch-cpu-f64", "mps-df64"]
    )
    parser.add_argument("--systems", nargs="+", default=["cooke"])
    parser.add_argument("--rays", nargs="*", type=int, default=[1000, 10000, 100000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--wavelength", type=float, default=WAVELENGTH)
    parser.add_argument(
        "--workloads",
        nargs="*",
        default=["trace", "spot", "fd_jacobian"],
        help="empty (--workloads with no value) runs only --batch / --group-sweep",
    )
    parser.add_argument(
        "--batch",
        nargs="+",
        default=[],
        metavar="BxN",
        help="batched-design workloads, e.g. 1000x1000 100x100000",
    )
    parser.add_argument(
        "--batch-variants",
        nargs="+",
        default=list(BATCH_VARIANTS),
        choices=BATCH_VARIANTS,
    )
    parser.add_argument(
        "--group-sweep",
        nargs="+",
        type=int,
        default=[],
        help="threadgroup sizes for the trace_group workload",
    )
    parser.add_argument("--parallel-workers", nargs="*", type=int, default=[])
    parser.add_argument("--out", default=None)
    parser.add_argument("--md", default=None, help="render the report to this file")
    parser.add_argument(
        "--from-json",
        nargs="+",
        default=[],
        help="render --md from these JSONs instead of measuring",
    )
    return parser


def _single_rows(args, report: dict, sysmap: dict, oracle: dict) -> None:
    """Measure every (system, N, workload, config) row of ``args``."""
    import optiland.backend as be

    for sys_name in args.systems:
        for n in args.rays:
            for workload in args.workloads:
                for config in args.configs:
                    if (
                        CONFIGS[config].get("workers")
                        and workload not in POOL_WORKLOADS
                    ):
                        continue
                    if CONFIGS[config].get("workers"):
                        continue  # pool rows are produced by the batch section
                    rec = _one_row(args, sysmap, oracle, sys_name, n, workload, config)
                    report["results"].append(rec)
                    be.set_backend("numpy")


def _measure(
    optic, n, workload, repeats, wavelength
) -> tuple[float, list[float], dict, dict]:
    """Run one workload: ``(median, samples, payload, extra)``.

    The payload is what the correctness record compares, so the oracle and the
    measured row are produced by the *same* function -- a reference computed by
    a second code path would compare two implementations, not two backends.
    """
    extra: dict[str, Any] = {}
    with no_grad():
        if workload == "trace":
            median, samples, result = timed(lambda: trace_workload(optic, n), repeats)
            payload = {
                "rays": rays_to_numpy(result),
                "image": image_row_to_numpy(optic),
            }
        elif workload == "trace_group":
            rays = launch_bundle(optic, n, wavelength)
            median, samples, result = timed(
                lambda: trace_group_workload(optic, rays), repeats
            )
            payload = {
                "rays": rays_to_numpy(result),
                "image": image_row_to_numpy(optic),
            }
            steps = payload["rays"]["x"].size * (len(optic.surfaces.surfaces) - 1)
            extra["surface_steps"] = int(steps)
            extra["surface_steps_per_s"] = steps / median if median else None
        elif workload == "spot":
            median, samples, result = timed(lambda: spot_workload(optic, n), repeats)
            payload = {"spot": np.asarray(result, dtype=np.float64)}
        elif workload == "fd_jacobian":
            median, samples, result = timed(
                lambda: fd_jacobian_workload(optic, n), max(1, repeats // 2)
            )
            payload = {"grad": np.asarray(result, dtype=np.float64)}
            extra["denominators"] = fd_sequential_denominators(optic).tolist()
            extra["merit_scale"] = float(np.mean(spot_workload(optic, n)))
        elif workload == "fd_jacobian_fused":
            median, samples, result = timed(
                lambda: fd_jacobian_fused_workload(optic, n, wavelength),
                max(1, repeats // 2),
            )
            payload = {"grad": np.asarray(result.jacobian, dtype=np.float64)}
            extra["denominators"] = np.asarray(result.steps, dtype=np.float64).tolist()
            extra["merits"] = np.asarray(result.merits, dtype=np.float64).tolist()
            extra["merit_scale"] = float(np.max(np.abs(result.merits)))
            extra["fused_batch"] = bool(getattr(result.result, "fused", False))
        else:
            raise ValueError(workload)
    return median, samples, payload, extra


def ensure_oracle(sysmap, oracle, sys_name, n, workload, wavelength) -> dict[str, Any]:
    """The NumPy float64 reference for ``(system, workload, n)``, computed once.

    Plan section 9 requires a correctness record on *every* timed row, and the
    group sweep of 9.4 names no NumPy configuration, so the oracle is produced
    here whether or not ``numpy`` is in ``--configs``.  It is run untimed
    (one repeat, no warm-up): only its values are used.
    """
    key = (sys_name, workload, n)
    if key in oracle:
        return oracle[key]
    apply_config("numpy")
    optic = sysmap[sys_name]()
    _, _, payload, _ = _measure(optic, n, workload, 1, wavelength)
    if workload == "fd_jacobian_fused":
        payload["grad_cpu_small_step"] = fd_reference_small_step(optic, n, wavelength)
    oracle[key] = payload
    return payload


def fd_reference_small_step(optic: Any, n: int, wavelength: float) -> np.ndarray:
    """``batch_trace.fd_reference_cpu`` at ``FD_STEP_SMALL`` (plan 9.2).

    Reported beside every ``fd_jacobian_fused`` row as
    ``max_abs_diff_vs_fd_reference_cpu``.  It is evidence, not the pass rule:
    it uses a *different* stencil, so it carries a different truncation error
    and cannot separate representation noise the way the same-stencil
    comparison does.
    """
    from optiland.raytrace import batch_trace as bt

    variables = fd_variables(optic)
    return np.asarray(
        bt.fd_reference_cpu(
            optic,
            variables,
            rms_merit,
            FD_STEP_SMALL,
            Hx=0.0,
            Hy=1.0,
            wavelength=wavelength,
            num_rays=hexapolar_rings(n),
            distribution="hexapolar",
            record="image",
        ),
        dtype=np.float64,
    )


def _one_row(args, sysmap, oracle, sys_name, n, workload, config) -> dict[str, Any]:
    """One timed row plus its correctness record and counter deltas."""
    reference = ensure_oracle(sysmap, oracle, sys_name, n, workload, args.wavelength)
    apply_config(config)
    optic = sysmap[sys_name]()
    reset_counters()
    before = counters()
    median, samples, payload, extra = _measure(
        optic, n, workload, args.repeats, args.wavelength
    )
    deltas = counter_delta(before, counters())
    eps = eps_of(config)
    if workload in ("trace", "trace_group"):
        correctness = merge_records(
            external_rule(payload["rays"], reference["rays"], eps=eps, label="rays"),
            external_rule(
                payload["image"], reference["image"], eps=eps, label="image row"
            ),
        )
    elif workload == "spot":
        scale = float(np.max(np.abs(reference["spot"])))
        correctness = merge_records(
            array_record(
                payload["spot"],
                reference["spot"],
                bound=max(EXTERNAL_POS_ABS, EXTERNAL_FACTOR * eps * scale),
                label="rms",
            )
        )
    else:
        bound = fd_bound(extra.get("merit_scale", 1.0), extra["denominators"], eps)
        correctness = merge_records(
            array_record(payload["grad"], reference["grad"], bound=bound, label="J")
        )
        extra["fd_bound"] = bound
        cpu = reference.get("grad_cpu_small_step")
        if cpu is not None:
            extra["max_abs_diff_vs_fd_reference_cpu"] = float(
                np.max(np.abs(payload["grad"] - cpu))
            )
    status = row_status(config, correctness, deltas)
    rec: dict[str, Any] = {
        "system": sys_name,
        "workload": workload,
        "n": n,
        "config": config,
        "median_s": median,
        "samples_s": samples,
        "correctness": correctness,
        "counters": deltas,
        "status": status,
        **extra,
    }
    worst = correctness.get("worst", "")
    print(
        f"{sys_name:18s} {workload:18s} n={n:<8d} {config:15s} "
        f"median {median * 1e3:9.2f} ms  {status:24s} {worst}",
        flush=True,
    )
    return rec


def _group_sweep_rows(args, report: dict, sysmap: dict, oracle: dict) -> None:
    """The ``trace_group`` workload once per threadgroup size (plan 9.4)."""
    import optiland.backend as be

    previous = os.environ.get("OPTILAND_METAL_FUSED_TRACE_GROUP")
    try:
        for group in args.group_sweep:
            os.environ["OPTILAND_METAL_FUSED_TRACE_GROUP"] = str(group)
            for sys_name in args.systems:
                for n in args.rays:
                    for config in args.configs:
                        if not is_fused(config):
                            continue
                        rec = _one_row(
                            args, sysmap, oracle, sys_name, n, "trace_group", config
                        )
                        rec["group"] = group
                        report["results"].append(rec)
                        be.set_backend("numpy")
    finally:
        if previous is None:
            os.environ.pop("OPTILAND_METAL_FUSED_TRACE_GROUP", None)
        else:
            os.environ["OPTILAND_METAL_FUSED_TRACE_GROUP"] = previous


def _batch_rows(args, report: dict, sysmap: dict) -> None:
    """The ``--batch`` rows: NumPy reference, worker pool, and the fused kernel."""
    import optiland.backend as be

    for spec in args.batch:
        b, n = parse_batch(spec)
        for sys_name in args.systems:
            builder = sysmap[sys_name]
            if not batch_design_set_applies(builder, args.batch_variants):
                print(
                    f"{sys_name}: skipped for --batch; plan 9.2's design set is "
                    f"the CookeTriplet's (surfaces "
                    f"{sorted({i for _, i in BATCH_VARIABLE.values()})}) and this "
                    f"system does not have them",
                    flush=True,
                )
                continue
            for variant in args.batch_variants:
                startup: dict[int, float] = {}
                # The NumPy sequential loop is both the reference row and the
                # oracle every fused row's per-design record is judged against
                # (plan 9.2), so it runs whether or not `numpy` is in
                # --configs; only the timed ROW is conditional.
                apply_config("numpy")
                optic = builder()
                wall, reference = numpy_batch_reference(
                    optic, variant, b, n, args.wavelength
                )
                if "numpy" in args.configs:
                    report["results"].append(
                        _batch_record(
                            "numpy", sys_name, variant, b, n, wall, reference, optic
                        )
                    )
                for config in args.configs:
                    cfg = CONFIGS[config]
                    apply_config(config)
                    if config == "numpy":
                        continue
                    if cfg.get("workers"):
                        workers = int(cfg["workers"])
                        if workers not in startup:
                            startup[workers] = pool_startup_seconds(workers)
                        pool = run_pool_batch(
                            builder, variant, b, n, workers, args.wavelength
                        )
                        rec = {
                            "system": sys_name,
                            "workload": "trace_batch",
                            "config": config,
                            "variant": variant,
                            "B": b,
                            "N_requested": n,
                            "median_s": pool["wall_s"],
                            "correctness": {"pass": True, "parts": [], "worst": "n/a"},
                            "counters": {},
                            "status": "ok",
                            "batch": {
                                "B": b,
                                "N": hexapolar_count(hexapolar_rings(n)),
                                "variant": variant,
                                # ``wall_s`` is one whole ``evaluate_parallel``
                                # call and so INCLUDES pool start-up;
                                # ``steady_wall_s`` takes the separately
                                # measured start-up back off, and that is the
                                # steady state plan 9.3's T4-T6 compare
                                # against.  Both are reported (plan 9.2).
                                "wall_s": pool["wall_s"],
                                "pool_startup_s": startup[workers],
                                "steady_wall_s": max(
                                    pool["wall_s"] - startup[workers], 0.0
                                ),
                                "design_traces_per_s": (
                                    b / max(pool["wall_s"] - startup[workers], 1e-12)
                                ),
                                "design_traces_per_s_with_startup": b / pool["wall_s"],
                                "workers": workers,
                            },
                        }
                        report["results"].append(rec)
                        print(
                            f"{sys_name:18s} trace_batch        B={b:<7d} "
                            f"{config:15s} wall {pool['wall_s']:8.3f} s  "
                            f"(pool start-up {startup[workers]:.2f} s)",
                            flush=True,
                        )
                    elif is_metal(config):
                        measured = run_trace_batch(
                            config,
                            builder,
                            variant,
                            b,
                            n,
                            args.repeats,
                            args.wavelength,
                        )
                        rec = _fused_batch_record(
                            config, sys_name, variant, b, n, measured, reference
                        )
                        report["results"].append(rec)
                    be.set_backend("numpy")


def _batch_record(config, sys_name, variant, b, n, wall, rows, optic) -> dict[str, Any]:
    """The NumPy sequential reference row."""
    rays = int(rows["x"].shape[1])
    surfaces = len(optic.surfaces.surfaces)
    print(
        f"{sys_name:18s} trace_batch        B={b:<7d} {config:15s} wall {wall:8.3f} s",
        flush=True,
    )
    return {
        "system": sys_name,
        "workload": "trace_batch",
        "config": config,
        "variant": variant,
        "B": b,
        "N_requested": n,
        "median_s": wall,
        "correctness": {"pass": True, "parts": [], "worst": "oracle"},
        "counters": {},
        "status": "ok",
        "batch": {
            "B": b,
            "N": rays,
            "S": surfaces,
            "variant": variant,
            "wall_s": wall,
            "design_traces_per_s": b / wall if wall else None,
            "rays_per_s": b * rays / wall if wall else None,
        },
    }


def _fused_batch_record(
    config, sys_name, variant, b, n, measured, reference
) -> dict[str, Any]:
    """One fused ``trace_batch`` row with its per-design correctness record."""
    wall = measured["median_s"]
    rays = measured["N"]
    steps = b * rays * (measured["S"] - 1)
    # Plan 9: "a row without a passing record is reported FAIL and never
    # summarised as a speed-up".  No NumPy reference in --configs means no
    # record at all, which is a failure of the run's setup, not a pass.
    correctness: dict[str, Any] = {
        "pass": False,
        "parts": [],
        "worst": "no numpy reference row in --configs",
    }
    worst_design = None
    max_over_designs = None
    if reference is not None:
        eps = eps_of(config)
        parts = []
        worst_ratio = -1.0
        for design in range(b):
            got = {k: v[design] for k, v in measured["image"].items()}
            ref = {k: v[design] for k, v in reference.items()}
            part = external_rule(got, ref, eps=eps, label=f"design {design}")
            if part["worst_ratio"] > worst_ratio:
                worst_ratio = part["worst_ratio"]
                worst_design = design
                max_over_designs = max(part["max_abs_diff"].values(), default=0.0)
            if not part["pass"]:
                parts.append(part)
        if not parts:
            parts = [
                external_rule(
                    {k: v[worst_design] for k, v in measured["image"].items()},
                    {k: v[worst_design] for k, v in reference.items()},
                    eps=eps,
                    label=f"worst design {worst_design}",
                )
            ]
        correctness = merge_records(*parts)
    status = row_status(config, correctness, measured["counters"])
    if is_fused(config) and not measured["fused"]:
        status = "FAIL(reason=not_fused)"
    print(
        f"{sys_name:18s} trace_batch        B={b:<7d} {config:15s} "
        f"wall {wall:8.3f} s  kernel {measured['kernel_s']:7.3f} s  "
        f"{status}",
        flush=True,
    )
    return {
        "system": sys_name,
        "workload": "trace_batch",
        "config": config,
        "variant": variant,
        "B": b,
        "N_requested": n,
        "median_s": wall,
        "samples_s": measured["samples_s"],
        "correctness": correctness,
        "counters": measured["counters"],
        "status": status,
        "batch": {
            "B": b,
            "N": rays,
            "S": measured["S"],
            "variant": variant,
            "launch_shared": measured["launch_shared"],
            "wall_s": wall,
            "kernel_s": measured["kernel_s"],
            "host_s": wall - measured["kernel_s"],
            "design_traces_per_s": b / wall if wall else None,
            "rays_per_s": b * rays / wall if wall else None,
            "surface_steps_per_s": steps / measured["kernel_s"]
            if measured["kernel_s"]
            else None,
            "peak_bytes": measured["peak_bytes"],
            "worst_design": worst_design,
            "max_over_designs": max_over_designs,
        },
    }


def main() -> None:
    args = build_parser().parse_args()

    if args.from_json:
        reports = [json.loads(Path(p).read_text()) for p in args.from_json]
        if not args.md:
            raise SystemExit("--from-json needs --md")
        render_markdown(reports, args.md)
        return

    import optiland.backend as be

    report: dict[str, Any] = {
        "provenance": provenance(),
        "args": {
            k: v for k, v in vars(args).items() if k not in ("out", "md", "from_json")
        },
        "results": [],
    }
    sysmap = systems()
    oracle: dict[tuple[str, str, int], Any] = {}

    if args.workloads and args.rays:
        _single_rows(args, report, sysmap, oracle)
    if args.group_sweep:
        _group_sweep_rows(args, report, sysmap, oracle)
    if args.batch:
        _batch_rows(args, report, sysmap)

    if args.parallel_workers:
        from optiland.parallel import WorkerConfig, evaluate_parallel

        for config in args.configs:
            cfg = CONFIGS[config]
            wc = WorkerConfig(
                device=cfg.get("device", "cpu"),
                precision=cfg.get("precision", "float64"),
                backend=cfg["backend"],
                metal_mode=cfg.get("mode", "df64"),
            )
            for workers in args.parallel_workers:
                jobs = [
                    (sysmap[args.systems[0]]().to_dict(), args.rays[0])
                    for _ in range(8)
                ]
                t0 = time.perf_counter()
                evaluate_parallel(_parallel_job, jobs, workers=[wc] * workers)
                elapsed = time.perf_counter() - t0
                report["results"].append(
                    {
                        "workload": "parallel_fd_jacobian_x8jobs",
                        "config": config,
                        "workers": workers,
                        "wall_s": elapsed,
                    }
                )
                print(
                    f"parallel {config:14s} workers={workers}: "
                    f"8 fd_jacobian jobs in {elapsed:.2f} s",
                    flush=True,
                )

    report["targets"] = compute_targets(report["results"])
    be.set_backend("numpy")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, default=float))
        print("wrote", args.out)
    if args.md:
        render_markdown([report], args.md)

    failing = [r for r in report["results"] if r.get("status", "ok") != "ok"]
    if failing:
        print(f"{len(failing)} row(s) FAILED their correctness or counter check")


def _parallel_job(optic_dict: Any, n: int) -> list[float]:
    """The historical ``--parallel-workers`` job (one fd_jacobian per job)."""
    from optiland.optic import Optic

    return fd_jacobian_workload(Optic.from_dict(optic_dict), n).tolist()


if __name__ == "__main__":
    main()
