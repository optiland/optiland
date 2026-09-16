"""End-to-end oracle: sample systems on NumPy float64 vs emulated float64 (mps).

For every selected system the script runs, on the NumPy backend (the float64
oracle), on torch CPU float64 (the oracle's own operation-order noise floor,
same code path as mps) and on the torch backend with ``device='mps'`` /
``precision='float64'`` in every requested representation (``df64``,
``sf64``):

* **traces**: a hexapolar bundle at the normalized fields (0, 0), (0, 0.7),
  (0, 1) and three wavelengths, recording every surface
  (``optic.surfaces.x/y/z/L/M/N/opd/intensity``);
* **analyses**: ``SpotDiagram`` RMS radii, ``RayFan`` aberration curves,
  ``OPD`` RMS and maps, ``GeometricMTF`` at a few frequencies, paraxial
  quantities (f1, f2, EPD, XPD, FNO, magnification) and ``Distortion``.

Per quantity it reports the maximum absolute difference, the difference
relative to the quantity's magnitude, NaN (vignetting) agreement, and, for
the emulated runs, the Metal op counters accumulated per phase (GPU
launches ``gpu:<kernel>``, host ops ``host:<op>``, CPU fallbacks
``cpu_fallback:<op>`` with the Optiland call site that triggered each one).

Pass criterion per system and mode: the max trace position difference is
below ``--tol`` (1e-11 mm) **or** below ``--tol-rel`` times the system's
path scale (default 64 machine epsilons of the representation), and the NaN
patterns agree. The relative clause exists for long-path systems such as the
Hubble telescope, where torch CPU float64 itself differs from NumPy by more
than 1e-11 mm; the CPU floor is printed next to every emulated number.

Usage (from the project root)::

    .venv/bin/python Optiland-Metal/scripts/metal_oracle_e2e.py \
        [--systems all|CookeTriplet,HubbleTelescope,...] [--modes df64,sf64] \
        [--rings 16] [--tol 1e-11] [--tol-rel 2.3e-13] [--no-cpu-floor] \
        [--no-analyses] [--grad] [--json NOTES/oracle-e2e.json] \
        [--md NOTES/06-oracle-report.md] [--from-json NOTES/oracle-e2e.json]

The exit status is non-zero when any system fails the criterion, when NaN
patterns disagree, or when an emulated run raised.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
import traceback
from collections import Counter
from typing import Any

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import numpy as np  # noqa: E402

import optiland.backend as be  # noqa: E402

QUANTITIES = ("x", "y", "z", "L", "M", "N", "opd", "i")
POSITION = ("x", "y", "z")
_ATTR = {q: ("intensity" if q == "i" else q) for q in QUANTITIES}
MACHINE_EPS = {"df64": 2.0**-48, "sf64": 2.0**-53}
FIELDS: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.0, 0.7), (0.0, 1.0))
PARAXIAL = ("f1", "f2", "EPD", "XPD", "FNO", "magnification")
PHASES = ("trace", "spot", "rayfan", "opd", "mtf", "paraxial", "distortion")
# analysis sampling (small enough for a few minutes per mode over all systems)
RAYFAN_POINTS = 257
OPD_RINGS = 16
MTF_RAYS = 32  # uniform grid per axis
MTF_POINTS = 32  # frequency samples and histogram bins
MTF_REPORT_INDEX = (4, 8, 16)  # frequencies quoted in the markdown table
DISTORTION_POINTS = 16
SPOT_RINGS = 16

# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------


def _sample(module: str, cls: str) -> Any:
    mod = __import__(module, fromlist=[cls])
    return getattr(mod, cls)()


def _zernike_singlet() -> Any:
    """Singlet with a Zernike (fringe) front surface, built here.

    ``optiland.samples`` has no Zernike or polynomial sample, so this mirrors
    the singlet used by ``tests/test_optic_deprecated.py`` with non-zero
    astigmatism (Z5), coma (Z8) and spherical (Z9) terms so that the
    Newton-Raphson intersection and the Zernike sag/normal code run.
    """
    from optiland.optic import Optic

    lens = Optic()
    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    lens.surfaces.add(
        index=1,
        surface_type="zernike",
        radius=50.0,
        thickness=5.0,
        is_stop=True,
        material="N-BK7",
        norm_radius=10.0,
        coefficients=[0.0, 0.0, 0.0, 0.0, 2e-3, 0.0, 0.0, 1e-3, 5e-4],
    )
    lens.surfaces.add(index=2, radius=-50.0, thickness=45.0)
    lens.surfaces.add(index=3)
    lens.set_aperture("EPD", 10)
    lens.fields.set_type("angle")
    lens.fields.add(y=0)
    lens.fields.add(y=3)
    lens.wavelengths.add(0.55, is_primary=True)
    return lens


def _polynomial_singlet() -> Any:
    """Singlet with an XY-polynomial front surface (C20, C02, C22, C40 terms)."""
    from optiland.optic import Optic

    lens = Optic()
    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    lens.surfaces.add(
        index=1,
        surface_type="polynomial",
        radius=50.0,
        thickness=5.0,
        is_stop=True,
        material="N-BK7",
        coefficients=[
            [0.0, 0.0, 1e-4, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [2e-4, 0.0, 5e-6, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-2e-6, 0.0, 0.0, 0.0, 0.0],
        ],
    )
    lens.surfaces.add(index=2, radius=-50.0, thickness=45.0)
    lens.surfaces.add(index=3)
    lens.set_aperture("EPD", 10)
    lens.fields.set_type("angle")
    lens.fields.add(y=0)
    lens.fields.add(y=3)
    lens.wavelengths.add(0.55, is_primary=True)
    return lens


SYSTEMS: dict[str, Any] = {
    "CookeTriplet": lambda: _sample("optiland.samples.objectives", "CookeTriplet"),
    "ReverseTelephoto": lambda: _sample(
        "optiland.samples.objectives", "ReverseTelephoto"
    ),
    "HubbleTelescope": lambda: _sample(
        "optiland.samples.telescopes", "HubbleTelescope"
    ),
    "AsphericSinglet": lambda: _sample("optiland.samples.simple", "AsphericSinglet"),
    "ZernikeSinglet": _zernike_singlet,
    "PolynomialSinglet": _polynomial_singlet,
}
SYSTEM_NOTES = {
    "AsphericSinglet": "even asphere (optiland.samples.simple); +5 deg field and "
    "±0.05 µm wavelengths added for the multi-field/-wavelength sweep",
    "HubbleTelescope": "mirrors only; ±0.05 µm wavelengths added",
    "ZernikeSinglet": "built in the script (no Zernike sample in optiland.samples)",
    "PolynomialSinglet": "built in the script (no polynomial sample in "
    "optiland.samples)",
}

# Findings that explain every non-trivial number in the report. They were
# measured with this script (see the ``--grad`` and ``--no-analyses`` flags and
# the investigations recorded in NOTES/06-oracle-report.md section 6).
REPORT_NOTES: tuple[str, ...] = (
    "**df64 traces** sit at 2-4 eps (2^-48) of the path scale on the five lens "
    "systems and 17.5 eps on the Hubble telescope (path 16 m, radii 11 m); the "
    "absolute Hubble number (4.7e-10 mm) is 32x the torch-cpu-vs-numpy floor "
    "(1.5e-11 mm), i.e. exactly the 2^(53-48) ratio of the two significands, so "
    "there is no dispatch error, only the shorter df64 significand.",
    "**sf64 traces** agree with numpy to 1.7-13 eps (2^-53) of the path scale, "
    "i.e. 2e-16 to 1.4e-15 relative, on every system. On Hubble the sf64 error "
    "(1.09e-11 mm) is *below* the torch-cpu-vs-numpy floor (1.46e-11 mm): sf64 "
    "is at the operation-order floor of float64 itself.",
    "**Why sf64 is not bit-identical with torch-cpu float64** (Cooke: 1.03e-13 "
    "mm = 1.3e-15 relative; the deviation already appears on the object "
    "surface, i.e. in ray generation): the v1 sf64 transcendentals are the "
    "inexact bridge through df64 (`codegen.SF64_INEXACT`: sin, cos, tan, atan2, "
    "asin, acos, exp, log, pow, hypot, ...), rounded to 48 bits, not 53. The "
    "trace path launches `tan`/`cos`/`sin` (field angles to direction cosines "
    "and ray starts), `pow`/`exp` (dispersion formulas) and the fused "
    "`conic_candidates` kernel on the bundle. Arithmetic (`add/sub/mul/div/sqrt`) "
    "and reductions are correctly rounded and match the CPU bit for bit; the "
    "residual ~1e-15 relative offset is the transcendental bridge (milestone M4 "
    "replaces it with a native musl port) plus numpy-vs-torch operation order "
    "(the torch-cpu floor column, 5e-15 to 3.6e-14 mm on the lens systems).",
    "**Autograd changes the forward value of Newton-Raphson geometries.** With "
    "`be.grad_mode` enabled the torch backend (CPU and mps alike) applies the "
    "DiffOptics one-step implicit correction `t - F(t)/F'(t)` after the primal "
    "solve (`geometries/newton_raphson.py: distance`). The primal solve stops at "
    "the configured `tol` (1e-6 for the even_asphere / zernike / polynomial "
    "configs), so that extra Newton step moves positions by up to ~tol: "
    "`--grad --no-analyses` measured, vs numpy, AsphericSinglet 1.0e-6 mm "
    "(both modes), PolynomialSinglet 5.3e-10 mm, ZernikeSinglet 4.0e-11 mm, "
    "CookeTriplet unchanged (no NR surface), while sf64 vs torch-cpu stayed at "
    "1.0e-14 to 2.0e-14 mm on the same runs. It is an Optiland code-path "
    "difference, not an emulation error; the report therefore runs torch with "
    "autograd disabled so numpy and torch execute the same primal path.",
    "**ZernikeSinglet df64 distortion (3.5e-7 %, rel 2e-5)** is conditioning of "
    "Optiland's rotational distortion model, not a dispatch error: the plate "
    "scale is fitted from a chief ray at Hy = 1e-10 (`distortion_strategies/"
    "model.py: RotationalDistortionModel.fit`), dividing `y_ref - y_c` = 2.4e-10 "
    "mm. In df64 the image height of that near-axis chief ray differs from "
    "numpy by exactly one float64 ulp of y (8.7e-19 mm at the 0.50 and 0.60 um "
    "wavelengths; the fused `conic_candidates` kernel runs even for 1-ray "
    "batches), which the fit amplifies 4e9-fold into a 3.5e-9 relative plate "
    "scale error; sf64 reproduces numpy's ulp and shows 4e-14 %. Every other "
    "system is at 1e-13 % or better.",
    "**OPD maps in waves** carry the optical-path error divided by the "
    "wavelength (0.55e-3 mm): df64 1e-9 waves on the lens systems and 5e-7 "
    "waves on Hubble (OPL ~16 m), sf64 1e-11 / 1.3e-8 waves; the OPD rms values "
    "agree to 1.3e-10 (df64) and 1.4e-11 (sf64) waves on the lens systems and "
    "4e-8 / 1e-10 waves on Hubble.",
    "**Vignetting patterns**: no ray of this sweep produced NaN (no sample has "
    "rays missing a surface), so the NaN check is vacuous here; the "
    "zero-intensity masks are the live test (Hubble's central obscuration "
    "zeroes 513 recorded entries over the 3 fields x 3 wavelengths) and agree "
    "in both modes.",
    "**Geometric MTF** goes through `be.histogram`, the one CPU fallback on the "
    "whole run (6 calls per system: 3 fields x tangential/sagittal, `mtf/"
    "geometric.py: _compute_field_data`; decode, `torch.histogram` on CPU "
    "float64, re-encode, `cpu_fallback:histogram`). Everything else, traces "
    "and the other five analyses, ran as Metal launches or on the "
    "dual-residency host path: zero `cpu_fallback:*` / `cpu_complex:*` counters.",
    "**Paraxial quantities** (f1, f2, EPD, XPD, FNO, magnification) are bit-"
    "identical to numpy in both modes: the paraxial trace works on 1-element "
    "tensors, which the dual-residency path evaluates on the CPU in float64.",
    "**Host threshold 256** puts the 817-ray traces, the 257-point ray fans, "
    "the 16-ring spot / OPD bundles and the 1024-ray MTF grids on the GPU, and "
    "the chief-ray distortion sweep (16 rays) and all scalar bookkeeping on the "
    "host; the per-phase gpu/host counts in section 4 show the split.",
)


def build(name: str) -> Any:
    """Instantiate ``name`` on the current backend and normalize its sweep.

    Systems with a single wavelength get two more (primary ± 0.05 µm) and
    systems with a single field get a 5 degree field so that the (0, 0.7) and
    (0, 1) normalized fields are off-axis. The same optic definition is used
    on every backend.
    """
    optic = SYSTEMS[name]()
    wls = optic.wavelengths.wavelengths
    if len(wls) < 3:
        p = optic.primary_wavelength
        optic.wavelengths.add(value=p - 0.05)
        optic.wavelengths.add(value=p + 0.05)
    if max(abs(f.y) for f in optic.fields.fields) == 0:
        optic.fields.add(y=5.0)
    return optic


# ---------------------------------------------------------------------------
# Instrumentation of the Metal counters
# ---------------------------------------------------------------------------
FALLBACK_SITES: dict[str, str] = {}


class _SiteCounter(Counter):
    """Counter that records the Optiland call site of each new CPU fallback."""

    def __setitem__(self, key: str, value: int) -> None:
        if key.startswith("cpu_fallback:") and key not in FALLBACK_SITES:
            FALLBACK_SITES[key] = _optiland_site()
        super().__setitem__(key, value)


def _optiland_site() -> str:
    """Innermost ``optiland/`` frame outside the backend package, as file:line."""
    frames = traceback.extract_stack()
    for fr in reversed(frames):
        fn = fr.filename.replace(os.sep, "/")
        if "/optiland/" in fn and "/optiland/backend/" not in fn:
            short = fn.split("/optiland/", 1)[1]
            return f"optiland/{short}:{fr.lineno} ({fr.name})"
    return "<no optiland frame>"


def _instrument_counters() -> None:
    from optiland.backend.torch_backend.metal import tensor as _t

    if not isinstance(_t._STATS, _SiteCounter):
        _t._STATS.__class__ = _SiteCounter


def _stats() -> dict[str, int]:
    return dict(be.metal_stats()) if be.get_backend() == "torch" else {}


def _diff(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    return {k: v - a.get(k, 0) for k, v in b.items() if v - a.get(k, 0)}


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def _np(x: Any) -> np.ndarray:
    return np.asarray(be.to_numpy(x), dtype=np.float64)


def run_system(name: str, rings: int, analyses: bool) -> dict[str, Any]:
    """Trace and analyze ``name`` on the current backend; return NumPy arrays.

    Returns a dict with one entry per phase (``trace``, ``spot``, ``rayfan``,
    ``opd``, ``mtf``, ``paraxial``, ``distortion``), each mapping quantity
    names to float64 arrays, plus ``phase_stats`` (Metal counter deltas per
    phase, empty on non-Metal backends) and ``phase_seconds``.
    """
    from optiland.analysis import Distortion, RayFan, SpotDiagram
    from optiland.mtf import GeometricMTF
    from optiland.wavefront import OPD

    optic = build(name)
    wavelengths = [w.value for w in optic.wavelengths.wavelengths][:3]
    out: dict[str, Any] = {
        "phase_stats": {},
        "phase_seconds": {},
        "wavelengths_um": wavelengths,
    }
    before = _stats()
    t0 = time.time()

    # -- traces ----------------------------------------------------------
    trace: dict[str, Any] = {}
    for hx, hy in FIELDS:
        for wl in wavelengths:
            optic.trace(Hx=hx, Hy=hy, wavelength=wl, num_rays=rings)
            sg = optic.surfaces
            key = f"H({hx:g},{hy:g})/w{wl:g}"
            trace[key] = {q: _np(getattr(sg, _ATTR[q])) for q in QUANTITIES}
    positions = _np(optic.surfaces.positions)
    heights = [
        np.nanmax(np.abs(np.concatenate([f["x"].ravel(), f["y"].ravel()])))
        for f in trace.values()
    ]
    out["scale"] = float(
        np.nanmax(np.abs(positions[np.isfinite(positions)])) + max(heights)
    )
    out["trace"] = trace
    out["phase_seconds"]["trace"] = time.time() - t0
    now = _stats()
    out["phase_stats"]["trace"] = _diff(before, now)
    before, t0 = now, time.time()
    if not analyses:
        return out

    # -- spot diagram ----------------------------------------------------
    spot = SpotDiagram(
        optic, fields=list(FIELDS), wavelengths="all", num_rings=SPOT_RINGS
    )
    out["spot"] = {
        "rms_spot_radius": np.array(
            [[float(_np(r)) for r in row] for row in spot.rms_spot_radius()]
        ),
        "geometric_spot_radius": np.array(
            [[float(_np(r)) for r in row] for row in spot.geometric_spot_radius()]
        ),
        "centroid": np.array(
            [[float(_np(cx)), float(_np(cy))] for cx, cy in spot.centroid()]
        ),
    }
    out["phase_seconds"]["spot"] = time.time() - t0
    now = _stats()
    out["phase_stats"]["spot"] = _diff(before, now)
    before, t0 = now, time.time()

    # -- ray fan ---------------------------------------------------------
    fan = RayFan(
        optic, fields=list(FIELDS), wavelengths="all", num_points=RAYFAN_POINTS
    )
    ex, ey = [], []
    for f in fan.fields:
        ex.append([_np(fan.data[f"{f.coord}"][f"{w}"]["x"]) for w in wavelengths])
        ey.append([_np(fan.data[f"{f.coord}"][f"{w}"]["y"]) for w in wavelengths])
    out["rayfan"] = {"ex": np.array(ex), "ey": np.array(ey)}
    out["phase_seconds"]["rayfan"] = time.time() - t0
    now = _stats()
    out["phase_stats"]["rayfan"] = _diff(before, now)
    before, t0 = now, time.time()

    # -- OPD -------------------------------------------------------------
    rms, maps = [], []
    for field in FIELDS:
        opd = OPD(optic, field=field, wavelength="primary", num_rays=OPD_RINGS)
        rms.append(float(_np(opd.rms())))
        maps.append(_np(opd.get_data(opd.fields[0], opd.wavelengths[0]).opd))
    out["opd"] = {"rms_waves": np.array(rms), "map_waves": np.array(maps)}
    out["phase_seconds"]["opd"] = time.time() - t0
    now = _stats()
    out["phase_stats"]["opd"] = _diff(before, now)
    before, t0 = now, time.time()

    # -- geometric MTF ---------------------------------------------------
    mtf = GeometricMTF(
        optic,
        fields=list(FIELDS),
        wavelength="primary",
        num_rays=MTF_RAYS,
        num_points=MTF_POINTS,
    )
    out["mtf"] = {
        "freq_cyc_per_mm": _np(mtf.freq),
        "mtf": np.array([[_np(t), _np(s)] for t, s in mtf.mtf]),
    }
    out["phase_seconds"]["mtf"] = time.time() - t0
    now = _stats()
    out["phase_stats"]["mtf"] = _diff(before, now)
    before, t0 = now, time.time()

    # -- paraxial --------------------------------------------------------
    out["paraxial"] = {
        k: np.array(float(_np(getattr(optic.paraxial, k)()))) for k in PARAXIAL
    }
    out["phase_seconds"]["paraxial"] = time.time() - t0
    now = _stats()
    out["phase_stats"]["paraxial"] = _diff(before, now)
    before, t0 = now, time.time()

    # -- distortion ------------------------------------------------------
    dist = Distortion(optic, wavelengths="all", num_points=DISTORTION_POINTS)
    pct = np.array([_np(d) for d in dist.data])
    # Optiland samples Hy = linspace(1e-10, 1, n); the innermost points divide
    # two nearly equal ~1e-9 mm heights, so their percentage is ill-conditioned
    # in any float64 implementation. ``percent_outer`` keeps Hy >= 0.25.
    hy = np.linspace(1e-10, 1, DISTORTION_POINTS)
    out["distortion"] = {"percent": pct, "percent_outer": pct[:, hy >= 0.25]}
    out["phase_seconds"]["distortion"] = time.time() - t0
    now = _stats()
    out["phase_stats"]["distortion"] = _diff(before, now)
    return out


def run_numpy(name: str, rings: int, analyses: bool) -> dict[str, Any]:
    """Oracle run on the NumPy backend."""
    be.set_backend("numpy")
    return run_system(name, rings, analyses)


def _set_grad(grad: bool) -> None:
    """Enable or disable torch autograd for the emulated / CPU runs.

    With autograd enabled the Newton-Raphson geometries (even/odd asphere,
    Zernike, polynomial, ...) apply the DiffOptics one-step implicit
    correction ``t - F(t)/F'(t)`` after the primal solve; since the primal
    solve stops at the configured ``tol`` (1e-6 for the asphere configs)
    that extra step moves the forward value by up to ~tol relative to the
    NumPy backend, which returns the primal root. The oracle therefore runs
    torch (CPU and mps) with autograd disabled by default so both backends
    execute the same primal path; ``--grad`` enables it.
    """
    if grad:
        be.grad_mode.enable()
    else:
        be.grad_mode.disable()


def run_torch_cpu(
    name: str, rings: int, analyses: bool, grad: bool = False
) -> dict[str, Any]:
    """Noise-floor run on torch CPU float64 (same code path as mps)."""
    be.set_backend("torch")
    be.set_device("cpu")
    be.set_precision("float64")
    _set_grad(grad)
    return run_system(name, rings, analyses)


def run_mps(
    name: str, rings: int, analyses: bool, mode: str, grad: bool = False
) -> dict[str, Any]:
    """Emulated run on torch / mps / float64 in representation ``mode``."""
    be.set_backend("torch")
    be.set_device("mps")
    be.set_precision("float64")
    _set_grad(grad)
    be.set_metal_mode(mode)
    be.metal_reset_stats()
    _instrument_counters()
    return run_system(name, rings, analyses)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_arrays(
    r: np.ndarray, g: np.ndarray, zero_mask: bool = False
) -> dict[str, Any]:
    """Max |delta|, magnitude-relative delta and NaN agreement of two arrays.

    With ``zero_mask`` the exact-zero patterns must agree as well (used for
    ray intensities, where Optiland marks vignetted / obscured rays with 0).
    """
    r = np.asarray(r, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    if r.shape != g.shape:
        raise ValueError(f"shape {g.shape} != {r.shape}")
    nan_r, nan_g = np.isnan(r), np.isnan(g)
    agree = bool(np.array_equal(nan_r, nan_g))
    zero_agree = bool(np.array_equal(r == 0, g == 0)) if zero_mask else True
    agree &= zero_agree
    both = ~(nan_r | nan_g)
    if both.any():
        d = np.abs(r[both] - g[both])
        max_abs = float(np.max(d))
        mag = float(np.max(np.abs(r[both])))
        rel = max_abs / mag if mag > 0 else (0.0 if max_abs == 0 else float("inf"))
        big = np.abs(r[both]) >= 1e-3 * mag if mag > 0 else np.zeros_like(d, bool)
        rel_elem = float(np.max(d[big] / np.abs(r[both][big]))) if big.any() else 0.0
    else:
        max_abs, mag, rel, rel_elem = 0.0, 0.0, 0.0, 0.0
    return {
        "max_abs": max_abs,
        "magnitude": mag,
        "rel": rel,
        "rel_elem": rel_elem,
        "nan_agree": agree,
        "zero_agree": zero_agree,
        "n": int(r.size),
        "n_nan": int(nan_r.sum()),
        "n_nan_got": int(nan_g.sum()),
        "n_zero": int((r == 0).sum()) if zero_mask else 0,
    }


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Worst case of two comparison records."""
    return {
        "max_abs": max(a["max_abs"], b["max_abs"]),
        "magnitude": max(a["magnitude"], b["magnitude"]),
        "rel": max(a["rel"], b["rel"]),
        "rel_elem": max(a["rel_elem"], b["rel_elem"]),
        "nan_agree": a["nan_agree"] and b["nan_agree"],
        "zero_agree": a["zero_agree"] and b["zero_agree"],
        "n": a["n"] + b["n"],
        "n_nan": a["n_nan"] + b["n_nan"],
        "n_nan_got": a["n_nan_got"] + b["n_nan_got"],
        "n_zero": a["n_zero"] + b["n_zero"],
    }


def compare(ref: dict[str, Any], got: dict[str, Any]) -> dict[str, Any]:
    """Compare two ``run_system`` results phase by phase."""
    rep: dict[str, Any] = {"trace": {}, "trace_fields": {}}
    worst: dict[str, dict[str, Any]] = {}
    for key, rq in ref["trace"].items():
        frep = {
            q: compare_arrays(rq[q], got["trace"][key][q], zero_mask=(q == "i"))
            for q in QUANTITIES
        }
        rep["trace_fields"][key] = frep
        for q in QUANTITIES:
            worst[q] = _merge(worst[q], frep[q]) if q in worst else frep[q]
    rep["trace"] = worst
    rep["nan_agree"] = all(v["nan_agree"] for v in worst.values())
    rep["max_position_delta"] = max(worst[q]["max_abs"] for q in POSITION)
    rep["scale_mm"] = ref["scale"]
    rep["max_position_delta_rel"] = rep["max_position_delta"] / ref["scale"]
    for phase in PHASES[1:]:
        if phase not in ref or phase not in got:
            continue
        rep[phase] = {
            k: compare_arrays(ref[phase][k], got[phase][k]) for k in ref[phase]
        }
        rep["nan_agree"] &= all(v["nan_agree"] for v in rep[phase].values())
    rep["values"] = {
        "spot_rms_ref": ref["spot"]["rms_spot_radius"].tolist() if "spot" in ref else 0,
        "spot_rms_got": got["spot"]["rms_spot_radius"].tolist() if "spot" in got else 0,
        "opd_rms_ref": ref["opd"]["rms_waves"].tolist() if "opd" in ref else 0,
        "opd_rms_got": got["opd"]["rms_waves"].tolist() if "opd" in got else 0,
        "paraxial_ref": {k: float(v) for k, v in ref.get("paraxial", {}).items()},
        "paraxial_got": {k: float(v) for k, v in got.get("paraxial", {}).items()},
        "mtf_freq": ref["mtf"]["freq_cyc_per_mm"].tolist() if "mtf" in ref else 0,
        "mtf_ref": ref["mtf"]["mtf"].tolist() if "mtf" in ref else 0,
        "mtf_got": got["mtf"]["mtf"].tolist() if "mtf" in got else 0,
    }
    return rep


def summarize_stats(phase_stats: dict[str, dict[str, int]]) -> dict[str, Any]:
    """Totals and per-phase GPU / host / fallback counts."""
    total: Counter[str] = Counter()
    per_phase: dict[str, Any] = {}
    for phase, st in phase_stats.items():
        total.update(st)
        per_phase[phase] = {
            "gpu": sum(v for k, v in st.items() if k.startswith("gpu:")),
            "host": sum(v for k, v in st.items() if k.startswith("host:")),
            "cpu_fallback": {
                k[13:]: v for k, v in st.items() if k.startswith("cpu_fallback:")
            },
            "cpu_complex": {
                k[12:]: v for k, v in st.items() if k.startswith("cpu_complex:")
            },
        }
    gpu = {k[4:]: v for k, v in total.items() if k.startswith("gpu:")}
    host = {k[5:]: v for k, v in total.items() if k.startswith("host:")}
    fb = {k[13:]: v for k, v in total.items() if k.startswith("cpu_fallback:")}
    return {
        "gpu_launches": sum(gpu.values()),
        "gpu_kernels": len(gpu),
        "gpu_by_kernel": dict(sorted(gpu.items(), key=lambda kv: -kv[1])),
        "host_ops": sum(host.values()),
        "host_by_op": dict(sorted(host.items(), key=lambda kv: -kv[1])),
        "cpu_fallbacks": sum(fb.values()),
        "cpu_fallback_by_op": fb,
        "cpu_fallback_sites": {
            k: FALLBACK_SITES.get("cpu_fallback:" + k, "?") for k in fb
        },
        "per_phase": per_phase,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _e(v: float) -> str:
    return f"{v:.2e}"


def format_console(
    name: str, mode: str, rep: dict[str, Any], floor: dict[str, Any] | None, st: Any
) -> str:
    """Human-readable block for one system and mode."""
    lines = [f"== {name} ({mode}) =="]
    lines.append(
        "  trace: max |delta| vs numpy float64 over all surfaces, fields, "
        "wavelengths" + ("   [torch-cpu float64 floor]" if floor else "")
    )
    for q in QUANTITIES:
        unit = "mm" if q in POSITION or q == "opd" else "  "
        w = rep["trace"][q]
        line = f"    {q:>3}: {w['max_abs']:.3e} {unit} rel {w['rel']:.2e}"
        if floor:
            line += f"   [{floor['trace'][q]['max_abs']:.3e}]"
        lines.append(line)
    rel = rep["max_position_delta_rel"]
    lines.append(
        f"  path scale {rep['scale_mm']:.4g} mm -> position delta / scale "
        f"{rel:.3e} ({rel / MACHINE_EPS[mode]:.1f} eps)"
    )
    lines.append(f"  NaN patterns agree (all phases): {rep['nan_agree']}")
    for phase in PHASES[1:]:
        if phase not in rep:
            continue
        parts = [
            f"{k} {v['max_abs']:.2e} (rel {v['rel']:.1e})"
            for k, v in rep[phase].items()
        ]
        lines.append(f"  {phase:>10}: " + "; ".join(parts))
    lines.append(
        f"  metal_stats: {st['gpu_launches']} GPU launches over "
        f"{st['gpu_kernels']} kernels, {st['host_ops']} host ops, "
        f"{st['cpu_fallbacks']} CPU fallbacks"
    )
    top = list(st["gpu_by_kernel"].items())[:12]
    lines.append("    top GPU kernels: " + ", ".join(f"{k}={v}" for k, v in top))
    lines.append(
        "    per phase (gpu/host/fallback): "
        + ", ".join(
            f"{p}={v['gpu']}/{v['host']}/{sum(v['cpu_fallback'].values())}"
            for p, v in st["per_phase"].items()
        )
    )
    lines.append(
        "    CPU fallbacks: "
        + (
            ", ".join(
                f"{k}={v} @ {st['cpu_fallback_sites'][k]}"
                for k, v in sorted(st["cpu_fallback_by_op"].items())
            )
            or "none"
        )
    )
    return "\n".join(lines)


def write_markdown(path: str, doc: dict[str, Any]) -> None:
    """Write the NOTES report (tables per mode, fallback list, method)."""
    modes = doc["modes"]
    res = doc["results"]
    names = [n for n in doc["systems"] if n in res]
    L: list[str] = []
    L.append("# 06 — End-to-end oracle report: NumPy float64 vs Metal df64 / sf64")
    L.append("")
    L.append(
        f"Generated {doc['generated']} by `Optiland-Metal/scripts/metal_oracle_e2e.py`"
        f" (rings={doc['rings']}, fields {list(FIELDS)}, 3 wavelengths per system,"
        f" host threshold {doc['host_threshold']}, autograd"
        f" {'enabled' if doc['grad'] else 'disabled'}, torch {doc['torch']},"
        f" {doc['device']}). Raw numbers: `NOTES/oracle-e2e.json`."
    )
    L.append("")
    L.append("## 1. Summary")
    L.append("")
    L.append(
        "| mode | worst trace position error (mm) | system | rel. to path scale "
        "| worst vs torch-cpu (mm) | pass |"
    )
    L.append("|---|---|---|---|---|---|")
    for mode in modes:
        worst = None
        for n in names:
            r = res[n].get(mode)
            if not r or "error" in r:
                continue
            if worst is None or r["vs_numpy"]["max_position_delta"] > worst[1]:
                worst = (n, r["vs_numpy"]["max_position_delta"])
        if worst is None:
            continue
        r = res[worst[0]][mode]
        cpu = r.get("vs_torch_cpu", {}).get("max_position_delta", float("nan"))
        all_pass = all(
            res[n].get(mode, {}).get("passed", False) for n in names if mode in res[n]
        )
        L.append(
            f"| {mode} | {_e(worst[1])} | {worst[0]} | "
            f"{_e(r['vs_numpy']['max_position_delta_rel'])} "
            f"({r['vs_numpy']['max_position_delta_rel'] / MACHINE_EPS[mode]:.1f} eps) "
            f"| {_e(cpu)} | {'PASS' if all_pass else 'FAIL'} |"
        )
    L.append("")
    L.append(
        "Criterion per system: position error < 1e-11 mm absolute **or** < 64 eps of"
        " the mode relative to the path scale, and the NaN patterns (every phase)"
        " and the zero-intensity patterns of the traces (Optiland marks vignetted"
        " and obscured rays with intensity 0) agree."
        " `rel` is max |delta| divided by the max |value| of that"
        " quantity (over the finite entries); `rel_elem` in the JSON is the max"
        " element-wise relative error over entries above 1e-3 of that magnitude."
    )
    L.append("")
    L.append("## 2. Traces (every surface, 3 fields x 3 wavelengths)")
    for mode in modes:
        L.append("")
        L.append(f"### 2.{modes.index(mode) + 1} {mode}: max |delta| vs numpy float64")
        L.append("")
        L.append(
            "| system | x | y | z | L | M | N | opd | i | pos. rel. to scale | "
            "torch-cpu floor (pos.) | vs torch-cpu (pos.) | NaN / vignetting "
            "(n) | pass |"
        )
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for n in names:
            r = res[n].get(mode)
            if not r:
                continue
            if "error" in r:
                L.append(f"| {n} | error: {r['error'].strip().splitlines()[-1]} |")
                continue
            v = r["vs_numpy"]
            fl = res[n].get("torch_cpu_vs_numpy")
            cpu = r.get("vs_torch_cpu")
            cells = [_e(v["trace"][q]["max_abs"]) for q in QUANTITIES]
            L.append(
                f"| {n} | "
                + " | ".join(cells)
                + f" | {_e(v['max_position_delta_rel'])} "
                f"({v['max_position_delta_rel'] / MACHINE_EPS[mode]:.1f} eps)"
                f" | {_e(fl['max_position_delta']) if fl else 'n/a'}"
                f" | {_e(cpu['max_position_delta']) if cpu else 'n/a'}"
                f" | {'agree' if v['nan_agree'] else 'DISAGREE'}"
                f" ({v['trace']['i']['n_nan']} NaN, {v['trace']['i']['n_zero']} zero-i)"
                f" | {'PASS' if r['passed'] else 'FAIL'} |"
            )
    L.append("")
    L.append("## 3. Analyses")
    an = [
        ("spot", "rms_spot_radius", "spot RMS (mm)"),
        ("spot", "centroid", "spot centroid (mm)"),
        ("rayfan", "ey", "ray fan ey (mm)"),
        ("rayfan", "ex", "ray fan ex (mm)"),
        ("opd", "rms_waves", "OPD rms (waves)"),
        ("opd", "map_waves", "OPD map (waves)"),
        ("mtf", "mtf", "geometric MTF"),
        ("distortion", "percent_outer", "distortion (%, Hy >= 0.25)"),
    ]
    for mode in modes:
        L.append("")
        L.append(
            f"### 3.{modes.index(mode) + 1} {mode}: max |delta| vs numpy "
            "(rel = delta / max |value|)"
        )
        L.append("")
        L.append("| system | " + " | ".join(a[2] for a in an) + " | paraxial (rel) |")
        L.append("|---|" + "---|" * (len(an) + 1))
        for n in names:
            r = res[n].get(mode)
            if not r or "error" in r or "spot" not in r["vs_numpy"]:
                continue
            v = r["vs_numpy"]
            cells = []
            for phase, key, _ in an:
                c = v[phase][key]
                cells.append(f"{_e(c['max_abs'])} ({c['rel']:.0e})")
            prel = max(c["rel"] for c in v["paraxial"].values())
            L.append(f"| {n} | " + " | ".join(cells) + f" | {prel:.1e} |")
    L.append("")
    L.append("### 3.3 Values (numpy oracle; emulated values agree to the deltas above)")
    L.append("")
    L.append(
        "| system | f2 (mm) | EPD (mm) | FNO | spot RMS on-axis, primary (mm) | "
        "OPD rms on-axis (waves) | MTF tangential @ freq idx "
        f"{list(MTF_REPORT_INDEX)} (on-axis) |"
    )
    L.append("|---|---|---|---|---|---|---|")
    for n in names:
        r = res[n].get(modes[0])
        if not r or "error" in r or "spot" not in r["vs_numpy"]:
            continue
        vals = r["vs_numpy"]["values"]
        px = vals["paraxial_ref"]
        mt = vals["mtf_ref"][0][0]
        fq = vals["mtf_freq"]
        mtf_cells = ", ".join(f"{mt[i]:.6f} @ {fq[i]:.1f}/mm" for i in MTF_REPORT_INDEX)
        L.append(
            f"| {n} | {px['f2']:.6f} | {px['EPD']:.6f} | {px['FNO']:.6f} | "
            f"{vals['spot_rms_ref'][0][1]:.9g} | {vals['opd_rms_ref'][0]:.9g} | "
            f"{mtf_cells} |"
        )
    L.append("")
    L.append("## 4. Metal op counters per system (whole run: traces + analyses)")
    for mode in modes:
        L.append("")
        L.append(f"### 4.{modes.index(mode) + 1} {mode}")
        L.append("")
        L.append(
            "| system | GPU launches | kernels | host ops | CPU fallbacks | "
            "per phase gpu/host/fallback | wall mps (s) | wall numpy (s) |"
        )
        L.append("|---|---|---|---|---|---|---|---|")
        for n in names:
            r = res[n].get(mode)
            if not r or "error" in r:
                continue
            st = r["stats"]
            per = ", ".join(
                f"{p} {v['gpu']}/{v['host']}/{sum(v['cpu_fallback'].values())}"
                for p, v in st["per_phase"].items()
            )
            fb = ", ".join(f"{k}={v}" for k, v in st["cpu_fallback_by_op"].items())
            L.append(
                f"| {n} | {st['gpu_launches']} | {st['gpu_kernels']} | "
                f"{st['host_ops']} | {st['cpu_fallbacks']}"
                f"{' (' + fb + ')' if fb else ''} | {per} | "
                f"{r['seconds_mps']:.1f} | {res[n]['seconds_numpy']:.2f} |"
            )
    L.append("")
    L.append("## 5. CPU fallbacks observed")
    L.append("")
    fbs: dict[str, dict[str, Any]] = {}
    for n in names:
        for mode in modes:
            r = res[n].get(mode)
            if not r or "error" in r:
                continue
            for op, cnt in r["stats"]["cpu_fallback_by_op"].items():
                d = fbs.setdefault(op, {"site": r["stats"]["cpu_fallback_sites"][op]})
                d.setdefault("where", {})[f"{n}/{mode}"] = cnt
                phases = [
                    p
                    for p, v in r["stats"]["per_phase"].items()
                    if op in v["cpu_fallback"]
                ]
                d.setdefault("phases", set()).update(phases)
    if not fbs:
        L.append(
            "None. Every op of every trace and analysis phase ran either as a Metal"
            " kernel launch or on the dual-residency host path; no"
            " `cpu_fallback:*` counter incremented."
        )
    else:
        L.append("| op | Optiland call site | phases | counts (system/mode) |")
        L.append("|---|---|---|---|")
        for op, d in sorted(fbs.items()):
            counts = ", ".join(f"{k}={v}" for k, v in d["where"].items())
            L.append(
                f"| `{op}` | `{d['site']}` | {', '.join(sorted(d['phases']))} | "
                f"{counts} |"
            )
    L.append("")
    L.append("## 6. Notes")
    L.append("")
    for line in list(REPORT_NOTES) + list(doc.get("notes", [])):
        L.append(f"- {line}")
    L.append("")
    L.append("System notes:")
    for n in names:
        if n in SYSTEM_NOTES:
            L.append(f"- {n}: {SYSTEM_NOTES[n]}")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--systems", default="all", help="'all' or comma-separated names")
    ap.add_argument("--modes", default="df64,sf64", help="comma-separated modes")
    ap.add_argument("--mode", default=None, help="alias for --modes (single mode)")
    ap.add_argument("--rings", type=int, default=16, help="hexapolar rings per trace")
    ap.add_argument(
        "--tol", type=float, default=1e-11, help="absolute position tolerance [mm]"
    )
    ap.add_argument(
        "--tol-rel",
        type=float,
        default=None,
        help="position tolerance relative to the path scale (default: 64 eps)",
    )
    ap.add_argument(
        "--no-cpu-floor", action="store_true", help="skip the torch CPU float64 run"
    )
    ap.add_argument(
        "--no-analyses", action="store_true", help="traces only (no analyses)"
    )
    ap.add_argument(
        "--grad",
        action="store_true",
        help="run torch (cpu and mps) with autograd enabled (implicit NR correction)",
    )
    ap.add_argument("--json", default=None, help="write the full report here")
    ap.add_argument("--md", default=None, help="write the markdown report here")
    ap.add_argument(
        "--from-json",
        default=None,
        help="do not run anything; render --md from this previously written JSON",
    )
    args = ap.parse_args(argv)
    if args.from_json:
        with open(args.from_json) as fh:
            doc = json.load(fh)
        if not args.md:
            ap.error("--from-json needs --md")
        write_markdown(args.md, doc)
        return 0 if doc.get("passed") else 1
    modes = (args.mode or args.modes).split(",")
    for m in modes:
        if m not in MACHINE_EPS:
            ap.error(f"unknown mode {m}")

    names = list(SYSTEMS) if args.systems == "all" else args.systems.split(",")
    unknown = [n for n in names if n not in SYSTEMS]
    if unknown:
        ap.error(f"unknown systems {unknown}; known: {list(SYSTEMS)}")

    import torch

    results: dict[str, Any] = {}
    ok = True
    for name in names:
        t0 = time.time()
        ref = run_numpy(name, args.rings, not args.no_analyses)
        t1 = time.time()
        entry: dict[str, Any] = {
            "seconds_numpy": t1 - t0,
            "wavelengths_um": ref["wavelengths_um"],
            "scale_mm": ref["scale"],
        }
        cpu = None
        if not args.no_cpu_floor:
            cpu = run_torch_cpu(name, args.rings, not args.no_analyses, args.grad)
            fl = compare(ref, cpu)
            fl.pop("trace_fields")
            fl.pop("values")
            entry["torch_cpu_vs_numpy"] = fl
            entry["seconds_torch_cpu"] = time.time() - t1
            be.set_backend("numpy")
        results[name] = entry
        for mode in modes:
            tol_rel = (
                args.tol_rel if args.tol_rel is not None else 64 * MACHINE_EPS[mode]
            )
            t2 = time.time()
            try:
                got = run_mps(name, args.rings, not args.no_analyses, mode, args.grad)
            except Exception:  # noqa: BLE001 - reported, then continue
                print(f"== {name} ({mode}) == FAILED\n{traceback.format_exc()}")
                entry[mode] = {"error": traceback.format_exc()}
                ok = False
                be.set_backend("numpy")
                continue
            t3 = time.time()
            rep = compare(ref, got)
            st = summarize_stats(got["phase_stats"])
            r: dict[str, Any] = {
                "vs_numpy": rep,
                "stats": st,
                "seconds_mps": t3 - t2,
                "phase_seconds": got["phase_seconds"],
            }
            if cpu is not None:
                vc = compare(cpu, got)
                vc.pop("trace_fields")
                vc.pop("values")
                r["vs_torch_cpu"] = vc
            passed = (
                rep["max_position_delta"] < args.tol
                or rep["max_position_delta_rel"] < tol_rel
            ) and rep["nan_agree"]
            r["passed"] = passed
            entry[mode] = r
            print(format_console(name, mode, rep, entry.get("torch_cpu_vs_numpy"), st))
            if "vs_torch_cpu" in r:
                print(
                    f"  vs torch-cpu float64: position {_e(vc['max_position_delta'])}"
                    f" mm (rel {_e(vc['max_position_delta_rel'])})"
                )
            print(f"  wall: numpy {t1 - t0:.2f} s, mps {t3 - t2:.2f} s")
            if passed:
                clause = (
                    "abs" if rep["max_position_delta"] < args.tol else "scale-relative"
                )
                print(f"  PASS ({clause} criterion)")
            else:
                ok = False
                print(
                    f"  ** FAIL: position delta {rep['max_position_delta']:.3e} mm "
                    f">= {args.tol:g} and relative "
                    f"{rep['max_position_delta_rel']:.3e} >= {tol_rel:g}, "
                    "or NaN mismatch"
                )
            be.set_backend("numpy")

    print()
    worst: dict[str, float] = {}
    for mode in modes:
        vals = [
            results[n][mode]["vs_numpy"]["max_position_delta"]
            for n in names
            if mode in results[n] and "vs_numpy" in results[n][mode]
        ]
        worst[mode] = max(vals) if vals else float("nan")
        print(f"Overall max position delta ({mode}): {worst[mode]:.3e} mm")
    print("PASS" if ok else "FAIL")

    from optiland.backend.torch_backend.metal import tensor as _t

    doc = {
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "device": "Apple M1 Max (mps)",
        "host_threshold": _t.get_host_threshold(),
        "modes": modes,
        "systems": names,
        "rings": args.rings,
        "grad": args.grad,
        "fields": list(FIELDS),
        "tol": args.tol,
        "tol_rel": {m: 64 * MACHINE_EPS[m] for m in modes},
        "worst_position_error_mm": worst,
        "passed": ok,
        "results": results,
        "notes": [],
    }
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(doc, fh, indent=1, default=str)
    if args.md:
        write_markdown(args.md, doc)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
