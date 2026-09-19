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
        [--md NOTES/06-oracle-report.md] [--from-json NOTES/oracle-e2e.json] \
        [--fused auto|off|require|both] [--tier-a] [--mtf-rays 38] [--diag] \
        [--strict]

The fused trace (plan 8.1).  ``--fused both`` runs each system twice in one
process -- once with the hook off, which is reference R1, then once on the
kernel (under ``require``, so an unexpected refusal raises; a system in
:data:`KNOWN_INELIGIBLE` runs under ``1`` and must instead produce zero
launches and the predicted refusal count) -- and reports a ``fused vs per-op``
column under the tolerance rule of plan 7.1: raw-component equality wherever
the traced bundles were larger than 1024 rays (tier A), the
``64 * eps * scale`` bound below that (tier B).  Every run carries an
independent census of candidate bundles that the two identities of plan 1.3
are checked against, plus a prediction of the kernel launch count from
``_slab_plan``; a violation fails the run.

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

#: Analysis sampling actually used; ``--tier-a`` and the four per-knob options
#: overwrite it (plan 8.1).  ``run_system`` reads this dict, never the
#: constants above, so a CLI override reaches every phase.
SAMPLING: dict[str, int] = {
    "rayfan_points": RAYFAN_POINTS,
    "opd_rings": OPD_RINGS,
    "mtf_rays": MTF_RAYS,
    "spot_rings": SPOT_RINGS,
}

#: ``--tier-a``: the sampling plan 8.1 names, meant to push every compared
#: bundle above :data:`TIER_A_MIN_RAYS`.  Whether a phase reaches it is
#: **measured**, not assumed, and one of them does not: ``GeometricMTF``'s
#: uniform grid is clipped to the unit disc, so ``--mtf-rays 33`` traces 797
#: rays, not 1089, and the MTF phase stays tier B.  Measured grid sizes:
#: 33 -> 797, 36 -> 952, 37 -> 1009, 38 -> 1060, so ``--mtf-rays 38`` is the
#: smallest uniform grid above the tier-A floor.  The plan's value is kept
#: here; a run that needs the MTF phase at tier A passes ``--mtf-rays 38``.
TIER_A_SAMPLING: dict[str, int] = {
    "rayfan_points": 1025,
    "opd_rings": 19,
    "mtf_rays": 33,
    "spot_rings": 19,
}

#: Tier A needs ``N > 1024`` (plan 7.1).
TIER_A_MIN_RAYS = 1024

#: Systems whose fields sweep an x field too: tilted or decentred systems are
#: not rotationally symmetric, so (0.7, 0) is a different trace (plan 8.1).
X_FIELD_SYSTEMS: frozenset[str] = frozenset({"TiltedTriplet", "TiltedFoldMirror"})

#: Systems traced but not analysed: Optiland's paraxial, MTF and distortion
#: analyses assume an axial image plane, which a 45-degree fold does not have.
TRACE_ONLY_SYSTEMS: frozenset[str] = frozenset({"TiltedFoldMirror"})

#: Systems the gate must refuse, and the ``FusedTraceSkip`` reason it must use
#: (plan 8.1).  ``scripts/trace_fixtures.KNOWN_INELIGIBLE`` is merged in, so a
#: WP5 entry added there shows up here without a second list.
KNOWN_INELIGIBLE: dict[str, str] = {
    "ZernikeSinglet": "geometry_type",
    "PolynomialSinglet": "geometry_type",
}

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


def _fixture(name: str) -> Any:
    """Build one of ``scripts/trace_fixtures.py``'s systems.

    Some fixtures return ``(optic, ray_builder)``; the oracle generates its own
    rays through ``optic.trace``, so only the optic is taken.
    """
    import trace_fixtures

    built = getattr(trace_fixtures, name)()
    return built[0] if isinstance(built, tuple) else built


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
    # Fused-trace coverage (plan 8.1): tilts and decentres, a reflective fold,
    # the deepest sample (44 surfaces, chunking), a rectangular aperture and
    # the odd-power asphere.
    "TiltedTriplet": lambda: _fixture("tilted_triplet"),
    "TiltedFoldMirror": lambda: _fixture("tilted_fold_mirror"),
    "UVProjectionLens": lambda: _fixture("uv_projection"),
    "RectApertureSinglet": lambda: _fixture("rect_aperture"),
    "OddAsphereSinglet": lambda: _fixture("odd_asphere_singlet"),
}
SYSTEM_NOTES = {
    "AsphericSinglet": "even asphere (optiland.samples.simple); +5 deg field and "
    "±0.05 µm wavelengths added for the multi-field/-wavelength sweep",
    "HubbleTelescope": "mirrors only; ±0.05 µm wavelengths added",
    "ZernikeSinglet": "built in the script (no Zernike sample in optiland.samples)",
    "PolynomialSinglet": "built in the script (no polynomial sample in "
    "optiland.samples)",
    "TiltedTriplet": "scripts/trace_fixtures.tilted_triplet(): the Cooke "
    "triplet with surfaces 3 and 4 tilted (rx, ry, rz) and decentred; the "
    "(0.7, 0) field is added because the system is no longer rotationally "
    "symmetric",
    "TiltedFoldMirror": "scripts/trace_fixtures.tilted_fold_mirror(): a "
    "45-degree fold behind an absorbing slab; traces only, the analyses "
    "assume an axial image plane",
    "UVProjectionLens": "optiland.samples.UVProjectionLens through "
    "scripts/trace_fixtures.uv_projection(): 44 surfaces, the chunking path",
    "RectApertureSinglet": "scripts/trace_fixtures.rect_aperture(): a "
    "rectangular aperture on the stop, so rays are clipped in x, in y and in "
    "both",
    "OddAsphereSinglet": "scripts/trace_fixtures.odd_asphere_singlet(): the "
    "odd-power asphere (Newton iteration with odd exponents)",
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


def fields_for(name: str) -> tuple[tuple[float, float], ...]:
    """The normalized fields swept for ``name`` (plan 8.1)."""
    if name in X_FIELD_SYSTEMS:
        return (*FIELDS, (0.7, 0.0))
    return FIELDS


def analyses_for(name: str, analyses: bool) -> bool:
    """Whether the analyses run for ``name`` (see :data:`TRACE_ONLY_SYSTEMS`)."""
    return analyses and name not in TRACE_ONLY_SYSTEMS


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


def _raw(value: Any) -> list[np.ndarray]:
    """Host copies of ``value``'s raw components, never decoded (plan 7.1).

    A ``MetalFloat64`` yields its df64 ``hi``/``lo`` float32 words or its sf64
    int64 bit patterns; anything else yields one float64 array.  Tier A
    compares these, not the decoded values, so a difference that lives only in
    the low word cannot hide.
    """
    comps = getattr(value, "components", None)
    if comps is None:
        return [np.asarray(be.to_numpy(value), dtype=np.float64)]
    return [np.asarray(c.detach().cpu().numpy()) for c in comps]


def _host_threshold() -> int:
    """The dual-residency threshold, read from the environment like the gate."""
    return int(os.environ.get("OPTILAND_METAL_HOST_THRESHOLD", "256"))


def _fused_traces() -> int:
    """The driver's ``fused_trace:traces`` counter right now."""
    return _stats().get("fused_trace:traces", 0)


def _candidate_rays(group: Any, rays: Any, skip: int) -> int | None:
    """``rays.x.numel()`` when this call is a fused-trace candidate, else None.

    The structural checks of plan 1.3, evaluated without importing the gate, so
    this census stays an independent predictor of ``fused_trace:candidates``
    (plan 8.1, 8.3) rather than the gate agreeing with itself.
    """
    from optiland.rays import RealRays
    from optiland.surfaces.surface_group import SurfaceGroup

    if type(group) is not SurfaceGroup or type(rays) is not RealRays or skip != 0:
        return None
    x = getattr(rays, "x", None)
    if type(x).__name__ != "MetalFloat64" or be.grad_mode.requires_grad:
        return None
    n = int(x.numel())
    return n if n > _host_threshold() else None


def _predict_launches(group: Any, n: int) -> int:
    """Kernel launches one fused trace of ``n`` rays through ``group`` needs.

    ``weighted_steps`` is recomputed here from the surface list (plan 3.5:
    ``1 + max_iter`` on a Newton row, 1 otherwise) and handed to the driver's
    own ``_slab_plan``, which plan 8.1 names as the predictor of
    ``gpu:fused_trace``.
    """
    from optiland.backend.torch_backend.metal import library, trace

    weighted = 0
    for surface in list(group.surfaces)[1:]:
        geometry = surface.geometry
        if type(geometry).__name__ in ("EvenAsphere", "OddAsphere"):
            weighted += 1 + int(geometry.max_iter)
        else:
            weighted += 1
    plan = trace._slab_plan(1, n, weighted, trace._max_steps(), library.DEFAULT_CHUNK)
    return len(plan)


class Census:
    """Independent per-phase census of fused-trace candidates (plan 8.1).

    Wraps ``SurfaceGroup.trace`` for the duration of one ``run_system`` call.
    ``candidates`` counts the bundles that pass the structural checks,
    ``launches`` accumulates the predicted kernel launches of the calls that
    actually fused, and ``n_min`` / ``n_max`` record the bundle sizes the phase
    traced, which is what the tier-A label is derived from.
    """

    def __init__(self) -> None:
        self.phase = "trace"
        self.candidates: Counter = Counter()
        self.launches: Counter = Counter()
        self.n_min: dict[str, int] = {}
        self.n_max: dict[str, int] = {}
        self._original: Any = None
        self._group: Any = None

    def __enter__(self) -> Census:  # noqa: PYI034 - concrete, never subclassed
        from optiland.surfaces.surface_group import SurfaceGroup

        self._group = SurfaceGroup
        original = SurfaceGroup.trace
        self._original = original
        census = self

        # The wrapper must hold the original function in a closure cell of its
        # own: that is what ``trace_mirror._delegates`` walks, so wrapping
        # ``SurfaceGroup.trace`` for the census is not read as mirror drift
        # (which would refuse every candidate for the rest of the process).
        def _traced(group, rays, skip=0, record=True):  # noqa: ANN001
            return census._call(original, group, rays, skip, record)

        SurfaceGroup.trace = _traced
        return self

    def __exit__(self, *exc: object) -> None:
        self._group.trace = self._original

    def _call(
        self, original: Any, group: Any, rays: Any, skip: int, record: bool
    ) -> Any:
        n = _candidate_rays(group, rays, skip)
        if n is None:
            return original(group, rays, skip=skip, record=record)
        phase = self.phase
        self.candidates[phase] += 1
        self.n_min[phase] = min(self.n_min.get(phase, n), n)
        self.n_max[phase] = max(self.n_max.get(phase, n), n)
        before = _fused_traces()
        out = original(group, rays, skip=skip, record=record)
        if _fused_traces() > before:
            self.launches[phase] += _predict_launches(group, n)
        return out

    def report(self) -> dict[str, Any]:
        """The census as JSON: per-phase candidates, launches and bundle sizes."""
        return {
            "candidates": dict(self.candidates),
            "predicted_launches": dict(self.launches),
            "n_min": dict(self.n_min),
            "n_max": dict(self.n_max),
            "total_candidates": sum(self.candidates.values()),
        }


def run_system(
    name: str,
    rings: int,
    analyses: bool,
    census: Census | None = None,
    raw: bool = False,
) -> dict[str, Any]:
    """Trace and analyze ``name`` on the current backend; return NumPy arrays.

    Returns a dict with one entry per phase (``trace``, ``spot``, ``rayfan``,
    ``opd``, ``mtf``, ``paraxial``, ``distortion``), each mapping quantity
    names to float64 arrays, plus ``phase_stats`` (Metal counter deltas per
    phase, empty on non-Metal backends) and ``phase_seconds``.  With ``raw``
    the recorded surfaces are kept as raw components too (``trace_raw``), which
    is what a tier-A comparison needs.
    """
    from optiland.analysis import Distortion, RayFan, SpotDiagram
    from optiland.mtf import GeometricMTF
    from optiland.wavefront import OPD

    optic = build(name)
    fields = fields_for(name)
    analyses = analyses_for(name, analyses)
    wavelengths = [w.value for w in optic.wavelengths.wavelengths][:3]
    out: dict[str, Any] = {
        "phase_stats": {},
        "phase_seconds": {},
        "wavelengths_um": wavelengths,
        "fields": [list(f) for f in fields],
        "sampling": dict(SAMPLING),
        "rings": rings,
    }
    before = _stats()
    t0 = time.time()

    # -- traces ----------------------------------------------------------
    if census is not None:
        census.phase = "trace"
    trace: dict[str, Any] = {}
    trace_raw: dict[str, Any] = {}
    for hx, hy in fields:
        for wl in wavelengths:
            optic.trace(Hx=hx, Hy=hy, wavelength=wl, num_rays=rings)
            sg = optic.surfaces
            key = f"H({hx:g},{hy:g})/w{wl:g}"
            trace[key] = {q: _np(getattr(sg, _ATTR[q])) for q in QUANTITIES}
            if raw:
                trace_raw[key] = {q: _raw(getattr(sg, _ATTR[q])) for q in QUANTITIES}
    if raw:
        out["trace_raw"] = trace_raw
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
    if census is not None:
        census.phase = "spot"
    spot = SpotDiagram(
        optic,
        fields=list(fields),
        wavelengths="all",
        num_rings=SAMPLING["spot_rings"],
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
    if census is not None:
        census.phase = "rayfan"
    fan = RayFan(
        optic,
        fields=list(fields),
        wavelengths="all",
        num_points=SAMPLING["rayfan_points"],
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
    if census is not None:
        census.phase = "opd"
    rms, maps = [], []
    for field in fields:
        opd = OPD(
            optic,
            field=field,
            wavelength="primary",
            num_rays=SAMPLING["opd_rings"],
        )
        rms.append(float(_np(opd.rms())))
        maps.append(_np(opd.get_data(opd.fields[0], opd.wavelengths[0]).opd))
    out["opd"] = {"rms_waves": np.array(rms), "map_waves": np.array(maps)}
    out["phase_seconds"]["opd"] = time.time() - t0
    now = _stats()
    out["phase_stats"]["opd"] = _diff(before, now)
    before, t0 = now, time.time()

    # -- geometric MTF ---------------------------------------------------
    if census is not None:
        census.phase = "mtf"
    mtf = GeometricMTF(
        optic,
        fields=list(fields),
        wavelength="primary",
        num_rays=SAMPLING["mtf_rays"],
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
    if census is not None:
        census.phase = "paraxial"
    out["paraxial"] = {
        k: np.array(float(_np(getattr(optic.paraxial, k)()))) for k in PARAXIAL
    }
    out["phase_seconds"]["paraxial"] = time.time() - t0
    now = _stats()
    out["phase_stats"]["paraxial"] = _diff(before, now)
    before, t0 = now, time.time()

    # -- distortion ------------------------------------------------------
    if census is not None:
        census.phase = "distortion"
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
    name: str,
    rings: int,
    analyses: bool,
    mode: str,
    grad: bool = False,
    fused: str | None = None,
) -> tuple[dict[str, Any], Census]:
    """Emulated run on torch / mps / float64 in representation ``mode``.

    ``fused`` is the value ``OPTILAND_METAL_FUSED_TRACE`` is set to for the
    duration of the run (plan 8.1); the hook reads it per trace.  The returned
    :class:`Census` is the independent predictor of the driver's counters.
    """
    be.set_backend("torch")
    be.set_device("mps")
    be.set_precision("float64")
    _set_grad(grad)
    be.set_metal_mode(mode)
    be.metal_reset_stats()
    _instrument_counters()
    previous = os.environ.get("OPTILAND_METAL_FUSED_TRACE")
    if fused is not None:
        os.environ["OPTILAND_METAL_FUSED_TRACE"] = fused
    try:
        with Census() as census:
            out = run_system(name, rings, analyses, census=census, raw=True)
    finally:
        if fused is not None:
            if previous is None:
                os.environ.pop("OPTILAND_METAL_FUSED_TRACE", None)
            else:
                os.environ["OPTILAND_METAL_FUSED_TRACE"] = previous
    return out, census


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


#: The six structural refusal reasons (plan 1.3), restated here rather than
#: imported so that this census stays independent of the gate.  Every other
#: reason is a feature reason and belongs in identity 2; an unknown reason is
#: therefore counted as a feature reason and makes the identity fail loudly.
STRUCTURAL_REASONS: frozenset[str] = frozenset(
    {
        "group_type",
        "rays_type",
        "rays_shape",
        "host_resident",
        "requires_grad",
        "skip",
    }
)


def _tier(n: int) -> str:
    """The comparison tier a bundle of ``n`` rays supports (plan 7.1)."""
    if n > TIER_A_MIN_RAYS:
        return "A"
    if n > _host_threshold():
        return "B"
    return "host"


def _feature_skips(stats: dict[str, int]) -> int:
    """Refusals of candidate bundles for a feature the kernel lacks."""
    return sum(
        v
        for k, v in stats.items()
        if k.startswith("fused_trace_skip:")
        and k.removeprefix("fused_trace_skip:") not in STRUCTURAL_REASONS
    )


def census_identities(
    census: Census, phase_stats: dict[str, dict[str, int]], switch: str
) -> dict[str, Any]:
    """Plan 1.3's identities and the launch prediction, per phase.

    With the hook off (``switch == "0"``) the identities are vacuous -- the
    hook returns before the gate, so not one ``fused_trace*`` counter may
    exist -- and that absence is what is checked instead.
    """
    problems: list[str] = []
    per_phase: dict[str, Any] = {}
    phases = set(phase_stats) | set(census.candidates)
    for phase in sorted(phases):
        st = phase_stats.get(phase, {})
        counted = census.candidates.get(phase, 0)
        candidates = st.get("fused_trace:candidates", 0)
        traces = st.get("fused_trace:traces", 0)
        late = st.get("fused_trace:late_fallback", 0)
        feature = _feature_skips(st)
        launches = st.get("gpu:fused_trace", 0)
        predicted = census.launches.get(phase, 0)
        if switch == "0":
            present = sorted(k for k in st if k.startswith("fused_trace"))
            if present:
                problems.append(f"{phase}: hook off but {present} were counted")
        else:
            if counted != candidates:
                problems.append(
                    f"{phase}: census {counted} != fused_trace:candidates {candidates}"
                )
            if candidates != traces + feature + late:
                problems.append(
                    f"{phase}: candidates {candidates} != traces {traces} + "
                    f"feature skips {feature} + late_fallback {late}"
                )
            if launches != predicted:
                problems.append(
                    f"{phase}: gpu:fused_trace {launches} != predicted "
                    f"_slab_plan launches {predicted}"
                )
        per_phase[phase] = {
            "census_candidates": counted,
            "candidates": candidates,
            "traces": traces,
            "feature_skips": feature,
            "late_fallback": late,
            "launches": launches,
            "predicted_launches": predicted,
            "n_min": census.n_min.get(phase, 0),
            "n_max": census.n_max.get(phase, 0),
            "tier": _tier(census.n_max.get(phase, 0)),
        }
    return {"problems": problems, "per_phase": per_phase}


def ineligible_check(
    name: str,
    reason: str,
    census: Census,
    totals: dict[str, int],
    n_fields: int,
    n_wavelengths: int,
) -> list[str]:
    """What a ``KNOWN_INELIGIBLE`` system must produce (plan 8.1).

    Zero launches, every candidate refused with the predicted reason, and --
    the exactly predicted number the plan names -- one candidate per field and
    wavelength in the trace phase.
    """
    problems: list[str] = []
    launches = totals.get("gpu:fused_trace", 0)
    if launches != 0:
        problems.append(f"{name}: gpu:fused_trace {launches} != 0")
    refused = totals.get(f"fused_trace_skip:{reason}", 0)
    counted = sum(census.candidates.values())
    if refused != counted:
        problems.append(
            f"{name}: fused_trace_skip:{reason} {refused} != {counted} "
            "candidate bundles"
        )
    expected = n_fields * n_wavelengths
    got = census.candidates.get("trace", 0)
    if got != expected:
        problems.append(
            f"{name}: {got} candidate bundles in the trace phase, expected "
            f"{expected} ({n_fields} fields x {n_wavelengths} wavelengths)"
        )
    return problems


def plan_runs(fused: str, name: str) -> list[tuple[str, str]]:
    """The ``(label, OPTILAND_METAL_FUSED_TRACE)`` runs for one system (plan 8.1).

    ``both`` runs the per-op path first -- that run is R1, the reference the
    fused one is compared against -- and then the fused path, under ``require``
    unless the system is known to be refused, where ``require`` would raise by
    design and ``1`` plus the predicted refusal histogram is the assertion.
    """
    fused_switch = "1" if name in KNOWN_INELIGIBLE else "require"
    return {
        "off": [("perop", "0")],
        "auto": [("fused", "1")],
        "require": [("fused", fused_switch)],
        "both": [("perop", "0"), ("fused", fused_switch)],
    }[fused]


def _quantity_scale(
    phase: str,
    name: str,
    rec: dict[str, Any],
    scale_mm: float,
    wavelength_um: float,
    context: dict[str, Any] | None = None,
) -> float:
    """The scale the tier-B bound of plan 7.1 uses for one quantity.

    Positions and optical path lengths are compared against the system's path
    scale, direction cosines and intensities against 1, OPD in waves against
    the path scale expressed in waves (plan 7.1: "opd the same with the OPL
    scale").  The geometric MTF carries the position difference through
    ``|mean exp(2*pi*i*f*x)|``, whose derivative in ``x`` is ``2*pi*f``, so its
    scale is the path scale times ``2*pi*f_max`` -- a derived amplification,
    not a widened tolerance.  Everything else is compared against its own
    magnitude, floored at 1.
    """
    if phase == "trace":
        return scale_mm if name in (*POSITION, "opd") else 1.0
    if phase == "opd":
        return scale_mm / (wavelength_um * 1e-3)
    if phase in ("spot", "rayfan"):
        return scale_mm
    if phase == "mtf" and name == "mtf" and context is not None:
        freq = np.asarray(context.get("freq_cyc_per_mm", [0.0]), dtype=np.float64)
        return 2 * np.pi * float(np.max(np.abs(freq))) * scale_mm
    return max(1.0, rec["magnitude"])


def compare_fused(
    perop: dict[str, Any],
    fused: dict[str, Any],
    mode: str,
    census: Census,
    scale: float,
) -> dict[str, Any]:
    """The ``fused vs per-op`` column (plan 8.1) under the rule of plan 7.1.

    Tier A -- a phase whose bundles were larger than
    :data:`TIER_A_MIN_RAYS` -- is exact: raw-component equality on every
    recorded surface for the trace phase (the df64 hi/lo words or the sf64 bit
    patterns, never the decoded value), and an identical analysis result
    elsewhere, since identical inputs run through identical per-op reductions.
    Tier B is the bound ``64 * MACHINE_EPS[mode] * scale`` per quantity.

    What fails the run: the trace phase at any tier, and any tier-A phase.  A
    tier-B *analysis* phase is reported with its numbers and its bound but does
    not fail the run on its own -- the same policy the vs-NumPy criterion has
    always had, where the analyses are reported and the traces are gated.
    """
    eps = MACHINE_EPS[mode]
    wavelength = (perop.get("wavelengths_um") or [0.55])[0]
    phases: dict[str, Any] = {}

    n = census.n_max.get("trace", 0)
    tier = _tier(n)
    assert tier != "A" or n > TIER_A_MIN_RAYS, (
        f"phase trace labelled tier A with N = {n} <= {TIER_A_MIN_RAYS}"
    )
    mismatches: list[str] = []
    for key, rows in perop.get("trace_raw", {}).items():
        got_rows = fused.get("trace_raw", {}).get(key, {})
        for q, comps in rows.items():
            got = got_rows.get(q, [])
            same = len(got) == len(comps) and all(
                np.array_equal(a, b, equal_nan=True)
                for a, b in zip(comps, got, strict=True)
            )
            if not same:
                mismatches.append(f"{key}/{q}")
    decoded: dict[str, Any] = {}
    n_diff = 0
    for key, rows in perop["trace"].items():
        for q in QUANTITIES:
            got = fused["trace"][key][q]
            rec = compare_arrays(rows[q], got, zero_mask=(q == "i"))
            both = ~(np.isnan(rows[q]) | np.isnan(got))
            n_diff += int(np.count_nonzero(rows[q][both] != got[both]))
            decoded[q] = _merge(decoded[q], rec) if q in decoded else rec
    for q, rec in decoded.items():
        rec["scale"] = _quantity_scale("trace", q, rec, scale, wavelength)
        rec["bound"] = 64 * eps * rec["scale"]
        rec["within_bound"] = rec["max_abs"] <= rec["bound"]
    nan_agree = all(v["nan_agree"] for v in decoded.values())
    within = all(v["within_bound"] for v in decoded.values())
    worst = max(v["max_abs"] for v in decoded.values())
    phases["trace"] = {
        "tier": tier,
        "n": n,
        "raw_equal": not mismatches,
        "raw_mismatches": mismatches[:20],
        "n_raw_mismatches": len(mismatches),
        "n_differing_elements": n_diff,
        "max_abs": worst,
        "bound": max(v["bound"] for v in decoded.values()),
        "within_bound": within,
        "nan_agree": nan_agree,
        "gated": True,
        "passed": bool(
            (not mismatches and nan_agree) if tier == "A" else (within and nan_agree)
        ),
        "quantities": decoded,
    }

    for phase in PHASES[1:]:
        if phase not in perop or phase not in fused:
            continue
        n = census.n_max.get(phase, 0)
        tier = _tier(n)
        assert tier != "A" or n > TIER_A_MIN_RAYS, (
            f"phase {phase} labelled tier A with N = {n} <= {TIER_A_MIN_RAYS}"
        )
        recs = {}
        for k in perop[phase]:
            rec = compare_arrays(perop[phase][k], fused[phase][k])
            rec["scale"] = _quantity_scale(
                phase, k, rec, scale, wavelength, perop[phase]
            )
            rec["bound"] = 64 * eps * rec["scale"]
            rec["within_bound"] = rec["max_abs"] <= rec["bound"]
            recs[k] = rec
        worst = max((v["max_abs"] for v in recs.values()), default=0.0)
        nan_agree = all(v["nan_agree"] for v in recs.values())
        within = all(v["within_bound"] for v in recs.values())
        exact = worst == 0.0
        phases[phase] = {
            "tier": tier,
            "n": n,
            "max_abs": worst,
            "bound": max((v["bound"] for v in recs.values()), default=0.0),
            "within_bound": within,
            "exact": exact,
            "nan_agree": nan_agree,
            "gated": tier == "A",
            "passed": bool((exact and nan_agree) if tier == "A" else True),
            "advisory": None
            if tier == "A"
            else ("within the tier-B bound" if within else "ABOVE the tier-B bound"),
            "quantities": recs,
        }

    return {
        "phases": phases,
        "passed": all(v["passed"] for v in phases.values()),
        "tier_a_phases": sorted(k for k, v in phases.items() if v["tier"] == "A"),
        "advisories": {
            k: v["advisory"]
            for k, v in phases.items()
            if v.get("advisory") and not v.get("within_bound", True)
        },
        "max_abs": max(v["max_abs"] for v in phases.values()),
    }


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
    name: str,
    mode: str,
    rep: dict[str, Any],
    floor: dict[str, Any] | None,
    st: Any,
    label: str = "",
) -> str:
    """Human-readable block for one system and mode."""
    lines = [f"== {name} ({mode}{', ' + label if label else ''}) =="]
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


def _fused_cell(record: dict[str, Any]) -> str:
    """The ``fused vs per-op`` cell of the trace table."""
    fvp = record.get("fused_vs_perop")
    if not fvp:
        return "n/a"
    trace = fvp["phases"]["trace"]
    if trace["tier"] == "A":
        return (
            "tier A: raw components equal"
            if trace["raw_equal"]
            else f"tier A: **{len(trace['raw_mismatches'])} raw mismatch(es)**"
        )
    return f"tier {trace['tier']}: {_e(trace['max_abs'])} (bound {_e(trace['bound'])})"


def write_markdown(path: str, doc: dict[str, Any]) -> None:
    """Write the NOTES report (tables per mode, fallback list, method)."""
    modes = doc["modes"]
    res = doc["results"]
    names = [n for n in doc["systems"] if n in res]
    L: list[str] = []
    L.append("# 06 — End-to-end oracle report: NumPy float64 vs Metal df64 / sf64")
    L.append("")
    prov = doc.get("provenance", {})
    L.append(
        f"Generated {doc['generated']} by `Optiland-Metal/scripts/metal_oracle_e2e.py`"
        f" (rings={doc['rings']}, fields {list(FIELDS)}, 3 wavelengths per system,"
        f" host threshold {doc['host_threshold']}, autograd"
        f" {'enabled' if doc['grad'] else 'disabled'}, fused"
        f" {doc.get('fused', 'auto')}, sampling {doc.get('sampling', {})},"
        f" torch {doc['torch']}, {doc['device']}). Commit"
        f" `{prov.get('commit', '')}`, mirror table"
        f" `{prov.get('mirror_table_hash', '')[:16]}`."
        " Raw numbers: `NOTES/oracle-e2e.json`."
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
            "torch-cpu floor (pos.) | vs torch-cpu (pos.) | fused vs per-op | "
            "NaN / vignetting (n) | pass |"
        )
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
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
                f" | {_fused_cell(r)}"
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
    L.append("## 6. Fused trace vs the per-op path")
    L.append("")
    if doc.get("fused", "auto") != "both":
        L.append(
            f"This run used `--fused {doc.get('fused', 'auto')}`, so there is no "
            "in-process per-op reference to compare against; `--fused both` "
            "runs the per-op path (R1) and the kernel in one process."
        )
    else:
        L.append(
            "Tier A is raw-component equality (plan 7.1): the df64 hi/lo words "
            "or the sf64 bit patterns of every recorded surface, never the "
            "decoded value. A phase is labelled tier A only when the bundles it "
            "traced were larger than "
            f"{TIER_A_MIN_RAYS} rays; the sizes are measured by the census, not "
            "assumed from the sampling knobs."
        )
        L.append("")
        L.append(
            "| system | mode | phase | tier | N | result | census "
            "(candidates/traces) | launches (got/predicted) |"
        )
        L.append("|---|---|---|---|---|---|---|---|")
        for n in names:
            for mode in modes:
                r = res[n].get(mode, {})
                fvp = r.get("fused_vs_perop")
                if not fvp:
                    continue
                ident = r.get("runs", {}).get("fused", {}).get("identities", {})
                per_phase = ident.get("per_phase", {})
                for phase, v in fvp["phases"].items():
                    cen = per_phase.get(phase, {})
                    if v["tier"] == "A" and "raw_equal" in v:
                        result = "raw equal" if v["raw_equal"] else "**RAW DIFFER**"
                    elif v["tier"] == "A":
                        result = (
                            "identical"
                            if v["max_abs"] == 0
                            else f"**{_e(v['max_abs'])}**"
                        )
                    else:
                        result = f"{_e(v['max_abs'])} <= {_e(v['bound'])}"
                    L.append(
                        f"| {n} | {mode} | {phase} | {v['tier']} | {v['n']} | "
                        f"{result} | {cen.get('census_candidates', 0)}/"
                        f"{cen.get('traces', 0)} | {cen.get('launches', 0)}/"
                        f"{cen.get('predicted_launches', 0)} |"
                    )
        L.append("")
        ineligible = doc.get("known_ineligible", {})
        if ineligible:
            L.append(
                "Systems the gate must refuse, and what was measured "
                "(`gpu:fused_trace` must be 0 and every candidate bundle must "
                "carry the predicted reason):"
            )
            for n, reason in ineligible.items():
                for mode in modes:
                    rec = res.get(n, {}).get(mode, {}).get("runs", {}).get("fused", {})
                    bad = rec.get("ineligible", {}).get("problems", [])
                    counters = rec.get("fused_counters", {})
                    L.append(
                        f"- {n} ({mode}): expected `{reason}`, counters "
                        f"{counters} -> {'OK' if not bad else 'FAIL ' + str(bad)}"
                    )
        problems = doc.get("census_problems", [])
        L.append("")
        L.append(
            f"Census and eligibility problems: {len(problems)}"
            + ("" if not problems else " -- " + "; ".join(problems[:10]))
        )
    L.append("")
    L.append("## 7. Notes")
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


def provenance(n_per_phase: dict[str, Any]) -> dict[str, Any]:
    """``{commit, mirror_table_hash, N_per_phase}`` (plan 8.1).

    The mirror table hash pins which Python sources the kernel was verified
    against (plan 3.7); ``N_per_phase`` is the measured largest bundle each
    phase traced, which is what the tier labels are derived from.
    """
    import subprocess

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo,
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
    return {
        "commit": commit,
        "mirror_table_hash": table,
        "N_per_phase": n_per_phase,
    }


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
    ap.add_argument(
        "--fused",
        choices=("auto", "off", "require", "both"),
        default="auto",
        help="fused trace: 'off' runs the per-op path, 'require' refuses to "
        "fall back for an eligible system, 'both' runs the per-op path (R1) "
        "and then the fused one in the same process and compares them",
    )
    ap.add_argument(
        "--tier-a",
        action="store_true",
        help=f"analysis sampling large enough for tier A ({TIER_A_SAMPLING})",
    )
    ap.add_argument("--rayfan-points", type=int, default=None)
    ap.add_argument("--opd-rings", type=int, default=None)
    ap.add_argument("--mtf-rays", type=int, default=None)
    ap.add_argument("--spot-rings", type=int, default=None)
    ap.add_argument(
        "--diag",
        action="store_true",
        help="OPTILAND_METAL_TRACE_DIAG=1: the kernel writes its status and "
        "iteration planes and the status histogram is printed",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="OPTILAND_METAL_STRICT=1: any CPU fallback raises (plan 1.4)",
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

    if args.tier_a:
        SAMPLING.update(TIER_A_SAMPLING)
    for key, value in (
        ("rayfan_points", args.rayfan_points),
        ("opd_rings", args.opd_rings),
        ("mtf_rays", args.mtf_rays),
        ("spot_rings", args.spot_rings),
    ):
        if value is not None:
            SAMPLING[key] = value
    if args.diag:
        os.environ["OPTILAND_METAL_TRACE_DIAG"] = "1"
    if args.strict:
        os.environ["OPTILAND_METAL_STRICT"] = "1"

    names = list(SYSTEMS) if args.systems == "all" else args.systems.split(",")
    unknown = [n for n in names if n not in SYSTEMS]
    if unknown:
        ap.error(f"unknown systems {unknown}; known: {list(SYSTEMS)}")

    import torch

    results: dict[str, Any] = {}
    ok = True
    problems: list[str] = []
    n_per_phase: dict[str, Any] = {}
    for name in names:
        t0 = time.time()
        ref = run_numpy(name, args.rings, not args.no_analyses)
        t1 = time.time()
        entry: dict[str, Any] = {
            "seconds_numpy": t1 - t0,
            "wavelengths_um": ref["wavelengths_um"],
            "scale_mm": ref["scale"],
            "fields": ref["fields"],
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
            runs = plan_runs(args.fused, name)
            got_by_label: dict[str, Any] = {}
            census_by_label: dict[str, Census] = {}
            records: dict[str, Any] = {}
            failed = False
            for label, switch in runs:
                t2 = time.time()
                try:
                    got, census = run_mps(
                        name,
                        args.rings,
                        not args.no_analyses,
                        mode,
                        args.grad,
                        fused=switch,
                    )
                except Exception:  # noqa: BLE001 - reported, then continue
                    print(
                        f"== {name} ({mode}, {label}) == FAILED\n"
                        f"{traceback.format_exc()}"
                    )
                    entry[mode] = {"error": traceback.format_exc(), "run": label}
                    ok, failed = False, True
                    be.set_backend("numpy")
                    break
                t3 = time.time()
                got_by_label[label], census_by_label[label] = got, census
                rep = compare(ref, got)
                st = summarize_stats(got["phase_stats"])
                totals: Counter[str] = Counter()
                for phase_st in got["phase_stats"].values():
                    totals.update(phase_st)
                ident = census_identities(census, got["phase_stats"], switch)
                passed = (
                    rep["max_position_delta"] < args.tol
                    or rep["max_position_delta_rel"] < tol_rel
                ) and rep["nan_agree"]
                record: dict[str, Any] = {
                    "switch": switch,
                    "vs_numpy": rep,
                    "stats": st,
                    "census": census.report(),
                    "identities": ident,
                    "fused_counters": {
                        k: v
                        for k, v in sorted(totals.items())
                        if k.startswith(("fused_trace", "gpu:fused_trace"))
                    },
                    "seconds_mps": t3 - t2,
                    "phase_seconds": got["phase_seconds"],
                    "passed": bool(passed and not ident["problems"]),
                }
                if cpu is not None:
                    vc = compare(cpu, got)
                    vc.pop("trace_fields")
                    vc.pop("values")
                    record["vs_torch_cpu"] = vc
                if ident["problems"]:
                    problems += [
                        f"{name}/{mode}/{label}: {m}" for m in ident["problems"]
                    ]
                if label == "fused" and name in KNOWN_INELIGIBLE:
                    reason = KNOWN_INELIGIBLE[name]
                    bad = ineligible_check(
                        name,
                        reason,
                        census,
                        dict(totals),
                        len(ref["fields"]),
                        len(ref["wavelengths_um"]),
                    )
                    record["ineligible"] = {"reason": reason, "problems": bad}
                    problems += [f"{mode}: {m}" for m in bad]
                    if bad:
                        record["passed"] = False
                if args.strict:
                    fb = {
                        k: v for k, v in totals.items() if k.startswith("cpu_fallback:")
                    }
                    record["cpu_fallbacks_under_strict"] = fb
                    if fb:
                        problems.append(f"{name}/{mode}/{label}: cpu_fallback {fb}")
                        record["passed"] = False
                if args.diag:
                    record["status_histogram"] = {
                        k.removeprefix("fused_trace:diag:"): v
                        for k, v in sorted(totals.items())
                        if k.startswith("fused_trace:diag:")
                    }
                records[label] = record
                n_per_phase.setdefault(name, {})[mode + "/" + label] = {
                    phase: census.n_max.get(phase, 0) for phase in census.n_max
                }
                print(
                    format_console(
                        name,
                        mode,
                        rep,
                        entry.get("torch_cpu_vs_numpy"),
                        st,
                        label,
                    )
                )
                if "vs_torch_cpu" in record:
                    print(
                        f"  vs torch-cpu float64: position "
                        f"{_e(record['vs_torch_cpu']['max_position_delta'])}"
                        f" mm (rel "
                        f"{_e(record['vs_torch_cpu']['max_position_delta_rel'])})"
                    )
                print(
                    "  census: "
                    + ", ".join(
                        f"{phase} {v['census_candidates']}/{v['candidates']} cand, "
                        f"{v['traces']} traces, {v['launches']}/"
                        f"{v['predicted_launches']} launches, N<={v['n_max']}"
                        f" (tier {v['tier']})"
                        for phase, v in ident["per_phase"].items()
                        if v["census_candidates"] or v["candidates"]
                    )
                    or "no candidate bundle"
                )
                if ident["problems"]:
                    for m in ident["problems"]:
                        print(f"  ** CENSUS: {m}")
                if record.get("ineligible", {}).get("problems"):
                    for m in record["ineligible"]["problems"]:
                        print(f"  ** INELIGIBLE: {m}")
                if args.diag and record.get("status_histogram"):
                    print(f"  status histogram: {record['status_histogram']}")
                print(f"  wall: numpy {t1 - t0:.2f} s, mps {t3 - t2:.2f} s")
                if not record["passed"]:
                    ok = False
                    print(
                        f"  ** FAIL: position delta "
                        f"{rep['max_position_delta']:.3e} mm >= {args.tol:g} and "
                        f"relative {rep['max_position_delta_rel']:.3e} >= "
                        f"{tol_rel:g}, a NaN mismatch, or a census violation"
                    )
                else:
                    clause = (
                        "abs"
                        if rep["max_position_delta"] < args.tol
                        else "scale-relative"
                    )
                    print(f"  PASS ({clause} criterion)")
                be.set_backend("numpy")
            if failed:
                continue
            primary = "perop" if "perop" in records else "fused"
            merged = dict(records[primary])
            merged["runs"] = records
            if "perop" in got_by_label and "fused" in got_by_label:
                fvp = compare_fused(
                    got_by_label["perop"],
                    got_by_label["fused"],
                    mode,
                    census_by_label["fused"],
                    ref["scale"],
                )
                merged["fused_vs_perop"] = fvp
                merged["passed"] = bool(
                    merged["passed"] and records["fused"]["passed"] and fvp["passed"]
                )
                print(
                    "  fused vs per-op: "
                    + ", ".join(
                        f"{phase} tier {v['tier']} N={v['n']} "
                        + (
                            f"raw {'equal' if v['raw_equal'] else 'DIFFER'}"
                            if "raw_equal" in v
                            else f"max {_e(v['max_abs'])}"
                        )
                        for phase, v in fvp["phases"].items()
                    )
                )
                for phase, note in fvp["advisories"].items():
                    v = fvp["phases"][phase]
                    print(
                        f"  advisory: {phase} (tier {v['tier']}, N={v['n']}) is "
                        f"{note}: max {_e(v['max_abs'])} vs {_e(v['bound'])}; "
                        "the analyses are reported, the traces are gated"
                    )
                if not fvp["passed"]:
                    ok = False
                    bad_phases = [
                        f"{phase} (max {_e(v['max_abs'])}, bound {_e(v['bound'])}"
                        + (
                            f", raw mismatches {v['raw_mismatches'][:4]}"
                            if v.get("raw_mismatches")
                            else ""
                        )
                        + ")"
                        for phase, v in fvp["phases"].items()
                        if not v["passed"]
                    ]
                    print(f"  ** FAIL fused vs per-op: {'; '.join(bad_phases)}")
                    problems.append(f"{name}/{mode}: fused vs per-op {bad_phases}")
            entry[mode] = merged

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
    if problems:
        print(f"{len(problems)} census / eligibility problem(s):")
        for message in problems:
            print(f"  ** {message}")
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
        "fused": args.fused,
        "tier_a": args.tier_a,
        "sampling": dict(SAMPLING),
        "fields": list(FIELDS),
        "fields_per_system": {n: [list(f) for f in fields_for(n)] for n in names},
        "trace_only_systems": sorted(TRACE_ONLY_SYSTEMS & set(names)),
        "known_ineligible": {n: r for n, r in KNOWN_INELIGIBLE.items() if n in names},
        "tol": args.tol,
        "tol_rel": {m: 64 * MACHINE_EPS[m] for m in modes},
        "worst_position_error_mm": worst,
        "passed": ok,
        "census_problems": problems,
        "provenance": provenance(n_per_phase),
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
