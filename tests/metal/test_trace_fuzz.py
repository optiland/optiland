"""WP8: randomised fuzz over the supported feature set (plan section 6 "Fuzz").

The conformance files (``test_trace_kernel.py``, ``test_trace_conformance.py``)
compare the kernel against the per-op path on systems a human chose.  This file
compares it on systems nobody chose: 64 seeds, each building a random surface
sequence out of the v1 feature set of plan 1.1 -- 2 to 10 interior surfaces
drawn from ``Plane`` / ``StandardGeometry(inf)`` / conic / ``EvenAsphere`` /
``OddAsphere``, random radii, conics and 0-5 coefficients, three-angle poses
with each angle zero with probability 1/2, the five aperture kinds, absorbing
and non-absorbing glasses, and mirrors -- traced with a random launch bundle
that contains NaN rays, zero-intensity rays and backward rays.

Three assertions per (seed, mode):

1. **Tier A** (``N = 2048 > 1024``, plan 7.1): raw-component equality of every
   recorded row and every final plane -- the df64 ``hi``/``lo`` words, or the
   sf64 int64 bit patterns.  No tolerance exists on this path.
2. **Tier B-1** (``N = 300``, the ``256 < N <= 1024`` site of plan 7.1):
   ``|delta| <= 64 * eps_mode * scale`` per quantity, with the NaN masks and
   the ``i == 0`` masks equal exactly.
3. **The status histogram** equals the pure-NumPy predictor of plan 7.1
   (``_trace_compare.predict_status``), which is derived from the mirrored
   Python sources and not from the kernel.

Both bundles are traced through the *same* ``Optic`` object, once with
``OPTILAND_METAL_FUSED_TRACE=0`` and once with ``1``, so a difference cannot
come from a re-built system; the reference run is asserted not to move a single
fused counter, and the fused run is asserted to have fused exactly once.

``test_fuzz_catalog_sf64_bitexact`` is the second half of plan section 6's fuzz
row: every one of the 29 shipped sample systems, in sf64, on a *random* pupil
bundle rather than the hexapolar sampling the conformance sweep uses, must be
bit-identical to the per-op path.  sf64 is correctly rounded binary64, so there
is no representation slack to hide behind there.

One documented limit, with its lock test: fuzz seed 52 in df64 exceeds tier
B-1's direction-cosine bound (120.875 eps against the plan's 64 eps) at
N = 300, and is **raw-component equal** at N = 2048.  1024 is
``BaseMaterial._MAX_VALUE_KEY_ARRAY_SIZE``, so this is the per-op path's own
sub-1024 dispersion behaviour amplified by a ten-surface random draw, not a
kernel divergence; :data:`TIER_B_DOCUMENTED_LIMITS` records the exact size and
:func:`check_tier_b` fails if it grows *or* if it disappears.

``test_fuzz_corpus_covers_the_feature_set`` keeps the corpus honest: it
compiles the record tables of all 64 systems and asserts that every geometry
code, every aperture code and every surface flag of ``trace_layout`` actually
occurs.  A generator that quietly stopped emitting odd aspheres would otherwise
leave 64 green tests behind.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import warnings  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

torch = pytest.importorskip("torch")
if not torch.backends.mps.is_available():  # pragma: no cover - hardware gate
    pytest.skip("Metal GPU required", allow_module_level=True)

import optiland.backend as be  # noqa: E402
from optiland.backend.torch_backend import metal  # noqa: E402
from optiland.backend.torch_backend.metal import tensor as T  # noqa: E402
from optiland.backend.torch_backend.metal import trace, trace_layout  # noqa: E402
from optiland.backend.torch_backend.metal.tensor import MACHINE_EPS  # noqa: E402
from optiland.backend.torch_backend.metal.trace_record import (  # noqa: E402
    canonical_w0,
)
from optiland.geometries import StandardGeometry  # noqa: E402
from optiland.optic import Optic  # noqa: E402
from optiland.physical_apertures.elliptical import EllipticalAperture  # noqa: E402
from optiland.physical_apertures.offset_radial import (  # noqa: E402
    OffsetRadialAperture,
)
from optiland.physical_apertures.radial import RadialAperture  # noqa: E402
from optiland.physical_apertures.rectangular import RectangularAperture  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import trace_fixtures as fx  # noqa: E402

from tests.metal import _trace_compare as tc  # noqa: E402

MODES = tc.MODES

#: Plan section 6 "Fuzz": 64 seeds.
SEEDS: tuple[int, ...] = tuple(range(64))

#: The tier-A bundle size of the fuzz row (``> 1024``, plan 7.1).
N_TIER_A = 2048

#: Tier-B site 1 of plan 7.1 (``256 < N <= 1024``).
N_TIER_B = 300

#: The one wavelength every fuzz system is defined and traced at.
W0 = 0.5876

#: Radius of the launch disc, in mm.  Every aperture the generator draws is
#: wider than 5 mm in at least one direction and narrower in another, so a
#: bundle of this size is partly clipped by most of them.
LAUNCH_RADIUS = 5.0

#: Where the launch bundle starts, in mm (before every system's first surface).
LAUNCH_Z = -20.0

#: Glasses the generator draws from.  ``"air"`` and ``"mirror"`` have ``k = 0``;
#: the four real glasses are absorbing at 0.5876 um, so ``FL_ABSORBING`` and the
#: kernel's ``exp`` path are both exercised (plan section 6 "Fuzz": glasses with
#: ``k > 0`` and ``k = 0``).  Each is named with its catalog so that no draw
#: depends on Optiland's multi-catalog resolution order.
GLASSES: tuple[Any, ...] = (
    "air",
    "N-BK7",
    "N-SF11",
    "N-SK16",
    ("F2", "schott"),
)

#: The surface kinds, and the weights they are drawn with.
SURFACE_KINDS: tuple[str, ...] = ("plane", "std_inf", "conic", "even", "odd")
SURFACE_WEIGHTS: tuple[float, ...] = (0.15, 0.10, 0.40, 0.20, 0.15)

#: Probability that a surface is a mirror, and the cap on mirrors per system.
MIRROR_P = 0.12
MAX_MIRRORS = 2

#: Fractions of the launch bundle that are special (plan section 6 "Fuzz":
#: "random pupils, some NaN, some ``i = 0``, some backward").
NAN_FRACTION = 0.02
ZERO_INTENSITY_FRACTION = 0.05
BACKWARD_FRACTION = 0.04

#: The direction cosines, which plan 7.1's tier-B rule bounds at ``64 * eps``
#: (scale 1) rather than against the path scale.
COS_ATTRS: frozenset[str] = frozenset({"L", "M", "N"})

#: Documented tier-B limits: ``(seed, mode) -> {quantity, delta, ulps, why}``.
#:
#: **This is not a widened tolerance.**  The plan rule is still evaluated for
#: these rows and must still fail (:func:`check_tier_b` asserts that it does);
#: what is locked is the exact size of the failure, so that a regression which
#: makes it worse -- or a fix which removes it -- fails this file.
#:
#: The one entry is fuzz seed 52 in df64.  Measured, with the certificate that
#: makes it a *per-op* property and not a kernel bug: the same system, the same
#: generator, traced fused vs per-op at N = 300 / 600 / 1024 differs, and at
#: N = 1025 / 2048 is **raw-component equal** (``tc.assert_tier_a`` on this very
#: seed passes in the same test, at N = 2048).  1024 is
#: ``BaseMaterial._MAX_VALUE_KEY_ARRAY_SIZE``: at or below it the per-op path
#: evaluates the dispersion on the whole wavelength array instead of the
#: one-element representative the kernel's ``w0`` mirrors, so the two paths use
#: legitimately different refractive indices.  That difference is what plan
#: 7.1's tier-B site 1 exists for; its constant ``64 * eps`` is calibrated on
#: the shipped systems, and this seed -- ten surfaces, two mirrors, eight
#: tilted poses, four glasses -- amplifies it to 120.875 eps on one direction
#: cosine (positions stay at 0.70 of their bound).  Plan 0.2.2 requires the
#: integrator's sign-off before a tier-B site is added; this entry is recorded
#: in ``status.md`` as a finding awaiting it.
TIER_B_DOCUMENTED_LIMITS: dict[tuple[int, str], dict[str, Any]] = {
    (52, "df64"): {
        "quantity": "N",
        "delta": 4.2943426592501055e-13,
        "ulps": 120.875,
        "why": "sub-1024 dispersion site (plan 7.1 tier B), amplified by the draw",
    },
}


# ---------------------------------------------------------------------------
# Environment and backend
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _driver_state():
    """Zero the counters and the driver's one-shot state around every test."""
    trace.reset_driver_state()
    T.reset_stats()
    yield
    trace.reset_driver_state()
    T.reset_stats()


@pytest.fixture
def mps_backend():
    """torch / mps / float64 with autograd off -- the fused path's setting.

    ``torch.no_grad()`` is load-bearing for the Newton geometries: with
    autograd merely enabled ``NewtonRaphsonGeometry.distance`` takes the
    DiffOptics branch and returns one extra refinement step instead of the
    primal root the kernel mirrors (``test_trace_kernel.mps_backend``).
    """
    previous = metal.get_mode()
    be.set_backend("torch")
    be.set_device("mps")
    be.set_precision("float64")
    be.grad_mode.disable()
    with torch.no_grad():
        assert not torch.is_grad_enabled()
        yield
    metal.set_mode(previous)
    be.grad_mode.disable()
    be.set_backend("numpy")


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FuzzSystem:
    """One generated system and the draw that produced it.

    Attributes:
        optic: the built ``Optic`` (on whatever backend was active).
        surfaces: one ``(kind, material, aperture, pose)`` tuple per interior
            surface, for the failure message.
    """

    optic: Any
    surfaces: tuple[tuple[str, str, str, str], ...]

    def describe(self) -> str:
        """A one-line-per-surface description, for an assertion message."""
        return "\n".join(
            f"  s{i + 1}: {kind:8s} {material:12s} {aperture:22s} {pose}"
            for i, (kind, material, aperture, pose) in enumerate(self.surfaces)
        )


def _draw_aperture(rng: np.random.Generator) -> Any:
    """One of the five supported aperture kinds (plan 1.1), uniformly."""
    kind = int(rng.integers(0, 5))
    if kind == 0:
        return None
    if kind == 1:
        return RadialAperture(r_max=float(rng.uniform(6.0, 12.0)))
    if kind == 2:
        return OffsetRadialAperture(
            r_max=float(rng.uniform(6.0, 12.0)),
            r_min=float(rng.uniform(0.0, 1.5)),
            offset_x=float(rng.uniform(-1.0, 1.0)),
            offset_y=float(rng.uniform(-1.0, 1.0)),
        )
    if kind == 3:
        return RectangularAperture(
            x_min=-float(rng.uniform(5.0, 10.0)),
            x_max=float(rng.uniform(5.0, 10.0)),
            y_min=-float(rng.uniform(5.0, 10.0)),
            y_max=float(rng.uniform(5.0, 10.0)),
        )
    return EllipticalAperture(
        a=float(rng.uniform(5.0, 10.0)),
        b=float(rng.uniform(5.0, 10.0)),
        offset_x=float(rng.uniform(-1.0, 1.0)),
        offset_y=float(rng.uniform(-1.0, 1.0)),
    )


def _draw_surface(rng: np.random.Generator) -> dict[str, Any]:
    """The geometry half of one surface's ``surfaces.add`` keywords.

    Magnitudes are chosen so that the *gate* accepts every draw: radii well
    inside the df64 normal band once squared (``SR_R2``), aperture parameters
    likewise, and the shipped Newton defaults for ``max_iter`` and ``tol``.  A
    draw that the gate refused would turn this file into a test of the
    fallback, not of the kernel.
    """
    kind = SURFACE_KINDS[int(rng.choice(len(SURFACE_KINDS), p=SURFACE_WEIGHTS))]
    kw: dict[str, Any] = {"_kind": kind}
    if kind == "plane":
        kw["radius"] = be.inf
    elif kind == "std_inf":
        # ``GeometryFactory`` collapses an infinite radius to ``Plane``, so
        # ``GEOM_STD_INF`` is reached by replacing the geometry after the
        # build (``_force_standard_infinite`` below, as
        # ``trace_fixtures.planes_both_kinds`` does).  The two are NOT
        # interchangeable: ``Plane.distance`` is a bare ``-z / N`` while the
        # infinite-radius conic branch floors the divisor at 1e-14.
        kw["radius"] = be.inf
    else:
        sign = 1.0 if rng.random() < 0.5 else -1.0
        kw["radius"] = sign * float(rng.uniform(18.0, 220.0))
        kw["conic"] = float(rng.uniform(-1.5, 1.5)) if rng.random() < 0.6 else 0.0
    if kind in ("even", "odd"):
        kw["surface_type"] = "even_asphere" if kind == "even" else "odd_asphere"
        count = int(rng.integers(0, 6))
        kw["coefficients"] = [
            float(rng.uniform(-1.0, 1.0)) * 10.0 ** (-4 - 2 * i) for i in range(count)
        ]
    return kw


def _draw_pose(rng: np.random.Generator) -> dict[str, float]:
    """Three angles, each zero with probability 1/2 (plan section 6 "Fuzz").

    ``rx``/``ry`` stay small (0.03 rad, the tilt magnitude of
    ``trace_fixtures.tilted_triplet``) so a tilted surface still intercepts the
    bundle; ``rz`` is free because a rotation about the axis moves no ray of a
    rotationally symmetric surface but still runs the kernel's ``rz`` branch.
    """
    pose: dict[str, float] = {}
    for axis, limit in (("rx", 0.03), ("ry", 0.03), ("rz", 0.5)):
        if rng.random() < 0.5:
            pose[axis] = float(rng.uniform(-limit, limit))
    return pose


def build_random_system(seed: int) -> FuzzSystem:
    """The system of ``seed``: 2-10 interior surfaces out of the v1 feature set.

    Mirrors flip the sign of every following thickness, the way
    ``HubbleTelescope`` does, so the beam keeps travelling towards the
    surfaces that follow it instead of leaving the system at surface 1.  A
    glass surface is always followed by one that exits to air, so no ray is
    left inside a medium at the image.
    """
    rng = np.random.default_rng(seed)
    count = int(rng.integers(2, 11))
    lens = Optic()
    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)

    direction = 1.0
    in_glass = False
    mirrors = 0
    described: list[tuple[str, str, str, str]] = []
    std_inf: list[tuple[int, float]] = []
    for k in range(count):
        kw = _draw_surface(rng)
        kind = kw.pop("_kind")
        if kind == "std_inf":
            std_inf.append((k + 1, float(rng.uniform(-2.0, 2.0))))
        reflective = bool(rng.random() < MIRROR_P) and mirrors < MAX_MIRRORS
        if reflective:
            material: Any = "mirror"
            mirrors += 1
            direction = -direction
            in_glass = False
        elif in_glass:
            material = "air"
            in_glass = False
        else:
            material = GLASSES[int(rng.integers(0, len(GLASSES)))]
            in_glass = material != "air"
        kw["material"] = material
        kw["thickness"] = direction * float(rng.uniform(1.0, 12.0))
        pose = _draw_pose(rng)
        kw.update(pose)
        aperture = _draw_aperture(rng)
        if aperture is not None:
            kw["aperture"] = aperture
        if k == 0:
            kw["is_stop"] = True
        lens.surfaces.add(index=k + 1, **kw)
        described.append(
            (
                kind,
                material if isinstance(material, str) else material[0],
                type(aperture).__name__ if aperture is not None else "-",
                ", ".join(f"{a}={v:+.4f}" for a, v in pose.items()) or "-",
            )
        )
    if in_glass:  # pragma: no cover - the last draw is usually already air
        lens.surfaces.add(index=count + 1, radius=be.inf, thickness=5.0)
        described.append(("plane", "air", "-", "-"))
        count += 1
    lens.surfaces.add(index=count + 1)
    for index, conic in std_inf:
        surface = lens.surfaces.surfaces[index]
        surface.geometry = StandardGeometry(
            coordinate_system=surface.geometry.cs, radius=be.inf, conic=conic
        )
    lens.set_aperture(aperture_type="EPD", value=2.0 * LAUNCH_RADIUS)
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0.0)
    lens.wavelengths.add(value=W0, is_primary=True)
    return FuzzSystem(optic=lens, surfaces=tuple(described))


def fuzz_bundle(seed: int, n: int) -> Any:
    """``n`` random launch rays: random pupil, some NaN, some ``i = 0``, backward.

    The NaN rays carry a NaN *position* with a finite direction (the ``MISS``
    shape), the zero-intensity rays are geometrically ordinary, and the
    backward rays point away from the system so every surface distance is
    negative -- the ``t < 0`` branch of plan 7.2.
    """
    rng = np.random.default_rng(1_000_003 * (seed + 1) + n)
    radius = LAUNCH_RADIUS * np.sqrt(rng.random(n))
    theta = rng.uniform(0.0, 2.0 * math.pi, n)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.full(n, LAUNCH_Z)
    ell = rng.uniform(-0.05, 0.05, n)
    m = rng.uniform(-0.05, 0.05, n)
    nz = np.sqrt(1.0 - ell * ell - m * m)
    backward = rng.random(n) < BACKWARD_FRACTION
    nz[backward] = -nz[backward]
    intensity = np.ones(n)
    intensity[rng.random(n) < ZERO_INTENSITY_FRACTION] = 0.0
    nan = rng.random(n) < NAN_FRACTION
    x[nan] = np.nan
    return fx.make_rays(x, y, z, ell, m, nz, intensity, W0)


# ---------------------------------------------------------------------------
# Tracing a pair
# ---------------------------------------------------------------------------


def _trace(group: Any, rays: Any, mode: str) -> tuple[tc.Capture, Any]:
    """Trace a clone of ``rays`` through ``group``; capture the raw words.

    The returned bundle is handed back beside the capture because the tier-B
    rule needs it decoded (the capture keeps raw components only).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = group.trace(tc.copy_rays(rays))
    return tc.capture(group, out, mode), out


@dataclass(frozen=True)
class Pair:
    """One per-op / fused comparison of the same bundle through the same optic."""

    ref: tc.Capture
    got: tc.Capture
    ref_final: dict[str, np.ndarray]
    got_final: dict[str, np.ndarray]
    rows: list[dict[str, np.ndarray]]
    deltas: dict[str, int]
    status: np.ndarray | None
    iters: np.ndarray | None


def trace_pair(system: FuzzSystem, rays: Any, mode: str, monkeypatch) -> Pair:
    """Trace ``rays`` twice through ``system``: hook off, then hook on.

    The reference run is asserted not to move any fused counter, which is what
    makes it a reference; the fused run is asserted to have fused exactly once
    (the caller handles the late-fallback case, which fuses zero times and
    writes nothing).
    """
    group = system.optic.surfaces
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    monkeypatch.setenv("OPTILAND_METAL_TRACE_DIAG", "0")
    before_ref = dict(T.stats())
    ref, ref_out = _trace(group, rays, mode)
    rows = tc.decoded_rows(group)
    ref_final = _decoded_final(ref_out)
    after_ref = T.stats()
    moved = {
        k: after_ref[k] - before_ref.get(k, 0)
        for k in after_ref
        if k.startswith(("fused_trace:", "fused_trace_skip:", "gpu:fused_trace"))
        and after_ref[k] != before_ref.get(k, 0)
    }
    assert not moved, f"the hook-off reference run moved a fused counter: {moved}"

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    monkeypatch.setenv("OPTILAND_METAL_TRACE_DIAG", "1")
    before = dict(T.stats())
    got, got_out = _trace(group, rays, mode)
    now = T.stats()
    deltas = {
        k: now.get(k, 0) - before.get(k, 0)
        for k in now
        if "fused" in k and now.get(k, 0) != before.get(k, 0)
    }
    got_final = _decoded_final(got_out)
    status = iters = None
    if deltas.get("fused_trace:traces", 0) == 1:
        planes = trace.diag_from(group)
        assert planes is not None, "DIAG was on but no planes were recorded"
        status = planes[0].cpu().numpy()[0]
        iters = planes[1].cpu().numpy()[0]
    return Pair(
        ref=ref,
        got=got,
        ref_final=ref_final,
        got_final=got_final,
        rows=rows,
        deltas=deltas,
        status=status,
        iters=iters,
    )


def assert_fused_once(pair: Pair, what: str) -> None:
    """Exactly one candidate, one trace, no refusal and no late fallback."""
    assert pair.deltas.get("fused_trace:candidates", 0) == 1, (
        f"{what}: the bundle was not a candidate; deltas {pair.deltas}"
    )
    assert pair.deltas.get("fused_trace:traces", 0) == 1, (
        f"{what}: fused_trace:traces moved by "
        f"{pair.deltas.get('fused_trace:traces', 0)}, not 1; deltas {pair.deltas}"
    )
    refusals = {
        k: v for k, v in pair.deltas.items() if k.startswith("fused_trace_skip:")
    }
    assert not refusals, (
        f"{what}: the generator produced a draw the gate refuses: {refusals}"
    )
    assert pair.deltas.get("fused_trace:late_fallback", 0) == 0, (
        f"{what}: late fallback fired; deltas {pair.deltas}"
    )


def assert_status_histogram(
    pair: Pair, system: FuzzSystem, mode: str, what: str
) -> dict[str, int]:
    """The kernel's status histogram equals the pure-NumPy predictor (plan 7.1).

    The undecidable bands of plan 7.1 are the only exclusions: a float64
    oracle cannot decide an inclusive aperture bound (including one of size
    ZERO, round-1 finding R1-V1-06), nor the sign of a refraction radicand at
    the critical angle (R1-V2-03), nor a Newton loop whose convergence test is
    inside df64's round-off or whose iterate leaves float32's range
    (R1-V1-07), that the mode's own arithmetic decides; ``ST_CLIPPED``,
    ``ST_TIR`` and the Newton bits are masked off on those ``(s, i)`` entries
    -- in the prediction and in the measurement alike -- and on nothing else.
    Every band is empty in sf64, where this stays an exact comparison.
    """
    assert pair.status is not None and pair.iters is not None, f"{what}: no DIAG planes"
    w0 = canonical_w0(W0, mode)
    records = tc.compile_tables(system.optic, mode, w0)
    prediction = tc.predict_status(pair.rows, system.optic, mode=mode, records=records)
    assert pair.status.shape == prediction.bits.shape, (
        f"{what}: status plane {pair.status.shape} vs prediction "
        f"{prediction.bits.shape}"
    )
    got = tc.mask_uncertain(pair.status, prediction)
    want = tc.mask_uncertain(prediction.bits, prediction)
    got_counts = tc.bit_counts(got)
    want_counts = tc.bit_counts(want)
    assert got_counts == want_counts, (
        f"{what}: status histogram {got_counts} vs predicted {want_counts}\n"
        f"{system.describe()}"
    )
    got_iters = tc.mask_iters(pair.iters, prediction)
    want_iters = tc.mask_iters(prediction.iters, prediction)
    assert np.array_equal(got_iters, want_iters), (
        f"{what}: iters differ from the prediction at "
        f"{int(np.count_nonzero(got_iters != want_iters))} entries\n"
        f"{system.describe()}"
    )
    return got_counts


# ---------------------------------------------------------------------------
# 1. The corpus itself
# ---------------------------------------------------------------------------


def _corpus_codes(mode: str) -> tuple[set[int], set[int], int]:
    """``(geometry codes, aperture codes, OR of every flag)`` over all 64 seeds."""
    geometries: set[int] = set()
    apertures: set[int] = set()
    flags = 0
    for seed in SEEDS:
        system = build_random_system(seed)
        records = tc.compile_tables(system.optic, mode, canonical_w0(W0, mode))
        surf_int = np.asarray(records.surf_int)[0]
        for row in surf_int[1:]:
            geometries.add(int(row[trace_layout.SI_GEOM]))
            apertures.add(int(row[trace_layout.SI_APCODE]))
            flags |= int(row[trace_layout.SI_FLAGS])
        geometries.add(int(surf_int[0][trace_layout.SI_GEOM]))
    return geometries, apertures, flags


def test_fuzz_corpus_covers_the_feature_set(mps_backend):
    """Every geometry code, aperture code and surface flag occurs in the corpus.

    Compiled through the real record compiler, so this is what the *kernel*
    would see, not what the generator believes it drew.  Without it a
    generator that stopped emitting (say) odd aspheres would leave 128 green
    comparisons behind and nothing would notice.
    """
    metal.set_mode("df64")
    geometries, apertures, flags = _corpus_codes("df64")

    want_geometries = {
        trace_layout.GEOM_OBJECT,
        trace_layout.GEOM_PLANE,
        trace_layout.GEOM_STD_INF,
        trace_layout.GEOM_CONIC,
        trace_layout.GEOM_EVEN,
        trace_layout.GEOM_ODD,
    }
    assert geometries == want_geometries, (
        f"geometry codes in the corpus: {sorted(geometries)}, "
        f"want {sorted(want_geometries)}"
    )
    want_apertures = {
        trace_layout.AP_NONE,
        trace_layout.AP_RADIAL,
        trace_layout.AP_OFFSET_RADIAL,
        trace_layout.AP_RECT,
        trace_layout.AP_ELLIPSE,
    }
    assert apertures == want_apertures, (
        f"aperture codes in the corpus: {sorted(apertures)}, "
        f"want {sorted(want_apertures)}"
    )
    want_flags = (
        trace_layout.FL_HAS_RX
        | trace_layout.FL_HAS_RY
        | trace_layout.FL_HAS_RZ
        | trace_layout.FL_REFLECTIVE
        | trace_layout.FL_HAS_APERTURE
        | trace_layout.FL_AP_IN_ROOT
        | trace_layout.FL_ABSORBING
        | trace_layout.FL_RADIUS_INF
    )
    missing = want_flags & ~flags
    assert missing == 0, f"surface flags never set in the corpus: {missing:#04x}"


def test_fuzz_bundles_contain_their_special_rays():
    """The launch bundle really carries NaN, ``i = 0`` and backward rays.

    Pure NumPy (the draw happens before ``be.array``), so it runs without the
    GPU and fails loudly if a fraction constant is ever zeroed.
    """
    rng = np.random.default_rng(1_000_003 * 1 + N_TIER_A)
    rng.random(N_TIER_A)  # radius
    rng.uniform(0.0, 2.0 * math.pi, N_TIER_A)  # theta
    rng.uniform(-0.05, 0.05, N_TIER_A)  # L
    rng.uniform(-0.05, 0.05, N_TIER_A)  # M
    backward = int(np.count_nonzero(rng.random(N_TIER_A) < BACKWARD_FRACTION))
    zero = int(np.count_nonzero(rng.random(N_TIER_A) < ZERO_INTENSITY_FRACTION))
    nan = int(np.count_nonzero(rng.random(N_TIER_A) < NAN_FRACTION))
    assert backward > 0 and zero > 0 and nan > 0, (backward, zero, nan)
    assert backward + zero + nan < N_TIER_A // 4, (backward, zero, nan)


# ---------------------------------------------------------------------------
# 2. The fuzz comparison (plan section 6 "Fuzz")
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("seed", SEEDS)
def test_fuzz_random_systems(mps_backend, monkeypatch, seed, mode):
    """A random system, both bundle sizes: tier A, tier B-1 and the histogram.

    Every failure message carries the surface-by-surface draw, so a red seed is
    reproducible from the message alone.
    """
    metal.set_mode(mode)
    system = build_random_system(seed)

    rays_a = fuzz_bundle(seed, N_TIER_A)
    pair_a = trace_pair(system, rays_a, mode, monkeypatch)
    what = f"fuzz[seed={seed}][{mode}][N={N_TIER_A}]"
    assert_fused_once(pair_a, f"{what}\n{system.describe()}")
    tc.assert_tier_a(pair_a.got, pair_a.ref, f"{what}\n{system.describe()}")
    counts = assert_status_histogram(pair_a, system, mode, what)

    rays_b = fuzz_bundle(seed, N_TIER_B)
    pair_b = trace_pair(system, rays_b, mode, monkeypatch)
    what_b = f"fuzz[seed={seed}][{mode}][N={N_TIER_B}]"
    assert_fused_once(pair_b, f"{what_b}\n{system.describe()}")
    assert tc.TIER_B_MIN_RAYS < N_TIER_B <= tc.TIER_B_MAX_RAYS
    check_tier_b(system, pair_b, seed, mode, f"{what_b}\n{system.describe()}")
    print(f"{what}: S={len(pair_a.ref.rows)} status {counts}")


def check_tier_b(
    system: FuzzSystem, pair: Pair, seed: int, mode: str, what: str
) -> None:
    """Plan 7.1's tier-B rule on ``pair``, or the documented limit's lock test.

    For every (seed, mode) but the one row of
    :data:`TIER_B_DOCUMENTED_LIMITS` this is ``_trace_compare.assert_tier_b``
    with the plan's own constants -- the rule lives there and is not restated
    here.  For a listed row it is the "documented limit + lock test" closure of
    plan section 6: the plan rule must STILL fail (a limit that has healed is a
    stale exemption and fails here), the measured worst difference must not have
    grown past the recorded value, and every other quantity must be inside its
    bound.
    """
    got_rows = tc.decoded_rows(system.optic.surfaces)
    scale = tc.system_scale(system.optic, pair.rows)
    limit = TIER_B_DOCUMENTED_LIMITS.get((seed, mode))
    if limit is None:
        tc.assert_tier_b(got_rows, pair.rows, mode=mode, scale=scale, what=what)
        assert_final_tier_b(pair, mode, scale, what)
        return

    with pytest.raises(AssertionError):
        tc.assert_tier_b(got_rows, pair.rows, mode=mode, scale=scale, what=what)

    eps = MACHINE_EPS[mode]
    worst = tier_b_worst(got_rows, pair.rows, pair)
    quantity = limit["quantity"]
    bound = (
        (tc.TIER_FACTOR * eps)
        if quantity in COS_ATTRS
        else (tc.TIER_FACTOR * eps * scale)
    )
    assert worst[quantity] > bound, (
        f"{what}: the documented tier-B limit on {quantity} has healed "
        f"({worst[quantity]:.6e} <= {bound:.6e}); remove the "
        f"TIER_B_DOCUMENTED_LIMITS entry instead of carrying it"
    )
    assert worst[quantity] <= limit["delta"], (
        f"{what}: the documented tier-B limit on {quantity} GREW: "
        f"{worst[quantity]:.17e} > {limit['delta']:.17e} "
        f"({worst[quantity] / eps:.6f} eps vs {limit['ulps']:.6f} eps)"
    )
    for attr, value in sorted(worst.items()):
        if attr == quantity or attr == "intensity":
            continue
        other = (
            (tc.TIER_FACTOR * eps)
            if attr in COS_ATTRS
            else (tc.TIER_FACTOR * eps * scale)
        )
        assert value <= other, (
            f"{what}: {attr} is outside the tier-B bound too "
            f"({value:.3e} > {other:.3e}); the documented limit covers only "
            f"{quantity}"
        )


def tier_b_worst(
    got_rows: list[dict[str, np.ndarray]],
    ref_rows: list[dict[str, np.ndarray]],
    pair: Pair,
) -> dict[str, float]:
    """Worst ``|delta|`` per quantity over the rows AND the returned bundle."""
    worst: dict[str, float] = {}
    pairs = list(zip(got_rows, ref_rows, strict=True))
    pairs.append((_as_tier_b_row(pair.got_final), _as_tier_b_row(pair.ref_final)))
    for got, ref in pairs:
        for attr in sorted(set(got) & set(ref)):
            a, b = got[attr], ref[attr]
            both = np.isfinite(a) & np.isfinite(b)
            delta = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
            worst[attr] = max(worst.get(attr, 0.0), delta)
    return worst


#: The returned bundle's planes the tier-B rule reads (``L0/M0/N0`` are the
#: launch directions and are never NaN).
FINAL_ATTRS: tuple[str, ...] = ("x", "y", "z", "L", "M", "N", "i", "opd")


def _decoded_final(rays: Any) -> dict[str, np.ndarray]:
    """The returned bundle's planes as float64 host arrays."""
    return {attr: tc.decode(getattr(rays, attr)) for attr in FINAL_ATTRS}


def assert_final_tier_b(pair: Pair, mode: str, scale: float, what: str) -> None:
    """Plan 7.1 tier B on the RETURNED bundle, not only on the recorded rows.

    ``assert_tier_b`` covers the snapshots; the eleven final planes are what a
    caller of ``SurfaceGroup.trace`` actually holds, so they get the same
    eps-derived bound, the same NaN rule and the same ``i == 0`` rule.
    """
    tc.assert_tier_b(
        [_as_tier_b_row(pair.got_final)],
        [_as_tier_b_row(pair.ref_final)],
        mode=mode,
        scale=scale,
        what=f"{what}: returned bundle",
    )


def _as_tier_b_row(final: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """A final-plane dict in the shape ``assert_tier_b`` reads a row in."""
    row = {k: v for k, v in final.items() if k != "i"}
    row["intensity"] = final["i"]
    return row


# ---------------------------------------------------------------------------
# 3. The shipped catalog, in sf64, on a random bundle
# ---------------------------------------------------------------------------


def _catalog_bundle(optic: Any, name: str, n: int = N_TIER_A) -> Any:
    """``n`` rays through ``optic``'s own pupil at a per-name random field.

    Built by the optic's own ray generator (so a system with a ray aimer is
    aimed exactly as ``optic.trace`` would aim it) on a random pupil sampling
    rather than the hexapolar rings the conformance sweep uses: the point of
    this row is a sampling nobody tuned.
    """
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    radius = np.sqrt(rng.random(n))
    theta = rng.uniform(0.0, 2.0 * math.pi, n)
    field_r = 0.99 * math.sqrt(float(rng.random()))
    field_t = 2.0 * math.pi * float(rng.random())
    generator = optic.ray_tracer.ray_generator
    return generator.generate_rays(
        be.array(np.full(n, field_r * math.cos(field_t))),
        be.array(np.full(n, field_r * math.sin(field_t))),
        be.array(radius * np.cos(theta)),
        be.array(radius * np.sin(theta)),
        optic.primary_wavelength,
    )


@pytest.mark.parametrize("name", sorted(fx.CATALOG))
def test_fuzz_catalog_sf64_bitexact(mps_backend, monkeypatch, name):
    """Every shipped system, sf64, random bundle: int64 bit patterns equal.

    sf64 is correctly rounded binary64 on both paths, so the only thing this
    can find is a mirror bug; there is no representation slack to absorb one.
    """
    if name in fx.KNOWN_INELIGIBLE:  # pragma: no cover - the map is empty today
        pytest.skip(f"{name}: {fx.KNOWN_INELIGIBLE[name]}")
    metal.set_mode("sf64")
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    monkeypatch.setenv("OPTILAND_METAL_TRACE_DIAG", "0")
    optic = fx.build(name)
    rays = _catalog_bundle(optic, name)

    group = optic.surfaces
    before_ref = dict(T.stats())
    ref, _ = _trace(group, rays, "sf64")
    after_ref = T.stats()
    moved = {
        k: after_ref[k] - before_ref.get(k, 0)
        for k in after_ref
        if k.startswith(("fused_trace:", "fused_trace_skip:", "gpu:fused_trace"))
        and after_ref[k] != before_ref.get(k, 0)
    }
    assert not moved, f"{name}: the hook-off reference run moved {moved}"

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    before = dict(T.stats())
    got, _ = _trace(group, rays, "sf64")
    now = T.stats()
    deltas = {
        k: now.get(k, 0) - before.get(k, 0)
        for k in now
        if "fused" in k and now.get(k, 0) != before.get(k, 0)
    }
    assert deltas.get("fused_trace:traces", 0) == 1, f"{name}: deltas {deltas}"
    refusals = {k: v for k, v in deltas.items() if k.startswith("fused_trace_skip:")}
    assert not refusals, f"{name}: {refusals}"
    tc.assert_tier_a(got, ref, f"catalog-sf64[{name}]")
