"""WP6-a: the feature matrix of plan 7.2, rows 1-25, against the WP5 fixtures.

Every row of plan 7.2 from "spherical refraction" through "tier-B site 1" runs
here, in both representations, against `scripts/trace_fixtures.py`.  Four kinds
of assertion, in order of strength:

1. **Tier A (plan 7.1)** -- ``test_fused_equals_perop``: the fused trace and
   the per-op trace of the *same* bundle agree on the RAW COMPONENTS (df64
   ``hi``/``lo`` words, sf64 int64 bit patterns) of all eight recorded planes of
   every surface row and all eleven final planes.  No tolerance exists on this
   path.  The two traces run in one process on two independently built optics
   from one launch bundle cloned component-wise, so nothing but the trace
   differs.
2. **Predicted status (plan 7.1)** -- ``test_predicted_status``: the pure-NumPy
   oracle ``_trace_compare.predict_status`` predicts the ``status``/``iters``
   planes per ``(surface, ray)`` from the record tables and the reference rows,
   and the kernel's DIAG planes must equal that prediction entry for entry.
   Each row of the matrix also asserts its "expected bits > 0" column, so a
   branch fixture that stopped exercising its branch fails instead of passing
   vacuously.
3. **External (plan 7.1)** -- ``test_fused_vs_numpy``: the same system traced
   on the NumPy float64 backend with its own ray generation, compared after
   decoding under the external rule (positions ``< 1e-11 mm`` or
   ``64 eps scale``, cosines ``< 64 eps``, NaN patterns equal, ``i == 0``
   patterns equal outside the per-aperture rim band).
4. **Counters (plan 7.2, last-but-four row)** -- ``test_predicted_counters``:
   every counter the trace moves is predicted from an independent census, never
   read off the run.

``predict_status`` itself is unit-tested on hand-built records
(``test_predict_status_*``) so that the oracle cannot pass by agreeing with the
thing it checks [fix: L2.16].

Sampling (plan 7.1): 4096-ray golden-angle pupil bundles -- above the 1024-ray
floor tier A needs in df64 and above the 256-ray host-residency threshold --
fields (0, 0), (0, 0.7), (0, 1) and (0.7, 0) for the tilted systems, all three
wavelengths where the system has three, one 4096-ray random bundle, and the
edge bundles each fixture builds for itself.  Tier-B site 1 is the 300-ray
Cooke bundle of ``test_tier_b_small_bundle`` and is the ONLY comparison in this
file that is not raw-component equality.

Section 7 (WP6-b) adds plan section 6's standing **round 0**:
``test_divergence_injection[site-mode]`` breaks one mirrored expression per
site -- five in the Python path, four through the kernel's
``#ifdef OPTILAND_TRACE_BREAK_*`` hooks -- proves with a *certificate* that the
break is observable at all, and only then asserts that the tier-A comparison
above notices it.  Without that, every "fused == per-op" assertion in this file
would be worth exactly as much as its untested ability to fail.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import inspect  # noqa: E402
import textwrap  # noqa: E402
import warnings  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from typing import TYPE_CHECKING, Any  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

torch = pytest.importorskip("torch")
if not torch.backends.mps.is_available():  # pragma: no cover - hardware gate
    pytest.skip("Metal GPU required", allow_module_level=True)

import optiland.backend as be  # noqa: E402
from optiland.backend.torch_backend import metal  # noqa: E402
from optiland.backend.torch_backend.metal import tensor as T  # noqa: E402
from optiland.backend.torch_backend.metal import (  # noqa: E402
    trace,
    trace_layout,
    trace_mirror,
)
from optiland.backend.torch_backend.metal.trace_adapters import (  # noqa: E402
    FusedTraceSkip,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import trace_fixtures as fx  # noqa: E402

from tests.metal import _trace_compare as tc  # noqa: E402
from tests.metal import _trace_probe_lib as tp  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

MODES = tc.MODES

#: The bundle size every tier-A row uses.
N_RAYS = fx.DEFAULT_RAYS

#: Fields, as ``(Hx, Hy)``, for a fixture that is launched through its optic's
#: own pupil (plan 7.1).
FIELDS_AXIAL: tuple[tuple[float, float], ...] = ((0.0, 0.0),)
FIELDS_FULL: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.0, 0.7), (0.0, 1.0))
FIELDS_TILTED: tuple[tuple[float, float], ...] = FIELDS_FULL + ((0.7, 0.0),)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One (fixture, sampling) row of the feature matrix.

    Attributes:
        id: The parametrization id, matching the ``test name`` column of plan
            7.2 (``...[cooke]`` -> ``cooke``).
        fixture: The ``trace_fixtures`` entry name.
        field: ``(Hx, Hy)`` when the bundle comes from the optic's pupil,
            None when the fixture ships its own rays factory.
        wavelength: An explicit wavelength, or None for the primary one.
        bits: Status bits the row must show at least once (plan 7.2's
            "expected bits > 0" column), by ``_trace_compare`` name.
        rays: The bundle size.
    """

    id: str
    fixture: str
    field: tuple[float, float] | None = None
    wavelength: float | None = None
    bits: tuple[str, ...] = ()
    rays: int = N_RAYS


def _pupil_cases(
    name: str,
    fields: tuple[tuple[float, float], ...],
    *,
    prefix: str | None = None,
    bits: tuple[str, ...] = (),
    wavelengths: tuple[float, ...] = (),
) -> list[Case]:
    """One case per field (and per extra wavelength) of a pupil-launched row."""
    base = prefix or name
    out: list[Case] = []
    for hx, hy in fields:
        suffix = "" if (hx, hy) == (0.0, 0.0) else f"-f{hx:g},{hy:g}"
        out.append(Case(f"{base}{suffix}", name, field=(hx, hy), bits=bits))
    for w in wavelengths:
        out.append(
            Case(f"{base}-w{w:g}", name, field=(0.0, 0.0), wavelength=w, bits=bits)
        )
    return out


#: Plan 7.2 rows 1-7 and 10-14, 19-23: the ones whose assertion is
#: "fused == per-op, raw components" over a whole bundle.
CASES: tuple[Case, ...] = tuple(
    [
        # spherical refraction, absorption, image refraction + OPD
        *_pupil_cases("cooke", FIELDS_FULL, wavelengths=(0.48, 0.65)),
        # reflection, annulus in clip and root selection, r_max = inf
        *_pupil_cases("hubble", FIELDS_FULL, bits=("clipped",)),
        # reflection under rx+ry+rz+decenter with absorption
        Case("tilted_fold", "tilted_fold_mirror"),
        # even asphere Newton
        *_pupil_cases("aspheric_singlet", FIELDS_FULL),
        # even asphere powers >= 4 / R = inf seed / odd asphere
        Case("even5", "even_asphere_5coeff"),
        Case("even_inf", "even_asphere_inf_radius"),
        # even asphere with an INEXACT conic stored as a Python float: the
        # only matrix row whose SR_K1 has a non-zero df64 low word and whose
        # `(1 + k) * r2` product is launched with the slot on the right
        # (round-2 finding R2-V1-03).
        Case("even_inexact_conic", "inexact_conic_asphere"),
        *_pupil_cases("odd_asphere_singlet", FIELDS_AXIAL, prefix="odd"),
        # Plane and StandardGeometry(inf).  Plan 7.2 lists NZ_FLOORED for this
        # row; the WP5 bundle is tilted to (0.2, 0.1) and floors nothing
        # (measured: no status bit at all).  The |N| <= 1e-14 edge case the
        # plan asks for lives in test_std_inf_floors_a_grazing_ray, which
        # builds its own bundle.
        Case("planes", "planes_both_kinds"),
        # decenter / tilt, five poses
        *[
            Case(f"tilt-{pose}", f"tilted_triplet_{pose}", field=(0.7, 0.0))
            for pose in fx.TILTED_TRIPLET_POSES
        ],
        # fold mirror
        Case("fold", "fold_mirror"),
        # rect / ellipse / offset-radial / r_max = 0
        Case("ap-rect", "rect_aperture", bits=("clipped",)),
        Case("ap-ellipse", "ellipse_aperture", bits=("clipped",)),
        Case("ap-offset_radial", "offset_radial_aperture", bits=("clipped",)),
        Case("ap-zero_rmax", "zero_rmax_aperture", bits=("clipped",)),
        # finite conjugate, 44 surfaces
        *_pupil_cases("uv_projection", FIELDS_AXIAL, prefix="uv"),
        # non-zero image thickness, apodized launch
        Case("nonzero_image_thickness", "nonzero_image_thickness", field=(0.0, 0.0)),
        Case("apodized", "apodized_bundle"),
        # off-axis parabola far-root preference
        Case("oap", "off_axis_parabola_far_root"),
        # backward propagation, t < 0
        Case("backward_plane", "backward_plane"),
        Case("backward_newton", "backward_newton"),
        # branch fixtures (their named tests assert the branch; the tier-A row
        # asserts that the branch reproduces bit for bit)
        Case("tir", "tir_singlet", bits=("tir",)),
        Case("miss", "miss_bundle", bits=("miss", "clipped")),
        # Plan 7.2 lists MISS for the reverse bundle; the fixture is built so
        # that BOTH roots are non-positive and the vertex-nearest fallback
        # returns a NEGATIVE root -- finite, not NaN (its docstring says
        # "implementation-defined per root").  test_reverse_miss asserts the
        # branch that is actually taken.
        Case("reverse", "reverse_bundle"),
        Case("grazing", "grazing_bundle"),
        Case("exact_grazing", "exact_grazing_bundle"),
        Case("rim", "rim_bundle"),
    ]
)

CASES_BY_ID: dict[str, Case] = {c.id: c for c in CASES}


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

    ``torch.no_grad()`` is load-bearing and not a formality.  ``be.grad_mode
    .disable()`` only clears the ``requires_grad`` flag new arrays are created
    with (``torch_backend/config.py:31-33``); it leaves
    ``torch.is_grad_enabled()`` at its default True.  With autograd enabled
    ``NewtonRaphsonGeometry.distance`` takes the DiffOptics branch and returns
    ``t - F(t) / (dF/dt)`` instead of the primal ``result.t``
    (``newton_raphson.py:571-608``) -- one extra refinement step, which differs
    from the primal root in the low word.  The kernel mirrors the primal solve,
    so the primal solve is the reference, and ``torch.no_grad()`` is what makes
    R1 that path (WP1's ``test_trace_units.metal_backend`` says the same).
    Measured here without it: every Newton fixture's ``opd`` and direction
    cosines differ from the fused trace in the last 1-2 ulp
    (``aspheric_singlet`` 594/4096 rays, ``backward_newton`` 5/4096,
    ``even_asphere_5coeff`` 1/4096, ``grazing_bundle`` 1470/4096) -- see the
    status note for WP2: the gate refuses ``requires_grad`` only when a tensor
    actually carries the flag, not when autograd is merely enabled.
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


@pytest.fixture
def numpy_backend():
    """The NumPy float64 reference backend (R2)."""
    be.set_backend("numpy")
    yield
    be.set_backend("numpy")


# ---------------------------------------------------------------------------
# Building and tracing
# ---------------------------------------------------------------------------


def build_case(case: Case) -> tuple[Any, Callable[[], Any]]:
    """``(optic, make_rays)`` for ``case`` on the ACTIVE backend.

    ``make_rays`` is called once per trace; every call returns a fresh bundle
    built the same way, so two traces of the same case start from the same
    numbers.
    """
    built = fx.FIXTURES[case.fixture]()
    if isinstance(built, tuple):
        optic, factory = built

        def make() -> Any:
            return factory(optic, case.rays)

        return optic, make

    optic = built
    hx, hy = case.field if case.field is not None else (0.0, 0.0)
    wavelength = case.wavelength

    def make() -> Any:
        return fx.pupil_bundle(optic, case.rays, Hx=hx, Hy=hy, wavelength=wavelength)

    return optic, make


def bundle_w0(rays: Any) -> float:
    """The canonical wavelength of ``rays``, exactly as the gate reads it.

    ``can_fuse_trace`` decodes ``w[0]`` off the device (plan 3.8), so this is
    the value the record tables are built at -- not ``optic.primary_wavelength``,
    which several fixtures deliberately do not match (``miss_bundle`` launches
    a 0.5876 um bundle into a Cooke triplet whose primary is 0.55 um).
    """
    return float(np.asarray(tc.decode(rays.w)).reshape(-1)[0])


def trace_perop(case: Case, monkeypatch) -> tuple[Any, Any, tc.Capture, float]:
    """Trace ``case`` with the hook switched off (reference R1)."""
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    optic, make = build_case(case)
    rays = make()
    w0 = bundle_w0(rays)
    out = optic.surfaces.trace(tc.copy_rays(rays))
    return optic, out, tc.capture(optic.surfaces, out, metal.get_mode()), w0


def trace_fused(
    case: Case, monkeypatch, *, diag: bool = False
) -> tuple[Any, Any, tc.Capture]:
    """Trace ``case`` through the fused kernel."""
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    monkeypatch.setenv("OPTILAND_METAL_TRACE_DIAG", "1" if diag else "0")
    optic, make = build_case(case)
    rays = make()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = optic.surfaces.trace(tc.copy_rays(rays))
    return optic, out, tc.capture(optic.surfaces, out, metal.get_mode())


def assert_was_fused(before: dict[str, int], what: str) -> None:
    """The fused path actually ran: exactly one candidate and one trace."""
    now = T.stats()
    traces = now.get("fused_trace:traces", 0) - before.get("fused_trace:traces", 0)
    skips = {
        k: now.get(k, 0) - before.get(k, 0)
        for k in now
        if k.startswith("fused_trace_skip:")
    }
    skips = {k: v for k, v in skips.items() if v}
    assert traces == 1, f"{what}: fused_trace:traces moved by {traces}, not 1; {skips}"


# ---------------------------------------------------------------------------
# 1. Tier A: fused == per-op, raw components (plan 7.1, plan 7.2 rows 1-25)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_fused_equals_perop(mps_backend, monkeypatch, case, mode):
    """Plan 7.1 tier A: every raw word of every recorded row and final plane."""
    metal.set_mode(mode)
    _, _, ref, _ = trace_perop(case, monkeypatch)
    before = dict(T.stats())
    _, _, got = trace_fused(case, monkeypatch)
    assert_was_fused(before, case.id)
    tc.assert_tier_a(got, ref, f"{case.id}[{mode}]")


@pytest.mark.parametrize("mode", MODES)
def test_fused_equals_perop_random_bundle(mps_backend, monkeypatch, mode):
    """The random bundle of plan 7.1: uniform ``Px, Py, Hx, Hy`` in the disc."""
    metal.set_mode(mode)
    rng = np.random.default_rng(20260917)
    n = N_RAYS
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    rp = np.sqrt(rng.uniform(0.0, 1.0, n))
    px, py = rp * np.cos(theta), rp * np.sin(theta)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    rh = np.sqrt(rng.uniform(0.0, 1.0, n))
    hx, hy = rh * np.cos(theta), rh * np.sin(theta)

    def run(fused: str) -> tc.Capture:
        monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", fused)
        optic = fx.cooke()
        generator = optic.ray_tracer.ray_generator
        rays = generator.generate_rays(
            be.array(hx),
            be.array(hy),
            be.array(px),
            be.array(py),
            optic.primary_wavelength,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = optic.surfaces.trace(rays)
        return tc.capture(optic.surfaces, out, mode)

    ref = run("0")
    before = dict(T.stats())
    got = run("1")
    assert_was_fused(before, "random")
    tc.assert_tier_a(got, ref, f"random[{mode}]")


# ---------------------------------------------------------------------------
# 2. Predicted status and iters (plan 7.1 "Status/iters histograms")
# ---------------------------------------------------------------------------


def predicted_and_actual(case: Case, monkeypatch, mode: str):
    """``(prediction, status, iters, reference rows)`` for one case."""
    optic_ref, _, _, w0 = trace_perop(case, monkeypatch)
    rows = tc.decoded_rows(optic_ref.surfaces)
    records = tc.compile_tables(optic_ref, mode, w0)
    prediction = tc.predict_status(rows, optic_ref, mode=mode, records=records)

    before = dict(T.stats())
    optic_f, _, _ = trace_fused(case, monkeypatch, diag=True)
    assert_was_fused(before, case.id)
    planes = trace.diag_from(optic_f.surfaces)
    assert planes is not None, f"{case.id}: DIAG planes were not recorded"
    status = planes[0].cpu().numpy()[0]
    iters = planes[1].cpu().numpy()[0]
    return prediction, status, iters, rows


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_predicted_status(mps_backend, monkeypatch, case, mode):
    """The kernel's status/iters planes equal the pure-NumPy prediction.

    The undecidable bands of plan 7.1 are the only exceptions: the float64
    oracle cannot decide an inclusive aperture bound, nor the sign of a
    refraction radicand at the critical angle (round-1 finding R1-V2-03), nor
    a Newton loop whose convergence test sits inside df64's round-off or whose
    iterate leaves float32's range (round-1 finding R1-V1-07), that the mode's
    own arithmetic decides; those ``(s, i)`` entries are excluded from the
    ``CLIPPED`` / ``TIR`` / Newton comparison -- and from nothing else.  All
    three bands are empty in sf64, so this is an exact comparison there.
    """
    metal.set_mode(mode)
    prediction, status, iters, _ = predicted_and_actual(case, monkeypatch, mode)
    assert status.shape == prediction.bits.shape

    got = tc.mask_uncertain(status, prediction)
    want = tc.mask_uncertain(prediction.bits, prediction)
    bad = np.argwhere(got != want)
    assert bad.size == 0, (
        f"{case.id}[{mode}]: status differs at {len(bad)} (surface, ray) "
        f"entries; first {bad[:5].tolist()} got {got[tuple(bad[0])]:#04x} "
        f"want {want[tuple(bad[0])]:#04x}"
    )
    got_iters = tc.mask_iters(iters, prediction)
    want_iters = tc.mask_iters(prediction.iters, prediction)
    bad = np.argwhere(got_iters != want_iters)
    assert bad.size == 0, (
        f"{case.id}[{mode}]: iters differs at {len(bad)} entries; "
        f"first {bad[:5].tolist()}"
    )

    counts = tc.bit_counts(status)
    for name in case.bits:
        assert counts[name] > 0, f"{case.id}[{mode}]: expected {name} > 0, got {counts}"
    assert counts["nonuniform_w"] == 0, f"{case.id}[{mode}]: {counts}"


# ---------------------------------------------------------------------------
# 3. External: fused vs NumPy float64 (plan 7.1 R2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_fused_vs_numpy(mps_backend, monkeypatch, case, mode):
    """Plan 7.1's external rule against a NumPy float64 trace of the same optic."""
    metal.set_mode(mode)
    before = dict(T.stats())
    optic_gpu, _, _ = trace_fused(case, monkeypatch)
    assert_was_fused(before, case.id)
    gpu_rows = tc.decoded_rows(optic_gpu.surfaces)
    apertures = [s.aperture for s in optic_gpu.surfaces.surfaces]

    be.set_backend("numpy")
    try:
        optic_np, make = build_case(case)
        out = optic_np.surfaces.trace(make())
        np_rows = tc.decoded_rows(optic_np.surfaces)
        del out
    finally:
        be.set_backend("torch")
        be.set_device("mps")
        be.set_precision("float64")
        be.grad_mode.disable()
        metal.set_mode(mode)

    scale = tc.system_scale(optic_gpu, np_rows)
    opl = max(
        (
            float(np.max(np.abs(r["opd"][np.isfinite(r["opd"])])))
            for r in np_rows
            if "opd" in r and np.isfinite(r["opd"]).any()
        ),
        default=scale,
    )
    rim = [
        tc.rim_band_mask(aperture, row.get("x"), row.get("y"))
        if aperture is not None and "x" in row
        else np.zeros(gpu_rows[0]["x"].shape, dtype=bool)
        for aperture, row in zip(apertures, np_rows, strict=True)
    ]
    tc.assert_vs_numpy(
        gpu_rows,
        np_rows,
        mode=mode,
        scale=scale,
        opl_scale=max(opl, scale),
        rim=rim,
        what=f"{case.id}[{mode}] vs numpy",
    )


# ---------------------------------------------------------------------------
# 4. The named branch tests of plan 7.2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_newton_not_converged_flagged(mps_backend, monkeypatch, mode):
    """``nonconverging_asphere`` with ``max_iter = 1``: the bit and ``iters``."""
    metal.set_mode(mode)
    case = Case("nonconverging", "nonconverging_asphere")
    prediction, status, iters, _ = predicted_and_actual(case, monkeypatch, mode)
    counts = tc.bit_counts(status)
    assert counts["newton_not_converged"] > 0, counts
    newton_rows = np.argwhere(status & trace_layout.ST_NEWTON_NOT_CONVERGED)
    for s, _ in newton_rows[:1]:
        assert int(iters[s].max()) == 1, (
            f"max_iter = 1 must give iters == 1, got {iters[s].max()}"
        )
    assert np.array_equal(iters, prediction.iters)
    assert counts["newton_not_converged"] == prediction.counts["newton_not_converged"]


@pytest.mark.parametrize("mode", MODES)
def test_late_fallback_counted(mps_backend, monkeypatch, mode):
    """``long_path_asphere``: TOL_CROSSOVER, one late fallback, result == R1."""
    metal.set_mode(mode)
    case = Case("long_path", "long_path_asphere")
    _, _, ref, _ = trace_perop(case, monkeypatch)

    before = dict(T.stats())
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    monkeypatch.setenv("OPTILAND_METAL_TRACE_DIAG", "0")
    optic, make = build_case(case)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = optic.surfaces.trace(make())
    got = tc.capture(optic.surfaces, out, mode)
    now = T.stats()

    late = now.get("fused_trace:late_fallback", 0) - before.get(
        "fused_trace:late_fallback", 0
    )
    traces = now.get("fused_trace:traces", 0) - before.get("fused_trace:traces", 0)
    # Predicted, not observed: the seed distance is ~5000 mm and Python's
    # round-off floor 8 * eps * max(1, |t|) overtakes tol = 1e-10 at 3.5e3 mm
    # in df64 and 1.1e5 mm in sf64 (fixture docstring, plan 1.2).  So df64
    # must fall back and sf64 must complete on the kernel.
    if mode == "df64":
        assert late == 1, f"late_fallback moved by {late}, not 1"
        assert traces == 0, f"a late fallback must not count a trace, got {traces}"
    else:
        assert late == 0, f"sf64 clears the crossover; late_fallback moved {late}"
        assert traces == 1, f"sf64 must complete on the kernel, traces {traces}"
    tc.assert_tier_a(got, ref, f"long_path[{mode}]")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", ("plane", "newton"))
def test_backward_propagation(mps_backend, monkeypatch, kind, mode):
    """``t < 0`` is a legal virtual propagation and the OPD decreases."""
    metal.set_mode(mode)
    case = CASES_BY_ID[f"backward_{kind}"]
    _, _, ref, _ = trace_perop(case, monkeypatch)
    before = dict(T.stats())
    optic, _, got = trace_fused(case, monkeypatch)
    assert_was_fused(before, case.id)
    tc.assert_tier_a(got, ref, f"backward_{kind}[{mode}]")

    rows = tc.decoded_rows(optic.surfaces)
    decreased = rows[1]["opd"] < rows[0]["opd"]
    assert decreased.any(), (
        f"backward_{kind}: no ray's OPD decreased at surface 1, so the "
        "negative-t branch was not exercised"
    )


@pytest.mark.parametrize("mode", MODES)
def test_tir_is_nan_with_intensity_unchanged(mps_backend, monkeypatch, mode):
    """TIR: NaN direction cosines, and ``refract`` leaves the intensity alone."""
    metal.set_mode(mode)
    case = CASES_BY_ID["tir"]
    prediction, status, _, rows = predicted_and_actual(case, monkeypatch, mode)
    counts = tc.bit_counts(status)
    assert counts["tir"] == prediction.counts["tir"] > 0, (
        f"tir[{mode}]: {counts} vs {prediction.counts}"
    )
    s = int(np.argwhere(status & trace_layout.ST_TIR)[0][0])
    tir = (status[s] & trace_layout.ST_TIR) != 0
    assert np.isnan(rows[s]["L"][tir]).all(), "a TIR ray must carry NaN L"
    assert np.isfinite(rows[s]["x"][tir]).all(), "a TIR ray still hit the surface"
    # refract (real_rays.py:189-211) has no TIR branch: it never touches the
    # intensity.  Absorption in the same step does, so the exact comparison
    # only applies when the record says the pre-material is transparent.
    optic_ref, _, _, w0 = trace_perop(case, monkeypatch)
    records = tc.compile_tables(optic_ref, mode, w0)
    flags = int(records.surf_int[0, s, trace_layout.SI_FLAGS])
    if flags & trace_layout.FL_ABSORBING:
        assert np.isfinite(rows[s]["intensity"][tir]).all()
        assert (rows[s]["intensity"][tir] > 0.0).all(), (
            "refract must not zero or NaN a TIR ray's intensity"
        )
    else:
        assert (rows[s]["intensity"][tir] == rows[s - 1]["intensity"][tir]).all(), (
            "refract has no TIR branch: the intensity is untouched"
        )


@pytest.mark.parametrize("mode", MODES)
def test_miss_is_nan_and_clips_downstream(mps_backend, monkeypatch, mode):
    """A miss is NaN, and the NaN intensity from the absorbing glass is clipped."""
    metal.set_mode(mode)
    case = CASES_BY_ID["miss"]
    prediction, status, _, rows = predicted_and_actual(case, monkeypatch, mode)
    counts = tc.bit_counts(status)
    assert counts["miss"] == prediction.counts["miss"] > 0, counts
    assert counts["clipped"] > 0, counts
    s = int(np.argwhere(status & trace_layout.ST_MISS)[0][0])
    miss = (status[s] & trace_layout.ST_MISS) != 0
    assert np.isnan(rows[s]["x"][miss]).all()
    # The fixture's measured intensity sequence: 1 after surface 1, NaN after
    # the absorbing glasses (exp of a NaN path length), 0 exactly at the
    # clipping surface (the clip ASSIGNS zero, it does not multiply), NaN again
    # after the next absorbing element.  A kernel that clipped before
    # propagating, or that treated NaN as "inside", produces another sequence.
    clip_rows = np.argwhere(status & trace_layout.ST_CLIPPED)
    assert clip_rows.size > 0, counts
    s_clip = int(clip_rows[0][0])
    assert (rows[s_clip]["intensity"][miss] == 0.0).all(), (
        f"the clip at surface {s_clip} must assign exactly zero"
    )
    assert np.isnan(rows[-1]["intensity"][miss]).all(), (
        "the trailing absorbing element multiplies that zero by exp(NaN)"
    )


@pytest.mark.parametrize("mode", MODES)
def test_reverse_miss(mps_backend, monkeypatch, mode):
    """``reverse_bundle``: the 'both roots non-positive' branch.

    Plan 7.2 predicts ``MISS`` here.  Measured, the branch produces no NaN:
    the bundle enters the Cooke triplet travelling in ``-z``, so neither conic
    root satisfies the ``t > 0`` admissibility rule, ``valid1`` and ``valid2``
    are both false and ``_conic_candidates`` falls back to the
    vertex-nearest root -- which is finite and NEGATIVE.  The observable is
    therefore the negative root, not a NaN, and the oracle predicts the same
    (no ``MISS`` anywhere).  Reported as a plan/fixture divergence rather than
    asserted away: the fixture's own docstring already says
    "implementation-defined per root".
    """
    metal.set_mode(mode)
    case = CASES_BY_ID["reverse"]
    prediction, status, _, rows = predicted_and_actual(case, monkeypatch, mode)
    counts = tc.bit_counts(status)
    assert counts == prediction.counts, counts
    assert counts["miss"] == 0, (
        "the vertex-nearest fallback returns a finite negative root; a NaN "
        f"here would mean the fallback stopped running ({counts})"
    )
    reached = rows[1]["z"] - rows[0]["z"]
    assert np.all(np.isfinite(reached))
    assert np.all(rows[1]["opd"] < rows[0]["opd"]), (
        "every root is negative, so every OPD contribution is negative"
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", ("grazing", "exact_grazing"))
def test_grazing_artefacts_reproduced(mps_backend, monkeypatch, kind, mode):
    """The ``sign(0) == 0`` and floor artefacts are reproduced, not fixed."""
    metal.set_mode(mode)
    case = CASES_BY_ID[kind]
    _, _, ref, _ = trace_perop(case, monkeypatch)
    before = dict(T.stats())
    _, _, got = trace_fused(case, monkeypatch)
    assert_was_fused(before, kind)
    tc.assert_tier_a(got, ref, f"{kind}[{mode}]")

    prediction, status, _, _ = predicted_and_actual(case, monkeypatch, mode)
    counts = tc.bit_counts(status)
    assert counts == prediction.counts, f"{kind}[{mode}]: {counts}"
    if kind == "grazing":
        # The CONTROL (the fixture's own word): |N| = 0.0447 is nine orders
        # above the 1e-14 divisor floor and |dF/dt| thirteen orders above
        # 32 * eps, so a floor test that fires here is measuring the fixture.
        assert counts["nz_floored"] == 0 and counts["df_floored"] == 0, counts
    else:
        # sign(0) == 0: the in-plane rays floor |N| at every one of the three
        # infinite-radius surfaces.
        n_graze = max(1, N_RAYS // 8)
        assert counts["nz_floored"] == 3 * n_graze, counts
        assert counts["df_floored"] == 0, (
            "the in-plane rays sit at r <= 6, where the seed residual "
            "1e-5 r^2 is already under tol = 1e-3, so the kernel's per-thread "
            "loop never evaluates dF/dt for them; "
            "test_df_floored_tangent_rays drives that branch instead"
        )


@pytest.mark.parametrize("mode", MODES)
def test_std_inf_floors_a_grazing_ray(mps_backend, monkeypatch, mode):
    """Plan 7.2's ``planes`` edge case: ``L = 0.999`` and one ray at ``N = 0``.

    ``planes_both_kinds`` ships a (0.2, 0.1) bundle that floors nothing, so
    the edge case gets its own bundle here.  It is launched into the SAME
    fixture optic, whose surface 2 is a live ``StandardGeometry(inf)``: its
    ``-z / max(|N|, 1e-14)`` floors the divisor and returns a finite distance
    where ``Plane``'s bare ``-z / N`` would return NaN.  The first surface is a
    ``Plane``, so the in-plane ray reaches it with ``t = inf`` and is finite
    nowhere after -- which is exactly the difference between the two codes and
    is reproduced, not repaired.
    """
    metal.set_mode(mode)

    def make_bundle(optic: Any, n: int = N_RAYS) -> Any:
        del optic
        lx = 0.999
        nz = np.full(n, np.sqrt(1.0 - lx * lx))
        nz[: n // 8] = 0.0  # exactly in the plane of both surfaces
        px, py = np.zeros(n), np.linspace(-3.0, 3.0, n)
        return fx.make_rays(px, py, np.full(n, -5.0), np.full(n, lx), np.zeros(n), nz)

    def run(fused: str, diag: str) -> tuple[Any, tc.Capture]:
        monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", fused)
        monkeypatch.setenv("OPTILAND_METAL_TRACE_DIAG", diag)
        optic, _ = fx.planes_both_kinds()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = optic.surfaces.trace(make_bundle(optic))
        return optic, tc.capture(optic.surfaces, out, mode)

    _, ref = run("0", "0")
    before = dict(T.stats())
    optic, got = run("1", "1")
    assert_was_fused(before, "planes-grazing")
    tc.assert_tier_a(got, ref, f"planes-grazing[{mode}]")

    planes = trace.diag_from(optic.surfaces)
    assert planes is not None
    status = planes[0].cpu().numpy()[0]
    counts = tc.bit_counts(status)
    assert counts["nz_floored"] == N_RAYS // 8, (
        f"exactly the {N_RAYS // 8} in-plane rays must floor |N| at the "
        f"StandardGeometry(inf) surface, got {counts}"
    )


@pytest.mark.parametrize("mode", MODES)
def test_df_floored_tangent_rays(mps_backend, monkeypatch, mode):
    """``ST_DF_FLOORED``: an UNCONVERGED ray whose ``dF/dt`` is exactly zero.

    No WP5 fixture reaches this bit (measured: ``exact_grazing_bundle`` sets
    ``NZ_FLOORED`` on 3 x N/8 entries and ``DF_FLOORED`` on none), because its
    in-plane rays sit at ``r <= 6`` where the seed residual ``1e-5 r^2`` is
    already under ``tol = 1e-3``: a converged thread breaks out of the Newton
    loop before any ``dF/dt`` is evaluated.  Python's batch loop *does*
    evaluate and floor ``dF/dt`` for them, but it freezes their step, so the
    distances still agree -- the difference is in the diagnostic bit only.

    Moving the in-plane rays out to ``y > 10.1`` makes them unconverged at
    their seed while ``dF/dt = f_x L + f_y M - N`` stays exactly zero (they sit
    on the y axis, so ``f_x = 0``, and ``M = N = 0``), which is the only way
    into ``_regularize_signed``'s floored branch.
    """
    metal.set_mode(mode)
    n_graze = N_RAYS // 8

    def make_bundle(n: int = N_RAYS) -> Any:
        rest = n - n_graze
        y_graze = 10.5 + 1.5 * (np.arange(n_graze) + 0.5) / n_graze
        theta = np.arange(rest) * (np.pi * (3.0 - np.sqrt(5.0)))
        return fx.make_rays(
            np.r_[np.zeros(n_graze), 12.0 * np.cos(theta)],
            np.r_[y_graze, 12.0 * np.sin(theta)],
            np.r_[np.zeros(n_graze), np.full(rest, -10.0)],
            np.r_[np.ones(n_graze), np.zeros(rest)],
            np.zeros(n),
            np.r_[np.zeros(n_graze), np.ones(rest)],
        )

    def run(fused: str, diag: str) -> tuple[Any, tc.Capture]:
        monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", fused)
        monkeypatch.setenv("OPTILAND_METAL_TRACE_DIAG", diag)
        optic = fx._flat_newton_system()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = optic.surfaces.trace(make_bundle())
        return optic, tc.capture(optic.surfaces, out, mode)

    optic_ref, ref = run("0", "0")
    before = dict(T.stats())
    optic, got = run("1", "1")
    assert_was_fused(before, "df-floored")
    tc.assert_tier_a(got, ref, f"df_floored[{mode}]")

    planes = trace.diag_from(optic.surfaces)
    assert planes is not None
    status = planes[0].cpu().numpy()[0]
    iters = planes[1].cpu().numpy()[0]
    counts = tc.bit_counts(status)
    assert counts["df_floored"] == n_graze, counts
    assert counts["newton_not_converged"] == n_graze, counts

    rows = tc.decoded_rows(optic_ref.surfaces)
    records = tc.compile_tables(optic_ref, mode, 0.5876)
    prediction = tc.predict_status(rows, optic_ref, mode=mode, records=records)
    assert np.array_equal(prediction.bits, status)
    assert np.array_equal(prediction.iters, iters)


#: ``rim_bundle``'s two apertures, by the surface row that carries them.
RIM_APERTURES: tuple[tuple[str, int], ...] = (("rect", 1), ("annulus", 2))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("aperture", RIM_APERTURES, ids=lambda a: a[0])
def test_rim_inclusive(mps_backend, monkeypatch, aperture, mode):
    """Rays exactly on an aperture edge are INSIDE, in both modes.

    ``rim_bundle`` places 16 probe rays on exact power-of-two coordinates at
    normal incidence, so ``x*x + y*y`` is exact in both representations and the
    inclusive ``<=`` / ``>=`` bounds are decided by arithmetic rather than by
    rounding: this assertion uses NO rim band at all (plan 7.1), in either
    mode.  Surface 1 carries ``RectangularAperture(-10, 10, -5, 5)`` and
    surface 2 ``RadialAperture(r_max = 8, r_min = 2)``; each row's intensity is
    predicted from the fixture's own inside/outside column, and surface 2's
    prediction is the AND of the two (surface 1's clip already zeroed the
    intensity, and the clip assigns rather than multiplies).
    """
    metal.set_mode(mode)
    kind, row = aperture
    case = CASES_BY_ID["rim"]
    _, _, ref, _ = trace_perop(case, monkeypatch)
    before = dict(T.stats())
    optic, _, got = trace_fused(case, monkeypatch)
    assert_was_fused(before, "rim")
    tc.assert_tier_a(got, ref, f"rim[{mode}]")

    rows = tc.decoded_rows(optic.surfaces)
    for index, probe in enumerate(fx._RIM_PROBES):
        x, y, inside_rect, inside_annulus = probe
        del x, y
        want = (
            bool(inside_rect)
            if kind == "rect"
            else bool(inside_rect and inside_annulus)
        )
        intensity = float(rows[row]["intensity"][index])
        assert (intensity != 0.0) == want, (
            f"rim probe {index} {probe} at the {kind} aperture: intensity "
            f"{intensity}, expected {'inside' if want else 'clipped'}"
        )


@pytest.mark.parametrize("mode", MODES)
def test_aperture_root_preference_observable(mps_backend, monkeypatch, mode):
    """The aperture reorders the two conic roots, and the far root is taken."""
    metal.set_mode(mode)
    case = CASES_BY_ID["oap"]
    _, _, ref, _ = trace_perop(case, monkeypatch)
    before = dict(T.stats())
    optic, _, got = trace_fused(case, monkeypatch)
    assert_was_fused(before, "oap")
    tc.assert_tier_a(got, ref, f"oap[{mode}]")

    # The preference is only observable if some ray lands on the root the
    # vertex rule would NOT have chosen: compare against the same system with
    # the aperture removed from root selection.
    rows = tc.decoded_rows(optic.surfaces)
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    plain, make = build_case(case)
    for surface in plain.surfaces.surfaces:
        surface.aperture = None
    out = plain.surfaces.trace(make())
    del out
    plain_rows = tc.decoded_rows(plain.surfaces)
    differ = ~np.isclose(
        rows[1]["z"], plain_rows[1]["z"], rtol=0.0, atol=1e-9, equal_nan=True
    )
    assert differ.any(), (
        "off_axis_parabola_far_root: removing the aperture changed no "
        "intersection, so the root preference is not observable here"
    )


@pytest.mark.parametrize("mode", MODES)
def test_trailing_propagate_after_fused(mps_backend, monkeypatch, mode):
    """A non-zero image thickness still propagates after the fused trace."""
    metal.set_mode(mode)
    case = CASES_BY_ID["nonzero_image_thickness"]
    _, _, ref, _ = trace_perop(case, monkeypatch)
    before = dict(T.stats())
    optic, out, got = trace_fused(case, monkeypatch)
    assert_was_fused(before, "nonzero_image_thickness")
    tc.assert_tier_a(got, ref, f"nonzero_image_thickness[{mode}]")

    rows = tc.decoded_rows(optic.surfaces)
    z_image = rows[-1]["z"]
    z_prev = rows[-2]["z"]
    assert np.all(np.abs(z_image - z_prev) > 0.0), (
        "the image surface sits at a non-zero thickness from the one before it"
    )
    assert np.array_equal(tc.decode(out.z), z_image, equal_nan=True), (
        "the returned bundle sits on the image surface"
    )


@pytest.mark.parametrize("mode", MODES)
def test_mixed_wavelength_refused_pre_launch(mps_backend, monkeypatch, mode):
    """A two-wavelength bundle is refused BEFORE any launch, with the reason."""
    metal.set_mode(mode)
    case = Case("mixed_wavelength", "mixed_wavelength_bundle")
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    optic_ref, make_ref = build_case(case)
    ref_out = optic_ref.surfaces.trace(make_ref())
    ref = tc.capture(optic_ref.surfaces, ref_out, mode)

    before = dict(T.stats())
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    optic, make = build_case(case)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = optic.surfaces.trace(make())
    got = tc.capture(optic.surfaces, out, mode)
    now = T.stats()

    key = f"fused_trace_skip:{FusedTraceSkip.MIXED_WAVELENGTH.value}"
    assert now.get(key, 0) - before.get(key, 0) == 1, (
        f"expected exactly one {key}; deltas "
        f"{ {k: now[k] - before.get(k, 0) for k in now if 'fused' in k} }"
    )
    assert now.get("gpu:fused_trace", 0) == before.get("gpu:fused_trace", 0), (
        "the refusal must happen before any launch"
    )
    assert now.get("fused_trace:traces", 0) == before.get("fused_trace:traces", 0)
    tc.assert_tier_a(got, ref, f"mixed_wavelength[{mode}] fallback")


@pytest.mark.parametrize("mode", MODES)
def test_tier_b_small_bundle(mps_backend, monkeypatch, mode):
    """Tier-B site 1 (plan 7.1): ``256 < N <= 1024`` on the Cooke triplet.

    This is the single comparison in the file that is not raw-component
    equality, and the bound is derived from ``MACHINE_EPS[mode]`` and the
    system scale, never from what the run happens to produce.  In df64 the two
    paths genuinely differ below ``_MAX_VALUE_KEY_ARRAY_SIZE`` (the per-op path
    evaluates the dispersion on the whole 300-element wavelength array, the
    kernel on the one-element view of plan 3.8), which is exactly why site 1
    exists; in sf64 they coincide.
    """
    metal.set_mode(mode)
    case = Case("cooke-tierb", "cooke", field=(0.0, 0.7), rays=fx.TIER_B_RAYS)
    assert tc.TIER_B_MIN_RAYS < case.rays <= tc.TIER_B_MAX_RAYS
    optic_ref, _, _, _ = trace_perop(case, monkeypatch)
    ref_rows = tc.decoded_rows(optic_ref.surfaces)
    before = dict(T.stats())
    optic, _, _ = trace_fused(case, monkeypatch)
    assert_was_fused(before, "cooke-tierb")
    got_rows = tc.decoded_rows(optic.surfaces)
    scale = tc.system_scale(optic, ref_rows)
    tc.assert_tier_b(
        got_rows, ref_rows, mode=mode, scale=scale, what=f"cooke-tierb[{mode}]"
    )


# ---------------------------------------------------------------------------
# 5. Counters (plan 7.2)
# ---------------------------------------------------------------------------


COUNTER_CASES: tuple[Case, ...] = tuple(
    CASES_BY_ID[i] for i in ("cooke", "hubble", "aspheric_singlet", "uv", "planes")
)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", COUNTER_CASES, ids=lambda c: c.id)
def test_predicted_counters(mps_backend, monkeypatch, case, mode):
    """Every counter is predicted from an independent census (plan 0.2.6, 8.3).

    The predictions:

    * ``gpu:fused_trace`` equals ``len(_slab_plan(...))`` -- computed by
      calling the driver's own planner, never tabulated;
    * ``fused_trace:surface_steps`` equals ``B * N * (S - 1)`` with ``S`` read
      off the surface list;
    * ``fused_trace:candidates == fused_trace:traces == 1``;
    * ``gpu:conic_candidates`` is ABSENT on the fused path and, on the ``=0``
      reference run of the same system, equals the number of finite-radius
      conic faces counted from the surface list.
    """
    metal.set_mode(mode)

    # --- the reference run: the per-op path reaches be.conic_intersection once
    # per finite-radius StandardGeometry face.
    T.reset_stats()
    optic_ref, _, _, w0 = trace_perop(case, monkeypatch)
    ref_stats = dict(T.stats())
    # Independent census: ``be.conic_intersection`` is reached once per
    # surface whose distance goes through ``StandardGeometry``'s FINITE-radius
    # branch -- which includes every Newton geometry, because its seed is
    # ``super().distance()`` (newton_raphson.py:347).  ``Plane`` has its own
    # bare ``-z / N`` and the infinite-radius branch never builds candidates.
    from optiland.geometries import Plane as _Plane
    from optiland.geometries import StandardGeometry as _Std

    surfaces = list(optic_ref.surfaces.surfaces)
    expected_conic = sum(
        1
        for s in surfaces[1:]
        if isinstance(s.geometry, _Std)
        and not isinstance(s.geometry, _Plane)
        and np.isfinite(float(be.to_numpy(s.geometry.radius)))
    )
    assert ref_stats.get("gpu:conic_candidates", 0) == expected_conic, (
        f"{case.id}[{mode}]: the per-op path launched "
        f"{ref_stats.get('gpu:conic_candidates', 0)} conic-candidate kernels, "
        f"expected {expected_conic} (one per finite-radius conic face)"
    )
    assert not any(k.startswith("fused_trace") for k in ref_stats), (
        f"{case.id}[{mode}]: the =0 run moved a fused counter: {ref_stats}"
    )

    # --- the fused run
    T.reset_stats()
    trace.reset_driver_state()
    optic, _, _ = trace_fused(case, monkeypatch)
    stats = dict(T.stats())

    s_count = len(list(optic.surfaces.surfaces))
    n = case.rays
    records = tc.compile_tables(optic_ref, mode, w0)
    plan = trace._slab_plan(
        1, n, int(records.weighted_steps), trace._max_steps(), 1 << 20
    )

    assert stats.get("gpu:fused_trace") == len(plan), (
        f"{case.id}[{mode}]: {stats.get('gpu:fused_trace')} launches vs "
        f"{len(plan)} planned slabs"
    )
    assert stats.get("fused_trace:chunks") == len(plan)
    assert stats.get("fused_trace:candidates") == 1
    assert stats.get("fused_trace:traces") == 1
    assert stats.get("fused_trace:designs") == 1
    assert stats.get("fused_trace:surface_steps") == 1 * n * (s_count - 1), (
        f"{case.id}[{mode}]: surface_steps "
        f"{stats.get('fused_trace:surface_steps')} vs {n * (s_count - 1)}"
    )
    assert "gpu:conic_candidates" not in stats, (
        f"{case.id}[{mode}]: the fused path must not reach be.conic_intersection"
    )
    assert not any(k.startswith("fused_trace_skip:") for k in stats), (
        f"{case.id}[{mode}]: unexpected refusals {stats}"
    )


# ---------------------------------------------------------------------------
# 6. predict_status on hand-built records [fix: L2.16]
# ---------------------------------------------------------------------------


@dataclass
class HandOptic:
    """The two attributes :func:`predict_status` reads off an optic."""

    surfaces: Any


@dataclass
class HandGroup:
    """A surface group of hand-built surfaces."""

    surfaces: list[Any] = field(default_factory=list)


@dataclass
class HandSurface:
    """A surface with only an ``aperture`` (the rim band's one input)."""

    aperture: Any = None


@dataclass
class HandRecords:
    """The three tables :func:`predict_status` reads."""

    surf_int: np.ndarray
    surf_real: np.ndarray
    coef: np.ndarray


def hand_tables(s: int, c: int = 1) -> HandRecords:
    """Zeroed tables for ``s`` surfaces, with a usable identity pose."""
    surf_int = np.zeros((1, s, trace_layout.SI_STRIDE), dtype=np.int32)
    surf_real = np.zeros((1, s, trace_layout.SR_STRIDE), dtype=np.float64)
    coef = np.zeros((1, s, c), dtype=np.float64)
    surf_int[0, 0, trace_layout.SI_GEOM] = trace_layout.GEOM_OBJECT
    surf_real[0, :, trace_layout.SR_TOL] = 1e-10
    surf_real[0, :, trace_layout.SR_U] = 1.0
    surf_real[0, :, trace_layout.SR_U2] = 1.0
    return HandRecords(surf_int, surf_real, coef)


def hand_rows(n: int, s: int) -> list[dict[str, np.ndarray]]:
    """``s`` rows of ``n`` rays, all at the origin travelling along +z."""
    rows = []
    for _ in range(s):
        rows.append(
            {
                "x": np.zeros(n),
                "y": np.zeros(n),
                "z": np.zeros(n),
                "L": np.zeros(n),
                "M": np.zeros(n),
                "N": np.ones(n),
                "intensity": np.ones(n),
                "opd": np.zeros(n),
            }
        )
    return rows


def hand_predict(rows, records, *, apertures=None, mode="df64"):
    """``predict_status`` on hand-built inputs."""
    s = len(rows)
    group = HandGroup([HandSurface(None) for _ in range(s)])
    if apertures is not None:
        for index, aperture in apertures.items():
            group.surfaces[index].aperture = aperture
    return tc.predict_status(
        rows, HandOptic(group), mode=mode, records=records, design=0
    )


def test_predict_status_plane_is_quiet():
    """A plane at normal incidence sets no bit and takes no iteration."""
    rows = hand_rows(4, 2)
    rows[0]["z"] = np.full(4, -5.0)
    tables = hand_tables(2)
    tables.surf_int[0, 1, trace_layout.SI_GEOM] = trace_layout.GEOM_PLANE
    prediction = hand_predict(rows, tables)
    assert prediction.bits.shape == (2, 4)
    assert not prediction.bits.any(), prediction.counts
    assert not prediction.iters.any()


def test_predict_status_plane_miss_on_zero_n():
    """``Plane.distance`` is a bare ``-z / N``, with NO floor (plane.py:84-86).

    So ``N = 0`` is only a MISS when ``z`` is zero too (``0 / 0``); a ray
    travelling *in* the plane of a surface it has not reached yet gets
    ``t = inf``, which is not NaN and therefore not a miss.  The distinction
    matters: it is the whole reason ``GEOM_PLANE`` and ``GEOM_STD_INF`` are two
    codes (see the next test).
    """
    rows = hand_rows(3, 2)
    rows[0]["z"] = np.array([-5.0, 0.0, -5.0])
    rows[0]["N"] = np.array([1.0, 0.0, 0.0])
    tables = hand_tables(2)
    tables.surf_int[0, 1, trace_layout.SI_GEOM] = trace_layout.GEOM_PLANE
    prediction = hand_predict(rows, tables)
    assert prediction.counts["miss"] == 1
    assert (prediction.bits[1] & trace_layout.ST_MISS != 0).tolist() == [
        False,
        True,
        False,
    ]


def test_predict_status_std_inf_floors_n_instead_of_missing():
    """``StandardGeometry(inf)`` floors ``|N|`` at 1e-14 -- NZ_FLOORED, no MISS.

    This is the difference that makes ``GEOM_PLANE`` and ``GEOM_STD_INF`` two
    codes rather than one (WP5 finding 1), and it is exactly what
    ``planes_both_kinds`` exercises end to end.
    """
    rows = hand_rows(3, 2)
    rows[0]["z"] = np.full(3, -5.0)
    rows[0]["N"] = np.array([1.0, 0.0, 1e-20])
    tables = hand_tables(2)
    tables.surf_int[0, 1, trace_layout.SI_GEOM] = trace_layout.GEOM_STD_INF
    prediction = hand_predict(rows, tables)
    assert prediction.counts["miss"] == 0
    assert prediction.counts["nz_floored"] == 2
    assert (prediction.bits[1] & trace_layout.ST_NZ_FLOORED != 0).tolist() == [
        False,
        True,
        True,
    ]


def test_predict_status_clipped_is_inclusive():
    """``contains`` is inclusive on both bounds, and NaN is outside."""
    from optiland.physical_apertures.radial import RadialAperture

    n = 4
    rows = hand_rows(n, 2)
    rows[0]["z"] = np.full(n, -5.0)
    rows[0]["x"] = np.array([0.0, 2.0, 3.0, np.nan])
    tables = hand_tables(2)
    tables.surf_int[0, 1, trace_layout.SI_GEOM] = trace_layout.GEOM_PLANE
    tables.surf_int[0, 1, trace_layout.SI_FLAGS] = trace_layout.FL_HAS_APERTURE
    tables.surf_int[0, 1, trace_layout.SI_APCODE] = trace_layout.AP_RADIAL
    tables.surf_real[0, 1, trace_layout.SR_AP0] = 4.0  # r_max**2
    tables.surf_real[0, 1, trace_layout.SR_AP1] = 0.0  # r_min**2
    prediction = hand_predict(
        rows, tables, apertures={1: RadialAperture(r_max=2.0, r_min=0.0)}
    )
    clipped = (prediction.bits[1] & trace_layout.ST_CLIPPED) != 0
    assert clipped.tolist() == [False, False, True, True], (
        "r = r_max is INSIDE; r > r_max and a NaN position are outside"
    )


def test_predict_status_tir_from_the_radicand():
    """TIR is a NaN ``sqrt`` in ``refract``, never a separate branch."""
    n = 3
    rows = hand_rows(n, 2)
    rows[0]["z"] = np.full(n, -5.0)
    # 30 / 60 / 80 degrees onto a plane whose normal is +z.
    angles = np.deg2rad([5.0, 60.0, 80.0])
    rows[0]["L"] = np.sin(angles)
    rows[0]["N"] = np.cos(angles)
    tables = hand_tables(2)
    tables.surf_int[0, 1, trace_layout.SI_GEOM] = trace_layout.GEOM_PLANE
    tables.surf_real[0, 1, trace_layout.SR_U] = 1.5
    tables.surf_real[0, 1, trace_layout.SR_U2] = 2.25
    prediction = hand_predict(rows, tables)
    tir = (prediction.bits[1] & trace_layout.ST_TIR) != 0
    # critical angle asin(1 / 1.5) = 41.8 deg
    assert tir.tolist() == [False, True, True]


def test_predict_status_newton_iterations_are_per_ray():
    """``iters`` is the ray's OWN count, not the batch's single integer.

    The Python loop freezes a converged ray (``be.where(converged, 0, step)``)
    and only breaks when every ray has converged, so a ray that converges on
    step 1 must be predicted as ``iters == 1`` even while its neighbours keep
    stepping (design 4.12, WP5 finding 4).
    """
    n = 2
    rows = hand_rows(n, 2)
    rows[0]["z"] = np.full(n, -5.0)
    rows[0]["x"] = np.array([0.0, 1.5])
    tables = hand_tables(2, c=1)
    tables.surf_int[0, 1, trace_layout.SI_GEOM] = trace_layout.GEOM_EVEN
    tables.surf_int[0, 1, trace_layout.SI_NCOEFF] = 1
    tables.surf_int[0, 1, trace_layout.SI_MAXITER] = 100
    tables.surf_real[0, 1, trace_layout.SR_R] = 50.0
    tables.surf_real[0, 1, trace_layout.SR_K] = 0.0
    tables.surf_real[0, 1, trace_layout.SR_K1] = 1.0
    tables.surf_real[0, 1, trace_layout.SR_R2] = 2500.0
    tables.coef[0, 1, 0] = 1e-4
    prediction = hand_predict(rows, tables)
    assert prediction.counts["newton_not_converged"] == 0, prediction.counts
    assert int(prediction.iters[1, 0]) == 0, (
        "the axial ray's conic seed is already the root: no Newton step"
    )
    assert int(prediction.iters[1, 1]) >= 1, (
        "the off-axis ray must take at least one step"
    )


def test_predict_status_newton_not_converged_with_one_iteration():
    """``max_iter = 1`` on a strongly aspheric surface: the bit and ``iters``."""
    n = 2
    rows = hand_rows(n, 2)
    rows[0]["z"] = np.full(n, -10.0)
    rows[0]["x"] = np.array([6.0, 8.0])
    # Tilted, like ``nonconverging_asphere``: at normal incidence x and y do
    # not depend on t, the residual is linear and one Newton step is exact.
    rows[0]["L"] = np.full(n, 0.3)
    rows[0]["N"] = np.full(n, np.sqrt(1.0 - 0.09))
    tables = hand_tables(2, c=1)
    tables.surf_int[0, 1, trace_layout.SI_GEOM] = trace_layout.GEOM_EVEN
    tables.surf_int[0, 1, trace_layout.SI_NCOEFF] = 1
    tables.surf_int[0, 1, trace_layout.SI_MAXITER] = 1
    tables.surf_real[0, 1, trace_layout.SR_R] = 20.0
    tables.surf_real[0, 1, trace_layout.SR_K1] = 1.0
    tables.surf_real[0, 1, trace_layout.SR_R2] = 400.0
    tables.coef[0, 1, 0] = -1.0e-3
    prediction = hand_predict(rows, tables)
    assert prediction.counts["newton_not_converged"] == n, prediction.counts
    assert prediction.iters[1].tolist() == [1, 1]


def test_predict_status_tol_crossover():
    """A seed past the crossover raises ``tol`` above the user's and flags it."""
    n = 2
    rows = hand_rows(n, 2)
    rows[0]["z"] = np.array([-5.0, -1e4])
    tables = hand_tables(2, c=1)
    tables.surf_int[0, 1, trace_layout.SI_GEOM] = trace_layout.GEOM_EVEN
    tables.surf_int[0, 1, trace_layout.SI_NCOEFF] = 1
    tables.surf_int[0, 1, trace_layout.SI_MAXITER] = 100
    tables.surf_real[0, 1, trace_layout.SR_R] = 1e6
    tables.surf_real[0, 1, trace_layout.SR_K1] = 1.0
    tables.surf_real[0, 1, trace_layout.SR_R2] = 1e12
    tables.coef[0, 1, 0] = 0.0
    prediction = hand_predict(rows, tables, mode="df64")
    crossed = (prediction.bits[1] & trace_layout.ST_TOL_CROSSOVER) != 0
    assert crossed.tolist() == [False, True], (
        "the df64 floor 8 * 2**-48 * 1e4 = 2.8e-10 beats tol = 1e-10"
    )
    prediction = hand_predict(rows, tables, mode="sf64")
    crossed = (prediction.bits[1] & trace_layout.ST_TOL_CROSSOVER) != 0
    assert crossed.tolist() == [False, False], (
        "the sf64 floor 8 * 2**-53 * 1e4 = 8.9e-12 is still under tol = 1e-10"
    )


def test_predict_status_first_event_rows_is_the_plan_sketch():
    """``first_event_rows`` is the first-occurrence view, not the bit plane."""
    n = 3
    rows = hand_rows(n, 4)
    for s in (2, 3):
        rows[s]["x"] = np.array([np.nan, 0.0, 0.0])
        rows[s]["y"] = np.array([np.nan, 0.0, 0.0])
    events = tc.first_event_rows(rows)
    assert events["miss"].tolist() == [2, -1, -1]
    assert events["tir"].tolist() == [-1, -1, -1]


def test_predict_status_counts_match_the_bit_plane():
    """``counts`` is exactly ``bit_counts(bits)``, so neither can drift."""
    rows = hand_rows(5, 2)
    rows[0]["z"] = np.array([-5.0, 0.0, 0.0, -5.0, 0.0])
    rows[0]["N"] = np.array([1.0, 0.0, 0.0, 1.0, 0.0])
    tables = hand_tables(2)
    tables.surf_int[0, 1, trace_layout.SI_GEOM] = trace_layout.GEOM_PLANE
    prediction = hand_predict(rows, tables)
    assert prediction.counts == tc.bit_counts(prediction.bits)
    assert prediction.counts["miss"] == 3


# ---------------------------------------------------------------------------
# 7. Round 0: divergence injection with sensitivity certificates (plan 6)
# ---------------------------------------------------------------------------
#
# Every conformance assertion above says "the fused trace equals the per-op
# trace".  That statement is worth exactly as much as the comparison's ability
# to FAIL.  Round 0 proves that ability site by site: break one mirrored
# expression, show first that the break is observable at all (the
# **certificate**), then show that the tier-A comparison notices.  A site whose
# certificate cannot be produced is removed rather than kept as decoration
# (plan section 6).
#
# Two kinds of site, with two different certificates:
#
# * **Python sites** patch the per-op path -- the reference R1 -- while the
#   kernel keeps mirroring the unpatched physics.  The certificate is plan
#   section 6's own: the fixture traced twice on mps per-op, with and without
#   the injected change, must differ in at least one recorded value.  The
#   injected function is also a MIRRORED fingerprint row, so the injection
#   additionally makes `trace_mirror.check_all()` name it -- asserted here,
#   because it is the reason the fused half of the test has to set the
#   developer drift hatch: under the default policy the driver would refuse
#   every candidate with `mirror_drift` and the comparison would pass by
#   falling back rather than by agreeing.
# * **MSL sites** patch the kernel through the `#ifdef OPTILAND_TRACE_BREAK_*`
#   hooks WP1 compiled in (`tp.BREAK_SITES`), by prepending the `#define` to
#   the driver's own `trace_source`.  Nothing in Python changes, so no
#   fingerprint moves -- which is precisely why round 0 exists on this side:
#   `check_all()` hashes Python sources and can never see a kernel edit.  The
#   certificate is that the broken library's trace differs from the clean
#   one's, and it is asserted in BOTH directions per mode: a site that
#   `tp.BREAK_SITES` says is invisible in a mode must leave that mode's trace
#   untouched, and the tier-A comparison must still pass there.


@dataclass(frozen=True)
class Site:
    """One round-0 divergence-injection site (plan section 6's table).

    Attributes:
        id: The parametrization id.
        kind: ``"python"`` (patches the per-op path) or ``"msl"`` (patches
            the kernel through a ``BREAK_`` hook).
        build: ``() -> (optic, make_rays)`` on the active backend.
        quantity: The recorded quantity the certificate must move, as
            ``_trace_compare`` names it (``"L"``, ``"intensity"``, ...).
        modes: The modes in which the injection must be visible.  For an MSL
            site this must equal ``tp.BREAK_SITES[site]``.
        mirror: For a Python site, the fingerprint qualname the injection
            drifts; None for an MSL site.
        inject: Applies the injection through the test's monkeypatch.
        break_site: For an MSL site, the ``tp.BREAK_SITES`` key.
    """

    id: str
    kind: str
    build: Callable[[], tuple[Any, Callable[[], Any]]]
    quantity: str
    modes: tuple[str, ...]
    mirror: str | None = None
    inject: Callable[[Any], None] | None = None
    break_site: str | None = None


def _rebuilt_with(func: Any, subs: tuple[tuple[str, str, int], ...]) -> Any:
    """``func`` recompiled from its own source with ``subs`` applied.

    Each substitution declares how many occurrences it expects, so an
    upstream edit that moves the expression the injection targets fails the
    test loudly instead of injecting a stale body (or nothing at all).  The
    copy is compiled in a copy of the original module's globals, with
    ``from __future__ import annotations`` prepended because the sources
    being copied carry annotations that are only importable under
    ``TYPE_CHECKING``.
    """
    src = textwrap.dedent(inspect.getsource(func))
    for old, new, count in subs:
        found = src.count(old)
        assert found == count, (
            f"{func.__qualname__}: expected {count} occurrence(s) of {old!r}, "
            f"found {found}; the injection target moved upstream"
        )
        src = src.replace(old, new)
    namespace = dict(func.__globals__)
    exec(  # noqa: S102 - a copy of the function under test, by design
        compile("from __future__ import annotations\n" + src, "<injected>", "exec"),
        namespace,
    )
    return namespace[func.__name__]


def _inject_refraction(monkeypatch) -> None:
    """``RealRays.refract`` with ``- u * n * dot`` flipped to ``+``."""
    from optiland.rays.real_rays import RealRays

    monkeypatch.setattr(
        RealRays,
        "refract",
        _rebuilt_with(
            RealRays.refract,
            (
                ("- u * nx * dot", "+ u * nx * dot", 1),
                ("- u * ny * dot", "+ u * ny * dot", 1),
                ("- u * nz * dot", "+ u * nz * dot", 1),
            ),
        ),
    )


def _inject_rotation(monkeypatch) -> None:
    """``RealRays.rotate_x`` with the opposite sign (``sin(rx) -> sin(-rx)``).

    Only the sine terms change, which is exactly the transposed rotation --
    the classic localize/globalize sign slip.  ``tilted_triplet("rx")``
    carries one rx pose, so both the localize (``-rx``) and the globalize
    (``+rx``) call are affected and the surface ends up tilted the other way.
    """
    from optiland.rays.real_rays import RealRays

    monkeypatch.setattr(
        RealRays,
        "rotate_x",
        _rebuilt_with(RealRays.rotate_x, (("be.sin(rx)", "be.sin(-rx)", 4),)),
    )


def _inject_aperture(monkeypatch) -> None:
    """``RectangularAperture.contains`` with ``<=`` made exclusive."""
    from optiland.physical_apertures.rectangular import RectangularAperture

    monkeypatch.setattr(
        RectangularAperture,
        "contains",
        _rebuilt_with(RectangularAperture.contains, (("<=", "<", 4),)),
    )


def _inject_absorption(monkeypatch) -> None:
    """``HomogeneousPropagation.propagate`` without the mm -> um ``1e3``."""
    from optiland.propagation.homogeneous import HomogeneousPropagation

    monkeypatch.setattr(
        HomogeneousPropagation,
        "propagate",
        _rebuilt_with(
            HomogeneousPropagation.propagate,
            (("-alpha * t * 1e3", "-alpha * t", 1),),
        ),
    )


def _inject_newton_floor(monkeypatch) -> None:
    """``newton_raphson._CONV_EPS_MULTIPLIER = 1e6``: the tolerance floor."""
    from optiland.geometries import newton_raphson

    monkeypatch.setattr(newton_raphson, "_CONV_EPS_MULTIPLIER", 1e6)


#: The Newton-floor site's system: ``long_path_asphere`` with the distance to
#: the Newton surface cut from 5000 mm to 2000 mm.
#:
#: **Deviation from plan section 6's table, measured.**  The table names
#: ``aspheric_singlet`` for this site and asks for ``iters`` and ``t`` to
#: move.  They do not: measured at 4096 rays in both modes, raising
#: ``_CONV_EPS_MULTIPLIER`` to 1e6 changes nothing on ``aspheric_singlet``,
#: ``even_asphere_5coeff``, ``even_asphere_inf_radius``,
#: ``odd_asphere_singlet``, ``backward_newton`` or ``nonconverging_asphere``
#: -- every one of them drives its residual from above ``tol`` to ~0 in a
#: single step, so the iteration it exits on is the same whether the
#: convergence test is 1e-10 or the injected floor.  The floor can only bite
#: where ``|t|`` is large enough to lift ``1e6 * eps * |t|`` above ``tol``,
#: and the only WP5 fixture with such a distance is ``long_path_asphere`` --
#: whose 5000 mm seed is ALSO past the df64 round-off crossover, so in df64
#: the driver takes the late fallback and the kernel never runs (WP6-a's
#: ``test_late_fallback_counted``).  At 2000 mm both conditions hold in both
#: modes: the nominal floor ``8 * eps * |t|`` stays under ``tol = 1e-10``
#: (no ``TOL_CROSSOVER``, no late fallback, measured ``traces == 1``) while
#: the injected floor ``1e6 * eps * |t|`` -- 7.1e-6 in df64, 2.2e-7 in sf64 --
#: is far above it.  ``long_path_batch_values`` already uses 2000 mm as its
#: "safely inside the crossover" design, for the same reason.
NEWTON_FLOOR_THICKNESS = 2000.0


def _build_newton_floor_site() -> tuple[Any, Callable[[], Any]]:
    """``long_path_asphere`` with surface 1's thickness set to 2000 mm."""
    optic, factory = fx.long_path_asphere()
    optic.updater.set_thickness(NEWTON_FLOOR_THICKNESS, 1)

    def make() -> Any:
        return factory(optic, N_RAYS)

    return optic, make


def _case_builder(case_id: str) -> Callable[[], tuple[Any, Callable[[], Any]]]:
    """``build_case`` bound to one of the feature-matrix cases."""

    def build() -> tuple[Any, Callable[[], Any]]:
        return build_case(CASES_BY_ID[case_id])

    return build


def _fixture_builder(
    name: str, field: tuple[float, float] | None = None
) -> Callable[[], tuple[Any, Callable[[], Any]]]:
    """``build_case`` on a fixture that has no feature-matrix row."""

    def build() -> tuple[Any, Callable[[], Any]]:
        return build_case(Case(name, name, field=field))

    return build


#: Plan section 6's round-0 table, one row per site.
SITES: tuple[Site, ...] = (
    Site(
        id="newton_tolerance_floor",
        kind="python",
        build=_build_newton_floor_site,
        quantity="opd",
        modes=MODES,
        mirror="optiland.geometries.newton_raphson:_CONV_EPS_MULTIPLIER",
        inject=_inject_newton_floor,
    ),
    Site(
        id="refraction",
        kind="python",
        build=_case_builder("cooke-f0,0.7"),
        quantity="L",
        modes=MODES,
        mirror="optiland.rays.real_rays:RealRays.refract",
        inject=_inject_refraction,
    ),
    Site(
        id="rotation",
        kind="python",
        build=_case_builder("tilt-rx"),
        quantity="x",
        modes=MODES,
        mirror="optiland.rays.real_rays:RealRays.rotate_x",
        inject=_inject_rotation,
    ),
    Site(
        id="aperture_inclusivity",
        kind="python",
        build=_case_builder("rim"),
        quantity="intensity",
        modes=MODES,
        mirror="optiland.physical_apertures.rectangular:RectangularAperture.contains",
        inject=_inject_aperture,
    ),
    Site(
        id="absorption",
        kind="python",
        build=_case_builder("cooke-f0,0.7"),
        quantity="intensity",
        modes=MODES,
        mirror="optiland.propagation.homogeneous:HomogeneousPropagation.propagate",
        inject=_inject_absorption,
    ),
    Site(
        id="msl_u_inkernel",
        kind="msl",
        build=_case_builder("cooke-f0,0.7"),
        quantity="L",
        modes=("df64",),
        break_site="U_INKERNEL",
    ),
    Site(
        id="msl_horner",
        kind="msl",
        build=_case_builder("even5"),
        quantity="opd",
        modes=MODES,
        break_site="HORNER",
    ),
    Site(
        id="msl_composed_rotation",
        kind="msl",
        build=_case_builder("tilt-rxryrz"),
        quantity="L",
        modes=MODES,
        break_site="COMPOSED_ROTATION",
    ),
    Site(
        id="msl_sqr",
        kind="msl",
        build=_case_builder("cooke-f0,0.7"),
        quantity="L",
        modes=("df64",),
        break_site="SQR",
    ),
)


def test_round0_sites_cover_every_break_hook():
    """Every ``#ifdef`` WP1 compiled in has a round-0 row, and the modes agree.

    ``tp.BREAK_SITES`` is the kernel's own declaration of which hooks exist
    and in which modes they are arithmetically visible; this file must not
    hold a second, drifting opinion.  A hook added to ``trace.metal`` without
    a site here fails this test rather than going unexercised.
    """
    rows = {s.break_site: s for s in SITES if s.kind == "msl"}
    assert set(rows) == set(tp.BREAK_SITES), (
        f"round-0 MSL rows {sorted(rows)} vs kernel hooks {sorted(tp.BREAK_SITES)}"
    )
    for name, site in rows.items():
        assert site.modes == tp.BREAK_SITES[name], (
            f"{name}: round 0 says {site.modes}, tp.BREAK_SITES says "
            f"{tp.BREAK_SITES[name]}"
        )
    assert {s.id for s in SITES if s.kind == "python"} == {
        "newton_tolerance_floor",
        "refraction",
        "rotation",
        "aperture_inclusivity",
        "absorption",
    }, "the Python half of plan section 6's table changed"


def test_round0_sqr_site_is_justified():
    """Plan section 6 keeps ``BREAK_SQR`` only if ``sqr`` is not ``mul(x, x)``.

    The decision is made by inspecting the headers and is printed, as the
    plan asks, rather than assumed: ``df64_core.h``'s ``sqr_core`` folds the
    two cross terms into one ``fma(2 * a.hi, a.lo, ...)`` while ``mul_core``
    rounds them separately, and ``sf64_core.h`` defines ``sqr(a)`` as
    literally ``mul(a, a)``.  So the site is KEPT, df64 only -- which is what
    ``tp.BREAK_SITES["SQR"]`` says and what the ``msl_sqr`` row asserts in
    both directions.
    """
    df = (tp.KERNEL_DIR / "df64_core.h").read_text(encoding="utf-8")
    sf = (tp.KERNEL_DIR / "sf64_core.h").read_text(encoding="utf-8")
    df_sqr = df[df.index("inline df64 sqr_core(df64 a)") :].split("}", 1)[0]
    df_mul = df[df.index("inline df64 mul_core(df64 a, df64 b)") :].split("}", 1)[0]
    kept = df_sqr.count("fma(") != df_mul.count("fma(")
    assert kept, "df64 sqr_core is mul_core: the BREAK_SQR site must be removed"
    assert "inline sf64 sqr(sf64 a) { return mul(a, a); }" in sf
    assert tp.BREAK_SITES["SQR"] == ("df64",)
    print(
        "round 0 BREAK_SQR: KEPT (df64 only) -- df64 sqr_core has "
        f"{df_sqr.count('fma(')} fma against mul_core's {df_mul.count('fma(')}; "
        "sf64 sqr is literally mul(a, a), so sf64 cannot diverge"
    )


def _trace_site(site: Site, monkeypatch, *, fused: str) -> tc.Capture:
    """Build ``site``'s system and trace one fresh bundle through it."""
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", fused)
    monkeypatch.setenv("OPTILAND_METAL_TRACE_DIAG", "0")
    optic, make = site.build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = optic.surfaces.trace(make())
    return tc.capture(optic.surfaces, out, metal.get_mode())


def tier_a_difference(got: tc.Capture, ref: tc.Capture, what: str) -> str | None:
    """``assert_tier_a``'s message, or None when the two captures agree."""
    try:
        tc.assert_tier_a(got, ref, what)
    except AssertionError as exc:
        return str(exc)
    return None


def differing_quantities(got: tc.Capture, ref: tc.Capture) -> dict[str, list[str]]:
    """``quantity -> ["surface 3", "rays"]`` for every raw word that differs."""
    out: dict[str, list[str]] = {}
    for s, (g_row, r_row) in enumerate(zip(got.rows, ref.rows, strict=True)):
        for attr in sorted(set(g_row) & set(r_row)):
            if any(
                not np.array_equal(g, r, equal_nan=True)
                for g, r in zip(g_row[attr], r_row[attr], strict=True)
            ):
                out.setdefault(attr, []).append(f"surface {s}")
    for attr, g_planes in got.final.items():
        if any(
            not np.array_equal(g, r, equal_nan=True)
            for g, r in zip(g_planes, ref.final[attr], strict=True)
        ):
            out.setdefault(attr, []).append("rays")
    return out


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("site", SITES, ids=lambda s: s.id)
def test_divergence_injection(mps_backend, monkeypatch, site, mode):
    """Plan section 6 round 0: certify the site, then break the comparison.

    Python site, in order:

    1. **Certificate.** The fixture traced twice on mps per-op, with and
       without the injection, must differ -- and must differ in the quantity
       plan section 6's table names, at a named surface.  Without this the
       step below could "fail" for any reason at all.
    2. **Drift.** The injection edits a MIRRORED fingerprint row, so
       ``trace_mirror.check_all()`` must name it.  This is the defence that
       would normally fire first, and it is why step 3 has to set
       ``OPTILAND_METAL_FUSED_TRACE_DRIFT=warn``: under the default policy
       the driver refuses every candidate with ``mirror_drift`` and the
       comparison would pass by falling back rather than by agreeing.
    3. **The comparison fails.** With the kernel actually running, tier A
       against the injected per-op trace must RAISE, and the message must
       name the quantity and the surface.
    4. **And it fails for the right reason.** The fused trace still equals
       the CLEAN per-op trace bit for bit: the kernel mirrors the unpatched
       physics, so the whole difference is the injection.

    MSL site: the certificate is that the ``BREAK_`` library's trace differs
    from the clean library's, asserted in both directions per mode --
    ``tp.BREAK_SITES`` declares ``U_INKERNEL`` and ``SQR`` invisible in sf64
    (sf64 division is correctly rounded and ``sf::sqr`` IS ``mul(a, a)``), so
    there the broken kernel must leave the trace untouched and tier A must
    still pass.
    """
    metal.set_mode(mode)
    what = f"{site.id}[{mode}]"

    if site.kind == "python":
        clean = _trace_site(site, monkeypatch, fused="0")

        assert site.inject is not None
        site.inject(monkeypatch)
        injured = _trace_site(site, monkeypatch, fused="0")

        # 1. certificate
        certificate = tier_a_difference(injured, clean, f"{what} certificate")
        assert certificate is not None, (
            f"{what}: the injection changed nothing on the per-op path; the "
            "site certifies nothing and must be removed (plan section 6)"
        )
        moved = differing_quantities(injured, clean)
        assert site.quantity in moved, (
            f"{what}: the certificate must move {site.quantity!r}; it moved "
            f"{ {k: v[:3] for k, v in moved.items()} }"
        )
        print(f"{what} certificate: {certificate.splitlines()[0]}")

        # 2. the same injection is drift
        assert site.mirror is not None
        problems = trace_mirror.check_all()
        assert any(site.mirror in p for p in problems), (
            f"{what}: the injection edits MIRRORED row {site.mirror}, so "
            f"check_all() must name it; it reported {problems}"
        )

        # 3. the kernel runs (drift hatch) and tier A fails
        monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE_DRIFT", "warn")
        trace.reset_driver_state()
        before = dict(T.stats())
        fused = _trace_site(site, monkeypatch, fused="1")
        assert_was_fused(before, what)
        message = tier_a_difference(fused, injured, what)
        assert message is not None, (
            f"{what}: the fused-vs-per-op tier-A comparison PASSED with the "
            "injection active, so it cannot detect this divergence"
        )
        caught = differing_quantities(fused, injured)
        assert site.quantity in caught, (
            f"{what}: tier A must catch {site.quantity!r}; it caught "
            f"{ {k: v[:3] for k, v in caught.items()} }"
        )
        print(
            f"{what}: tier A FAILS as required -- {site.quantity} at "
            f"{caught[site.quantity]}; first report: {message.splitlines()[0]}"
        )

        # 4. and the kernel still mirrors the unpatched physics
        tc.assert_tier_a(fused, clean, f"{what}: fused vs CLEAN per-op")
        return

    # --- MSL site
    reference = _trace_site(site, monkeypatch, fused="0")
    clean_fused = _trace_site(site, monkeypatch, fused="1")
    tc.assert_tier_a(clean_fused, reference, f"{what}: clean kernel")

    assert site.break_site is not None
    original = trace.trace_source
    define = f"#define OPTILAND_TRACE_BREAK_{site.break_site} 1\n"
    try:
        trace.trace_source = lambda m, _o=original: define + _o(m)
        trace.reset_driver_state(libraries=True)
        before = dict(T.stats())
        broken = _trace_site(site, monkeypatch, fused="1")
        assert_was_fused(before, what)
    finally:
        trace.trace_source = original
        trace.reset_driver_state(libraries=True)

    certificate = tier_a_difference(broken, clean_fused, f"{what} certificate")
    if mode not in site.modes:
        assert certificate is None, (
            f"{what}: tp.BREAK_SITES says this hook is invisible in {mode}, "
            f"but it moved the trace: {certificate}"
        )
        tc.assert_tier_a(broken, reference, f"{what}: invisible in {mode}")
        return

    assert certificate is not None, (
        f"{what}: the BREAK hook changed nothing, so it certifies nothing "
        "and the site must be removed (plan section 6)"
    )
    moved = differing_quantities(broken, clean_fused)
    assert site.quantity in moved, (
        f"{what}: the certificate must move {site.quantity!r}; it moved "
        f"{ {k: v[:3] for k, v in moved.items()} }"
    )
    print(f"{what} certificate: {certificate.splitlines()[0]}")

    message = tier_a_difference(broken, reference, what)
    assert message is not None, (
        f"{what}: the fused-vs-per-op tier-A comparison PASSED with the "
        "broken kernel, so it cannot detect this divergence"
    )
    caught = differing_quantities(broken, reference)
    assert site.quantity in caught, (
        f"{what}: tier A must catch {site.quantity!r}; it caught "
        f"{ {k: v[:3] for k, v in caught.items()} }"
    )
    print(
        f"{what}: tier A FAILS as required -- {site.quantity} at "
        f"{caught[site.quantity]}; first report: {message.splitlines()[0]}"
    )
