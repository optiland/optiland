"""Regression and lock tests closing verify-round 1 findings.

WP0 creates this file with one sanity test so the definition-of-done path
exists even for a round that produces no findings.  The wave-3 fix lane for
round 1 owns it from then on: every finding it closes lands here as a named
regression test (plan 0.1, 4/WP0), or as a documented-limit lock test.

Never widen a tolerance to make a test here pass (plan 0.2.2).

Round 1, iteration 1 closed four findings (``NOTES/fused-trace-research/
verify-round-1.md``):

============ ======================= =======================================
finding      closure                 test
============ ======================= =======================================
R1-V1-01     fix                     ``test_r1v101_diag_planes_are_never_stale``
R1-V1-02     fix                     ``test_r1v102_late_fallback_counts_its_reason``,
                                     ``test_r1v102_long_path_row_exposes_tol_crossover``
R1-V1-03     documented limit        ``test_r1v103_locked``,
                                     ``test_r1v103_locked_gate_scans_apertures_only``
R1-V2-01     fix                     ``test_r1v201_tier_a_sees_zero_sign``,
                                     ``test_r1v201_tier_a_sees_zero_sign_in_a_live_capture``
============ ======================= =======================================

Iteration 2 closed four more, all by fix + named regression test:

============ ======================= =======================================
finding      closure                 test
============ ======================= =======================================
R1-V1-04     fix (gate refusal)      ``test_r1v104_unnormalized_bundle_is_refused``,
                                     ``test_r1v104_normalized_bundle_still_fuses``,
                                     ``test_r1v104_normalize_branch_is_live``
R1-V1-05     fix (gate refusal)      ``test_r1v105_tensor_asphere_coefficient``,
                                     ``test_r1v105_tensor_coefficient_through_variable_update``
R1-V2-02     fix (kernel)            ``test_r1v202_infinite_newton_iterate_is_nan``
R1-V2-03     fix (status oracle)     ``test_r1v203_tir_band_is_excluded_...``,
                                     ``test_r1v203_nfloor_is_compared_...``
============ ======================= =======================================

Iteration 3 closed three more, all by fix + named regression test:

============ ======================= =======================================
finding      closure                 test
============ ======================= =======================================
R1-V1-06     fix (status oracle)     ``test_r1v106_zero_sized_aperture_has_a_rim_band``,
                                     ``test_r1v106_ordinary_aperture_band_is_still_narrow``
R1-V1-07     fix (status oracle)     ``test_r1v107_newton_band_is_excluded_...``,
                                     ``test_r1v107_newton_band_is_empty_on_the_shipped_fixtures``
R1-V2-04     fix (kernel)            ``test_r1v204_reflect_at_negative_zero_dot``,
                                     ``test_r1v204_be_sign_of_a_negative_zero_is_positive``,
                                     ``test_r1v204_refraction_is_unchanged``
============ ======================= =======================================

The documented limit R1-V1-03 is stated once, in full, in the docstring of
``test_r1v103_locked``; ``NOTES/fused-trace-research/documented-limits.md``
carries the same text for WP8's report section "Documented limits".
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import dataclasses  # noqa: E402
import warnings  # noqa: E402
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
    trace_record,
)
from optiland.backend.torch_backend.metal.trace_adapters import (  # noqa: E402
    FusedTraceSkip,
)
from optiland.geometries.odd_asphere import OddAsphere  # noqa: E402
from optiland.physical_apertures.elliptical import (  # noqa: E402
    EllipticalAperture,
)
from optiland.physical_apertures.radial import RadialAperture  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import trace_fixtures as fx  # noqa: E402

from tests.metal import _trace_compare as tc  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator

MODES = tc.MODES

#: The bundle size every tier-A comparison here uses (> 1024 in df64).
N_RAYS = fx.DEFAULT_RAYS

#: A bundle the gate refuses as ``host_resident`` (threshold 256, plan 3.3).
N_HOST_RESIDENT = 64

#: The sf64 int64 word for ``-0.0``.
SF64_NEG_ZERO = np.int64(-9223372036854775808)


def test_round_file_present():
    """Placeholder so round 1 always has a collectable test file."""
    assert True


# ---------------------------------------------------------------------------
# Environment and backend (same contract as test_trace_kernel.py)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _driver_state() -> Iterator[None]:
    """Zero the counters and the driver's one-shot state around every test."""
    trace.reset_driver_state()
    T.reset_stats()
    yield
    trace.reset_driver_state()
    T.reset_stats()


@pytest.fixture
def mps_backend() -> Iterator[None]:
    """torch / mps / float64 with autograd off -- the fused path's setting.

    ``torch.no_grad()`` is load-bearing: with autograd enabled
    ``NewtonRaphsonGeometry.distance`` takes the DiffOptics branch and returns
    one extra refinement step, which is not what the kernel mirrors
    (``test_trace_kernel.mps_backend`` documents the measurement).
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


def stats_delta(before: dict[str, int]) -> dict[str, int]:
    """Every ``fused_trace*`` counter that moved since ``before``."""
    now = T.stats()
    return {
        key: now.get(key, 0) - before.get(key, 0)
        for key in now
        if key.startswith("fused_trace") and now.get(key, 0) - before.get(key, 0) != 0
    }


def quiet_trace(group: Any, rays: Any) -> Any:
    """``group.trace(rays)`` with the driver's RuntimeWarnings suppressed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return group.trace(rays)


# ---------------------------------------------------------------------------
# R1-V1-01 / R1-V1-02 -- the DIAG planes of a trace that did not complete
# ---------------------------------------------------------------------------


def newton_optic(tol: float = 1e-10) -> Any:
    """An even-asphere singlet whose Newton tolerance is settable.

    ``tol = 0`` makes Python's round-off floor ``8 * eps * max(1, |t|)``
    exceed ``tol`` for every ray, so the kernel raises ``ST_TOL_CROSSOVER``
    on the whole design and the driver late-falls-back in BOTH modes -- unlike
    ``long_path_asphere``, which crosses over in df64 only.
    """
    return fx._build(
        [
            {"radius": be.inf, "thickness": be.inf},
            {
                "surface_type": "even_asphere",
                "radius": 20.0,
                "thickness": 6.0,
                "material": "N-BK7",
                "is_stop": True,
                "conic": 0.0,
                "coefficients": [-1.0e-3, -1.0e-5],
                "tol": tol,
            },
            {"radius": -40.0, "thickness": 40.0},
            {},
        ],
        epd=16.0,
    )


def tilted_bundle(num_rays: int) -> Any:
    """The bundle ``newton_optic`` is traced with (``_tilted_collimated_rays``)."""
    return fx.collimated_bundle(num_rays, radius=8.0, z=-10.0, L=0.3)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", ("late_fallback", "refusal"))
def test_r1v101_diag_planes_are_never_stale(mps_backend, monkeypatch, kind, mode):
    """A trace that did not complete on the kernel leaves no previous planes.

    Finding R1-V1-01.  ``trace.diag_from`` is documented as "the (status,
    iters) planes of ``group``'s last fused trace" and plan section 6 round 3
    requires them "overwritten per trace".  Before the fix ``_record_diag`` ran
    only on the success path, so after a second trace of the SAME group that
    fell back late (or was refused by the gate) ``diag_from`` still returned
    the FIRST trace's planes -- the identical tensor object, 4096 columns wide,
    for a trace of 2048 or 64 rays.  A consumer zipping ``diag_from(group)[0]``
    with the bundle it just traced silently read the wrong rays.

    Now the planes are dropped at the top of ``fused_trace`` and only that
    trace's own launch puts them back: absent after a refusal, and after a late
    fallback the discarded launch's own complete planes (R1-V1-02).
    """
    metal.set_mode(mode)
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    monkeypatch.setenv("OPTILAND_METAL_TRACE_DIAG", "1")

    optic = newton_optic()
    group = optic.surfaces
    n_surfaces = len(group.surfaces)

    before = dict(T.stats())
    quiet_trace(group, tilted_bundle(N_RAYS))
    first_delta = stats_delta(before)
    assert first_delta.get("fused_trace:traces", 0) == 1, first_delta
    first = trace.diag_from(group)
    assert first is not None, "the first trace recorded no DIAG planes"
    assert tuple(first[0].shape) == (1, n_surfaces, N_RAYS)
    first_ptr = first[0].data_ptr()

    if kind == "late_fallback":
        group.surfaces[1].geometry.tol = 0.0
        n2 = N_RAYS // 2
    else:
        n2 = N_HOST_RESIDENT

    before = dict(T.stats())
    quiet_trace(group, tilted_bundle(n2))
    second_delta = stats_delta(before)
    assert second_delta.get("fused_trace:traces", 0) == 0, (
        f"the second trace was supposed not to complete: {second_delta}"
    )
    second = trace.diag_from(group)

    if second is not None:
        assert second[0].data_ptr() != first_ptr, (
            f"{kind}[{mode}]: diag_from still returns the FIRST trace's planes "
            f"(shape {tuple(first[0].shape)}) after a trace of {n2} rays"
        )
        assert tuple(second[0].shape) == (1, n_surfaces, n2), (
            f"{kind}[{mode}]: the planes are {tuple(second[0].shape)}, not the "
            f"{n2}-ray trace's own"
        )

    if kind == "refusal":
        assert second_delta.get("fused_trace_skip:host_resident", 0) == 1, second_delta
        assert second is None, (
            "a trace the gate refused never reached the kernel, so it has no "
            f"planes of its own; diag_from returned {second}"
        )
    else:
        assert second_delta.get("fused_trace:late_fallback", 0) == 1, second_delta
        assert second is not None, (
            "a late fallback DID reach the kernel; its planes carry the "
            "ST_TOL_CROSSOVER bit that explains the fallback (R1-V1-02)"
        )
        counts = tc.bit_counts(second[0].cpu().numpy())
        assert counts["tol_crossover"] > 0, counts


@pytest.mark.parametrize("mode", MODES)
def test_r1v102_late_fallback_counts_its_reason(mps_backend, monkeypatch, mode):
    """A late fallback counts WHY it fired: ``fused_trace:diag:tol_crossover``.

    Finding R1-V1-02.  ``ST_TOL_CROSSOVER`` is set only by launches the driver
    discards, and before the fix ``_record_diag`` never saw one, so
    ``fused_trace:diag:tol_crossover`` could not be non-zero on any trace and
    nothing in the counters said why ``fused_trace:late_fallback`` moved.  That
    left the "expected bits > 0 = TOL_CROSSOVER" column of plan 7.2's
    "tolerance crossover / late fallback" row uncheckable through the public
    diagnostic surface.

    The counter is asserted against an independent census of the planes, not
    read off the run.
    """
    metal.set_mode(mode)
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    monkeypatch.setenv("OPTILAND_METAL_TRACE_DIAG", "1")

    optic = newton_optic(tol=0.0)
    group = optic.surfaces

    before = dict(T.stats())
    quiet_trace(group, tilted_bundle(N_RAYS))
    delta = stats_delta(before)

    assert delta.get("fused_trace:late_fallback", 0) == 1, delta
    assert delta.get("fused_trace:traces", 0) == 0, delta
    counted = delta.get("fused_trace:diag:tol_crossover", 0)
    diag_counters = {k: v for k, v in delta.items() if ":diag:" in k}
    assert counted > 0, (
        f"the late fallback counted no reason; fused_trace:diag:* = "
        f"{diag_counters}, full delta = {delta}"
    )

    planes = trace.diag_from(group)
    assert planes is not None
    census = tc.bit_counts(planes[0].cpu().numpy())
    assert counted == census["tol_crossover"], (
        f"the counter says {counted} and the planes say {census['tol_crossover']}"
    )


def test_r1v102_long_path_row_exposes_tol_crossover(mps_backend, monkeypatch):
    """Plan 7.2's own late-fallback fixture now shows its bit through DIAG.

    Finding R1-V1-02, the plan-row half: ``long_path_asphere`` crosses over in
    df64 (the seed distance is ~5000 mm, where Python's round-off floor
    overtakes ``tol = 1e-10``) and completes in sf64.  The df64 fallback is the
    row plan 7.2 writes ``TOL_CROSSOVER`` against, and its bit is now readable
    where the row says it should be.
    """
    metal.set_mode("df64")
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    monkeypatch.setenv("OPTILAND_METAL_TRACE_DIAG", "1")

    optic, rays_factory = fx.long_path_asphere()
    group = optic.surfaces

    before = dict(T.stats())
    quiet_trace(group, rays_factory(optic, N_RAYS))
    delta = stats_delta(before)

    assert delta.get("fused_trace:late_fallback", 0) == 1, delta
    assert delta.get("fused_trace:traces", 0) == 0, delta
    assert delta.get("fused_trace:diag:tol_crossover", 0) > 0, delta

    planes = trace.diag_from(group)
    assert planes is not None
    assert tc.bit_counts(planes[0].cpu().numpy())["tol_crossover"] > 0


# ---------------------------------------------------------------------------
# R1-V1-03 -- documented limit: df64 geometry slots outside float32's range
# ---------------------------------------------------------------------------


def conic_radius_optic(radius: float) -> Any:
    """The Cooke triplet with the stop surface's radius replaced."""
    return fx._cooke({4: {"radius": radius, "thickness": 4.75041, "is_stop": True}})


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("radius", (1e30, 1e-30), ids=("huge", "tiny"))
def test_r1v103_locked(mps_backend, monkeypatch, radius, mode):
    """DOCUMENTED LIMIT (R1-V1-03): df64 cannot hold every geometry parameter.

    **The limit.**  ``_fill_conic_scalars`` stores ``SR_R2 = radius ** 2``.  In
    df64 that word is a float32 pair, so ``radius = 1e30`` gives ``1e60``,
    which overflows float32 to ``+inf``, and ``radius = 1e-30`` gives
    ``1e-60``, which is below ``DF64_FLT_MIN`` and flushes to zero.  Both NaN
    the whole bundle in df64, while sf64 (int64 words, no float32 exponent)
    traces ``radius = 1e30`` normally.  The gate scans the four aperture slots
    ``SR_AP0..SR_AP3`` for exactly this hazard and refuses with
    ``aperture_params``; it does NOT scan ``SR_R``, ``SR_K``, ``SR_K1``,
    ``SR_R2``, ``SR_TOL`` or the coefficient array
    (``test_r1v103_locked_gate_scans_apertures_only`` pins the asymmetry).

    **Why it is a limit and not a bug.**  This is not a fused-vs-per-op
    divergence: the fused and per-op paths agree word for word (tier A, below),
    and the per-op GPU path fails the external rule against NumPy with a
    byte-identical message, so the df64 backend owns the behaviour and the
    kernel reproduces it faithfully.  Refusing at the gate would make the fused
    path REFUSE where the per-op path silently NaNs, which is an improvement
    and therefore forbidden by plan 0.2.1 "mirror, never improve".

    **What this test locks.**  The gate accepts the system, the kernel runs it,
    and the result equals the per-op path exactly -- in both modes, at both
    radii -- plus the measured finite-ray counts that make the df64/sf64
    asymmetry visible instead of implied.
    """
    metal.set_mode(mode)

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    optic_ref = conic_radius_optic(radius)
    rays = fx.pupil_bundle(optic_ref, N_RAYS, Hx=0.0, Hy=0.0)
    out_ref = quiet_trace(optic_ref.surfaces, tc.copy_rays(rays))
    ref = tc.capture(optic_ref.surfaces, out_ref, mode)

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    monkeypatch.setenv("OPTILAND_METAL_TRACE_DIAG", "0")
    before = dict(T.stats())
    optic_fused = conic_radius_optic(radius)
    out_fused = quiet_trace(
        optic_fused.surfaces, tc.copy_rays(fx.pupil_bundle(optic_fused, N_RAYS))
    )
    delta = stats_delta(before)
    got = tc.capture(optic_fused.surfaces, out_fused, mode)

    # The gate accepts an out-of-float32-range geometry slot: that IS the limit.
    assert delta.get("fused_trace:traces", 0) == 1, (
        f"radius = {radius:g} [{mode}]: the gate refused, which would be an "
        f"improvement over the per-op path; {delta}"
    )
    tc.assert_tier_a(got, ref, f"conic_radius_{radius:g}[{mode}]")

    # The measured consequence, pinned exactly (plan 0.2.6: predict, do not
    # inspect).  Only sf64 at radius = 1e30 keeps a finite bundle.
    finite = int(np.isfinite(tc.decode(out_fused.x)).sum())
    expected = N_RAYS if (mode == "sf64" and radius == 1e30) else 0
    assert finite == expected, (
        f"radius = {radius:g} [{mode}]: {finite}/{N_RAYS} rays have a finite "
        f"final x, expected {expected}"
    )


@pytest.mark.parametrize("mode", MODES)
def test_r1v103_locked_gate_scans_apertures_only(mps_backend, monkeypatch, mode):
    """The denormal scan covers ``SR_AP0..SR_AP3`` and no geometry slot.

    The other half of documented limit R1-V1-03: the identical hazard is
    refused on an aperture parameter (``trace_record.py``'s
    ``SR_AP0..SR_AP3`` scan, df64 only) and accepted on a geometry one.  A
    later upstream merge that widens the scan, or narrows it, fails here.
    """
    metal.set_mode(mode)
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")

    aperture_case = fx.denormal_aperture_params()
    ap_optic = aperture_case.optic
    ap_gate = trace_record.can_fuse_trace(
        ap_optic.surfaces, fx.pupil_bundle(ap_optic, N_RAYS), 0
    )
    if mode == "df64":
        assert ap_gate.ok is False
        assert ap_gate.reason is FusedTraceSkip.APERTURE_PARAMS
    else:
        assert ap_gate.ok is True, (
            f"sf64 has no float32 exponent to leave; {ap_gate.reason}"
        )

    for radius in (1e30, 1e-30):
        optic = conic_radius_optic(radius)
        gate = trace_record.can_fuse_trace(
            optic.surfaces, fx.pupil_bundle(optic, N_RAYS), 0
        )
        assert gate.ok is True, (
            f"radius = {radius:g} [{mode}]: the gate now refuses a geometry "
            f"slot ({gate.reason}); that is a change of contract, not a fix "
            "(plan 0.2.1)"
        )


# ---------------------------------------------------------------------------
# R1-V2-01 -- tier A must see the sign of a zero in df64
# ---------------------------------------------------------------------------


def test_r1v201_tier_a_sees_zero_sign():
    """``assert_raw_equal`` rejects ``-0.0`` against ``+0.0`` in df64 too.

    Finding R1-V2-01.  Plan 7.1 tier A is "raw-component equality -- hi/lo
    words, or int64 bit patterns", implemented as
    ``np.array_equal(..., equal_nan=True)``.  ``np.array_equal`` compares with
    ``==`` and ``-0.0 == 0.0``, so before the fix a df64 float32 word of
    ``-0.0`` was accepted against ``+0.0`` while the same divergence in sf64
    (int64 ``-9223372036854775808`` vs ``0``) was rejected: the two modes did
    not have the same acceptance criterion, and every df64 tier-A row in
    ``test_trace_kernel.py``, ``test_trace_conformance.py`` and
    ``test_trace_batch.py`` rested on it.

    The sign of a zero is load-bearing in the mirrored code: ``_conic.py``'s
    ``copysign(sqrt_d, b)`` decides which quadratic root becomes ``t1``,
    ``Plane.distance``'s ``-z / N`` returns ``+inf`` or ``-inf`` by it, and
    ``_sign_preserving_floor`` branches on ``nz >= 0``, which is True for
    ``-0.0``.
    """
    lo = np.zeros(4, dtype=np.float32)
    plus = np.array([0.0, 1.0, -2.0, 0.0], dtype=np.float32)
    minus = np.array([-0.0, 1.0, -2.0, 0.0], dtype=np.float32)

    # df64: the hi words differ only in the sign of one zero.
    with pytest.raises(AssertionError, match="SIGN OF A ZERO"):
        tc.assert_raw_equal([minus, lo], [plus, lo], "df64 hi word")
    # ... and in the lo word, which tier A compares just as strictly.
    with pytest.raises(AssertionError, match="SIGN OF A ZERO"):
        tc.assert_raw_equal([plus, minus], [plus, plus], "df64 lo word")

    # sf64 rejected it before the fix and still does.
    sf_plus = np.array([0, 1, 2, 0], dtype=np.int64)
    sf_minus = np.array([SF64_NEG_ZERO, 1, 2, 0], dtype=np.int64)
    with pytest.raises(AssertionError, match="differs on"):
        tc.assert_raw_equal([sf_minus], [sf_plus], "sf64 words")

    # Equality still means equality: identical words pass, identical signs of
    # zero pass, and the NaN-payload freedom plan 7.1 grants is untouched.
    tc.assert_raw_equal([plus, lo], [plus.copy(), lo.copy()], "identical")
    tc.assert_raw_equal([minus, lo], [minus.copy(), lo.copy()], "identical -0.0")
    nan_a = np.array([np.nan, -0.0], dtype=np.float32)
    nan_b = np.array([np.nan, -0.0], dtype=np.float32)
    nan_b[0] = np.float32(-np.nan)  # a different NaN payload/sign
    tc.assert_raw_equal([nan_a], [nan_b], "nan payloads")


def neg_zero_system() -> Any:
    """A plain singlet -- nothing here but the launch carries a sign of zero."""
    return fx._build(
        [
            {"radius": be.inf, "thickness": be.inf},
            {"radius": 25.0, "thickness": 6.0, "material": "N-BK7", "is_stop": True},
            {"radius": -40.0, "thickness": 45.0},
            {},
        ],
        epd=14.0,
    )


def neg_zero_column(num_rays: int = N_RAYS) -> Any:
    """Every ray on the axis with ``x = y = L = M = -0.0``."""
    zeros = np.full(num_rays, -0.0)
    return fx.make_rays(
        zeros,
        zeros,
        np.full(num_rays, -10.0),
        zeros,
        zeros,
        np.ones(num_rays),
    )


def flip_one_zero_sign(cap: tc.Capture) -> tuple[tc.Capture | None, str]:
    """A copy of ``cap`` with exactly one recorded ``-0.0`` word made ``+0.0``.

    The smallest divergence a kernel could introduce.  Returns
    ``(None, "")`` when the capture holds no negative zero at all, so a
    vacuous injection is reported instead of passing.
    """
    rows = [
        {attr: [c.copy() for c in comps] for attr, comps in row.items()}
        for row in cap.rows
    ]
    for s, row in enumerate(rows):
        for attr, comps in row.items():
            for c, arr in enumerate(comps):
                flat = arr.reshape(-1)
                if arr.dtype == np.float32:
                    neg = np.flatnonzero((flat == 0.0) & np.signbit(flat))
                    replacement = np.float32(0.0)
                elif arr.dtype == np.int64:
                    neg = np.flatnonzero(flat == SF64_NEG_ZERO)
                    replacement = np.int64(0)
                else:  # pragma: no cover - captures hold only these two dtypes
                    continue
                if neg.size:
                    flat[neg[0]] = replacement
                    return (
                        dataclasses.replace(cap, rows=tuple(rows)),
                        f"surface {s}.{attr} component {c} ray {int(neg[0])}",
                    )
    return None, ""


@pytest.mark.parametrize("mode", MODES)
def test_r1v201_tier_a_sees_zero_sign_in_a_live_capture(mps_backend, monkeypatch, mode):
    """The same injection, on a real fused capture, through ``assert_tier_a``.

    Finding R1-V2-01, second half.  An all-``-0.0`` axial column produces a
    capture holding thousands of negative-zero words; flipping exactly one of
    them to ``+0.0`` and re-running plan 7.1's own comparator passed in df64
    before the fix and must fail now.  The un-injected comparison is asserted
    first, so the test also records what the verifier measured: the kernel
    preserves every sign of zero today, in both modes.
    """
    metal.set_mode(mode)

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    optic_ref = neg_zero_system()
    out_ref = quiet_trace(optic_ref.surfaces, neg_zero_column())
    ref = tc.capture(optic_ref.surfaces, out_ref, mode)

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    monkeypatch.setenv("OPTILAND_METAL_TRACE_DIAG", "0")
    before = dict(T.stats())
    optic_fused = neg_zero_system()
    out_fused = quiet_trace(optic_fused.surfaces, neg_zero_column())
    delta = stats_delta(before)
    got = tc.capture(optic_fused.surfaces, out_fused, mode)

    assert delta.get("fused_trace:traces", 0) == 1, delta
    # The kernel is not wrong today: negative zeros included.
    tc.assert_tier_a(got, ref, f"neg_zero_column[{mode}]")

    injected, where = flip_one_zero_sign(got)
    assert injected is not None, (
        f"[{mode}] the capture holds no negative zero, so the injection would "
        "be vacuous and this test would prove nothing"
    )
    with pytest.raises(AssertionError) as excinfo:
        tc.assert_tier_a(injected, ref, f"injected[{mode}]")
    assert where.split(" component")[0] in str(excinfo.value), (
        f"[{mode}] tier A failed, but not at the injected word {where}: {excinfo.value}"
    )


# ---------------------------------------------------------------------------
# Iteration-2 shared helpers
# ---------------------------------------------------------------------------


def raw_differences(
    got: tc.Capture, ref: tc.Capture
) -> list[tuple[int, str, int, int]]:
    """Every recorded raw word of ``got`` that differs from ``ref``.

    Same predicate as :func:`tc.assert_tier_a` (NaN equals NaN, every hi/lo
    word or int64 pattern compared), but it *returns* the differences instead
    of raising, so a test can use "they differ" as a certificate that a case is
    live and "they do not" as the closure.
    """
    out: list[tuple[int, str, int, int]] = []
    for s, (g_row, r_row) in enumerate(zip(got.rows, ref.rows, strict=True)):
        for attr in sorted(g_row):
            for c, (g, r) in enumerate(zip(g_row[attr], r_row[attr], strict=True)):
                g = np.asarray(g).reshape(-1)
                r = np.asarray(r).reshape(-1)
                if np.issubdtype(g.dtype, np.integer):
                    same = g == r
                else:
                    same = (np.isnan(g) & np.isnan(r)) | (g == r)
                out.extend((s, attr, c, int(i)) for i in np.flatnonzero(~same))
    return out


def perop(build, rays, monkeypatch, mode):
    """Trace ``rays`` on a fresh ``build()`` with the hook OFF; return capture."""
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    optic = build()
    out = quiet_trace(optic.surfaces, rays)
    return optic, tc.capture(optic.surfaces, out, mode)


def hooked(build, rays, monkeypatch, mode, *, diag: str = "0"):
    """Trace ``rays`` on a fresh ``build()`` with the hook ON.

    Returns ``(optic, capture, counter delta)``.  The hook either fuses or
    falls back; the counters say which, and the capture is what the caller
    received either way.
    """
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    monkeypatch.setenv("OPTILAND_METAL_TRACE_DIAG", diag)
    optic = build()
    before = dict(T.stats())
    out = quiet_trace(optic.surfaces, rays)
    delta = stats_delta(before)
    return optic, tc.capture(optic.surfaces, out, mode), delta


# ---------------------------------------------------------------------------
# R1-V1-04 -- a bundle whose ``is_normalized`` flag is clear
# ---------------------------------------------------------------------------


def singlet_optic() -> Any:
    """The plain singlet the non-unit bundle is measured on."""
    return fx._build(
        [
            {"radius": be.inf, "thickness": be.inf},
            {"radius": 25.0, "thickness": 6.0, "material": "N-BK7", "is_stop": True},
            {"radius": -40.0, "thickness": 45.0},
            {},
        ],
        epd=12.0,
    )


def newton_singlet_optic() -> Any:
    """The same singlet with a Newton (even-asphere) first surface."""
    return fx._build(
        [
            {"radius": be.inf, "thickness": be.inf},
            {
                "surface_type": "even_asphere",
                "radius": 25.0,
                "thickness": 6.0,
                "material": "N-BK7",
                "is_stop": True,
                "coefficients": [-1.0e-4, 2.0e-7],
            },
            {"radius": -40.0, "thickness": 45.0},
            {},
        ],
        epd=12.0,
    )


#: ``(builder, "how the bundle is made")`` per R1-V1-04 case.
R1V104_CASES: dict[str, tuple[Any, float]] = {
    # name -> (optic builder, direction scale)
    "non_unit_singlet": (singlet_optic, 1.5),
    "non_unit_newton": (newton_singlet_optic, 1.5),
    "unit_cooke": (fx.cooke, 1.0),
}


def r1v104_bundle(optic: Any, scale: float) -> Any:
    """A pupil bundle whose direction cosines are scaled by ``scale``.

    At ``scale = 1`` the bundle is an ordinary one and only the *flag* is a
    lie; at 1.5 the directions are genuinely non-unit, which is what makes
    ``rays.normalize()`` move the trace by millimetres.
    """
    rays = fx.pupil_bundle(optic, N_RAYS, Hx=0.0, Hy=0.7)
    if scale != 1.0:
        rays.L = rays.L * scale
        rays.M = rays.M * scale
        rays.N = rays.N * scale
    return rays


def flagged(rays: Any, *, normalized: bool) -> Any:
    """A clone of ``rays`` -- the same encoded words -- with the flag set.

    ``tc.copy_rays`` rebuilds a ``RealRays``, whose ``__init__`` sets
    ``is_normalized = True`` (``real_rays.py:104``), so the flag is applied to
    the clone and never to the source: every trace in these tests launches from
    byte-identical words and differs only in the flag.
    """
    out = tc.copy_rays(rays)
    out.is_normalized = normalized
    return out


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", sorted(R1V104_CASES), ids=lambda c: c)
def test_r1v104_unnormalized_bundle_is_refused(mps_backend, monkeypatch, case, mode):
    """A bundle with ``is_normalized`` clear is refused, not traced.

    Finding R1-V1-04.  ``HomogeneousPropagation.propagate`` ends with
    ``if not rays.is_normalized: rays.normalize()``
    (``propagation/homogeneous.py:56-57``) -- on *every* surface -- and that
    function is a MIRRORED fingerprint row whose MSL (``propagate_absorb``)
    does not carry the branch.  Nothing else in the metal package mentioned the
    flag: no drift warning, no counted skip, no raise under ``require``, while
    the fused writeback preserved it (``test_trace_writeback.py:772``).  The
    fused path carried the flag and ignored its meaning, and the measured cost
    was 4.90 mm of ``x`` on 4096 of 4096 rays with non-unit directions, in both
    modes, and 4.5e-13 mm on 1413 of 4096 rays of an ordinary df64 pupil bundle
    whose flag alone was cleared.

    The closure is plan 1.2's own remedy: a counted gate refusal with
    transparent fallback.  The reason is ``propagation_model`` -- the
    propagation model the kernel mirrors does not cover this bundle -- because
    the ``FusedTraceSkip`` set is frozen by plan 3.3 and only the integrator
    may add a value (a dedicated ``rays_not_normalized`` is requested in
    ``status.md``).  It is a FEATURE reason, so ``require`` raises on it rather
    than falling back silently, which is what the finding asks for.
    """
    metal.set_mode(mode)
    build, scale = R1V104_CASES[case]

    optic = build()
    launch = r1v104_bundle(optic, scale)

    _, ref = perop(build, flagged(launch, normalized=False), monkeypatch, mode)
    optic_f, got, delta = hooked(
        build, flagged(launch, normalized=False), monkeypatch, mode
    )

    assert delta.get("fused_trace:candidates", 0) == 1, delta
    assert delta.get("fused_trace_skip:propagation_model", 0) == 1, delta
    assert delta.get("fused_trace:traces", 0) == 0, (
        f"{case}[{mode}]: the kernel traced a bundle it does not mirror; {delta}"
    )
    tc.assert_tier_a(got, ref, f"r1v104_{case}[{mode}]")

    # The gate's own verdict, and its class: a candidate refused for a feature.
    gate = trace_record.can_fuse_trace(
        optic_f.surfaces, flagged(launch, normalized=False), 0
    )
    assert gate.ok is False
    assert gate.reason is FusedTraceSkip.PROPAGATION_MODEL
    assert gate.structural is False

    # ... so ``require`` is loud instead of silent.
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "require")
    optic_r = build()
    with pytest.raises(trace.MetalFallbackError, match="propagation_model"):
        quiet_trace(optic_r.surfaces, flagged(launch, normalized=False))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", sorted(R1V104_CASES), ids=lambda c: c)
def test_r1v104_normalized_bundle_still_fuses(mps_backend, monkeypatch, case, mode):
    """The control: the SAME words, flagged normalized, fuse and agree.

    Finding R1-V1-04's own control.  The refusal must key on the flag and on
    nothing else -- a non-unit bundle that claims to be normalized is a bundle
    whose Python path never calls ``normalize()``, so the kernel mirrors it
    exactly and must still take it.  Without this half the gate could refuse
    every bundle and the first test would still pass.
    """
    metal.set_mode(mode)
    build, scale = R1V104_CASES[case]

    optic = build()
    launch = r1v104_bundle(optic, scale)

    _, ref = perop(build, flagged(launch, normalized=True), monkeypatch, mode)
    _, got, delta = hooked(build, flagged(launch, normalized=True), monkeypatch, mode)

    assert delta.get("fused_trace:traces", 0) == 1, delta
    assert not [k for k in delta if k.startswith("fused_trace_skip:")], delta
    tc.assert_tier_a(got, ref, f"r1v104_control_{case}[{mode}]")


@pytest.mark.parametrize("mode", MODES)
def test_r1v104_normalize_branch_is_live(mps_backend, monkeypatch, mode):
    """The certificate: clearing the flag really does move the Python path.

    Finding R1-V1-04, the sensitivity half -- without it the refusal above
    would be a refusal of nothing.  Both traces are per-op, from byte-identical
    launch words, and differ only in ``rays.is_normalized``:

    * non-unit directions (scaled by 1.5): the divergence is gross, 4.90 mm of
      ``x`` on the singlet -- measured on 4096/4096 rays in both modes;
    * an ordinary cooke pupil bundle, flag cleared: ``normalize()`` still
      divides by ``sqrt(L**2 + M**2 + N**2)``, which is not exactly 1 in df64
      (1413 of 4096 rays move in the surface-1 ``L`` low word) but IS exactly
      1.0 in sf64, where the division is exact and the two agree word for word.
      The mode asymmetry is asserted, not averaged over.
    """
    metal.set_mode(mode)

    optic = singlet_optic()
    launch = r1v104_bundle(optic, 1.5)
    optic_t, _ = perop(
        singlet_optic, flagged(launch, normalized=True), monkeypatch, mode
    )
    optic_f, _ = perop(
        singlet_optic, flagged(launch, normalized=False), monkeypatch, mode
    )
    rows_t = tc.decoded_rows(optic_t.surfaces)
    rows_f = tc.decoded_rows(optic_f.surfaces)
    dx = np.nanmax(np.abs(rows_t[-1]["x"] - rows_f[-1]["x"]))
    assert dx > 1.0, (
        f"[{mode}] a non-unit bundle must diverge by millimetres when the "
        f"normalize branch runs; max |dx| = {dx:g} mm"
    )

    cooke = fx.cooke()
    unit_launch = fx.pupil_bundle(cooke, N_RAYS, Hx=0.0, Hy=0.7)
    build = fx.cooke
    _, ref_t = perop(build, flagged(unit_launch, normalized=True), monkeypatch, mode)
    _, ref_f = perop(build, flagged(unit_launch, normalized=False), monkeypatch, mode)
    differences = raw_differences(ref_f, ref_t)
    if mode == "df64":
        assert differences, (
            "df64: re-normalising a unit bundle divides by a value that is not "
            "exactly 1, so the per-op path must move"
        )
    else:
        assert not differences, (
            f"sf64: the divisor is exactly 1.0, so nothing may move; "
            f"{len(differences)} words differ, first {differences[:3]}"
        )


# ---------------------------------------------------------------------------
# R1-V1-05 -- a df64 asphere coefficient stored as a backend tensor
# ---------------------------------------------------------------------------


#: The two coefficients ``coeff_asphere_optic`` carries.
R1V105_COEFFS: tuple[float, ...] = (-1.0e-4, 2.0e-7)


def coeff_asphere_optic(form: str) -> Any:
    """The even-asphere singlet with its coefficients stored as ``form``.

    ``form`` is ``"float"`` (Python floats, today's fixtures), ``"numpy"``
    (``numpy.float64``, what the batch API's value arrays hand the updater) or
    ``"tensor"`` (0-d backend scalars, what ``Variable.update`` stores when the
    caller passes ``be.array(v)`` -- plan 3.9).
    """
    optic = newton_singlet_optic()
    if form == "float":
        values = [float(c) for c in R1V105_COEFFS]
    elif form == "numpy":
        values = [np.float64(c) for c in R1V105_COEFFS]
    elif form == "tensor":
        values = [be.array(float(c)) for c in R1V105_COEFFS]
    else:  # pragma: no cover - programming error
        raise ValueError(form)
    optic.surfaces.surfaces[1].geometry.coefficients = values
    return optic


@pytest.mark.parametrize("mode", MODES)
def test_r1v105_tensor_asphere_coefficient(mps_backend, monkeypatch, mode):
    """A 0-d backend coefficient is refused in df64 and fused in sf64.

    Finding R1-V1-05, in the certificate form the finding requires.
    ``EvenAsphere.sag`` evaluates ``Ci * r2 ** (i + 1)``: a host-scalar df64 op
    when ``Ci`` is a Python or NumPy scalar and an array-array df64 op when it
    is a backend tensor, and the two round differently.  ``_fill_asphere``
    stores ``float(Ci)``, so the record is identical either way and the kernel
    can only ever mirror the host-scalar form -- it therefore equalled per-op
    (FLOAT) while the reference had moved to per-op(TENSOR).  The existing test
    ``test_trace_batch.py::test_updater_applies_scaled_values`` compares the
    compiled TABLES for both forms and passes, because the tables ARE equal; it
    is the per-op trace that moves, and nothing compared that.

    Three traces, one process, one launch bundle:

    * A per-op, coefficients as Python floats;
    * B per-op, coefficients as 0-d backend tensors;
    * C through the hook, coefficients as 0-d backend tensors.

    The certificate is ``A != B`` in df64 (the case is live: the Python path
    itself moved) and ``A == B`` in sf64 (both forms are correctly rounded
    binary64 there, so there is nothing to refuse).  The closure is ``C == B``
    word for word in both modes -- in df64 because the gate now refuses the
    form with a counted ``geometry_type`` and the per-op path runs, in sf64
    because the kernel fuses it and agrees.
    """
    metal.set_mode(mode)

    optic = coeff_asphere_optic("float")
    launch = fx.pupil_bundle(optic, N_RAYS, Hx=0.0, Hy=0.0)

    _, as_float = perop(
        lambda: coeff_asphere_optic("float"), tc.copy_rays(launch), monkeypatch, mode
    )
    _, as_tensor = perop(
        lambda: coeff_asphere_optic("tensor"), tc.copy_rays(launch), monkeypatch, mode
    )
    _, as_numpy = perop(
        lambda: coeff_asphere_optic("numpy"), tc.copy_rays(launch), monkeypatch, mode
    )

    # NumPy scalars take the host-scalar path, so the batch API's usual value
    # arrays are clean in both modes: the form that bites is a genuine backend
    # scalar, and this pins that distinction.
    assert not raw_differences(as_numpy, as_float), (
        f"[{mode}] numpy.float64 must behave exactly like float: "
        f"{len(raw_differences(as_numpy, as_float))} words differ"
    )

    sensitivity = raw_differences(as_tensor, as_float)
    if mode == "df64":
        assert sensitivity, (
            "df64: the certificate is empty -- per-op(float) and per-op(tensor) "
            "agree, so this test would prove nothing about the fused path"
        )
    else:
        assert not sensitivity, (
            f"sf64: both forms are correctly rounded binary64, so the Python "
            f"path must not move; {len(sensitivity)} words differ, first "
            f"{sensitivity[:3]}"
        )

    _, got, delta = hooked(
        lambda: coeff_asphere_optic("tensor"), tc.copy_rays(launch), monkeypatch, mode
    )
    if mode == "df64":
        assert delta.get("fused_trace:candidates", 0) == 1, delta
        assert delta.get("fused_trace_skip:geometry_type", 0) == 1, delta
        assert delta.get("fused_trace:traces", 0) == 0, delta
    else:
        assert delta.get("fused_trace:traces", 0) == 1, delta
        assert not [k for k in delta if k.startswith("fused_trace_skip:")], delta
    tc.assert_tier_a(got, as_tensor, f"r1v105_tensor_coefficients[{mode}]")


@pytest.mark.parametrize("mode", MODES)
def test_r1v105_tensor_coefficient_through_variable_update(
    mps_backend, monkeypatch, mode
):
    """The same form, reached through the public ``Variable.update`` path.

    Finding R1-V1-05's reachability half.  ``OpticUpdater.set_asphere_coeff``
    stores the raw value (``optic_updater.py:159-170``), so
    ``Variable(optic, 'asphere_coeff', ...).update(be.array(v))`` leaves a 0-d
    tensor inside ``EvenAsphere.coefficients`` -- the form plan 3.9 says the
    adapters must accept, and the form ``test_trace_batch.py`` itself
    constructs.  The two updates must put the same NUMBER, or the comparison
    below would be measuring a value change instead of a form change; that is
    asserted before anything is traced.
    """
    metal.set_mode(mode)
    from optiland.optimization.variable.variable import Variable

    def build(form: str):
        """``even_asphere_inf_radius`` with coefficient 0 driven to +3 %.

        The value, the fixture and the ray factory are the probe's
        (``i2v1_02b``), where the divergence measured 2 of 4096 rays.
        """

        def factory():
            optic, _ = fx.even_asphere_inf_radius()
            variable = Variable(
                optic, "asphere_coeff", surface_number=1, coeff_number=0
            )
            physical = float(variable.variable.inverse_scale(variable.value)) * 1.03
            scaled = variable.variable.scale(physical)
            variable.update(be.array(scaled) if form == "tensor" else float(scaled))
            return optic

        return factory

    optic_float = build("float")()
    optic_tensor = build("tensor")()
    stored_float = optic_float.surfaces.surfaces[1].geometry.coefficients[0]
    stored_tensor = optic_tensor.surfaces.surfaces[1].geometry.coefficients[0]
    assert not isinstance(stored_tensor, (float, np.floating)), (
        "Variable.update stored a plain float: the case cannot be reached this "
        "way any more, so this test no longer measures what it claims"
    )
    assert float(stored_tensor) == float(stored_float), (
        "the two forms must hold the same number, or the comparison would be "
        f"about the value: {float(stored_tensor)!r} vs {float(stored_float)!r}"
    )

    _, rays_of = fx.even_asphere_inf_radius()
    launch = rays_of(optic_float, N_RAYS)
    _, as_float = perop(build("float"), tc.copy_rays(launch), monkeypatch, mode)
    _, as_tensor = perop(build("tensor"), tc.copy_rays(launch), monkeypatch, mode)
    _, got, delta = hooked(build("tensor"), tc.copy_rays(launch), monkeypatch, mode)

    sensitivity = raw_differences(as_tensor, as_float)
    if mode == "df64":
        assert sensitivity, "df64: the public-API certificate is empty"
        assert delta.get("fused_trace_skip:geometry_type", 0) == 1, delta
        assert delta.get("fused_trace:traces", 0) == 0, delta
    else:
        assert not sensitivity, f"sf64: {len(sensitivity)} words moved"
        assert delta.get("fused_trace:traces", 0) == 1, delta
    tc.assert_tier_a(got, as_tensor, f"r1v105_variable_update[{mode}]")


# ---------------------------------------------------------------------------
# R1-V2-02 -- an infinite Newton iterate must become NaN, as Python's does
# ---------------------------------------------------------------------------

#: Draw 16 of the iteration-2 edge fuzz, captured verbatim from its generator
#: (``NOTES/fused-trace-research/probes/round1/i2/i2_93``).  Surface 2 is the
#: ``EvenAsphere`` whose Newton iterate leaves float32's range in df64.
FUZZ16_SURFACES: tuple[dict, ...] = (
    {"radius": be.inf, "thickness": be.inf},
    {
        "thickness": 10.89190632784297,
        "is_stop": True,
        "material": "mirror",
        "radius": 20.0,
        "conic": -0.5,
        "surface_type": "even_asphere",
        "coefficients": [0.0, -1e-30],
        "tol": 1e-06,
        "rz": 0.0,
        "dy": 0.3,
    },
    {
        "thickness": 6.480004077850667,
        "radius": -40.0,
        "conic": -1.0,
        "surface_type": "even_asphere",
        "coefficients": [1e-14, 2e-06, -0.0001],
        "tol": 1e-06,
        "rx": -0.03,
        "ry": -0.0,
    },
    {
        "thickness": 6.219640812812122,
        "material": "SF11",
        "radius": 60.0,
        "conic": -2.0,
        "dy": 0.0,
    },
    {},
)

#: That draw's entrance pupil diameter and bundle size.
FUZZ16_EPD = 14.739760533764612
FUZZ16_RAYS = 2048

#: The rays whose recorded row the pre-fix kernel filled with infinities.
FUZZ16_INFINITE_RAYS: tuple[int, ...] = (1438, 1742)

#: The surface those rays diverge on.
FUZZ16_SURFACE = 2


def fuzz16_optic() -> Any:
    return fx._build([dict(s) for s in FUZZ16_SURFACES], epd=FUZZ16_EPD)


def fuzz16_bundle(num_rays: int = FUZZ16_RAYS) -> Any:
    """The fuzz's own launch bundle for seed 16, reproduced exactly.

    Eight interleaved classes: ``N`` on the 1e-14 divisor floor and one ULP
    either side, ``L == N`` bit for bit, ordinary cosines, a backward ray,
    random angles, signed zeros and ``i = 0`` rays
    (``probes/round1/i2/i2_90_edge_fuzz.py::make_rays``).
    """
    floor = 1e-14
    v = float(np.sqrt(0.5))
    rng = np.random.default_rng(16 + 10_000)
    idx = np.arange(num_rays)
    cls = idx % 8
    nz = np.empty(num_rays)
    lx = np.empty(num_rays)
    my = np.zeros(num_rays)
    edge = np.array(
        [
            0.0,
            -0.0,
            floor,
            -floor,
            float(np.nextafter(floor, np.inf)),
            float(np.nextafter(floor, 0.0)),
        ]
    )
    nz[cls == 0] = np.resize(edge, int((cls == 0).sum()))
    lx[cls == 0] = 1.0
    nz[cls == 1] = v
    lx[cls == 1] = v
    nz[cls == 2] = np.sqrt(1.0 - 0.09)
    lx[cls == 2] = 0.3
    nz[cls == 3] = -np.sqrt(1.0 - 0.04)
    lx[cls == 3] = 0.2
    rest = cls >= 4
    angle = rng.uniform(-0.35, 0.35, int(rest.sum()))
    lx[rest] = np.sin(angle)
    nz[rest] = np.cos(angle)
    my[cls == 5] = 0.05
    x = rng.uniform(-6.0, 6.0, num_rays)
    y = rng.uniform(-6.0, 6.0, num_rays)
    x[cls == 6] = -0.0
    y[cls == 6] = -0.0
    z = np.where(idx % 2 == 0, -12.0, -8.0)
    intensity = np.ones(num_rays)
    intensity[cls == 7] = 0.0
    return fx.make_rays(x, y, z, lx, my, nz, intensity)


@pytest.mark.parametrize("mode", MODES)
def test_r1v202_infinite_newton_iterate_is_nan(mps_backend, monkeypatch, mode):
    """An infinite Newton iterate takes one more step and becomes NaN.

    Finding R1-V2-02.  ``trace.metal::newton_distance`` broke out of the loop
    on ``!is_finite(t)``, justified by "in Python it would keep stepping NaN
    into NaN ... with the same final t".  That holds for NaN, which is
    absorbing, and not for an infinity: ``_solve_distance_primal``
    (``newton_raphson.py:355-372``) breaks only on ``be.all(converged)``, so
    while any ray is unconverged an infinite iterate is stepped once more,
    ``F(t) = sag(x + tL, y + tM) - (z + tN)`` evaluates ``inf - inf = NaN``,
    and Python's final ``t`` is NaN.  In df64 the iterate really does overflow
    (the hi word is a float32), so the kernel recorded ``+-inf`` where the
    per-op path recorded NaN -- on the ``x``, ``y``, ``z`` and ``opd`` of an
    ordinary v1 ``EvenAsphere`` row, deterministically, in df64 only.  The
    break condition is now ``is_nan(t)``.

    The system and the launch bundle are draw 16 of the iteration-2 edge fuzz,
    reproduced verbatim (an attempt to reduce it to a hand-built system did not
    reproduce, so the draw IS the repro).  sf64 passed before the fix and must
    still pass: its iterate stays finite.
    """
    metal.set_mode(mode)
    launch = fuzz16_bundle()

    optic_r, ref = perop(fuzz16_optic, tc.copy_rays(launch), monkeypatch, mode)
    optic_f, got, delta = hooked(fuzz16_optic, tc.copy_rays(launch), monkeypatch, mode)

    assert delta.get("fused_trace:traces", 0) == 1, (
        f"[{mode}] the draw must reach the kernel for this to mean anything: {delta}"
    )
    differences = raw_differences(got, ref)
    assert not differences, (
        f"[{mode}] {len(differences)} raw words differ; first {differences[:6]}"
    )
    tc.assert_tier_a(got, ref, f"r1v202_fuzz16[{mode}]")

    # The mechanism, on the two rays the pre-fix kernel filled with infinities:
    # Python's value is NaN, so the kernel's must be NaN -- not +-inf.
    row_ref = tc.decoded_rows(optic_r.surfaces)[FUZZ16_SURFACE]
    row_got = tc.decoded_rows(optic_f.surfaces)[FUZZ16_SURFACE]
    for attr in ("x", "y", "z", "opd"):
        assert np.array_equal(np.isinf(row_got[attr]), np.isinf(row_ref[attr])), (
            f"[{mode}] surface {FUZZ16_SURFACE}.{attr}: the two paths do not "
            "even agree on which entries are infinite"
        )
        if mode != "df64":
            continue
        for ray in FUZZ16_INFINITE_RAYS:
            assert np.isnan(row_ref[attr][ray]), (
                f"ray {ray} is no longer the NaN case this test is about; "
                f"per-op {attr} = {row_ref[attr][ray]!r}"
            )
            assert np.isnan(row_got[attr][ray]), (
                f"[{mode}] surface {FUZZ16_SURFACE}.{attr} ray {ray}: the "
                f"kernel recorded {row_got[attr][ray]!r} where Python records "
                "NaN (an infinite Newton iterate must take one more step)"
            )


# ---------------------------------------------------------------------------
# R1-V2-03 -- what a float64 oracle cannot decide about a df64 trace
# ---------------------------------------------------------------------------

#: ``N`` values straddling ``std_inf_distance``'s 1e-14 divisor floor, in both
#: signs, repeated so neighbouring threads take different branches
#: (``probes/round1/i2/i2_10_nfloor_boundary.py``).
GRAZING_N: tuple[float, ...] = (
    0.0,
    -0.0,
    1e-14,
    -1e-14,
    float(np.nextafter(1e-14, np.inf)),
    -float(np.nextafter(1e-14, np.inf)),
    float(np.nextafter(1e-14, 0.0)),
    -float(np.nextafter(1e-14, 0.0)),
    2e-14,
    -2e-14,
    0.5e-14,
    -0.5e-14,
    1e-20,
    -1e-20,
    1e-40,
    -1e-40,
    1e-13,
    -1e-13,
    1e-7,
    -1e-7,
)


def grazing_slab_optic() -> Any:
    """Two ``StandardGeometry(inf)`` surfaces with glass between them.

    ``GeometryFactory`` collapses ``radius = inf`` to ``Plane`` (whose bare
    ``-z / N`` has no floor), so both geometries are installed by hand.  A ray
    that enters the slab at grazing incidence leaves it at EXACTLY the critical
    angle, which is the knife edge this pair of tests is about.
    """
    optic = fx._build(
        [
            {"radius": be.inf, "thickness": be.inf},
            {
                "surface_type": "plane",
                "thickness": 5.0,
                "material": "N-BK7",
                "is_stop": True,
            },
            {"surface_type": "plane", "thickness": 40.0},
            {},
        ],
        epd=8.0,
    )
    fx._force_standard_infinite(optic, 1)
    fx._force_standard_infinite(optic, 2)
    return optic


def grazing_bundle(num_rays: int = N_RAYS) -> Any:
    """``N`` straddling the floor, ``L`` making up the rest of the direction."""
    nz = np.resize(np.array(GRAZING_N, dtype=np.float64), num_rays)
    lx = np.sqrt(np.maximum(0.0, 1.0 - nz * nz))
    x = np.linspace(-3.0, 3.0, num_rays)
    z = np.where(np.arange(num_rays) % 2 == 0, -4.0, 4.0)
    return fx.make_rays(x, np.zeros(num_rays), z, lx, np.zeros(num_rays), nz)


def grazing_prediction_and_planes(monkeypatch, mode: str):
    """``(prediction, status, iters)`` for the grazing slab in ``mode``."""
    launch = grazing_bundle()
    w0 = float(np.asarray(tc.decode(launch.w)).reshape(-1)[0])

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    optic_ref = grazing_slab_optic()
    quiet_trace(optic_ref.surfaces, tc.copy_rays(launch))
    rows = tc.decoded_rows(optic_ref.surfaces)
    records = tc.compile_tables(optic_ref, mode, w0)
    prediction = tc.predict_status(rows, optic_ref, mode=mode, records=records)

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    monkeypatch.setenv("OPTILAND_METAL_TRACE_DIAG", "1")
    optic_f = grazing_slab_optic()
    before = dict(T.stats())
    quiet_trace(optic_f.surfaces, tc.copy_rays(launch))
    delta = stats_delta(before)
    assert delta.get("fused_trace:traces", 0) == 1, delta
    planes = trace.diag_from(optic_f.surfaces)
    assert planes is not None
    return prediction, planes[0].cpu().numpy()[0], planes[1].cpu().numpy()[0]


@pytest.mark.parametrize("mode", MODES)
def test_r1v203_tir_band_is_excluded_from_the_exact_count(
    mps_backend, monkeypatch, mode
):
    """The status oracle may not certify a TIR bit it cannot decide.

    Finding R1-V2-03.  ``_interact_bits`` tests ``1 - u2 * (1 - dot*dot) < 0``;
    at the critical angle that radicand is zero in exact arithmetic, so its
    sign is decided by round-off -- in df64 at 2**-48 and in the float64 oracle
    at 2**-53.  On this bundle the kernel flags TIR on 820 of 4096 rays at
    surface 2 and ``predict_status`` predicted 3688, with ``|radicand|`` for
    every disagreeing ray equal to 6.661e-16 while the non-grazing rays of the
    same bundle sit at 1.3.  The kernel is NOT wrong there -- plan 7.1 tier A
    passes on every raw word in both modes, and a census taken from the per-op
    path's own recorded rows equals the kernel's plane entry for entry -- the
    float64 oracle is the odd one out, so a fixture or fuzz draw with a ray at
    the critical angle failed a status assertion the kernel passed.

    ``ST_TIR`` now gets the treatment ``ST_CLIPPED`` already had: rays inside
    the band are marked ``tir_uncertain`` and excluded from the exact count by
    ``mask_uncertain``, in the prediction and in the measurement alike.  No
    tolerance was widened: every other bit, every other ray and both ``iters``
    planes are still compared exactly, and in sf64 the band is empty by
    construction (its width is ``MACHINE_EPS[mode] - MACHINE_EPS['sf64']``),
    so the sf64 count stays exactly as strict as it was.
    """
    metal.set_mode(mode)
    prediction, status, iters = grazing_prediction_and_planes(monkeypatch, mode)

    got = tc.mask_uncertain(status, prediction)
    want = tc.mask_uncertain(prediction.bits, prediction)
    bad = np.argwhere(got != want)
    assert bad.size == 0, (
        f"[{mode}] masked status differs at {len(bad)} (surface, ray) entries; "
        f"first {bad[:5].tolist()}"
    )
    assert np.array_equal(iters, prediction.iters)

    raw_tir = int(np.count_nonzero(status & trace_layout.ST_TIR))
    predicted_tir = int(np.count_nonzero(prediction.bits & trace_layout.ST_TIR))
    uncertain = int(prediction.tir_uncertain.sum())
    if mode == "df64":
        # The certificate that the band is doing work: without the exclusion
        # the raw counts disagree, which is exactly what the finding measured.
        assert raw_tir != predicted_tir, (
            "df64: the oracle and the kernel now agree on the raw TIR count, "
            "so this bundle no longer exercises the band and the test proves "
            "nothing"
        )
        assert uncertain > 0
    else:
        assert raw_tir == predicted_tir, (
            f"sf64 is binary64: the oracle must reproduce the kernel's TIR bit "
            f"exactly; {raw_tir} vs {predicted_tir}"
        )
        assert uncertain == 0, (
            f"sf64: the band must be empty by construction, {uncertain} entries"
        )


@pytest.mark.parametrize("mode", MODES)
def test_r1v203_nfloor_is_compared_in_the_mode_representation(
    mps_backend, monkeypatch, mode
):
    """The oracle compares against the constant the KERNEL compares against.

    The second half of the same probe check.  ``std_inf_distance`` floors the
    divisor at ``1e-14`` and raises ``ST_NZ_FLOORED`` when the floor CHANGED
    the divisor (``ns != N``).  The oracle compared a decoded df64 ``N``
    against the float64 literal, while the kernel compares it against
    ``decode(encode(1e-14))``; the two differ in the last bits, so every ray
    sitting exactly on the constant was mispredicted -- 410 of 4096 at surface
    1, measured, df64 only.  ``mode_scalar`` rounds the constant to the mode's
    own representation first.  This is not a band and not a tolerance: the
    comparison is exact again.
    """
    metal.set_mode(mode)
    prediction, status, _ = grazing_prediction_and_planes(monkeypatch, mode)

    bit = trace_layout.ST_NZ_FLOORED
    for s in range(status.shape[0]):
        kernel = (status[s] & bit) != 0
        predicted = (prediction.bits[s] & bit) != 0
        bad = np.flatnonzero(kernel != predicted)
        assert bad.size == 0, (
            f"[{mode}] surface {s}: NZ_FLOORED differs on {bad.size} rays "
            f"(kernel {int(kernel.sum())}, oracle {int(predicted.sum())}); "
            f"first {bad[:5].tolist()}"
        )


# ---------------------------------------------------------------------------
# R1-V2-04 -- `sign(-0.0)` at a mirror, where the per-op path returns +0.0
# ---------------------------------------------------------------------------

#: ``sqrt(0.5)``: the two transverse cosines of the armed classes below.
V_HALF = float(np.sqrt(0.5))

#: The four sign classes of ``(x, y, L, M, N)`` the mirror bundle interleaves.
#: Class 3 is the armed one: ``L0 < 0``, ``M0 < 0`` and ``N0 = -0.0`` make all
#: three addends of ``dot = (L0*nx + M0*ny) + N0*nz`` equal ``-0.0``, so the
#: sum is ``-0.0`` -- the only input on which ``be.sign`` and ``df::sign`` /
#: ``sf::sign`` disagree (probes/round1/i3/i3_04b).
NEG_ZERO_DOT_CLASSES: tuple[tuple[float, float, float, float, float], ...] = (
    (0.0, 0.0, V_HALF, V_HALF, 0.0),
    (0.0, 0.0, V_HALF, V_HALF, -0.0),
    (0.0, 0.0, -V_HALF, -V_HALF, 0.0),
    (0.0, 0.0, -V_HALF, -V_HALF, -0.0),
)

#: The class index whose ``dot`` is ``-0.0`` (one ray in four of the bundle).
NEG_ZERO_CLASS = 3


def mirror_slab_optic() -> Any:
    """A ``Plane`` MIRROR between two planes -- the reflect branch, flat."""
    return fx._build(
        [
            {"radius": be.inf, "thickness": be.inf},
            {
                "radius": be.inf,
                "thickness": 14.0,
                "is_stop": True,
                "material": "mirror",
            },
            {"radius": be.inf, "thickness": 9.0},
            {},
        ],
        epd=12.0,
    )


def refracting_slab_optic() -> Any:
    """The same system with glass in place of the mirror (the control twin).

    Refraction is immune to the sign of the zero -- the ``u * N0 + nz * root``
    term lands on ``+0.0`` under either convention -- so this twin must agree
    word for word BEFORE and AFTER the fix.  A fix that reached into
    ``refract`` would fail here.
    """
    return fx._build(
        [
            {"radius": be.inf, "thickness": 14.0, "is_stop": True},
            {"radius": be.inf, "thickness": 9.0, "material": "N-BK7"},
            {"radius": be.inf, "thickness": 9.0},
            {},
        ],
        epd=12.0,
    )


def neg_zero_dot_bundle(num_rays: int = N_RAYS) -> Any:
    """The four sign classes, interleaved (``i3_04b``'s bundle)."""
    arr = np.array(NEG_ZERO_DOT_CLASSES, dtype=np.float64)
    sel = np.arange(num_rays) % len(NEG_ZERO_DOT_CLASSES)
    return fx.make_rays(
        arr[sel, 0],
        arr[sel, 1],
        np.full(num_rays, -10.0),
        arr[sel, 2],
        arr[sel, 3],
        arr[sel, 4],
    )


def zero_word_is_negative(component: np.ndarray, ray: int) -> bool:
    """Is raw ``component[ray]`` a NEGATIVE zero, in either representation?"""
    flat = np.asarray(component).reshape(-1)
    value = flat[ray]
    if flat.dtype == np.int64:
        return bool(value == SF64_NEG_ZERO)
    return bool(value == 0.0 and np.signbit(value))


@pytest.mark.parametrize("mode", MODES)
def test_r1v204_reflect_at_negative_zero_dot(mps_backend, monkeypatch, mode):
    """``trace_ops<R>::sign`` must be ``be.sign``, which is ``+0.0`` at ``-0.0``.

    Finding R1-V2-04.  ``RealRays._align_surface_normal`` aligns the normal
    with ``be.sign(dot)``, and ``be.sign`` is NOT the ``sign`` kernel:
    ``ops_elementwise._sign`` runs it and then masks NaN and both zeros to
    ``+0.0`` (torch's ``(0 < x) - (x < 0)``; measured here and in
    ``test_r1v204_be_sign_of_a_negative_zero_is_positive``).  ``df::sign`` and
    ``sf::sign`` preserve the sign of a zero, so at ``dot == -0.0`` the kernel
    aligned the normal with the OPPOSITE zero and the reflect branch recorded
    ``N = N0 - (2*dot)*nz = -0.0 - (-0.0) = +0.0`` where the per-op path
    records ``-0.0 - (+0.0) = -0.0``: 1024 of these 4096 rays, in BOTH modes,
    on ``surface 1.N``.  The fix is trace-local (``trace.metal``'s
    ``trace_ops<R>::sign``); ``df64_core.h`` / ``sf64_core.h`` are shared with
    the elementwise backend and keep their raw ``sign``, with their comments
    corrected.

    Not a cosmetic bit: ``Plane.distance`` divides by ``N``, so its sign is the
    sign of the resulting infinity.  The blast radius measured in the finding
    (one recorded plane, healed at the next surface) is why nothing else moved.
    """
    metal.set_mode(mode)
    launch = neg_zero_dot_bundle()

    optic_r, ref = perop(mirror_slab_optic, tc.copy_rays(launch), monkeypatch, mode)
    optic_f, got, delta = hooked(
        mirror_slab_optic, tc.copy_rays(launch), monkeypatch, mode
    )
    assert delta.get("fused_trace:traces", 0) == 1, (
        f"[{mode}] the bundle must reach the kernel: {delta}"
    )

    differences = raw_differences(got, ref)
    assert not differences, (
        f"[{mode}] {len(differences)} raw words differ; first {differences[:6]}"
    )
    tc.assert_tier_a(got, ref, f"r1v204_mirror[{mode}]")

    # The mechanism, on the armed class: the per-op path records a NEGATIVE
    # zero for N at the mirror, so the kernel must record one too.
    ray = NEG_ZERO_CLASS
    ref_n = ref.rows[1]["N"]
    got_n = got.rows[1]["N"]
    assert zero_word_is_negative(ref_n[0], ray), (
        f"[{mode}] ray {ray} is no longer the -0.0 case this test is about; "
        f"per-op surface 1.N = {np.asarray(ref_n[0]).reshape(-1)[ray]!r}"
    )
    assert zero_word_is_negative(got_n[0], ray), (
        f"[{mode}] surface 1.N ray {ray}: the kernel recorded "
        f"{np.asarray(got_n[0]).reshape(-1)[ray]!r} where the per-op path "
        "records -0.0 (sign(-0.0) must be +0.0, so the aligned normal keeps "
        "its own sign)"
    )

    # Every class, so the fix cannot have passed by flipping every zero: the
    # recorded N keeps the sign of N0 (classes 1 and 3 launch with -0.0), and
    # the kernel matches the per-op path on all four.  Only class 3 has
    # dot == -0.0; classes 0-2 sum to +0.0 and agreed before the fix too.
    for cls, launch_n in enumerate(c[4] for c in NEG_ZERO_DOT_CLASSES):
        expected_negative = bool(np.signbit(launch_n))
        assert zero_word_is_negative(ref_n[0], cls) is expected_negative, (
            f"[{mode}] class {cls} no longer records the sign of its own N0; "
            "the bundle changed and this test is measuring something else"
        )
        assert zero_word_is_negative(got_n[0], cls) is expected_negative, (
            f"[{mode}] class {cls}: the kernel's surface 1.N sign differs "
            "from the per-op path's"
        )


@pytest.mark.parametrize("mode", MODES)
def test_r1v204_be_sign_of_a_negative_zero_is_positive(mps_backend, mode):
    """The certificate: what the kernel must mirror, measured on the backend.

    ``be.sign(-0.0)`` is ``+0.0`` in both modes -- numpy's and torch's value
    (``numpy.sign(-0.0)`` and ``torch.sign(-0.0)`` are ``+0.0``, measured on
    this checkout), produced by ``ops_elementwise._sign``'s host-side mask and
    not by the ``sign`` kernel, which preserves the sign.  If this ever
    changes, ``trace_ops<R>::sign`` is wrong and
    ``test_r1v204_reflect_at_negative_zero_dot`` is measuring the wrong thing.
    """
    metal.set_mode(mode)
    values = np.array([-0.0, 0.0, -1.5, 1.5], dtype=np.float64)
    out = be.sign(be.array(values))
    words = tc.raw(out)[0]
    assert not zero_word_is_negative(words, 0), (
        f"[{mode}] be.sign(-0.0) now carries a negative zero: "
        f"{np.asarray(words).reshape(-1)[0]!r}"
    )
    decoded = tc.decode(out)
    assert decoded.tolist() == [0.0, 0.0, -1.0, 1.0]
    assert not bool(np.signbit(decoded[0])) and not bool(np.signbit(decoded[1]))


@pytest.mark.parametrize("mode", MODES)
def test_r1v204_refraction_is_unchanged(mps_backend, monkeypatch, mode):
    """The control: the refracting twin agreed before the fix and still does.

    Refraction's extra ``u * N0 + nz * root`` term lands on ``+0.0`` under
    either sign convention (measured in the finding), which is why only the
    mirror case failed.  This twin is the guard that the closure stayed inside
    ``_align_surface_normal``: a fix that also moved ``refract`` fails here.
    """
    metal.set_mode(mode)
    launch = neg_zero_dot_bundle()
    _, ref = perop(refracting_slab_optic, tc.copy_rays(launch), monkeypatch, mode)
    _, got, delta = hooked(
        refracting_slab_optic, tc.copy_rays(launch), monkeypatch, mode
    )
    assert delta.get("fused_trace:traces", 0) == 1, delta
    differences = raw_differences(got, ref)
    assert not differences, (
        f"[{mode}] refraction moved: {len(differences)} raw words differ; "
        f"first {differences[:6]}"
    )
    tc.assert_tier_a(got, ref, f"r1v204_refract[{mode}]")


# ---------------------------------------------------------------------------
# R1-V1-06 -- an aperture edge of zero has no relative rim band
# ---------------------------------------------------------------------------


def zero_edge_optic(r_max: float = 0.0, tilt: float = 0.03) -> Any:
    """A singlet whose stop carries ``RadialAperture(r_max)``, tilted by ``rx``.

    The tilt is what makes the pole interesting: the local hit point of an
    axial ray is ``y0 + t*M`` with both terms ~0.3 mm, so whether it is
    EXACTLY zero -- which is what ``contains`` asks at ``r_max = 0`` -- is a
    property of the arithmetic, not of the ray.
    """
    entry: dict[str, Any] = {
        "radius": 50.0,
        "thickness": 5.0,
        "material": "N-BK7",
        "is_stop": True,
        "aperture": RadialAperture(r_max=r_max),
    }
    if tilt:
        entry["rx"] = tilt
    return fx._build(
        [
            {"radius": be.inf, "thickness": be.inf},
            entry,
            {"radius": -50.0, "thickness": 45.0},
            {},
        ],
        epd=20.0,
    )


def axial_disc_bundle(num_rays: int = N_RAYS, *, axial: bool = True) -> Any:
    """The shipped disc, with every 32nd ray moved exactly onto the axis.

    ``zero_rmax_aperture``'s own bundle has no ray at the origin (its
    golden-angle spiral starts at ``r = radius * sqrt(0.5 / num_rays)``), which
    is why the shipped fixture never armed this.  ``axial=False`` is that
    bundle, for the control.
    """
    px, py = fx._spiral(num_rays, 10.0)
    if axial:
        px[0::32] = 0.0
        py[0::32] = 0.0
    return fx.make_rays(
        px,
        py,
        np.full(num_rays, -10.0),
        np.zeros(num_rays),
        np.zeros(num_rays),
        np.ones(num_rays),
    )


def zero_edge_prediction(
    monkeypatch, mode: str, *, r_max: float = 0.0, axial: bool = True
):
    """``(prediction, status, per-op clip census, rows)`` for the zero edge."""
    launch = axial_disc_bundle(axial=axial)
    w0 = float(np.asarray(tc.decode(launch.w)).reshape(-1)[0])

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    optic_ref = zero_edge_optic(r_max=r_max)
    quiet_trace(optic_ref.surfaces, tc.copy_rays(launch))
    rows = tc.decoded_rows(optic_ref.surfaces)
    records = tc.compile_tables(optic_ref, mode, w0)
    prediction = tc.predict_status(rows, optic_ref, mode=mode, records=records)

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    monkeypatch.setenv("OPTILAND_METAL_TRACE_DIAG", "1")
    optic_f = zero_edge_optic(r_max=r_max)
    before = dict(T.stats())
    quiet_trace(optic_f.surfaces, tc.copy_rays(launch))
    delta = stats_delta(before)
    assert delta.get("fused_trace:traces", 0) == 1, delta
    planes = trace.diag_from(optic_f.surfaces)
    assert planes is not None
    # The per-op path's OWN verdict: a clip sets the intensity to exactly 0.
    census = (rows[1]["intensity"] == 0.0) & (rows[0]["intensity"] != 0.0)
    return prediction, planes[0].cpu().numpy()[0], census, rows


@pytest.mark.parametrize("mode", MODES)
def test_r1v106_zero_sized_aperture_has_a_rim_band(mps_backend, monkeypatch, mode):
    """A clip the mode's round-off decides may not be asserted exactly.

    Finding R1-V1-06.  ``rim_band_mask`` skipped an edge of zero and is
    otherwise relative, so at ``r_max = 0`` -- attack item 7, and the shipped
    ``zero_rmax_aperture`` fixture -- ``clip_uncertain`` was EMPTY and every
    ray had to be predicted exactly.  At that aperture ``contains`` is
    ``r2 <= 0 and r2 >= 0``, true only at the exact pole, so a ray whose local
    hit point is the pole is decided by whether ``x`` and ``y`` are exactly
    zero after the cancellation ``y0 + t*M`` at a tilted surface.  Measured:
    in df64 the kernel keeps the 128 axial rays and the float64 oracle clipped
    all 4096; in sf64 all three agree.

    The kernel is the side that is right, and the certificate is the per-op
    path's own recorded intensity -- a clip sets it to exactly zero -- which
    equals the kernel's plane ray for ray in both modes.  The band therefore
    gets an absolute floor (``rim_floor``: the mode's excess round-off over
    float64 times the magnitude of the terms whose cancellation made the hit
    point), which is exactly zero in sf64, so no sf64 count and no non-zero
    edge changes.
    """
    metal.set_mode(mode)
    prediction, status, census, _ = zero_edge_prediction(monkeypatch, mode)

    got = tc.mask_uncertain(status, prediction)
    want = tc.mask_uncertain(prediction.bits, prediction)
    bad = np.argwhere(got != want)
    assert bad.size == 0, (
        f"[{mode}] masked status differs at {len(bad)} (surface, ray) entries; "
        f"first {bad[:5].tolist()}"
    )

    kernel_clip = (status[1] & trace_layout.ST_CLIPPED) != 0
    assert np.array_equal(kernel_clip, census), (
        f"[{mode}] the kernel's CLIPPED plane and the per-op path's own "
        f"intensity census disagree on "
        f"{int(np.count_nonzero(kernel_clip != census))} rays -- the kernel, "
        "not the oracle, would then be the problem"
    )

    axial = np.zeros(status.shape[1], dtype=bool)
    axial[0::32] = True
    predicted_clip = (prediction.bits[1] & trace_layout.ST_CLIPPED) != 0
    banded = int(prediction.clip_uncertain[1].sum())
    if mode == "df64":
        # The certificate that the band is doing work: the RAW counts still
        # disagree, on the axial rays and only on them.
        assert int(kernel_clip.sum()) != int(predicted_clip.sum()), (
            "df64: the oracle and the kernel now agree on the raw clip count, "
            "so this bundle no longer exercises the band"
        )
        disputed = np.flatnonzero(kernel_clip != predicted_clip)
        assert disputed.size > 0 and bool(np.all(axial[disputed])), (
            f"df64: {disputed.size} disputed rays, not all of them axial"
        )
        assert banded >= disputed.size, (
            f"df64: the band covers {banded} rays but {disputed.size} are disputed"
        )
    else:
        assert np.array_equal(kernel_clip, predicted_clip), (
            "sf64 is binary64: the oracle must reproduce the kernel's clip "
            f"exactly ({int(kernel_clip.sum())} vs {int(predicted_clip.sum())})"
        )
        assert banded == 0, (
            f"sf64: the rim band must stay empty at a zero edge, {banded} rays"
        )


@pytest.mark.parametrize("axial", (True, False), ids=("on-the-edge", "off-the-edge"))
@pytest.mark.parametrize("mode", MODES)
def test_r1v106_ordinary_aperture_band_is_still_narrow(
    mps_backend, monkeypatch, mode, axial
):
    """The floor covers the rays ON a zero edge and no others.

    The control for the test above, on an ordinary ``r_max = 8`` mm annulus
    (``RadialAperture``'s default ``r_min`` is 0.0, so it still HAS a zero
    edge -- at the pole).  With the axial rays present the band must cover
    exactly them; with the plain disc, which has no ray at the pole, it must
    be empty.  Either way the kernel, the oracle and the per-op intensity
    census agree exactly, ray for ray, in both modes: an absolute floor of
    ``32 * (eps[mode] - eps['sf64']) * scale`` is ~1e-12 mm here, far too
    narrow to reach the 8 mm edge, and it is zero in sf64.
    """
    metal.set_mode(mode)
    prediction, status, census, _ = zero_edge_prediction(
        monkeypatch, mode, r_max=8.0, axial=axial
    )

    kernel_clip = (status[1] & trace_layout.ST_CLIPPED) != 0
    predicted_clip = (prediction.bits[1] & trace_layout.ST_CLIPPED) != 0
    assert int(kernel_clip.sum()) > 0, "the control must actually clip something"
    assert np.array_equal(kernel_clip, census)
    assert np.array_equal(kernel_clip, predicted_clip)

    band = prediction.clip_uncertain
    on_the_pole = np.zeros(status.shape[1], dtype=bool)
    if axial:
        on_the_pole[0::32] = True
    if mode == "df64" and axial:
        assert np.array_equal(band[1], on_the_pole), (
            f"df64: the band covers {int(band[1].sum())} rays where "
            f"{int(on_the_pole.sum())} sit on the r_min = 0 edge"
        )
    else:
        assert int(band.sum()) == 0, (
            f"[{mode}] the rim band covers {int(band.sum())} rays "
            f"(axial={axial}); the floor is too wide"
        )


# ---------------------------------------------------------------------------
# R1-V1-07 -- what a float64 oracle cannot decide about a df64 Newton loop
# ---------------------------------------------------------------------------

#: Draw 14 of the iteration-3 odd-item fuzz, captured verbatim from its
#: generator (``probes/round1/i3/i3v1_90_odd_items_fuzz.py``).  Surface 1 is an
#: odd asphere with ``max_iter = 0``, so its ``ST_NEWTON_NOT_CONVERGED`` bit IS
#: the single comparison ``|F(t_seed)| < tol`` -- the knife edge of this
#: finding.
FUZZ14_EPD = 14.0
FUZZ14_W0 = 2.5
FUZZ14_RAYS = 2048

#: Draw 9 of the same fuzz: ``tol = 1e3`` with ``max_iter = 254`` on an odd
#: asphere of infinite radius, whose float64 iterates reach 4.3e143 -- past
#: float32's 3.4e38, so the df64 iterate is an infinity and NaN one step later.
FUZZ9_EPD = 14.0
FUZZ9_W0 = 0.3
FUZZ9_RAYS = 2048


def fuzz14_optic() -> Any:
    return fx._build(
        [
            {"radius": be.inf, "thickness": be.inf},
            {
                "z": 0.0,
                "is_stop": True,
                "material": "mirror",
                "rx": 0.03,
                "rz": 0.7218,
                "aperture": RadialAperture(r_max=np.inf, r_min=0.0),
                "surface_type": "odd_asphere",
                "radius": -8.0,
                "conic": 0.5,
                "coefficients": [0.0],
                "tol": 1e-12,
                "max_iter": 0,
            },
            {
                "z": 2.0,
                "is_stop": False,
                "material": "SK16",
                "rx": 0.03,
                "ry": 0.7218,
                "rz": -0.03,
                "aperture": EllipticalAperture(a=6.0, b=2.0),
                "surface_type": "even_asphere",
                "radius": 20.0,
                "conic": 0.5,
                "coefficients": [-1e-4, 2e-6],
                "tol": 1e-8,
                "max_iter": 20,
            },
            {"z": 27.0},
        ],
        epd=FUZZ14_EPD,
        wavelengths=(FUZZ14_W0,),
    )


def fuzz9_optic() -> Any:
    optic = fx._build(
        [
            {"radius": be.inf, "thickness": be.inf},
            {
                "z": 0.0,
                "is_stop": True,
                "material": "F2",
                "rz": -0.03,
                "aperture": EllipticalAperture(a=6.0, b=6.0),
                "surface_type": "odd_asphere",
                "radius": 25.0,
                "conic": 0.5,
                "coefficients": [1e-30, 0.0, -1e-14],
                "tol": 1e3,
                "max_iter": 254,
            },
            {
                "z": 12.0,
                "is_stop": False,
                "rx": 0.7218,
                "ry": 0.03,
                "rz": 0.7218,
                "aperture": RadialAperture(r_max=np.inf, r_min=0.0),
                "radius": be.inf,
            },
            {
                "z": 20.0,
                "is_stop": False,
                "rx": 0.7218,
                "rz": -0.5,
                "aperture": EllipticalAperture(a=6.0, b=2.0),
                "radius": be.inf,
            },
            {"z": 52.0},
        ],
        epd=FUZZ9_EPD,
        wavelengths=(FUZZ9_W0,),
    )
    # The draw patches surface 1 to R = -inf, which GeometryFactory cannot
    # build (it collapses an infinite radius to a Plane).
    surface = optic.surfaces.surfaces[1]
    surface.geometry = OddAsphere(
        coordinate_system=surface.geometry.cs,
        radius=-np.inf,
        conic=0.5,
        coefficients=[1e-30, 0.0, -1e-14],
        tol=1e3,
        max_iter=254,
    )
    return optic


def fuzz_edge_bundle(wavelength: float, num_rays: int = FUZZ14_RAYS) -> Any:
    """The odd-item fuzz's own launch bundle (its ``_bundle``, verbatim).

    Vertex, wide-angle, exactly grazing (both signs of a zero ``N``),
    backward, ``i = 0``, NaN position, NaN direction and denormal-ish
    direction lanes, interleaved every 32 rays over a 7 mm spiral.
    """
    px, py = fx._spiral(num_rays, 7.0)
    lx = np.zeros(num_rays)
    my = np.zeros(num_rays)
    nz = np.ones(num_rays)
    z0 = np.full(num_rays, -12.0)
    intensity = np.ones(num_rays)
    px[0::32] = 0.0
    py[0::32] = 0.0
    lx[1::32] = 0.6
    nz[1::32] = np.sqrt(1.0 - 0.36)
    lx[2::32] = 1.0
    nz[2::32] = 0.0
    lx[3::32] = 1.0
    nz[3::32] = -0.0
    nz[4::32] = -1.0
    z0[4::32] = 12.0
    intensity[5::32] = 0.0
    px[6::32] = np.nan
    lx[7::32] = np.nan
    nz[7::32] = np.nan
    lx[8::32] = 1e-14
    nz[8::32] = np.sqrt(1.0 - 1e-28)
    return fx.make_rays(px, py, z0, lx, my, nz, intensity, wavelength)


def newton_prediction_and_planes(build, wavelength, monkeypatch, mode: str):
    """``(prediction, status, iters, capture pair)`` for a fuzz draw."""
    launch = fuzz_edge_bundle(wavelength)

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    optic_ref = build()
    out = quiet_trace(optic_ref.surfaces, tc.copy_rays(launch))
    ref = tc.capture(optic_ref.surfaces, out, mode)
    rows = tc.decoded_rows(optic_ref.surfaces)
    w0 = float(np.asarray(tc.decode(out.w)).reshape(-1)[0])
    records = tc.compile_tables(optic_ref, mode, w0)
    prediction = tc.predict_status(rows, optic_ref, mode=mode, records=records)

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    monkeypatch.setenv("OPTILAND_METAL_TRACE_DIAG", "1")
    optic_f = build()
    before = dict(T.stats())
    out_f = quiet_trace(optic_f.surfaces, tc.copy_rays(launch))
    delta = stats_delta(before)
    assert delta.get("fused_trace:traces", 0) == 1, delta
    got = tc.capture(optic_f.surfaces, out_f, mode)
    planes = trace.diag_from(optic_f.surfaces)
    assert planes is not None
    return (
        prediction,
        planes[0].cpu().numpy()[0],
        planes[1].cpu().numpy()[0],
        (got, ref),
    )


@pytest.mark.parametrize(
    ("draw", "build", "wavelength"),
    (
        pytest.param("seed14", fuzz14_optic, FUZZ14_W0, id="threshold"),
        pytest.param("seed9", fuzz9_optic, FUZZ9_W0, id="float32-range"),
    ),
)
@pytest.mark.parametrize("mode", MODES)
def test_r1v107_newton_band_is_excluded_from_the_exact_count(
    mps_backend, monkeypatch, mode, draw, build, wavelength
):
    """The status oracle may not certify a Newton verdict it cannot decide.

    Finding R1-V1-07.  ``_trace_compare._newton_distance`` re-runs the loop in
    NumPy float64 while the kernel and the per-op path run it in the MODE's
    arithmetic, so ``conv = |F| < tol``, ``near = |dF/dt| <= tau`` and "is the
    iterate finite" can land differently from identical inputs.  Two
    mechanisms, both measured (df64 only; all 64 draws agree exactly in sf64,
    and tier A is green on every draw in both modes):

    * ``threshold`` (draw 14) -- ``max_iter = 0``, so the
      ``ST_NEWTON_NOT_CONVERGED`` bit IS the comparison ``|F(t_seed)| < tol``.
      Four rays have ``|F|`` between 1.7e-14 and 2.4e-13 against
      ``tol = 1e-12``, and the df64 conic SEED lands 1.16e-12 mm from the
      float64 one (48 ULP relative, because ``sqrt(b*b - 4*a*c)`` is
      ill-conditioned on a near-tangent ray), which moves the residual past
      ``tol``.
    * ``float32-range`` (draw 9) -- the oracle's own iterates reach 4.3e143,
      past float32's 3.4e38, so the df64 iterate is an infinity and NaN one
      step later (the R1-V2-02 break) while float64 keeps stepping: ``iters``
      2 against 6 on 128 rays, plus a spurious ``ST_DF_FLOORED``.

    Both are now marked ``newton_uncertain`` and excluded -- from ``iters``
    and from the three bits the loop raises -- in the prediction and in the
    measurement alike.  Everything decided outside the loop (``MISS``,
    ``CLIPPED``, ``TIR``, ``TOL_CROSSOVER``), every other ray and every raw
    recorded word under tier A stay exactly as strict as before, and in sf64
    the band is empty by construction.
    """
    metal.set_mode(mode)
    prediction, status, iters, (got, ref) = newton_prediction_and_planes(
        build, wavelength, monkeypatch, mode
    )
    tc.assert_tier_a(got, ref, f"r1v107_{draw}[{mode}]")

    masked_status = tc.mask_uncertain(status, prediction)
    masked_want = tc.mask_uncertain(prediction.bits, prediction)
    bad = np.argwhere(masked_status != masked_want)
    assert bad.size == 0, (
        f"[{mode}] {draw}: masked status differs at {len(bad)} (surface, ray) "
        f"entries; first {bad[:5].tolist()}"
    )
    bad = np.argwhere(
        tc.mask_iters(iters, prediction) != tc.mask_iters(prediction.iters, prediction)
    )
    assert bad.size == 0, (
        f"[{mode}] {draw}: masked iters differs at {len(bad)} entries; "
        f"first {bad[:5].tolist()}"
    )

    raw_bits = np.argwhere(status != prediction.bits)
    raw_iters = np.argwhere(iters != prediction.iters)
    banded = int(prediction.newton_uncertain.sum())
    if mode == "df64":
        # The certificate that the band is doing work: without it the raw
        # planes still disagree, which is what the finding measured.
        assert raw_bits.size or raw_iters.size, (
            f"df64 {draw}: the oracle and the kernel now agree raw, so this "
            "draw no longer exercises the band and the test proves nothing"
        )
        assert banded > 0
        for s, i in [tuple(e) for e in raw_bits] + [tuple(e) for e in raw_iters]:
            assert bool(prediction.newton_uncertain[s, i]), (
                f"df64 {draw}: (surface {s}, ray {i}) disagrees with the "
                "oracle and is NOT inside the band"
            )
    else:
        assert raw_bits.size == 0 and raw_iters.size == 0, (
            f"sf64 is binary64: the oracle must reproduce the kernel's Newton "
            f"verdict exactly; {raw_bits.size} bits and {raw_iters.size} iters "
            "entries differ"
        )
        assert banded == 0, (
            f"sf64: the Newton band must be empty by construction, {banded} entries"
        )


#: The shipped Newton fixtures this control sweeps, with the bundle each one
#: is traced with in plan 7.2 (a fixture that ships its own rays factory
#: returns ``(optic, factory)``; the other two take a collimated bundle).
NEWTON_FIXTURES: tuple[str, ...] = (
    "aspheric_singlet",
    "odd_asphere_singlet",
    "even_asphere_5coeff",
    "even_asphere_inf_radius",
    "nonconverging_asphere",
)


def newton_fixture(name: str) -> tuple[Any, Any]:
    """``(optic, rays)`` for one shipped Newton fixture."""
    built = getattr(fx, name)()
    if isinstance(built, tuple):
        optic, factory = built
        return optic, factory(optic, N_RAYS)
    return built, fx.collimated_bundle(N_RAYS, radius=8.0, z=-10.0, L=0.3)


@pytest.mark.parametrize("fixture", NEWTON_FIXTURES)
@pytest.mark.parametrize("mode", MODES)
def test_r1v107_newton_band_is_empty_on_the_shipped_fixtures(
    mps_backend, monkeypatch, mode, fixture
):
    """The band may not open where the tolerance is above the round-off.

    The control for the test above, over every shipped Newton fixture of plan
    7.2 -- including ``nonconverging_asphere``, whose ``max_iter = 1`` cap is
    the one fixture that raises ``ST_NEWTON_NOT_CONVERGED`` on purpose.  They
    keep the shipped tolerances (1e-10 and the factory's 1e-6), three to seven
    orders above the band's ~3.7e-12 mm at these scales, so
    ``newton_uncertain`` must be EMPTY and the loop's verdict must be
    predicted exactly, ray for ray, in both modes.  The band only ever opens
    where a caller sets a tolerance at or below the arithmetic's own
    resolution -- which is what plan section 6's attack items 12/13/14 do.
    """
    metal.set_mode(mode)
    optic_ref, launch = newton_fixture(fixture)
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    out = quiet_trace(optic_ref.surfaces, tc.copy_rays(launch))
    rows = tc.decoded_rows(optic_ref.surfaces)
    w0 = float(np.asarray(tc.decode(out.w)).reshape(-1)[0])
    records = tc.compile_tables(optic_ref, mode, w0)
    prediction = tc.predict_status(rows, optic_ref, mode=mode, records=records)

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    monkeypatch.setenv("OPTILAND_METAL_TRACE_DIAG", "1")
    optic_f, _ = newton_fixture(fixture)
    before = dict(T.stats())
    quiet_trace(optic_f.surfaces, tc.copy_rays(launch))
    delta = stats_delta(before)
    assert delta.get("fused_trace:traces", 0) == 1, f"{fixture}: {delta}"
    planes = trace.diag_from(optic_f.surfaces)
    assert planes is not None
    status = planes[0].cpu().numpy()[0]
    iters = planes[1].cpu().numpy()[0]

    assert int(prediction.newton_uncertain.sum()) == 0, (
        f"[{mode}] {fixture}: the Newton band covers "
        f"{int(prediction.newton_uncertain.sum())} entries on a shipped "
        "tolerance; it is too wide"
    )
    assert int(prediction.iters.max()) > 0, (
        f"{fixture}: nothing iterated, so this control checks nothing"
    )
    assert np.array_equal(iters, prediction.iters), (
        f"[{mode}] {fixture}: iters differ at "
        f"{int(np.count_nonzero(iters != prediction.iters))} entries"
    )
    assert np.array_equal(
        tc.mask_uncertain(status, prediction),
        tc.mask_uncertain(prediction.bits, prediction),
    )
