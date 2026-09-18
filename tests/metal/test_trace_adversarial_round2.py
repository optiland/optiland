"""Regression and lock tests closing verify-round 2 findings.

WP0 creates this file with one sanity test so the definition-of-done path
exists even for a round that produces no findings.  The wave-3 fix lane for
round 2 owns it from then on: every finding it closes lands here as a named
regression test (plan 0.1, 4/WP0), or as a documented-limit lock test.

Never widen a tolerance to make a test here pass (plan 0.2.2).

Round 2, iteration 1 closed four findings
(``NOTES/fused-trace-research/verify-round-2.md``), all by fix + named
regression test:

============ ============================ ==================================
finding      closure                      test
============ ============================ ==================================
R2-V1-03     fix (kernel: operand side)   ``test_r2v103_inexact_conic_form``,
                                          ``test_r2v103_exact_conic_is_unmoved``,
                                          ``test_r2v103_inexact_radius_form``,
                                          ``test_r2v103_operand_side_bits_
                                          track_the_storage_form``,
                                          ``test_r2v103_matrix_row_is_
                                          inexact_and_right_sided``
R2-V1-04     fix (batch API: no_grad)     ``test_r2v104_trace_batch_legs_
                                          run_without_autograd``,
                                          ``test_r2v104_batch_equals_its_
                                          loop_on_a_newton_system``
R2-V1-01     fix (batch API: structural)  ``test_r2v101_nan_radius_
                                          decision_is_the_same``
R2-V1-02     fix (batch API: image row)   ``test_r2v102_rays_needs_the_
                                          real_image_row``,
                                          ``test_r2v102_image_index_is_
                                          carried_not_derived``
============ ============================ ==================================

Round 2, iteration 2 closed three more, two by fix + named regression test and
one as a documented limit (the only closure open to a fix lane for it -- see
the test's own docstring and
``NOTES/fused-trace-research/documented-limits.md``):

============ ============================ ==================================
finding      closure                      test
============ ============================ ==================================
R2-V1-05     fix (batch API: the late     ``test_r2v105_late_fallback_
             fallback re-traces the       equals_the_contract_loop``,
             caller's optic, not an       ``test_r2v105_retraced_rows_come_
             ``Optic.from_dict`` copy)    from_the_callers_optic``,
                                          ``test_r2v105_the_retrace_leaves_
                                          the_callers_records_alone``
R2-V1-07     fix (batch API: the          ``test_r2v107_late_fallback_
             re-traced design's status    planes_describe_their_rows``
             and iters are zeroed)
R2-V1-06     DOCUMENTED LIMIT + lock      ``test_r2v106_locked``
             (hook path, autograd merely
             enabled, Newton geometries)
============ ============================ ==================================

Round 2, iteration 3 closed three more -- two by fix + named regression test
(both about the state a batch leaves on the caller's optic, neither a
fused-vs-loop divergence) and one as a documented limit:

============ ============================ ==================================
finding      closure                      test
============ ============================ ==================================
R2-V1-08     fix (batch API: the optic is ``test_r2v108_thickness_survives_
             restored by reference, not   a_batch``
             by replaying values)
R2-V1-09     fix (same restore: the       ``test_r2v109_material_survives_
             caller's ``Material`` object a_batch``,
             comes back, dispersion and   ``test_r2v109_index_designs_use_
             absorption with it)          ideal_glass_like_the_loop``
R2-V1-10     DOCUMENTED LIMIT + lock      ``test_r2v110_locked``
             (tolerancing frames are
             tier B at N <= 1024)
============ ============================ ==================================

Every comparison is plan 7.1 **tier A**: raw components (df64 hi/lo float32
words, sf64 int64 bit patterns), ``equal_nan=True``, bundles above 1024 rays --
with exactly one deliberate exception, ``test_r2v110_locked``, whose whole
subject is the 469-ray tier-B site and which therefore asserts tier A FAILS
there and plan 7.1's tier-B bound holds. No tolerance is widened anywhere.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

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
from optiland.backend.torch_backend.metal import trace  # noqa: E402
from optiland.backend.torch_backend.metal import trace_layout as L  # noqa: E402
from optiland.backend.torch_backend.metal.trace_adapters import (  # noqa: E402
    GEOMETRY_ADAPTERS,
    is_backend_scalar,
)
from optiland.optimization.variable import Variable  # noqa: E402
from optiland.raytrace import batch_trace as BT  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import trace_fixtures as fx  # noqa: E402

from tests.metal import _trace_compare as tc  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator

MODES = tc.MODES

#: 19 hexapolar rings = 1,141 rays: above 1024, so every comparison is tier A.
RINGS = 19
N_BATCH_RAYS = 1 + 3 * RINGS * (RINGS + 1)

#: The bundle size the plain (non-batch) traces here use.
N_RAYS = fx.DEFAULT_RAYS

BATCH_KWARGS = {
    "Hx": 0.0,
    "Hy": 0.0,
    "wavelength": 0.55,
    "num_rays": RINGS,
    "distribution": "hexapolar",
}

#: Conic constants whose ``1 + k`` is NOT float32-exact, so ``SR_K1``'s df64
#: low word is non-zero and the operand side of ``(1 + k) * r2`` is visible.
INEXACT_CONICS = (-0.4, -0.3)

#: The float32-exact control: ``1 + (-0.5) = 0.5`` has a zero low word, so the
#: two operand orders collapse and nothing can differ either way.
EXACT_CONIC = -0.5


def test_round_file_present():
    """Placeholder so round 2 always has a collectable test file."""
    assert True


# ---------------------------------------------------------------------------
# Environment and backend (same contract as test_trace_adversarial_round1.py)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch) -> None:
    """No fused-trace switch leaks into or out of a test."""
    for name in (
        "OPTILAND_METAL_FUSED_TRACE",
        "OPTILAND_METAL_FUSED_TRACE_MAX_STEPS",
        "OPTILAND_METAL_FUSED_TRACE_MEMORY_FRACTION",
        "OPTILAND_METAL_FUSED_TRACE_MIN_RAYS",
        "OPTILAND_METAL_FUSED_TRACE_DRIFT",
        "OPTILAND_METAL_TRACE_DIAG",
    ):
        monkeypatch.delenv(name, raising=False)


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
    """torch / mps / float64 with autograd off -- the fused path's setting."""
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


def quiet_trace(group: Any, rays: Any) -> Any:
    """``group.trace(rays)`` with the driver's RuntimeWarnings suppressed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return group.trace(rays)


def raw_differences(got: tc.Capture, ref: tc.Capture) -> list[tuple]:
    """Every recorded raw word of ``got`` that differs from ``ref``.

    Same predicate as :func:`tc.assert_tier_a` (NaN equals NaN, every hi/lo word
    or int64 pattern compared), but it *returns* the differences, so a test can
    use "they differ" as the certificate that a case is live and "they do not"
    as the closure.
    """
    out: list[tuple] = []
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
    """Trace ``rays`` on a fresh ``build()`` with the hook OFF."""
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    optic = build()
    out = quiet_trace(optic.surfaces, rays)
    return tc.capture(optic.surfaces, out, mode)


def hooked(build, rays, monkeypatch, mode):
    """Trace ``rays`` on a fresh ``build()`` with the hook ON; + counter delta."""
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    optic = build()
    before = dict(T.stats())
    out = quiet_trace(optic.surfaces, rays)
    now = T.stats()
    delta = {
        key: now.get(key, 0) - before.get(key, 0)
        for key in now
        if key.startswith("fused_trace") and now.get(key, 0) - before.get(key, 0) != 0
    }
    return tc.capture(optic.surfaces, out, mode), delta


# ---------------------------------------------------------------------------
# R2-V1-03 -- the operand side of the two conic products
#
# `(1 + self.k) * r2` and `self.radius * (1 + sqrt(..))` are the only two
# expressions in `standard.py` / `even_asphere.py` / `odd_asphere.py` whose
# LAUNCHED kernel variant depends on how the geometry stores its scalar:
# a backend array keeps the slot on the left, a Python/NumPy number swaps it to
# the right through the reflected dunder.  df64's mul adds `a.hi*b.lo` before
# `a.lo*b.hi`, so the two orders differ in the low word whenever both operands
# have a non-zero low word.  `OpticUpdater.set_conic` assigns the raw value, so
# every `set_conic` and every conic `Variable.update` produces the right-sided
# form -- which the kernel used to mirror as if it were the left-sided one.
# ---------------------------------------------------------------------------


def newton_singlet(conic: float, form: str, radius_form: str = "constructor") -> Any:
    """An even-asphere singlet whose conic (and radius) storage form is chosen.

    ``form`` is ``"constructor"`` (``be.array`` -- a backend scalar, what
    ``StandardGeometry.__init__`` produces), ``"float"`` (a Python float, what
    ``OpticUpdater.set_conic`` assigns), ``"numpy"`` (``numpy.float64``, what a
    batch value array hands the updater) or ``"variable"`` (through the public
    ``Variable.update`` path).
    """
    optic = fx._build(
        [
            {"radius": be.inf, "thickness": be.inf},
            {
                "surface_type": "even_asphere",
                "radius": 25.0,
                "thickness": 6.0,
                "material": "N-BK7",
                "is_stop": True,
                "conic": conic,
                "coefficients": [-1.0e-4, 2.0e-6, -5.0e-8],
            },
            {"radius": -35.0, "thickness": 45.0},
            {},
        ],
        epd=14.0,
    )
    geometry = optic.surfaces.surfaces[1].geometry
    if form == "constructor":
        pass  # `_build` already wrapped it in be.array
    elif form == "float":
        optic.updater.set_conic(float(conic), 1)
    elif form == "numpy":
        optic.updater.set_conic(np.float64(conic), 1)
    elif form == "variable":
        variable = Variable(optic, "conic", surface_number=1)
        variable.update(variable.variable.scale(float(conic)))
    else:  # pragma: no cover - programming error
        raise ValueError(form)
    if radius_form == "float":
        # Not reachable through `OpticUpdater.set_radius` (it goes through
        # `StandardGeometry.set_radius`, which re-wraps in `be.array`); this is
        # the twin of the conic case and it pins the second bit.
        geometry.radius = float(be.to_numpy(geometry.radius))
    elif radius_form != "constructor":  # pragma: no cover - programming error
        raise ValueError(radius_form)
    return optic


CONIC_FORMS = ("float", "numpy", "variable")


@pytest.mark.parametrize("form", CONIC_FORMS)
@pytest.mark.parametrize("conic", INEXACT_CONICS)
@pytest.mark.parametrize("mode", MODES)
def test_r2v103_inexact_conic_form(mps_backend, monkeypatch, mode, conic, form):
    """A Newton row with an inexact conic fuses bit for bit, in every form.

    Finding R2-V1-03, in the certificate form the finding requires.

    * **Certificate** -- in df64 the per-op path itself MOVES between the
      backend-array form and the ``form`` under test, because the two launch
      different kernel variants of ``(1 + k) * r2``.  Without that the closure
      below would prove nothing.  In sf64 it must NOT move: sf64 is correctly
      rounded, hence commutative, which is why the bug was df64-only.
    * **Closure** -- the fused trace equals the per-op trace of the SAME optic,
      tier A, in both modes and in all three value forms, with the kernel
      actually running (``fused_trace:traces == 1``, no refusal, no late
      fallback).
    """
    metal.set_mode(mode)
    launch = fx.pupil_bundle(newton_singlet(conic, "float"), N_RAYS, Hx=0.0, Hy=0.0)

    as_backend_scalar = perop(
        lambda: newton_singlet(conic, "constructor"),
        tc.copy_rays(launch),
        monkeypatch,
        mode,
    )
    as_form = perop(
        lambda: newton_singlet(conic, form), tc.copy_rays(launch), monkeypatch, mode
    )

    sensitivity = raw_differences(as_form, as_backend_scalar)
    if mode == "df64":
        assert sensitivity, (
            f"df64 k={conic} form={form}: the certificate is empty -- the "
            "per-op path does not move with the storage form, so this test "
            "would prove nothing about the kernel"
        )
    else:
        assert not sensitivity, (
            f"sf64 k={conic} form={form}: sf64 mul is correctly rounded and "
            f"therefore commutative, so the per-op path must not move; "
            f"{len(sensitivity)} words differ, first {sensitivity[:3]}"
        )

    got, delta = hooked(
        lambda: newton_singlet(conic, form), tc.copy_rays(launch), monkeypatch, mode
    )
    assert delta.get("fused_trace:traces", 0) == 1, delta
    assert delta.get("fused_trace:late_fallback", 0) == 0, delta
    assert not [k for k in delta if k.startswith("fused_trace_skip:")], delta
    tc.assert_tier_a(got, as_form, f"r2v103_conic[{mode}][k={conic}][{form}]")


@pytest.mark.parametrize("mode", MODES)
def test_r2v103_exact_conic_is_unmoved(mps_backend, monkeypatch, mode):
    """The float32-exact control: ``k = -0.5`` cannot see the operand side.

    ``1 + (-0.5) = 0.5`` has a zero df64 low word, so ``mul(K1, r2)`` and
    ``mul(r2, K1)`` collapse to the same float32 sum.  Both the per-op path and
    the fused path must therefore be word-identical across the storage forms --
    before AND after the R2-V1-03 fix.  This is the row that would catch a
    "fix" that changed the arithmetic instead of only its operand order.
    """
    metal.set_mode(mode)
    launch = fx.pupil_bundle(
        newton_singlet(EXACT_CONIC, "float"), N_RAYS, Hx=0.0, Hy=0.0
    )
    reference = perop(
        lambda: newton_singlet(EXACT_CONIC, "constructor"),
        tc.copy_rays(launch),
        monkeypatch,
        mode,
    )
    for form in CONIC_FORMS:
        other = perop(
            lambda f=form: newton_singlet(EXACT_CONIC, f),
            tc.copy_rays(launch),
            monkeypatch,
            mode,
        )
        assert not raw_differences(other, reference), (
            f"[{mode}] a float32-exact conic must be blind to the storage "
            f"form, but {form} moved the per-op path"
        )
        got, delta = hooked(
            lambda f=form: newton_singlet(EXACT_CONIC, f),
            tc.copy_rays(launch),
            monkeypatch,
            mode,
        )
        assert delta.get("fused_trace:traces", 0) == 1, delta
        tc.assert_tier_a(got, reference, f"r2v103_exact[{mode}][{form}]")


@pytest.mark.parametrize("mode", MODES)
def test_r2v103_inexact_radius_form(mps_backend, monkeypatch, mode):
    """The radius twin of the conic case, with its own certificate.

    ``self.radius * (1 + be.sqrt(..))`` in ``sag`` and ``self.radius *
    be.sqrt(..)`` in ``_surface_normal`` swap sides with ``geometry.radius``'s
    storage form exactly as ``(1 + k) * r2`` does with ``geometry.k``'s, and
    ``FL_R_ON_RIGHT`` records it.  ``OpticUpdater.set_radius`` re-wraps in
    ``be.array`` so the public updater never reaches this form, but a direct
    attribute assignment does -- and the effect is much larger than the conic
    one (measured on the sag/normal probes at 4096 points: 457 raw words
    against the conic case's 8).
    """
    metal.set_mode(mode)
    inexact = 25.3  # not float32-exact
    optic = newton_singlet(EXACT_CONIC, "constructor")
    optic.surfaces.surfaces[1].geometry.radius = be.array(inexact)
    launch = fx.pupil_bundle(optic, N_RAYS, Hx=0.0, Hy=0.0)

    def build(radius_form: str) -> Any:
        built = newton_singlet(EXACT_CONIC, "constructor")
        geometry = built.surfaces.surfaces[1].geometry
        geometry.radius = (
            float(inexact) if radius_form == "float" else be.array(inexact)
        )
        return built

    reference = perop(
        lambda: build("constructor"), tc.copy_rays(launch), monkeypatch, mode
    )
    as_float = perop(lambda: build("float"), tc.copy_rays(launch), monkeypatch, mode)

    sensitivity = raw_differences(as_float, reference)
    if mode == "df64":
        assert sensitivity, (
            "df64: the certificate is empty -- the per-op path does not move "
            "with the radius storage form, so this test would prove nothing"
        )
    else:
        assert not sensitivity, (
            f"sf64 must not move with the storage form; {len(sensitivity)} "
            f"words differ, first {sensitivity[:3]}"
        )

    got, delta = hooked(lambda: build("float"), tc.copy_rays(launch), monkeypatch, mode)
    assert delta.get("fused_trace:traces", 0) == 1, delta
    tc.assert_tier_a(got, as_float, f"r2v103_radius[{mode}]")


def _geometry_row(optic: Any, index: int = 1) -> np.ndarray:
    """``GeometryAdapter.fill``'s ``surf_int`` row for one surface."""
    geometry = optic.surfaces.surfaces[index].geometry
    row_int = np.zeros(L.SI_STRIDE, dtype=np.int32)
    row_real = np.zeros(L.SR_STRIDE, dtype=np.float64)
    row_coef = np.zeros(max(1, len(getattr(geometry, "coefficients", []))))
    GEOMETRY_ADAPTERS[type(geometry)].fill(geometry, row_int, row_real, row_coef, {})
    return row_int, row_real


@pytest.mark.parametrize("form", ("constructor", *CONIC_FORMS))
def test_r2v103_operand_side_bits_track_the_storage_form(mps_backend, form):
    """``FL_K1_ON_RIGHT`` / ``FL_R_ON_RIGHT`` equal the storage-form predicate.

    The bits are the whole mechanism, so they are asserted directly and not
    only through their effect: set exactly when the geometry holds a
    Python/NumPy number, clear exactly when it holds a backend array.  A future
    upstream change that makes ``set_conic`` wrap in ``be.array`` (or that stops
    the constructor doing so) flips a bit here instead of silently
    un-mirroring the kernel.
    """
    metal.set_mode("df64")
    for radius_form in ("constructor", "float"):
        optic = newton_singlet(INEXACT_CONICS[0], form, radius_form=radius_form)
        geometry = optic.surfaces.surfaces[1].geometry
        row_int, _ = _geometry_row(optic)
        flags = int(row_int[L.SI_FLAGS])
        assert bool(flags & L.FL_K1_ON_RIGHT) == (not is_backend_scalar(geometry.k)), (
            f"conic form {form}: type(k) is {type(geometry.k).__name__}"
        )
        assert bool(flags & L.FL_R_ON_RIGHT) == (
            not is_backend_scalar(geometry.radius)
        ), f"radius form {radius_form}: {type(geometry.radius).__name__}"
        # The constructor form must leave BOTH clear: that is the state every
        # pre-fix fixture was in, and it is what keeps the matrix bit-identical.
        if form == "constructor" and radius_form == "constructor":
            assert not flags & (L.FL_K1_ON_RIGHT | L.FL_R_ON_RIGHT)


def test_r2v103_matrix_row_is_inexact_and_right_sided(mps_backend):
    """``inexact_conic_asphere`` really is the row plan 7.2 was missing.

    Before round 2 every Newton row in the conformance matrix had a
    float32-exact conic (``aspheric_singlet`` k = 0, ``even_asphere_5coeff``
    k = -0.5, ``even_asphere_inf_radius`` k = 0, ``long_path_asphere`` k = 0),
    so ``SR_K1``'s df64 low word was exactly zero in every end-to-end Newton
    trace the suite ran and neither the association nor the operand side of
    ``(1 + k) * r2`` could be observed.  This pins that the new fixture fixes
    both holes, so an innocent-looking edit to its conic cannot quietly reopen
    them -- and it pins that the OLD rows are still exact, since that is the
    property that makes them the control.
    """
    from optiland.backend.torch_backend.metal import encode

    metal.set_mode("df64")
    optic, _ = fx.inexact_conic_asphere()
    geometry = optic.surfaces.surfaces[1].geometry
    row_int, row_real = _geometry_row(optic)

    hi, lo = encode.encode_df64(np.array(row_real[L.SR_K1]))
    assert float(lo) != 0.0, (
        f"SR_K1 = {row_real[L.SR_K1]!r} is float32-exact (low word {float(lo)}), "
        "so this fixture cannot see an operand-side or association error"
    )
    assert float(hi) != 0.0
    assert not is_backend_scalar(geometry.k), (
        "the fixture must keep the Python-number conic form `set_conic` leaves"
    )
    assert int(row_int[L.SI_FLAGS]) & L.FL_K1_ON_RIGHT

    exact_rows = {
        "aspheric_singlet": fx.aspheric_singlet(),
        "even_asphere_5coeff": fx.even_asphere_5coeff()[0],
        "even_asphere_inf_radius": fx.even_asphere_inf_radius()[0],
    }
    for name, built in exact_rows.items():
        _, real = _geometry_row(built)
        _, low = encode.encode_df64(np.array(real[L.SR_K1]))
        assert float(low) == 0.0, (
            f"{name}'s conic became inexact; it is one of the controls that "
            "make the new row's certificate meaningful"
        )


# ---------------------------------------------------------------------------
# R2-V1-04 -- trace_batch's own reference took the DiffOptics Newton branch
# ---------------------------------------------------------------------------


def batch_conic_case() -> tuple[Any, list, np.ndarray]:
    """``aspheric_singlet`` with a conic variable over five designs.

    A FINITE-radius Newton system: the family whose grad branch R2-V1-04 is
    about, and the one ``test_trace_batch.py`` avoided before the fix.
    """
    optic = fx.aspheric_singlet()
    variable = Variable(optic, "conic", surface_number=1)
    values = np.array([[variable.variable.scale(k)] for k in (-0.4, -0.2, 0.0, 0.2)])
    return optic, [variable], values


@pytest.mark.parametrize("mode", MODES)
def test_r2v104_trace_batch_legs_run_without_autograd(mps_backend, monkeypatch, mode):
    """Both ``trace_batch`` legs trace with ``torch.is_grad_enabled()`` False.

    Finding R2-V1-04.  ``NewtonRaphsonGeometry.distance`` returns the DiffOptics
    one-step correction ``t - F(t)/(dF/dt)`` instead of the primal ``result.t``
    whenever autograd is merely ENABLED -- no tensor needs ``requires_grad``, so
    the gate does not refuse -- and the kernel mirrors the primal solve.  The
    spy is on the two leg entry points rather than on the arithmetic, so it
    states the property the module docstring claims.

    ``be.grad_mode.disable()`` is deliberately NOT used to arrange the "before"
    state: it does not clear ``torch.is_grad_enabled()`` (round-2 observation
    3), which is the flag ``newton_raphson.py`` branches on.
    """
    metal.set_mode(mode)
    seen: dict[str, list[bool]] = {"fused": [], "loop": []}
    for key, name in (("fused", "_trace_batch_fused"), ("loop", "_trace_batch_loop")):
        original = getattr(BT, name)

        def spy(*args, _o=original, _k=key, **kwargs):
            seen[_k].append(torch.is_grad_enabled())
            return _o(*args, **kwargs)

        monkeypatch.setattr(BT, name, spy)

    optic, variables, values = batch_conic_case()
    with torch.enable_grad():
        assert torch.is_grad_enabled(), "the 'before' state must have autograd on"
        BT.trace_batch(optic, variables, values, record="all", **BATCH_KWARGS)
        # The caller's grad state is restored, not silently left off.
        assert torch.is_grad_enabled()

    assert seen["fused"] == [False], seen
    # The loop leg only runs when the fused one refuses; force it and re-check.
    seen["fused"].clear()
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    optic, variables, values = batch_conic_case()
    with torch.enable_grad():
        BT.trace_batch(optic, variables, values, record="all", **BATCH_KWARGS)
    assert seen["loop"] == [False], seen
    assert seen["fused"] == [], "the switch was off; the fused leg must not run"


@pytest.mark.parametrize("mode", MODES)
def test_r2v104_batch_equals_its_loop_on_a_newton_system(
    mps_backend, monkeypatch, mode
):
    """Fused ``trace_batch`` equals its contract loop under the shipped grad state.

    The comparison is made with autograd ENABLED around the call, which is the
    library's default (``OptimizationProblem.__init__`` and ``Tolerancing`` both
    turn it on) and the state the finding was measured in.  Before the fix the
    fused rows and the loop rows disagreed in the df64 low word on every finite
    -radius Newton system (``aspheric_singlet`` 160/1141 rays at ``s1.z``).

    The certificate that the branch is live at all is asserted first: a plain
    per-op trace with autograd on must differ from the same trace under
    ``torch.no_grad()``.  It uses ``backward_newton``, the one Newton fixture
    whose grad branch moves the recorded words in BOTH modes (measured 1/1141
    rays at ``s1.z``); ``aspheric_singlet``'s own divergence is df64-only, so it
    could not certify the sf64 leg.  If upstream ever removes the grad branch
    the certificate empties and this test says so instead of passing vacuously.
    """
    metal.set_mode(mode)

    def build_backward():
        built, _ = fx.backward_newton()
        return built

    witness, rays_factory = fx.backward_newton()
    launch = rays_factory(witness, N_RAYS)
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    with torch.no_grad():
        primal = perop(build_backward, tc.copy_rays(launch), monkeypatch, mode)
    with torch.enable_grad():
        corrected = perop(build_backward, tc.copy_rays(launch), monkeypatch, mode)
    assert raw_differences(corrected, primal), (
        f"[{mode}] the certificate is empty: NewtonRaphsonGeometry.distance no "
        "longer branches on torch.is_grad_enabled(), so this test would prove "
        "nothing"
    )

    def run(switch: str) -> Any:
        monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", switch)
        built, variables, values = batch_conic_case()
        with torch.enable_grad():
            return BT.trace_batch(
                built, variables, values, record="all", **BATCH_KWARGS
            )

    fused = run("1")
    assert fused.fused is True, f"the kernel did not run: {fused.stats}"
    loop = run("0")
    assert loop.fused is False
    assert fused.rows == loop.rows
    for attr in ("x", "y", "z", "L", "M", "N", "intensity", "opd"):
        got = [c for c in _components(getattr(fused, attr))]
        ref = [c for c in _components(getattr(loop, attr))]
        for c, (g, r) in enumerate(zip(got, ref, strict=True)):
            assert np.array_equal(g, r, equal_nan=True), (
                f"[{mode}] {attr} component {c} differs on "
                f"{int((~((g == r) | (np.isnan(g) & np.isnan(r)))).sum())} cells"
            )


def _components(plane: Any) -> list[np.ndarray]:
    """The raw component arrays of one plane, as NumPy."""
    if type(plane).__name__ == "MetalFloat64":
        return [c.detach().contiguous().cpu().numpy() for c in plane.components]
    return [np.asarray(plane)]


# ---------------------------------------------------------------------------
# R2-V1-01 -- a NaN radius: one raise/not-raise decision per call
# ---------------------------------------------------------------------------


def nan_radius_case(bad: float) -> tuple[Any, list, np.ndarray]:
    """A Cooke triplet whose surface-5 radius goes finite, finite, ``bad``."""
    optic = fx.cooke()
    variable = Variable(optic, "radius", surface_number=5)
    scale = variable.variable.scale
    return (
        optic,
        [variable],
        np.array([[scale(79.0)], [scale(80.0)], [scale(bad)]]),
    )


@pytest.mark.parametrize("bad", (float("nan"), float("inf")))
@pytest.mark.parametrize("mode", MODES)
def test_r2v101_nan_radius_decision_is_the_same(mps_backend, monkeypatch, mode, bad):
    """One call, one raise/not-raise decision, whatever the switch.

    Finding R2-V1-01.  ``_structural_signature`` records
    ``math.isfinite(float(radius))``, which is False for NaN and for +-inf
    alike, while the fused path's ``_structural_check`` compared only the
    compiled ``SI_GEOM`` -- and ``_fill_conic`` splits ``GEOM_CONIC`` from
    ``GEOM_STD_INF`` with ``be.isinf``, which leaves a NaN radius in
    ``GEOM_CONIC``.  A design whose radius became NaN therefore raised on the
    contract loop and traced silently on the fused path, in both modes, against
    ``_structural_signature``'s own docstring ("``trace_batch`` raises the same
    ``ValueError`` on NumPy as on Metal").  The ``+inf`` twin is the control:
    it already agreed, and it must keep agreeing -- there the fused path
    diagnoses the same edit through ``SI_GEOM`` (``GEOM_CONIC`` ->
    ``GEOM_STD_INF``), so the two messages name different slots of the same
    surface; the decision, which is what the finding is about, is identical.
    """
    metal.set_mode(mode)
    messages = {}
    for switch in ("1", "0"):
        monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", switch)
        optic, variables, values = nan_radius_case(bad)
        with pytest.raises(ValueError) as excinfo:
            BT.trace_batch(optic, variables, values, record="image", **BATCH_KWARGS)
        messages[switch] = str(excinfo.value)
    for switch, text in messages.items():
        assert "design 2" in text and "surface 5" in text, (switch, text)
        assert "not structural edits" in text, (switch, text)
    if np.isnan(bad):
        # The fix: both paths reach the SAME slot, so the message is identical.
        assert "radius finiteness of surface 5" in messages["0"], messages["0"]
        assert messages["1"] == messages["0"], messages
    else:
        assert "radius finiteness of surface 5" in messages["0"], messages["0"]
        assert "geometry code of surface 5" in messages["1"], messages["1"]

    # The finite control: nothing raises and the kernel runs.
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    optic, variables, values = nan_radius_case(81.0)
    result = BT.trace_batch(optic, variables, values, record="image", **BATCH_KWARGS)
    assert result.fused is True, result.stats


# ---------------------------------------------------------------------------
# R2-V1-02 -- rays(b) with the image row absent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_r2v102_rays_needs_the_real_image_row(mps_backend, monkeypatch, mode):
    """``rays(b)`` returns the image, or raises -- never the highest row.

    Finding R2-V1-02.  ``_image_index()`` was ``max(self.rows)``, so
    ``self.rows.get(self._image_index())`` was never ``None`` while ``rows`` was
    non-empty: with ``record=[1, 5], write_final=False`` on an 8-surface Cooke,
    ``rays(0)`` handed back surface 5's state as if it were the image (asserted
    below to be bit-identical to the surface-5 row) and then applied the LAST
    surface's trailing propagation to it, while the ``ValueError`` its own
    docstring advertises was unreachable.
    """
    metal.set_mode(mode)
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    optic, variables, values = nan_radius_case(81.0)
    image = len(optic.surfaces.surfaces) - 1
    assert image == 7, "the fixture must be the 8-surface Cooke triplet"

    result = BT.trace_batch(
        optic, variables, values, record=[1, 5], write_final=False, **BATCH_KWARGS
    )
    assert result.n_rays == N_BATCH_RAYS  # tier-A bundle size
    assert result.rows == {1: 0, 5: 1}
    assert result.image_index == image
    with pytest.raises(ValueError) as excinfo:
        result.rays(0)
    assert f"image row ({image})" in str(excinfo.value), str(excinfo.value)
    assert "[1, 5]" in str(excinfo.value)

    # With the image row recorded it still works, and it really is that row.
    optic, variables, values = nan_radius_case(81.0)
    recorded = BT.trace_batch(
        optic, variables, values, record=[1, image], write_final=False, **BATCH_KWARGS
    )
    assert recorded.image_index == image
    rays = recorded.rays(0)
    row = recorded.rows[image]
    want = _components(BT._select(recorded.x, 0, row))
    # `rays()` applies the trailing propagation, which moves z but not x.
    got = _components(rays.x)
    for g, r in zip(got, want, strict=True):
        assert np.array_equal(g, r, equal_nan=True)

    # The default path stays safe: `_resolve_write_final` turns write_final on
    # when the image row is absent, so `rays(b)` reads the final planes.
    optic, variables, values = nan_radius_case(81.0)
    default = BT.trace_batch(optic, variables, values, record=[1, 5], **BATCH_KWARGS)
    assert default.final is not None
    assert default.rays(0).x is not None


def test_r2v102_image_index_is_carried_not_derived():
    """``_image_index`` reads the carried index, never ``max(self.rows)``.

    A pure unit assertion on the dataclass, so the mechanism is pinned even if
    every GPU test is skipped: a result that records rows 1 and 5 of an
    8-surface system reports 7, and a hand-built result that carries no index
    reports -1 (which makes ``rays`` raise) rather than the highest row.
    """
    result = BT.BatchTraceResult(rows={1: 0, 5: 1}, image_index=7)
    assert result._image_index() == 7
    assert BT.BatchTraceResult(rows={1: 0, 5: 1})._image_index() == -1


# ---------------------------------------------------------------------------
# R2-V1-05 -- the per-design late fallback re-traces the CALLER'S optic
#
# `_retrace_designs` traced `Optic.from_dict(optic.to_dict())`.  A dict round
# trip re-wraps every geometry scalar in `be.array`, which normalises exactly
# the storage form R2-V1-03 measured the df64 operand side on, so a re-traced
# design's rows were a per-op trace of a DIFFERENT system than the contract
# loop `batch_trace`'s own module docstring names as the reference.  The
# fixture the WP7 test used (`long_path_asphere`: conic 0.0, R = 200 mm, one
# 1e-7 coefficient over a 5 mm semi-aperture) cannot see either effect, which
# is why it passed with the defect live.
# ---------------------------------------------------------------------------

#: ``long_path_inexact_batch_values``' physical distances are
#: ``[1000, 2000, 5000, 1500]`` mm and df64's round-off floor overtakes
#: ``tol = 1e-10`` at ``|t| = 3.5e3`` mm, so exactly design 2 crosses over;
#: sf64's crossover is at 1.1e5 mm, so nothing does.
LATE_DESIGN = 2


def crossover_case(conic: float) -> tuple[Any, list, np.ndarray]:
    """``(optic, variables, values)`` for the sensitive crossover system."""
    return fx.long_path_inexact_batch_values(conic)


def batch_with(switch: str, conic: float, monkeypatch) -> Any:
    """One ``trace_batch`` of :func:`crossover_case` at ``switch``."""
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", switch)
    optic, variables, values = crossover_case(conic)
    return BT.trace_batch(optic, variables, values, record="all", **BATCH_KWARGS)


def row_differences(got: Any, ref: Any) -> list[str]:
    """Every recorded ``(design, surface, plane)`` where two results differ."""
    assert got.rows == ref.rows, f"row maps differ: {got.rows} vs {ref.rows}"
    out: list[str] = []
    for b in range(int(got.B)):
        for attr in ("x", "y", "z", "L", "M", "N", "intensity", "opd"):
            g = _components(BT._select(getattr(got, attr), b))
            r = _components(BT._select(getattr(ref, attr), b))
            for c, (x, y) in enumerate(zip(g, r, strict=True)):
                same = (x == y) | (np.isnan(x) & np.isnan(y))
                n = int((~same).sum())
                if n:
                    out.append(f"design {b} {attr} component {c}: {n} cells")
    return out


@pytest.mark.parametrize("conic", (-0.4, 0.0))
@pytest.mark.parametrize("mode", MODES)
def test_r2v105_late_fallback_equals_the_contract_loop(
    mps_backend, monkeypatch, mode, conic
):
    """A re-traced design's rows equal the contract loop's, tier A.

    Finding R2-V1-05, in the certificate form the finding asks for.

    * ``conic = -0.4`` is the live case: ``1 + k = 0.6`` is not float32-exact
      and the conic arrives through ``optic.updater.set_conic``, so
      ``geometry.k`` is a Python float and the dict round trip the re-trace
      used to make moved the operand side of ``(1 + k) * r2``.  Before the fix
      design 2's ``s3.x`` differed from the loop on 4 of 1141 rays (13 more
      quantities), deterministically.
    * ``conic = 0.0`` is the float32-exact **control**: the low word of
      ``1 + k`` is zero, both operand orders collapse, and this case agreed
      before the fix and must keep agreeing after it.
    * sf64 is the second control: correctly rounded, hence commutative, and
      its crossover is at 1.1e5 mm so nothing falls back at all.

    The comparison is every recorded row of every design, not just the
    re-traced one, so a fix that moved the divergence elsewhere is caught.
    """
    metal.set_mode(mode)
    fused = batch_with("1", conic, monkeypatch)
    assert fused.fused is True, f"the kernel did not run: {fused.stats}"

    expected = [False] * 4
    if mode == "df64":
        expected[LATE_DESIGN] = True
    assert fused.late_fallback_designs.tolist() == expected, fused.stats
    assert fused.stats.get("fused_trace:late_fallback", 0) == int(mode == "df64")

    loop = batch_with("0", conic, monkeypatch)
    assert loop.fused is False
    assert not row_differences(fused, loop), row_differences(fused, loop)


@pytest.mark.parametrize("mode", MODES)
def test_r2v105_retraced_rows_come_from_the_callers_optic(
    mps_backend, monkeypatch, mode
):
    """The re-traced slice is a per-op trace of the optic the caller holds.

    Finding R2-V1-05, the mechanism rather than the consequence.

    * **Certificate** -- in df64 a per-op trace of ``Optic.from_dict(
      optic.to_dict())`` is measurably NOT a per-op trace of ``optic``: the
      round trip re-wraps ``geometry.k`` (a Python float after ``set_conic``)
      in ``be.array``, which swaps the operand side of ``(1 + k) * r2``
      (R2-V1-03).  If that ever stops being true -- a round trip that preserves
      storage forms is the other fix the finding names -- this assertion is the
      one that says so, and this test should then be re-pointed rather than
      deleted.  sf64 must NOT move: it is correctly rounded and commutative.
    * **Closure** -- the fused result's re-traced rows are bit-identical to a
      per-op trace of the caller's own optic with that design applied.
    """
    metal.set_mode(mode)
    fused = batch_with("1", -0.4, monkeypatch)
    assert fused.fused is True, f"the kernel did not run: {fused.stats}"
    late = fused.late_fallback_designs.tolist()

    if mode == "sf64":
        assert late == [False] * 4, (
            "sf64's tolerance crossover is at 1.1e5 mm, so this system must "
            f"not fall back at all; got {late}"
        )
        return
    assert late[LATE_DESIGN] is True and sum(late) == 1, late

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    optic, variables, values = crossover_case(-0.4)
    BT._apply_design(optic, variables, values[LATE_DESIGN])
    launch = BT._generate_bundle(optic, **BATCH_KWARGS)

    from optiland.optic import Optic

    copy = Optic.from_dict(optic.to_dict())
    from_copy = tc.capture(
        copy.surfaces, quiet_trace(copy.surfaces, tc.copy_rays(launch)), mode
    )
    from_optic = tc.capture(
        optic.surfaces, quiet_trace(optic.surfaces, tc.copy_rays(launch)), mode
    )

    certificate = raw_differences(from_copy, from_optic)
    assert certificate, (
        "the certificate is empty: Optic.from_dict(optic.to_dict()) now traces "
        "exactly like the optic in df64, so this test would prove nothing "
        "about WHICH system the late fallback re-traces"
    )

    for attr in ("x", "y", "z", "L", "M", "N", "intensity", "opd"):
        for surface, row in fused.rows.items():
            got = _components(BT._select(getattr(fused, attr), LATE_DESIGN, row))
            want = from_optic.rows[surface][attr]
            for c, (g, w) in enumerate(zip(got, want, strict=True)):
                assert np.array_equal(g, w, equal_nan=True), (
                    f"[{mode}] the re-traced design's surface {surface} {attr} "
                    f"component {c} is not the caller's optic's: "
                    f"{int((~((g == w) | (np.isnan(g) & np.isnan(w)))).sum())} "
                    "cells differ"
                )


@pytest.mark.parametrize("mode", MODES)
def test_r2v105_the_retrace_leaves_the_callers_records_alone(
    mps_backend, monkeypatch, mode
):
    """The optic's own recorded planes survive the per-design re-trace.

    This is the property the ``Optic.from_dict`` copy bought and the one
    ``_surface_records_preserved`` now buys instead, so it is asserted rather
    than assumed: it holds before the fix and after it.  The marker is a
    recognisable array written onto every surface before the batch runs.
    """
    metal.set_mode(mode)
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    optic, variables, values = crossover_case(-0.4)

    markers = {}
    for index, surface in enumerate(optic.surfaces.surfaces):
        marker = be.array(np.full(3, float(index) + 0.5))
        surface.x = marker
        markers[index] = marker

    fused = BT.trace_batch(optic, variables, values, record="all", **BATCH_KWARGS)
    assert fused.fused is True, f"the kernel did not run: {fused.stats}"
    assert fused.late_fallback_designs.any() == (mode == "df64")

    for index, surface in enumerate(optic.surfaces.surfaces):
        assert surface.x is markers[index], (
            f"surface {index}'s recorded x was replaced by the re-trace; the "
            "fused batch path must leave the caller's optic's records alone"
        )


# ---------------------------------------------------------------------------
# R2-V1-07 -- the diagnostics planes describe the rows they come with
# ---------------------------------------------------------------------------


def _planes(a: Any) -> np.ndarray:
    """A ``(B, S, N)`` diagnostics plane as NumPy, torch or not."""
    return np.asarray(a.cpu() if hasattr(a, "cpu") else a)


@pytest.mark.parametrize("mode", MODES)
def test_r2v107_late_fallback_planes_describe_their_rows(
    mps_backend, monkeypatch, mode
):
    """A re-traced design's ``status`` / ``iters`` are zero, like its rows' run.

    Finding R2-V1-07.  ``_retrace_designs`` overwrote ``snap`` and ``final``
    for the design but not ``status`` / ``iters``, so design 2 came back with
    ``ST_TOL_CROSSOVER`` on 1141 of 1141 cells from the aborted kernel attempt
    beside per-op rows the kernel never produced -- two different runs in one
    object, against the dataclass docstring's "zeros on the fallback path".

    The kernel designs are the control: their ``iters`` must still carry the
    Newton row's real iteration counts, so this is not "zero everything".
    """
    metal.set_mode(mode)
    fused = batch_with("1", -0.4, monkeypatch)
    assert fused.fused is True, f"the kernel did not run: {fused.stats}"
    late = fused.late_fallback_designs
    assert late.tolist() == [b == LATE_DESIGN and mode == "df64" for b in range(4)]

    status = _planes(fused.status)
    iters = _planes(fused.iters)
    assert status.shape == iters.shape == (4, 5, N_BATCH_RAYS)

    for b in range(4):
        if late[b]:
            assert not status[b].any(), (
                f"design {b} was re-traced on the per-op path, but its status "
                f"plane still carries {int(np.count_nonzero(status[b]))} set "
                "cells from the kernel attempt whose rows were discarded "
                f"(TOL_CROSSOVER on "
                f"{int(np.count_nonzero(status[b] & L.ST_TOL_CROSSOVER))})"
            )
            assert not iters[b].any(), (
                f"design {b}: {int(np.count_nonzero(iters[b]))} iteration "
                "counts describe rows that were thrown away"
            )
        else:
            assert iters[b].any(), (
                f"design {b} ran on the kernel through a Newton row, so its "
                "iteration counts must be there: zeroing is only for the "
                "designs whose rows the kernel did not produce"
            )
            assert not (status[b] & L.ST_TOL_CROSSOVER).any(), (
                f"design {b} did not fall back, so no ray may carry ST_TOL_CROSSOVER"
            )


# ---------------------------------------------------------------------------
# R2-V1-06 -- DOCUMENTED LIMIT: the hook path with autograd merely enabled
#
# `NewtonRaphsonGeometry.distance` branches on `torch.is_grad_enabled()` alone
# (newton_raphson.py:566-570): with autograd on it returns the DiffOptics
# one-step correction `t - F(t)/stopgrad(dF/dt)`, with autograd off the primal
# `result.t`.  The kernel mirrors the primal solve.  The gate's `requires_grad`
# reason fires only when a tensor actually CARRIES the flag, so a default-state
# trace of a Newton system fuses and its df64 low words differ from the per-op
# path's.  `trace_batch` closed its half by tracing both legs under
# `torch.no_grad()` (R2-V1-04); the hook cannot do that without silently
# changing what an autograd caller gets back, and refusing at the gate is a
# user-visible policy change that ripples into acceptance gates this lane does
# not own.  The limit is recorded in
# NOTES/fused-trace-research/documented-limits.md and in METAL.md; this test is
# what pins it, so a later change of policy fails here instead of drifting.
# ---------------------------------------------------------------------------

#: ``(fixture name, whether the group has a Newton row)``.
#:
#: The verdict is the same in BOTH modes and for every Newton geometry -- this
#: lane measured the matrix rather than inheriting the finding's df64-only
#: reading, which came from a 1141-ray bundle.  At 4096 rays, fused vs per-op
#: with autograd enabled (quantities that differ / worst ray count):
#:
#: ===================== ================ ================
#: fixture               df64             sf64
#: ===================== ================ ================
#: aspheric_singlet      11 / 594         0 / 0
#: even_asphere_5coeff   20 / 592         13 / 1
#: inexact_conic_asphere 20 / 567         13 / 1
#: backward_newton       22 / 679         17 / 5
#: odd_asphere_singlet   26 / 672         17 / 22
#: nonconverging_asphere 43 / 1112        21 / 1112
#: cooke (no Newton row) 0 / 0            0 / 0
#: ===================== ================ ================
#:
#: so sf64 is NOT immune: it is the same one extra refinement step, and only
#: ``aspheric_singlet`` happens to absorb it at every ray of this bundle.
#: Under ``torch.no_grad()`` every cell of that table is 0 / 0.
GRAD_LOCK_CASES = (
    ("aspheric_singlet", True),
    ("even_asphere_5coeff", True),
    ("inexact_conic_asphere", True),
    ("backward_newton", True),
    ("odd_asphere_singlet", True),
    ("cooke", False),
)

#: The one Newton case whose divergence this bundle absorbs in sf64 (table
#: above).  Named rather than special-cased silently: with autograd enabled it
#: is the only (fixture, mode) pair where a Newton row agrees.
GRAD_LOCK_ABSORBED = {("aspheric_singlet", "sf64")}


@pytest.fixture
def mps_backend_grad_on() -> Iterator[None]:
    """torch / mps / float64 with autograd ENABLED -- the library's default.

    The difference from :func:`mps_backend` is the whole point of R2-V1-06:
    ``be.grad_mode.disable()`` clears only the ``requires_grad`` flag new
    arrays are created with, it does not clear ``torch.is_grad_enabled()``,
    and the Newton branch reads the latter.
    """
    previous = metal.get_mode()
    be.set_backend("torch")
    be.set_device("mps")
    be.set_precision("float64")
    be.grad_mode.disable()
    assert torch.is_grad_enabled(), "this fixture is about the grad-ON state"
    yield
    metal.set_mode(previous)
    be.grad_mode.disable()
    be.set_backend("numpy")


def grad_lock_build(name: str) -> tuple[Any, Any]:
    """``(build, rays)`` for one lock case, on the active backend."""

    def build() -> Any:
        built = fx.FIXTURES[name]()
        return built[0] if isinstance(built, tuple) else built

    built = fx.FIXTURES[name]()
    optic, factory = built if isinstance(built, tuple) else (built, None)
    rays = (
        factory(optic, N_RAYS)
        if factory is not None
        else fx.pupil_bundle(optic, N_RAYS, Hx=0.0, Hy=0.0)
    )
    return build, rays


@pytest.mark.parametrize(("name", "newton"), GRAD_LOCK_CASES)
@pytest.mark.parametrize("mode", MODES)
def test_r2v106_locked(mps_backend_grad_on, monkeypatch, mode, name, newton):
    """LOCK: with autograd merely enabled, a Newton group fuses and diverges.

    Documented limit R2-V1-06 (``NOTES/fused-trace-research/documented-limits
    .md``).  Asserted exactly, in the state the library ships in:

    1. the trace still fuses -- ``fused_trace:traces == 1``, no
       ``fused_trace_skip:*``, and ``OPTILAND_METAL_FUSED_TRACE=require`` does
       not raise (``requires_grad`` is structural and, by plan 1.2's literal
       condition, not met: no tensor carries the flag);
    2. a group with a Newton row disagrees with the per-op path -- in df64
       always, and in sf64 for every Newton fixture but one on this bundle
       (``GRAD_LOCK_CASES``' table): the limit is NOT df64-only;
    3. a group with **no** Newton row agrees in both modes, so the limit is
       exactly the Newton branch and nothing wider;
    4. under ``torch.no_grad()`` -- what ``trace_batch`` does for both its legs
       and what every conformance fixture does -- it agrees in both modes.

    The limit, its measurement and the decision it hands the integrator are in
    ``NOTES/fused-trace-research/documented-limits.md`` and, for users, in
    METAL.md's "Autograd" section.  ``trace_batch`` is NOT affected: R2-V1-04
    put both of its legs inside ``torch.no_grad()``.  If a later pass makes the
    gate refuse this configuration, assertion 1 is the one that fails, which is
    the point of a lock test.
    """
    metal.set_mode(mode)
    build, launch = grad_lock_build(name)

    assert torch.is_grad_enabled()
    reference = perop(build, tc.copy_rays(launch), monkeypatch, mode)
    got, delta = hooked(build, tc.copy_rays(launch), monkeypatch, mode)

    # 1. nothing notices.
    assert delta.get("fused_trace:traces", 0) == 1, delta
    assert not [k for k in delta if k.startswith("fused_trace_skip:")], delta
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "require")
    quiet_trace(build().surfaces, tc.copy_rays(launch))  # must not raise

    # 2./3. the divergence is exactly the Newton branch, in both modes.
    differences = raw_differences(got, reference)
    expect_divergence = newton and (name, mode) not in GRAD_LOCK_ABSORBED
    if expect_divergence:
        assert differences, (
            f"[{mode}] {name}: the limit is gone -- the fused path now agrees "
            "with the per-op path with autograd enabled. If that is a fix, "
            "close R2-V1-06 with a regression test and delete this lock."
        )
    else:
        assert not differences, (
            f"[{mode}] {name}: newton={newton} -- this case must agree "
            f"(it is the control for the limit's extent); {len(differences)} "
            f"words differ, first {differences[:3]}"
        )

    # 4. `torch.no_grad()` is the documented way out, in every case.
    with torch.no_grad():
        primal_ref = perop(build, tc.copy_rays(launch), monkeypatch, mode)
        primal_got, primal_delta = hooked(
            build, tc.copy_rays(launch), monkeypatch, mode
        )
    assert primal_delta.get("fused_trace:traces", 0) == 1, primal_delta
    tc.assert_tier_a(primal_got, primal_ref, f"r2v106_no_grad[{mode}][{name}]")


# ---------------------------------------------------------------------------
# Round 2, iteration 3 -- the state a batch leaves behind, and the contract
# `optiland/tolerancing/batched.py` states about a small bundle.
#
# R2-V1-08 and R2-V1-09 are NOT fused-vs-loop divergences: both paths leave the
# identical thing behind, because both go through `Variable.update`.  What was
# false was `trace_batch`'s docstring -- "it is mutated design by design and
# restored before the call returns" -- so the fix is in the restore, and the
# certificate of each test is that the OLD restore (the variable round trip
# alone, `_restore` without a state snapshot) still fails it.
# ---------------------------------------------------------------------------

#: The recorded attributes of a ``BatchTraceResult``, in ``snap`` plane order.
BATCH_ATTRS = ("x", "y", "z", "L", "M", "N", "intensity", "opd")

#: Cooke surfaces the round-3 batch findings are measured on: 5 carries the
#: thickness `ThicknessVariable.get_value()` reads as a position difference,
#: 1 carries the dispersive, absorbing glass an `index` variable destroys.
THICKNESS_SURFACE, INDEX_SURFACE = 5, 1


def raw_words(value: Any) -> tuple[float, ...]:
    """The raw words of a scalar, whatever its storage form.

    df64 gives the hi/lo float32 pair, sf64 the int64 pattern, a Python float
    itself.  Decoded float64 is never compared (day-1 P1).
    """
    if type(value).__name__ == "MetalFloat64":
        return tuple(
            float(c.detach().reshape(-1)[0].cpu().numpy()) for c in value.components
        )
    return tuple(np.asarray(be.to_numpy(value), dtype=np.float64).reshape(-1).tolist())


def optic_scalars(optic: Any) -> dict[str, tuple]:
    """Every geometry/material scalar a record row reads, as (type, raw words).

    The type is part of the value on purpose: R2-V1-08's second half is a
    Python ``float`` coming back as a ``MetalFloat64``, which every raw-word
    comparison alone would call equal.
    """
    out: dict[str, tuple] = {}
    for index, surface in enumerate(optic.surfaces.surfaces):
        geometry = surface.geometry
        for name in ("radius", "k"):
            if hasattr(geometry, name):
                value = getattr(geometry, name)
                out[f"s{index}.{name}"] = (type(value).__name__, raw_words(value))
        out[f"s{index}.cs.z"] = (
            type(geometry.cs.z).__name__,
            raw_words(geometry.cs.z),
        )
        out[f"s{index}.thickness"] = (
            type(surface.thickness).__name__,
            raw_words(surface.thickness),
        )
    return out


def moved_scalars(before: dict[str, tuple], after: dict[str, tuple]) -> list[str]:
    """``s<i>.<name>: <before> -> <after>`` for every scalar that moved."""
    return [
        f"{key}: {value} -> {after.get(key)}"
        for key, value in before.items()
        if after.get(key) != value
    ]


def batch_components(result: Any, b: int) -> dict[int, dict[str, list]]:
    """Design ``b``'s recorded rows as raw components, by surface index."""
    return {
        index: {
            attr: tc.raw(BT._select(getattr(result, attr), b, row))
            for attr in BATCH_ATTRS
        }
        for index, row in result.rows.items()
    }


def batch_row_differences(got: Any, ref: Any, b: int) -> list[tuple]:
    """Every raw word of design ``b`` where two results differ (NaN == NaN)."""
    a, c = batch_components(got, b), batch_components(ref, b)
    assert set(a) == set(c), f"row maps differ: {sorted(a)} vs {sorted(c)}"
    out: list[tuple] = []
    for index in sorted(a):
        for attr in BATCH_ATTRS:
            for k, (x, y) in enumerate(
                zip(a[index][attr], c[index][attr], strict=True)
            ):
                if np.issubdtype(x.dtype, np.integer):
                    same = x == y
                else:
                    same = (x == y) | (np.isnan(x) & np.isnan(y))
                out.extend((index, attr, k, int(i)) for i in np.flatnonzero(~same))
    return out


def batch_decoded_rows(result: Any, b: int) -> list[dict[str, np.ndarray]]:
    """Design ``b``'s recorded rows decoded to float64, in surface order.

    Tier B is defined on decoded values (plan 7.1), so this is the one place
    in this file where a decode is the right thing.
    """
    return [
        {
            attr: tc.decode(BT._select(getattr(result, attr), b, row))
            for attr in BATCH_ATTRS
        }
        for _, row in sorted(result.rows.items())
    ]


def launch_words(optic: Any) -> dict[str, list]:
    """The raw words of a bundle generated FROM this optic, per plane.

    A stronger probe of the restore promise than the scalars alone: the launch
    runs through ``optic.paraxial`` (EPD, entrance pupil, the object distance
    ``set_thickness`` rebuilds), so anything the restore left 1 ulp out in a
    position lands here even when the scalar it came from was put back.
    """
    bundle = BT._generate_bundle(optic, **BATCH_KWARGS)
    return {
        attr: tc.raw(getattr(bundle, attr))
        for attr in ("x", "y", "z", "L", "M", "N", "i", "w")
    }


def moved_launch(before: dict[str, list], after: dict[str, list]) -> list[str]:
    """The launch planes whose raw words moved."""
    return [
        f"{attr} word {k}"
        for attr in sorted(before)
        for k, (a, b) in enumerate(zip(before[attr], after[attr], strict=True))
        if not np.array_equal(a, b, equal_nan=True)
    ]


def batch_case(optic: Any, kind: str, surface_number: int, spread: float) -> tuple:
    """``(variables, values)`` whose FIRST row is the nominal value.

    The nominal row matters: a restore defect that only showed up for a
    perturbed design could be argued away as "the caller asked for that".
    """
    variable = Variable(optic, kind, surface_number=surface_number, wavelength=0.55)
    nominal = float(variable.variable.get_value())
    rows = (nominal, nominal + spread, nominal - spread)
    return [variable], np.array([[variable.variable.scale(v)] for v in rows])


def perop_capture(optic: Any, rays: Any, monkeypatch, mode: str) -> Any:
    """Capture a per-op trace of THIS optic (not a fresh build of it)."""
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    try:
        out = quiet_trace(optic.surfaces, rays)
        return tc.capture(optic.surfaces, out, mode)
    finally:
        monkeypatch.delenv("OPTILAND_METAL_FUSED_TRACE", raising=False)


@pytest.mark.parametrize("mode", MODES)
def test_r2v108_thickness_survives_a_batch(mps_backend, monkeypatch, mode):
    """A ``thickness`` variable leaves the attribute it found, in its own form.

    Finding R2-V1-08, closed by fix + this regression test.

    * **Certificate** -- the variable-level restore alone (``_restore`` with no
      state snapshot, which is what the call used to do) still moves it:
      ``ThicknessVariable.get_value()`` reads ``SurfaceGroup.get_thickness(5)``
      = ``cs.z[6] - cs.z[5]``, not ``surface.thickness``, so what comes back is
      a *position difference* (2.9520799999999987 for a stored 2.95208) wrapped
      in ``be.array``.  Without this leg the closure below would pass on a
      system whose round trip happened to be exact.
    * **Control** -- the same certificate leg on a ``radius`` variable, whose
      getter does read the attribute, must find nothing: the defect is the
      thickness round trip and nothing wider (``fd_jacobian`` on a radius
      already left the optic alone).
    * **Closure** -- after ``trace_batch`` every scalar of every surface is the
      value AND the type it was, after one call and after ten.
    """
    metal.set_mode(mode)

    # Certificate: the old restore, in place, on the two variable kinds.
    for kind, surface_number, spread, moves in (
        ("thickness", THICKNESS_SURFACE, 0.01, True),
        ("radius", 5, 0.5, False),
    ):
        with torch.no_grad():
            optic = fx.cooke()
            variables, values = batch_case(optic, kind, surface_number, spread)
            before = optic_scalars(optic)
            originals = BT._current_values(variables)
            BT._apply_design(optic, variables, values[1])
            BT._restore(optic, variables, originals)  # no state: the old behaviour
            moved = moved_scalars(before, optic_scalars(optic))
        if moves:
            assert moved, (
                f"[{mode}] the certificate is empty: the variable-level restore "
                "no longer moves the thickness, so this test cannot tell a fix "
                "from an accident. If the getter was fixed upstream, close "
                "R2-V1-08 against that instead."
            )
            assert any("thickness" in entry for entry in moved), moved
        else:
            assert not moved, (
                f"[{mode}] the {kind} control moved: {moved} -- R2-V1-08 is "
                "supposed to be specific to the thickness round trip"
            )

    # Closure: one call, then ten.
    with torch.no_grad():
        optic = fx.cooke()
        variables, values = batch_case(optic, "thickness", THICKNESS_SURFACE, 0.01)
        before = optic_scalars(optic)
        assert before[f"s{THICKNESS_SURFACE}.thickness"][0] == "float", (
            "the fixture no longer stores a Python float here, so the "
            "storage-form half of R2-V1-08 is not being measured"
        )
        launch_before = launch_words(optic)
        result = BT.trace_batch(optic, variables, values, **BATCH_KWARGS)
        assert result.fused, result.stats
        assert not moved_scalars(before, optic_scalars(optic)), moved_scalars(
            before, optic_scalars(optic)
        )
        # and the paraxial machinery downstream of those scalars, which
        # `set_thickness` rebuilds from them, launches the same bundle.
        assert not moved_launch(launch_before, launch_words(optic))
        for call in range(10):
            BT.trace_batch(optic, variables, values, **BATCH_KWARGS)
            moved = moved_scalars(before, optic_scalars(optic))
            assert not moved, f"[{mode}] moved after call {call}: {moved}"
        assert not moved_launch(launch_before, launch_words(optic))


@pytest.mark.parametrize("mode", MODES)
def test_r2v109_material_survives_a_batch(mps_backend, monkeypatch, mode):
    """An ``index`` variable gives the caller's glass back -- the object itself.

    Finding R2-V1-09, closed by fix + this regression test.

    ``Variable("index").update`` goes through ``OpticUpdater.set_index`` ->
    ``IdealMaterial(n=value, k=0)`` (``optic_updater.py:118-121``), so a restore
    that only replays the *value* replays an ideal glass: the dispersion and
    the absorption of the caller's ``Material`` are gone, and 23 of the 29
    shipped samples are absorbing (plan 1.1).

    * **Certificate** -- the variable-level restore alone still leaves an
      ``IdealMaterial`` with ``k = 0`` behind.
    * **Closure** -- after ``trace_batch`` the surface holds the SAME material
      object, still dispersive and still absorbing, and a per-op trace of the
      caller's optic is tier-A identical to one taken before the call.
    """
    metal.set_mode(mode)

    with torch.no_grad():
        optic = fx.cooke()
        variables, values = batch_case(optic, "index", INDEX_SURFACE, 0.001)
        original = optic.surfaces[INDEX_SURFACE].material_post
        originals = BT._current_values(variables)
        BT._apply_design(optic, variables, values[1])
        BT._restore(optic, variables, originals)  # no state: the old behaviour
        replaced = optic.surfaces[INDEX_SURFACE].material_post
    assert replaced is not original, (
        f"[{mode}] the certificate is empty: the variable-level restore no "
        "longer replaces the material, so this test proves nothing"
    )
    assert float(replaced.k(0.55)) == 0.0, float(replaced.k(0.55))

    with torch.no_grad():
        optic = fx.cooke()
        variables, values = batch_case(optic, "index", INDEX_SURFACE, 0.001)
        surface = optic.surfaces[INDEX_SURFACE]
        original = surface.material_post
        # The fixture has to be dispersive and absorbing or the closure is
        # vacuous -- this is the half of the finding a value restore destroys.
        assert float(original.n(0.45)) != float(original.n(0.65))
        assert float(original.k(0.55)) > 0.0
        rays = fx.collimated_bundle(N_BATCH_RAYS, radius=5.0, z=-10.0)
        before = perop_capture(optic, tc.copy_rays(rays), monkeypatch, mode)
        launch_before = launch_words(optic)
        result = BT.trace_batch(optic, variables, values, **BATCH_KWARGS)
        assert result.fused, result.stats
        after = perop_capture(optic, tc.copy_rays(rays), monkeypatch, mode)
        # The index moves the paraxial quantities, so the launch this optic
        # generates is the sharpest witness that the glass really came back.
        assert not moved_launch(launch_before, launch_words(optic))

    assert surface.material_post is original, (
        f"[{mode}] the surface holds {type(surface.material_post).__name__}, "
        f"not the {type(original).__name__} the call found"
    )
    assert optic.surfaces[INDEX_SURFACE + 1].material_pre is original
    tc.assert_tier_a(after, before, f"r2v109_after_vs_before[{mode}]")


@pytest.mark.parametrize("mode", MODES)
def test_r2v109_index_designs_use_ideal_glass_like_the_loop(
    mps_backend, monkeypatch, mode
):
    """PINNED: an ``index`` design is an ideal glass, on both paths.

    The half of R2-V1-09 that is not a restore defect and is not this module's
    to change: ``Variable("index").update`` *replaces* the glass, so no row of
    an ``index`` variable reproduces a dispersive system -- not even the row
    carrying the nominal index, whose launch moves too because the paraxial
    quantities move with the index.  It is documented on ``trace_batch``'s
    ``values`` argument and in METAL.md.

    What must hold, and is asserted here, is that the batch is faithful to the
    reference it names: the fused design 0 equals the CONTRACT LOOP's design 0
    at tier A, and the loop leaves the same ``IdealMaterial`` behind before the
    restore puts the caller's glass back.  If a later change makes an ``index``
    design keep the dispersion, this test fails and the docstring is wrong.
    """
    metal.set_mode(mode)

    with torch.no_grad():
        optic = fx.cooke()
        variables, values = batch_case(optic, "index", INDEX_SURFACE, 0.001)
        original = optic.surfaces[INDEX_SURFACE].material_post
        rays = fx.collimated_bundle(N_BATCH_RAYS, radius=5.0, z=-10.0)
        untouched = perop_capture(optic, tc.copy_rays(rays), monkeypatch, mode)

        fused = BT.trace_batch(optic, variables, values, **BATCH_KWARGS)
        assert fused.fused, fused.stats

        monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
        loop = BT.trace_batch(optic, variables, values, **BATCH_KWARGS)
        monkeypatch.delenv("OPTILAND_METAL_FUSED_TRACE", raising=False)
    assert not loop.fused, loop.stats

    # 1. the two paths' nominal design agrees, word for word.
    differences = batch_row_differences(fused, loop, 0)
    assert not differences, (
        f"[{mode}] fused vs contract loop on design 0: {len(differences)} words "
        f"differ, first {differences[:3]}"
    )

    # 2. and that design is NOT the untouched system -- the pinned part.
    nominal = batch_components(fused, 0)
    moved = [
        f"s{index}.{attr} word {k}"
        for index, row in sorted(nominal.items())
        for attr in BATCH_ATTRS
        if attr in untouched.rows[index]
        for k, (a, b) in enumerate(
            zip(row[attr], untouched.rows[index][attr], strict=True)
        )
        if not np.array_equal(a, b, equal_nan=True)
    ]
    assert moved, (
        f"[{mode}] an `index` design now reproduces the dispersive system. "
        "That is a behaviour change in OpticUpdater.set_index, not a bug here "
        "-- update trace_batch's `values` docstring and METAL.md with it."
    )

    # 3. and the caller's glass is back afterwards (the fixed half).
    assert optic.surfaces[INDEX_SURFACE].material_post is original


# ---------------------------------------------------------------------------
# R2-V1-10 -- DOCUMENTED LIMIT: a batch of 1024 rays or fewer is tier B
#
# `optiland/tolerancing/batched.py` promised, in its module docstring and in
# `monte_carlo_batched`'s, that "the batched frame equals the sequential frame
# cell for cell".  That is plan 7.1 TIER A, and tier A is defined for N > 1024
# only.  Below `materials/base.py`'s `_MAX_VALUE_KEY_ARRAY_SIZE = 1024` the
# per-op path evaluates a dispersive index on the whole `w` array; above it, on
# one uniform representative -- which is what the kernel always uses (plan 3.8).
# Both acceptance tests happened to sample at 19 rings (N = 1,141), the one
# side of the threshold where the promise is true, and the batch gate skips the
# bundle checks (design 6.3) so nothing stops a small batch from fusing.
#
# The magnitudes are INSIDE plan 7.1's own tier-B envelope (max |delta|
# 6.5e-16 against 64 * eps_df64 * scale = 2.3e-13), so this is not a kernel bug
# and the closure is a documented limit: the contract now carries the
# qualification (`batched.py`'s "How close ... is, exactly", `df.attrs["tier"]`
# and `TIER_A_MIN_RAYS`), the two acceptance tests gained an N <= 1024 leg that
# asserts the TIER-B rule instead of exact equality, and this test pins the
# mechanism so that a later fix -- or a later regression into a wider
# difference -- fails here.  Nothing was widened: the N = 1,141 legs still
# assert exact equality.
# ---------------------------------------------------------------------------

#: ``(hexapolar rings, ray count)``.  12 rings is 469 rays -- below
#: ``_MAX_VALUE_KEY_ARRAY_SIZE`` and above the 256-ray ``host_resident``
#: threshold, so it fuses and lands on tier-B site 1.  19 rings is the control.
SITE1_BUNDLES = ((12, 469), (19, 1141))


def ideal_cooke() -> Any:
    """``cooke()`` with every glass replaced by a constant-index material.

    The control that names the mechanism: ``_MAX_VALUE_KEY_ARRAY_SIZE`` only
    changes how a *dispersive* index is evaluated, so with ``IdealMaterial``
    everywhere the two key paths give the same number and N = 469 must be
    bit-exact.  ``ObjectSurface`` exposes ``material_pre``/``material_post`` as
    read-only properties; its material is air either way, and every other
    surface's ``material_pre`` is the previous surface's ``material_post``.
    """
    from optiland.materials import IdealMaterial

    optic = fx.cooke()
    replaced = 0
    for surface in optic.surfaces.surfaces:
        material = getattr(surface, "material_post", None)
        if material is None:
            continue
        try:
            surface.material_post = IdealMaterial(n=float(material.n(0.55)), k=0.0)
        except AttributeError:  # pragma: no cover - ObjectSurface
            continue
        replaced += 1
    assert replaced >= 6, replaced
    optic.updater.update()
    return optic


@pytest.mark.parametrize("glass", ("dispersive", "ideal"))
@pytest.mark.parametrize(("rings", "n_rays"), SITE1_BUNDLES)
@pytest.mark.parametrize("mode", MODES)
def test_r2v110_locked(mps_backend, monkeypatch, mode, rings, n_rays, glass):
    """LOCK: at N <= 1024 a dispersive df64 batch is tier B, not tier A.

    Documented limit R2-V1-10 (``NOTES/fused-trace-research/documented-limits
    .md``), asserted in the certificate form the finding asks for -- one
    parametrization, four legs:

    1. ``df64``/``dispersive``/``N = 469``: the fused batch does **not** equal
       the contract loop word for word (tier A fails, and that it fails is
       asserted, not tolerated), and it **does** satisfy plan 7.1's tier-B rule
       ``|delta| <= 64 * eps * scale`` with equal NaN and ``i == 0`` masks;
    2. ``ideal``/``N = 469``: bit-exact -- the mechanism is the dispersive
       index lookup and nothing else;
    3. ``N = 1141``: bit-exact in both glasses -- the threshold is real;
    4. ``sf64``: bit-exact everywhere -- sf64 has no float32 key array.

    Both legs are the same ``trace_batch`` call with the kernel on and off, so
    nothing but the kernel differs.  If a later pass makes leg 1 bit-exact,
    this test fails and R2-V1-10 is closed by a fix instead; if the difference
    ever grows past the tier-B bound, it fails as a kernel bug.
    """
    metal.set_mode(mode)
    build = fx.cooke if glass == "dispersive" else ideal_cooke
    kwargs = dict(BATCH_KWARGS, num_rays=rings, record=True)

    with torch.no_grad():
        optic = build()
        variables, values = batch_case(optic, "radius", 5, 0.5)
        fused = BT.trace_batch(optic, variables, values, **kwargs)
        monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
        loop = BT.trace_batch(optic, variables, values, **kwargs)
        monkeypatch.delenv("OPTILAND_METAL_FUSED_TRACE", raising=False)

    assert fused.fused, fused.stats
    assert not loop.fused, loop.stats
    assert int(fused.n_rays) == n_rays, (fused.n_rays, n_rays)

    differences = [
        (b, *d)
        for b in range(int(values.shape[0]))
        for d in batch_row_differences(fused, loop, b)
    ]
    tier_b_expected = mode == "df64" and glass == "dispersive" and n_rays <= 1024

    if not tier_b_expected:
        assert not differences, (
            f"[{mode}] {glass} N={n_rays}: this leg is the CONTROL and must be "
            f"tier A; {len(differences)} words differ, first {differences[:3]}"
        )
        return

    assert differences, (
        f"[{mode}] {glass} N={n_rays}: the limit is gone -- the batch now "
        "equals the contract loop word for word below 1024 rays. If that is a "
        "fix, close R2-V1-10 with a regression test, delete this lock and "
        "restore the unqualified promise in optiland/tolerancing/batched.py."
    )
    for b in range(int(values.shape[0])):
        got = batch_decoded_rows(fused, b)
        ref = batch_decoded_rows(loop, b)
        tc.assert_tier_b(
            got,
            ref,
            mode=mode,
            scale=tc.system_scale(optic, ref),
            what=f"r2v110[{mode}][{glass}][N={n_rays}][design {b}]",
        )
