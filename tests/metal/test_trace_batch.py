"""WP7: the batched multi-design trace API (``optiland/raytrace/batch_trace.py``).

Every fused assertion here is plan 7.1 **tier A**: raw-component equality (df64
hi/lo float32 words, sf64 int64 bit patterns) against the *contract loop* --
the same designs driven one at a time through ``optic.trace`` on the same
backend.  No tolerance appears anywhere, and no test compares decoded float64.

Two traps this file is built to avoid:

1. **Comparing the per-op path with itself.**  ``trace_batch`` falls back to the
   contract loop for every refusal, so a test that only compares "batch vs
   loop" passes loudly while asserting nothing if the kernel never ran.
   ``assert_fused`` therefore pins ``result.fused is True`` on every fused leg,
   and its failure message names the refusal reason and any mirror drift.
2. **Hiding a kernel divergence behind a convenient fixture.**  The
   ``asphere_coeff`` row uses ``even_asphere_inf_radius`` rather than
   ``aspheric_singlet`` because, at the time of writing, the kernel disagrees
   with the per-op path in the df64 low word for every *finite-radius* Newton
   geometry (reproduced with ``trace.fused_trace`` alone, no batch code
   involved; recorded in ``NOTES/fused-trace-research/status.md``).  That is a
   WP1/WP6 conformance finding, not a batch-API one; the fixture used here
   still exercises the ``asphere_coeff`` variable, its tier-1 row refresh and
   the coefficient-list contract of plan 3.9.

Run: ``pytest tests/metal/test_trace_batch.py -q -p no:cacheprovider -o addopts=``
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import inspect  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import optiland.backend as be  # noqa: E402
from optiland.optimization.variable import Variable  # noqa: E402
from optiland.raytrace import batch_trace as BT  # noqa: E402
from optiland.surfaces.surface_group import SurfaceGroup  # noqa: E402

_SCRIPTS = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
import trace_fixtures  # noqa: E402

torch = pytest.importorskip("torch")
HAS_MPS = bool(torch.backends.mps.is_available())

MODES = ("df64", "sf64")

#: Whether WP4's hook is in ``SurfaceGroup.trace``.  It changes how many gate
#: refusals a *fallback* leg contributes, so counter predictions name it rather
#: than guess (the same probe ``test_trace_writeback.py`` uses).
HOOKED = "_fused_metal_trace" in inspect.getsource(SurfaceGroup.trace)

#: Day-1 Q13: 19 hexapolar rings = 1,141 rays, above
#: ``_MAX_VALUE_KEY_ARRAY_SIZE`` (1024), so every comparison is tier A.
RINGS = 19
N_RAYS = 1 + 3 * RINGS * (RINGS + 1)

#: Designs per batch in the equality sweep (plan WP7).
B = 5

#: The recorded attributes of ``BatchTraceResult``, in ``snap`` plane order.
ATTRS = ("x", "y", "z", "L", "M", "N", "intensity", "opd")

TRACE_KWARGS = {
    "Hx": 0.0,
    "Hy": 0.0,
    "wavelength": 0.55,
    "num_rays": RINGS,
    "distribution": "hexapolar",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
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
    yield


@pytest.fixture(autouse=True)
def _driver_state():
    """Zero the counters and the driver's one-shot state around every test."""
    if not HAS_MPS:
        yield
        return
    from optiland.backend.torch_backend.metal import tensor as T
    from optiland.backend.torch_backend.metal import trace

    trace.reset_driver_state()
    T.reset_stats()
    yield
    trace.reset_driver_state()
    T.reset_stats()


@pytest.fixture
def mps_backend():
    """torch / mps / float64 with autograd off (the fused path's setting)."""
    if not HAS_MPS:  # pragma: no cover - hardware gate
        pytest.skip("Metal GPU required")
    from optiland.backend.torch_backend import metal

    previous = metal.get_mode()
    be.set_backend("torch")
    be.set_device("mps")
    be.set_precision("float64")
    be.grad_mode.disable()
    yield
    metal.set_mode(previous)
    be.grad_mode.disable()
    be.set_backend("numpy")


def use_mode(mode):
    """Select the Metal representation for this test."""
    from optiland.backend.torch_backend import metal

    metal.set_mode(mode)


# ---------------------------------------------------------------------------
# Comparison helpers (raw components only)
# ---------------------------------------------------------------------------
def components(plane):
    """The raw component arrays of one plane, as NumPy.

    df64 gives two float32 words, sf64 one int64 bit pattern.  Decoded float64
    is never compared: ``decode(encode(x)) != x`` at the 2^-48 level (day-1 P1).
    """
    if type(plane).__name__ == "MetalFloat64":
        return [c.detach().contiguous().cpu().numpy() for c in plane.components]
    return [np.asarray(plane)]


def raw_equal(a, b):
    """Raw-component equality, NaN == NaN."""
    ca, cb = components(a), components(b)
    if len(ca) != len(cb):
        return False
    return all(
        x.shape == y.shape and np.array_equal(x, y, equal_nan=True)
        for x, y in zip(ca, cb, strict=True)
    )


def diff_report(a, b, what):
    """A message naming the first differing cells, for an assertion."""
    for k, (x, y) in enumerate(zip(components(a), components(b), strict=True)):
        if x.shape != y.shape:
            return f"{what}: component {k} shape {x.shape} vs {y.shape}"
        same = (x == y) | (np.isnan(x) & np.isnan(y))
        bad = np.argwhere(~same)
        if bad.size:
            return (
                f"{what}: component {k} differs in {len(bad)} of {x.size} cells, "
                f"first at {bad[0].tolist()} ({x[tuple(bad[0])]!r} vs "
                f"{y[tuple(bad[0])]!r})"
            )
    return f"{what}: equal"


def assert_rows_equal(got, want, *, attrs=ATTRS):
    """Every recorded row of two results agrees component for component."""
    assert got.rows == want.rows, f"row maps differ: {got.rows} vs {want.rows}"
    for attr in attrs:
        a, b = getattr(got, attr), getattr(want, attr)
        assert (a is None) == (b is None), f"{attr}: one result recorded nothing"
        if a is None:
            continue
        assert raw_equal(a, b), diff_report(a, b, attr)


def assert_fused(result):
    """Pin that the kernel really ran; name the reason when it did not."""
    if result.fused:
        return
    from optiland.backend.torch_backend.metal import trace_mirror

    skips = {k: v for k, v in result.stats.items() if k.startswith("fused_trace_skip")}
    raise AssertionError(
        "trace_batch fell back to the contract loop, so this comparison would "
        f"compare the per-op path with itself. Refusals: {skips or 'none'}. "
        f"Mirror drift: {trace_mirror.check_all() or 'none'}."
    )


def stats_of(result, prefix="fused_trace"):
    """The result's counter delta, restricted to the fused-trace keys."""
    return {k: v for k, v in result.stats.items() if prefix in k}


def run(optic, variables, values, **kwargs):
    """``trace_batch`` with this file's launch settings."""
    merged = dict(TRACE_KWARGS)
    merged.update(kwargs)
    return BT.trace_batch(optic, variables, values, **merged)


def contract_loop(optic, variables, values, monkeypatch, **kwargs):
    """The reference: the same designs through ``optic.trace``, kernel off."""
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    try:
        result = run(optic, variables, values, **kwargs)
    finally:
        monkeypatch.delenv("OPTILAND_METAL_FUSED_TRACE", raising=False)
    assert result.fused is False
    return result


# ---------------------------------------------------------------------------
# The variable sweep: one case per variable type (plan WP7)
# ---------------------------------------------------------------------------
def _scaled(variable, physical):
    """``physical`` in the variable's scaled units (plan 3.9 [fix: L1.10])."""
    return [variable.variable.scale(p) for p in physical]


def case_radius():
    optic = trace_fixtures.cooke()
    v = Variable(optic, "radius", surface_number=5)
    physical = [79.0, 79.5, 79.68360, 80.0, 80.5]
    return optic, [v], np.array([_scaled(v, physical)]).T


def case_reciprocal_radius():
    optic = trace_fixtures.cooke()
    v = Variable(optic, "reciprocal_radius", surface_number=5)
    base = float(v.value)
    return optic, [v], np.array([[base * f] for f in (0.96, 0.98, 1.0, 1.02, 1.04)])


def case_conic():
    optic = trace_fixtures.cooke()
    v = Variable(optic, "conic", surface_number=5)
    return optic, [v], np.array([[k] for k in (-0.2, -0.1, 0.0, 0.1, 0.2)])


def case_thickness():
    optic = trace_fixtures.cooke()
    v = Variable(optic, "thickness", surface_number=5)
    physical = [2.85, 2.90, 2.95208, 3.00, 3.05]
    return optic, [v], np.array([_scaled(v, physical)]).T


def case_asphere_coeff():
    # even_asphere_inf_radius, not aspheric_singlet: see the module docstring.
    optic, _ = trace_fixtures.even_asphere_inf_radius()
    v = Variable(optic, "asphere_coeff", surface_number=1, coeff_number=0)
    base = float(v.value)
    span = abs(base) * 0.05 if base else 1e-3
    return optic, [v], np.array([[base + f * span] for f in (-2, -1, 0, 1, 2)])


def case_decenter():
    optic = trace_fixtures.cooke()
    v = Variable(optic, "decenter", surface_number=5, axis="x")
    return optic, [v], np.array([[d] for d in (-0.05, -0.02, 0.0, 0.02, 0.05)])


def case_tilt():
    optic = trace_fixtures.cooke()
    v = Variable(optic, "tilt", surface_number=5, axis="y")
    return optic, [v], np.array([[t] for t in (-0.02, -0.01, 0.0, 0.01, 0.02)])


#: The physical indices ``case_index`` drives surface 5 to.
INDEX_PHYSICAL = (1.58, 1.60, 1.62, 1.64, 1.66)


def case_index():
    optic = trace_fixtures.cooke()
    v = Variable(optic, "index", surface_number=5, wavelength=0.55)
    # IndexVariable's default scaler is LinearScaler(1.0, -1.5), so a physical
    # index of 1.58 is the scaled value 0.08 (plan 3.9 [fix: L1.10]).
    return optic, [v], np.array([_scaled(v, INDEX_PHYSICAL)]).T


def case_material():
    optic = trace_fixtures.cooke()
    glasses = ["N-BK7", "N-SF11", "SK16", "F2", "N-SF5"]
    v = Variable(optic, "material", surface_number=5, glass_selection=glasses)
    return optic, [v], np.array([[g] for g in glasses], dtype=object)


CASES = {
    "radius": case_radius,
    "reciprocal_radius": case_reciprocal_radius,
    "conic": case_conic,
    "thickness": case_thickness,
    "asphere_coeff": case_asphere_coeff,
    "decenter": case_decenter,
    "tilt": case_tilt,
    "index": case_index,
    "material": case_material,
}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("vartype", sorted(CASES))
def test_batch_equals_contract_loop(mps_backend, monkeypatch, mode, vartype):
    """Every recorded row of every design equals the contract loop, raw."""
    use_mode(mode)
    optic, variables, values = CASES[vartype]()
    assert values.shape == (B, 1)

    fused = run(optic, variables, values, record="all")
    assert_fused(fused)
    loop = contract_loop(optic, variables, values, monkeypatch, record="all")

    assert fused.B == loop.B == B
    assert fused.n_rays == loop.n_rays == N_RAYS
    assert_rows_equal(fused, loop)
    assert stats_of(fused)["fused_trace:designs"] == B


@pytest.mark.parametrize("mode", MODES)
def test_batch_equals_contract_loop_multivariable(mps_backend, monkeypatch, mode):
    """Two variables of different types in one batch, still row for row."""
    use_mode(mode)
    optic = trace_fixtures.cooke()
    radius = Variable(optic, "radius", surface_number=5)
    tilt = Variable(optic, "tilt", surface_number=6, axis="x")
    values = np.array(
        [
            [radius.variable.scale(r), t]
            for r, t in zip(
                (79.0, 79.5, 79.68360, 80.0, 80.5),
                (-0.02, -0.01, 0.0, 0.01, 0.02),
                strict=True,
            )
        ]
    )
    fused = run(optic, [radius, tilt], values, record="all")
    assert_fused(fused)
    loop = contract_loop(optic, [radius, tilt], values, monkeypatch, record="all")
    assert_rows_equal(fused, loop)


# ---------------------------------------------------------------------------
# Units: values are the optimizer's scaled units  [fix: L1.10]
# ---------------------------------------------------------------------------
def test_values_are_scaled_units(mps_backend, monkeypatch):
    """``var.variable.scale(physical)`` lands as ``physical`` on the geometry."""
    use_mode("df64")
    optic = trace_fixtures.cooke()
    v = Variable(optic, "radius", surface_number=5)
    physical = 80.25
    scaled = v.variable.scale(physical)
    # RadiusVariable's default scaler is LinearScaler(1/100, -1.0).
    assert scaled == pytest.approx(physical / 100.0 - 1.0, abs=0.0, rel=0.0)

    BT._apply_design(optic, [v], [scaled])
    assert float(optic.surfaces[5].geometry.radius) == physical

    # And end to end: the batch row built from the scaled value traces exactly
    # like the system whose radius IS the physical value.
    values = np.array([[scaled]])
    fused = run(optic, [v], values, record="all")
    assert_fused(fused)

    reference = trace_fixtures.cooke()
    reference.surfaces[5].geometry.radius = be.array(physical)
    reference.updater.update()
    loop = contract_loop(reference, [], np.zeros((1, 0)), monkeypatch, record="all")
    assert_rows_equal(fused, loop)


# ---------------------------------------------------------------------------
# Tier 1: the row cache and its canary
# ---------------------------------------------------------------------------
def compile_designs(optic, variables, values, *, mode, tier1, record=True):
    """Drive ``_compile_designs`` directly and hand back its tables."""
    from optiland.backend.torch_backend.metal.trace_record import canonical_w0

    w0 = canonical_w0(TRACE_KWARGS["wavelength"], mode)
    originals = BT._current_values(variables)
    try:
        return BT._compile_designs(
            optic,
            variables,
            values,
            w0=w0,
            mode=mode,
            record=record,
            tier1=tier1,
            shared=True,
            bundles=[],
            launch_kwargs=dict(TRACE_KWARGS),
        )
    finally:
        BT._restore(optic, variables, originals)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", sorted(CASES))
def test_tier1_equals_tier0_bitwise(mps_backend, mode, name):
    """The row cache reproduces the full compile, table for table."""
    use_mode(mode)
    optic, variables, values = CASES[name]()
    tier1 = compile_designs(optic, variables, values, mode=mode, tier1=True)
    tier0 = compile_designs(optic, variables, values, mode=mode, tier1=False)
    assert tier1.row_cache_used is True
    assert tier0.row_cache_used is False
    assert tier1.canary_mismatch is False
    assert np.array_equal(tier1.records.surf_int, tier0.records.surf_int)
    assert np.array_equal(
        tier1.records.surf_real, tier0.records.surf_real, equal_nan=True
    )
    assert np.array_equal(tier1.records.coef, tier0.records.coef, equal_nan=True)
    assert np.array_equal(tier1.records.snap_rows, tier0.records.snap_rows)
    assert np.array_equal(tier1.records.step_cost, tier0.records.step_cost)


def test_tier1_disabled_with_pickups(mps_backend, monkeypatch):
    """A pickup turns the row cache off, and the result is the tier-0 one."""
    use_mode("df64")
    optic, variables, values = case_radius()
    # A *radius* pickup: a thickness pickup writes a shape-(1,) ``cs.z`` and the
    # gate then refuses the whole system with ``pose_nonfinite`` (reported to
    # WP2 in status.md), which would refuse this test for the wrong reason.
    optic.pickups.add(2, "radius", 6, scale=1.0)
    optic.updater.update()

    assert BT._row_cache_enabled("auto", optic, variables) is False
    assert BT._row_cache_enabled("auto", trace_fixtures.cooke(), variables) is True

    fused = run(optic, variables, values, record="all")
    assert_fused(fused)
    assert fused.row_cache_used is False
    forced = run(optic, variables, values, record="all", row_cache=False)
    assert_rows_equal(fused, forced)
    loop = contract_loop(optic, variables, values, monkeypatch, record="all")
    assert_rows_equal(fused, loop)


def test_tier1_canary_catches_injected_updater_change(mps_backend, monkeypatch):
    """A hidden updater side effect that tier 1 cannot see fails the canary.

    ``set_thickness`` is patched to also write surface 1's conic, a row a
    thickness variable never touches, so tier 1's cached table keeps design 0's
    conic while a tier-0 compile reads the perturbed one.  The injected change
    is a pure function of the value, so the tier-0 recompile is deterministic.
    """
    use_mode("df64")
    from optiland.optic.optic_updater import OpticUpdater

    original = OpticUpdater.set_thickness

    def patched(self, value, surface_number):
        original(self, value, surface_number)
        self.optic.surfaces[1].geometry.k = be.array(float(value) * 0.01)

    monkeypatch.setattr(OpticUpdater, "set_thickness", patched)

    optic, variables, values = case_thickness()
    fused = run(optic, variables, values, record="all")
    assert_fused(fused)
    assert fused.canary_mismatch is True
    assert fused.row_cache_used is False
    assert stats_of(fused)["fused_trace:tier1_canary_mismatch"] == 1

    forced = run(optic, variables, values, record="all", row_cache=False)
    assert forced.canary_mismatch is False
    assert_rows_equal(fused, forced)
    loop = contract_loop(optic, variables, values, monkeypatch, record="all")
    assert_rows_equal(fused, loop)


def test_touched_rows_cover_the_updater(mps_backend):
    """The tier-1 row sets are the ones the updater can actually move."""
    optic = trace_fixtures.cooke()
    s = len(optic.surfaces.surfaces)
    thickness = Variable(optic, "thickness", surface_number=3)
    assert BT._touched_rows(thickness, s) == set(range(3, s))
    object_thickness = Variable(optic, "thickness", surface_number=0)
    assert BT._touched_rows(object_thickness, s) == {0}
    index = Variable(optic, "index", surface_number=3, wavelength=0.55)
    assert BT._touched_rows(index, s) == {3, 4}
    radius = Variable(optic, "radius", surface_number=3)
    assert BT._touched_rows(radius, s) == {3}
    last_index = Variable(optic, "index", surface_number=s - 1, wavelength=0.55)
    assert BT._touched_rows(last_index, s) == {s - 1}


# ---------------------------------------------------------------------------
# Shared launch
# ---------------------------------------------------------------------------
def test_shared_launch_rule(mps_backend, monkeypatch):
    """Post-stop variables share one launch set; pre-stop ones never do."""
    use_mode("df64")
    optic, variables, values = case_radius()
    stop = int(optic.surfaces.stop_index)
    assert stop == 4 and BT._surface_number(variables[0]) == 5

    assert BT._shared_launch_reason(optic, variables) is None
    shared = run(optic, variables, values, record="all")
    assert_fused(shared)
    assert shared.launch_shared is True

    per_design = run(optic, variables, values, record="all", shared_launch=False)
    assert_fused(per_design)
    assert per_design.launch_shared is False
    # Sharing is an optimisation, never the contract: it must not change a bit.
    assert_rows_equal(shared, per_design)

    # And the invariance is real, not assumed: the launch sets themselves are
    # bit-identical across designs (note 06 section 3.2, max|delta| = 0.0).
    bundles = []
    originals = BT._current_values(variables)
    try:
        for row in values:
            BT._apply_design(optic, variables, row)
            bundles.append(BT._generate_bundle(optic, **TRACE_KWARGS))
    finally:
        BT._restore(optic, variables, originals)
    for other in bundles[1:]:
        for attr in ("x", "y", "z", "L", "M", "N", "i", "w"):
            got, want = getattr(other, attr), getattr(bundles[0], attr)
            assert raw_equal(got, want), diff_report(got, want, f"launch.{attr}")

    # A pre-stop variable is refused by the rule, and the refusal is earned:
    # the launch sets really do move.
    pre = trace_fixtures.cooke()
    pre_var = Variable(pre, "radius", surface_number=1)
    reason = BT._shared_launch_reason(pre, [pre_var])
    assert reason is not None and "before the stop" in reason
    pre_values = np.array([[pre_var.variable.scale(r)] for r in (21.0, 23.0)])
    pre_result = run(pre, [pre_var], pre_values, record="all")
    assert_fused(pre_result)
    assert pre_result.launch_shared is False

    moved = []
    originals = BT._current_values([pre_var])
    try:
        for row in pre_values:
            BT._apply_design(pre, [pre_var], row)
            moved.append(BT._generate_bundle(pre, **TRACE_KWARGS))
    finally:
        BT._restore(pre, [pre_var], originals)
    assert any(
        not raw_equal(getattr(moved[0], attr), getattr(moved[1], attr))
        for attr in ("x", "y", "z", "L", "M", "N", "i", "w")
    ), (
        "a pre-stop radius change did not move the launch set; the shared-launch "
        "rule would be untestable theatre"
    )
    loop = contract_loop(pre, [pre_var], pre_values, monkeypatch, record="all")
    assert_rows_equal(pre_result, loop)


def test_shared_launch_reasons(mps_backend):
    """Each condition of note 06 section 4 is checked, and says which failed."""
    use_mode("df64")
    optic = trace_fixtures.cooke()
    post = Variable(optic, "radius", surface_number=5)
    assert BT._shared_launch_reason(optic, [post]) is None

    optic.ray_tracer.ray_aiming_config = {"mode": "iterative"}
    assert "aiming" in BT._shared_launch_reason(optic, [post])
    optic.ray_tracer.ray_aiming_config = {"mode": "paraxial"}

    optic.fields.set_type("paraxial_image_height")
    assert "field type" in BT._shared_launch_reason(optic, [post])


# ---------------------------------------------------------------------------
# Structure, records, writeback
# ---------------------------------------------------------------------------
def test_structural_change_raises(mps_backend):
    """A design that changes the geometry code is not a perturbation."""
    use_mode("df64")
    optic = trace_fixtures.cooke()
    v = Variable(optic, "reciprocal_radius", surface_number=5)
    base = float(v.value)
    # 0 means an infinite radius (reciprocal_radius.py:66), which the adapter
    # compiles as GEOM_STD_INF instead of GEOM_CONIC.
    values = np.array([[base], [0.0]])
    with pytest.raises(ValueError, match="geometry"):
        run(optic, [v], values, record="all")
    # The same ValueError on every backend: the contract loop compares a
    # structural signature where the fused path compares its tables.
    monkeypatch_env = os.environ.get("OPTILAND_METAL_FUSED_TRACE")
    os.environ["OPTILAND_METAL_FUSED_TRACE"] = "0"
    try:
        with pytest.raises(ValueError, match="radius finiteness"):
            run(optic, [v], values, record="all")
    finally:
        if monkeypatch_env is None:
            os.environ.pop("OPTILAND_METAL_FUSED_TRACE", None)
        else:
            os.environ["OPTILAND_METAL_FUSED_TRACE"] = monkeypatch_env
    # The optic is restored even though the call raised.
    assert float(optic.surfaces[5].geometry.radius) == pytest.approx(79.68360, abs=1e-9)


@pytest.mark.parametrize("mode", MODES)
def test_record_policies(mps_backend, mode):
    """Each policy records exactly the rows it names, with the same values."""
    use_mode(mode)
    optic, variables, values = case_radius()
    s = len(optic.surfaces.surfaces)

    every = run(optic, variables, values, record="all")
    assert_fused(every)
    assert every.rows == dict.fromkeys(range(s)) or every.rows == {
        i: i for i in range(s)
    }
    assert every.x.shape == (B, s, N_RAYS)

    image = run(optic, variables, values, record="image")
    assert_fused(image)
    assert image.rows == {s - 1: 0}
    assert image.x.shape == (B, 1, N_RAYS)

    stop = run(optic, variables, values, record="stop")
    assert_fused(stop)
    assert stop.rows == {int(optic.surfaces.stop_index): 0}

    explicit = run(optic, variables, values, record=[1, 3])
    assert_fused(explicit)
    assert explicit.rows == {1: 0, 3: 1}
    assert explicit.x.shape == (B, 2, N_RAYS)

    nothing = run(optic, variables, values, record="none")
    assert_fused(nothing)
    assert nothing.rows == {} and nothing.x is None
    assert nothing.final is not None  # write_final flips on (day-1 Q10)

    # Same rows, same values, whichever policy asked for them.
    same_rows = ((image, {s - 1: 0}), (stop, stop.rows), (explicit, {1: 0, 3: 1}))
    for policy, rows in same_rows:
        for surface, row in rows.items():
            for attr in ATTRS:
                got = BT._select(getattr(policy, attr), 2, row)
                want = BT._select(getattr(every, attr), 2, every.rows[surface])
                assert raw_equal(got, want), diff_report(
                    got, want, f"{attr}[{surface}]"
                )

    with pytest.raises(ValueError, match="listed twice"):
        run(optic, variables, values, record=[2, 2])
    with pytest.raises(ValueError, match="outside"):
        run(optic, variables, values, record=[s])


@pytest.mark.parametrize("mode", MODES)
def test_write_final_default(mps_backend, mode):
    """``write_final`` is off exactly when the image row is recorded (Q10)."""
    use_mode(mode)
    optic, variables, values = case_radius()
    s = len(optic.surfaces.surfaces)

    image = run(optic, variables, values, record="image")
    assert_fused(image)
    assert image.final is None

    partial = run(optic, variables, values, record=[1])
    assert_fused(partial)
    assert partial.final is not None
    assert partial.final["x"].shape == (B, N_RAYS)

    both = run(optic, variables, values, record="all", write_final=True)
    assert_fused(both)
    assert both.final is not None
    # The final planes are the image row: same state, two buffers.
    for plane, attr in (("x", "x"), ("y", "y"), ("z", "z"), ("opd", "opd")):
        got = BT._select(both.final[plane], 3)
        want = BT._select(getattr(both, attr), 3, both.rows[s - 1])
        assert raw_equal(got, want), diff_report(got, want, f"final.{plane}")


def test_install_isolation(mps_backend):
    """``install`` writes design b's recorded rows and nothing else."""
    use_mode("df64")
    optic, variables, values = case_radius()
    result = run(optic, variables, values, record=[1, 7])
    assert_fused(result)

    trace_before = type(optic.surfaces).trace
    result.install(optic, 3)
    assert type(optic.surfaces).trace is trace_before, "install patched a method"

    for surface_index, row in result.rows.items():
        surface = optic.surfaces[surface_index]
        for attr in ATTRS:
            got = getattr(surface, attr)
            want = BT._select(getattr(result, attr), 3, row)
            assert raw_equal(got, want), diff_report(
                got, want, f"s{surface_index}.{attr}"
            )

    for surface_index in range(len(optic.surfaces.surfaces)):
        if surface_index in result.rows:
            continue
        assert be.size(optic.surfaces[surface_index].x) == 0

    # Installing another design replaces the rows; nothing accumulates.
    result.install(optic, 0)
    for surface_index, row in result.rows.items():
        got = optic.surfaces[surface_index].x
        want = BT._select(result.x, 0, row)
        assert raw_equal(got, want)


@pytest.mark.parametrize("mode", MODES)
def test_forced_tiny_slabs_identical(mps_backend, monkeypatch, mode):
    """A pathologically small ``MAX_STEPS`` changes the chunking, not a bit."""
    use_mode(mode)
    optic, variables, values = case_radius()
    whole = run(optic, variables, values, record="all")
    assert_fused(whole)
    assert stats_of(whole)["fused_trace:chunks"] == 1

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE_MAX_STEPS", "64")
    chunked = run(optic, variables, values, record="all")
    assert_fused(chunked)
    assert stats_of(chunked)["fused_trace:chunks"] > 1
    assert_rows_equal(whole, chunked)


def test_memory_budget_refusal(mps_backend, monkeypatch):
    """Over budget, the batch refuses with ``memory`` and falls back cleanly."""
    use_mode("df64")
    optic, variables, values = case_radius()
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE_MEMORY_FRACTION", "1e-9")
    refused = run(optic, variables, values, record="all")
    assert refused.fused is False
    # One refusal from the batch gate, plus one per design from the contract
    # loop's own traces once WP4's hook forwards them to the same gate.
    assert stats_of(refused)["fused_trace_skip:memory"] == 1 + (B if HOOKED else 0)
    assert "gpu:fused_trace" not in refused.stats

    monkeypatch.delenv("OPTILAND_METAL_FUSED_TRACE_MEMORY_FRACTION")
    fused = run(optic, variables, values, record="all")
    assert_fused(fused)
    assert_rows_equal(refused, fused)


def test_memory_refusal_raises_under_require(mps_backend, monkeypatch):
    """``require`` turns the feature refusal into an exception (plan 1.3)."""
    use_mode("df64")
    from optiland.backend.torch_backend.metal.trace import MetalFallbackError

    optic, variables, values = case_radius()
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE_MEMORY_FRACTION", "1e-9")
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "require")
    with pytest.raises(MetalFallbackError, match="memory"):
        run(optic, variables, values, record="all")


@pytest.mark.parametrize("mode", MODES)
def test_per_design_materials(mps_backend, monkeypatch, mode):
    """An index variable flows through ``material.n(w0)`` on the mutated object."""
    use_mode(mode)
    from optiland.backend.torch_backend.metal import trace_layout as L
    from optiland.backend.torch_backend.metal.trace_record import canonical_w0

    optic, variables, values = case_index()
    compiled = compile_designs(optic, variables, values, mode=mode, tier1=False)
    w0 = canonical_w0(TRACE_KWARGS["wavelength"], mode)
    assert compiled.records.w0 == w0

    # Surface 5's post index is design b's value, and surface 6's PRE index is
    # the same object (day-1 Q5 finding b).
    for b, physical in enumerate(INDEX_PHYSICAL):
        assert compiled.records.surf_real[b, 5, L.SR_NPOST] == physical
        assert compiled.records.surf_real[b, 6, L.SR_NPRE] == physical

    fused = run(optic, variables, values, record="all")
    assert_fused(fused)
    loop = contract_loop(optic, variables, values, monkeypatch, record="all")
    assert_rows_equal(fused, loop)
    # The designs are genuinely different traces, not one trace copied B times.
    assert not raw_equal(BT._select(fused.x, 0, 7), BT._select(fused.x, 4, 7))


# ---------------------------------------------------------------------------
# Fallbacks: other backends and the kill switch
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ["numpy", "torch-cpu"])
def test_runs_on_numpy_and_torch_cpu(backend):
    """The contract loop is the implementation on every non-Metal backend."""
    previous = be.get_backend()
    if backend == "numpy":
        be.set_backend("numpy")
    else:
        be.set_backend("torch")
        be.set_device("cpu")
        be.set_precision("float64")
        be.grad_mode.disable()
    try:
        optic, variables, values = case_radius()
        result = run(optic, variables, values, record="all")
        assert result.fused is False
        assert result.row_cache_used is False
        assert result.stats == {}
        s = len(optic.surfaces.surfaces)
        assert tuple(result.x.shape) == (B, s, N_RAYS)
        assert result.rows == {i: i for i in range(s)}

        # Against a hand-written per-design loop, value for value.
        originals = BT._current_values(variables)
        try:
            for b, row in enumerate(values):
                BT._apply_design(optic, variables, row)
                rays = BT._generate_bundle(optic, **TRACE_KWARGS)
                optic.surfaces.trace(rays, record=True)
                for surface_index in range(s):
                    want = optic.surfaces[surface_index].x
                    got = BT._select(result.x, b, surface_index)
                    assert np.array_equal(
                        np.asarray(be.to_numpy(got)),
                        np.asarray(be.to_numpy(want)),
                        equal_nan=True,
                    )
        finally:
            BT._restore(optic, variables, originals)
    finally:
        be.set_backend(previous)


def test_batch_honours_kill_switch(mps_backend, monkeypatch):
    """``=0`` runs the contract loop: no launch, no exception  [fix: L4.9]."""
    use_mode("df64")
    optic, variables, values = case_radius()
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    off = run(optic, variables, values, record="all")
    assert off.fused is False
    assert off.stats == {} or "gpu:fused_trace" not in off.stats
    assert not any(k.startswith("fused_trace") for k in off.stats)

    monkeypatch.delenv("OPTILAND_METAL_FUSED_TRACE")
    on = run(optic, variables, values, record="all")
    assert_fused(on)
    assert_rows_equal(off, on)


def test_tol_floor_scale_is_refused(mps_backend):
    """Day-1 Q3 decided the per-design re-trace; the alternative is not silent."""
    use_mode("df64")
    optic, variables, values = case_radius()
    with pytest.raises(NotImplementedError, match="tol_floor_scale"):
        run(optic, variables, values, tol_floor_scale=1.0)


# ---------------------------------------------------------------------------
# Late fallback and the trailing propagation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_batch_late_fallback_per_design(mps_backend, monkeypatch, mode):
    """One design crosses the Newton tolerance crossover; only it falls back."""
    use_mode(mode)
    optic, variables, values = trace_fixtures.long_path_batch_values()
    assert values.shape == (4, 1)
    kwargs = {"wavelength": 0.5876, "record": "all"}

    fused = run(optic, variables, values, **kwargs)
    assert_fused(fused)

    # df64's round-off floor overtakes tol at |t| = 3.5e3 mm, so the 5,000 mm
    # design crosses over and nothing else does; sf64's crossover is at 1.1e5.
    expected = [False, False, mode == "df64", False]
    assert fused.late_fallback_designs.tolist() == expected
    assert stats_of(fused).get("fused_trace:late_fallback", 0) == int(mode == "df64")

    # The plan's two comparisons (WP7): the fallen-back design against a
    # standalone per-op trace, the others against single-design fused traces.
    # The whole-batch tier-A comparison is deliberately NOT made here: this
    # fixture is a finite-radius Newton system, where the kernel currently
    # disagrees with the per-op path in the low word (module docstring).
    loop = contract_loop(optic, variables, values, monkeypatch, **kwargs)
    for b, fell_back in enumerate(expected):
        if fell_back:
            reference = loop
        else:
            reference = run(optic, variables, values[b : b + 1], **kwargs)
        if not fell_back:
            assert_fused(reference)
            assert reference.late_fallback_designs.tolist() == [False]
        for attr in ATTRS:
            for surface, row in fused.rows.items():
                got = BT._select(getattr(fused, attr), b, row)
                want = BT._select(getattr(reference, attr), b if fell_back else 0, row)
                assert raw_equal(got, want), diff_report(
                    got, want, f"design {b} surface {surface} {attr}"
                )

    # The optic is back where it started.
    assert float(optic.surfaces[1].thickness) == pytest.approx(5000.0, abs=0.0)


@pytest.mark.parametrize("mode", MODES)
def test_batch_nonzero_image_thickness(mps_backend, monkeypatch, mode):
    """``rays(b)`` carries the trailing propagate the tracer applies itself."""
    use_mode(mode)
    optic = trace_fixtures.nonzero_image_thickness()
    assert float(optic.surfaces[-1].thickness) == 5.0
    v = Variable(optic, "radius", surface_number=5)
    values = np.array([[v.variable.scale(r)] for r in (79.0, 79.68360, 80.5)])

    fused = run(optic, [v], values, record="all")
    assert_fused(fused)

    originals = BT._current_values([v])
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    try:
        for b, row in enumerate(values):
            BT._apply_design(optic, [v], row)
            want = optic.trace(
                TRACE_KWARGS["Hx"],
                TRACE_KWARGS["Hy"],
                TRACE_KWARGS["wavelength"],
                RINGS,
                "hexapolar",
            )
            got = fused.rays(b)
            for attr in ("x", "y", "z", "L", "M", "N", "i", "opd"):
                a, c = getattr(got, attr), getattr(want, attr)
                assert raw_equal(a, c), diff_report(a, c, f"rays({b}).{attr}")
    finally:
        monkeypatch.delenv("OPTILAND_METAL_FUSED_TRACE", raising=False)
        BT._restore(optic, [v], originals)


@pytest.mark.parametrize("mode", MODES)
def test_rays_from_final_and_from_image_row_agree(mps_backend, mode):
    """``rays(b)`` is the same bundle whichever buffer it reads."""
    use_mode(mode)
    optic, variables, values = case_radius()
    from_row = run(optic, variables, values, record="image")
    from_final = run(optic, variables, values, record="image", write_final=True)
    assert_fused(from_row)
    assert from_row.final is None and from_final.final is not None
    a, b = from_row.rays(2), from_final.rays(2)
    for attr in ("x", "y", "z", "L", "M", "N", "i", "opd"):
        assert raw_equal(getattr(a, attr), getattr(b, attr)), diff_report(
            getattr(a, attr), getattr(b, attr), attr
        )
    assert a.L0 is None and b.L0 is not None


def test_rms_spot_matches_the_operand_expression(mps_backend):
    """``rms_spot`` is ``RayOperand.rms_spot_size``' expression, per design."""
    use_mode("df64")
    optic, variables, values = case_radius()
    result = run(optic, variables, values, record="image")
    assert_fused(result)
    got = be.to_numpy(result.rms_spot())
    assert got.shape == (B,)
    for b in range(B):
        x = be.to_numpy(BT._select(result.x, b, 0))
        y = be.to_numpy(BT._select(result.y, b, 0))
        r2 = (x - np.nanmean(x)) ** 2 + (y - np.nanmean(y)) ** 2
        assert got[b] == pytest.approx(np.sqrt(np.nanmean(r2)), rel=1e-12)


# ---------------------------------------------------------------------------
# CONTRACT-row value tests (plan 3.7; named by trace_mirror.FINGERPRINTS)
# ---------------------------------------------------------------------------
def test_updater_applies_scaled_values(mps_backend):
    """``OpticUpdater``'s setters put the value the variable asked for.

    CONTRACT rows ``set_radius``, ``set_conic``, ``set_thickness``,
    ``set_asphere_coeff``, ``set_index`` and ``update``: their *values* are
    pinned here rather than their source hashed, so a refactor of the updater
    does not raise a false drift alarm but a change of meaning fails.
    """
    use_mode("df64")
    optic = trace_fixtures.cooke()

    def unscaled(variable, physical):
        """What ``Variable.update`` will hand the updater, exactly."""
        return float(variable.variable.inverse_scale(variable.variable.scale(physical)))

    radius = Variable(optic, "radius", surface_number=5)
    want = unscaled(radius, 81.5)
    BT._apply_design(optic, [radius], [radius.variable.scale(81.5)])
    assert float(optic.surfaces[5].geometry.radius) == want
    assert want == pytest.approx(81.5, rel=1e-15)

    conic = Variable(optic, "conic", surface_number=5)
    BT._apply_design(optic, [conic], [-0.75])
    assert float(optic.surfaces[5].geometry.k) == -0.75  # IdentityScaler

    thickness = Variable(optic, "thickness", surface_number=5)
    want = unscaled(thickness, 3.5)
    BT._apply_design(optic, [thickness], [thickness.variable.scale(3.5)])
    assert float(optic.surfaces[5].thickness) == want
    assert want == pytest.approx(3.5, rel=1e-15)

    index = Variable(optic, "index", surface_number=5, wavelength=0.55)
    want = unscaled(index, 1.7)
    BT._apply_design(optic, [index], [index.variable.scale(1.7)])
    material = optic.surfaces[5].material_post
    assert float(np.ravel(be.to_numpy(material.n(be.array([0.55]))))[0]) == want
    # set_index replaces the object and both sides see the replacement (Q5).
    assert optic.surfaces[6].material_pre is material

    asphere, _ = trace_fixtures.even_asphere_inf_radius()
    coeff = Variable(asphere, "asphere_coeff", surface_number=1, coeff_number=0)
    scaled = coeff.variable.scale(2.5e-4)
    want = float(coeff.variable.inverse_scale(scaled))
    BT._apply_design(asphere, [coeff], [scaled])
    stored = asphere.surfaces[1].geometry.coefficients[0]
    assert float(stored) == want
    assert want == pytest.approx(2.5e-4, rel=1e-12)
    # ``set_asphere_coeff`` stores the raw value (optic_updater.py:159-170), so
    # a backend scalar arrives as a 0-d tensor inside ``coefficients`` and the
    # adapter has to accept it (plan 3.9); the record must be the same either
    # way.
    from optiland.backend.torch_backend.metal.trace_record import compile_records

    as_float = compile_records(asphere.surfaces, 0.55, "df64", record="image")
    BT._apply_design(asphere, [coeff], [be.array(scaled)])
    stored_tensor = asphere.surfaces[1].geometry.coefficients[0]
    assert hasattr(stored_tensor, "ndim")
    as_tensor = compile_records(asphere.surfaces, 0.55, "df64", record="image")
    assert np.array_equal(as_float.coef, as_tensor.coef)
    assert np.array_equal(as_float.surf_real, as_tensor.surf_real, equal_nan=True)

    # `update()` is idempotent on a system with no pickups or solves (Q4).
    before = float(optic.surfaces[5].geometry.radius)
    optic.updater.update()
    optic.updater.update()
    assert float(optic.surfaces[5].geometry.radius) == before


def test_pose_refresh_after_update(mps_backend):
    """Thickness changes rebuild every downstream ``cs.z`` (CONTRACT row).

    ``SurfaceGroup._update_coordinate_systems`` / ``set_thickness`` keep
    ``cs.z[k]`` equal to the cumulative sum of the upstream thicknesses, which
    is exactly why a thickness variable invalidates every downstream record row
    (design 6.2).
    """
    use_mode("df64")
    optic = trace_fixtures.cooke()
    thickness = Variable(optic, "thickness", surface_number=2)
    BT._apply_design(optic, [thickness], [thickness.variable.scale(7.0)])

    surfaces = optic.surfaces.surfaces
    assert float(surfaces[1].geometry.cs.z) == 0.0
    cumulative = 0.0
    for k in range(2, len(surfaces)):
        cumulative += float(surfaces[k - 1].thickness)
        assert float(surfaces[k].geometry.cs.z) == pytest.approx(
            cumulative, rel=0.0, abs=1e-12
        )
    assert float(surfaces[2].thickness) == 7.0

    # And the record rows follow: every row from 2 on moved, none before it.
    before = compile_designs(
        trace_fixtures.cooke(),
        [],
        np.zeros((1, 0)),
        mode="df64",
        tier1=False,
    ).records
    after = compile_designs(
        optic, [], np.zeros((1, 0)), mode="df64", tier1=False
    ).records
    from optiland.backend.torch_backend.metal import trace_layout as L

    for k in range(1, 2):
        assert before.surf_real[0, k, L.SR_TZ] == after.surf_real[0, k, L.SR_TZ]
    for k in range(3, len(surfaces)):
        assert before.surf_real[0, k, L.SR_TZ] != after.surf_real[0, k, L.SR_TZ]


def test_variable_update_units(mps_backend):
    """``Variable.update`` takes scaled units; ``reset`` restores (CONTRACT)."""
    use_mode("df64")
    optic = trace_fixtures.cooke()
    radius = Variable(optic, "radius", surface_number=5)
    initial = radius.initial_value

    # RadiusVariable: LinearScaler(1/100, -1.0); the round trip is exact.
    physical = 81.5
    scaled = radius.variable.scale(physical)
    # LinearScaler evaluates ``value * factor + offset`` (linear.py:21-28); the
    # association is part of the contract, so the prediction uses it verbatim.
    assert scaled == physical * (1 / 100.0) + (-1.0)
    unscaled = radius.variable.inverse_scale(scaled)
    assert unscaled == (scaled - (-1.0)) / (1 / 100.0)
    assert unscaled == pytest.approx(physical, rel=1e-15)

    radius.update(scaled)
    assert float(radius.variable.get_value()) == unscaled
    assert radius.value == pytest.approx(scaled, rel=1e-15)

    radius.reset()
    assert radius.value == pytest.approx(initial, rel=1e-15)
    assert float(optic.surfaces[5].geometry.radius) == pytest.approx(79.68360, abs=1e-9)

    # ThicknessVariable: LinearScaler(1/10, -1.0).
    thickness = Variable(optic, "thickness", surface_number=5)
    assert thickness.variable.scale(3.5) == 3.5 * (1 / 10.0) + (-1.0)
    thickness.update(thickness.variable.scale(3.5))
    assert float(optic.surfaces[5].thickness) == pytest.approx(3.5, rel=1e-15)

    # A list or tuple is unwrapped to its first entry (variable.py:186-188).
    thickness.update([thickness.variable.scale(4.0)])
    assert float(optic.surfaces[5].thickness) == pytest.approx(4.0, rel=1e-15)


# ---------------------------------------------------------------------------
# Batched tolerancing consumers (plan 3.9, WP7 part 2)
# ---------------------------------------------------------------------------
#: The nominal radii of the two Cooke surfaces the perturbations sample around.
#: ``Perturbation`` uses an ``IdentityScaler`` and ``Variable.update`` *sets*
#: the value, so a sampler is centred on the nominal value, never on zero
#: (``tests/test_monte_carlo.py:23`` builds its samplers the same way).
TOL_R1, TOL_R5 = 22.01359, 79.68360

#: Samples per tolerancing run and designs per fused launch: 16 = 2 chunks of
#: 8, so the chunk loop of ``monte_carlo_batched`` really runs twice.
TOL_ITERATIONS, TOL_CHUNK = 16, 8


def _rms_operand_data(optic, **overrides):
    """``rms_spot_size`` input data at this file's launch settings."""
    data = {
        "optic": optic,
        "surface_number": -1,
        "Hx": TRACE_KWARGS["Hx"],
        "Hy": TRACE_KWARGS["Hy"],
        "num_rays": RINGS,
        "wavelength": TRACE_KWARGS["wavelength"],
        "distribution": "hexapolar",
    }
    data.update(overrides)
    return data


def build_tolerancing(*, extra_operand=None, compensator=False, seed=7):
    """A CookeTriplet with two seeded radius perturbations and one operand.

    A fresh ``Optic`` and fresh samplers every call, so the batched run and the
    sequential reference draw the same sequence from the same seeds.

    ``be.grad_mode.disable()`` comes first and is load-bearing: the previous
    ``Tolerancing`` in the same test left autograd **on**
    (``OptimizationProblem.__init__``), and an ``Optic`` built under grad holds
    ``requires_grad`` leaves, which the gate refuses *structurally* (plan 1.2)
    no matter what the trace later does.  This reproduces the natural order --
    build the system, then tolerance it -- rather than papering over it.
    """
    from optiland.tolerancing.core import Tolerancing
    from optiland.tolerancing.perturbation import DistributionSampler

    be.grad_mode.disable()
    optic = trace_fixtures.cooke()
    tolerancing = Tolerancing(optic)
    tolerancing.add_operand("rms_spot_size", input_data=_rms_operand_data(optic))
    if extra_operand is not None:
        kind, data = extra_operand
        tolerancing.add_operand(kind, input_data=dict(data, optic=optic))
    tolerancing.add_perturbation(
        "radius",
        DistributionSampler("normal", seed=seed, loc=TOL_R1, scale=0.05),
        surface_number=1,
    )
    tolerancing.add_perturbation(
        "radius",
        DistributionSampler("normal", seed=seed + 1, loc=TOL_R5, scale=0.20),
        surface_number=5,
    )
    if compensator:
        tolerancing.add_compensator("thickness", surface_number=6)
    return tolerancing


def sequential_monte_carlo(tolerancing, iterations, monkeypatch, *, grad_off=True):
    """``MonteCarlo.run`` with the kernel off and autograd off: the reference.

    ``grad_off=False`` is for the *refusal* comparisons, where the contract is
    "the existing loop runs completely unchanged": there the helper itself runs
    ``MonteCarlo.run`` under whatever grad setting ``Tolerancing`` left, so the
    reference must too.

    Autograd is disabled *after* the analysis object exists because
    ``Tolerancing``/``OptimizationProblem`` enables it on construction
    (``optimization/problem.py:63-66``).  Grad-off is the right reference and
    not a convenience: **measured**, the per-op Metal path is not
    grad-invariant in df64.  Same optic, same radii, same operand call, kernel
    off throughout, only ``be.grad_mode`` different: 53 of 1,141 image-row
    ``x`` low words and 35 ``y`` low words change, moving the RMS spot by
    1.7e-13 relative (sf64 is unaffected).  Plan section 7 puts grad-enabled
    traces on the per-op path by decision and the whole harness runs
    ``OPTILAND_TEST_MPS_GRAD=0``, so the fused path's contract is with the
    grad-off per-op path.  ``monte_carlo_batched`` matches by evaluating its
    batch under ``_forward_only`` (its deviation 4).
    """
    from optiland.tolerancing.monte_carlo import MonteCarlo

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    try:
        analysis = MonteCarlo(tolerancing)
        if grad_off:
            be.grad_mode.disable()
        analysis.run(iterations)
        return analysis.get_results()
    finally:
        monkeypatch.delenv("OPTILAND_METAL_FUSED_TRACE", raising=False)


def assert_frames_identical(got, want):
    """Same columns, same order, same float64 cells (NaN == NaN)."""
    assert list(got.columns) == list(want.columns), (
        f"columns differ: {list(got.columns)} vs {list(want.columns)}"
    )
    assert len(got) == len(want)
    for column in want.columns:
        a = np.asarray([float(v) for v in got[column]], dtype=np.float64)
        b = np.asarray([float(v) for v in want[column]], dtype=np.float64)
        bad = np.argwhere(~((a == b) | (np.isnan(a) & np.isnan(b))))
        assert not bad.size, (
            f"column {column!r} differs in {len(bad)} of {a.size} rows, first "
            f"at {int(bad[0][0])}: {a[bad[0][0]]!r} vs {b[bad[0][0]]!r}"
        )


@pytest.mark.parametrize("mode", MODES)
def test_monte_carlo_batched_matches_loop(mps_backend, monkeypatch, mode):
    """The batched frame equals ``MonteCarlo.run`` on mps per-op, cell for cell.

    Both frames are produced from the same seeds, so the perturbation columns
    are the same draws and the operand column is the same ``be.*`` expression
    on tier-A-identical records (plan 3.9).
    """
    use_mode(mode)
    from optiland.tolerancing import batched as TB

    problem = build_tolerancing()
    # The Tolerancing constructor turned autograd on (problem.py:63-66).
    assert be.grad_mode.requires_grad is True
    got = TB.monte_carlo_batched(problem, TOL_ITERATIONS, chunk=TOL_CHUNK)
    assert got.attrs["batched"] is True
    assert got.attrs["fused"] is True, (
        "monte_carlo_batched fell back to the contract loop, so this would "
        f"compare the per-op path with itself; reason: {got.attrs['reason']}"
    )
    # ``_forward_only`` traced without it and put the caller's setting back.
    assert be.grad_mode.requires_grad is True
    assert len(got) == TOL_ITERATIONS

    want = sequential_monte_carlo(build_tolerancing(), TOL_ITERATIONS, monkeypatch)
    assert_frames_identical(got, want)


@pytest.mark.parametrize("mode", MODES)
def test_sensitivity_batched_matches_loop(mps_backend, monkeypatch, mode):
    """``sensitivity_batched`` equals ``SensitivityAnalysis.run``, cell for cell.

    The sweep is one design per (perturbation, sample) pair with every other
    perturbed variable nominal, which is what ``reset()`` + a single
    ``apply()`` leaves the system as.
    """
    use_mode(mode)
    from optiland.tolerancing import batched as TB
    from optiland.tolerancing.core import Tolerancing
    from optiland.tolerancing.perturbation import RangeSampler
    from optiland.tolerancing.sensitivity_analysis import SensitivityAnalysis

    def build():
        be.grad_mode.disable()  # see build_tolerancing's docstring
        optic = trace_fixtures.cooke()
        tolerancing = Tolerancing(optic)
        tolerancing.add_operand("rms_spot_size", input_data=_rms_operand_data(optic))
        tolerancing.add_perturbation(
            "radius", RangeSampler(TOL_R1 - 0.1, TOL_R1 + 0.1, 3), surface_number=1
        )
        tolerancing.add_perturbation(
            "conic", RangeSampler(-0.1, 0.1, 3), surface_number=5
        )
        return tolerancing

    got = TB.sensitivity_batched(build(), chunk=4)
    assert got.attrs["batched"] is True
    assert got.attrs["fused"] is True
    assert len(got) == 6

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    try:
        analysis = SensitivityAnalysis(build())
        be.grad_mode.disable()  # see sequential_monte_carlo's docstring
        analysis.run()
        want = analysis.get_results()
    finally:
        be.grad_mode.disable()
        monkeypatch.delenv("OPTILAND_METAL_FUSED_TRACE", raising=False)

    assert list(got["perturbation_type"]) == list(want["perturbation_type"])
    assert_frames_identical(
        got[got.columns[1:]],
        want[want.columns[1:]],
    )


@pytest.mark.parametrize("mode", MODES)
def test_batched_operand_values_vary(mps_backend, mode):
    """Every design gets its own operand value  [fix: L1.2].

    The trap this closes: ``install()`` writing one design's rows and every
    operand reading the same ones would still produce a plausible frame.
    """
    use_mode(mode)
    from optiland.tolerancing import batched as TB

    frame = TB.monte_carlo_batched(build_tolerancing(), TOL_ITERATIONS, chunk=TOL_CHUNK)
    assert frame.attrs["fused"] is True
    values = np.asarray(frame["0: rms spot size"], dtype=np.float64)
    assert values.shape == (TOL_ITERATIONS,)
    assert np.isfinite(values).all(), "the fixture must not vignette any design"
    assert len(set(values.tolist())) == TOL_ITERATIONS, (
        f"{TOL_ITERATIONS} designs produced {len(set(values.tolist()))} distinct "
        "operand values"
    )
    # The perturbation columns vary too, so the designs really are different.
    for column in list(frame.columns)[:2]:
        assert len(set(np.asarray(frame[column], dtype=np.float64).tolist())) == (
            TOL_ITERATIONS
        )


def test_batched_refuses_unbatchable_operand(mps_backend, monkeypatch):
    """An operand outside ``BATCHABLE_OPERANDS`` runs the existing loop."""
    use_mode("df64")
    from optiland.tolerancing import batched as TB

    extra = (
        "real_x_intercept",
        {
            "surface_number": -1,
            "Hx": 0.0,
            "Hy": 0.0,
            "Px": 0.3,
            "Py": 0.0,
            "wavelength": TRACE_KWARGS["wavelength"],
        },
    )
    assert "real_x_intercept" not in TB.BATCHABLE_OPERANDS
    assert TB._refusal_reason(build_tolerancing(extra_operand=extra)) == (
        "operand_type:real_x_intercept"
    )

    iterations = 4
    got = TB.monte_carlo_batched(build_tolerancing(extra_operand=extra), iterations)
    assert got.attrs["batched"] is False
    assert got.attrs["reason"] == "operand_type:real_x_intercept"
    assert len(got) == iterations

    # grad_off=False: the refusal path runs the untouched loop, so does this.
    want = sequential_monte_carlo(
        build_tolerancing(extra_operand=extra),
        iterations,
        monkeypatch,
        grad_off=False,
    )
    assert_frames_identical(got, want)


def test_batched_refuses_compensators(mps_backend, monkeypatch):
    """A compensator is a nested optimisation per sample, not a design row."""
    use_mode("df64")
    from optiland.tolerancing import batched as TB

    assert TB._refusal_reason(build_tolerancing(compensator=True)) == (
        "compensator_variables"
    )

    iterations = 2
    got = TB.monte_carlo_batched(build_tolerancing(compensator=True), iterations)
    assert got.attrs["batched"] is False
    assert got.attrs["reason"] == "compensator_variables"
    assert any(name.startswith("C0:") for name in got.columns), (
        f"the compensator column is missing: {list(got.columns)}"
    )
    assert len(got) == iterations


def test_batched_refuses_wavelength_all_and_mixed(mps_backend):
    """``wavelength='all'`` and two wavelengths are both refused (plan 3.9)."""
    use_mode("df64")
    from optiland.tolerancing import batched as TB

    all_waves = build_tolerancing()
    all_waves.operands[0].input_data["wavelength"] = "all"
    assert TB._refusal_reason(all_waves) == "wavelength_all"

    mixed = build_tolerancing(
        extra_operand=("rms_spot_size", _rms_operand_data(None, wavelength=0.65))
    )
    assert TB._refusal_reason(mixed) == "mixed_wavelength"


def test_batchable_reader_mirrors_the_operand(mps_backend):
    """The reader is the operand's post-trace expression, to the last bit.

    Traced once per-op, the reader reads ``optic.surfaces`` exactly as
    ``RayOperand.rms_spot_size`` does after its own trace, so the two float64
    values are identical -- not merely close.
    """
    use_mode("df64")
    from optiland.optimization.operand.ray import RayOperand
    from optiland.tolerancing import batched as TB

    optic = trace_fixtures.cooke()
    data = _rms_operand_data(optic)
    want = float(
        RayOperand.rms_spot_size(
            optic,
            data["surface_number"],
            data["Hx"],
            data["Hy"],
            data["num_rays"],
            data["wavelength"],
            data["distribution"],
        )
    )
    got = float(TB.BATCHABLE_OPERANDS["rms_spot_size"].read(optic, data))
    assert got == want

    # ``rms_spot()`` on a one-design batch is the same number (design 6.4).
    v = Variable(optic, "radius", surface_number=5)
    values = np.array([[v.value]])
    result = run(optic, [v], values, record="all")
    assert_fused(result)
    result.install(optic, 0)
    assert float(TB.BATCHABLE_OPERANDS["rms_spot_size"].read(optic, data)) == want


def test_batched_honours_kill_switch(mps_backend, monkeypatch):
    """``=0`` still produces the right frame, through the contract loop."""
    use_mode("df64")
    from optiland.tolerancing import batched as TB

    iterations = 4
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    try:
        off = TB.monte_carlo_batched(build_tolerancing(), iterations)
    finally:
        monkeypatch.delenv("OPTILAND_METAL_FUSED_TRACE", raising=False)
    assert off.attrs["batched"] is True and off.attrs["fused"] is False

    on = TB.monte_carlo_batched(build_tolerancing(), iterations)
    assert on.attrs["fused"] is True
    assert_frames_identical(on, off)
