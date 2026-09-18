"""WP3: the fused-trace driver -- planning, launch, sentinel, writeback, stats.

Every assertion here predicts an exact result (plan 0.2.6): a slab plan that
tiles its grid exactly once with a *counted* number of slabs, raw-component
equality between a chunked and an unchunked launch, one NaN in one snapshot
row, exact counter deltas, exact refusal reasons.

Layering.  The driver sits between two lanes that are still landing, so these
tests attack it at the three layers that exist today and say so loudly:

1. **Host planning** (``_slab_plan``, the env readers, the late-fallback
   reduction) -- pure, always live.
2. **Launch** -- the real ten-buffer bind and dispatch against WP1's kernel.
   ``test_stub_roundtrip`` runs a *degenerate* record set (``S = 1``: the
   object row alone, no surface steps), which the physics body will reproduce
   exactly, so the test stays load-bearing after WP1's body lands.
3. **Driver / writeback** -- ``fused_trace`` end to end with a hand-built
   ``TraceRecords`` for CookeTriplet injected through ``trace._trace_record``
   (WP2's ``metal/trace_record.py`` does not exist yet; plan 4/WP3
   "hand-built ``TraceRecords`` for CookeTriplet until WP2 lands").  The
   assertions are structural -- shapes, frames, identity, aliasing, counters,
   refusals -- never "the numbers are right", which only the per-op
   comparison can decide.

The tests that *must* compare against the per-op path are marked ``skipif`` on
what they need and go live by themselves; the skip reason names what is
missing.  ``NEEDS_PHYSICS`` is WP1's kernel body (live from I1);
``NEEDS_HOOK`` is WP4's ``SurfaceGroup.trace`` hook (wave 2), which only
``test_trailing_propagate_unchanged`` needs -- everything else drives
``trace.fused_trace`` directly, so it cannot pass by comparing the per-op path
with itself.
"""

from __future__ import annotations

import copy
import gc
import inspect
import os

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

from dataclasses import dataclass, field  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

torch = pytest.importorskip("torch")
if not torch.backends.mps.is_available():  # pragma: no cover - hardware gate
    pytest.skip("Metal GPU required", allow_module_level=True)

import optiland.backend as be  # noqa: E402
from optiland.backend.torch_backend import metal  # noqa: E402
from optiland.backend.torch_backend.metal import compile as _compile  # noqa: E402
from optiland.backend.torch_backend.metal import (  # noqa: E402
    library,
    trace,
    trace_layout,
    trace_mirror,
)
from optiland.backend.torch_backend.metal import tensor as T  # noqa: E402
from optiland.backend.torch_backend.metal.tensor import MetalFloat64  # noqa: E402
from optiland.backend.torch_backend.metal.trace_adapters import (  # noqa: E402
    FusedTraceSkip,
)
from optiland.rays import RealRays  # noqa: E402
from optiland.surfaces.surface_group import SurfaceGroup  # noqa: E402

MODES = ("df64", "sf64")

#: ``trace_body`` is still the placeholder while it declares that it reads no
#: table: the stub ends with ``(void)surf_real;`` precisely because a stub
#: cannot read ``surf_real`` / ``coef`` / ``consts``.  That line disappears the
#: moment the physics body lands, which is when a per-op comparison starts to
#: mean something.
_TRACE_METAL = _compile._KERNEL_DIR / "trace.metal"
PHYSICS = "(void)surf_real;" not in _TRACE_METAL.read_text(encoding="utf-8")
NEEDS_PHYSICS = pytest.mark.skipif(
    not PHYSICS,
    reason=(
        "kernels/trace.metal still carries the WP1 placeholder trace_body "
        "(it reads no surface table); a per-op comparison would compare the stub"
    ),
)

#: A test that reaches the driver through ``optic.trace`` goes through WP4's
#: hook in ``SurfaceGroup.trace`` (plan 3.4), which is wave 2.  While the hook
#: is absent the fused leg of such a test silently runs the per-op loop, so the
#: test compares the per-op path with itself and PASSES while asserting
#: nothing.  That is worse than a red test, so it is skipped -- loudly, naming
#: the missing hook -- until ``SurfaceGroup.trace`` actually calls it.  The
#: driver-level tests below do not need the hook and stay live at I1.
HOOKED = "_fused_metal_trace" in inspect.getsource(SurfaceGroup.trace)
NEEDS_HOOK = pytest.mark.skipif(
    not HOOKED,
    reason=(
        "SurfaceGroup.trace does not call _fused_metal_trace yet (WP4, wave 2); "
        "an optic.trace comparison would compare the per-op path with itself"
    ),
)

N_RAYS = 1024  # > HOST_THRESHOLD (256), so every plane is GPU-resident


# ---------------------------------------------------------------------------
# Fixtures
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
    """torch / mps / float64 with autograd off (the fused path's setting)."""
    previous = metal.get_mode()
    be.set_backend("torch")
    be.set_device("mps")
    be.set_precision("float64")
    be.grad_mode.disable()
    yield
    metal.set_mode(previous)
    be.grad_mode.disable()
    be.set_backend("numpy")


# ---------------------------------------------------------------------------
# Hand-built records (WP2's TraceRecords, duck-typed)
# ---------------------------------------------------------------------------


@dataclass
class HandRecords:
    """The fields of ``TraceRecords`` the driver reads (plan 3.3).

    Only the *structure* is hand-built: the stub kernel never reads
    ``surf_real`` or ``coef``, and WP2 owns the values.
    """

    mode: str
    B: int
    S: int
    C: int
    surf_int: np.ndarray
    surf_real: np.ndarray
    coef: np.ndarray
    snap_rows: np.ndarray
    step_cost: np.ndarray
    n_rows: int
    has_newton: bool
    weighted_steps: int
    w0: float = 0.55


def hand_records(
    mode: str,
    S: int,
    *,
    B: int = 1,
    C: int = 0,
    record: bool = True,
    max_iter: int = 0,
    newton_rows: tuple[int, ...] = (),
) -> HandRecords:
    """Tables for ``S`` surfaces: row 0 the object surface, the rest planes.

    ``newton_rows`` marks rows whose ``step_cost`` is ``1 + max_iter``
    (plan 3.3 ``step_cost``), which is what the weighted chunking reacts to.
    """
    surf_int = np.zeros((B, S, trace_layout.SI_STRIDE), dtype=np.int32)
    surf_int[:, 0, trace_layout.SI_GEOM] = trace_layout.GEOM_OBJECT
    surf_int[:, 1:, trace_layout.SI_GEOM] = trace_layout.GEOM_PLANE
    step_cost = np.ones(S, dtype=np.int32)
    for s in newton_rows:
        surf_int[:, s, trace_layout.SI_GEOM] = trace_layout.GEOM_EVEN
        surf_int[:, s, trace_layout.SI_MAXITER] = max_iter
        step_cost[s] = 1 + max_iter
    if record:
        rows = np.tile(np.arange(S, dtype=np.int32), (B, 1))
        n_rows = S
    else:
        rows = np.full((B, S), -1, dtype=np.int32)
        n_rows = 0
    surf_int[:, :, trace_layout.SI_SNAPROW] = rows
    surf_int[:, :, trace_layout.SI_STEPCOST] = step_cost
    return HandRecords(
        mode=mode,
        B=B,
        S=S,
        C=C,
        surf_int=surf_int,
        surf_real=np.zeros((B, S, trace_layout.SR_STRIDE)),
        coef=np.zeros((B, S, C)),
        snap_rows=rows,
        step_cost=step_cost,
        n_rows=n_rows,
        has_newton=bool(newton_rows),
        weighted_steps=int(step_cost[1:].sum()),
    )


@dataclass
class FakeGate:
    """A ``GateResult`` (plan 3.3)."""

    ok: bool
    reason: Any = None
    structural: bool = False
    mode: str | None = None
    w0: float | None = None
    n: int = 0
    s: int = 0


@dataclass
class FakeRecordModule:
    """Stands in for WP2's ``metal/trace_record.py`` (plan 4/WP3 dependency)."""

    records: HandRecords
    gate: FakeGate
    calls: list[str] = field(default_factory=list)
    bytes_per_trace: int = 0

    def can_fuse_trace(self, group, rays, skip, *, wavelength=None):  # noqa: D102
        self.calls.append("can_fuse_trace")
        return self.gate

    def compile_records(self, group, w0, mode, *, record=True, designs=1):  # noqa: D102
        self.calls.append(f"compile_records(record={record})")
        if not record:
            blank = hand_records(mode, self.records.S, B=self.records.B, record=False)
            return blank
        return self.records

    def memory_bytes(self, records, N, launch_designs, write_final):  # noqa: D102
        self.calls.append("memory_bytes")
        return self.bytes_per_trace


def cooke_group(mode: str):
    """A live CookeTriplet surface group under the current backend."""
    from optiland.samples.objectives import CookeTriplet

    metal.set_mode(mode)
    lens = CookeTriplet()
    return lens, lens.surfaces


def make_rays(n: int = N_RAYS, seed: int = 11, wavelength: float = 0.55) -> RealRays:
    """A GPU-resident bundle of ``n`` plausible rays.

    The wavelength plane is broadcast to ``n`` because that is what
    ``RayGenerator.generate_rays`` produces (ray_generator.py:91) and what
    WP2's gate requires: every plane the same GPU-resident ``(n,)`` shape.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-8.0, 8.0, n)
    y = rng.uniform(-8.0, 8.0, n)
    z = rng.uniform(-40.0, -1.0, n)
    ell = rng.uniform(-0.3, 0.3, n)
    em = rng.uniform(-0.3, 0.3, n)
    nn = np.sqrt(1.0 - ell * ell - em * em)
    return RealRays(
        be.array(x),
        be.array(y),
        be.array(z),
        be.array(ell),
        be.array(em),
        be.array(nn),
        be.array(np.ones(n)),
        be.array(np.full(n, wavelength)),
    )


def raw(value: Any) -> list[np.ndarray]:
    """Host copies of a ``MetalFloat64``'s raw components (never decoded)."""
    return [c.detach().cpu().numpy() for c in value.components]


#: The eight per-surface snapshot planes and the eleven final planes the
#: driver writes back (plan 3.2 ``S_*`` / ``F_*``).
SNAP_ATTRS = ("x", "y", "z", "L", "M", "N", "intensity", "opd")
FINAL_ATTRS = ("x", "y", "z", "L", "M", "N", "i", "opd", "L0", "M0", "N0")


def copy_rays(rays: RealRays) -> RealRays:
    """A second bundle holding the SAME encoded words as ``rays``.

    Component-level ``be.copy``, never a ``to_numpy`` / ``be.array`` round
    trip: re-encoding shifts the df64 low word, which would make a tier-A
    comparison compare two different launch bundles (WP1-c finding 4).
    """
    out = RealRays(
        *(be.copy(getattr(rays, a)) for a in ("x", "y", "z", "L", "M", "N", "i", "w"))
    )
    out.opd = be.copy(rays.opd)
    return out


def pupil_rays(
    optic: Any,
    n: int = N_RAYS,
    *,
    Hx: float = 0.0,
    Hy: float = 0.0,
    wavelength: float = 0.55,
) -> RealRays:
    """``n`` rays through ``optic``'s pupil, from the optic's own generator.

    A golden-angle spiral over the unit pupil disc, so the bundle is exactly
    ``n`` rays and deterministic.  The call goes through
    ``RayGenerator.generate_rays`` -> ``paraxial.EPL()``, which is why the
    optic must carry an aperture stop: without one this raises ``ValueError``
    before any trace runs.
    """
    k = np.arange(n, dtype=np.float64)
    r = np.sqrt((k + 0.5) / n)
    theta = k * (np.pi * (3.0 - np.sqrt(5.0)))
    generator = optic.ray_tracer.ray_generator
    return generator.generate_rays(
        be.array(np.full(n, Hx)),
        be.array(np.full(n, Hy)),
        be.array(r * np.cos(theta)),
        be.array(r * np.sin(theta)),
        wavelength,
    )


def compare_raw(got: list[np.ndarray], ref: list[np.ndarray], what: str) -> None:
    """Tier-A equality (plan 7.1): raw components, never a tolerance."""
    assert len(got) == len(ref), f"{what}: {len(got)} components vs {len(ref)}"
    for c, (g, r) in enumerate(zip(got, ref, strict=True)):
        assert g.shape == r.shape, f"{what}: component {c} {g.shape} vs {r.shape}"
        if not np.array_equal(g, r, equal_nan=True):
            bad = np.flatnonzero(~((np.isnan(g) & np.isnan(r)) | (g == r)))
            raise AssertionError(
                f"{what}: component {c} differs on {bad.size}/{g.size} entries; "
                f"first indices {bad[:8].tolist()}"
            )


def install_fake(monkeypatch, records: HandRecords, gate: FakeGate) -> FakeRecordModule:
    """Route ``trace._trace_record()`` at a hand-built record compiler."""
    fake = FakeRecordModule(records=records, gate=gate)
    monkeypatch.setattr(trace, "_trace_record", lambda: fake)
    return fake


def delta(before: dict[str, int]) -> dict[str, int]:
    """Counter deltas since ``before``."""
    now = T.stats()
    keys = set(now) | set(before)
    return {
        k: now.get(k, 0) - before.get(k, 0)
        for k in keys
        if now.get(k, 0) != before.get(k, 0)
    }


# ---------------------------------------------------------------------------
# 1. Host planning: the slab plan is the single source of the chunking
# ---------------------------------------------------------------------------

CHUNK = library.DEFAULT_CHUNK
MAX = trace.DEFAULT_MAX_STEPS


@pytest.mark.parametrize(
    ("B", "N", "ws", "max_steps", "expected"),
    [
        # one design, Cooke-shaped: 7 weighted steps per path, one slab
        (1, 100_000, 7, MAX, 1),
        # the forced-tiny budget of test_chunking_identical: 4096 // 7 = 585
        (1, 100_000, 7, 4096, (100_000 + 584) // 585),
        # ten designs, tiny budget: B_chunk collapses to 1, 2 ray slabs each
        (10, 1_000, 7, 4096, 10 * 2),
        # a budget below one single step still yields a 1x1 slab per cell
        (3, 4, 1_000_000, 1, 12),
        # 44-row spherical vs the same system with two aspheres (see below)
        (1, 1_000_000, 43, MAX, 1),
        (1, 1_000_000, 243, MAX, 4),
    ],
)
def test_slab_plan_counts_are_predicted(B, N, ws, max_steps, expected):
    plan = trace._slab_plan(B, N, ws, max_steps, CHUNK)
    assert len(plan) == expected


@pytest.mark.parametrize(
    ("B", "N", "ws", "max_steps"),
    [(1, 1000, 7, MAX), (1, 20_000, 7, 4096), (7, 999, 11, 4096), (3, 4, 10**6, 1)],
)
def test_slab_plan_tiles_the_grid_exactly_once(B, N, ws, max_steps):
    seen = np.zeros((B, N), dtype=np.int64)
    for design_base, b_extent, ray_base, n_extent in trace._slab_plan(
        B, N, ws, max_steps, CHUNK
    ):
        assert b_extent >= 1 and n_extent >= 1
        assert design_base + b_extent <= B
        assert ray_base + n_extent <= N
        seen[design_base : design_base + b_extent, ray_base : ray_base + n_extent] += 1
    assert np.array_equal(seen, np.ones((B, N), dtype=np.int64))


def test_slab_plan_weights_newton_rows():
    """[fix: L4.5]: an asphere row costs ``1 + max_iter``, so it chunks finer."""
    spherical = hand_records("df64", 44)
    aspheric = hand_records("df64", 44, max_iter=100, newton_rows=(7, 21))
    assert spherical.weighted_steps == 43
    assert aspheric.weighted_steps == 41 + 2 * 101 == 243
    n = 1_000_000
    plan_s = trace._slab_plan(1, n, spherical.weighted_steps, MAX, CHUNK)
    plan_a = trace._slab_plan(1, n, aspheric.weighted_steps, MAX, CHUNK)
    assert len(plan_s) == 1
    assert len(plan_a) == 4
    assert len(plan_a) > len(plan_s)


def test_slab_plan_empty_grid():
    assert trace._slab_plan(0, 10, 7, MAX, CHUNK) == []
    assert trace._slab_plan(10, 0, 7, MAX, CHUNK) == []


def test_env_semantics(monkeypatch):
    """Every switch is read per call, with the documented default."""
    for name in (
        "OPTILAND_METAL_FUSED_TRACE",
        "OPTILAND_METAL_FUSED_TRACE_MAX_STEPS",
        "OPTILAND_METAL_FUSED_TRACE_MIN_RAYS",
        "OPTILAND_METAL_FUSED_TRACE_MEMORY_FRACTION",
        "OPTILAND_METAL_FUSED_TRACE_GROUP",
        "OPTILAND_METAL_FUSED_TRACE_DRIFT",
        "OPTILAND_METAL_TRACE_DIAG",
    ):
        monkeypatch.delenv(name, raising=False)
    assert trace._switch() == "1"
    assert trace._max_steps() == 2**26
    assert trace._min_rays() == 0
    assert trace._memory_fraction() == 0.25
    assert trace._group_size() is None
    assert trace._drift_policy() == "refuse"
    assert trace._diag() is False

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "require")
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE_MAX_STEPS", "4096")
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE_MIN_RAYS", "512")
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE_MEMORY_FRACTION", "0.5")
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE_GROUP", "64")
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE_DRIFT", "warn")
    monkeypatch.setenv("OPTILAND_METAL_TRACE_DIAG", "1")
    assert trace._switch() == "require"
    assert trace._max_steps() == 4096
    assert trace._min_rays() == 512
    assert trace._memory_fraction() == 0.5
    assert trace._group_size() == 64
    assert trace._drift_policy() == "warn"
    assert trace._diag() is True

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE_MAX_STEPS", "0")
    with pytest.raises(ValueError, match="MAX_STEPS"):
        trace._max_steps()
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE_GROUP", "96")
    with pytest.raises(ValueError, match="power of two"):
        trace._group_size()


def test_late_fallback_mask_is_per_design():
    """The reduction is per design, and is skipped when no row is Newton."""
    status = torch.zeros(3, 4, 8, dtype=torch.uint8, device="mps")
    status[1, 2, 5] = trace_layout.ST_TOL_CROSSOVER
    status[2, 0, 0] = trace_layout.ST_MISS  # a different bit must not count
    assert np.array_equal(
        trace._late_fallback_mask(status, True), np.array([False, True, False])
    )
    assert np.array_equal(
        trace._late_fallback_mask(status, False), np.array([False, False, False])
    )


def test_provenance_stamp():
    """``table_hash()`` is a stable 64-hex stamp for the oracle/suite JSONs."""
    h1 = trace_mirror.table_hash()
    h2 = trace_mirror.table_hash()
    assert h1 == h2
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)


# ---------------------------------------------------------------------------
# 2. Library and launch packing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_include_order_is_pinned(mode):
    """conic.metal before trace.metal, trace_layout.h before trace.metal."""
    src = trace.trace_source(mode)
    i_conic = src.index("OPTILAND_CONIC_METAL")
    i_layout = src.index("OPTILAND_TRACE_LAYOUT_H")
    i_trace = src.index("OPTILAND_TRACE_METAL")
    assert i_conic < i_trace
    assert i_layout < i_trace
    assert trace.TRACE_FILES == ("conic.metal", "trace_layout.h", "trace.metal")


@pytest.mark.parametrize("mode", MODES)
def test_library_is_cached_and_warmed_up(mps_backend, mode):
    """One library per mode per process, after ``get_library`` has self-tested."""
    lib = trace._kernel_library(mode)
    assert trace._kernel_library(mode) is lib
    assert mode in library._LIBRARIES  # get_library(mode) ran the self-test
    assert hasattr(lib, trace.ENTRY[mode])
    # The warm-up dispatch is not a user trace: it must not be counted.
    assert not any(k.startswith("fused_trace") for k in T.stats())


def test_check_component_rejects_every_unsafe_tensor():
    good = torch.zeros(8, dtype=torch.float32, device="mps")
    trace._check_component(good, torch.float32, "ok")  # does not raise
    with pytest.raises(TypeError):
        trace._check_component([1.0], torch.float32, "list")
    with pytest.raises(TypeError):
        trace._check_component(
            MetalFloat64.from_numpy(np.zeros(8), "df64", host=False),
            torch.float32,
            "wrapper",
        )
    with pytest.raises(ValueError, match="mps"):
        trace._check_component(
            torch.zeros(8, dtype=torch.float32), torch.float32, "cpu"
        )
    with pytest.raises(TypeError, match="dtype"):
        trace._check_component(good, torch.int64, "dtype")
    with pytest.raises(ValueError, match="contiguous"):
        trace._check_component(
            torch.zeros(8, 2, dtype=torch.float32, device="mps")[:, 0],
            torch.float32,
            "strided",
        )
    with pytest.raises(ValueError, match="negated"):
        trace._check_component(torch._neg_view(good), torch.float32, "neg")


@pytest.mark.parametrize("mode", MODES)
def test_pack_launch_matches_the_ray_components(mps_backend, mode):
    """The nine planes are concatenated in ``Q_*`` order, bit for bit."""
    metal.set_mode(mode)
    rays = make_rays()
    packed = trace._pack_launch(rays, mode, N_RAYS)
    ncomp = 2 if mode == "df64" else 1
    assert len(packed) == ncomp
    for k, buf in enumerate(packed):
        got = buf.cpu().numpy().reshape(trace_layout.Q_PLANES, N_RAYS)
        for q, attr in enumerate(trace._LAUNCH_ATTRS):
            want = raw(getattr(rays, attr))[k]
            if want.size == 1:  # the uniform wavelength plane is broadcast
                want = np.repeat(want, N_RAYS)
            assert np.array_equal(got[q], want), (attr, k)


@pytest.mark.parametrize("mode", MODES)
def test_pack_launch_broadcasts_a_uniform_plane(mps_backend, mode):
    """A one-element plane is broadcast, and a wrong-length one is refused."""
    metal.set_mode(mode)
    rays = make_rays()
    rays.w = be.array(np.full(1, 0.55))  # the pre-broadcast shape
    packed = trace._pack_launch(rays, mode, N_RAYS)
    got = packed[0].cpu().numpy().reshape(trace_layout.Q_PLANES, N_RAYS)
    want = raw(rays.w)[0]
    assert want.size == 1
    assert np.array_equal(got[trace_layout.Q_W], np.repeat(want, N_RAYS))

    rays.w = be.array(np.full(N_RAYS // 2, 0.55))
    with pytest.raises(ValueError, match="rays.w"):
        trace._pack_launch(rays, mode, N_RAYS)


# ---------------------------------------------------------------------------
# 3. Launch: the ten bindings, the slabs and the sentinel
# ---------------------------------------------------------------------------


def run_records(records: HandRecords, mode: str, n: int, *, write_final: bool = True):
    """``launch_trace`` on a fresh bundle.

    Returns ``(result, launch_raw, counts)`` where ``counts`` is the counter
    delta of the launch alone: building the bundle is not part of what the
    driver's counters are asserted on.
    """
    metal.set_mode(mode)
    trace._kernel_library(mode)
    rays = make_rays(n)
    launch = trace._pack_launch(rays, mode, n)
    launch_raw = [b.cpu().numpy().reshape(trace_layout.Q_PLANES, n) for b in launch]
    before = T.stats()
    result = trace.launch_trace(
        records, launch, launch_stride=0, N=n, write_final=write_final, mode=mode
    )
    return result, launch_raw, delta(before)


@pytest.mark.parametrize("mode", MODES)
def test_stub_roundtrip(mps_backend, mode):
    """The degenerate trace (object row only, no surface steps) is the identity.

    ``S = 1`` means the kernel writes the object row and nothing else, in the
    stub and in the physics body alike (plan 3.2 object-row contract), so this
    smoke test on the full ten-buffer path is permanent.
    """
    records = hand_records(mode, 1)
    assert records.weighted_steps == 0
    result, launch_raw, counts = run_records(records, mode, N_RAYS)

    assert result.launches == len(trace._slab_plan(1, N_RAYS, 0, MAX, CHUNK)) == 1
    assert counts == {"gpu:fused_trace": 1, "fused_trace:chunks": 1}

    # final == launch on the eight state planes, on RAW COMPONENTS (day-1 P1)
    for k, comp in enumerate(result.final):
        got = comp.cpu().numpy()
        for q_final, q_launch in enumerate(
            (
                trace_layout.Q_X,
                trace_layout.Q_Y,
                trace_layout.Q_Z,
                trace_layout.Q_L,
                trace_layout.Q_M,
                trace_layout.Q_N,
                trace_layout.Q_I,
                trace_layout.Q_OPD,
            )
        ):
            assert np.array_equal(got[q_final, 0], launch_raw[k][q_launch]), q_final
    # the object row is the launch state
    for k, comp in enumerate(result.snap):
        got = comp.cpu().numpy()
        for q in range(trace_layout.S_PLANES):
            assert np.array_equal(got[q, 0, 0], launch_raw[k][q]), q

    assert int(result.status.sum().item()) == 0
    assert not bool((result.iters == trace_layout.ITERS_UNWRITTEN).any().item())
    assert not result.late_fallback_designs.any()


@pytest.mark.parametrize("mode", MODES)
def test_chunking_identical(mps_backend, monkeypatch, mode):
    """A forced tiny ``MAX_STEPS`` changes the slab count, never the bytes."""
    n = 20_000
    records = hand_records(mode, 8)
    assert records.weighted_steps == 7
    monkeypatch.delenv("OPTILAND_METAL_FUSED_TRACE_MAX_STEPS", raising=False)
    whole, _, _ = run_records(records, mode, n)
    assert whole.launches == 1

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE_MAX_STEPS", "4096")
    chunked, _, counts = run_records(records, mode, n)
    predicted = len(trace._slab_plan(1, n, 7, 4096, CHUNK))
    assert predicted == (n + 584) // 585 == 35
    assert chunked.launches == predicted
    assert counts == {
        "gpu:fused_trace": predicted,
        "fused_trace:chunks": predicted,
    }
    for a, b in zip(whole.snap, chunked.snap, strict=True):
        assert np.array_equal(a.cpu().numpy(), b.cpu().numpy())
    for a, b in zip(whole.final, chunked.final, strict=True):
        assert np.array_equal(a.cpu().numpy(), b.cpu().numpy())


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("switch", ["1", "require"])
def test_iters_sentinel_detects_unvisited(mps_backend, monkeypatch, mode, switch):
    """A dropped slab is caught by the sentinel in EVERY switch setting.

    Day-1 P9 measured the failure this guards: a command buffer the driver
    aborts returns success from ``torch.mps.synchronize()`` with a million
    cells unwritten.
    """
    n = 20_000
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", switch)
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE_MAX_STEPS", "4096")
    records = hand_records(mode, 8)

    full, _, _ = run_records(records, mode, n)
    assert not bool((full.iters == trace_layout.ITERS_UNWRITTEN).any().item())

    real = trace._slab_plan
    monkeypatch.setattr(trace, "_slab_plan", lambda *a, **k: real(*a, **k)[:-1])
    before = T.stats()
    with pytest.raises(trace.MetalFallbackError, match="sentinel"):
        run_records(records, mode, n)
    assert delta(before).get("fused_trace:unvisited") == 1


@pytest.mark.parametrize("mode", MODES)
def test_launch_stride_contract(mps_backend, mode):
    """Only the two layouts the kernel derives ``Lb`` from are accepted."""
    metal.set_mode(mode)
    records = hand_records(mode, 4, B=2)
    launch = trace._pack_launch(make_rays(512), mode, 512)
    with pytest.raises(ValueError, match="launch_stride"):
        trace.launch_trace(
            records, launch, launch_stride=2, N=512, write_final=False, mode=mode
        )
    with pytest.raises(ValueError, match="elements"):
        # stride 1 needs Lb = B = 2 bundles; only one was packed
        trace.launch_trace(
            records, launch, launch_stride=1, N=512, write_final=False, mode=mode
        )


# ---------------------------------------------------------------------------
# 4. The driver: gate, refusals, writeback
# ---------------------------------------------------------------------------


def fused_cooke(monkeypatch, mode: str, *, record: bool = True, **gate_kw):
    """Run ``fused_trace`` on CookeTriplet with hand-built records."""
    lens, group = cooke_group(mode)
    s = len(group.surfaces)
    records = hand_records(mode, s, record=record)
    gate = FakeGate(ok=True, mode=mode, w0=0.55, n=N_RAYS, s=s, **gate_kw)
    fake = install_fake(monkeypatch, records, gate)
    rays = make_rays()
    group.reset()  # the hook does this before calling the driver (plan 3.4)
    return lens, group, rays, records, fake


@pytest.mark.parametrize("mode", MODES)
def test_records_shape_and_frame(mps_backend, monkeypatch, mode):
    """Eight attributes per surface, shape (N,), all rows present, object row 0."""
    lens, group, rays, records, _ = fused_cooke(monkeypatch, mode)
    launch = {a: raw(getattr(rays, a)) for a in ("x", "y", "z", "L", "M", "N")}

    assert trace.fused_trace(group, rays, 0, True) is True

    assert len(group.surfaces) == records.S
    for surf in group.surfaces:
        for attr in ("x", "y", "z", "L", "M", "N", "intensity", "opd"):
            value = getattr(surf, attr)
            assert isinstance(value, MetalFloat64), attr
            assert value.mode == mode
            assert tuple(value.shape) == (N_RAYS,), attr
            assert be.size(value) == N_RAYS
        # a paraxial-only attribute and the AOI stay empty (design 2.6)
        assert be.size(surf.aoi) == 0
        assert be.size(surf.u) == 0

    # every row survives SurfaceGroup's `size > 0` filter: none is dropped
    assert tuple(be.to_numpy(group.x).shape) == (records.S, N_RAYS)

    # row 0 IS the launch state (design 5), compared on raw components
    for attr, want in launch.items():
        got = raw(getattr(group.surfaces[0], attr))
        for k in range(len(want)):
            assert np.array_equal(got[k], want[k]), attr


@pytest.mark.parametrize("mode", MODES)
def test_rays_identity_and_l0(mps_backend, monkeypatch, mode):
    """``rays`` keeps its identity; L0/M0/N0 are written; ``w`` is untouched."""
    lens, group, rays, records, _ = fused_cooke(monkeypatch, mode)
    same = rays
    w_before = rays.w
    normalized_before = rays.is_normalized

    assert trace.fused_trace(group, rays, 0, True) is True

    assert rays is same
    assert rays.w is w_before
    assert rays.is_normalized is normalized_before
    for attr in ("x", "y", "z", "L", "M", "N", "i", "opd", "L0", "M0", "N0"):
        value = getattr(rays, attr)
        assert isinstance(value, MetalFloat64), attr
        assert tuple(value.shape) == (N_RAYS,), attr
    # the eleven final planes are one storage, distinct from the snapshots
    ptr = rays.x.components[0].untyped_storage().data_ptr()
    assert rays.N0.components[0].untyped_storage().data_ptr() == ptr
    assert group.surfaces[0].x.components[0].untyped_storage().data_ptr() != ptr


@pytest.mark.parametrize("mode", MODES)
def test_record_false_raises_on_stack(mps_backend, monkeypatch, mode):
    """``record=False``: no snapshots, ``SurfaceGroup.x`` raises as today."""
    lens, group, rays, records, fake = fused_cooke(monkeypatch, mode, record=False)

    assert trace.fused_trace(group, rays, 0, False) is True
    assert "compile_records(record=False)" in fake.calls

    for surf in group.surfaces:
        assert be.size(surf.x) == 0
        assert be.size(surf.intensity) == 0

    # ``SurfaceGroup.x`` fails exactly as it does on an untraced group: the
    # concrete type is the backend's (RuntimeError under torch, ValueError
    # under numpy), so the control decides it rather than this test.
    control = cooke_group(mode)[1]
    control.reset()
    with pytest.raises(Exception) as untraced:  # noqa: B017 - type is the assertion
        _ = control.x
    with pytest.raises(type(untraced.value)):
        _ = group.x

    # the rays are still written
    assert tuple(rays.x.shape) == (N_RAYS,)


@pytest.mark.parametrize("mode", MODES)
def test_inplace_mutation_isolated(mps_backend, monkeypatch, mode):
    """``surfaces[0].x[0] = nan`` lands in exactly one cell of the snap storage."""
    lens, group, rays, records, _ = fused_cooke(monkeypatch, mode)
    assert trace.fused_trace(group, rays, 0, True) is True

    before = sum(
        int(np.count_nonzero(np.isnan(be.to_numpy(s.x)))) for s in group.surfaces
    )
    assert before == 0
    group.surfaces[0].x[0] = float("nan")
    after = [np.isnan(be.to_numpy(s.x)) for s in group.surfaces]
    assert int(sum(int(a.sum()) for a in after)) == 1
    assert bool(after[0][0])
    assert not bool(after[1].any())


@pytest.mark.parametrize("mode", MODES)
def test_views_are_gpu_resident(mps_backend, monkeypatch, mode):
    """The snapshots are zero-copy views; arithmetic stays on the GPU (P2)."""
    lens, group, rays, records, _ = fused_cooke(monkeypatch, mode)
    assert trace.fused_trace(group, rays, 0, True) is True

    base = group.surfaces[0].x.components[0].untyped_storage().data_ptr()
    for surf in group.surfaces:
        assert surf.x.is_host_resident is False
        assert surf.x.components[0].untyped_storage().data_ptr() == base

    T.reset_stats()
    _ = group.surfaces[2].x * 2.0
    st = T.stats()
    assert sum(v for k, v in st.items() if k.startswith("gpu:")) == 1
    assert not any(k.startswith("host:") for k in st)
    assert not any(k.startswith("cpu_") for k in st)


@pytest.mark.parametrize("mode", MODES)
def test_no_wrapper_passed_to_kernel(mps_backend, monkeypatch, mode):
    """Every buffer the kernel binds is a plain, contiguous, mps tensor."""
    lens, group, rays, records, _ = fused_cooke(monkeypatch, mode)
    seen: list[Any] = []
    real = trace._launch_slabs

    def spy(lib, recs, launch, **kw):
        class Proxy:
            pass

        proxy = Proxy()
        # Both twins (the full body and the spherical-only instantiation the
        # driver selects when no surface is Newton-solved) capture their args.
        for name in (trace.ENTRY[kw["mode"]], trace.ENTRY_SPHERICAL[kw["mode"]]):
            entry = getattr(lib, name)

            def capture(*args, entry=entry, **kwargs):
                seen.extend(args)
                return entry(*args, **kwargs)

            setattr(proxy, name, capture)
        return real(proxy, recs, launch, **kw)

    monkeypatch.setattr(trace, "_launch_slabs", spy)
    assert trace.fused_trace(group, rays, 0, True) is True

    assert len(seen) == (16 if mode == "df64" else 10)
    for arg in seen:
        assert type(arg) is torch.Tensor, type(arg).__name__
        assert arg.device.type == "mps"
        assert arg.is_contiguous()


# ---------------------------------------------------------------------------
# 5. Refusals, late fallback, drift, DIAG
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_kill_switch_is_read_per_trace(mps_backend, monkeypatch, mode):
    """``OPTILAND_METAL_FUSED_TRACE=0`` refuses before anything is imported."""
    lens, group, rays, records, fake = fused_cooke(monkeypatch, mode)
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    before = T.stats()
    assert trace.fused_trace(group, rays, 0, True) is False
    assert delta(before) == {}
    assert fake.calls == []
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    assert trace.fused_trace(group, rays, 0, True) is True


@pytest.mark.parametrize(
    "reason",
    [FusedTraceSkip.HOST_RESIDENT, FusedTraceSkip.GEOMETRY_TYPE],
    ids=lambda r: r.value,
)
def test_refusal_counts_and_require_splits_structural_from_feature(
    mps_backend, monkeypatch, reason
):
    """Structural refusals never raise under ``require``; feature ones do."""
    mode = "df64"
    lens, group = cooke_group(mode)
    s = len(group.surfaces)
    structural = reason in (FusedTraceSkip.HOST_RESIDENT,)
    gate = FakeGate(ok=False, reason=reason, structural=structural, mode=mode, n=0)
    install_fake(monkeypatch, hand_records(mode, s), gate)
    rays = make_rays()

    before = T.stats()
    assert trace.fused_trace(group, rays, 0, True) is False
    counts = delta(before)
    assert counts.get(f"fused_trace_skip:{reason.value}") == 1
    # a structural refusal is not a candidate; a feature refusal is
    assert counts.get("fused_trace:candidates", 0) == (0 if structural else 1)
    assert not any(k.startswith("gpu:") for k in counts)

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "require")
    if structural:
        assert trace.fused_trace(group, rays, 0, True) is False
    else:
        with pytest.raises(trace.MetalFallbackError, match=reason.value):
            trace.fused_trace(group, rays, 0, True)


@pytest.mark.parametrize("mode", MODES)
def test_late_fallback_path(mps_backend, monkeypatch, mode):
    """A tolerance crossover returns False with nothing written; require raises."""
    lens, group, rays, records, _ = fused_cooke(monkeypatch, mode)
    monkeypatch.setattr(
        trace, "_late_fallback_mask", lambda status, has_newton: np.array([True])
    )
    before = T.stats()
    assert trace.fused_trace(group, rays, 0, True) is False
    counts = delta(before)
    assert counts.get("fused_trace:late_fallback") == 1
    assert counts.get("fused_trace:traces", 0) == 0
    for surf in group.surfaces:
        assert be.size(surf.x) == 0  # nothing written back
    assert rays.L0 is None

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "require")
    with pytest.raises(trace.MetalFallbackError, match="crossover"):
        trace.fused_trace(group, rays, 0, True)


@pytest.mark.parametrize("mode", MODES)
def test_drift_refuses(mps_backend, monkeypatch, mode):
    """A drifted MIRRORED source refuses every candidate, warning once."""
    lens, group, rays, records, _ = fused_cooke(monkeypatch, mode)
    monkeypatch.setattr(trace_mirror, "check_all", lambda: ["optiland.x.Y.sag"])
    trace.reset_driver_state()

    before = T.stats()
    with pytest.warns(trace.FusedTraceDriftWarning, match="optiland.x.Y.sag"):
        assert trace.fused_trace(group, rays, 0, True) is False
    assert delta(before).get("fused_trace_skip:mirror_drift") == 1

    # one warning per process, not one per trace
    with warnings_recorded() as rec:
        assert trace.fused_trace(group, rays, 0, True) is False
    assert [
        w for w in rec if issubclass(w.category, trace.FusedTraceDriftWarning)
    ] == []

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "require")
    with pytest.raises(trace.MetalFallbackError, match="mirror_drift"):
        trace.fused_trace(group, rays, 0, True)

    # the developer escape hatch proceeds (never set by a harness)
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE_DRIFT", "warn")
    assert trace.fused_trace(group, rays, 0, True) is True


def warnings_recorded():
    """``warnings.catch_warnings(record=True)`` with ``always``."""
    import warnings as _w

    class _Ctx:
        def __enter__(self):
            self._cm = _w.catch_warnings(record=True)
            out = self._cm.__enter__()
            _w.simplefilter("always")
            return out

        def __exit__(self, *exc):
            return self._cm.__exit__(*exc)

    return _Ctx()


@pytest.mark.parametrize("mode", MODES)
def test_memory_budget_refuses(mps_backend, monkeypatch, mode):
    """``memory_bytes`` above the budget refuses before anything is allocated."""
    lens, group, rays, records, fake = fused_cooke(monkeypatch, mode)
    fake.bytes_per_trace = 10**18
    before = T.stats()
    assert trace.fused_trace(group, rays, 0, True) is False
    counts = delta(before)
    assert counts.get("fused_trace_skip:memory") == 1
    assert not any(k.startswith("gpu:") for k in counts)


@pytest.mark.parametrize("mode", MODES)
def test_counters_are_exact(mps_backend, monkeypatch, mode):
    """The trace/design/surface-step counters equal their closed-form values."""
    lens, group, rays, records, _ = fused_cooke(monkeypatch, mode)
    # The mirror check runs once per process and its structural identities
    # build a probe Optic, so its host ops belong to start-up, not to a trace.
    trace._drift_qualnames()
    before = T.stats()
    assert trace.fused_trace(group, rays, 0, True) is True
    counts = delta(before)
    plan = trace._slab_plan(1, N_RAYS, records.weighted_steps, MAX, CHUNK)
    assert counts == {
        "fused_trace:candidates": 1,
        "fused_trace:traces": 1,
        "fused_trace:designs": 1,
        "fused_trace:surface_steps": N_RAYS * (records.S - 1),
        "fused_trace:chunks": len(plan),
        "gpu:fused_trace": len(plan),
    }
    # the census identity of plan 1.3 holds for this trace
    assert counts["fused_trace:candidates"] == counts["fused_trace:traces"]


@pytest.mark.parametrize("mode", MODES)
def test_diag_not_deepcopied(mps_backend, monkeypatch, mode):
    """The DIAG planes hang off a WeakKeyDictionary, not off the group.

    Deviation from the plan's wording (which predicts the allocated-memory
    delta of the deepcopy): the copy is taken with ``record=False`` so no
    snapshot views are duplicated, and the assertion is the exact one the fix
    is about -- the planes are neither copied nor kept alive.
    """
    monkeypatch.setenv("OPTILAND_METAL_TRACE_DIAG", "1")
    lens, group, rays, records, _ = fused_cooke(monkeypatch, mode, record=False)
    assert trace.fused_trace(group, rays, 0, False) is True

    planes = trace.diag_from(group)
    assert planes is not None
    status, iters = planes
    assert tuple(status.shape) == (1, records.S, N_RAYS)
    assert len(trace._DIAG_PLANES) == 1

    clone = copy.deepcopy(lens)
    assert trace.diag_from(clone.surfaces) is None
    assert len(trace._DIAG_PLANES) == 1  # nothing was copied

    del group, lens, planes, status, iters
    gc.collect()
    assert len(trace._DIAG_PLANES) == 0  # nothing was kept alive


# ---------------------------------------------------------------------------
# 5b. The driver on WP2's real record compiler (no test double)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_drives_the_real_record_compiler(mps_backend, mode):
    """Gate -> compile -> launch -> writeback with nothing monkeypatched.

    Values are the stub kernel's until WP1's body lands, so this asserts the
    integration only: the gate accepts a real bundle, the compiled records
    drive one launch, and the writeback shapes and counters are exact.
    """
    pytest.importorskip(
        "optiland.backend.torch_backend.metal.trace_record",
        reason="WP2's record compiler has not landed yet",
    )
    lens, group = cooke_group(mode)
    rays = make_rays()
    group.reset()
    trace._drift_qualnames()
    T.reset_stats()

    assert trace.fused_trace(group, rays, 0, True) is True

    st = T.stats()
    s = len(group.surfaces)
    assert {k: v for k, v in st.items() if k.startswith(("fused_trace", "gpu:"))} == {
        "fused_trace:readback": 1,
        "fused_trace:candidates": 1,
        "fused_trace:traces": 1,
        "fused_trace:designs": 1,
        "fused_trace:surface_steps": N_RAYS * (s - 1),
        "fused_trace:chunks": 1,
        "gpu:fused_trace": 1,
    }
    assert not any(k.startswith("cpu_") for k in st)
    assert tuple(be.to_numpy(group.x).shape) == (s, N_RAYS)
    for attr in ("x", "y", "z", "L", "M", "N", "i", "opd", "L0", "M0", "N0"):
        assert tuple(getattr(rays, attr).shape) == (N_RAYS,), attr


# ---------------------------------------------------------------------------
# 6. Live at I1: what only a per-op comparison can decide
# ---------------------------------------------------------------------------


def tilted_plane_optic(mode: str):
    """A three-surface system whose middle plane is tilted AND decentred.

    Surface 1 carries ``rx = 0.05`` and ``dy = 2.0``: the localize/globalize
    pair (``SR_CNRX``/``SR_SRX`` and ``SR_TY``) is exercised, and the recorded
    intersection points are only right if the kernel globalizes them back.

    Two fixture bugs lived here and are pinned by the asserts below:

    * no surface was marked as the aperture stop, so ``generate_rays`` ->
      ``paraxial.EPL()`` raised ``ValueError`` before any trace ran (the I1
      blocker; it was hidden while ``NEEDS_PHYSICS`` skipped the test);
    * the decentre was written ``y=2.0``, which ``add_surface`` does not
      forward to the coordinate system -- the surface was tilted but NOT
      decentred, so the ``SR_TY`` half of the claim was never tested.
    """
    from optiland.optic import Optic

    metal.set_mode(mode)
    optic = Optic()
    optic.add_surface(index=0, thickness=np.inf)
    optic.add_surface(
        index=1, thickness=5.0, radius=np.inf, dy=2.0, rx=0.05, is_stop=True
    )
    optic.add_surface(index=2)
    optic.set_aperture("EPD", 10.0)
    optic.set_field_type("angle")
    optic.add_field(y=0.0)
    optic.add_wavelength(0.55, is_primary=True)

    group = optic.surfaces
    assert group.stop_index == 1
    cs = group.surfaces[1].geometry.cs
    pose = {a: float(be.to_numpy(getattr(cs, a))) for a in ("x", "y", "z", "rx")}
    assert pose == {"x": 0.0, "y": 2.0, "z": 0.0, "rx": 0.05}
    return optic, group


@NEEDS_PHYSICS
@pytest.mark.parametrize("mode", MODES)
def test_global_frame_record(mps_backend, monkeypatch, mode):
    """Snapshots are in the GLOBAL frame on a tilted, decentred surface.

    Driver level, so it is live at I1 (WP4's hook is wave 2): the reference is
    the per-op ``group.trace`` on a component-level copy of the same launch
    bundle, the fused leg is ``fused_trace`` under ``require``, and agreement
    is plan 7.1 tier A -- raw components, ``equal_nan=True``, no tolerance.
    """
    optic, group = tilted_plane_optic(mode)
    rays = pupil_rays(optic)
    assert tuple(rays.x.shape) == (N_RAYS,)  # > HOST_THRESHOLD: not host-resident
    launch_z = np.asarray(be.to_numpy(rays.z))
    reference = copy_rays(rays)

    # --- reference: the per-op GPU path (no hook, so no env lookup happens) --
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    group.reset()
    group.trace(reference, record=True)
    ref_snap = [
        {a: raw(getattr(surf, a)) for a in SNAP_ATTRS} for surf in group.surfaces
    ]
    ref_final = {a: raw(getattr(reference, a)) for a in FINAL_ATTRS}

    # --- the fused path, under `require` ------------------------------------
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "require")
    group.reset()
    trace._drift_qualnames()
    T.reset_stats()
    assert trace.fused_trace(group, rays, 0, True) is True

    s = len(group.surfaces)
    st = T.stats()
    assert {k: v for k, v in st.items() if k.startswith(("fused_trace", "gpu:"))} == {
        "fused_trace:readback": 1,
        "fused_trace:candidates": 1,
        "fused_trace:traces": 1,
        "fused_trace:designs": 1,
        "fused_trace:surface_steps": N_RAYS * (s - 1),
        "fused_trace:chunks": 1,
        "gpu:fused_trace": 1,
    }

    for index, surf in enumerate(group.surfaces):
        for attr in SNAP_ATTRS:
            compare_raw(
                raw(getattr(surf, attr)),
                ref_snap[index][attr],
                f"[{mode}] surface {index} {attr}",
            )
    for attr in FINAL_ATTRS:
        compare_raw(raw(getattr(rays, attr)), ref_final[attr], f"[{mode}] final {attr}")

    # The comparison must not be vacuous.  ``equal_nan=True`` makes an all-NaN
    # record compare equal to anything, and an untilted plane would record a
    # constant global z, so both are ruled out with exact predicates: every
    # recorded word is finite, the bundle starts on ONE plane (range exactly
    # 0.0) and the tilt spreads the recorded global z over a range that is
    # not.
    snap = {
        a: np.asarray(be.to_numpy(getattr(group.surfaces[1], a)))
        for a in ("x", "y", "z")
    }
    for a, v in snap.items():
        assert np.all(np.isfinite(v)), f"[{mode}] surface 1 {a} is not finite"
    assert float(np.ptp(launch_z)) == 0.0
    assert float(np.ptp(snap["z"])) > 0.0


@NEEDS_PHYSICS
@NEEDS_HOOK
@pytest.mark.parametrize("mode", MODES)
def test_trailing_propagate_unchanged(mps_backend, monkeypatch, mode):
    """``optic.trace`` fused equals ``=0``, including the trailing propagate.

    The only test in this file that goes through WP4's hook, which is what the
    trailing propagate needs: ``RealRayTracer.trace`` propagates the returned
    bundle to the image surface AFTER ``SurfaceGroup.trace`` returns
    (``real_ray_tracer.py``), so this is the one comparison that covers the
    step the kernel does not run.  ``fused_trace:traces == 1`` is asserted on
    the ``=1`` leg: a silently-unhooked or silently-refused run fails here
    instead of passing by comparing the per-op path with itself.
    """
    from optiland.samples.objectives import CookeTriplet

    metal.set_mode(mode)
    lens = CookeTriplet()
    # 64 hexapolar RINGS = 1 + 3*64*65 = 12,481 rays, i.e. far above
    # HOST_THRESHOLD (256), so the bundle is a real fused candidate.
    n_rays = 1 + 3 * 64 * 65

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    T.reset_stats()
    rays_ref = lens.trace(
        Hx=0.0, Hy=1.0, wavelength=0.55, num_rays=64, distribution="hexapolar"
    )
    assert tuple(rays_ref.x.shape) == (n_rays,)
    assert T.stats().get("fused_trace:traces", 0) == 0
    ref = {a: raw(getattr(rays_ref, a)) for a in FINAL_ATTRS}

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    trace._drift_qualnames()
    T.reset_stats()
    rays_got = lens.trace(
        Hx=0.0, Hy=1.0, wavelength=0.55, num_rays=64, distribution="hexapolar"
    )
    assert T.stats().get("fused_trace:traces", 0) == 1
    for attr in FINAL_ATTRS:
        compare_raw(raw(getattr(rays_got, attr)), ref[attr], f"[{mode}] final {attr}")
