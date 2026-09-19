"""WP4: the ``SurfaceGroup.trace`` hook, the conftest switches and the census.

What this file defends (plan 3.4, 4/WP4, 1.3, 8.3):

1. **The hook costs the NumPy path nothing.**  ``_fused_metal_trace`` returns
   False on its first line for anything that is not a ``MetalFloat64`` bundle,
   *before* an environment lookup and before any import.  Two tests measure
   that rather than argue it: a fresh subprocess proves no
   ``optiland.backend.torch_backend.metal`` module is imported by a NumPy
   ``optic.trace`` and that the call costs under a microsecond with
   ``os.environ.get`` rigged to raise, and a 180k-ray scenario measures the
   hook against the pre-hook loop body.
2. **The gate fires where it must and nowhere else.**  Every shipped sample
   fuses in one trace with a *predicted* number of launches (``_slab_plan`` is
   the predictor, never a measured table), every ``FusedTraceSkip`` value is
   produced by its WP5 fixture with the counter at exactly 1 and no launch,
   and the per-op path keeps its own routing (``gpu:conic_candidates``).
3. **``require`` is structural-safe** (plan 1.3): chief rays, 6-ring bundles,
   ``PolarizedRays`` and grad-on traces pass through it untouched.
4. **The census is an independent predictor.**  The predicate lives in
   ``tests/conftest.py`` and is imported from there, so the census these tests
   check is literally the one the suite writes to its JSONL.

Run: ``pytest tests/metal/test_trace_gate.py -q -p no:cacheprovider -o addopts=``
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import contextlib  # noqa: E402
import functools  # noqa: E402
import importlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import textwrap  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

torch = pytest.importorskip("torch")
if not torch.backends.mps.is_available():  # pragma: no cover - hardware gate
    pytest.skip("Metal GPU required", allow_module_level=True)

import optiland.backend as be  # noqa: E402
from optiland.backend.torch_backend.metal import (  # noqa: E402
    library,
    trace,
    trace_mirror,
)
from optiland.backend.torch_backend.metal import tensor as T  # noqa: E402
from optiland.backend.torch_backend.metal import trace_record as R  # noqa: E402
from optiland.backend.torch_backend.metal.trace_adapters import (  # noqa: E402
    STRUCTURAL_REASONS,
    FusedTraceSkip,
)
from optiland.geometries.standard import StandardGeometry  # noqa: E402
from optiland.surfaces.surface_group import (  # noqa: E402
    SurfaceGroup,
    _fused_metal_trace,
)
from tests.conftest import _is_fuse_candidate  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = str(_ROOT / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
import trace_fixtures  # noqa: E402

MODES = ("df64", "sf64")

#: Above ``HOST_THRESHOLD`` (256) and above ``_MAX_VALUE_KEY_ARRAY_SIZE``, so a
#: bundle of this size is GPU-resident and takes the uniform-material branch.
N_RAYS = 4096

#: Hexapolar rings for the ``optic.trace`` census tests: 1 + 3*12*13 = 469
#: rays, comfortably past ``HOST_THRESHOLD`` so the trace is a candidate.
CENSUS_RINGS = 12
CENSUS_RAYS = 1 + 3 * CENSUS_RINGS * (CENSUS_RINGS + 1)

#: The pending perf gate's ``surfaces_trace_spherical`` scenario size.
PERF_RAYS = 180_000

#: The plan's budget for the hook on the NumPy path (plan 4/WP4).
PERF_BUDGET_SECONDS = 1e-3
CHEAP_CALL_SECONDS = 1e-6

CATALOG = dict(trace_fixtures.CATALOG)
KNOWN_INELIGIBLE = dict(trace_fixtures.KNOWN_INELIGIBLE)
CATALOG_NAMES = sorted(CATALOG)

FEATURE_REASONS = tuple(r for r in FusedTraceSkip if r not in STRUCTURAL_REASONS)
ALL_REASONS = sorted(FusedTraceSkip, key=lambda r: r.value)

#: Refusals whose fixture must not be driven through ``SurfaceGroup.trace``.
#:
#: ``group_type``: a ``SequencedSurfaceGroup`` is not a ``SurfaceGroup`` and
#: has its own ``trace``, so the hook is never installed on it (plan 1.2, "the
#: sequenced loop is not hooked"); the reason is only reachable through the
#: driver entry point, which is exactly what ``test_sequenced_group_never_fused``
#: pairs with.
#: ``too_many_rays``: the bundle declares 2**30 + 1 rays as stride-0 views, so
#: the *fallback* loop would materialise ~8 GiB per plane.  The refusal is the
#: whole contract there, so the driver is called directly and no loop runs.
HOOKLESS = frozenset({FusedTraceSkip.GROUP_TYPE, FusedTraceSkip.TOO_MANY_RAYS})

#: Refusals whose result cannot be compared against a per-op trace of the same
#: bundle (the per-op loop would be the 8 GiB trace described above).
NO_PEROP_COMPARE = frozenset({FusedTraceSkip.TOO_MANY_RAYS})

#: Measured on this checkout: ``FusedTraceSkip.POLARIZATION`` is returned
#: nowhere in ``metal/trace_record.py`` -- a ``BaseCoatingPolarized`` is
#: refused with ``coating`` -- so the value is unreachable and the fixture
#: cannot produce it.  ``trace_record.py`` is WP2's file, not WP4's; the
#: one-branch split is requested in ``NOTES/fused-trace-research/status.md``
#: (WP5 section 4) and repeated in WP4's section.  The mark is *strict*: the
#: day the split lands this test fails until the xfail is removed.
UNREACHABLE_REASONS = {
    FusedTraceSkip.POLARIZATION: (
        "FusedTraceSkip.POLARIZATION is returned nowhere in "
        "metal/trace_record.py (WP2): a BaseCoatingPolarized is refused with "
        "`coating`. See the WP5 request in NOTES/fused-trace-research/status.md"
    )
}


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _driver_state():
    """Zero the counters and the driver's one-shot state around every test."""
    trace.reset_driver_state()
    T.reset_stats()
    yield
    trace.reset_driver_state()
    T.reset_stats()


def _backend(mode: str):
    be.set_backend("torch")
    be.set_device("mps")
    be.set_precision("float64")
    be.grad_mode.disable()
    previous = be.metal_mode()
    be.set_metal_mode(mode)
    be.metal_reset_stats()
    try:
        yield mode
    finally:
        be.set_metal_mode(previous)
        be.grad_mode.disable()
        be.set_backend("numpy")


@pytest.fixture(params=MODES, ids=lambda m: f"mode={m}")
def mps_mode(request):
    """The torch-mps backend in each representation, grad off."""
    yield from _backend(request.param)


@pytest.fixture
def mps_df64():
    """The torch-mps backend in df64, grad off."""
    yield from _backend("df64")


@contextlib.contextmanager
def metal_backend(mode: str):
    """``_backend`` as a context manager, for tests parametrized over modes
    without a fixture (they also need ``monkeypatch`` ordering of their own)."""
    generator = _backend(mode)
    next(generator)
    try:
        yield mode
    finally:
        generator.close()


@pytest.fixture
def numpy_backend():
    """The NumPy backend, restored afterwards."""
    previous = be.get_backend()
    be.set_backend("numpy")
    try:
        yield
    finally:
        be.set_backend(previous)


@contextlib.contextmanager
def census(capture: bool = False):
    """Count ``SurfaceGroup.trace`` calls with the conftest's own predicate.

    Yields a ``SimpleNamespace`` with ``candidates`` (the census), ``calls``
    (every call, candidate or not) and, with ``capture``, ``launches``: the raw
    components of each incoming bundle, snapshotted *before* the trace mutates
    it in place.
    """
    counts = SimpleNamespace(candidates=0, calls=0, launches=[])
    original = SurfaceGroup.trace

    def _census_trace(self, rays, skip=0, record=True):
        counts.calls += 1
        if _is_fuse_candidate(self, rays, skip):
            counts.candidates += 1
        if capture:
            counts.launches.append(_raw(rays))
        return original(self, rays, skip=skip, record=record)

    SurfaceGroup.trace = _census_trace
    try:
        yield counts
    finally:
        SurfaceGroup.trace = original


def _fused(stats: dict[str, int]) -> dict[str, int]:
    """The ``fused_trace*`` slice of a counter snapshot."""
    return {k: v for k, v in stats.items() if k.startswith("fused_trace")}


def _skips(stats: dict[str, int]) -> dict[str, int]:
    return {
        k[len("fused_trace_skip:") :]: v
        for k, v in stats.items()
        if k.startswith("fused_trace_skip:")
    }


def _feature_skips(stats: dict[str, int]) -> int:
    values = {r.value for r in FEATURE_REASONS}
    return sum(v for k, v in _skips(stats).items() if k in values)


def _raw(rays) -> dict[str, tuple[np.ndarray, ...]]:
    """Every ray plane as raw components (df64 hi/lo words, sf64 bit patterns).

    Tier A of plan 7.1: equality is on the components, never on a decoded
    float, so a 1-ulp difference cannot hide behind a decode.
    """
    out: dict[str, tuple[np.ndarray, ...]] = {}
    for name in ("x", "y", "z", "L", "M", "N", "i", "opd", "w"):
        value = getattr(rays, name)
        parts = getattr(value, "components", None) or (value,)
        out[name] = tuple(np.asarray(p.detach().cpu().numpy()) for p in parts)
    return out


def _raw_equal(left, right) -> list[str]:
    """Plane names whose raw components differ (NaN patterns count as equal)."""
    bad = []
    for name, parts in left.items():
        others = right[name]
        if len(parts) != len(others):
            bad.append(name)
            continue
        for a, b in zip(parts, others, strict=True):
            equal = (
                np.array_equal(a, b, equal_nan=True)
                if np.issubdtype(a.dtype, np.floating)
                else np.array_equal(a, b)
            )
            if not equal:
                bad.append(name)
                break
    return bad


def _numpy_bundle(n: int, wavelength: float = 0.55):
    """A collimated NumPy bundle for the Cooke triplet's 10 mm pupil."""
    return trace_fixtures.collimated_bundle(
        n, radius=5.0, z=-10.0, wavelength=wavelength
    )


def _subprocess_json(script: str, extra_env: dict[str, str] | None = None) -> dict:
    """Run ``script`` in a fresh interpreter and parse its last stdout line."""
    env = dict(os.environ)
    env.setdefault("PYTORCH_MPS_FAST_MATH", "0")
    env["MPLBACKEND"] = "Agg"
    env["QT_QPA_PLATFORM"] = "offscreen"
    env.pop("OPTILAND_METAL_FUSED_TRACE", None)
    env.update(extra_env or {})
    done = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, (
        done.returncode,
        done.stdout[-4000:],
        done.stderr[-4000:],
    )
    lines = [line for line in done.stdout.splitlines() if line.strip()]
    assert lines, done.stderr[-4000:]
    return json.loads(lines[-1])


# ---------------------------------------------------------------------------
# 1. The NumPy path pays nothing
# ---------------------------------------------------------------------------
_NUMPY_SUBPROCESS = r"""
import json, sys, timeit, types
import numpy as np

import optiland.backend as be

def metal_modules():
    return sorted(m for m in sys.modules
                  if m.startswith("optiland.backend.torch_backend.metal"))

# ``optiland/backend/__init__.py`` imports the ``metal`` package itself at
# package import (nothing to do with the hook); that is the baseline every
# later snapshot must still equal.
baseline = metal_modules()

be.set_backend("numpy")
from optiland.rays import RealRays
from optiland.samples.objectives import CookeTriplet
from optiland.surfaces import surface_group as sg

optic = CookeTriplet()
optic.trace(Hx=0.0, Hy=0.0, wavelength=0.55, num_rays=8, distribution="hexapolar")

after_trace = metal_modules()

n = 1024
rays = RealRays(np.zeros(n), np.zeros(n), np.full(n, -10.0), np.zeros(n),
                np.zeros(n), np.ones(n), np.ones(n), np.full(n, 0.55))

class _Boom:
    def get(self, *args, **kwargs):
        raise AssertionError("the NumPy path looked up the environment")

# Replace the module-level ``os`` the hook closes over: any environment lookup
# now raises, so "returns False before the lookup" is measured, not argued.
sg.os = types.SimpleNamespace(environ=_Boom())
group = optic.surfaces
returned = sg._fused_metal_trace(group, rays, 0, True)

number = 100000
best = min(timeit.repeat(lambda: sg._fused_metal_trace(group, rays, 0, True),
                         number=number, repeat=7)) / number
print(json.dumps({
    "baseline": baseline,
    "after_trace": after_trace,
    "after_hook": metal_modules(),
    "returned": returned,
    "seconds": best,
}))
"""


def test_hook_is_installed(numpy_backend):
    """``SurfaceGroup.trace`` calls the hook, and the hook is the plan's.

    Without this the whole file could pass by comparing the per-op path with
    itself: an unhooked ``trace`` refuses nothing and counts nothing.
    """
    assert "_fused_metal_trace" in SurfaceGroup.trace.__code__.co_names
    rays = _numpy_bundle(1024)  # NumPy planes: not a ``MetalFloat64`` bundle
    assert _fused_metal_trace(object(), rays, 0, True) is False


def test_numpy_path_never_imports_metal_and_is_cheap():
    """A NumPy trace imports no metal module and the hook costs < 1 us."""
    out = _subprocess_json(_NUMPY_SUBPROCESS)
    # The only metal module in a NumPy process is the package ``optiland.backend``
    # itself imports; neither the trace nor the hook adds one.
    assert out["baseline"] == ["optiland.backend.torch_backend.metal"], out["baseline"]
    assert out["after_trace"] == out["baseline"], out["after_trace"]
    assert out["after_hook"] == out["baseline"], out["after_hook"]
    assert out["returned"] is False
    assert out["seconds"] < CHEAP_CALL_SECONDS, out["seconds"]


def test_numpy_perf_gate_scenario_cost(numpy_backend):
    """The hook costs the 180k-ray NumPy scenario under a millisecond.

    The statistic is the **median of 5 paired** deltas, not the difference of
    two medians: the two arms are measured adjacent in time, so the machine
    drift that dominates a 40 ms NumPy trace (measured spread 2-5 ms over five
    repeats) cancels inside each pair instead of leaking into the estimate of a
    hook that costs ~0.1 us.  The threshold is the plan's, untouched.
    """
    optic = trace_fixtures.cooke()
    group = optic.surfaces
    hooked = SurfaceGroup.trace

    def pre_hook_trace(self, rays, skip=0, record=True):
        """``SurfaceGroup.trace``'s body as it stood before the hook."""
        self.reset()
        for surface in self.surfaces[skip:]:
            surface.trace(rays, record=record)
        return rays

    # Warm both paths (imports, allocator) outside the measurement.
    for body in (hooked, pre_hook_trace):
        for _ in range(3):
            body(group, _numpy_bundle(PERF_RAYS))

    deltas = []
    samples: dict[str, list[float]] = {"hook": [], "pre": []}
    for _ in range(5):
        paired: dict[str, float] = {}
        for label, body in (("hook", hooked), ("pre", pre_hook_trace)):
            rays = _numpy_bundle(PERF_RAYS)
            start = time.perf_counter()
            body(group, rays)
            paired[label] = time.perf_counter() - start
            samples[label].append(paired[label])
        deltas.append(paired["hook"] - paired["pre"])

    delta = statistics.median(deltas)
    assert delta < PERF_BUDGET_SECONDS, (
        f"hook {statistics.median(samples['hook']) * 1e3:.3f} ms vs pre-hook "
        f"{statistics.median(samples['pre']) * 1e3:.3f} ms over {PERF_RAYS} "
        f"rays: median paired delta {delta * 1e3:.3f} ms"
    )


_NON_METAL_SUBPROCESS = r"""
import json, os, sys
import numpy as np
os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import optiland.backend as be
be.set_backend("torch")
be.set_device({device!r})
be.set_precision({precision!r})
be.grad_mode.disable()

from optiland.rays import RealRays
from optiland.samples.objectives import CookeTriplet

def bundle(n=2048):
    k = np.arange(n, dtype=np.float64)
    ang = k * (np.pi * (3.0 - np.sqrt(5.0)))
    rad = 5.0 * np.sqrt((k + 0.5) / n)
    return RealRays(be.array(rad * np.cos(ang)), be.array(rad * np.sin(ang)),
                    be.array(np.full(n, -10.0)), be.array(np.zeros(n)),
                    be.array(np.zeros(n)), be.array(np.ones(n)),
                    be.array(np.ones(n)), be.array(np.full(n, 0.55)))

optic = CookeTriplet()
planes = {{}}
for switch in ("1", "0"):
    os.environ["OPTILAND_METAL_FUSED_TRACE"] = switch
    rays = bundle()
    optic.surfaces.trace(rays)
    planes[switch] = [np.asarray(be.to_numpy(getattr(rays, a)), dtype=np.float64)
                      for a in ("x", "y", "z", "L", "M", "N", "i", "opd")]

equal = all(np.array_equal(a, b, equal_nan=True)
            for a, b in zip(planes["1"], planes["0"]))
modules = sorted(m for m in sys.modules
                 if m.startswith("optiland.backend.torch_backend.metal"))
stats = {{}}
if "optiland.backend.torch_backend.metal" in sys.modules:
    from optiland.backend.torch_backend import metal
    stats = {{k: v for k, v in metal.stats().items() if k.startswith("fused_trace")}}
print(json.dumps({{"equal": bool(equal), "modules": modules, "stats": stats}}))
"""


@pytest.mark.parametrize(
    ("label", "device", "precision"),
    [("torch-cpu-f64", "cpu", "float64"), ("mps-f32", "mps", "float32")],
    ids=lambda v: v if isinstance(v, str) else str(v),
)
def test_non_metal_paths_untouched(label, device, precision):
    """torch-CPU float64 and mps float32 never reach the fused driver."""
    out = _subprocess_json(
        _NON_METAL_SUBPROCESS.format(device=device, precision=precision)
    )
    assert out["equal"] is True, label
    assert "optiland.backend.torch_backend.metal.trace" not in out["modules"], (
        label,
        out["modules"],
    )
    assert out["stats"] == {}, (label, out["stats"])


# ---------------------------------------------------------------------------
# 2. The gate fires where it must
# ---------------------------------------------------------------------------
def _predicted_launches(group, mode: str, wavelength: float, n: int) -> int:
    """``len(_slab_plan(...))`` for one design of ``n`` rays through ``group``."""
    records = R.compile_records(
        group, R.canonical_w0(wavelength, mode), mode, record=True
    )
    plan = trace._slab_plan(
        1, n, records.weighted_steps, trace._max_steps(), library.DEFAULT_CHUNK
    )
    return len(plan)


@pytest.mark.parametrize("name", CATALOG_NAMES)
def test_gate_fires_on_every_claimed_sample(mps_mode, name):
    """Each shipped sample fuses in one trace with a predicted launch count."""
    mode = mps_mode
    optic = CATALOG[name]()
    group = optic.surfaces
    wavelength = float(optic.primary_wavelength)
    rays = trace_fixtures.pupil_bundle(optic, N_RAYS, wavelength=wavelength)

    T.reset_stats()
    group.trace(rays)
    stats = T.stats()

    expected_reason = KNOWN_INELIGIBLE.get(name)
    if expected_reason is not None:
        assert stats.get(f"fused_trace_skip:{expected_reason}") == 1, _fused(stats)
        assert "gpu:fused_trace" not in stats, stats.get("gpu:fused_trace")
        return

    assert stats.get("fused_trace:candidates") == 1, _fused(stats)
    assert stats.get("fused_trace:traces") == 1, _fused(stats)
    assert _skips(stats) == {}, _skips(stats)
    assert "gpu:conic_candidates" not in stats, stats.get("gpu:conic_candidates")

    expected = _predicted_launches(group, mode, wavelength, N_RAYS)
    assert stats.get("gpu:fused_trace") == expected, (
        stats.get("gpu:fused_trace"),
        expected,
    )
    assert stats.get("fused_trace:surface_steps") == N_RAYS * (len(group.surfaces) - 1)


@pytest.mark.parametrize("mode", MODES)
def test_reference_routing(mode, monkeypatch):
    """With the hook off, ``StandardGeometry.distance`` still reaches the
    fused conic kernel once per finite-radius spherical surface.

    This pins the R1 reference's routing, so an upstream re-route
    (``upstream/i329-conic-intersection``) fails here instead of silently
    changing what the conformance tests compare against.
    """
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    with metal_backend(mode):
        optic = trace_fixtures.cooke()
        group = optic.surfaces
        expected = sum(
            1
            for surface in group.surfaces
            if type(surface.geometry) is StandardGeometry
            and np.isfinite(float(be.to_numpy(surface.geometry.radius)))
        )
        assert expected == 6, expected  # the Cooke triplet's six glass faces

        rays = trace_fixtures.pupil_bundle(optic, N_RAYS)
        backend = be._backends["torch"]
        original = backend.conic_intersection
        calls = []

        def _counting(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(backend, "conic_intersection", _counting)
        T.reset_stats()
        group.trace(rays)
        stats = T.stats()

        assert len(calls) == expected, len(calls)
        assert stats.get("gpu:conic_candidates") == expected, (
            stats.get("gpu:conic_candidates"),
            expected,
        )
        assert _fused(stats) == {}, _fused(stats)


# ---------------------------------------------------------------------------
# 3. Refusals
# ---------------------------------------------------------------------------
def _apply_case(case, monkeypatch) -> None:
    for name, value in case.env.items():
        monkeypatch.setenv(name, value)
    for patch in case.patches:
        monkeypatch.setattr(
            importlib.import_module(patch.module), patch.attr, patch.value, raising=True
        )
    if case.grad:
        be.grad_mode.enable()
        monkeypatch.setattr(be.grad_mode, "requires_grad", True, raising=True)


def _case_group(case):
    return case.group if case.group is not None else case.optic.surfaces


def _case_rays(case):
    factory = case.rays if case.rays is not None else trace_fixtures.pupil_bundle
    return factory(case.optic, case.num_rays)


def _capture(call):
    """``(value, None)`` or ``(None, exception)`` -- outcomes are comparable."""
    try:
        return call(), None
    except Exception as error:  # noqa: BLE001 - the outcome IS the assertion
        return None, error


def _fresh_case(reason, monkeypatch, *, switch: str):
    """A newly built refusal fixture with its patches and ``switch`` applied."""
    case = trace_fixtures.REFUSAL_FIXTURES[reason]()
    _apply_case(case, monkeypatch)
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", switch)
    return case


def _run_case(case, *, hookless: bool):
    """Trace ``case`` through the hook (or the driver) and return the bundle."""
    group = _case_group(case)
    rays = _case_rays(case)
    if hookless:
        assert trace.fused_trace(group, rays, case.skip, True) is False
    else:
        group.trace(rays, skip=case.skip)
    return rays


@pytest.mark.parametrize("reason", ALL_REASONS, ids=lambda r: r.value)
def test_refusal_reason(mps_df64, monkeypatch, reason):
    """Every ``FusedTraceSkip`` value: counted once, no launch, per-op result.

    The comparison is *cross-path on the same backend*: the hook's fallback and
    an ``OPTILAND_METAL_FUSED_TRACE=0`` run of the same fixture must end the
    same way -- both returning raw-equal bundles, or both raising the same
    exception.  ``RefusalCase.per_op_raises`` is not used as the predicate,
    because WP5 measured it on the NumPy reference path and three fixtures
    behave differently under torch-mps (recorded in status.md): ``rays_shape``
    raises ``RuntimeError`` rather than ``ValueError``, and ``aperture_params``
    and ``bsdf`` raise where NumPy does not.

    Under ``require`` a structural reason must stay silent and a feature reason
    must raise ``MetalFallbackError`` (plan 1.3).
    """
    if reason in UNREACHABLE_REASONS:
        pytest.xfail(UNREACHABLE_REASONS[reason])

    structural = reason in STRUCTURAL_REASONS
    hookless = reason in HOOKLESS
    case = trace_fixtures.REFUSAL_FIXTURES[reason]()
    _apply_case(case, monkeypatch)

    T.reset_stats()
    fused_rays, fused_error = _capture(lambda: _run_case(case, hookless=hookless))
    stats = T.stats()

    assert not isinstance(fused_error, trace.MetalFallbackError), fused_error
    assert stats.get(f"fused_trace_skip:{reason.value}") == 1, (
        reason.value,
        _fused(stats),
    )
    assert "gpu:fused_trace" not in stats, stats.get("gpu:fused_trace")
    assert stats.get("fused_trace:traces", 0) == 0, _fused(stats)
    if structural:
        assert stats.get("fused_trace:candidates", 0) == 0, _fused(stats)
    else:
        assert stats.get("fused_trace:candidates") == 1, _fused(stats)

    # The fallback ends exactly as the per-op path does.
    if reason not in NO_PEROP_COMPARE and not hookless:
        reference_case = _fresh_case(reason, monkeypatch, switch="0")
        reference_rays, reference_error = _capture(
            lambda: _run_case(reference_case, hookless=False)
        )
        assert type(fused_error) is type(reference_error), (
            fused_error,
            reference_error,
        )
        if fused_error is None and case.deterministic:
            assert _raw_equal(_raw(fused_rays), _raw(reference_rays)) == []
        monkeypatch.delenv("OPTILAND_METAL_FUSED_TRACE", raising=False)

    # ``require``: structural refusals stay silent, feature refusals raise.
    trace.reset_driver_state()
    required_case = _fresh_case(reason, monkeypatch, switch="require")
    _, required_error = _capture(lambda: _run_case(required_case, hookless=hookless))
    if structural:
        assert not isinstance(required_error, trace.MetalFallbackError), required_error
    else:
        assert isinstance(required_error, trace.MetalFallbackError), required_error
        assert reason.value in str(required_error), str(required_error)


def test_require_is_structural_safe(mps_df64, monkeypatch):
    """Chief rays, 6-ring bundles, PolarizedRays and grad-on traces survive
    ``OPTILAND_METAL_FUSED_TRACE=require`` (plan 1.3, [fix: L4.1])."""
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "require")
    optic = trace_fixtures.cooke()

    # 1. a 6-ring hexapolar bundle: 127 rays, host-resident.
    six_ring = optic.trace(Hx=0.0, Hy=0.0, wavelength=0.55, num_rays=6)
    assert int(six_ring.x.numel()) == 127

    # 2. a chief ray through ``trace_generic``: N = 1.
    optic.ray_tracer.trace_generic(Hx=0.0, Hy=0.0, Px=0.0, Py=0.0, wavelength=0.55)

    # 3. a ``PolarizedRays`` bundle above the threshold.
    polarized = trace_fixtures.REFUSAL_FIXTURES[FusedTraceSkip.RAYS_TYPE]()
    _case_group(polarized).trace(_case_rays(polarized))

    # 4. autograd on.
    grad_case = trace_fixtures.REFUSAL_FIXTURES[FusedTraceSkip.REQUIRES_GRAD]()
    _apply_case(grad_case, monkeypatch)
    _case_group(grad_case).trace(_case_rays(grad_case))
    be.grad_mode.disable()

    stats = T.stats()
    refused = set(_skips(stats))
    assert refused, _fused(stats)
    assert refused <= {r.value for r in STRUCTURAL_REASONS}, refused
    assert _feature_skips(stats) == 0, _skips(stats)


# ---------------------------------------------------------------------------
# 4. Census identities
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", CATALOG_NAMES)
def test_census_identities(mps_df64, name):
    """The conftest census equals ``fused_trace:candidates`` (plan 1.3, 8.3)."""
    optic = CATALOG[name]()
    wavelength = float(optic.primary_wavelength)
    T.reset_stats()
    with census() as counts:
        optic.trace(
            Hx=0.0,
            Hy=0.0,
            wavelength=wavelength,
            num_rays=CENSUS_RINGS,
            distribution="hexapolar",
        )
    stats = T.stats()

    assert counts.candidates >= 1, (counts, _fused(stats))
    # Both identities below HOLD while the gate is fully off for a counted
    # feature reason, so neither can see the census wrapper switching the
    # kernel off (I2 section 3).  Only this assertion can.
    assert stats.get("fused_trace_skip:mirror_drift", 0) == 0, _fused(stats)
    assert counts.candidates == stats.get("fused_trace:candidates", 0), (
        counts.candidates,
        _fused(stats),
    )
    assert stats.get("fused_trace:candidates", 0) == (
        stats.get("fused_trace:traces", 0)
        + _feature_skips(stats)
        + stats.get("fused_trace:late_fallback", 0)
    ), _fused(stats)


def test_sequenced_group_never_fused(mps_df64):
    """``SequencedOptic.trace`` is not hooked: census 0, no counters."""
    from optiland.sequences import SequencedOptic

    optic = trace_fixtures.cooke()
    steps = list(range(len(optic.surfaces.surfaces)))
    sequence = SequencedOptic(optic, "all", steps)
    rays = trace_fixtures.pupil_bundle(optic, N_RAYS)

    T.reset_stats()
    with census() as counts:
        sequence.surfaces.trace(rays)
    stats = T.stats()

    assert counts.calls == 0, counts
    assert counts.candidates == 0, counts
    assert _fused(stats) == {}, _fused(stats)


# ---------------------------------------------------------------------------
# 5. State: re-traces, interleaving, ray aiming
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_retrace_after_geometry_replacement(mode, monkeypatch):
    """Replacing a geometry recompiles the records; a refused replacement
    falls back instead of reusing the previous trace's tables."""
    with metal_backend(mode):
        optic = trace_fixtures.cooke()
        group = optic.surfaces
        group.trace(trace_fixtures.pupil_bundle(optic, N_RAYS))
        assert T.stats().get("fused_trace:traces") == 1

        surface = group.surfaces[2]
        surface.geometry = StandardGeometry(
            surface.geometry.cs, be.array(-31.5), be.array(0.0)
        )

        T.reset_stats()
        fused = trace_fixtures.pupil_bundle(optic, N_RAYS)
        group.trace(fused)
        assert T.stats().get("fused_trace:traces") == 1, _fused(T.stats())

        monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
        reference = trace_fixtures.pupil_bundle(optic, N_RAYS)
        group.trace(reference)
        monkeypatch.delenv("OPTILAND_METAL_FUSED_TRACE")
        assert _raw_equal(_raw(fused), _raw(reference)) == []

        # A replacement the kernel cannot mirror must be refused, not reused.
        from optiland.geometries.toroidal import ToroidalGeometry

        surface.geometry = ToroidalGeometry(
            surface.geometry.cs,
            radius_x=-31.5,
            radius_y=-40.0,
            conic=0.0,
        )
        T.reset_stats()
        group.trace(trace_fixtures.pupil_bundle(optic, N_RAYS))
        stats = T.stats()
        assert stats.get("fused_trace_skip:geometry_type") == 1, _fused(stats)
        assert stats.get("fused_trace:traces", 0) == 0, _fused(stats)


@pytest.mark.parametrize("mode", MODES)
def test_paraxial_interleave_keeps_working(mode):
    """Paraxial traces around a fused trace neither fuse nor change value."""
    with metal_backend(mode):
        optic = trace_fixtures.cooke()
        before = _plain(optic.paraxial.f2())

        T.reset_stats()
        with census() as counts:
            optic.surfaces.trace(trace_fixtures.pupil_bundle(optic, N_RAYS))
            after_trace = _plain(optic.paraxial.f2())
        stats = T.stats()

        assert counts.candidates == 1, counts
        assert stats.get("fused_trace:candidates") == 1, _fused(stats)
        assert stats.get("fused_trace:traces") == 1, _fused(stats)
        assert np.array_equal(before, after_trace, equal_nan=True), (
            before,
            after_trace,
        )


def _plain(value) -> np.ndarray:
    return np.asarray(be.to_numpy(value), dtype=np.float64)


@pytest.mark.parametrize("mode", MODES)
def test_ray_aiming_midflight_does_not_disturb(mode, monkeypatch):
    """Ray aiming's per-surface traces never enter the fused path.

    ``rays/ray_aiming/initialization.py:174`` and ``iterative.py:753`` loop over
    ``self.optic.surfaces[i].trace(rays)`` themselves, so they never reach
    ``SurfaceGroup.trace`` at all: the group-level census counts 0 for them and
    then counts the full trace that follows.  The aiming solve must also be
    *unaffected* by the switch, which is checked on the launch bundle
    ``SurfaceGroup.trace`` receives -- raw components, so a 1-ulp shift in an
    aim point fails here.  (The traced *output* is WP6's tier-A comparison, not
    this test's: at this bundle size df64 legitimately differs, see the
    ``_MAX_VALUE_KEY_ARRAY_SIZE`` finding in status.md.)
    """
    with metal_backend(mode):
        optic = trace_fixtures.cooke()
        optic.ray_tracer.set_aiming("iterative", max_iter=5, tol=1e-6)

        from optiland.surfaces.standard_surface import Surface

        original_surface_trace = Surface.trace
        surface_calls = []

        def _counting_surface_trace(self, rays, record=True):
            surface_calls.append(1)
            return original_surface_trace(self, rays, record=record)

        monkeypatch.setattr(Surface, "trace", _counting_surface_trace)

        def _aimed_trace():
            return optic.trace(
                Hx=0.0,
                Hy=0.7,
                wavelength=0.55,
                num_rays=CENSUS_RINGS,
                distribution="hexapolar",
            )

        T.reset_stats()
        with census(capture=True) as fused_counts:
            _aimed_trace()
        stats = T.stats()

        assert surface_calls, "ray aiming never ran"
        assert fused_counts.calls == 1, fused_counts
        # Every group-level trace in the run is a candidate: the mid-flight
        # aiming traces are surface-level and never reach the census.
        assert fused_counts.candidates == fused_counts.calls, fused_counts
        assert fused_counts.candidates == stats.get("fused_trace:candidates", 0), (
            fused_counts,
            _fused(stats),
        )
        assert stats.get("fused_trace:traces") == 1, _fused(stats)

        monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
        with census(capture=True) as perop_counts:
            _aimed_trace()
        assert perop_counts.calls == fused_counts.calls
        for aimed, reference in zip(
            fused_counts.launches, perop_counts.launches, strict=True
        ):
            assert _raw_equal(aimed, reference) == []


# ---------------------------------------------------------------------------
# 6. The conftest switches themselves
# ---------------------------------------------------------------------------
def _set_test_backend_body():
    """The fork-local ``set_test_backend`` fixture function, unwrapped."""
    from tests import conftest as tests_conftest

    fixture = tests_conftest.set_test_backend
    unwrap = getattr(fixture, "_get_wrapped_function", None)  # pytest >= 8.4
    if unwrap is not None:
        return unwrap()
    return fixture.__pytest_wrapped__.obj  # pragma: no cover - older pytest


@pytest.mark.parametrize(
    ("value", "expected"), [("1", True), ("0", False), (None, True)]
)
def test_conftest_grad_switch(monkeypatch, value, expected):
    """``OPTILAND_TEST_MPS_GRAD`` drives ``be.grad_mode`` on the torch-mps
    parametrization; unset keeps today's grad-on behaviour."""
    if value is None:
        monkeypatch.delenv("OPTILAND_TEST_MPS_GRAD", raising=False)
    else:
        monkeypatch.setenv("OPTILAND_TEST_MPS_GRAD", value)

    generator = _set_test_backend_body()(SimpleNamespace(param="torch-mps"))
    try:
        next(generator)
        assert be.grad_mode.requires_grad is expected
    finally:
        next(generator, None)
        be.set_backend("torch")
        be.set_device("mps")
        be.grad_mode.disable()
        be.set_backend("numpy")


def test_conftest_census_predicate(mps_df64):
    """``_is_fuse_candidate`` accepts a GPU bundle and rejects each structural
    non-candidate, without importing the gate."""
    optic = trace_fixtures.cooke()
    group = optic.surfaces
    rays = trace_fixtures.pupil_bundle(optic, N_RAYS)

    assert _is_fuse_candidate(group, rays, 0) is True
    assert _is_fuse_candidate(group, rays, 1) is False
    assert _is_fuse_candidate(object(), rays, 0) is False
    assert (
        _is_fuse_candidate(group, trace_fixtures.pupil_bundle(optic, 256), 0) is False
    )

    polarized = trace_fixtures.REFUSAL_FIXTURES[FusedTraceSkip.RAYS_TYPE]()
    assert _is_fuse_candidate(group, _case_rays(polarized), 0) is False

    be.grad_mode.enable()
    try:
        assert _is_fuse_candidate(group, rays, 0) is False
    finally:
        be.grad_mode.disable()


# ---------------------------------------------------------------------------
# 7. Drift: instrumenting a mirrored function is not drift (I2 fix lane)
# ---------------------------------------------------------------------------
# The census this plan mandates (conftest's ``_fused_trace_census``, the
# ``census()`` helper above, plan 8.1's oracle census) replaces
# ``SurfaceGroup.trace`` with a wrapper.  ``trace_mirror`` resolves a MIRRORED
# row through the live class attribute, so before the I2 fix every census
# reported ``mirror_drift`` and the driver refused *every* candidate for the
# rest of the process -- measured: a 4096-ray Cooke df64 trace went from
# ``{'fused_trace:traces': 1, 'fused_trace:surface_steps': 28672}`` to
# ``{'fused_trace_skip:mirror_drift': 1}`` on one bare wrapper.  The fix
# follows ``__wrapped__``, closure cells and same-qualname module globals from
# the live attribute to the function being wrapped; these tests pin both
# halves of it: a wrapper is transparent, a *replacement* is still drift.

#: Holds the original for the module-global wrapper form of
#: :func:`_group_trace_wrapper` (set by :func:`_wrap_group_trace`).
_WRAPPED_GROUP_TRACE = None


def _module_global_wrapper(self, rays, skip=0, record=True):
    """A wrapper that keeps the original in a module global, not a closure."""
    return _WRAPPED_GROUP_TRACE(self, rays, skip=skip, record=record)


def _wrap_group_trace(form: str, original):
    """One ``SurfaceGroup.trace`` wrapper per wrapping style."""
    global _WRAPPED_GROUP_TRACE

    def closure_wrapper(self, rays, skip=0, record=True):
        return original(self, rays, skip=skip, record=record)

    if form == "closure":
        return closure_wrapper
    if form == "nested":
        inner = closure_wrapper

        def outer(self, rays, skip=0, record=True):
            return inner(self, rays, skip=skip, record=record)

        return outer
    if form == "functools_wraps":
        return functools.wraps(original)(closure_wrapper)
    if form == "module_global":
        _WRAPPED_GROUP_TRACE = original
        return _module_global_wrapper
    if form == "no_source":
        # Compiled from a string: ``inspect.getsource`` raises OSError, which
        # is how the failure first showed up in a ``python -c`` probe.
        namespace = {"original": original}
        exec(  # noqa: S102 - a wrapper built the way a harness builds one
            compile(
                "def w(self, rays, skip=0, record=True):\n"
                "    return original(self, rays, skip=skip, record=record)\n",
                "<no-source>",
                "exec",
            ),
            namespace,
        )
        return namespace["w"]
    raise AssertionError(f"unknown wrapper form {form!r}")


WRAPPER_FORMS = ("closure", "nested", "functools_wraps", "module_global", "no_source")


@pytest.mark.parametrize("form", WRAPPER_FORMS)
def test_mirror_check_sees_through_a_wrapper(form):
    """Every wrapping style leaves ``check_all()`` empty."""
    original = SurfaceGroup.trace
    assert trace_mirror.check_all() == [], "the table is already drifted"
    SurfaceGroup.trace = _wrap_group_trace(form, original)
    try:
        assert trace_mirror.check_all() == [], form
    finally:
        SurfaceGroup.trace = original


def test_mirror_check_still_catches_a_rebuilt_body():
    """A replacement that does not hold the original is still drift.

    This is round 0's injection mechanism (``test_trace_kernel.py``'s
    ``_rebuilt_with``): the function is recompiled from edited source into a
    copy of its module's globals, so nothing reachable from it is the original
    and the row must still be reported.
    """
    from optiland.rays.real_rays import RealRays

    original = RealRays.refract
    source = textwrap.dedent(inspect.getsource(original))
    marker = "- u * nx * dot"
    assert source.count(marker) == 1, "the injection target moved upstream"
    namespace = dict(original.__globals__)
    exec(  # noqa: S102 - round 0's own injection, in miniature
        compile(
            "from __future__ import annotations\n"
            + source.replace(marker, "+ u * nx * dot"),
            "<injected>",
            "exec",
        ),
        namespace,
    )
    RealRays.refract = namespace["refract"]
    try:
        problems = trace_mirror.check_all()
    finally:
        RealRays.refract = original
    assert len(problems) == 1, problems
    assert "optiland.rays.real_rays:RealRays.refract" in problems[0], problems


def test_mirror_check_still_catches_a_changed_constant(monkeypatch):
    """A MIRRORED constant is hashed by value; a wrapper cannot hide that."""
    from optiland.geometries import newton_raphson

    monkeypatch.setattr(newton_raphson, "_CONV_EPS_MULTIPLIER", 1e6)
    problems = trace_mirror.check_all()
    assert len(problems) == 1, problems
    assert "_CONV_EPS_MULTIPLIER" in problems[0], problems


@pytest.mark.parametrize("mode", MODES)
def test_bare_wrapper_on_group_trace_keeps_the_kernel(mode):
    """The end-to-end pin: a census wrapper must not switch the gate off.

    Exactly the shape the conftest fixture installs (a bare closure, no
    ``functools.wraps``), around the same 4096-ray Cooke bundle the rest of
    this file uses.  The counters must be the unwrapped ones.
    """
    with metal_backend(mode):
        optic = trace_fixtures.cooke()
        rays = trace_fixtures.pupil_bundle(optic, N_RAYS)
        original = SurfaceGroup.trace
        calls = []

        def _bare_census(self, rays, skip=0, record=True):
            calls.append(1)
            return original(self, rays, skip=skip, record=record)

        SurfaceGroup.trace = _bare_census
        trace.reset_driver_state()
        T.reset_stats()
        try:
            optic.surfaces.trace(rays)
        finally:
            SurfaceGroup.trace = original
        stats = T.stats()

    assert calls == [1], calls
    assert stats.get("fused_trace_skip:mirror_drift", 0) == 0, _fused(stats)
    assert stats.get("fused_trace:candidates") == 1, _fused(stats)
    assert stats.get("fused_trace:traces") == 1, _fused(stats)


def test_bare_wrapper_on_surface_trace_keeps_the_kernel(mps_df64):
    """The same, for the ``Surface.trace`` wrapper the ray-aiming test installs,
    with the conftest-shaped group wrapper on top of it (two mirrored rows
    wrapped at once, one of them twice)."""
    from optiland.surfaces.standard_surface import Surface

    optic = trace_fixtures.cooke()
    rays = trace_fixtures.pupil_bundle(optic, N_RAYS)
    original_surface_trace = Surface.trace
    surface_calls = []

    def _counting_surface_trace(self, rays, record=True):
        surface_calls.append(1)
        return original_surface_trace(self, rays, record=record)

    Surface.trace = _counting_surface_trace
    try:
        with census() as counts:
            trace.reset_driver_state()
            T.reset_stats()
            optic.surfaces.trace(rays)
            stats = T.stats()
    finally:
        Surface.trace = original_surface_trace

    assert counts.candidates == 1, counts
    assert stats.get("fused_trace_skip:mirror_drift", 0) == 0, _fused(stats)
    assert stats.get("fused_trace:traces") == 1, _fused(stats)
    # The fused path replaces the per-surface loop, so the wrapped
    # ``Surface.trace`` is not called at all on a fused trace.
    assert surface_calls == [], surface_calls


def test_render_table_round_trips_the_checked_in_table():
    """``--update`` rewrites the table it read; the renderer must not churn."""
    text = trace_mirror._TABLE_PATH.read_text(encoding="utf-8")
    start = text.index(trace_mirror._BEGIN) + len(trace_mirror._BEGIN) + 1
    end = text.index(trace_mirror._END)
    rows = [
        (
            fp.qualname,
            fp.msl_function,
            fp.klass,
            fp.sha256,
            fp.verified_note,
            fp.source_sha,
        )
        for fp in trace_mirror.FINGERPRINTS
    ]
    assert trace_mirror._render_table(rows) == text[start:end].rstrip("\n")


def test_render_table_wraps_a_long_verified_note():
    """A long ``--verified`` note must not push the file over ruff's E501.

    The note the I2 re-baseline wanted was 211 characters and made
    ``trace_mirror.py`` fail ``ruff check``, which is a commit gate (plan
    10.1); it was shortened by hand.  The renderer wraps it now, and the note
    still round-trips exactly.
    """
    note = (
        "re-verified at I2 after the WP4 hook landed: the mirrored loop body "
        "(for surface in self.surfaces[skip:]: surface.trace(rays, "
        "record=record)), self.reset() and return rays are unchanged; the only "
        "edit is the if not _fused_metal_trace(...) branch in front of them, "
        'and a 2-word "tail" with an embedded backslash \\ and quote "'
    )
    assert len(note) > 211, len(note)
    rendered = trace_mirror._render_table(
        [("a.b:C.d", "trace_body", trace_mirror.MIRRORED, "0" * 64, note, "2f9f2912")]
    )
    assert max(len(line) for line in rendered.splitlines()) <= 88, rendered
    namespace = {"MIRRORED": trace_mirror.MIRRORED, "CONTRACT": trace_mirror.CONTRACT}
    exec(rendered, namespace)  # noqa: S102 - the renderer's own output
    assert namespace["_ROWS"][0][4] == note
