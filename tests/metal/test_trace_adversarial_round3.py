"""Regression and lock tests closing verify-round 3 findings.

WP0 creates this file with one sanity test so the definition-of-done path
exists even for a round that produces no findings.  The wave-3 fix lane for
round 3 owns it from then on: every finding it closes lands here as a named
regression test (plan 0.1, 4/WP0), or as a documented-limit lock test.

Never widen a tolerance to make a test here pass (plan 0.2.2).

Round 3, iteration 1 closed five findings
(``NOTES/fused-trace-research/verify-round-3.md``), all by fix + named
regression test:

============ ============================== ================================
finding      closure                        test
============ ============================== ================================
R3-V1-01     fix (driver: the explicit      ``test_r3v101_the_driver_never_
             ``torch.mps.synchronize()``    device_synchronizes``,
             is gone; the sentinel          ``test_r3v101_concurrent_traces_
             readback already ordered       do_not_abort_the_process``
             every slab)
R3-V1-02     fix (driver: the hook's        ``test_r3v102_truthy_record_
             ``record`` is taken for its    values_trace_like_the_per_op_
             truth value, as upstream       path``
             takes it)
R3-V1-03     fix (same normalisation:       ``test_r3v103_record_policy_
             the silent half)               values_record_every_surface``
R3-V1-04     fix (hook: the three early     ``test_r3v104_diag_is_dropped_
             returns drop the group's       when_the_kill_switch_is_off``,
             DIAG planes)                   ``test_r3v104_diag_is_dropped_
                                            when_the_bundle_is_not_metal``
R3-V2-01     fix (batch API: ``install``    ``test_r3v201_install_drops_the_
             drops them too)                stale_diag_planes``
============ ============================== ================================

Round 3, iteration 2 closed five more:

============ ============================== ================================
finding      closure                        test
============ ============================== ================================
R3-V1-05     DOCUMENTED LIMIT + lock        ``test_r3v105_locked_one_main_
             (cold-start concurrency is     thread_trace_builds_the_
             unsupported on the torch-MPS   library``, ``test_r3v105_locked_
             backend, fused OR per-op)      threads_after_a_main_thread_
                                            warm_up_agree``
R3-V1-06     fix (driver: a notice about    ``test_r3v106_a_promoted_warning_
             a fallback the caller never    cannot_make_the_diag_notice_
             asked for can no longer be     raise``, ``test_r3v106_a_promoted_
             promoted to an exception)      warning_cannot_make_the_fallback_
                                            raise``
R3-V1-07     DOCUMENTED LIMIT + lock        ``test_r3v107_locked``
             (a recorded row is a view
             that pins its whole ``snap``)
R3-V2-02     fix (five MIRRORED rows +      ``test_r3v202_the_five_census_
             a completeness census in       functions_carry_a_mirrored_row``,
             test_trace_mirror_sources.py)  ``test_r3v202_an_edit_to_a_census_
                                            function_refuses_the_fused_path``
R3-V2-03     fix (gate: the pose is         ``test_r3v203_a_coordinate_system_
             accepted only for an exact     subclass_is_refused``
             ``CoordinateSystem``)
============ ============================== ================================

Round 3, iteration 3 closed three more:

============ ============================== ================================
finding      closure                        test
============ ============================== ================================
R3-V1-08     DOCUMENTED LIMIT + lock        ``test_r3v108_locked_the_budget_
             (plan 9.2's T8 threshold       is_live_bytes_and_it_meets_the_
             names ``driver_allocated_      target``, ``test_r3v108_locked_
             memory()`` while its 807 MB    the_process_holds_more_than_it_
             prediction is plan 3.6's       allocates``
             LIVE bytes; the model is
             right to 1.3 %)
R3-V1-09     DOCUMENTED LIMIT + lock        ``test_r3v109_locked``,
             (one contiguous ``snap``       ``test_r3v109_locked_the_single_
             crosses the MPS allocator's    snap_block_is_what_crosses``
             8-12 MB heap trigger that
             one ray plane does not)
R3-V2-04     fix (three MIRRORED rows for   ``test_r3v204_the_conic_solver_
             the per-op GPU conic solver,   carries_a_mirrored_row``,
             a census carve-out that lets   ``test_r3v204_an_edit_to_the_
             the alarm see it, and the      conic_solver_refuses_the_fused_
             duplicated solver epsilon      path``, ``test_r3v204_the_solver_
             pinned to ``MACHINE_EPS``)     epsilon_is_one_number_not_two``
============ ============================== ================================

R3-V1-08 and R3-V1-09 are one mechanism with two faces: PyTorch's MPS caching
allocator reserves a 1 024 MiB heap for a single allocation above ~8-12 MB,
and the fused path's ``snap`` is one contiguous buffer per raw component.
Neither is a kernel defect -- the design's own memory model is within 1.3 % of
the measurement and ``empty_cache()`` returns every byte -- and neither is
closable inside this package: the T8 row lives in ``scripts/metal_benchmark.py``
(WP8's file) and the heap lives in torch.  Both are documented limits, and the
request to WP8 is in ``NOTES/fused-trace-research/status.md``.

R3-V2-02 and R3-V2-03 are also one mechanism seen twice: a source digest
cannot see a *subclass* that overrides a mirrored method, and it cannot see a
mirrored function nobody wrote a row for.  The project's answer to the first
is an exact-type check at the gate -- six of the seven mirrored families
already had one -- and to the second a mechanical census of what actually
executes inside a trace.

R3-V1-04 and R3-V2-01 are one mechanism -- the DIAG planes were dropped only
inside ``metal/trace.py::fused_trace``, so every path that rewrote a group's
state without reaching it left ``diag_from`` describing an older trace -- with
two fix sites, and both verifiers reached it independently.  They are closed
together by :func:`optiland.backend.torch_backend.metal.trace.forget_diag`,
which the hook and ``BatchTraceResult.install`` call through ``sys.modules``
so that neither imports the driver to do it.

Every comparison is plan 7.1 **tier A**: raw components (df64 hi/lo float32
words, sf64 int64 bit patterns), ``equal_nan=True``, bundles of 4,096 rays.
No tolerance is widened anywhere, and no test here asserts only that something
is absent: each diagnostics test first proves the planes ARE there for the
trace that produced them, so a ``diag_from`` that always returned None would
fail it.
"""

from __future__ import annotations

import ast
import functools
import json
import logging
import os
import re
import subprocess
import sys
import textwrap
import types
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
from optiland.backend.torch_backend.metal import conic as metal_conic  # noqa: E402
from optiland.backend.torch_backend.metal import tensor as T  # noqa: E402
from optiland.backend.torch_backend.metal import (  # noqa: E402
    trace,
    trace_mirror,
    trace_record,
)
from optiland.backend.torch_backend.metal import trace_layout as L  # noqa: E402
from optiland.backend.torch_backend.metal.trace_adapters import (  # noqa: E402
    FusedTraceSkip,
)
from optiland.coordinate_system import CoordinateSystem  # noqa: E402
from optiland.optimization.variable import Variable  # noqa: E402
from optiland.raytrace.batch_trace import trace_batch  # noqa: E402
from optiland.surfaces import surface_group as sg  # noqa: E402
from optiland.utils import machine_eps  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))
import trace_fixtures as fx  # noqa: E402

from tests.metal import _trace_compare as tc  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator

MODES = tc.MODES

#: The bundle every plain trace here uses: 4,096 rays, so every comparison is
#: tier A (plan 7.1) and every bundle is GPU-resident (> 256).
N_RAYS = fx.DEFAULT_RAYS

#: 19 hexapolar rings = 1,141 rays for the batch: above 1024, and deliberately
#: NOT ``N_RAYS``, so a stale plane is visible as a ray count.
RINGS = 19
N_BATCH_RAYS = 1 + 3 * RINGS * (RINGS + 1)

BATCH_KWARGS = {
    "Hx": 0.0,
    "Hy": 0.0,
    "wavelength": 0.55,
    "num_rays": RINGS,
    "distribution": "hexapolar",
}

#: Values upstream's ``record`` parameter accepts because it is only ever used
#: for its truth value (``surface.trace(rays, record=record)`` -> ``if
#: record:``), paired with the truth the per-op path gives them.  Before the
#: R3-V1-02 fix the first four raised ``TypeError`` on the fused path and the
#: fifth raised ``ValueError: unknown record policy``.
TRUTHY_RECORD_VALUES: tuple[tuple[str, Any, bool], ...] = (
    ("int-1", 1, True),
    ("int-0", 0, False),
    ("np-True", np.True_, True),
    ("np-False", np.False_, False),
    ("str-yes", "yes", True),
)

#: The silent half (R3-V1-03): values that ARE in ``compile_records``' policy
#: domain and mean something else there.  The per-op path records every
#: surface for both.
POLICY_RECORD_VALUES: tuple[tuple[str, Any], ...] = (
    ("str-image", "image"),
    ("list-1-3", [1, 3]),
)


def test_round_file_present():
    """Placeholder so round 3 always has a collectable test file."""
    assert True


# ---------------------------------------------------------------------------
# Environment and backend (same contract as test_trace_adversarial_round2.py)
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


def quiet_trace(group: Any, rays: Any, **kwargs: Any) -> Any:
    """``group.trace(rays, **kwargs)`` with the driver's RuntimeWarnings off."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return group.trace(rays, **kwargs)


def row_lengths(group: Any) -> list[int]:
    """How many rays each surface's recorded row holds; 0 when it has none."""
    out: list[int] = []
    for surface in group.surfaces:
        value = getattr(surface, "x", None)
        out.append(0 if value is None else int(be.size(value)))
    return out


def fused_delta(before: dict[str, int]) -> dict[str, int]:
    """The ``fused_trace*`` counters that moved since ``before``."""
    now = T.stats()
    return {
        key: now.get(key, 0) - before.get(key, 0)
        for key in now
        if key.startswith("fused_trace") and now.get(key, 0) - before.get(key, 0) != 0
    }


# ---------------------------------------------------------------------------
# R3-V1-01 -- the fused path must not abort the process under concurrency
#
# `_launch_slabs` used to end its dispatch loop with `torch.mps.synchronize()`,
# the only such call in the package; the per-op path makes none.  That call
# commits the process-wide MPS command buffer from the calling thread, so a
# second thread tracing another optic at the same time -- upstream #833 runs
# GUI analyses on cancellable worker jobs, and the hook is default-on -- had
# an encoder open on the buffer being committed and the process died: SIGSEGV
# at two threads, SIGABRT with `failed assertion _status <
# MTLCommandBufferStatusCommitted at line 323 in
# -[IOGPUMetalCommandBuffer setCurrentCommandEncoder:]` at three or more.
#
# The sentinel readback that follows the loop already waits for every slab, so
# the call bought nothing: it is gone, and the ordering is unchanged.
# ---------------------------------------------------------------------------


def test_r3v101_the_driver_never_device_synchronizes(mps_backend, monkeypatch):
    """A fused trace makes no device-wide ``torch.mps.synchronize()`` call.

    The counting patch is installed AFTER a warm-up trace, so a library build
    or a lazy pipeline state cannot be mistaken for the driver's own call.
    """
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    optic = fx.cooke()
    group = optic.surfaces
    rays = fx.pupil_bundle(optic, N_RAYS)
    quiet_trace(group, tc.copy_rays(rays))

    calls: list[int] = []
    original = torch.mps.synchronize

    def counting_synchronize(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.mps, "synchronize", counting_synchronize)
    before = dict(T.stats())
    out = quiet_trace(group, tc.copy_rays(rays))
    delta = fused_delta(before)

    assert delta.get("fused_trace:traces") == 1, delta
    assert out.x is not None
    assert calls == [], (
        "the driver called torch.mps.synchronize() "
        f"{len(calls)} time(s); that commit from the calling thread is what "
        "aborts the process when another thread has an encoder open (R3-V1-01)"
    )


#: Run in a fresh interpreter: the failure mode under test is a process abort,
#: which no in-process assertion can survive.  Each thread traces a different
#: optic, and every capture is compared raw-component-wise (tier A) against a
#: single-threaded per-op reference, so a "fix" that merely hides the crash
#: while corrupting the results fails this test too.
_THREADS_SUBPROCESS = r"""
import json, os, sys, threading, warnings

sys.path.insert(0, "scripts")
os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")
os.environ["OPTILAND_METAL_TRACE_DIAG"] = "0"

import torch

import optiland.backend as be
import trace_fixtures as fx
from optiland.backend.torch_backend import metal
from tests.metal import _trace_compare as tc

THREADS = {threads}
TRACES = {traces}
MODE = {mode!r}
BUILDERS = (fx.cooke, fx.aspheric_singlet, fx.hubble, fx.odd_asphere_singlet)


def setup(fused):
    os.environ["OPTILAND_METAL_FUSED_TRACE"] = fused
    be.set_backend("torch")
    be.set_device("mps")
    be.set_precision("float64")
    be.grad_mode.disable()
    metal.set_mode(MODE)


def quiet(group, rays):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return group.trace(rays)


setup("0")
work, refs = [], []
with torch.no_grad():
    for k in range(THREADS):
        optic = BUILDERS[k % len(BUILDERS)]()
        rays = fx.pupil_bundle(optic, 2048)
        out = quiet(optic.surfaces, tc.copy_rays(rays))
        refs.append(tc.capture(optic.surfaces, out, MODE))

setup("1")
with torch.no_grad():
    for k in range(THREADS):
        optic = BUILDERS[k % len(BUILDERS)]()
        rays = fx.pupil_bundle(optic, 2048)
        quiet(optic.surfaces, tc.copy_rays(rays))   # warm every code path first
        work.append((optic, rays))

caps, errors = {{}}, []
barrier = threading.Barrier(THREADS)


def worker(k):
    try:
        optic, rays = work[k]
        with torch.no_grad():
            barrier.wait()
            for _ in range(TRACES):
                out = quiet(optic.surfaces, tc.copy_rays(rays))
            caps[k] = tc.capture(optic.surfaces, out, MODE)
    except BaseException as exc:
        errors.append("thread %d: %s: %s" % (k, type(exc).__name__, exc))


threads = [threading.Thread(target=worker, args=(k,)) for k in range(THREADS)]
for t in threads:
    t.start()
for t in threads:
    t.join(timeout=300)

mismatch = []
for k, ref in enumerate(refs):
    if k not in caps:
        mismatch.append("thread %d produced no capture" % k)
        continue
    try:
        tc.assert_tier_a(caps[k], ref, "thread %d" % k)
    except AssertionError as exc:
        mismatch.append(str(exc).replace("\n", " ")[:300])

print(json.dumps({{
    "alive": [t.is_alive() for t in threads],
    "errors": errors,
    "mismatch": mismatch,
}}))
"""


@pytest.mark.parametrize("threads", [2, 4])
def test_r3v101_concurrent_traces_do_not_abort_the_process(threads):
    """``threads`` threads tracing different optics survive, and agree.

    Before the fix this killed the interpreter every repetition -- measured
    here as -6 (SIGABRT, `failed assertion _status <
    MTLCommandBufferStatusCommitted`) at both thread counts, and as -11
    (SIGSEGV, silent) by the verifier at two -- while the per-op path survives
    the same shape at 2, 3, 4, 6 and 8 threads.
    """
    script = _THREADS_SUBPROCESS.format(threads=threads, traces=8, mode="df64")
    env = dict(os.environ)
    env["PYTORCH_MPS_FAST_MATH"] = "0"
    env["MPLBACKEND"] = "Agg"
    env["QT_QPA_PLATFORM"] = "offscreen"
    env.pop("OPTILAND_METAL_FUSED_TRACE", None)
    done = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert done.returncode == 0, (
        f"{threads} concurrent fused traces killed the interpreter with "
        f"returncode {done.returncode} (negative = signal; -11 SIGSEGV, "
        f"-6 SIGABRT)",
        done.stdout[-2000:],
        done.stderr[-2000:],
    )
    lines = [line for line in done.stdout.splitlines() if line.strip()]
    out = json.loads(lines[-1])
    assert out["errors"] == [], out["errors"]
    assert out["alive"] == [False] * threads, out["alive"]
    assert out["mismatch"] == [], out["mismatch"]


# ---------------------------------------------------------------------------
# R3-V1-02 / R3-V1-03 -- `record` is upstream's, and upstream's is a truth value
#
# `SurfaceGroup.trace(rays, record=...)` documents a bool and uses it only as
# `if record:`.  The hook forwarded the value verbatim into `compile_records`,
# whose domain is `bool | str | Sequence[int]`, so `record=1` raised TypeError
# (loud) and `record="image"` recorded one of Cooke's eight rows where the
# per-op path records all eight (silent, and it made the plan 1.5 rollback
# change the answer).  `fused_trace` now passes `bool(record)`.
# ---------------------------------------------------------------------------


def _compare_record_value(mode: str, monkeypatch, value: Any) -> tuple[list, list]:
    """Trace the same bundle with the hook off and on; return both row shapes.

    Asserts the fused leg really took the kernel and that the two legs agree
    raw-component-wise, and hands back the row lengths for the caller's own
    prediction.
    """
    metal.set_mode(mode)
    optic = fx.cooke()
    rays = fx.pupil_bundle(optic, N_RAYS)

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    reference = fx.cooke()
    ref_out = quiet_trace(reference.surfaces, tc.copy_rays(rays), record=value)
    ref = tc.capture(reference.surfaces, ref_out, mode)
    ref_rows = row_lengths(reference.surfaces)

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    before = dict(T.stats())
    got_out = quiet_trace(optic.surfaces, tc.copy_rays(rays), record=value)
    delta = fused_delta(before)
    got = tc.capture(optic.surfaces, got_out, mode)
    got_rows = row_lengths(optic.surfaces)

    assert delta.get("fused_trace:traces") == 1, (value, delta)
    tc.assert_tier_a(got, ref, f"record={value!r} [{mode}]")
    return ref_rows, got_rows


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    ("label", "value", "truth"),
    TRUTHY_RECORD_VALUES,
    ids=[row[0] for row in TRUTHY_RECORD_VALUES],
)
def test_r3v102_truthy_record_values_trace_like_the_per_op_path(
    mps_backend, monkeypatch, mode, label, value, truth
):
    """``record=1 / 0 / np.True_ / np.False_ / 'yes'`` still traces, both ways.

    The prediction is exact: a truthy value records all eight Cooke rows with
    ``N_RAYS`` rays each, a falsy one records none, and the fused and per-op
    legs are raw-component equal.
    """
    ref_rows, got_rows = _compare_record_value(mode, monkeypatch, value)
    expected = [N_RAYS] * 8 if truth else [0] * 8
    assert ref_rows == expected, (label, ref_rows)
    assert got_rows == expected, (label, got_rows)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    ("label", "value"),
    POLICY_RECORD_VALUES,
    ids=[row[0] for row in POLICY_RECORD_VALUES],
)
def test_r3v103_record_policy_values_record_every_surface(
    mps_backend, monkeypatch, mode, label, value
):
    """``record='image'`` and ``record=[1, 3]`` record all eight rows, as upstream.

    They are truthy, and upstream's ``record`` is a truth value; the snapshot
    policy of ``compile_records`` belongs to the batch API, which calls it
    directly.  Before the fix the fused leg recorded ``[0]*7 + [N]`` and
    ``[0, N, 0, N, 0, 0, 0, 0]`` respectively, with no error on either path.
    """
    ref_rows, got_rows = _compare_record_value(mode, monkeypatch, value)
    expected = [N_RAYS] * 8
    assert ref_rows == expected, (label, ref_rows)
    assert got_rows == expected, (label, got_rows)


# ---------------------------------------------------------------------------
# R3-V1-04 / R3-V2-01 -- `diag_from` describes the trace the group just had
#
# `fused_trace` drops a group's DIAG planes at the top, so they are that
# trace's own or absent.  Three paths rewrite a group's state without reaching
# it: the hook's early returns (the plan 1.5 `=0` rollback, a non-Metal
# bundle, no driver) and `BatchTraceResult.install`.  All four call
# `trace.forget_diag` now.
# ---------------------------------------------------------------------------


def _fused_planes(optic: Any, mode: str, n: int = N_RAYS) -> tuple:
    """Trace ``optic`` with DIAG on and return the planes it must have.

    The certificate for every test below: if this ever returns None the group
    never reached the kernel and the "planes are gone" assertions that follow
    would be vacuous.
    """
    group = optic.surfaces
    quiet_trace(group, fx.pupil_bundle(optic, n))
    planes = trace.diag_from(group)
    assert planes is not None, "no DIAG planes after a fused trace"
    assert tuple(planes[0].shape) == (1, len(group.surfaces), n), planes[0].shape
    return planes


@pytest.mark.parametrize("mode", MODES)
def test_r3v104_diag_is_dropped_when_the_kill_switch_is_off(
    mps_backend, monkeypatch, mode
):
    """The plan 1.5 runtime rollback leaves no planes behind.

    A fused DIAG trace of ``N_RAYS`` rays, then ``OPTILAND_METAL_FUSED_TRACE``
    set to ``0`` and the SAME group traced with half as many: ``diag_from``
    used to return the first trace's planes -- the same tensor, over twice the
    rays the caller holds -- which is exactly the unchecked zip its docstring
    licenses, out of range by a factor of two.
    """
    metal.set_mode(mode)
    monkeypatch.setenv("OPTILAND_METAL_TRACE_DIAG", "1")
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    optic = fx.cooke()
    group = optic.surfaces
    first = _fused_planes(optic, mode)

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    half = N_RAYS // 2
    out = quiet_trace(group, fx.pupil_bundle(optic, half))

    assert int(be.size(out.x)) == half
    assert row_lengths(group) == [half] * len(group.surfaces)
    after = trace.diag_from(group)
    assert after is None, (
        "the kill-switch trace left the previous trace's planes: "
        f"{tuple(after[0].shape)} for a {half}-ray bundle "
        f"(same tensor: {after[0].data_ptr() == first[0].data_ptr()})"
    )


@pytest.mark.parametrize("mode", MODES)
def test_r3v104_diag_is_dropped_when_the_bundle_is_not_metal(
    mps_backend, monkeypatch, mode
):
    """The hook's ray-type early return drops them too.

    The hook is called directly with a stand-in bundle, because an optic built
    under Metal holds Metal geometry parameters: tracing a real NumPy bundle
    through such a group raises inside the per-op loop before the planes
    matter (verifier 1's probe part D).  The early return is reached either
    way, and it is the return under test.
    """
    metal.set_mode(mode)
    monkeypatch.setenv("OPTILAND_METAL_TRACE_DIAG", "1")
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    optic = fx.cooke()
    group = optic.surfaces
    _fused_planes(optic, mode)

    not_metal = types.SimpleNamespace(x=np.zeros(N_RAYS))
    assert sg._fused_metal_trace(group, not_metal, 0, True) is False
    assert trace.diag_from(group) is None


@pytest.mark.parametrize("fused_batch", [True, False], ids=["fused", "loop"])
@pytest.mark.parametrize("mode", MODES)
def test_r3v201_install_drops_the_stale_diag_planes(
    mps_backend, monkeypatch, mode, fused_batch
):
    """``install`` leaves no planes describing the pre-batch trace.

    The batch never traces the caller's optic through the hook, and its own
    diagnostics are per design in ``result.status``; ``install`` rewrites the
    group's records from them.  ``diag_from`` used to keep returning the
    pre-batch trace's ``(1, S, N_RAYS)`` planes over the batch's 1,141-ray
    rows.  Both legs are covered: the fused batch and the contract loop.
    """
    metal.set_mode(mode)
    monkeypatch.setenv("OPTILAND_METAL_TRACE_DIAG", "1")
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    optic = fx.cooke()
    group = optic.surfaces
    _fused_planes(optic, mode)

    variable = Variable(optic, "radius", surface_number=1)
    base = float(variable.value)
    values = np.array([[base], [base * 1.001], [base * 0.999], [base * 1.002]])
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1" if fused_batch else "0")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = trace_batch(optic, [variable], values, record="all", **BATCH_KWARGS)
        result.install(optic, 2)

    assert result.fused is fused_batch
    assert tuple(result.status.shape) == (4, len(group.surfaces), N_BATCH_RAYS)
    assert row_lengths(group) == [N_BATCH_RAYS] * len(group.surfaces)
    after = trace.diag_from(group)
    assert after is None, (
        f"install() left planes of shape {tuple(after[0].shape)} on a group "
        f"holding {N_BATCH_RAYS}-ray rows"
    )


# ---------------------------------------------------------------------------
# R3-V1-06 -- a fallback the caller never asked for must not be able to RAISE
#
# Process state that lives in the WARNING FILTER, not in the driver: under
# `warnings.simplefilter("error")` -- `python -W error`, `pytest -W error`, a
# `filterwarnings = error` ini section -- every warning becomes an exception.
# The per-op path emits none of the driver's warnings, so each one was a place
# where an additive, default-on hook raised where the path it mirrors returns:
# 4 of 4 checks, both modes.  Two sites: `_record_diag`'s Newton notice under
# DIAG=1 (the per-op path hides non-convergence entirely) and the one-shot
# `FusedTraceUnavailableWarning` that ANNOUNCES the transparent fallback.
#
# `trace._emit` now catches the promotion and logs the same text on the
# module's logger instead, which `logging.lastResort` still prints to stderr.
# The loud channel is unchanged: `require` raises `MetalFallbackError`, and
# every refusal is counted whether or not anybody is listening.
# ---------------------------------------------------------------------------

_TRACE_LOGGER = "optiland.backend.torch_backend.metal.trace"


@pytest.mark.parametrize("mode", MODES)
def test_r3v106_a_promoted_warning_cannot_make_the_diag_notice_raise(
    mps_backend, monkeypatch, caplog, mode
):
    """DIAG=1 on a non-converging Newton system returns under ``-W error``.

    The reference leg is the per-op path in the SAME filter, so the assertion
    is "fused returns where per-op returns", not "no warning exists".
    """
    metal.set_mode(mode)
    monkeypatch.setenv("OPTILAND_METAL_TRACE_DIAG", "1")
    optic, rays_of = fx.nonconverging_asphere()
    rays = rays_of(optic, num_rays=N_RAYS)

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    reference_optic, reference_rays_of = fx.nonconverging_asphere()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ref_out = reference_optic.surfaces.trace(
            tc.copy_rays(reference_rays_of(reference_optic, num_rays=N_RAYS))
        )
    ref = tc.capture(reference_optic.surfaces, ref_out, mode)

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    quiet_trace(optic.surfaces, tc.copy_rays(rays))  # warm the library first

    before = dict(T.stats())
    caplog.clear()
    with (
        caplog.at_level(logging.WARNING, logger=_TRACE_LOGGER),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("error")
        out = optic.surfaces.trace(tc.copy_rays(rays))
    delta = fused_delta(before)
    got = tc.capture(optic.surfaces, out, mode)

    assert delta.get("fused_trace:traces") == 1, delta
    assert delta.get("fused_trace:diag:newton_not_converged", 0) > 0, (
        "the fixture no longer produces a non-converging Newton row, so this "
        f"test would pass vacuously; {delta}"
    )
    tc.assert_tier_a(got, ref, f"nonconverging_asphere DIAG -W error [{mode}]")

    logged = [r.getMessage() for r in caplog.records if r.name == _TRACE_LOGGER]
    assert any("Newton did not converge" in m for m in logged), (
        "the promoted warning was swallowed instead of demoted to the "
        f"logger; records: {logged}"
    )


@pytest.mark.parametrize("mode", MODES)
def test_r3v106_a_promoted_warning_cannot_make_the_fallback_raise(
    mps_backend, monkeypatch, caplog, mode
):
    """An unbuildable trace library falls back silently under ``-W error``.

    The library is broken with a monkeypatch rather than a source edit, which
    is the shape a real failure takes (an OS without Metal, a driver bug at
    ``compile_shader``): the driver must count ``fused_trace_skip:unavailable``
    and hand the caller the per-op answer, not an exception.
    """
    metal.set_mode(mode)
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    optic = fx.cooke()
    rays = fx.pupil_bundle(optic, N_RAYS)

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    reference = fx.cooke()
    ref_out = quiet_trace(reference.surfaces, tc.copy_rays(rays))
    ref = tc.capture(reference.surfaces, ref_out, mode)

    def broken(_mode: str) -> Any:
        raise RuntimeError("test: the trace library cannot be built")

    monkeypatch.setattr(trace, "_kernel_library", broken)
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")

    before = dict(T.stats())
    caplog.clear()
    with (
        caplog.at_level(logging.WARNING, logger=_TRACE_LOGGER),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("error")
        out = optic.surfaces.trace(tc.copy_rays(rays))
    delta = fused_delta(before)
    got = tc.capture(optic.surfaces, out, mode)

    assert delta.get("fused_trace_skip:unavailable") == 1, delta
    assert delta.get("fused_trace:traces", 0) == 0, delta
    tc.assert_tier_a(got, ref, f"unavailable library -W error [{mode}]")

    logged = [r.getMessage() for r in caplog.records if r.name == _TRACE_LOGGER]
    assert any("fused trace unavailable" in m for m in logged), logged


def test_r3v106_the_notice_is_still_an_ordinary_warning_by_default(
    mps_backend, monkeypatch
):
    """Demoting a PROMOTED warning must not stop the warning being emitted.

    Without ``-W error`` the notice is still a ``FusedTraceUnavailableWarning``
    that ``pytest.warns`` and every ordinary filter can see; only the promotion
    is refused.  Guards against a "fix" that simply deleted the warning.
    """
    metal.set_mode("df64")
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    optic = fx.cooke()
    rays = fx.pupil_bundle(optic, N_RAYS)

    def broken(_mode: str) -> Any:
        raise RuntimeError("test: the trace library cannot be built")

    monkeypatch.setattr(trace, "_kernel_library", broken)
    with pytest.warns(trace.FusedTraceUnavailableWarning, match="unavailable"):
        optic.surfaces.trace(tc.copy_rays(rays))

    # ...and only once per process (plan 1.5), which the demotion keeps.
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        optic.surfaces.trace(tc.copy_rays(rays))
    assert [w for w in seen if w.category is trace.FusedTraceUnavailableWarning] == []


# ---------------------------------------------------------------------------
# R3-V2-02 -- every function the MSL mirrors carries a fingerprint row
#
# The drift machinery was attacked from the inside in iteration 1 and held.
# This is its COVERAGE: five physics functions the kernel reproduces carried
# no row, so an edit to any of them left `check_all()` empty, the kernel
# running, and `fused` still equal to the PRE-edit answer while the Python
# path had moved -- silently and uncounted, in both modes.
#
# The rows are the fix; the census in
# `tests/metal/test_trace_mirror_sources.py::
# test_every_executed_physics_function_is_fingerprinted_or_exempt` is what
# stops a sixth from appearing at the next upstream merge.
# ---------------------------------------------------------------------------

#: The five qualnames of finding R3-V2-02 and the MSL that reproduces each.
R3V202_ROWS: tuple[tuple[str, str], ...] = (
    ("optiland.geometries.standard:StandardGeometry.surface_normal", "normal_of"),
    (
        "optiland.geometries.newton_raphson:NewtonRaphsonGeometry.surface_normal",
        "normal_of",
    ),
    ("optiland.surfaces.object_surface:ObjectSurface._trace_real", "trace_body"),
    ("optiland.geometries.base:BaseGeometry.localize", "localize"),
    ("optiland.geometries.base:BaseGeometry.globalize", "globalize"),
)


def _inject_statement(source: str) -> str:
    """``source`` plus one statement: an edit every one of the five admits.

    Two of the five (``BaseGeometry.localize``/``globalize``) have no
    ``return`` to rewrite, so the injection is a trailing assignment at the
    body's indentation instead.  It changes the AST -- which is what the
    fingerprint hashes -- without changing what the function computes, so the
    test measures the TABLE's verdict and nothing else.
    """
    lines = source.rstrip("\n").split("\n")
    last = lines[-1]
    indent = " " * (len(last) - len(last.lstrip()))
    return source.rstrip("\n") + "\n" + indent + "_r3v202_injected = 1\n"


def test_r3v202_the_five_census_functions_carry_a_mirrored_row():
    """Each of the five is a MIRRORED row whose digest matches this checkout."""
    rows = {
        fp.qualname: fp
        for fp in trace_mirror.FINGERPRINTS
        if fp.klass == trace_mirror.MIRRORED
    }
    for qualname, msl in R3V202_ROWS:
        assert qualname in rows, (
            f"{qualname} is mirrored by kernels/trace.metal::{msl} but carries "
            "no MIRRORED row, so an upstream edit to it would leave the kernel "
            "serving the old physics (R3-V2-02)"
        )
        assert rows[qualname].msl_function == msl, rows[qualname]
        assert trace_mirror.source_digest(qualname) == rows[qualname].sha256


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("qualname", [q for q, _ in R3V202_ROWS])
def test_r3v202_an_edit_to_a_census_function_refuses_the_fused_path(
    mps_backend, monkeypatch, mode, qualname
):
    """An edit to any of the five now drifts, warns, counts and refuses.

    Before the five rows existed this test's three counter assertions read
    ``mirror_drift = 0`` and ``traces = 1``: the kernel ran, and it ran the
    physics it was verified against rather than the physics the Python path
    had just been given.
    """
    metal.set_mode(mode)
    optic = fx.cooke()
    rays = fx.pupil_bundle(optic, N_RAYS)

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    reference = fx.cooke()
    ref_out = quiet_trace(reference.surfaces, tc.copy_rays(rays))
    ref = tc.capture(reference.surfaces, ref_out, mode)

    victim = trace_mirror._resolve(qualname)
    real_getsource = trace_mirror.inspect.getsource
    original_source = real_getsource(victim)
    tampered = _inject_statement(original_source)
    assert tampered != original_source, qualname

    def fake_getsource(obj: Any) -> str:
        return tampered if obj is victim else real_getsource(obj)

    monkeypatch.setattr(trace_mirror.inspect, "getsource", fake_getsource)
    trace.reset_driver_state()
    assert [p for p in trace_mirror.check_all() if qualname in p], (
        "the injection did not move the table's verdict, so this test would "
        "pass vacuously"
    )

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    before = dict(T.stats())
    with pytest.warns(trace.FusedTraceDriftWarning, match=re.escape(qualname)):
        out = optic.surfaces.trace(tc.copy_rays(rays))
    delta = fused_delta(before)
    got = tc.capture(optic.surfaces, out, mode)

    assert delta.get("fused_trace_skip:mirror_drift") == 1, delta
    assert delta.get("fused_trace:traces", 0) == 0, delta
    assert delta.get("fused_trace:candidates") == 1, delta
    tc.assert_tier_a(got, ref, f"drift {qualname} [{mode}]")


def test_r3v202_drift_in_a_census_function_raises_under_require(
    mps_backend, monkeypatch
):
    """``require`` turns the refusal into ``MetalFallbackError``, as for any row."""
    metal.set_mode("df64")
    optic = fx.cooke()
    rays = fx.pupil_bundle(optic, N_RAYS)

    qualname = "optiland.geometries.base:BaseGeometry.localize"
    victim = trace_mirror._resolve(qualname)
    real_getsource = trace_mirror.inspect.getsource
    tampered = _inject_statement(real_getsource(victim))

    def fake_getsource(obj: Any) -> str:
        return tampered if obj is victim else real_getsource(obj)

    monkeypatch.setattr(trace_mirror.inspect, "getsource", fake_getsource)
    trace.reset_driver_state()
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "require")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(trace.MetalFallbackError, match="mirror_drift"):
            optic.surfaces.trace(tc.copy_rays(rays))


# ---------------------------------------------------------------------------
# R3-V2-03 -- the pose is the last mirrored class without an exact-type gate
#
# `trace_record` keys every mirrored family by exact type: `type(group)`,
# `type(rays)`, `type(surface)`, `GEOMETRY_ADAPTERS.get(type(geometry))`,
# `APERTURE_ADAPTERS`, `INTERACTION_ADAPTERS`,
# `type(material.propagation_model)`.  The pose checked only
# `cs.reference_cs is not None`, so a `CoordinateSystem` subclass whose
# `localize` adds a shift FUSED in both modes and disagreed with the per-op
# path -- `CoordinateSystem.localize`/`globalize` are MIRRORED rows, but a
# subclass overrides them without touching the base source the digest is taken
# from, so `check_all()` cannot see it.
# ---------------------------------------------------------------------------


class ShiftedCoordinateSystem(CoordinateSystem):
    """A pose whose ``localize`` moves the rays 0.05 mm before the real one."""

    SHIFT = 0.05

    def localize(self, rays: Any) -> Any:
        rays.y = rays.y + self.SHIFT
        return super().localize(rays)


def _with_pose(optic: Any, index: int, cls: type) -> Any:
    """Replace surface ``index``'s pose with an instance of ``cls``, same values."""
    cs = optic.surfaces.surfaces[index].geometry.cs
    replacement = cls(x=cs.x, y=cs.y, z=cs.z, rx=cs.rx, ry=cs.ry, rz=cs.rz)
    optic.surfaces.surfaces[index].geometry.cs = replacement
    return optic


@pytest.mark.parametrize("mode", MODES)
def test_r3v203_a_coordinate_system_subclass_is_refused(mps_backend, monkeypatch, mode):
    """A pose subclass refuses with ``reference_cs``; an exact pose still fuses.

    Three legs, in order: the CERTIFICATE (the override really moves the
    per-op answer, so a refusal is not vacuous), the refusal itself, and the
    exact-class control that proves the new check did not simply turn the
    fused path off for every pose.
    """
    metal.set_mode(mode)
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")

    plain = fx.cooke()
    rays = fx.pupil_bundle(plain, N_RAYS)
    plain_out = quiet_trace(plain.surfaces, tc.copy_rays(rays))
    plain_ref = tc.capture(plain.surfaces, plain_out, mode)

    shifted = _with_pose(fx.cooke(), 2, ShiftedCoordinateSystem)
    shifted_out = quiet_trace(shifted.surfaces, tc.copy_rays(rays))
    shifted_ref = tc.capture(shifted.surfaces, shifted_out, mode)
    with pytest.raises(AssertionError):
        tc.assert_tier_a(shifted_ref, plain_ref, "certificate")

    # The gate names the subclass, in its own counted reason.
    gate = trace_record.can_fuse_trace(shifted.surfaces, rays, 0)
    assert gate.ok is False
    assert gate.reason is FusedTraceSkip.REFERENCE_CS
    assert gate.structural is False

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    subclassed = _with_pose(fx.cooke(), 2, ShiftedCoordinateSystem)
    before = dict(T.stats())
    got_out = quiet_trace(subclassed.surfaces, tc.copy_rays(rays))
    delta = fused_delta(before)
    got = tc.capture(subclassed.surfaces, got_out, mode)

    assert delta.get("fused_trace_skip:reference_cs") == 1, delta
    assert delta.get("fused_trace:traces", 0) == 0, delta
    tc.assert_tier_a(got, shifted_ref, f"CoordinateSystem subclass [{mode}]")

    # Control: an exact-class pose rebuilt the same way still fuses and agrees.
    control = _with_pose(fx.cooke(), 2, CoordinateSystem)
    before = dict(T.stats())
    control_out = quiet_trace(control.surfaces, tc.copy_rays(rays))
    delta = fused_delta(before)
    assert delta.get("fused_trace:traces") == 1, delta
    tc.assert_tier_a(
        tc.capture(control.surfaces, control_out, mode),
        plain_ref,
        f"exact CoordinateSystem control [{mode}]",
    )


# ---------------------------------------------------------------------------
# R3-V1-07 -- DOCUMENTED LIMIT: a recorded row is a view that pins its `snap`
# ---------------------------------------------------------------------------


def _storage_ratio(value: Any) -> list[int]:
    """Bytes of allocation each raw component of ``value`` keeps alive, / its own."""
    ratios = []
    for component in value.components:
        own = component.numel() * component.element_size()
        ratios.append(component.untyped_storage().nbytes() // own)
    return ratios


@pytest.mark.parametrize("mode", MODES)
def test_r3v107_locked(mps_backend, monkeypatch, mode):
    """DOCUMENTED LIMIT (R3-V1-07): one held row pins the whole ``snap``.

    **The limit.**  The driver writes every recorded row back as a *zero-copy
    view* of the one ``snap`` buffer (design 5, day-1 P2), which is what makes
    the writeback free.  A caller that keeps one row from each of a series of
    traces therefore keeps that trace's entire ``snap`` alive: 8 planes x
    ``n_rows`` surfaces, exactly ``8 * n_rows`` times the bytes of the row it
    thinks it is holding.  Measured over eight traces of a Cooke triplet at
    N = 1,250,000 in df64, holding ``surfaces[3].y`` from each:
    ``current_allocated_memory`` 905.6 -> 5,273.6 MB, **+624 MB per
    iteration** against the per-op path's +9.7 MB -- the 64x this test pins --
    with ``driver_allocated_memory`` at 7.2 GB by the eighth trace.  It is
    retention, not a leak: the same loop without the held rows is flat
    (827.6 MB at trace 1 and at trace 8) and both legs are tier A throughout.

    **Why it is a limit and not a bug.**  It is not a fused-vs-per-op
    divergence -- the numbers agree word for word, which this test also
    asserts -- and the zero-copy writeback is the design's, budgeted per launch
    by plan 3.6.  Bounding *accumulated* retention would mean copying every
    recorded row on every trace, i.e. paying the writeback cost the fused path
    exists to avoid, for a hazard that no shipped consumer meets.

    **The mitigation, and why exposure is small.**  Either copy the row
    (``be.copy(row)``) or reach records through ``SurfaceGroup.x/y/...``, which
    stacks into fresh storage.  Every shipped analysis does the latter.  Both
    mitigations are asserted below, so a change that broke them fails here.
    """
    metal.set_mode(mode)
    optic = fx.cooke()
    rays = fx.pupil_bundle(optic, N_RAYS)

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    reference = fx.cooke()
    ref_out = quiet_trace(reference.surfaces, tc.copy_rays(rays))
    ref = tc.capture(reference.surfaces, ref_out, mode)
    per_op_row = reference.surfaces.surfaces[3].y
    assert _storage_ratio(per_op_row) == [1] * len(per_op_row.components), (
        "the per-op path's recorded row owns its storage; if it stopped "
        "doing so this comparison would no longer name the fused path's cost"
    )

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    before = dict(T.stats())
    out = quiet_trace(optic.surfaces, tc.copy_rays(rays))
    delta = fused_delta(before)
    assert delta.get("fused_trace:traces") == 1, delta
    tc.assert_tier_a(tc.capture(optic.surfaces, out, mode), ref, f"cooke [{mode}]")

    rows = row_lengths(optic.surfaces)
    n_rows = sum(1 for length in rows if length)
    snap_planes = L.S_OPD + 1
    assert (snap_planes, n_rows) == (8, 8), (rows, snap_planes)

    fused_row = optic.surfaces.surfaces[3].y
    expected = snap_planes * n_rows  # 8 planes x 8 recorded surfaces = 64
    assert _storage_ratio(fused_row) == [expected] * len(fused_row.components), (
        f"a recorded row on the fused path keeps {_storage_ratio(fused_row)}x "
        f"its own bytes alive; the documented limit is {expected}x (8 snap "
        f"planes x {n_rows} recorded rows)"
    )

    # Both documented mitigations, so neither can regress silently.
    copied = be.copy(fused_row)
    assert _storage_ratio(copied) == [1] * len(copied.components)
    stacked = optic.surfaces.y
    assert _storage_ratio(stacked) == [1] * len(stacked.components)
    assert int(be.size(stacked)) == n_rows * N_RAYS


# ---------------------------------------------------------------------------
# R3-V1-05 -- DOCUMENTED LIMIT: concurrent tracing needs a main-thread warm-up
# ---------------------------------------------------------------------------

#: The shape the documented limit calls SUPPORTED: the process builds the
#: trace library and the torch-MPS kernels both paths share on the main
#: thread, and only then are threads started, each of which builds its OWN
#: optic and traces it for the first time concurrently.  That is the probe's
#: `precompiled` leg, measured clean 8/8 where the cold-start leg aborted.
_COLD_START_SUBPROCESS = r"""
import json, os, sys, threading, warnings

sys.path.insert(0, "scripts")
os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")
os.environ["OPTILAND_METAL_TRACE_DIAG"] = "0"
os.environ["OPTILAND_METAL_FUSED_TRACE"] = "1"

import torch

import optiland.backend as be
import trace_fixtures as fx
from optiland.backend.torch_backend import metal
from tests.metal import _trace_compare as tc

THREADS = {threads}
TRACES = {traces}
MODE = {mode!r}
BUILDERS = (fx.cooke, fx.aspheric_singlet, fx.hubble, fx.odd_asphere_singlet)


def quiet(group, rays):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return group.trace(rays)


be.set_backend("torch")
be.set_device("mps")
be.set_precision("float64")
be.grad_mode.disable()
metal.set_mode(MODE)

# Per-op references, and THE DOCUMENTED MITIGATION: one main-thread trace.
refs = []
with torch.no_grad():
    os.environ["OPTILAND_METAL_FUSED_TRACE"] = "0"
    for k in range(THREADS):
        optic = BUILDERS[k % len(BUILDERS)]()
        rays = fx.pupil_bundle(optic, 2048)
        out = quiet(optic.surfaces, tc.copy_rays(rays))
        refs.append(tc.capture(optic.surfaces, out, MODE))
    os.environ["OPTILAND_METAL_FUSED_TRACE"] = "1"
    warm = fx.cooke()
    quiet(warm.surfaces, tc.copy_rays(fx.pupil_bundle(warm, 2048)))

caps, errors = {{}}, []
barrier = threading.Barrier(THREADS)


def worker(k):
    try:
        with torch.no_grad():
            # Built INSIDE the thread: the thread's first fused trace of this
            # optic is concurrent with the other three.
            optic = BUILDERS[k % len(BUILDERS)]()
            rays = fx.pupil_bundle(optic, 2048)
            barrier.wait()
            for _ in range(TRACES):
                out = quiet(optic.surfaces, tc.copy_rays(rays))
            caps[k] = tc.capture(optic.surfaces, out, MODE)
    except BaseException as exc:
        errors.append("thread %d: %s: %s" % (k, type(exc).__name__, exc))


threads = [threading.Thread(target=worker, args=(k,)) for k in range(THREADS)]
for t in threads:
    t.start()
for t in threads:
    t.join(timeout=300)

mismatch = []
for k, ref in enumerate(refs):
    if k not in caps:
        mismatch.append("thread %d produced no capture" % k)
        continue
    try:
        tc.assert_tier_a(caps[k], ref, "thread %d" % k)
    except AssertionError as exc:
        mismatch.append(str(exc).replace("\n", " ")[:300])

print(json.dumps({{
    "alive": [t.is_alive() for t in threads],
    "errors": errors,
    "mismatch": mismatch,
}}))
"""


@pytest.mark.parametrize("mode", MODES)
def test_r3v105_locked_one_main_thread_trace_builds_the_library(
    mps_backend, monkeypatch, mode
):
    """DOCUMENTED LIMIT (R3-V1-05): the mitigation is one main-thread trace.

    **The limit.**  Concurrent tracing from a COLD START -- threads whose very
    first trace in the process is concurrent -- is unsupported on the
    torch-MPS backend, on the fused path *and* on the per-op path.  Measured
    at 4 threads: fused sf64 aborted 3 times in its first 6 runs (once
    SIGTRAP, twice SIGSEGV, all silent) and then went 20/20 clean; fused df64
    7/8; the **per-op control at the same shape 7/8, its one failure a hang
    past 300 s**.  A sampled hang
    (``NOTES/fused-trace-research/probes/round3/r3i2_03_hung_thread_sample.txt``)
    shows one worker owning torch's ``DispatchQueue_75: metal gpu stream``
    inside ``MetalShaderLibrary::getLibraryPipelineState`` while the other
    three block in ``_dispatch_sync_f_slow`` on that same queue, on torch's
    OWN kernels (``or_kernel_mps``, ``fill_mps_kernel``,
    ``structured_cat_out_mps``) that both paths use.  The trace library's own
    build is ruled out: a probe-only lock around ``trace._kernel_library``
    does not help (df64 7/8) and a warm process that drops the library cache
    and rebuilds it concurrently is clean 48/48 on both paths and both modes.

    **Why it is a limit and not a fix.**  There is no reproducible trigger in
    this package's code, and the evidence points into PyTorch's MPS dispatch
    queue, which the per-op path shares.  A code change here would be a guess.

    **The mitigation.**  Trace once on the main thread before starting
    threads; that is all it takes, and this test pins it: one ordinary trace
    populates the driver's per-mode library cache, so no worker thread has to
    build a Metal library or a pipeline state.
    ``test_r3v105_locked_threads_after_a_main_thread_warm_up_agree`` runs the
    resulting shape for real.  Once warm, 4 and 6 threads x 20 traces are
    clean and tier-A exact
    (``test_r3v101_concurrent_traces_do_not_abort_the_process``).
    """
    metal.set_mode(mode)
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    trace.reset_driver_state(libraries=True)
    assert mode not in trace._LIBS

    optic = fx.cooke()
    before = dict(T.stats())
    quiet_trace(optic.surfaces, tc.copy_rays(fx.pupil_bundle(optic, N_RAYS)))
    delta = fused_delta(before)

    assert delta.get("fused_trace:traces") == 1, delta
    assert mode in trace._LIBS, (
        "one main-thread trace no longer leaves the mode's trace library "
        "cached, so the documented mitigation for R3-V1-05 no longer holds"
    )


def test_r3v105_locked_threads_after_a_main_thread_warm_up_agree():
    """Four threads whose first trace of their OWN optic is concurrent.

    The supported half of documented limit R3-V1-05, run for real and in a
    fresh interpreter (the unsupported half's failure mode is a process
    abort, which no in-process assertion survives).  This is deliberately NOT
    ``test_r3v101_concurrent_traces_do_not_abort_the_process``'s shape: there
    every thread's optic is traced on the main thread first, here only a
    single unrelated Cooke trace is, so each thread still compiles its own
    records and takes its first launch concurrently.  Every capture is
    compared raw-component-wise against a single-threaded per-op reference.
    """
    script = _COLD_START_SUBPROCESS.format(threads=4, traces=5, mode="df64")
    env = dict(os.environ)
    env["PYTORCH_MPS_FAST_MATH"] = "0"
    env["MPLBACKEND"] = "Agg"
    env["QT_QPA_PLATFORM"] = "offscreen"
    env.pop("OPTILAND_METAL_FUSED_TRACE", None)
    done = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert done.returncode == 0, (
        "four threads tracing after a main-thread warm-up killed the "
        f"interpreter with returncode {done.returncode} (negative = signal)",
        done.stdout[-2000:],
        done.stderr[-2000:],
    )
    lines = [line for line in done.stdout.splitlines() if line.strip()]
    out = json.loads(lines[-1])
    assert out["errors"] == [], out["errors"]
    assert out["alive"] == [False] * 4, out["alive"]
    assert out["mismatch"] == [], out["mismatch"]


# ---------------------------------------------------------------------------
# R3-V2-04 -- the completeness alarm could not look at the conic solver
#
# R3-V2-02's census drops every `optiland.backend.*` qualname with the reason
# "both paths run the same df64_core.h / sf64_core.h arithmetic, so an edit
# under here moves the two answers together".  For the elementwise arithmetic
# that is true.  `optiland/backend/torch_backend/metal/conic.py` is not
# arithmetic: it is the per-op GPU path's own conic solver -- the tier-A
# reference plan 7.1 measures the kernel against -- and the fused kernel runs
# none of it.  Verifier 2 injected a root-order edit into `conic_candidates`
# and got R3-V2-02's exact shape back on cooke and hubble, both modes:
# certificate on the Python path, `check_all()` clean, `mirror_drift = 0`,
# `fused_trace:traces = 1`, fused != per-op at tier A while fused == the
# pre-edit answer.
#
# Closed the way the finding's option (a) asks, plus the epsilon half of its
# option (c):
#   * three MIRRORED rows -- `conic_candidates` (the root pair and the five
#     flag bits), `_ConicMetal.forward` (its composition with
#     `_select_distance`) and `_scalar_float` (the host read of R and k that
#     `trace_adapters._as_float` mirrors into SR_R / SR_K);
#   * a carve-out in `test_trace_mirror_sources.py::_CONIC_PREFIXES` so the
#     census can see that tree at all, with a GPU census leg behind it;
#   * the solver epsilon, which was duplicated as the literals `2.0**-48` /
#     `2.0**-53` although it is `MACHINE_EPS[mode]` -- a MIRRORED row the
#     kernel reads as `consts[C_EPS]` -- is now read from that table, so the
#     two copies cannot part.
# ---------------------------------------------------------------------------

#: The three qualnames of finding R3-V2-04 and the MSL each one answers to.
R3V204_ROWS: tuple[tuple[str, str], ...] = (
    ("optiland.backend.torch_backend.metal.conic:conic_candidates", "conic_candidates"),
    (
        "optiland.backend.torch_backend.metal.conic:_ConicMetal.forward",
        "conic_distance",
    ),
    (
        "optiland.backend.torch_backend.metal.conic:_scalar_float",
        "host:compile_records",
    ),
)


def test_r3v204_the_conic_solver_carries_a_mirrored_row():
    """Each of the three is a MIRRORED row whose digest matches this checkout."""
    rows = {
        fp.qualname: fp
        for fp in trace_mirror.FINGERPRINTS
        if fp.klass == trace_mirror.MIRRORED
    }
    for qualname, msl in R3V204_ROWS:
        assert qualname in rows, (
            f"{qualname} is the per-op GPU path's conic solver, which plan 7.1 "
            "makes the kernel's tier-A reference, but it carries no MIRRORED "
            "row: an edit to it moves the per-op answer while the kernel keeps "
            "serving the physics it was verified against (R3-V2-04)"
        )
        assert rows[qualname].msl_function == msl, rows[qualname]
        assert trace_mirror.source_digest(qualname) == rows[qualname].sha256


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("qualname", [q for q, _ in R3V204_ROWS])
def test_r3v204_an_edit_to_the_conic_solver_refuses_the_fused_path(
    mps_backend, monkeypatch, mode, qualname
):
    """An edit to the conic solver now drifts, warns, counts and refuses.

    Before the three rows existed this test's counter assertions read
    ``mirror_drift = 0`` and ``traces = 1``: the kernel ran, against a
    reference the Python path had just stopped computing.
    """
    metal.set_mode(mode)
    optic = fx.cooke()
    rays = fx.pupil_bundle(optic, N_RAYS)

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    reference = fx.cooke()
    ref_out = quiet_trace(reference.surfaces, tc.copy_rays(rays))
    ref = tc.capture(reference.surfaces, ref_out, mode)

    victim = trace_mirror._resolve(qualname)
    real_getsource = trace_mirror.inspect.getsource
    tampered = _inject_statement(real_getsource(victim))

    def fake_getsource(obj: Any) -> str:
        return tampered if obj is victim else real_getsource(obj)

    monkeypatch.setattr(trace_mirror.inspect, "getsource", fake_getsource)
    trace.reset_driver_state()
    assert [p for p in trace_mirror.check_all() if qualname in p], (
        "the injection did not move the table's verdict, so this test would "
        "pass vacuously"
    )

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    before = dict(T.stats())
    with pytest.warns(trace.FusedTraceDriftWarning, match=re.escape(qualname)):
        out = optic.surfaces.trace(tc.copy_rays(rays))
    delta = fused_delta(before)
    got = tc.capture(optic.surfaces, out, mode)

    assert delta.get("fused_trace_skip:mirror_drift") == 1, delta
    assert delta.get("fused_trace:traces", 0) == 0, delta
    assert delta.get("fused_trace:candidates") == 1, delta
    tc.assert_tier_a(got, ref, f"drift {qualname} [{mode}]")


@pytest.mark.parametrize("mode", MODES)
def test_r3v204_the_solver_epsilon_is_one_number_not_two(mps_backend, mode):
    """The conic solver's epsilon is ``MACHINE_EPS[mode]``, not a copy of it.

    ``metal/conic.py`` passed the solver epsilon as the hard-coded literals
    ``2.0**-48`` / ``2.0**-53``.  Both happen to equal ``MACHINE_EPS[mode]``
    -- a MIRRORED row that the kernel reads as ``consts[C_EPS]`` (plan 3.2)
    and that ``optiland.utils.machine_eps`` reads for the Python path -- so
    the per-op reference and the kernel carried two independent copies of one
    mirrored constant.  A digest of ``conic_candidates`` cannot see the other
    copy move, and re-baselining ``MACHINE_EPS`` (plan 0.2.8) would not have
    touched the literal.  There is one copy now, and this test pins it.
    """
    metal.set_mode(mode)
    tree = ast.parse(
        textwrap.dedent(trace_mirror.inspect.getsource(metal_conic.conic_candidates))
    )
    trace_mirror._strip_docstring(tree)  # the docstring NAMES the old literals
    powers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow)
    ]
    assert powers == [], (
        f"a power-of-two literal is back in conic_candidates "
        f"({[ast.unparse(node) for node in powers]}): the solver epsilon is "
        "MACHINE_EPS[mode], and a second copy of a mirrored constant is "
        "exactly what R3-V2-04 closed"
    )
    assert "MACHINE_EPS[mode]" in ast.unparse(tree)

    # The three copies that must stay one number: the table, the kernel's
    # consts[C_EPS], and what the Python path's machine_eps() reports.
    expected = {"df64": 2.0**-48, "sf64": 2.0**-53}[mode]
    assert T.MACHINE_EPS[mode] == expected
    assert float(trace._consts_array(mode)[L.C_EPS]) == expected
    assert machine_eps(be.array([1.0, 2.0])) == expected


# ---------------------------------------------------------------------------
# R3-V1-08 / R3-V1-09 -- DOCUMENTED LIMITS: the MPS allocator's 1 GiB heap
#
# Plan 3.6 budgets the LIVE bytes of one launch against a fraction of
# `recommended_max_memory()`.  Plan 9.2's T8 row states its threshold on
# `torch.mps.driver_allocated_memory()`, which is a different quantity: what
# the process HOLDS from the Metal driver, i.e. live blocks plus PyTorch's MPS
# caching allocator's heaps.  On this machine a single MPS allocation above
# ~8-12 MB makes the allocator take a 1 024 MiB heap (measured with plain
# torch, no Optiland, in `probes/round3/r3i3_07`), and the fused path's `snap`
# is ONE contiguous buffer per component, so it crosses that at N ~ 60 000
# where the per-op path's largest single block -- one ray plane -- does not
# until ~1e6 rays.
#
# Neither finding is a kernel defect and neither is closable by a fix in this
# package: R3-V1-08 is a measurement choice in `scripts/metal_benchmark.py`
# (WP8's file, which this lane does not own) and R3-V1-09 is torch's
# allocator.  Both are documented limits with lock tests, and the request to
# WP8 is in `NOTES/fused-trace-research/status.md`.
# ---------------------------------------------------------------------------

MIB = 1024 * 1024

#: Runs ONE memory measurement in a FRESH interpreter and prints one JSON
#: line.  One measurement per process, not several: PyTorch's MPS caching
#: allocator keeps the heaps it has taken, so a second cell in the same
#: process is absorbed by the first cell's heap and reads a number that is
#: neither path's (measured: N = 60 000 fused reads 1 072.6 MiB alone and
#: 136.6 MiB after an N = 55 000 cell in the same process).  The probes this
#: closes -- `probes/round3/r3i3_07`, `r3i3_09` -- fork per cell for the same
#: reason.
#:
#: Legs: ``control`` (plain torch, no Optiland: the precondition both limits
#: rest on), ``t8 <mode>`` (plan 9.2's T8 shape), ``trace <mode> <N> <0|1>``
#: (one ordinary hook trace of cooke).
_MEMORY_SUBPROCESS = r"""
import gc, json, os, sys, warnings

sys.path.insert(0, "scripts")
os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")
os.environ["OPTILAND_METAL_TRACE_DIAG"] = "0"

import torch

MIB = 1024 * 1024
LEG = sys.argv[1]
out = {"leg": LEG}


def quiet(fn):
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn()


if LEG == "control":
    # No Optiland in this process at all: one allocation, twice.
    torch.mps.empty_cache()
    small = torch.zeros(8 * MIB // 4, dtype=torch.float32, device="mps")
    out["held_8mb"] = torch.mps.driver_allocated_memory()
    del small
    torch.mps.empty_cache()
    big = torch.zeros(12 * MIB // 4, dtype=torch.float32, device="mps")
    out["held_12mb"] = torch.mps.driver_allocated_memory()
    del big
    torch.mps.empty_cache()
    out["after_empty_cache"] = torch.mps.driver_allocated_memory()
else:
    import numpy as np

    import optiland.backend as be
    import trace_fixtures as fx
    from optiland.backend.torch_backend import metal
    from tests.metal import _trace_compare as tc

    MODE = sys.argv[2]
    os.environ["OPTILAND_METAL_FUSED_TRACE"] = "1"
    be.set_backend("torch")
    be.set_device("mps")
    be.set_precision("float64")
    be.grad_mode.disable()
    metal.set_mode(MODE)
    out["mode"] = MODE

    if LEG == "t8":
        # Plan 9.2's T8 shape: 100 designs x 1e5 rays, record="image", shared
        # launch.  The result is held, so the whole launch is still live.
        from optiland.optimization.variable import Variable
        from optiland.raytrace.batch_trace import trace_batch

        optic = fx.cooke()
        var = Variable(optic, "radius", surface_number=1)
        B, N = 100, 100000
        values = np.full((B, 1), var.variable.scale(22.01359)) + np.linspace(
            -0.01, 0.01, B
        ).reshape(B, 1)
        torch.mps.empty_cache()
        res = quiet(
            lambda: trace_batch(
                optic,
                [var],
                values,
                Hx=0.0,
                Hy=0.0,
                wavelength=optic.primary_wavelength,
                num_rays=N,
                distribution="random",
                record="image",
                shared_launch="post",
            )
        )
    else:
        N, fused = int(sys.argv[3]), sys.argv[4]
        optic = fx.cooke()
        rays = quiet(lambda: fx.pupil_bundle(optic, num_rays=N))
        os.environ["OPTILAND_METAL_FUSED_TRACE"] = fused
        torch.mps.empty_cache()
        res = quiet(lambda: optic.surfaces.trace(tc.copy_rays(rays)))
        out["fused"] = fused
    out["N"] = N
    out["live"] = torch.mps.current_allocated_memory()
    out["held"] = torch.mps.driver_allocated_memory()
    # Every reference to the launch, including the recorded rows the optic
    # keeps as zero-copy views of `snap` (documented limit R3-V1-07).
    res = optic = rays = None
    gc.collect()
    torch.mps.empty_cache()
    out["after_empty_cache"] = torch.mps.driver_allocated_memory()

print(json.dumps(out))
"""


def _run_memory_leg(*argv: str) -> dict[str, Any]:
    """One ``_MEMORY_SUBPROCESS`` run; returns its JSON."""
    env = dict(os.environ)
    env["PYTORCH_MPS_FAST_MATH"] = "0"
    env["MPLBACKEND"] = "Agg"
    env["QT_QPA_PLATFORM"] = "offscreen"
    env.pop("OPTILAND_METAL_FUSED_TRACE", None)
    done = subprocess.run(
        [sys.executable, "-c", _MEMORY_SUBPROCESS, *argv],
        cwd=str(_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert done.returncode == 0, (
        f"the memory probe {argv} exited {done.returncode}",
        done.stdout[-2000:],
        done.stderr[-2000:],
    )
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    assert lines, (done.stdout[-2000:], done.stderr[-2000:])
    return json.loads(lines[-1])


@functools.cache
def _allocator_heap_or_skip() -> dict[str, Any]:
    """The plain-torch control, once per session.

    Both limits are properties of PyTorch's MPS caching allocator, so both
    state their precondition rather than assuming it: one 12 MB allocation in
    a process that holds nothing else makes it reserve a 1 024 MiB heap.
    Where that is not true the limits do not apply and the measured tests skip
    instead of asserting an allocator property that is not there.
    """
    control = _run_memory_leg("control")
    if control["held_12mb"] < 512 * MIB:
        pytest.skip(  # pragma: no cover - other torch builds / other machines
            "this torch's MPS allocator does not reserve a heap for a single "
            f"12 MB allocation (held {control['held_12mb'] / MIB:.1f} MiB), so "
            "documented limits R3-V1-08 / R3-V1-09 do not apply here"
        )
    return control


def test_r3v108_locked_the_budget_is_live_bytes_and_it_meets_the_target(mps_backend):
    """DOCUMENTED LIMIT (R3-V1-08): T8's threshold names the wrong quantity.

    **The limit.**  Plan 9.2 row T8 asks for peak
    ``driver_allocated_memory() < 1 GB`` during the 1e2 x 1e5,
    ``record="image"``, shared-launch batch, and predicts 807 MB.  That
    prediction is plan 3.6's table, which counts the **live** bytes of one
    launch.  Measured on the T8 shape itself: live **780.0 MiB**
    (817.9 MB) against the model's **807.4 MB** -- the design's memory model
    is right to **1.3 %** -- while ``driver_allocated_memory()`` reads
    **1 080.6 MiB (df64)** and **1 697.2 MiB (sf64)**, i.e. 1.06x and 1.66x
    the threshold, for that same 780 MiB of live data.
    ``torch.mps.empty_cache()`` returns it to 0.6 / 1.2 MiB, so nothing is
    leaked or retained: the whole gap is PyTorch's MPS caching allocator
    taking a 1 024 MiB heap (mechanism: R3-V1-09).

    **What this test pins.**  That the quantity the design budgets is the live
    one, that it meets T8's number with 25 % to spare (807.4 MB against
    1 073.7 MB), and that nothing in the
    driver or the gate ever reads ``driver_allocated_memory()`` -- so no
    counter, no ``fused_trace_skip:memory`` and no gate decision can see the
    held bytes.  The measured half is
    ``test_r3v108_locked_the_process_holds_more_than_it_allocates``.

    **For the benchmark run (WP8, `scripts/metal_benchmark.py`, not this
    lane's file).**  T8 must not be reported as a FAIL of the kernel's memory
    behaviour on a number that measures the allocator: quote
    ``current_allocated_memory()`` beside ``driver_allocated_memory()``, or
    restate the threshold as "live bytes < 1 GB; held bytes reported" with the
    1 024 MiB heap named.  The request is in ``status.md``.
    """
    metal.set_mode("df64")
    optic = fx.cooke()
    w0 = trace_record.canonical_w0(float(optic.primary_wavelength), "df64")
    records = trace_record.compile_records(
        optic.surfaces, w0, "df64", record="image", designs=100
    )
    B, N, S = 100, 100000, records.S
    assert (records.B, records.n_rows, S) == (B, 1, 8), records

    # Plan 3.6's T8 row, term by term: 640 MB + 160 MB + 7.2 MB = 807 MB.
    snap = L.S_PLANES * B * records.n_rows * N * 8
    status_iters = 2 * B * S * N
    launch = L.Q_PLANES * 1 * N * 8
    tables = B * S * (L.SR_STRIDE * 8 + L.SI_STRIDE * 4 + records.C * 8)
    assert (snap, status_iters, launch) == (640_000_000, 160_000_000, 7_200_000)

    total = trace_record.memory_bytes(records, N, launch_designs=1, write_final=False)
    assert total == snap + status_iters + launch + tables == 807_436_800
    assert total < 1024**3, "the design's own budget misses plan 9.2's T8 target"

    # Nothing in the package consults the allocator's held bytes, so the
    # budget, the `memory` refusal and every counter are live-byte quantities.
    for module in (trace, trace_record):
        source = trace_mirror.inspect.getsource(module)
        assert "driver_allocated_memory" not in source, module.__name__
    assert "recommended_max_memory" in trace_mirror.inspect.getsource(trace)


def test_r3v108_locked_the_process_holds_more_than_it_allocates():
    """The measured half of R3-V1-08, in a fresh interpreter.

    Run in a subprocess because PyTorch's MPS caching allocator retains across
    tests: in-process this would report whatever ran before it.  The plain
    torch control in the same process is the precondition -- one 12 MB
    allocation, no Optiland, and the process holds 1 024.4 MiB.
    """
    control = _allocator_heap_or_skip()
    out = _run_memory_leg("t8", "df64")
    live, held = out["live"], out["held"]
    model = 807_436_800  # plan 3.6's T8 row, exactly (see the test above)

    assert abs(live - 817_889_280) < 0.02 * 817_889_280, (
        f"live bytes moved: {live / MIB:.1f} MiB against the documented "
        "780.0 MiB; the limit's arithmetic no longer describes this shape"
    )
    assert live > model, live
    assert (live - model) / model < 0.03, (
        f"the design's memory model is documented as right to 1.3 % "
        f"(live {live / 1e6:.1f} MB vs model {model / 1e6:.1f} MB); it is now "
        f"off by {(live - model) / model:.1%}"
    )
    assert held >= 1024 * MIB, (
        f"the process now holds {held / MIB:.1f} MiB for the T8 shape, i.e. "
        "it MEETS plan 9.2's target.  If torch's allocator changed, "
        "documented limit R3-V1-08 is obsolete and the report must say so"
    )
    assert out["after_empty_cache"] <= 8 * MIB, (
        "empty_cache() no longer returns the held bytes, so this is retention "
        f"and not allocator granularity: {out['after_empty_cache'] / MIB:.1f} MiB"
    )
    # The control says the gap is allocator granularity and nothing else: the
    # same process holding one 8 MB block holds 32.4 MiB, one 12 MB block
    # 1 024.4 MiB.
    assert control["held_8mb"] < 512 * MIB, control


# ---------------------------------------------------------------------------
# R3-V1-09 -- DOCUMENTED LIMIT: one contiguous `snap` crosses the heap
# threshold that one ray plane does not
# ---------------------------------------------------------------------------

#: N at which the fused path's single `snap` allocation crosses the measured
#: 8-12 MB trigger for an 8-row Cooke: 8 planes x 8 rows x N x 4 bytes.
_R3V109_N = 60_000


@pytest.mark.parametrize("mode", MODES)
def test_r3v109_locked(mps_backend, monkeypatch, mode):
    """DOCUMENTED LIMIT (R3-V1-09): the fused path makes one big allocation.

    **The limit.**  The driver allocates ``snap`` as ONE contiguous buffer per
    raw component -- ``_empty_components(S_PLANES * B * n_rows * N, mode)``,
    ``metal/trace.py`` -- while the per-op path's largest single block is one
    ray plane (``N * 4`` bytes in df64, ``N * 8`` in sf64).  PyTorch's MPS
    caching allocator reserves a **1 024 MiB heap for a single allocation
    above ~8-12 MB** (measured with plain torch and no Optiland,
    ``probes/round3/r3i3_07``), so at N = 60 000 an 8-row Cooke trace leaves
    the process holding **1 072.6 MiB (df64) / 1 073.2 MiB (sf64)** for a
    33.6 MiB live set -- **22.1x** the per-op path's 48.6 MiB -- and 1 080.6 /
    1 081.2 MiB at N = 120 001 against 88.6 / 96.6 MiB.  Up to N = 55 000 the
    two paths are within 2.5x (the sweep from N = 20 000 is in the documented
    limit's table).  The number is reached at the FIRST trace and
    does not move through 60, it is the same for a constant N and for an N
    cycling over five sizes (so it is not fragmentation), and
    ``torch.mps.empty_cache()`` releases all of it (1 081 -> 0.6 MiB).

    **What is NOT claimed.**  The crossover sits at the same N in both modes
    although one ``snap`` component is 14.65 MiB there in df64 and 29.30 MiB
    in sf64, and sf64 at N = 32 768 has a 16.00 MiB component and does not
    cross.  "One allocation over ~8-12 MB" is the shape of the trigger, per
    the plain-torch control; the exact rule inside torch's MPS allocator was
    not determined and this test does not pretend to pin it.

    **Why it is a limit and not a bug.**  Plan 3.6 budgets one launch's LIVE
    bytes against ``recommended_max_memory()``, a constant that knows nothing
    about what the process already holds, so neither the gate nor
    ``fused_trace_skip:memory`` nor any counter sees this (pinned by
    ``test_r3v108_locked_the_budget_is_live_bytes_and_it_meets_the_target``).
    Splitting ``snap`` into one buffer per surface row would change plan 3.2's
    frozen buffer layout and the zero-copy writeback with it -- an integrator
    decision, not a fix lane's, and one that trades a documented allocator
    property for a kernel-side cost.

    **The mitigation.** ``torch.mps.empty_cache()`` after a large trace, which
    returns the process to its live set; a consumer on a 16 GB machine (plan
    3.6's own example) that traces 60 000+ rays should expect to hold a
    gigabyte until it does.

    This test pins the MECHANISM with exact numbers rather than a memory
    measurement, which no in-process assertion can make honestly (the
    allocator retains across tests);
    ``test_r3v109_locked_the_single_snap_block_is_what_crosses`` runs the
    measurement in a fresh interpreter.
    """
    metal.set_mode(mode)
    itemsize = 4 if mode == "df64" else 8
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")

    optic = fx.cooke()
    rays = fx.pupil_bundle(optic, _R3V109_N)
    assert int(be.size(rays.x)) == _R3V109_N
    before = dict(T.stats())
    out = quiet_trace(optic.surfaces, tc.copy_rays(rays))
    delta = fused_delta(before)
    assert delta.get("fused_trace:traces") == 1, delta

    rows = row_lengths(optic.surfaces)
    n_rows = sum(1 for length in rows if length)
    snap_planes = L.S_OPD + 1
    assert (snap_planes, n_rows) == (8, 8), (rows, snap_planes)

    fused_row = optic.surfaces.surfaces[3].y
    blocks = [c.untyped_storage().nbytes() for c in fused_row.components]
    expected = snap_planes * n_rows * _R3V109_N * itemsize
    assert blocks == [expected] * len(blocks), (
        f"the fused path's single snap allocation is {blocks} bytes; the "
        f"documented limit is one contiguous {expected} byte block per raw "
        "component (8 snap planes x 8 recorded rows x N)"
    )
    assert expected == {"df64": 15_360_000, "sf64": 30_720_000}[mode]

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    reference = fx.cooke()
    ref_out = quiet_trace(reference.surfaces, tc.copy_rays(rays))
    per_op_row = reference.surfaces.surfaces[3].y
    per_op_blocks = [c.untyped_storage().nbytes() for c in per_op_row.components]
    assert per_op_blocks == [_R3V109_N * itemsize] * len(per_op_blocks), (
        "the per-op path's largest recorded block is one ray plane; if that "
        "changed, this comparison no longer names the fused path's cost"
    )

    # The one comparison that decides which side of the allocator's 8-12 MB
    # trigger each path lands on, at the N the limit names.
    assert expected > 12 * MIB > _R3V109_N * itemsize, (blocks, per_op_blocks)

    tc.assert_tier_a(
        tc.capture(optic.surfaces, out, mode),
        tc.capture(reference.surfaces, ref_out, mode),
        f"cooke N={_R3V109_N} [{mode}]",
    )


def test_r3v109_locked_the_single_snap_block_is_what_crosses():
    """The measured half of R3-V1-09, in a fresh interpreter.

    Two cells, ONE PROCESS EACH -- N = 60 000 fused and per-op -- because the
    allocator keeps what it has taken: measured, an N = 55 000 cell in the
    same process absorbs the N = 60 000 allocation and the fused number reads
    136.6 MiB instead of 1 072.6 MiB.  60 000 is the first N at which the
    fused path holds a gigabyte and the per-op path does not; the full sweep
    from N = 20 000 is in the documented limit's table.
    """
    _allocator_heap_or_skip()
    above_fused = _run_memory_leg("trace", "df64", str(_R3V109_N), "1")
    above_perop = _run_memory_leg("trace", "df64", str(_R3V109_N), "0")

    assert above_fused["held"] >= 1024 * MIB, (
        f"at N = 60 000 the fused path held {above_fused['held'] / MIB:.1f} "
        "MiB in the measurement this limit records (1 072.6 MiB); if torch's "
        "allocator changed, documented limit R3-V1-09 is obsolete"
    )
    assert above_perop["held"] < 128 * MIB, (
        f"the per-op control also crossed ({above_perop['held'] / MIB:.1f} "
        "MiB), so the limit is no longer about the one contiguous snap block"
    )
    assert above_fused["held"] > 8 * above_fused["live"], (
        above_fused["held"],
        above_fused["live"],
    )
    assert above_fused["after_empty_cache"] <= 8 * MIB, (
        "empty_cache() no longer returns the reserve, so this would be "
        "retention rather than allocator granularity: "
        f"{above_fused['after_empty_cache'] / MIB:.1f} MiB"
    )
    assert (
        above_fused["live"] == above_perop["live"]
        or abs(above_fused["live"] - above_perop["live"]) < 4 * MIB
    ), (
        "the two paths no longer hold the same live set at this N, so a "
        "held-bytes comparison between them would be measuring something else",
        above_fused["live"],
        above_perop["live"],
    )
