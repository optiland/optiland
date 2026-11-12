"""Tests for the threads of the compiled trace (``be.set_trace_threads``).

A trace on several threads must give the bits of a trace on one thread, for
any number of threads: the same ray arrays and records, with NaN at the same
positions. Each thread runs the operations of one thread on its own rays, so
the tests compare bits, not values within a tolerance: a tolerance would
hide a part that ran other operations, or a ray that no part traced.

The threads apply only to the compiled NumPy trace, so the tests run on the
NumPy backend and do not use ``set_test_backend``;
``test_torch_backend_ignores_the_threads`` checks Torch. The ``small_parts``
fixture lets small bundles use threads, in parts of one block of rays, so
that the tests are fast and have many parts. It also sets ``_MIN_RAYS`` of
the traces in parts to 10**12, so that a trace that leaves the plan runs
the generic loop over the surfaces, as in ``tests/test_trace_plan.py``.

Manuel Fragata Mendes, 2026
"""

from __future__ import annotations

import concurrent.futures
import os
import time
from types import SimpleNamespace

import numpy as np
import pytest

import optiland.backend as be
import optiland.surfaces._chunked_trace as chunked
import optiland.surfaces._trace_plan as plan
from optiland.rays import RealRays
from optiland.samples.objectives import CookeTriplet

from .trace_plan_test_utils import (
    SYSTEMS,
    _asphere_triplet,
    _assert_same,
    _fan,
    _state,
    _strong_asphere,
)

# The limits that the small_parts fixture lowers, with their defaults.
DEFAULTS = {
    name: getattr(plan, name)
    for name in (
        "_MIN_THREADED_RAYS",
        "_MIN_PART_BLOCKS",
        "_MIN_THREADED_INTENSITY_RAYS",
    )
}
THREADS = 3
WAVELENGTHS = (0.48, 0.55, 0.65)


@pytest.fixture(autouse=True)
def small_parts(monkeypatch):
    """Trace each bundle on threads, in parts of one block of rays."""
    for name in DEFAULTS:
        monkeypatch.setattr(plan, name, 1)
    monkeypatch.setattr(chunked, "_MIN_RAYS", 10**12)
    previous = be.get_trace_threads()
    yield
    be.set_trace_threads(previous)


@pytest.fixture
def thread_counts(monkeypatch):
    """The threads of the kernel and of the intensity pass of each trace."""
    counts = {"kernel": [], "intensity": []}
    run_kernel, absorbed_intensity = plan._run_kernel, plan._absorbed_intensity

    def counted_kernel(*args, threads):
        counts["kernel"].append(threads)
        return run_kernel(*args, threads=threads)

    def counted_intensity(*args, threads):
        counts["intensity"].append(threads)
        return absorbed_intensity(*args, threads=threads)

    monkeypatch.setattr(plan, "_run_kernel", counted_kernel)
    monkeypatch.setattr(plan, "_absorbed_intensity", counted_intensity)
    return counts


def _traced(build, rays, record, threads):
    """Trace ``rays()`` through ``build()`` with SurfaceGroup.trace."""
    be.set_trace_threads(threads)
    optic = build()
    traced = rays()
    optic.surfaces.trace(traced, record=record)
    return _state(optic, traced)


def _hexapolar(build, record, threads):
    """optic.trace of a hexapolar bundle of 20 rings (1261 rays)."""
    be.set_trace_threads(threads)
    optic = build()
    rays = optic.trace(
        Hx=0.0,
        Hy=0.7,
        wavelength=optic.primary_wavelength,
        num_rays=20,
        distribution="hexapolar",
        record=record,
    )
    return _state(optic, rays)


def test_one_thread_by_default(thread_counts):
    assert be.get_trace_threads() == 1
    CookeTriplet().surfaces.trace(_fan(5000), record=True)
    assert thread_counts["kernel"] == [1]
    assert thread_counts["intensity"] == [1]


def test_set_and_get_the_number_of_threads():
    be.set_trace_threads(2)
    assert be.get_trace_threads() == 2
    be.set_trace_threads(np.int64(4))
    assert be.get_trace_threads() == 4
    assert type(be.get_trace_threads()) is int
    be.set_trace_threads(1)
    assert be.get_trace_threads() == 1


def test_minus_one_sets_the_number_of_logical_cpus():
    """os.process_cpu_count on Python 3.13 and later, os.cpu_count before."""
    count_cpus = getattr(os, "process_cpu_count", os.cpu_count)
    be.set_trace_threads(-1)
    assert be.get_trace_threads() == (count_cpus() or 1)


def test_minus_one_sets_one_thread_if_the_count_is_unknown(monkeypatch):
    name = "process_cpu_count" if hasattr(os, "process_cpu_count") else "cpu_count"
    monkeypatch.setattr(os, name, lambda: None)
    be.set_trace_threads(-1)
    assert be.get_trace_threads() == 1


@pytest.mark.parametrize(
    "n", [1.5, 2.0, True, False, "2", None, np.float64(2.0), np.bool_(True)]
)
def test_a_number_of_another_type_raises_type_error(n):
    be.set_trace_threads(2)
    with pytest.raises(TypeError, match="must be an integer"):
        be.set_trace_threads(n)
    assert be.get_trace_threads() == 2


@pytest.mark.parametrize("n", [0, -2, -100])
def test_zero_or_a_number_below_minus_one_raises_value_error(n):
    be.set_trace_threads(2)
    with pytest.raises(ValueError, match="-1 or 1 or more"):
        be.set_trace_threads(n)
    assert be.get_trace_threads() == 2


def test_the_kernel_releases_the_gil_and_uses_no_numba_threads():
    """The private pool runs the kernels; Numba's threading layer does not.

    With nogil=True, a kernel releases the GIL, so the threads of the pool
    trace their parts at the same time. Without parallel=True, a kernel
    does not use the threading layer of Numba, so numba.set_num_threads and
    NUMBA_NUM_THREADS do not change it.
    """
    kernel = plan._kernel()
    for dispatcher in (kernel.trace_plan, kernel.trace_basic_plan):
        options = dispatcher.targetoptions
        assert options["nogil"] is True
        assert not options.get("parallel", False)


def test_the_setting_is_independent_of_torch_threads():
    torch = pytest.importorskip("torch")
    torch_threads = torch.get_num_threads()
    be.set_trace_threads(THREADS)
    assert torch.get_num_threads() == torch_threads
    torch.set_num_threads(1)
    try:
        assert be.get_trace_threads() == THREADS
    finally:
        torch.set_num_threads(torch_threads)


@pytest.mark.filterwarnings("ignore:paraxial ray tracing")
@pytest.mark.parametrize("record", [True, False])
@pytest.mark.parametrize("name", sorted(SYSTEMS))
def test_threaded_trace_is_identical(thread_counts, name, record):
    """optic.trace of each system of the plan tests, on 1 and 3 threads."""
    single = _hexapolar(SYSTEMS[name], record, 1)
    multi = _hexapolar(SYSTEMS[name], record, THREADS)
    assert THREADS in thread_counts["kernel"]
    _assert_same(single, multi)


@pytest.mark.parametrize("name", ["immersed_image", "thick_image"])
def test_final_step_in_parts_on_threads_is_identical(thread_counts, monkeypatch, name):
    """Each part of a thread holds three parts of the final step.

    The final step propagates the rays past the image surface. With parts of
    three blocks on the threads and of one block for the final step, each
    thread traces and propagates its part in three kernel calls. The result
    is that of one thread and that of the generic trace.
    """
    monkeypatch.setattr(plan, "_MIN_PART_BLOCKS", 3)
    monkeypatch.setattr(plan, "_FINAL_PART", 256)
    kernel = plan._kernel()
    ranges = []
    propagate = kernel._propagate_past_last_surface

    def spy(first, last, *args):
        ranges.append((first, last))
        return propagate(first, last, *args)

    monkeypatch.setattr(kernel, "_propagate_past_last_surface", spy)
    single = _hexapolar(SYSTEMS[name], record=True, threads=1)
    multi = _hexapolar(SYSTEMS[name], record=True, threads=THREADS)
    assert thread_counts["kernel"] == [1, THREADS]
    n = single["ray.x"].shape[0]
    parts = [(first, min(first + 256, n)) for first in range(0, n, 256)]
    # Two traces: each range of the final step once per trace.
    assert sorted(ranges) == sorted(parts * 2)
    _assert_same(single, multi)
    monkeypatch.setattr(plan, "_ENABLED", False)
    _assert_same(multi, _hexapolar(SYSTEMS[name], record=True, threads=1))


@pytest.mark.parametrize("record", [True, False])
@pytest.mark.parametrize("n", [1, 255, 256, 257, 1000, 2049])
def test_ray_counts_and_wavelengths_are_identical(thread_counts, n, record):
    """Whole and partial blocks, and one refractive index per ray.

    The fan has three wavelengths, so each glass has one index per ray. The
    Cooke triplet has a glass with k > 0, so the intensity pass also runs
    on threads.
    """

    def rays():
        return _fan(n, WAVELENGTHS)

    single = _traced(CookeTriplet, rays, record, 1)
    multi = _traced(CookeTriplet, rays, record, THREADS)
    assert thread_counts["kernel"] == [1, THREADS]
    assert thread_counts["intensity"] == [1, THREADS]
    _assert_same(single, multi)


class _Filled:
    """NumPy, but ``empty`` returns an array filled with ``value``."""

    def __init__(self, value):
        self.value = value

    def __getattr__(self, name):
        return getattr(np, name)

    def empty(self, shape, dtype=float):
        out = np.empty(shape, dtype=dtype)
        out.fill(self.value if out.dtype != np.bool_ else bool(self.value))
        return out


def test_the_parts_write_each_ray(monkeypatch, thread_counts):
    """Outputs that start with other values end with the same values.

    So the parts of the kernel and of the intensity pass write each ray.
    """
    results = []
    for value in (7.0, 0.0):
        monkeypatch.setattr(plan, "np", _Filled(value))
        rays = _traced(CookeTriplet, lambda: _fan(1000, WAVELENGTHS), True, 4)
        results.append(rays)
    assert thread_counts["kernel"] == [4, 4]
    assert thread_counts["intensity"] == [4, 4]
    _assert_same(*results)


def _steep_rays():
    """700 rays from z = -3 with angles up to 0.9 rad, for _strong_asphere."""
    n = 700
    angle = np.linspace(-0.9, 0.9, n)
    return RealRays(
        np.zeros(n),
        np.linspace(-5, 5, n),
        np.full(n, -3.0),
        np.zeros(n),
        np.sin(angle),
        np.cos(angle),
        np.ones(n),
        np.full(n, 0.55),
    )


@pytest.mark.parametrize(
    ("coefficients", "max_iter"),
    [
        pytest.param((), 100, id="no_coefficients"),
        pytest.param((-1e-2, 2e-5), 100, id="converged"),
        pytest.param((-1e-2, 2e-5), 2, id="two_steps"),
    ],
)
def test_newton_is_identical(thread_counts, coefficients, max_iter):
    """The Newton-Raphson iteration of an even asphere, on several threads.

    The rays of a block take different numbers of steps, and some steep rays
    start on the vertex plane. Each part starts at a block boundary, so it
    iterates the blocks of a trace on one thread.
    """

    def build():
        return _strong_asphere(coefficients, max_iter)

    single = _traced(build, _steep_rays, True, 1)
    multi = _traced(build, _steep_rays, True, THREADS)
    assert thread_counts["kernel"] == [1, THREADS]
    _assert_same(single, multi)


def test_seeds_are_the_maxima_over_all_parts(monkeypatch):
    """A NaN seed in the third part of four reaches the floor check.

    Each part has its own seed summary, and the summary of the trace is the
    maximum over the parts. It must be the summary of one thread.
    """
    seen = []
    floor_applies = plan._floor_applies

    def spy(trace_plan, seed_stats):
        seen.append(np.array(seed_stats))
        return floor_applies(trace_plan, seed_stats)

    monkeypatch.setattr(plan, "_floor_applies", spy)

    def rays():
        out = _fan(1000)
        out.y = np.linspace(0.0, 2.0, 1000)
        out.z[600] = np.nan
        return out

    single = _traced(_asphere_triplet, rays, True, 1)
    multi = _traced(_asphere_triplet, rays, True, THREADS)
    _assert_same(single, multi)
    assert len(seen) == 2
    assert np.array_equal(seen[0], seen[1])
    # Ray 600 has a NaN seed at the two even aspheres, surfaces 1 and 4.
    assert np.flatnonzero(seen[0][:, 0]).tolist() == [1, 4]


def test_tolerance_floor_uses_the_generic_trace(monkeypatch):
    """With a tolerance below the round-off floor, a threaded trace falls back.

    The floor depends on the seeds of all rays, so _floor_applies reads the
    maxima over the parts. It gives True, trace_with_plan returns False, and
    the generic trace traces the rays.
    """
    calls = []
    original = plan.trace_with_plan

    def counted(*args, **kwargs):
        calls.append(original(*args, **kwargs))
        return calls[-1]

    monkeypatch.setattr(plan, "trace_with_plan", counted)

    def build():
        optic = _asphere_triplet()
        optic.surfaces.surfaces[4].geometry.tol = 1e-30
        return optic

    multi = _traced(build, lambda: _fan(1000), True, THREADS)
    assert calls == [False]
    monkeypatch.setattr(plan, "_ENABLED", False)
    generic = _traced(build, lambda: _fan(1000), True, 1)
    _assert_same(multi, generic)


def test_small_bundles_use_one_thread(monkeypatch, thread_counts):
    """The default limits: 8192 rays for the kernel, 65536 for the intensity pass.

    The small_parts fixture lowers the limits for the other tests, so this
    test sets the defaults again. The Cooke triplet has a glass with k > 0,
    so each trace runs the intensity pass, which uses at most 4 threads.
    """
    for name, value in DEFAULTS.items():
        monkeypatch.setattr(plan, name, value)
    be.set_trace_threads(8)
    surfaces = CookeTriplet().surfaces
    kernel_limit = DEFAULTS["_MIN_THREADED_RAYS"]
    for n in (kernel_limit - 1, kernel_limit, DEFAULTS["_MIN_THREADED_INTENSITY_RAYS"]):
        surfaces.trace(_fan(n), record=False)
    assert thread_counts["kernel"] == [1, 8, 8]
    assert thread_counts["intensity"] == [1, 1, plan._MAX_INTENSITY_THREADS]


def test_traces_in_several_python_threads(thread_counts):
    """Four Python threads trace at the same time and share the pool.

    Each trace gets the bits of a trace on one thread.
    """

    def rays():
        return _fan(3000, WAVELENGTHS)

    reference = _traced(CookeTriplet, rays, True, 1)
    be.set_trace_threads(THREADS)
    thread_counts["kernel"].clear()

    def trace(_):
        optic = CookeTriplet()
        traced = rays()
        optic.surfaces.trace(traced, record=True)
        return _state(optic, traced)

    with concurrent.futures.ThreadPoolExecutor(4) as callers:
        results = list(callers.map(trace, range(4)))
    assert thread_counts["kernel"] == [THREADS] * 4
    for result in results:
        _assert_same(reference, result)


def test_run_parts_runs_each_part_once():
    be.set_trace_threads(4)
    done = []
    plan._run_parts(50, 4, done.append)
    assert sorted(done) == list(range(50))


def test_run_parts_raises_after_all_parts_have_stopped():
    """The exception of a part comes after the other parts have ended.

    Each other part takes 10 ms, so a missing wait would raise before they
    end. The other threads trace all other parts.
    """
    be.set_trace_threads(4)
    done = []

    def run(k):
        if k == 5:
            raise RuntimeError("part 5")
        time.sleep(0.01)
        done.append(k)

    with pytest.raises(RuntimeError, match="part 5"):
        plan._run_parts(12, 4, run)
    assert sorted(done) == [k for k in range(12) if k != 5]
    # The pool still works.
    plan._run_parts(12, 4, lambda k: None)


def test_no_pool_for_one_thread_or_one_part(monkeypatch):
    """With one thread or one part, the calling thread runs all parts."""
    made = []
    monkeypatch.setattr(plan, "_executor", lambda: made.append(True))
    done = []
    be.set_trace_threads(4)
    plan._run_parts(1, 4, done.append)
    plan._run_parts(3, 1, done.append)
    be.set_trace_threads(1)
    plan._run_parts(3, 4, done.append)
    assert made == []
    assert done == [0, 0, 1, 2, 0, 1, 2]


def test_a_forked_process_makes_a_new_pool(monkeypatch):
    """After a fork, the threads of the parent's pool do not exist.

    The test cannot fork on Windows, so it changes the process id that the
    pool sees.
    """
    monkeypatch.setattr(plan, "_pool", None)
    be.set_trace_threads(3)
    parent = plan._executor()
    assert plan._executor() is parent
    monkeypatch.setattr(plan, "os", SimpleNamespace(getpid=lambda: -1))
    child = plan._executor()
    assert child is not parent
    assert plan._executor() is child
    parent.shutdown()
    child.shutdown()


def test_a_new_number_of_threads_makes_a_new_pool(monkeypatch):
    """The old pool finishes its queued parts and then stops."""
    monkeypatch.setattr(plan, "_pool", None)
    be.set_trace_threads(3)
    old = plan._executor()
    be.set_trace_threads(5)
    new = plan._executor()
    assert new is not old
    assert plan._pool.workers == 4
    with pytest.raises(RuntimeError, match="shutdown"):
        old.submit(int)
    new.shutdown()


def test_torch_backend_ignores_the_threads(thread_counts):
    """The Torch backend uses the generic trace for any number of threads."""
    pytest.importorskip("torch")
    be.set_trace_threads(THREADS)
    be.set_backend("torch")
    try:
        CookeTriplet().surfaces.trace(_fan(1000), record=True)
    finally:
        be.set_backend("numpy")
    assert thread_counts["kernel"] == []


def test_rays_that_miss_are_identical(thread_counts):
    """Rays far outside the lens give NaN at the same positions."""

    def rays():
        return _fan(1000, height=40.0, angle=0.3)

    single = _traced(CookeTriplet, rays, True, 1)
    multi = _traced(CookeTriplet, rays, True, THREADS)
    assert np.isnan(multi["ray.x"]).any()
    _assert_same(single, multi)
