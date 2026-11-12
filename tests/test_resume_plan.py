"""Tests for the compiled trace plan in resumed partial traces.

Partial-trace finite differences take a snapshot of the ray state before
each surface (``RealRayTracer._trace``) and resume it at a surface k
(``_trace_from``). With NumPy, the resume traces the surfaces from k with
the compiled trace plan, and propagates the rays past the last surface in
the final step of the plan. The plan exists only on the NumPy backend, so
these tests run on NumPy; ``test_torch_backend_resumes_with_the_loop``
checks Torch. A fast path must give the bits of the generic code, so each
test resumes once with the plan and once with its private switch
``_ENABLED`` off, and compares the rays and the records of all surfaces bit
for bit. Compiled code can give a NaN another sign bit, so the sign of NaN
is not compared (see the developer page on performance internals).

Manuel Fragata Mendes, 2026
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
import optiland.surfaces._trace_plan as plan
from optiland.physical_apertures import RectangularAperture
from optiland.samples.objectives import CookeTriplet

from .trace_plan_test_utils import SYSTEMS, _asphere_triplet, _assert_same, _state
from .utils import assert_bits_equal

# Spheres, an even asphere, a conic mirror and apertures. The image surfaces
# of immersed_image and thick_image have a thickness that is not 0, so the
# final step of each resume moves the rays.
RESUMED = (
    "cooke",
    "double_gauss",
    "hubble",
    "aspheric_singlet",
    "clipped",
    "immersed_image",
    "thick_image",
)


@pytest.fixture
def calls(monkeypatch):
    """The results of the calls of trace_with_plan, in order.

    RealRayTracer looks up trace_with_plan in its module at each call, so
    the spy replaces the module attribute.
    """
    results = []
    original = plan.trace_with_plan

    def spy(*args, **kwargs):
        results.append(original(*args, **kwargs))
        return results[-1]

    monkeypatch.setattr(plan, "trace_with_plan", spy)
    return results


@pytest.fixture
def floor_checks(monkeypatch):
    """The results of the calls of _floor_applies, in order."""
    results = []
    original = plan._floor_applies

    def spy(*args):
        results.append(original(*args))
        return results[-1]

    monkeypatch.setattr(plan, "_floor_applies", spy)
    return results


def _grid():
    """A grid of 5 x 5 normalized pupil coordinates."""
    g = np.linspace(-0.7, 0.7, 5)
    px, py = np.meshgrid(g, g)
    return be.array(px.ravel()), be.array(py.ravel())


def _snapshot(optic):
    """Trace a field point with a snapshot of the rays before each surface.

    Returns:
        The traced rays and the snapshot.
    """
    px, py = _grid()
    return optic.ray_tracer._trace(0.0, 0.7, px, py, optic.primary_wavelength)


def _resume(build, k, record, monkeypatch, enabled):
    """Take a snapshot trace of ``build()``, then resume it at surface ``k``.

    Before the resume, each surface gets stale records. The resumed surfaces
    must reset them, and the surfaces before ``k`` must keep them.

    Returns:
        The ray arrays and the records of all surfaces after the resume.
    """
    optic = build()
    _, snapshot = _snapshot(optic)
    for j, surface in enumerate(optic.surfaces.surfaces):
        surface.u = np.array([float(j)])
        surface.aoi = np.array([-float(j)])
        if not record:
            surface.x = np.array([1.0 + j])
    monkeypatch.setattr(plan, "_ENABLED", enabled)
    rays = optic.ray_tracer._trace_from(k, snapshot, record)
    return _state(optic, rays)


@pytest.mark.parametrize("record", [True, False])
@pytest.mark.parametrize("name", RESUMED)
def test_a_resume_with_the_plan_is_identical(calls, monkeypatch, name, record):
    """A resume at each surface gives the rays and the records of the loop.

    The surfaces from k get the records of the resume, and the surfaces
    before k keep their stale records.
    """
    build = SYSTEMS[name]
    surface_count = build().surfaces.num_surfaces
    for k in range(surface_count):
        calls.clear()
        with_plan = _resume(build, k, record, monkeypatch, enabled=True)
        assert calls == [True], k
        _assert_same(with_plan, _resume(build, k, record, monkeypatch, enabled=False))
        kept = [with_plan[f"s{j}.u"].size == 1 for j in range(surface_count)]
        assert kept == [j < k for j in range(surface_count)], k


def test_a_resume_outside_the_domain_uses_the_loop(calls):
    """A rectangular aperture is outside the domain of the plan.

    The resume then runs the loop over the surfaces, which gives the rays of
    the snapshot trace.
    """
    optic = CookeTriplet()
    optic.surfaces.surfaces[2].aperture = RectangularAperture(-5, 5, -5, 5)
    traced, snapshot = _snapshot(optic)
    rays = optic.ray_tracer._trace_from(1, snapshot, True)
    assert calls == [False]
    for name in ("x", "y", "z", "L", "M", "N", "i", "opd"):
        assert_bits_equal(getattr(rays, name), getattr(traced, name))


def test_a_resume_after_the_tolerance_floor_uses_the_loop(
    calls, floor_checks, monkeypatch
):
    """The kernel runs, then the tolerance floor of an asphere gives False.

    The plan has released the records of the resumed surfaces, and the loop
    resets them.
    """

    def build():
        optic = _asphere_triplet()
        optic.surfaces.surfaces[4].geometry.tol = 1e-30
        return optic

    with_plan = _resume(build, 2, True, monkeypatch, enabled=True)
    assert calls == [False]
    assert floor_checks == [True]
    _assert_same(with_plan, _resume(build, 2, True, monkeypatch, enabled=False))


def test_torch_backend_resumes_with_the_loop(calls):
    """NumPy only: with Torch, trace_with_plan returns False."""
    pytest.importorskip("torch")
    be.set_backend("torch")
    try:
        optic = CookeTriplet()
        _, snapshot = _snapshot(optic)
        optic.ray_tracer._trace_from(2, snapshot, True)
    finally:
        be.set_backend("numpy")
    assert calls == [False]
