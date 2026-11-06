"""Shared paraxial path in a ray-generation scope (issue #934).

A trace builds the paraxial path once instead of once for each paraxial
call of the ray aimer. The shared path must never outlive its scope, so a
geometry change after a trace must always give fresh first-order values.
"""

from __future__ import annotations

import threading

import pytest

import optiland.backend as be
import optiland.surfaces.surface_group as surface_group_module
from optiland.paraxial_path import shared_path_scope
from optiland.samples.objectives import CookeTriplet

from .utils import assert_allclose


@pytest.fixture
def count_builds(monkeypatch):
    """Count the calls of the path builder that SurfaceGroup uses."""
    calls = []
    build = surface_group_module.build_paraxial_path

    def counting_build(surfaces):
        calls.append(1)
        return build(surfaces)

    monkeypatch.setattr(surface_group_module, "build_paraxial_path", counting_build)
    return calls


@pytest.mark.parametrize("aiming", ["paraxial", "iterative", "robust"])
def test_trace_builds_the_path_once(set_test_backend, count_builds, aiming):
    optic = CookeTriplet()
    optic.ray_tracer.set_aiming(aiming)
    wavelength = optic.primary_wavelength
    optic.trace(0.0, 1.0, wavelength, 6, "hexapolar")

    count_builds.clear()
    optic.trace(0.0, 1.0, wavelength, 6, "hexapolar")
    assert len(count_builds) == 1

    count_builds.clear()
    optic.trace_generic(0.0, 1.0, 0.0, 1.0, wavelength)
    assert len(count_builds) == 1


def test_scope_gives_one_path_and_ends_with_the_scope(set_test_backend):
    surfaces = CookeTriplet().surfaces
    with shared_path_scope(surfaces):
        first = surfaces.build_paraxial_path()
        assert surfaces.build_paraxial_path() is first
        with shared_path_scope(surfaces):
            assert surfaces.build_paraxial_path() is first
        assert surfaces.build_paraxial_path() is first
    assert surfaces.build_paraxial_path() is not first
    assert surfaces.build_paraxial_path() is not surfaces.build_paraxial_path()


def test_scope_is_per_surface_group(set_test_backend):
    a = CookeTriplet().surfaces
    b = CookeTriplet().surfaces
    with shared_path_scope(a):
        assert a.build_paraxial_path() is a.build_paraxial_path()
        assert b.build_paraxial_path() is not b.build_paraxial_path()


def test_scope_has_no_effect_in_other_threads(set_test_backend):
    surfaces = CookeTriplet().surfaces
    same_in_thread = []

    def worker():
        same_in_thread.append(
            surfaces.build_paraxial_path() is surfaces.build_paraxial_path()
        )

    with shared_path_scope(surfaces):
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
    assert same_in_thread == [False]


def test_geometry_change_after_a_trace_gives_fresh_values(set_test_backend):
    optic = CookeTriplet()
    wavelength = optic.primary_wavelength
    optic.trace(0.0, 1.0, wavelength, 6, "hexapolar")
    f2_before = optic.paraxial.f2()
    epl_before = optic.paraxial.EPL()

    optic.surfaces[1].geometry.cs.z = optic.surfaces[1].geometry.cs.z - 1.0
    reference = CookeTriplet()
    reference.surfaces[1].geometry.cs.z = reference.surfaces[1].geometry.cs.z - 1.0

    optic.trace(0.0, 1.0, wavelength, 6, "hexapolar")
    assert_allclose(optic.paraxial.EPL(), reference.paraxial.EPL())
    assert_allclose(optic.paraxial.f2(), reference.paraxial.f2())
    assert not be.allclose(
        be.as_array_1d(optic.paraxial.EPL()), be.as_array_1d(epl_before)
    )
    assert not be.allclose(
        be.as_array_1d(optic.paraxial.f2()), be.as_array_1d(f2_before)
    )
