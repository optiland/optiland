from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import Plane
from optiland.materials import IdealMaterial
from optiland.rays import ParaxialRays, RealRays
from optiland.surfaces.object_surface import ObjectSurface


@pytest.fixture
def setup_object_surface():
    cs = CoordinateSystem()
    geometry = Plane(cs)
    material = IdealMaterial(1, 0)
    object_surface = ObjectSurface(geometry=geometry, material_post=material)
    return object_surface, geometry


def test_is_infinite(set_test_backend, setup_object_surface):
    object_surface, geometry = setup_object_surface

    geometry.cs.z = -be.array(be.inf)
    assert object_surface.is_infinite

    geometry.cs.z = be.array(-100)
    assert not object_surface.is_infinite


def test_set_aperture(set_test_backend, setup_object_surface):
    object_surface, _ = setup_object_surface
    object_surface.set_aperture()


def test_trace(set_test_backend, setup_object_surface):
    object_surface, _ = setup_object_surface
    x = np.random.rand(10)
    rays = RealRays(x, x, x, x, x, x, x, x)
    traced_rays = object_surface.trace(rays)
    assert traced_rays == rays


def test_trace_paraxial(set_test_backend, setup_object_surface):
    object_surface, _ = setup_object_surface
    y = be.array([1])
    u = be.array([0])
    z = be.array([-10])
    w = be.array([1])
    rays = ParaxialRays(y, u, z, w)
    object_surface._trace_paraxial(rays)


def test_trace_real(set_test_backend, setup_object_surface):
    object_surface, _ = setup_object_surface
    x = be.random_uniform(size=10)
    rays = RealRays(x, x, x, x, x, x, x, x)
    object_surface._trace_real(rays)


def test_interact(set_test_backend, setup_object_surface):
    object_surface, _ = setup_object_surface
    x = be.random_uniform(size=10)
    rays = RealRays(x, x, x, x, x, x, x, x)
    interacted_rays = object_surface._interact(rays)
    assert interacted_rays == rays


def test_material_pre_is_material_post(setup_object_surface):
    object_surface, _ = setup_object_surface
    assert object_surface.material_pre == object_surface.material_post
