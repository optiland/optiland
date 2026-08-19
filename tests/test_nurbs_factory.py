from __future__ import annotations

import pytest

from optiland import backend as be
from optiland.geometries.nurbs.nurbs_geometry import NurbsGeometry
from optiland.materials.ideal import IdealMaterial
from optiland.surfaces.factories.surface_factory import SurfaceFactory
from tests.utils import assert_allclose


class MockSurface:
    def __init__(self):
        self.material_post = IdealMaterial(n=1.0)


class MockSurfaceGroup:
    def __init__(self, num_surfaces):
        self.num_surfaces = num_surfaces
        self.surfaces = [MockSurface()]


@pytest.mark.parametrize("backend", be.list_available_backends())
def test_nurbs_factory(set_test_backend, backend):
    be.set_backend(backend)
    config = {
        "surface_type": "nurbs",
        "radius": 100,
        "conic": -1,
        "comment": "NURBS test surface",
        "index": 1,
        "is_stop": False,
        "material": "air",
    }
    factory = SurfaceFactory(surface_group=MockSurfaceGroup(2))
    surf = factory.create_surface(**config)
    assert isinstance(surf.geometry, NurbsGeometry)
    assert_allclose(be.asarray(surf.geometry.radius), be.asarray(100))
    assert_allclose(be.asarray(surf.geometry.k), be.asarray(-1))
