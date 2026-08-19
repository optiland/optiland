from __future__ import annotations

import pytest

from optiland import backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.nurbs.nurbs_geometry import NurbsGeometry
from optiland.materials.ideal import IdealMaterial
from optiland.optic.optic import Optic
from optiland.optimization.variable.nurbs import (
    NurbsPointsVariable,
    NurbsWeightsVariable,
)
from optiland.surfaces.standard_surface import Surface
from optiland.surfaces.surface_group import SurfaceGroup
from tests.utils import assert_allclose


@pytest.fixture
def optic(backend):
    be.set_backend(backend)
    cs = CoordinateSystem()
    geo = NurbsGeometry(cs, nurbs_norm_x=1, nurbs_norm_y=1)
    geo.fit_surface()
    air = IdealMaterial(n=1.0)
    surf = Surface(previous_surface=None, geometry=geo, material_post=air)
    sg = SurfaceGroup([surf])
    optic = Optic()
    optic.surfaces = sg
    return optic


@pytest.mark.parametrize("backend", be.list_available_backends())
def test_nurbs_points_variable(optic, backend):
    var = NurbsPointsVariable(optic, 0, (0, 0, 0), apply_scaling=False)
    val = var.get_value()
    assert_allclose(val, be.asarray(-1.0, dtype=val.dtype), atol=1e-6)
    var.update_value(1.0)
    val = var.get_value()
    assert_allclose(val, be.asarray(1.0, dtype=val.dtype), atol=1e-6)


@pytest.mark.parametrize("backend", be.list_available_backends())
def test_nurbs_weights_variable(optic, backend):
    var = NurbsWeightsVariable(optic, 0, (0, 0), apply_scaling=False)
    val = var.get_value()
    assert_allclose(val, be.asarray(1.0, dtype=val.dtype), atol=1e-6)
    var.update_value(2.0)
    val = var.get_value()
    assert_allclose(val, be.asarray(2.0, dtype=val.dtype), atol=1e-6)
