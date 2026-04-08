"""Tests for GridSagVariable"""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.grid_sag import GridSagGeometry
from optiland.materials.ideal import IdealMaterial
from optiland.optic import Optic
from optiland.optimization.variable.grid_sag import GridSagVariable
from optiland.surfaces.standard_surface import Surface


@pytest.fixture
def grid_sag_optic():
    """Creates an Optic with a GridSagGeometry surface."""
    optic = Optic()
    cs = CoordinateSystem()
    x = [-1, 0, 1]
    y = [-1, 0, 1]
    sag = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    grid_sag_geom = GridSagGeometry(cs, x, y, sag)
    surface = Surface(
        geometry=grid_sag_geom,
        previous_surface=None,
        material_post=IdealMaterial(1.5),
    )
    optic.surfaces.add(new_surface=surface, index=0)
    return optic


def test_grid_sag_variable_get_value(grid_sag_optic):
    """Test that GridSagVariable correctly gets the sag grid."""
    variable = GridSagVariable(grid_sag_optic, 0)
    expected_sag = be.asarray([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    assert be.allclose(variable.get_value(), expected_sag)


def test_grid_sag_variable_update_value(grid_sag_optic):
    """Test that GridSagVariable correctly updates the sag grid."""
    variable = GridSagVariable(grid_sag_optic, 0)
    new_sag = be.asarray([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    variable.update_value(new_sag)
    assert be.allclose(grid_sag_optic.surfaces[0].geometry.sag_grid, new_sag)


@pytest.mark.parametrize("backend", be.list_available_backends())
def test_grid_sag_variable_torch_backend(backend, grid_sag_optic):
    """Test GridSagVariable with the PyTorch backend."""
    be.set_backend(backend)
    if be.get_backend() == "torch":
        variable = GridSagVariable(grid_sag_optic, 0)
        original_sag = variable.get_value()

        # Simulate a simple optimization step
        sag_tensor = be.asarray(original_sag, dtype=be.float32)
        sag_tensor.requires_grad_(True)
        loss = be.sum(sag_tensor**2)
        loss.backward()

        assert sag_tensor.grad is not None
        assert be.allclose(
            sag_tensor.grad, 2 * be.asarray(original_sag, dtype=be.float32)
        )
