from __future__ import annotations

import pytest

from optiland import backend as be
from optiland.phase.grid import GridPhaseProfile

from .utils import assert_allclose


@pytest.fixture
def grid_data():
    x = be.linspace(-1, 1, 50)
    y = be.linspace(-2, 2, 50)
    phase_grid = be.array([[i**2 + j**3 for i in x] for j in y])
    return x, y, phase_grid


def test_grid_phase_profile_init(grid_data):
    x, y, phase_grid = grid_data
    profile = GridPhaseProfile(x, y, phase_grid)
    assert profile.x_coords is not None
    assert profile.y_coords is not None
    assert profile.phase_grid is not None


def test_grid_phase_profile_get_phase(grid_data):
    x, y, phase_grid = grid_data
    profile = GridPhaseProfile(x, y, phase_grid)

    phase = profile.get_phase(be.array([x[1]]), be.array([y[2]]))
    assert_allclose(phase, phase_grid[2, 1], atol=1e-6)

    phase_interp = profile.get_phase(be.array([0.5]), be.array([0.5]))
    assert isinstance(phase_interp.item(), float)


def test_grid_phase_profile_get_gradient(grid_data):
    x, y, phase_grid = grid_data
    profile = GridPhaseProfile(x, y, phase_grid)

    grad_x, grad_y, grad_z = profile.get_gradient(be.array([0.5]), be.array([1.0]))

    assert_allclose(grad_x, be.array([1.0]), atol=1e-2)
    assert_allclose(grad_y, be.array([3.0]), atol=1e-2)
    assert_allclose(grad_z, be.array([0.0]), atol=1e-6)


def test_grid_phase_profile_get_paraxial_gradient(grid_data):
    x, y, phase_grid = grid_data
    profile = GridPhaseProfile(x, y, phase_grid)

    y_vals = be.array([0.5, 1.0, 1.5])
    paraxial_grad = profile.get_paraxial_gradient(y_vals)

    expected_grad = 3 * y_vals**2
    assert_allclose(paraxial_grad, expected_grad, atol=1e-2)


def test_grid_phase_profile_to_from_dict(grid_data):
    x, y, phase_grid = grid_data
    profile = GridPhaseProfile(x, y, phase_grid)

    data = profile.to_dict()

    assert data["phase_type"] == "grid"
    assert "x_coords" in data
    assert "y_coords" in data
    assert "phase_grid" in data

    new_profile = GridPhaseProfile.from_dict(data)

    assert isinstance(new_profile, GridPhaseProfile)
    assert_allclose(new_profile.x_coords, x, atol=1e-6)
    assert_allclose(new_profile.y_coords, y, atol=1e-6)
    assert_allclose(new_profile.phase_grid, phase_grid, atol=1e-6)
