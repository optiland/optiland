from __future__ import annotations

import pytest

from optiland import backend as be
from optiland.phase.interpolators import GridInterpolator

from .utils import assert_allclose


@pytest.fixture
def interpolator_data():
    x = be.linspace(-1, 1, 100)
    y = be.linspace(-2, 2, 100)
    grid = be.array([[i**2 + j**3 for i in x] for j in y])
    return x, y, grid


def test_interpolator_height_exact(interpolator_data):
    x, y, grid = interpolator_data
    interp = GridInterpolator(x, y, grid)

    h = interp.height(be.array([x[5]]), be.array([y[7]]))
    assert_allclose(h, be.array([grid[7, 5]]), atol=1e-6)


def test_interpolator_height_interpolated(interpolator_data):
    x, y, grid = interpolator_data
    interp = GridInterpolator(x, y, grid)

    h = interp.height(be.array([0.15]), be.array([-0.33]))
    assert isinstance(h.item(), float)


def test_interpolator_gradient_values(interpolator_data):
    x, y, grid = interpolator_data
    interp = GridInterpolator(x, y, grid)

    px = be.array([0.5])
    py = be.array([1.0])

    dh_dx, dh_dy = interp.gradient(px, py)

    eps = 1e-4

    h_x_plus = interp.height(px + eps, py)
    h_x_minus = interp.height(px - eps, py)
    num_dx = (h_x_plus - h_x_minus) / (2 * eps)

    h_y_plus = interp.height(px, py + eps)
    h_y_minus = interp.height(px, py - eps)
    num_dy = (h_y_plus - h_y_minus) / (2 * eps)

    assert_allclose(dh_dx, num_dx, atol=1e-2)
    assert_allclose(dh_dy, num_dy, atol=1e-2)


def test_interpolator_boundary_values(interpolator_data):
    x, y, grid = interpolator_data
    interp = GridInterpolator(x, y, grid)

    boundary_points = [
        (x[0], y[0]),
        (x[-1], y[-1]),
        (x[0], y[-1]),
        (x[-1], y[0]),
    ]

    for xx, yy in boundary_points:
        h = interp.height(be.array([xx]), be.array([yy]))
        assert isinstance(h.item(), float)


def test_interpolator_gradient_boundary(interpolator_data):
    x, y, grid = interpolator_data
    interp = GridInterpolator(x, y, grid)

    px = be.array([x[0]])
    py = be.array([y[0]])

    dh_dx, dh_dy = interp.gradient(px, py)

    assert dh_dx.shape == (1,)
    assert dh_dy.shape == (1,)


def test_torch_interpolator_gradient_keeps_graph(set_test_backend):
    if be.get_backend() != "torch":
        pytest.skip("Torch-specific autograd test")

    coef = be.array(1.0)
    coef.requires_grad_(True)

    x = be.linspace(-1.0, 1.0, 5)
    y = be.linspace(-1.0, 1.0, 5)
    grid = coef * (y[:, None] ** 2 + x[None, :] ** 2)

    interp = GridInterpolator(x, y, grid)

    px = be.array([0.25])
    py = be.array([0.25])

    dh_dx, dh_dy = interp.gradient(px, py)

    assert dh_dx.requires_grad
    assert dh_dy.requires_grad
