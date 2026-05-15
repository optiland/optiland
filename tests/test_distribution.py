from __future__ import annotations

import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import optiland.backend as be
from optiland import distribution

from .utils import assert_allclose


@pytest.mark.parametrize("num_points", [10, 25, 106, 512])
def test_line_x(set_test_backend, num_points):
    d = distribution.create_distribution("line_x")
    d.generate_points(num_points=num_points)

    assert_allclose(d.x, be.linspace(-1, 1, num_points))
    assert_allclose(d.y, be.zeros(num_points))

    d = distribution.create_distribution("positive_line_x")
    d.generate_points(num_points=num_points)

    assert_allclose(d.x, be.linspace(0, 1, num_points))
    assert_allclose(d.y, be.zeros(num_points))


@pytest.mark.parametrize("num_points", [9, 60, 111, 509])
def test_line_y(set_test_backend, num_points):
    d = distribution.create_distribution("line_y")
    d.generate_points(num_points=num_points)

    assert_allclose(d.x, be.zeros(num_points))
    assert_allclose(d.y, be.linspace(-1, 1, num_points))

    d = distribution.create_distribution("positive_line_y")
    d.generate_points(num_points=num_points)

    assert_allclose(d.x, be.zeros(num_points))
    assert_allclose(d.y, be.linspace(0, 1, num_points))


@pytest.mark.parametrize("num_points", [8, 26, 154, 689])
def test_random(set_test_backend, num_points):
    seed = 42
    d = distribution.RandomDistribution(seed=seed)
    d.generate_points(num_points=num_points)

    rng = be.default_rng(seed=seed)
    r = be.random_uniform(size=num_points, generator=rng)
    theta = be.random_uniform(0, 2 * be.pi, size=num_points, generator=rng)

    x = be.sqrt(r) * be.cos(theta)
    y = be.sqrt(r) * be.sin(theta)

    assert_allclose(d.x, x)
    assert_allclose(d.y, y)


@pytest.mark.parametrize("num_rings", [3, 7, 15, 220])
def test_hexapolar(set_test_backend, num_rings):
    d = distribution.create_distribution("hexapolar")
    d.generate_points(num_rings=num_rings)

    x = be.zeros(1)
    y = be.zeros(1)
    r = be.linspace(0, 1, num_rings + 1)

    for i in range(num_rings):
        num_theta = 6 * (i + 1)
        theta = be.linspace(0, 2 * be.pi, num_theta + 1)[:-1]
        x = be.concatenate([x, r[i + 1] * be.cos(theta)])
        y = be.concatenate([y, r[i + 1] * be.sin(theta)])

    assert_allclose(d.x, x)
    assert_allclose(d.y, y)


@pytest.mark.parametrize("num_points", [15, 56, 161, 621])
def test_cross(set_test_backend, num_points):
    d = distribution.create_distribution("cross")
    d.generate_points(num_points=num_points)

    # Expected points construction based on the new logic in CrossDistribution
    y_line_x_expected = be.zeros(num_points)
    y_line_y_expected = be.linspace(-1, 1, num_points)

    x_line_x_expected_full = be.linspace(-1, 1, num_points)
    x_line_y_expected_full = be.zeros(num_points)

    if num_points % 2 == 1:  # Odd number of points
        # Remove the middle element from the x-axis line as it's the duplicated origin
        mid_idx = num_points // 2
        x_line_x_to_concat = be.concatenate(
            (x_line_x_expected_full[:mid_idx], x_line_x_expected_full[mid_idx + 1 :])
        )
        x_line_y_to_concat = be.concatenate(
            (x_line_y_expected_full[:mid_idx], x_line_y_expected_full[mid_idx + 1 :])
        )
    else:  # Even number of points (origin is not in the middle of linspace for an odd-length array)
        x_line_x_to_concat = x_line_x_expected_full
        x_line_y_to_concat = x_line_y_expected_full

    # Concatenate in the same order as in the implementation
    expected_x = be.concatenate((y_line_x_expected, x_line_x_to_concat))
    expected_y = be.concatenate((y_line_y_expected, x_line_y_to_concat))

    assert_allclose(d.x, expected_x)
    assert_allclose(d.y, expected_y)


def test_view_distribution(set_test_backend):
    d = distribution.create_distribution("random")
    d.generate_points(num_points=10)
    fig, ax = d.view()
    assert fig is not None
    assert ax is not None
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_invalid_distribution_error(set_test_backend):
    with pytest.raises(ValueError):
        distribution.create_distribution(distribution_type="invalid")


@pytest.mark.parametrize("num_points", [10, 25, 50, 100])
def test_uniform_distribution(set_test_backend, num_points):
    d = distribution.create_distribution("uniform")
    d.generate_points(num_points=num_points)

    x = be.linspace(-1, 1, num_points)
    x, y = be.meshgrid(x, x)
    r2 = x**2 + y**2
    x = x[r2 <= 1]
    y = y[r2 <= 1]

    assert be.all(d.x >= -1.0) and be.all(d.x <= 1.0)
    assert be.all(d.y >= -1.0) and be.all(d.y <= 1.0)
    assert_allclose(d.x, x)
    assert_allclose(d.y, y)


@pytest.mark.parametrize("num_rings", range(1, 7))
def test_gaussian_quad_distribution(num_rings, set_test_backend):
    # From: G. W. Forbes, "Optical system assessment for design: numerical ray tracing
    # in the Gaussian pupil," J. Opt. Soc. Am. A 5, 1943-1956 (1988).
    radius_dict = {
        1: be.array([0.70711]),
        2: be.array([0.45970, 0.88807]),
        3: be.array([0.33571, 0.70711, 0.94196]),
        4: be.array([0.26350, 0.57446, 0.81853, 0.96466]),
        5: be.array([0.21659, 0.48038, 0.70711, 0.87706, 0.97626]),
        6: be.array([0.18375, 0.41158, 0.61700, 0.78696, 0.91138, 0.98300]),
    }
    d = distribution.GaussianQuadrature()
    # Only check radial distribution, set num_spokes to 1:
    d.generate_points(num_rings=num_rings, num_spokes=1)
    r = be.hypot(d.x, d.y)
    assert_allclose(r, radius_dict[num_rings], atol=1e-4)
    assert_allclose(d.weights.sum(), 1.0)


def test_gaussian_quad_distribution_errors(set_test_backend):
    with pytest.raises(ValueError):
        d = distribution.GaussianQuadrature()
        d.generate_points(num_rings=0)


def test_quadrature_r_squared(set_test_backend):
    d = distribution.GaussianQuadrature()
    d.generate_points(2)
    assert_allclose(((d.x**2 + d.y**2) * d.weights).sum(), 0.5)


def test_quadrature_zernike(set_test_backend):
    def spherical(x, y):
        return 5**0.5 * (6 * (x**2 + y**2) ** 2 - 6 * (x**2 + y**2) + 1)

    d = distribution.GaussianQuadrature()
    d.generate_points(4)
    assert_allclose((spherical(d.x, d.y) * d.weights).sum(), 0.0)


def test_quadrature_exp(set_test_backend):
    d = distribution.GaussianQuadrature()
    d.generate_points(7)
    assert_allclose(
        ((be.exp(-(d.x**2) - d.y**2)) * d.weights).sum(), (be.exp(1) - 1) / be.exp(1)
    )


@pytest.mark.parametrize("num_points", [16, 64, 256, 1024])
def test_sobol_distribution(set_test_backend, num_points):
    seed = 42
    d = distribution.SobolDistribution(seed=seed)
    d.generate_points(num_points=num_points)

    sampler = be.sobol_sampler(dim=2, num_samples=num_points, scramble=True, seed=seed)
    u1 = sampler[:, 0]
    u2 = sampler[:, 1]
    r = be.sqrt(u1)
    theta = 2 * be.pi * u2
    x = r * be.cos(theta)
    y = r * be.sin(theta)

    assert_allclose(d.x, x)
    assert_allclose(d.y, y)
    assert be.all(d.x**2 + d.y**2 <= 1.0 + 1e-7)
