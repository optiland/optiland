"""Input and numerical equivalence contracts for conic execution paths."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import optiland.backend as be
from optiland.geometries.standard import _conic_intersection_distance


@pytest.fixture(autouse=True)
def numpy_backend():
    be.set_backend("numpy")
    be.set_precision("float64")
    yield
    be.set_backend("numpy")
    be.set_precision("float64")


def _rays(values):
    return SimpleNamespace(
        **dict(zip(("x", "y", "z", "L", "M", "N"), values, strict=True))
    )


@pytest.mark.parametrize("size", [0, 1, 17, 8193])
@pytest.mark.parametrize("strided", [False, True])
def test_numpy_compiled_and_broadcast_paths_agree(size, strided):
    y = np.linspace(-2, 2, size * 2 if strided else size)
    if strided:
        y = y[::2]
    rays = _rays((y * 0, y, y * 0 - 3, y * 0, y * 0, y * 0 + 1))
    rays.y = y  # Preserve the non-contiguous input.
    actual = _conic_intersection_distance(rays, np.array(12.0), np.array(0.5))
    # A broadcast dimension takes the general array path.
    expected = _conic_intersection_distance(
        rays, np.array([[12.0]]), np.array([[0.5]])
    )
    assert actual.shape == (size,)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, expected.reshape(-1))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_scalar_directions_and_array_parameters_broadcast(dtype):
    x = np.array([[0], [0.5]], dtype=dtype)
    zero = np.array(0, dtype=dtype)
    rays = _rays((x, zero, zero - 1, zero, zero, zero + 1))
    radius = np.array([[2, 4, 8]], dtype=dtype)
    actual = _conic_intersection_distance(rays, radius, zero)
    expected = radius - np.sqrt(radius * radius - x * x) + 1
    assert actual.shape == (2, 3)
    assert actual.dtype == dtype
    np.testing.assert_allclose(actual, expected, rtol=2e-7)


def test_mixed_parameter_precision_matches_general_array_evaluation():
    rays = _rays(tuple(np.array([v], dtype=np.float64) for v in (1, 2, -3, 0, 0, 1)))
    radius = np.array(12, dtype=np.float32)
    conic = np.array(0.3, dtype=np.float32)
    actual = _conic_intersection_distance(rays, radius, conic)
    expected = _conic_intersection_distance(
        rays, radius.reshape(1, 1), conic.reshape(1, 1)
    )
    np.testing.assert_array_equal(actual, expected.reshape(-1))


def test_nonfinite_lanes_are_misses_without_mutating_inputs():
    x = np.array([0.0, np.nan, np.inf])
    rays = _rays(
        (x, np.zeros(3), np.full(3, -1.0), np.zeros(3), np.zeros(3), np.ones(3))
    )
    copies = {key: value.copy() for key, value in vars(rays).items()}
    with np.errstate(invalid="ignore"):
        actual = _conic_intersection_distance(rays, np.array(1.0), np.array(0.0))
    assert actual[0] == 1.0
    assert np.isnan(actual[1:]).all()
    for key, value in copies.items():
        np.testing.assert_array_equal(getattr(rays, key), value)


def test_masked_arrays_keep_the_general_numpy_execution_contract():
    y = np.ma.array([0.1, 0.2], mask=[False, True])
    rays = _rays((y * 0, y, y * 0 - 1, y * 0, y * 0, y * 0 + 1))
    actual = _conic_intersection_distance(rays, np.array(2.0), np.array(0.0))
    expected = _conic_intersection_distance(
        rays, np.array([[2.0]]), np.array([[0.0]])
    )
    np.testing.assert_array_equal(actual, expected.reshape(-1))


@pytest.mark.parametrize("radius", [-12.0, 12.0])
@pytest.mark.parametrize("conic", [-2.0, -1.0, 0.0, 0.5])
def test_random_conics_match_independent_polynomial_roots(radius, conic):
    rng = np.random.default_rng(753)
    positions = rng.uniform(-30, 30, (3, 128))
    directions = rng.normal(size=(3, 128))
    directions /= np.linalg.norm(directions, axis=0)
    actual = _conic_intersection_distance(
        _rays((*positions, *directions)), np.array(radius), np.array(conic)
    )
    expected = []
    for position, direction in zip(positions.T, directions.T, strict=True):
        # Construct the polynomial independently from a diagonal quadric form.
        quadric = np.diag([1, 1, 1 + conic])
        a = direction @ quadric @ direction
        b = 2 * (position @ quadric @ direction - radius * direction[2])
        c = position @ quadric @ position - 2 * radius * position[2]
        roots = np.roots([a, b, c])
        roots = roots[np.isreal(roots)].real
        if not roots.size:
            expected.append(np.nan)
            continue
        hit_z = position[2] + roots * direction[2]
        valid = (roots > 0) & (1 - (1 + conic) * hit_z / radius >= 0)
        expected.append(
            roots[valid].min() if valid.any() else roots[np.argmin(abs(hit_z))]
        )
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
