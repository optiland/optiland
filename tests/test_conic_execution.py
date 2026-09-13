"""Input and numerical equivalence contracts for conic execution paths."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from numba import config

import optiland.backend as be
from optiland.backend.numpy_backend import conic as numpy_conic
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
    expected = _conic_intersection_distance(rays, np.array([[12.0]]), np.array([[0.5]]))
    assert actual.shape == (size,)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, expected.reshape(-1))


@pytest.mark.skipif(config.DISABLE_JIT, reason="Requires compiled/Python comparison")
@pytest.mark.parametrize("layout", ["empty", "single", "contiguous", "strided"])
@pytest.mark.parametrize(
    "kernel_name", ["_numpy_conic_distance", "_numpy_conic_candidates"]
)
def test_numpy_compiled_loops_match_python_and_analytic_sphere(kernel_name, layout):
    """Check loop outputs and masks, including misses and singular/self hits."""
    # Unit sphere centered at (0, 0, 1): ordinary hits, a tangent, a miss,
    # a chord starting on the surface, a signed fallback, and nonfinite rays.
    values = np.array(
        [
            [0, 0.6, 1, 2, 1, 0, np.nan, np.inf],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [-1, -1, -1, -1, 1, 4, -1, -1],
            [0, 0, 0, 0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 1, 1],
        ],
        dtype=np.float64,
    )
    count = {"empty": 0, "single": 1}.get(layout, values.shape[1])
    values = values[:, :count]
    if layout == "strided":
        values = np.repeat(values, 2, axis=1)[:, ::2]
        assert all(not value.flags.c_contiguous for value in values)
    original = values.copy()
    kernel = getattr(numpy_conic, kernel_name)
    with np.errstate(invalid="ignore"):
        compiled = kernel(*values, 1.0, 0.0)
        # Exercise the very same loop under Python tracing in ordinary CI,
        # while checking that Numba preserves its numerical behavior.
        interpreted = kernel.py_func(*values, 1.0, 0.0)
    for actual, expected in zip(compiled, interpreted, strict=True):
        assert actual.shape == expected.shape == (count,)
        assert actual.dtype == expected.dtype
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=0)
    np.testing.assert_array_equal(values, original)

    regular = np.array([True, True, False, False, True, True, False, False])[:count]
    if kernel_name == "_numpy_conic_distance":
        distance = np.array([1, 1.2, 2, np.nan, 2, -4, np.nan, np.nan])[:count]
        np.testing.assert_allclose(compiled[0], distance, rtol=2e-14, atol=0)
        np.testing.assert_array_equal(compiled[1], regular)
        assert compiled[0].dtype == np.float64
        assert compiled[1].dtype == np.bool_
    else:
        # Check both physical sphere roots independently of loop equivalence;
        # no numeric candidate value is required for an unsolvable ray.
        for actual, expected in (
            (compiled.first, [3, 2.8, 2, np.nan, 2, -4, np.nan, np.nan]),
            (compiled.second, [1, 1.2, 2, np.nan, 0, -2, np.nan, np.nan]),
        ):
            np.testing.assert_allclose(
                actual[compiled.solvable],
                np.array(expected)[:count][compiled.solvable],
                rtol=2e-14,
                atol=0,
            )
        first_valid = [False, False, True, False, True, False, False, False]
        second_valid = [True, True, True, False, False, False, False, False]
        solvable = [True, True, True, False, True, True, False, False]
        np.testing.assert_array_equal(compiled.first_valid, first_valid[:count])
        np.testing.assert_array_equal(compiled.second_valid, second_valid[:count])
        np.testing.assert_array_equal(compiled.pick_second, second_valid[:count])
        np.testing.assert_array_equal(compiled.solvable, solvable[:count])
        np.testing.assert_array_equal(compiled.regular, regular)
        assert all(value.dtype == np.float64 for value in compiled[:2])
        assert all(value.dtype == np.bool_ for value in compiled[2:])


@pytest.mark.skipif(config.DISABLE_JIT, reason="Requires compiled/Python comparison")
@pytest.mark.parametrize("radius", [-2.0, 2.0])
def test_numpy_python_loop_preserves_parabola_aperture_selection(monkeypatch, radius):
    """Both loops preserve signed roots and the aperture's farther-root choice."""
    y = np.array([3.0, 2.0, 0.0, -3.0])
    values = (y * 0, y, y * 0 + radius / 2, y * 0, y * 0 - 1, y * 0)
    # At z=R/2, the parabola x**2+y**2=2*R*z crosses y=+/-2.
    # The second ray starts on it; the final ray requires the signed fallback.
    for contains, expected in (
        (None, [1, 4, 2, -5]),
        (lambda x, y: y < 0, [5, 4, 2, -5]),
    ):
        compiled = be.conic_intersection(*values, radius, -1.0, contains=contains)
        with monkeypatch.context() as patch:
            for name in ("_numpy_conic_distance", "_numpy_conic_candidates"):
                patch.setattr(numpy_conic, name, getattr(numpy_conic, name).py_func)
            interpreted = be.conic_intersection(
                *values, radius, -1.0, contains=contains
            )
        np.testing.assert_allclose(compiled, expected, rtol=2e-14, atol=0)
        np.testing.assert_array_equal(interpreted, compiled)


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
    expected = _conic_intersection_distance(rays, np.array([[2.0]]), np.array([[0.0]]))
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
