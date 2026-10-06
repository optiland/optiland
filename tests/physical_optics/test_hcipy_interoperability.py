"""Cross-package phase, coordinate, and power contracts for scalar fields."""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.physical_optics import ScalarField
from optiland.physical_optics.interoperability import from_hcipy, to_hcipy
from tests.utils import assert_allclose

hp = pytest.importorskip("hcipy")


@pytest.mark.parametrize("shape", [(3, 5), (4, 6), (3, 6), (4, 5)])
@pytest.mark.parametrize("has_center", [False, True])
def test_roundtrip_preserves_phase_coordinates_and_power(
    set_test_backend, shape, has_center
):
    ny, nx = shape
    grid = hp.make_uniform_grid(
        [nx, ny], [nx * 2e-5, ny * 3e-5], center=[7e-5, -4e-5], has_center=has_center
    )
    samples = (np.arange(nx * ny).reshape(shape) + 1.0) * np.exp(
        1j * np.arange(nx * ny).reshape(shape) * 0.13
    )
    source = hp.Wavefront(hp.Field(samples.ravel(), grid), wavelength=633e-9)
    imported = from_hcipy(source, refractive_index=1.5)
    assert imported.shape == shape
    assert imported.wavelength == pytest.approx(0.000633)
    assert imported.refractive_index == 1.5
    assert_allclose(imported.data, samples / 1000, rtol=1e-12, atol=1e-15)
    assert_allclose(imported.power, source.total_power, rtol=1e-12, atol=1e-15)
    x, y = imported.coordinates()
    assert_allclose(x, grid.separated_coords[0] * 1000, rtol=1e-12, atol=1e-15)
    assert_allclose(y, grid.separated_coords[1] * 1000, rtol=1e-12, atol=1e-15)
    restored = to_hcipy(imported)
    assert_allclose(restored.electric_field.shaped, samples, rtol=1e-12, atol=1e-12)
    assert_allclose(restored.grid.x, grid.x, rtol=1e-12, atol=1e-18)
    assert_allclose(restored.grid.y, grid.y, rtol=1e-12, atol=1e-18)
    assert_allclose(restored.total_power, source.total_power, rtol=1e-12, atol=1e-15)
    assert restored.wavelength == pytest.approx(source.wavelength)
    restored.electric_field[0] = 0
    assert_allclose(imported.data, samples / 1000, rtol=1e-12, atol=1e-15)


def test_reject_vector_field(set_test_backend):
    grid = hp.make_pupil_grid(4)
    source = hp.Wavefront(hp.Field(np.ones((2, grid.size), dtype=complex), grid), 1e-6)
    with pytest.raises(ValueError, match="scalar"):
        from_hcipy(source)


def test_reject_custom_quadrature(set_test_backend):
    grid = hp.make_pupil_grid(4)
    grid.weights = np.ones(grid.size)
    source = hp.Wavefront(hp.Field(np.ones(grid.size), grid), 1e-6)
    with pytest.raises(ValueError, match="weights"):
        from_hcipy(source)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_unit_conversion_rejected(set_test_backend, value):
    field = ScalarField(be.ones((4, 4)), dx=0.1, wavelength=0.001)
    with pytest.raises(ValueError, match="length_scale"):
        to_hcipy(field, length_scale=value)
    source = to_hcipy(field)
    with pytest.raises(ValueError, match="length_scale"):
        from_hcipy(source, length_scale=value)


def test_zero_field_is_valid_and_independent(set_test_backend):
    source = hp.Wavefront(
        hp.Field(np.zeros(16, dtype=complex), hp.make_pupil_grid(4)), 0.01
    )
    imported = from_hcipy(source, length_scale=1.0)
    assert_allclose(imported.power, 0, rtol=0, atol=0)
    source.electric_field[:] = 1
    assert_allclose(imported.power, 0, rtol=0, atol=0)


def test_reject_irregular_and_polar_grids(set_test_backend):
    for grid in [
        hp.CartesianGrid(
            hp.UnstructuredCoords(
                [np.array([0.0, 1.0, 0.0, 2.0]), np.array([0.0, 0.0, 1.0, 1.0])]
            )
        ),
        hp.PolarGrid(hp.RegularCoords([0.1, 0.2], [4, 4], [0.1, 0.0])),
    ]:
        source = hp.Wavefront(hp.Field(np.ones(grid.size), grid), 1e-6)
        with pytest.raises(ValueError, match="Cartesian grid"):
            from_hcipy(source)


def test_nonfinite_samples_rejected(set_test_backend):
    source = hp.Wavefront(hp.Field(np.ones(16), hp.make_pupil_grid(4)), 1e-6)
    source.electric_field[3] = float("nan")
    with pytest.raises(ValueError, match="samples"):
        from_hcipy(source)
    field = ScalarField(be.array(np.full((4, 4), float("inf"))), dx=1, wavelength=0.1)
    with pytest.raises(ValueError, match="samples"):
        to_hcipy(field)


def test_import_uses_configured_torch_precision_without_discarding_phase(
    set_test_backend,
):
    values = np.array([1 + 2j, 3 - 4j, -1j, -0.5 + 0.2j], dtype=np.complex64)
    source = hp.Wavefront(hp.Field(values, hp.make_pupil_grid(2)), 1e-6)
    imported = from_hcipy(source, length_scale=1.0)
    expected_dtype = np.complex128 if be.get_backend() == "torch" else np.complex64
    assert be.to_numpy(imported.data).dtype == expected_dtype
    assert_allclose(imported.data, values.reshape(2, 2), rtol=0, atol=0)
    if be.get_backend() == "torch":
        objective = imported.power
        objective.backward()
        assert imported.data.grad_fn is not None


def test_type_and_backend_mismatch_rejected(set_test_backend):
    with pytest.raises(TypeError, match="Wavefront"):
        from_hcipy(np.zeros((3, 3)))
    with pytest.raises(TypeError, match="ScalarField"):
        to_hcipy(np.zeros((3, 3)))
    field = ScalarField(be.ones((3, 3)), dx=1, wavelength=0.1)
    other = "torch" if be.get_backend() == "numpy" else "numpy"
    original = be.get_backend()
    try:
        be.set_backend(other)
        with pytest.raises(RuntimeError, match="backend changed"):
            to_hcipy(field)
    finally:
        be.set_backend(original)
