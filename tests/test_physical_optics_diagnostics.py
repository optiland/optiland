from __future__ import annotations

import warnings
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import optiland.backend as be
from optiland.physical_optics import (
    BoundaryDiagnostic,
    ScalarField,
    boundary_diagnostic,
    gaussian_field,
)
from tests.utils import assert_allclose, assert_array_equal


def _complex_array(data):
    # be.array uses the backend's real default dtype, discarding imaginary data.
    if be.get_backend() == "torch":
        torch = pytest.importorskip("torch")
        return torch.tensor(data, dtype=torch.complex128)
    return np.asarray(data, dtype=np.complex128)


def _mask(shape, width):
    rows, columns = np.indices(shape)
    return (
        (rows < width)
        | (rows >= shape[0] - width)
        | (columns < width)
        | (columns >= shape[1] - width)
    )


def _reference(data, width):
    intensity = np.abs(data) ** 2
    total = intensity.sum()
    return intensity[_mask(data.shape, width)].sum() / total if total else 0.0


@pytest.mark.parametrize("width", [1, 2, 3])
@pytest.mark.parametrize(
    "case", ["zero", "constant", "centered", "edge", "corner", "random"]
)
def test_boundary_fraction_matches_independent_mask(set_test_backend, width, case):
    shape = (6, 10)
    data = np.zeros(shape, dtype=np.complex128)
    if case == "constant":
        data[:] = 2 + 3j
    elif case == "centered":
        data[2:4, 4:6] = 1j
    elif case == "edge":
        data[0, 4] = 2j
        data[3, 5] = 1
    elif case == "corner":
        data[0, 0] = 1
        data[3, 5] = 1j
    elif case == "random":
        rng = np.random.default_rng(20261004)
        data = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    field = ScalarField(_complex_array(data), dx=0.2, dy=0.7, wavelength=0.0005)
    result = boundary_diagnostic(field, edge_width=width, threshold=0.1)
    expected = _reference(data, width)
    assert isinstance(result, BoundaryDiagnostic)
    assert_allclose(result.boundary_fraction, expected, rtol=1e-13, atol=1e-14)
    assert result.exceeds_threshold is bool(expected > 0.1)
    assert result.edge_width == width
    assert result.threshold == 0.1
    assert 0 <= float(be.to_numpy(result.boundary_fraction)) <= 1
    if case == "constant":
        band_size = shape[0] * shape[1] - (shape[0] - 2 * width) * (
            shape[1] - 2 * width
        )
        assert_allclose(
            result.boundary_fraction, band_size / np.prod(shape), rtol=1e-13, atol=1e-14
        )
    if case == "corner" and width < 3:
        assert_allclose(result.boundary_fraction, 0.5, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(2, 2), (2, 7), (7, 2), (5, 9), (9, 5)])
def test_small_and_transposed_grids(set_test_backend, shape):
    field = ScalarField(be.ones(shape), dx=1, wavelength=0.5)
    width = min(shape) // 2
    result = boundary_diagnostic(field, edge_width=width, threshold=1)
    assert_allclose(
        result.boundary_fraction, _mask(shape, width).mean(), rtol=1e-13, atol=1e-14
    )
    assert not result.exceeds_threshold


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_amplitude_scale_invariance_without_overflow(set_test_backend, scale):
    data = np.arange(1, 61, dtype=float).reshape(6, 10) * (1 + 0.5j)
    field = ScalarField(
        _complex_array(data * scale), dx=1e-100, dy=1e100, wavelength=0.5
    )
    result = boundary_diagnostic(field, edge_width=2, threshold=0.1)
    assert_allclose(
        result.boundary_fraction, _reference(data, 2), rtol=1e-13, atol=1e-14
    )


@pytest.mark.parametrize("shape", [(6, 10), (10, 6), (2, 7), (7, 2)])
def test_full_grid_band_is_exactly_one(set_test_backend, shape):
    rng = np.random.default_rng(20261004)
    data = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    field = ScalarField(_complex_array(data), dx=1, wavelength=0.5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = boundary_diagnostic(
            field, edge_width=min(shape) // 2, threshold=1, warn=True
        )
    assert_allclose(result.boundary_fraction, 1, rtol=0, atol=0)
    assert not result.exceeds_threshold
    assert not caught


def test_small_boundary_fraction_is_not_lost_to_subtraction(set_test_backend):
    data = np.zeros((6, 10), dtype=complex)
    data[0, 0] = 1e-10j
    data[3, 5] = 1
    field = ScalarField(_complex_array(data), dx=1, wavelength=0.5)
    result = boundary_diagnostic(field, edge_width=1, threshold=1e-21)
    assert_allclose(result.boundary_fraction, 1e-20, rtol=1e-13, atol=0)
    assert result.exceeds_threshold


@pytest.mark.parametrize("warn", [False, True])
@pytest.mark.parametrize("threshold", [0.49, 0.5, 0.51])
def test_warning_is_optional_and_strict(set_test_backend, warn, threshold):
    data = np.zeros((6, 10), dtype=complex)
    data[0, 0] = data[3, 5] = 1
    field = ScalarField(_complex_array(data), dx=1, wavelength=0.5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = boundary_diagnostic(
            field, edge_width=1, threshold=threshold, warn=warn
        )
    exceeds = threshold < 0.5
    assert result.exceeds_threshold is exceeds
    assert len(caught) == int(warn and exceeds)
    if caught:
        assert caught[0].category is RuntimeWarning
        assert "periodic FFT" in str(caught[0].message)
        assert caught[0].filename == __file__


def test_diagnostic_does_not_mutate_propagated_field(set_test_backend):
    initial = gaussian_field(
        (12, 18), dx=0.01, dy=0.02, wavelength=0.0005, waist_radius=0.025
    )
    field = initial.propagate(2.0)
    before = be.to_numpy(field.data).copy()
    original_data = field.data
    metadata = (
        field.shape,
        field.dx,
        field.dy,
        field.wavelength,
        field.refractive_index,
    )
    with pytest.warns(RuntimeWarning):
        result = boundary_diagnostic(field, edge_width=2, threshold=1e-20, warn=True)
    assert field.data is original_data
    assert_array_equal(field.data, before)
    assert metadata == (
        field.shape,
        field.dx,
        field.dy,
        field.wavelength,
        field.refractive_index,
    )
    assert_allclose(
        result.boundary_fraction, _reference(before, 2), rtol=1e-13, atol=1e-14
    )
    with pytest.raises(FrozenInstanceError):
        result.threshold = 0.5


@pytest.mark.parametrize("width", [0, -1, 4, 100])
def test_invalid_width_value(set_test_backend, width):
    field = ScalarField(be.ones((6, 10)), dx=1, wavelength=0.5)
    with pytest.raises(ValueError, match="edge_width"):
        boundary_diagnostic(field, edge_width=width, threshold=0.1)


@pytest.mark.parametrize("width", [True, False, 1.0, "1", None])
def test_invalid_width_type(set_test_backend, width):
    field = ScalarField(be.ones((6, 10)), dx=1, wavelength=0.5)
    with pytest.raises(TypeError, match="edge_width"):
        boundary_diagnostic(field, edge_width=width, threshold=0.1)


@pytest.mark.parametrize("threshold", [0, -0.1, np.nan, np.inf, -np.inf])
def test_invalid_threshold_value(set_test_backend, threshold):
    field = ScalarField(be.ones((6, 10)), dx=1, wavelength=0.5)
    with pytest.raises(ValueError, match="threshold"):
        boundary_diagnostic(field, edge_width=1, threshold=threshold)


@pytest.mark.parametrize("threshold", [True, False, 0.1j, "0.1", None])
def test_invalid_threshold_type(set_test_backend, threshold):
    field = ScalarField(be.ones((6, 10)), dx=1, wavelength=0.5)
    with pytest.raises(TypeError, match="threshold"):
        boundary_diagnostic(field, edge_width=1, threshold=threshold)


@pytest.mark.parametrize(
    "value", [np.nan, np.inf, -np.inf, complex(0, np.nan), complex(0, np.inf)]
)
@pytest.mark.parametrize("position", [(0, 0), (3, 5)])
def test_nonfinite_amplitudes_are_rejected(set_test_backend, value, position):
    data = np.ones((6, 10), dtype=complex)
    data[position] = value
    field = ScalarField(_complex_array(data), dx=1, wavelength=0.5)
    with pytest.raises(ValueError, match="finite"):
        boundary_diagnostic(field, edge_width=1, threshold=0.1)


def test_invalid_field_and_warning_types(set_test_backend):
    with pytest.raises(TypeError, match="ScalarField"):
        boundary_diagnostic(be.ones((6, 10)), edge_width=1, threshold=0.1)
    field = ScalarField(be.ones((6, 10)), dx=1, wavelength=0.5)
    with pytest.raises(TypeError, match="warn"):
        boundary_diagnostic(field, edge_width=1, threshold=0.1, warn="yes")


def test_backend_change_is_rejected(set_test_backend):
    if len(be.list_available_backends()) < 2:
        pytest.skip("requires both backends")
    field = ScalarField(be.ones((6, 10)), dx=1, wavelength=0.5)
    original_backend = be.get_backend()
    try:
        be.set_backend("torch" if original_backend == "numpy" else "numpy")
        with pytest.raises(RuntimeError, match="active backend changed"):
            boundary_diagnostic(field, edge_width=1, threshold=0.1)
    finally:
        be.set_backend(original_backend)


@pytest.mark.parametrize("precision", ["complex64", "complex128"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("zero", [False, True])
def test_torch_metric_preserves_dtype_device_and_gradients(
    set_test_backend, precision, device, zero
):
    if be.get_backend() != "torch":
        pytest.skip("Torch-specific contract")
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    data = torch.zeros((6, 10), dtype=getattr(torch, precision), device=device)
    if not zero:
        data[0, 0] = 2j
        data[3, 5] = 1
    data.requires_grad_()
    field = ScalarField(data, dx=1, wavelength=0.5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = boundary_diagnostic(field, edge_width=1, threshold=0.1, warn=True)
    fraction = result.boundary_fraction
    assert fraction.dtype == data.real.dtype
    assert fraction.device == data.device
    assert fraction.ndim == 0
    assert fraction.requires_grad
    assert len(caught) == (0 if zero else 1)
    fraction.backward()
    assert torch.isfinite(data.grad).all()
    if zero:
        assert torch.count_nonzero(data.grad) == 0


def test_torch_gradient_matches_analytic_and_finite_difference(set_test_backend):
    if be.get_backend() != "torch":
        pytest.skip("Torch-specific contract")
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(20261004)
    data = rng.uniform(0.2, 1.5, size=(6, 10))
    direction = rng.normal(size=data.shape)
    samples = torch.tensor(data, dtype=torch.float64, requires_grad=True)
    field = ScalarField(samples, dx=0.2, dy=0.7, wavelength=0.5)
    result = boundary_diagnostic(field, edge_width=2, threshold=0.1)
    result.boundary_fraction.backward()
    expected = _reference(data, 2)
    gradient = 2 * data * (_mask(data.shape, 2) - expected) / np.sum(data**2)
    assert_allclose(samples.grad, gradient, rtol=1e-12, atol=1e-14)
    derivative = np.sum(gradient * direction)
    for step in (1e-3, 1e-4, 1e-5):
        difference = (
            _reference(data + step * direction, 2)
            - _reference(data - step * direction, 2)
        ) / (2 * step)
        np.testing.assert_allclose(difference, derivative, rtol=1e-5, atol=1e-10)
