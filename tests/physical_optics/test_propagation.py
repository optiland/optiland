from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.physical_optics import ScalarField, angular_spectrum, gaussian_field
from tests.utils import assert_allclose


@pytest.mark.parametrize("shape", [(31, 32), (32, 31), (33, 35)])
@pytest.mark.parametrize("evanescent", ["discard", "decay"])
def test_zero_distance_is_identity(set_test_backend, shape, evanescent):
    field = gaussian_field(
        shape=shape,
        dx=0.02,
        dy=0.03,
        wavelength=0.0006328,
        waist_radius=0.2,
    )

    propagated = angular_spectrum(field, distance=0.0, evanescent=evanescent)

    assert_allclose(propagated.data, field.data, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("backend_distance", [False, True])
def test_zero_distance_decay_preserves_evanescent_content(
    set_test_backend, backend_distance
):
    indices = be.arange_indices(32)
    checkerboard = 1 - 2 * (indices % 2)
    data = checkerboard[:, None] * checkerboard[None, :]
    field = ScalarField(data, dx=0.1, wavelength=1.0)

    distance = be.array(0.0) if backend_distance else 0.0
    propagated = field.propagate(distance, evanescent="decay")

    assert_allclose(propagated.data, field.data, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("distance", [-1e-9, 0.0, 1e-9])
@pytest.mark.parametrize("backend_distance", [False, True])
def test_evanescent_discard_applies_at_every_distance(
    set_test_backend, distance, backend_distance
):
    indices = be.arange_indices(32)
    checkerboard = 1 - 2 * (indices % 2)
    data = checkerboard[:, None] * checkerboard[None, :]
    field = ScalarField(data, dx=0.1, wavelength=1.0)

    if backend_distance:
        distance = be.array(distance)
    propagated = field.propagate(distance, evanescent="discard")

    assert_allclose(propagated.power, 0.0, atol=1e-12)


def test_zero_distance_discard_preserves_only_propagating_content(set_test_backend):
    indices = be.arange_indices(32)
    checkerboard = 1 - 2 * (indices % 2)
    data = 1 + checkerboard[:, None] * checkerboard[None, :]
    field = ScalarField(data, dx=0.1, wavelength=1.0)

    propagated = field.propagate(be.array(0.0), evanescent="discard")

    assert_allclose(propagated.data, be.ones((32, 32)), rtol=1e-12, atol=1e-12)


def test_plane_wave_accumulates_expected_phase(set_test_backend):
    wavelength = 0.5
    refractive_index = 1.4
    distance = 0.37
    field = ScalarField(
        be.ones((16, 18)),
        dx=1.0,
        dy=0.8,
        wavelength=wavelength,
        refractive_index=refractive_index,
    )

    propagated = field.propagate(distance)
    expected_phase = np.exp(1j * 2 * np.pi * refractive_index * distance / wavelength)

    assert_allclose(propagated.data, field.data * expected_phase, atol=1e-12)


def test_propagating_spectrum_conserves_power(set_test_backend):
    field = gaussian_field(
        shape=(128, 128),
        dx=0.01,
        wavelength=0.0006328,
        waist_radius=0.18,
    )

    propagated = field.propagate(120.0)

    assert_allclose(propagated.power, field.power, rtol=1e-12, atol=1e-12)


def test_gaussian_beam_radius_matches_paraxial_solution(set_test_backend):
    wavelength = 0.0006328
    waist_radius = 0.25
    distance = 100.0
    field = gaussian_field(
        shape=(256, 256),
        dx=0.01,
        wavelength=wavelength,
        waist_radius=waist_radius,
    )

    propagated = field.propagate(distance)
    x, y = propagated.coordinates()
    x_grid, y_grid = be.meshgrid(x, y)
    radius_squared = x_grid * x_grid + y_grid * y_grid
    measured_radius = be.sqrt(
        2 * be.sum(propagated.intensity * radius_squared) / be.sum(propagated.intensity)
    )
    rayleigh_range = np.pi * waist_radius**2 / wavelength
    expected_radius = waist_radius * np.sqrt(1 + (distance / rayleigh_range) ** 2)

    assert_allclose(measured_radius, expected_radius, rtol=2e-3, atol=1e-5)


def test_forward_backward_round_trip(set_test_backend):
    field = gaussian_field(
        shape=(64, 72),
        dx=0.02,
        dy=0.015,
        wavelength=0.0006328,
        waist_radius=0.16,
    )

    round_trip = field.propagate(20.0).propagate(-20.0)

    assert_allclose(round_trip.data, field.data, rtol=1e-11, atol=1e-11)


def test_evanescent_policies_remove_or_attenuate_high_frequency(set_test_backend):
    indices = be.arange_indices(32)
    checkerboard = 1 - 2 * (indices % 2)
    data = checkerboard[:, None] * checkerboard[None, :]
    field = ScalarField(data, dx=0.1, wavelength=1.0)

    discarded = field.propagate(0.2, evanescent="discard")
    decayed = field.propagate(0.2, evanescent="decay")

    assert_allclose(discarded.power, 0.0, atol=1e-12)
    assert float(be.to_numpy(decayed.power)) < float(be.to_numpy(field.power))
    assert float(be.to_numpy(decayed.power)) > 0.0


def test_invalid_propagation_arguments(set_test_backend):
    field = ScalarField(be.ones((8, 8)), dx=1.0, wavelength=0.5)

    with pytest.raises(ValueError, match="distance"):
        field.propagate(float("nan"))
    with pytest.raises(ValueError, match="distance"):
        field.propagate(be.array(float("inf")))
    with pytest.raises(TypeError, match="distance"):
        field.propagate(be.ones((2,)))
    with pytest.raises(TypeError, match="distance must be real"):
        field.propagate(field.data[0, 0] + 1j)
    with pytest.raises(ValueError, match="evanescent"):
        field.propagate(1.0, evanescent="invalid")


def test_propagation_preserves_field_precision(set_test_backend):
    try:
        be.set_precision("float32")
        field = ScalarField(be.ones((8, 8)), dx=1.0, wavelength=0.5)
        field_dtype = field.data.dtype
        real_dtype = field.data.real.dtype

        be.set_precision("float64")
        propagated = field.propagate(be.array(0.1))
        x, y = field.coordinates()

        assert propagated.data.dtype == field_dtype
        assert x.dtype == real_dtype
        assert y.dtype == real_dtype
    finally:
        be.set_precision("float64")


def test_numpy_and_torch_results_match():
    if "torch" not in be.list_available_backends():
        pytest.skip("PyTorch is not available")

    try:
        be.set_backend("numpy")
        numpy_field = gaussian_field(
            shape=(48, 52),
            dx=0.02,
            dy=0.015,
            wavelength=0.0006328,
            waist_radius=0.15,
        )
        numpy_result = be.to_numpy(numpy_field.propagate(30.0).data)

        be.set_backend("torch")
        be.set_device("cpu")
        be.set_precision("float64")
        torch_field = gaussian_field(
            shape=(48, 52),
            dx=0.02,
            dy=0.015,
            wavelength=0.0006328,
            waist_radius=0.15,
        )
        torch_result = be.to_numpy(torch_field.propagate(30.0).data)
    finally:
        be.set_backend("numpy")

    assert np.allclose(torch_result, numpy_result, rtol=1e-11, atol=5e-12)


@pytest.mark.parametrize("distance_value", [-0.17, 0.0, 0.17])
@pytest.mark.parametrize("evanescent", ["discard", "decay"])
def test_torch_autograd_flows_through_field_and_distance(distance_value, evanescent):
    if "torch" not in be.list_available_backends():
        pytest.skip("PyTorch is not available")

    try:
        be.set_backend("torch")
        be.set_device("cpu")
        be.set_precision("float64")
        be.grad_mode.enable()
        amplitude = be.ones((8, 8))
        distance = be.array(distance_value)
        field = ScalarField(amplitude, dx=1.0, wavelength=1.0)

        propagated = field.propagate(distance, evanescent=evanescent)
        loss = be.real(propagated.data[0, 0]) + propagated.power
        loss.backward()

        assert amplitude.grad is not None
        assert distance.grad is not None
        assert bool(be.all(be.isfinite(amplitude.grad)))
        assert bool(be.all(be.isfinite(distance.grad)))
    finally:
        be.set_backend("numpy")


@pytest.mark.parametrize("distance_value", [-0.17, 0.0, 0.17])
@pytest.mark.parametrize("evanescent", ["discard", "decay"])
@pytest.mark.parametrize("mode", [0, 1])
def test_torch_distance_gradient_matches_phase_and_finite_difference(
    distance_value, evanescent, mode
):
    if "torch" not in be.list_available_backends():
        pytest.skip("PyTorch is not available")

    try:
        be.set_backend("torch")
        be.set_device("cpu")
        be.set_precision("float64")
        be.grad_mode.enable()
        size = 16
        phase = 2 * np.pi * mode * be.cast(be.arange_indices(size)) / size
        data = be.ones((size, 1)) * be.exp(1j * phase)[None, :]
        field = ScalarField(data, dx=1.0, wavelength=1.0)
        distance = be.array(distance_value)

        loss = field.propagate(distance, evanescent=evanescent).data[0, 0].imag
        loss.backward()

        kz = np.sqrt((2 * np.pi) ** 2 - (2 * np.pi * mode / size) ** 2)
        expected_gradient = kz * np.cos(kz * distance_value)
        step = 1e-6
        forward = field.propagate(distance_value + step, evanescent=evanescent)
        backward = field.propagate(distance_value - step, evanescent=evanescent)
        finite_difference = (forward.data[0, 0].imag - backward.data[0, 0].imag) / (
            2 * step
        )

        assert distance.grad is not None
        assert_allclose(distance.grad, expected_gradient, rtol=1e-12, atol=1e-12)
        assert_allclose(distance.grad, finite_difference, rtol=1e-9, atol=1e-9)
    finally:
        be.set_backend("numpy")


@pytest.mark.parametrize("distance_value", [-0.02, 0.02])
def test_torch_evanescent_decay_gradient_away_from_zero(distance_value):
    if "torch" not in be.list_available_backends():
        pytest.skip("PyTorch is not available")

    try:
        be.set_backend("torch")
        be.set_device("cpu")
        be.set_precision("float64")
        be.grad_mode.enable()
        indices = be.arange_indices(32)
        checkerboard = 1 - 2 * (indices % 2)
        data = checkerboard[:, None] * checkerboard[None, :]
        field = ScalarField(data, dx=0.1, wavelength=1.0)
        distance = be.array(distance_value)

        power = field.propagate(distance, evanescent="decay").power
        power.backward()

        decay_rate = np.sqrt(2 * (np.pi / field.dx) ** 2 - (2 * np.pi) ** 2)
        expected_power = field.power * np.exp(-2 * abs(distance_value) * decay_rate)
        expected_gradient = -2 * np.sign(distance_value) * decay_rate * expected_power
        assert_allclose(power, expected_power, rtol=1e-12, atol=1e-12)
        assert_allclose(distance.grad, expected_gradient, rtol=1e-12, atol=1e-12)
    finally:
        be.set_backend("numpy")
