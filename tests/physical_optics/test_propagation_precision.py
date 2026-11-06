"""Independent phase, Gaussian-q, and gradient references for ASM precision."""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.physical_optics import ScalarField


def _array(values, precision):
    values = np.array(values, dtype=f"complex{2 * precision}", copy=True)
    if be.get_backend() == "torch":
        import torch

        return torch.as_tensor(values)
    return values


def _distance(value, kind):
    if kind == "python":
        return value
    if kind == "scalar32":
        return np.float32(value)
    if be.get_backend() == "torch":
        import torch

        return torch.tensor(value, dtype=getattr(torch, kind))
    return np.asarray(value, dtype=kind)


def _focus_metrics(precision, distance_kind, size, dx):
    # A quadratic input wavefront isolates propagation from train/geometry code.
    # Its paraxial q parameter independently specifies the focal plane and waist.
    wavelength, waist, focal_length = 0.0005, 0.2, 50.0
    rayleigh = np.pi * waist**2 / wavelength
    q_after = 1 / (1 / (1j * rayleigh) - 1 / focal_length)
    distance = -q_after.real
    focused_waist = np.sqrt(wavelength * q_after.imag / np.pi)
    x = (np.arange(size) - (size - 1) / 2) * dx
    xx, yy = np.meshgrid(x, x)
    radius_squared = xx**2 + yy**2
    data = np.exp(-radius_squared / waist**2) * np.exp(
        -1j * np.pi * radius_squared / (wavelength * focal_length)
    )
    field = ScalarField(_array(data, precision), dx, wavelength)
    output = field.propagate(_distance(distance, distance_kind))
    assert output.data.dtype == field.data.dtype
    if be.get_backend() == "torch":
        assert output.data.device == field.data.device
    # Double-precision observable reductions avoid conflating field propagation
    # with accumulation error in a float32 second moment.
    intensity = be.to_numpy(output.intensity).astype(np.float64)
    measured_width = np.sqrt(2 * np.sum(radius_squared * intensity) / np.sum(intensity))
    reference = (waist / focused_waist) ** 2 * np.exp(
        -2 * radius_squared / focused_waist**2
    )
    relative_l2 = np.linalg.norm(intensity - reference) / np.linalg.norm(reference)
    halo = np.sum(intensity[radius_squared > (5 * focused_waist) ** 2]) / np.sum(
        intensity
    )
    power_error = float(be.to_numpy(output.power / field.power)) - 1
    return measured_width / focused_waist - 1, relative_l2, halo, power_error


@pytest.mark.parametrize("precision", [32, 64])
@pytest.mark.parametrize("distance_kind", ["python", "float32", "float64"])
@pytest.mark.parametrize("size,dx", [(256, 0.008), (512, 0.004)])
def test_focused_gaussian_has_no_roundoff_halo(
    set_test_backend, precision, distance_kind, size, dx
):
    try:
        be.set_precision(f"float{precision}")
        width_error, relative_l2, halo, power_error = _focus_metrics(
            precision, distance_kind, size, dx
        )
    finally:
        be.set_precision("float64")

    # With >5 input-waist padding and >=4.8 focused-waist samples, the
    # complex128 ASM/paraxial difference is <1e-8. These float32 bounds allow
    # FFT/input roundoff but exclude the former ~27% width error / 7e-4 halo.
    tolerance = 5e-5 if precision == 32 else 2e-7
    assert abs(width_error) < tolerance
    assert relative_l2 < (2e-5 if precision == 32 else 2e-7)
    assert halo < (1e-8 if precision == 32 else 1e-16)
    assert abs(power_error) < (5e-7 if precision == 32 else 1e-12)


@pytest.mark.parametrize("precision", [32, 64])
@pytest.mark.parametrize("distance_kind", ["python", "scalar32", "float32", "float64"])
@pytest.mark.parametrize("distance_value", [-47.89123, 0.0, 47.89123])
@pytest.mark.parametrize("modes", [(0, 0), (4, -3)])
def test_optical_distance_absolute_fourier_mode_phase(
    set_test_backend, precision, distance_kind, distance_value, modes
):
    ny, nx, dx, dy = 48, 60, 0.006, 0.009
    mx, my = modes
    wavelength, n = 0.0005, 1.5
    phase = (
        2
        * np.pi
        * (mx * np.arange(nx)[None, :] / nx + my * np.arange(ny)[:, None] / ny)
    )
    data = np.exp(1j * phase)
    field = ScalarField(
        _array(data, precision), dx, wavelength, dy=dy, refractive_index=n
    )
    distance = _distance(distance_value, distance_kind)
    output = field.propagate(distance)
    # A float32 distance has already been rounded by its caller. Use that
    # represented physical distance, not its unavailable pre-rounding value.
    actual_distance = float(be.to_numpy(distance))
    kz = np.sqrt(
        (2 * np.pi * n / wavelength) ** 2
        - (2 * np.pi * mx / (nx * dx)) ** 2
        - (2 * np.pi * my / (ny * dy)) ** 2
    )
    expected = data * np.exp(1j * kz * actual_distance)
    tolerance = 2e-6 if precision == 32 else 3e-10
    np.testing.assert_allclose(
        be.to_numpy(output.data), expected, rtol=0, atol=tolerance
    )
    assert output.data.dtype == field.data.dtype


@pytest.mark.parametrize("precision", [32, 64])
def test_signed_round_trip_at_optical_distance(set_test_backend, precision):
    rng = np.random.default_rng(1234)
    data = rng.normal(size=(32, 40)) + 1j * rng.normal(size=(32, 40))
    field = ScalarField(_array(data, precision), 0.01, 0.0005)
    round_trip = field.propagate(47.89123).propagate(-47.89123)
    relative_l2 = np.linalg.norm(
        be.to_numpy(round_trip.data - field.data)
    ) / np.linalg.norm(be.to_numpy(field.data))
    assert relative_l2 < (5e-7 if precision == 32 else 1e-12)


@pytest.mark.parametrize("precision", [32, 64])
@pytest.mark.parametrize("mode", [1, 2, 3])
@pytest.mark.parametrize("distance", [-0.17, 0.0, 0.17])
@pytest.mark.parametrize("evanescent", ["discard", "decay"])
def test_propagating_cutoff_and_evanescent_modes(
    set_test_backend, precision, mode, distance, evanescent
):
    size, dx, wavelength = 16, 0.125, 1.0
    data = np.broadcast_to(
        np.exp(2j * np.pi * mode * np.arange(size) / size), (size, size)
    )
    field = ScalarField(_array(data, precision), dx, wavelength)
    output = field.propagate(distance, evanescent=evanescent)
    kz_squared = (2 * np.pi / wavelength) ** 2 - (2 * np.pi * mode / (size * dx)) ** 2
    if kz_squared >= 0:
        transfer = np.exp(1j * np.sqrt(kz_squared) * distance)
    elif evanescent == "discard":
        transfer = 0.0
    else:
        transfer = np.exp(-abs(distance) * np.sqrt(-kz_squared))
    np.testing.assert_allclose(
        be.to_numpy(output.data),
        data * transfer,
        rtol=0,
        atol=5e-7 if precision == 32 else 1e-12,
    )


@pytest.mark.parametrize("precision", [32, 64])
def test_spatial_unit_scaling_does_not_overflow_wavenumber_squared(
    set_test_backend, precision
):
    size, mode = 32, 3
    data = np.broadcast_to(
        np.exp(2j * np.pi * mode * np.arange(size) / size), (size, size)
    )
    values = _array(data, precision)
    output = ScalarField(values, 0.01, 0.0005).propagate(47.89123)
    # The dimensionless physical problem is identical. Squaring its dimensional
    # wavenumbers would overflow float32 after this unit rescaling.
    scale = 1e-18
    with np.errstate(over="raise", invalid="raise"):
        rescaled = ScalarField(values, 0.01 * scale, 0.0005 * scale).propagate(
            47.89123 * scale
        )
    assert np.isfinite(be.to_numpy(rescaled.data)).all()
    np.testing.assert_allclose(
        be.to_numpy(rescaled.data),
        be.to_numpy(output.data),
        rtol=0,
        atol=5e-7 if precision == 32 else 3e-10,
    )


@pytest.mark.parametrize("precision", [32, 64])
@pytest.mark.parametrize("distance_precision", [32, 64])
@pytest.mark.parametrize("distance_value", [-47.89123, 0.0, 47.89123])
def test_torch_optical_distance_and_data_gradients(
    set_test_backend, precision, distance_precision, distance_value
):
    if be.get_backend() != "torch":
        pytest.skip("Torch-specific gradient reference")
    import torch

    size, mode, dx, wavelength = 32, 3, 0.01, 0.0005
    data = np.broadcast_to(
        np.exp(2j * np.pi * mode * np.arange(size) / size), (size, size)
    )
    values = _array(data, precision).requires_grad_()
    field = ScalarField(values, dx, wavelength)
    distance = torch.tensor(
        distance_value,
        dtype=getattr(torch, f"float{distance_precision}"),
        requires_grad=True,
    )
    loss = field.propagate(distance).data[0, 0].imag
    loss.backward()
    actual_distance = distance.detach().item()
    kz = np.sqrt((2 * np.pi / wavelength) ** 2 - (2 * np.pi * mode / (size * dx)) ** 2)
    expected_derivative = kz * np.cos(kz * actual_distance)
    np.testing.assert_allclose(
        distance.grad.item(),
        expected_derivative,
        rtol=2e-6 if precision == 32 or distance_precision == 32 else 1e-9,
        atol=1e-8,
    )
    assert torch.isfinite(values.grad).all()
    assert values.grad.abs().sum() > 0


@pytest.mark.parametrize("precision", [32, 64])
@pytest.mark.parametrize("distance_value", [-0.02, 0.0, 0.02])
@pytest.mark.parametrize("evanescent", ["discard", "decay"])
def test_torch_evanescent_gradients_with_dtype_aware_reference(
    set_test_backend, precision, distance_value, evanescent
):
    if be.get_backend() != "torch":
        pytest.skip("Torch-specific gradient reference")
    import torch

    size, dx, wavelength = 32, 0.1, 1.0
    checkerboard = 1 - 2 * (np.arange(size) % 2)
    field = ScalarField(
        _array(checkerboard[:, None] * checkerboard[None, :], precision), dx, wavelength
    )
    distance = torch.tensor(distance_value, dtype=torch.float64, requires_grad=True)
    output = field.propagate(distance, evanescent=evanescent)
    output.power.backward()
    decay_rate = np.sqrt(2 * (np.pi / dx) ** 2 - (2 * np.pi / wavelength) ** 2)
    expected_power = (
        size**2 * dx**2 * np.exp(-2 * abs(distance_value) * decay_rate)
        if evanescent == "decay"
        else 0.0
    )
    expected_gradient = -2 * np.sign(distance_value) * decay_rate * expected_power
    tolerance = 5e-7 if precision == 32 else 1e-12
    np.testing.assert_allclose(
        output.power.item(), expected_power, rtol=tolerance, atol=1e-12
    )
    np.testing.assert_allclose(
        distance.grad.item(), expected_gradient, rtol=tolerance, atol=1e-12
    )
    if precision == 64 and distance_value != 0 and evanescent == "decay":
        # Several step sizes separate distance-gradient correctness from one
        # lucky finite difference; truncation scales as (decay_rate * step)^2.
        for step in (2e-6, 1e-6, 5e-7):
            plus = field.propagate(
                distance_value + step, evanescent="decay"
            ).power.item()
            minus = field.propagate(
                distance_value - step, evanescent="decay"
            ).power.item()
            np.testing.assert_allclose(
                distance.grad.item(), (plus - minus) / (2 * step), rtol=1e-8, atol=1e-10
            )


@pytest.mark.parametrize("distance_kind", ["python", "float32"])
def test_native_float32_factorization_has_no_gaussian_halo(
    set_test_backend, monkeypatch, distance_kind
):
    # Exercise the same formula without any float64 transfer arrays on CPU.
    # This is numerical coverage for the MPS fallback, not an MPS runtime test.
    from optiland.physical_optics import propagation
    from optiland.physical_optics.field import _cast_real_like

    monkeypatch.setattr(propagation, "_phase_precision", _cast_real_like)
    try:
        be.set_precision("float32")
        width_error, relative_l2, halo, power_error = _focus_metrics(
            32, distance_kind, 256, 0.008
        )
    finally:
        be.set_precision("float64")
    assert abs(width_error) < 5e-5
    assert relative_l2 < 2e-5
    assert halo < 1e-8
    assert abs(power_error) < 5e-7


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_torch_accelerator_device_and_distance_gradient(set_test_backend, device):
    if be.get_backend() != "torch":
        pytest.skip("Torch-specific device reference")
    import torch

    available = (
        torch.cuda.is_available()
        if device == "cuda"
        else torch.backends.mps.is_available()
    )
    if not available:
        pytest.skip(f"{device} device is unavailable")
    # Backend-default creation stays on CPU: propagation must follow the field,
    # not the backend default, including transfer grids and a CPU distance leaf.
    values = torch.ones(
        (16, 18), dtype=torch.complex64, device=device, requires_grad=True
    )
    distance = torch.tensor(0.17, dtype=torch.float64, requires_grad=True)
    field = ScalarField(values, 0.01, 0.0005)
    output = field.propagate(distance)
    output.data[0, 0].imag.backward()
    assert output.data.device == values.device
    assert output.data.dtype == values.dtype
    assert torch.isfinite(values.grad).all()
    assert torch.isfinite(distance.grad)
