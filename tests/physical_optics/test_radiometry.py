from __future__ import annotations

import io

import numpy as np
import pytest
from scipy.constants import c, epsilon_0

import optiland.backend as be
from optiland.physical_optics.field import ScalarField
from optiland.physical_optics.radiometry import (
    field_to_irradiance,
    irradiance_to_photon_rate,
    normalized_psf,
)
from tests.utils import assert_allclose, assert_array_equal

# Independent 35-digit Decimal references using exact SI h and c, not the
# conversion implementation: 500 nm -> 3.9728917142978574e-19 J per photon.
RATE_1_W_MM2_10_BY_20_UM = 5.0341165675427093e14
RATE_1_NW_500_NM = 2.5170582837713547e9


@pytest.mark.parametrize("precision,rtol", [("float64", 1e-12), ("float32", 5e-7)])
def test_analytical_photon_rate_and_precision(set_test_backend, precision, rtol):
    try:
        be.set_precision(precision)
        irradiance = be.ones((2, 3))
        rate = irradiance_to_photon_rate(
            irradiance, dx_mm=0.01, dy_mm=0.02, wavelength_nm=500.0
        )
        assert rate.dtype == irradiance.dtype
        assert_allclose(rate, RATE_1_W_MM2_10_BY_20_UM, rtol=rtol, atol=0)
        if be.get_backend() == "torch":
            assert rate.device == irradiance.device
    finally:
        be.set_precision("float64")


def test_captured_power_reference(set_test_backend):
    # Six cells with area 0.01 * 0.02 mm² contain exactly 1 nW altogether.
    irradiance = be.full((2, 3), 1e-9 / (6 * 0.01 * 0.02))
    rate = irradiance_to_photon_rate(
        irradiance, dx_mm=0.01, dy_mm=0.02, wavelength_nm=500.0
    )
    assert_allclose(be.sum(rate), RATE_1_NW_500_NM, rtol=1e-12, atol=0)
    # Exported rates have no exposure factor; the consumer applies its interval.
    assert_allclose(be.sum(rate * 0.01), RATE_1_NW_500_NM * 0.01, rtol=1e-12, atol=0)


def test_calibration_is_explicit_and_does_not_change_geometry(set_test_backend):
    data = be.ones((2, 3)) * (1 + 2j)
    field = ScalarField(data, dx=2, dy=3, wavelength=0.5, center=(4, -2))
    before = be.to_numpy(field.data).copy()
    irradiance = field_to_irradiance(field, irradiance_scale_w_per_mm2=0.4)
    assert_allclose(irradiance, 2.0, rtol=1e-12, atol=0)
    assert_array_equal(field.data, before)
    assert (field.dx, field.dy, field.wavelength, field.center) == (2, 3, 0.5, (4, -2))
    with pytest.raises(TypeError, match="irradiance_scale_w_per_mm2"):
        field_to_irradiance(field)


def test_peak_electric_phasor_requires_separate_calibration(set_test_backend):
    # Only this explicitly chosen scale interprets the samples as peak V/m.
    field = ScalarField(be.ones((2, 2)), dx=1, wavelength=0.5, refractive_index=1.5)
    peak_scale = 1.5 * epsilon_0 * c / (2 * 1e6)
    peak = field_to_irradiance(field, irradiance_scale_w_per_mm2=peak_scale)
    power_normalized = field_to_irradiance(field, irradiance_scale_w_per_mm2=1.0)
    assert_allclose(peak, peak_scale, rtol=1e-12, atol=0)
    assert_array_equal(power_normalized, np.ones((2, 2)))


def test_global_phase_and_amplitude_scaling(set_test_backend):
    data = be.array([[1.0, 2.0], [0.0, 3.0]]) + 1j
    field = ScalarField(data, dx=1, wavelength=0.5)
    shifted = ScalarField(data * np.exp(0.73j), dx=1, wavelength=0.5)
    amplified = ScalarField(data * 2, dx=1, wavelength=0.5)
    reference = field_to_irradiance(field, irradiance_scale_w_per_mm2=0.25)
    assert_allclose(
        field_to_irradiance(shifted, irradiance_scale_w_per_mm2=0.25),
        reference,
        rtol=1e-12,
        atol=1e-12,
    )
    assert_allclose(
        field_to_irradiance(amplified, irradiance_scale_w_per_mm2=0.25),
        4 * reference,
        rtol=1e-12,
        atol=0,
    )


def test_rectangular_binning_is_exact_and_conserves_rates(set_test_backend):
    irradiance = be.array(np.arange(1, 25, dtype=float).reshape(4, 6))
    before = be.to_numpy(irradiance).copy()
    # Choose the wavelength and cell area so the conversion is exactly one.
    # Synthetic unit conversion uses a rounded h*c cell width and 1 m light;
    # unlike a decimal near-unit wavelength, this factor is exactly 1 in float64.
    kwargs = dict(dx_mm=1.9864458571489286e-25, dy_mm=1.0, wavelength_nm=1e9)
    fine = irradiance_to_photon_rate(irradiance, **kwargs)
    binned = irradiance_to_photon_rate(irradiance, binning=(2, 3), **kwargs)
    assert binned.shape == (2, 2)
    assert_array_equal(binned, np.array([[30.0, 48.0], [102.0, 120.0]]))
    assert_allclose(be.sum(binned), be.sum(fine), rtol=1e-15, atol=0)
    assert_array_equal(irradiance, before)


def test_optical_loss_and_crop_are_not_renormalized(set_test_backend):
    field = ScalarField(be.ones((4, 6)), dx=1, wavelength=0.5)
    attenuated = ScalarField(field.data * 0.5, dx=1, wavelength=0.5)
    kwargs = dict(dx_mm=0.01, dy_mm=0.02, wavelength_nm=500.0)
    irradiance = field_to_irradiance(field, irradiance_scale_w_per_mm2=1e-12)
    rate = irradiance_to_photon_rate(irradiance, **kwargs)
    lossy_rate = irradiance_to_photon_rate(
        field_to_irradiance(attenuated, irradiance_scale_w_per_mm2=1e-12), **kwargs
    )
    cropped_rate = irradiance_to_photon_rate(irradiance[:2, :3], **kwargs)
    assert_allclose(be.sum(lossy_rate), 0.25 * be.sum(rate), rtol=1e-12, atol=0)
    assert_allclose(be.sum(cropped_rate), 0.25 * be.sum(rate), rtol=1e-12, atol=0)


@pytest.mark.parametrize("precision,rtol", [("float64", 1e-12), ("float32", 5e-7)])
def test_psf_normalization_and_numpy_export(set_test_backend, precision, rtol):
    try:
        be.set_precision(precision)
        intensity = be.array([[0.0, 2.0, 1.0], [1.0, 0.0, 4.0]])
        before = be.to_numpy(intensity).copy()
        kernel = normalized_psf(intensity)
        assert kernel.dtype == intensity.dtype
        assert_allclose(kernel, before / 8, rtol=rtol, atol=0)
        assert_allclose(be.sum(kernel), 1.0, rtol=rtol, atol=0)
        assert_allclose(normalized_psf(intensity * 10), kernel, rtol=rtol, atol=0)
        assert_array_equal(intensity, before)
        # Existing NPY serialization suffices; no filesystem writes or schema.
        stream = io.BytesIO()
        np.save(stream, be.to_numpy(kernel), allow_pickle=False)
        stream.seek(0)
        assert_array_equal(np.load(stream, allow_pickle=False), kernel)
    finally:
        be.set_precision("float64")


@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf, -np.inf])
def test_invalid_image_values(set_test_backend, bad):
    image = be.array([[1.0, bad]])
    for convert in (
        normalized_psf,
        lambda data: irradiance_to_photon_rate(
            data, dx_mm=1, dy_mm=1, wavelength_nm=500
        ),
    ):
        with pytest.raises(ValueError, match="finite|nonnegative"):
            convert(image)


@pytest.mark.parametrize("bad", [[], 1.0, "image"])
def test_invalid_image_types(set_test_backend, bad):
    with pytest.raises(TypeError, match="NumPy array or PyTorch tensor"):
        normalized_psf(bad)


@pytest.mark.parametrize("shape", [(0, 2), (2, 0), (2,), (2, 2, 2)])
def test_invalid_image_shapes(set_test_backend, shape):
    with pytest.raises(ValueError, match="two-dimensional"):
        normalized_psf(be.ones(shape))


def test_complex_image_rejected(set_test_backend):
    with pytest.raises(TypeError, match="real floating-point"):
        normalized_psf(be.ones((2, 2)) + 1j)


def test_zero_irradiance_and_zero_calibration_allowed_but_psf_rejected(
    set_test_backend,
):
    field = ScalarField(be.ones((2, 2)), dx=1, wavelength=0.5)
    zero = field_to_irradiance(field, irradiance_scale_w_per_mm2=0)
    rate = irradiance_to_photon_rate(zero, dx_mm=1, dy_mm=1, wavelength_nm=500)
    assert_array_equal(rate, np.zeros((2, 2)))
    with pytest.raises(ValueError, match="positive finite sum"):
        normalized_psf(zero)


@pytest.mark.parametrize("bad", [-1, np.nan, np.inf, -np.inf])
def test_invalid_calibration_values(set_test_backend, bad):
    field = ScalarField(be.ones((2, 2)), dx=1, wavelength=0.5)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        field_to_irradiance(field, irradiance_scale_w_per_mm2=bad)


@pytest.mark.parametrize("bad", [True, 1j, "1", [1]])
def test_invalid_calibration_types(set_test_backend, bad):
    field = ScalarField(be.ones((2, 2)), dx=1, wavelength=0.5)
    with pytest.raises(TypeError, match="real scalar"):
        field_to_irradiance(field, irradiance_scale_w_per_mm2=bad)


def test_invalid_field_and_nonfinite_field_samples(set_test_backend):
    with pytest.raises(TypeError, match="ScalarField"):
        field_to_irradiance(be.ones((2, 2)), irradiance_scale_w_per_mm2=1)
    field = ScalarField(be.full((2, 2), np.inf), dx=1, wavelength=0.5)
    with np.errstate(invalid="ignore"), pytest.raises(ValueError, match="finite"):
        field_to_irradiance(field, irradiance_scale_w_per_mm2=0)


@pytest.mark.parametrize("name", ["dx_mm", "dy_mm", "wavelength_nm"])
@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf])
def test_invalid_geometry_values(set_test_backend, name, bad):
    kwargs = dict(dx_mm=1, dy_mm=1, wavelength_nm=500)
    kwargs[name] = bad
    with pytest.raises(ValueError, match=name):
        irradiance_to_photon_rate(be.ones((2, 2)), **kwargs)


@pytest.mark.parametrize("bad", [True, "1", 1j, [1]])
def test_invalid_geometry_types(set_test_backend, bad):
    with pytest.raises(TypeError, match="dx_mm"):
        irradiance_to_photon_rate(
            be.ones((2, 2)), dx_mm=bad, dy_mm=1, wavelength_nm=500
        )


@pytest.mark.parametrize("binning", [(0, 1), (-1, 2), (3, 2), (2, 4), (5, 1)])
def test_invalid_binning_values(set_test_backend, binning):
    with pytest.raises(ValueError, match="binning"):
        irradiance_to_photon_rate(
            be.ones((4, 6)), dx_mm=1, dy_mm=1, wavelength_nm=500, binning=binning
        )


@pytest.mark.parametrize("binning", [1, [2, 3], (1,), (1, 2, 3), (1.0, 2), (True, 2)])
def test_invalid_binning_types(set_test_backend, binning):
    with pytest.raises(TypeError, match="binning"):
        irradiance_to_photon_rate(
            be.ones((4, 6)), dx_mm=1, dy_mm=1, wavelength_nm=500, binning=binning
        )


def test_overflowed_psf_sum_rejected(set_test_backend):
    with (
        np.errstate(over="ignore"),
        pytest.raises(ValueError, match="positive finite sum"),
    ):
        normalized_psf(be.full((2, 2), 1e308))


def test_backend_mismatch_rejected(set_test_backend):
    if "torch" not in be.list_available_backends():
        pytest.skip("PyTorch is not available")
    active = be.get_backend()
    other = "torch" if active == "numpy" else "numpy"
    be.set_backend(other)
    wrong_image = be.ones((2, 2))
    wrong_field = ScalarField(wrong_image, dx=1, wavelength=0.5)
    be.set_backend(active)
    with pytest.raises(TypeError, match="active"):
        normalized_psf(wrong_image)
    with pytest.raises(RuntimeError, match="active backend changed"):
        field_to_irradiance(wrong_field, irradiance_scale_w_per_mm2=1)


def test_torch_gradients_match_analytical_derivatives(set_test_backend):
    if be.get_backend() != "torch":
        pytest.skip("Autograd is specific to PyTorch")
    amplitude = be.ones((4, 6))
    field = ScalarField(amplitude, dx=1, wavelength=0.5)
    irradiance = field_to_irradiance(field, irradiance_scale_w_per_mm2=2e-12)
    rate = irradiance_to_photon_rate(
        irradiance, dx_mm=0.01, dy_mm=0.02, wavelength_nm=500, binning=(2, 3)
    )
    be.sum(rate).backward()
    assert_allclose(
        amplitude.grad, 4e-12 * RATE_1_W_MM2_10_BY_20_UM, rtol=1e-12, atol=0
    )
    intensity = be.array([[1.0, 2.0], [3.0, 4.0]])
    normalized_psf(intensity)[0, 0].backward()
    # d(I00 / sum(I))/dI00 = 9/100; other derivatives = -1/100.
    assert_allclose(
        intensity.grad,
        np.array([[0.09, -0.01], [-0.01, -0.01]]),
        rtol=1e-12,
        atol=1e-15,
    )


@pytest.mark.parametrize("precision,rtol", [("float64", 1e-12), ("float32", 5e-7)])
def test_rectangular_binning_physical_reference_and_dtype(
    set_test_backend, precision, rtol
):
    try:
        be.set_precision(precision)
        irradiance = be.ones((4, 6))
        rate = irradiance_to_photon_rate(
            irradiance, dx_mm=0.01, dy_mm=0.02, wavelength_nm=500, binning=(2, 3)
        )
        assert rate.dtype == irradiance.dtype
        assert_allclose(rate, 6 * RATE_1_W_MM2_10_BY_20_UM, rtol=rtol, atol=0)
    finally:
        be.set_precision("float64")


def test_gaussian_power_matches_analytical_integral(set_test_backend):
    from optiland.physical_optics.field import gaussian_field

    # Window reaches more than five waist radii; analytic omitted power is tiny.
    waist_mm = 0.1
    field = gaussian_field(
        (128, 160), dx=0.01, dy=0.008, wavelength=0.0005, waist_radius=waist_mm
    )
    irradiance = field_to_irradiance(field, irradiance_scale_w_per_mm2=1e-9)
    rate = irradiance_to_photon_rate(
        irradiance, dx_mm=0.01, dy_mm=0.008, wavelength_nm=500
    )
    expected_power_w = 1e-9 * np.pi * waist_mm**2 / 2
    expected_rate = expected_power_w * (RATE_1_NW_500_NM / 1e-9)
    assert_allclose(be.sum(rate), expected_rate, rtol=1e-12, atol=0)


def test_overflowed_rates_and_calibration_rejected(set_test_backend):
    image = be.full((2, 2), 1e308)
    with np.errstate(over="ignore"):
        with pytest.raises(ValueError, match="finite"):
            irradiance_to_photon_rate(image, dx_mm=1, dy_mm=1, wavelength_nm=500)
        field = ScalarField(be.full((2, 2), 1e150), dx=1, wavelength=0.5)
        with pytest.raises(ValueError, match="finite"):
            field_to_irradiance(field, irradiance_scale_w_per_mm2=1e20)
