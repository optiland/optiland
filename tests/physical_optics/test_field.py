from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.physical_optics import ScalarField, gaussian_field
from tests.utils import assert_allclose


def test_scalar_field_promotes_real_data_and_reports_shape(set_test_backend):
    field = ScalarField(be.ones((3, 4)), dx=0.2, dy=0.1, wavelength=0.0005)

    assert field.shape == (3, 4)
    assert np.iscomplexobj(be.to_numpy(field.data))
    assert "shape=(3, 4)" in repr(field)


def test_scalar_field_coordinates_are_centered(set_test_backend):
    field = ScalarField(be.ones((3, 4)), dx=2.0, dy=1.0, wavelength=0.5)

    x, y = field.coordinates()

    assert_allclose(x, [-3.0, -1.0, 1.0, 3.0])
    assert_allclose(y, [-1.0, 0.0, 1.0])


def test_scalar_field_intensity_and_power(set_test_backend):
    data = be.ones((3, 4)) * (1 + 2j)
    field = ScalarField(data, dx=0.2, dy=0.1, wavelength=0.0005)

    assert_allclose(field.intensity, be.full((3, 4), 5.0))
    assert_allclose(field.power, 5.0 * 12 * 0.2 * 0.1)


def test_gaussian_field_uses_waist_radius_definition(set_test_backend):
    waist_radius = 0.25
    field = gaussian_field(
        shape=(129, 129),
        dx=0.01,
        wavelength=0.0006328,
        waist_radius=waist_radius,
    )
    x, y = field.coordinates()
    x_grid, y_grid = be.meshgrid(x, y)
    radius_squared = x_grid * x_grid + y_grid * y_grid
    measured_radius = be.sqrt(
        2 * be.sum(field.intensity * radius_squared) / be.sum(field.intensity)
    )

    assert_allclose(measured_radius, waist_radius, rtol=1e-5, atol=1e-7)


def test_gaussian_field_accepts_backend_scalar_amplitude(set_test_backend):
    amplitude = be.ones(()) * (1.0 + 2.0j)

    field = gaussian_field(
        shape=(3, 3),
        dx=0.1,
        wavelength=0.5,
        waist_radius=0.25,
        amplitude=amplitude,
    )

    assert_allclose(field.data[1, 1], amplitude)


def test_gaussian_field_rejects_nonscalar_amplitude(set_test_backend):
    with pytest.raises(ValueError, match="amplitude must be scalar"):
        gaussian_field(
            shape=(3, 4),
            dx=0.1,
            wavelength=0.5,
            waist_radius=0.25,
            amplitude=be.ones((4,)),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dx": 0.0}, "dx"),
        ({"dy": -1.0}, "dy"),
        ({"wavelength": float("inf")}, "wavelength"),
        ({"refractive_index": 0.0}, "refractive_index"),
    ],
)
def test_scalar_field_rejects_invalid_physical_parameters(
    set_test_backend, kwargs, message
):
    parameters = {"dx": 1.0, "wavelength": 0.5} | kwargs

    with pytest.raises(ValueError, match=message):
        ScalarField(be.ones((3, 3)), **parameters)


def test_scalar_field_rejects_invalid_shape(set_test_backend):
    with pytest.raises(ValueError, match="two-dimensional"):
        ScalarField(be.ones((4,)), dx=1.0, wavelength=0.5)

    with pytest.raises(ValueError, match="at least two"):
        ScalarField(be.ones((1, 4)), dx=1.0, wavelength=0.5)


def test_scalar_field_rejects_backend_change():
    if "torch" not in be.list_available_backends():
        pytest.skip("PyTorch is not available")

    try:
        be.set_backend("numpy")
        field = ScalarField(be.ones((4, 4)), dx=1.0, wavelength=0.5)
        be.set_backend("torch")

        with pytest.raises(RuntimeError, match="active backend changed"):
            field.propagate(1.0)
    finally:
        be.set_backend("numpy")
