"""Grid coordinates must survive imports and homogeneous propagation."""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.physical_optics import ScalarField
from tests.utils import assert_allclose


def test_shifted_rectangular_grid_coordinates(set_test_backend):
    field = ScalarField(
        be.ones((3, 4)), dx=0.2, dy=0.3, wavelength=0.001, center=(0.1, -0.4)
    )
    x, y = field.coordinates()
    assert_allclose(x, [-0.2, 0.0, 0.2, 0.4], rtol=0, atol=1e-15)
    assert_allclose(y, [-0.7, -0.4, -0.1], rtol=0, atol=1e-15)
    propagated = field.propagate(1.0)
    assert propagated.center == field.center
    x_out, y_out = propagated.coordinates()
    assert_allclose(x_out, x, rtol=0, atol=0)
    assert_allclose(y_out, y, rtol=0, atol=0)
    assert_allclose(propagated.power, field.power, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("center", [(float("nan"), 0), (0, float("inf"))])
def test_nonfinite_grid_center_rejected(set_test_backend, center):
    with pytest.raises(ValueError, match="finite"):
        ScalarField(be.ones((2, 2)), dx=1, wavelength=0.1, center=center)


@pytest.mark.parametrize("center", [None, [1], (1, 2, 3), (1j, 0), "xy"])
def test_invalid_grid_center_rejected(set_test_backend, center):
    with pytest.raises(TypeError, match="center"):
        ScalarField(be.ones((2, 2)), dx=1, wavelength=0.1, center=center)
