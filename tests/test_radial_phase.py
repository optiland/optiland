from __future__ import annotations

from optiland import backend as be
from optiland.phase.radial import RadialPhaseProfile

from .utils import assert_allclose


def test_radial_phase_profile_init():
    profile = RadialPhaseProfile(coefficients=[1.0, 2.0])
    assert profile.coefficients == [1.0, 2.0]


def test_radial_phase_profile_get_phase(set_test_backend):
    # phi(r) = 0.5 * r^2 + 0.1 * r^4
    profile = RadialPhaseProfile(coefficients=[0.5, 0.1])
    x = be.array([1, 2, 0])
    y = be.array([0, 1, 0])
    # r^2 = [1, 5, 0]
    # expected_phase = [0.5*1 + 0.1*1, 0.5*5 + 0.1*25, 0] = [0.6, 5.0, 0]
    phase = profile.get_phase(x, y)
    assert_allclose(phase, be.array([0.6, 5.0, 0.0]))


def test_radial_phase_profile_get_gradient(set_test_backend):
    # phi(r) = 0.5 * r^2
    profile = RadialPhaseProfile(coefficients=[0.5])
    x = be.array([1, 0, 3])
    y = be.array([0, 2, 4])
    # d_phi/dr = r
    # grad_x = (d_phi/dr) * x/r = x
    # grad_y = (d_phi/dr) * y/r = y
    grad_x, grad_y, grad_z = profile.get_gradient(x, y)
    assert_allclose(grad_x, x)
    assert_allclose(grad_y, y)
    assert_allclose(grad_z, 0)

    # test r=0 case
    x_zero = be.array([0.0])
    y_zero = be.array([0.0])
    grad_x_zero, grad_y_zero, grad_z_zero = profile.get_gradient(x_zero, y_zero)
    assert_allclose(grad_x_zero, be.array([0.0]))
    assert_allclose(grad_y_zero, be.array([0.0]))
    assert_allclose(grad_z_zero, be.array([0.0]))


def test_radial_phase_profile_get_paraxial_gradient(set_test_backend):
    # phi(r) = 0.5 * r^2
    profile = RadialPhaseProfile(coefficients=[0.5])
    y = be.array([-2, -1, 0, 1, 2])
    # at x=0, d_phi/dy = d_phi/dr * sign(y) = r * sign(y) = |y| * sign(y) = y
    paraxial_grad = profile.get_paraxial_gradient(y)
    assert_allclose(paraxial_grad, y)


def test_radial_phase_profile_to_from_dict():
    profile = RadialPhaseProfile(coefficients=[1.0, 2.0, 3.0])
    data = profile.to_dict()
    assert data["phase_type"] == "radial"
    assert data["coefficients"] == [1.0, 2.0, 3.0]

    new_profile = RadialPhaseProfile.from_dict(data)
    assert isinstance(new_profile, RadialPhaseProfile)
    assert new_profile.coefficients == [1.0, 2.0, 3.0]
