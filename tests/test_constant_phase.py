from __future__ import annotations

from optiland import backend as be
from optiland.phase.constant import ConstantPhaseProfile

from .utils import assert_allclose


def test_constant_phase_profile_init():
    profile = ConstantPhaseProfile(phase=1.0)
    assert profile.phase == 1.0


def test_constant_phase_profile_get_phase(set_test_backend):
    profile = ConstantPhaseProfile(phase=2.0)
    x = be.array([1, 2, 3])
    y = be.array([4, 5, 6])
    phase = profile.get_phase(x, y)
    assert_allclose(phase, be.full_like(x, 2.0))


def test_constant_phase_profile_get_gradient(set_test_backend):
    profile = ConstantPhaseProfile()
    x = be.array([1, 2, 3])
    y = be.array([4, 5, 6])
    grad_x, grad_y, grad_z = profile.get_gradient(x, y)
    assert_allclose(grad_x, be.zeros_like(x))
    assert_allclose(grad_y, be.zeros_like(y))
    assert_allclose(grad_z, be.zeros_like(x))


def test_constant_phase_profile_get_paraxial_gradient(set_test_backend):
    profile = ConstantPhaseProfile()
    y = be.array([1, 2, 3])
    paraxial_grad = profile.get_paraxial_gradient(y)
    assert_allclose(paraxial_grad, be.zeros_like(y))


def test_constant_phase_profile_to_from_dict():
    profile = ConstantPhaseProfile(phase=3.0)
    data = profile.to_dict()
    assert data["phase_type"] == "constant"
    assert data["phase"] == 3.0

    new_profile = ConstantPhaseProfile.from_dict(data)
    assert isinstance(new_profile, ConstantPhaseProfile)
    assert new_profile.phase == 3.0
