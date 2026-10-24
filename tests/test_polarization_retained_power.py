"""Retained scalar power is independent of the last source-state evaluation."""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.rays import PolarizedRays, create_polarization
from tests.utils import assert_allclose


def bundle(power):
    return PolarizedRays(
        be.zeros(3),
        be.zeros(3),
        be.zeros(3),
        be.zeros(3),
        be.zeros(3),
        be.ones(3),
        power,
        be.full(3, 0.55),
    )


def test_retained_launch_power_and_losses_are_not_applied_twice(set_test_backend):
    launch = be.array([4, 2, 0])
    rays = bundle(launch)
    rays.attenuate(be.array([0.25, 0.5, 0.7]))
    rays.attenuate(0.25)
    assert_allclose(launch, [4, 2, 0])
    # Scalar attenuation leaves the Jones operator untouched.
    assert_allclose(rays.p, be.tile(be.eye(3), (3, 1, 1)))
    raw = rays.get_output_field(be.tile(be.array([1, 0, 0]), (3, 1)))
    assert_allclose(raw, be.array([[1, 0, 0], [1, 0, 0], [0, 0, 0]]))
    for name in ("V", "H", "RCP", "unpolarized", "H"):
        state = create_polarization(name)
        before = be.copy(rays.i)
        fields = rays.get_exit_fields(state)
        assert_allclose(rays.i, before)  # Field evaluation has no side effects.
        assert_allclose(
            sum(be.sum(be.abs(e) ** 2, axis=1) for e in fields), [0.25, 0.25, 0]
        )
        rays.update_intensity(state)
        assert_allclose(rays.i, [0.25, 0.25, 0])


def test_current_intensity_assignment_does_not_change_retained_power(set_test_backend):
    rays = bundle(be.array([4, 2, 1]))
    rays.attenuate(0.25)
    rays.i = be.zeros(3)
    rays.update_intensity(create_polarization("H"))
    assert_allclose(rays.i, [1, 0.5, 0.25])
    rays.clip(be.array([False, True, False]))
    rays.update_intensity(create_polarization("V"))
    assert_allclose(rays.i, [1, 0, 0.25])


@pytest.mark.parametrize("output", ["fields", "intensity"])
def test_launch_and_loss_gradients_survive_state_evaluation(output, set_test_backend):
    if be.get_backend() != "torch":
        pytest.skip("Autograd is a Torch capability")
    import torch

    launch = torch.tensor([4.0, 2.0, 3.0], requires_grad=True, dtype=torch.float64)
    factor = torch.tensor([0.25, 0.5, 0.75], requires_grad=True, dtype=torch.float64)
    rays = bundle(launch)
    rays.attenuate(factor)
    rays.clip(be.array([False, False, True]))
    rays.update_intensity(create_polarization("V"))
    state = create_polarization("unpolarized")
    if output == "fields":
        value = sum(be.sum(be.abs(e) ** 2) for e in rays.get_exit_fields(state))
    else:
        rays.update_intensity(state)
        value = be.sum(rays.i)
    grad_launch, grad_factor = torch.autograd.grad(value, (launch, factor))
    assert_allclose(value, 2)
    assert_allclose(grad_launch, [0.25, 0.5, 0])
    assert_allclose(grad_factor, [4, 2, 0])
