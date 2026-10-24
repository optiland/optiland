"""Scalar masking preserves precision, extinction and attenuation gradients."""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.rays import RealRays
from tests.utils import assert_allclose


def bundle(power):
    return RealRays(
        be.zeros(3),
        be.zeros(3),
        be.zeros(3),
        be.zeros(3),
        be.zeros(3),
        be.ones(3),
        power,
        be.full(3, 0.55),
    )


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("scalar", [False, True])
def test_attenuation_precision_and_input_ownership(precision, scalar, set_test_backend):
    be.set_precision(precision)
    try:
        power = be.array([4, 0, 2])
        rays = bundle(power)
        # NumPy's existing ray constructor promotes inputs to float64.
        # Attenuation must preserve the ray's actual storage precision.
        intensity_dtype = rays.i.dtype
        factor = 0.25 if scalar else be.array([0.25, np.nan, 0.5])
        rays.attenuate(factor)
        assert_allclose(rays.i, [1, 0, 0.5 if scalar else 1])
        assert rays.i.dtype == intensity_dtype
        assert_allclose(power, [4, 0, 2])
        if not scalar:
            assert_allclose(factor[[0, 2]], [0.25, 0.5])
            assert np.isnan(be.to_numpy(factor)[1])
    finally:
        be.set_precision("float64")


def test_scalar_ray_attenuation_gradients_mask_invalid_dark_factor(set_test_backend):
    if be.get_backend() != "torch":
        pytest.skip("Autograd is a Torch capability")
    import torch

    power = torch.tensor([4.0, 0.0, 2.0], dtype=torch.float64, requires_grad=True)
    factor = torch.tensor(
        [0.25, float("nan"), 0.5], dtype=torch.float64, requires_grad=True
    )
    rays = bundle(power)
    rays.attenuate(factor)
    grad_power, grad_factor = torch.autograd.grad(rays.i.sum(), (power, factor))
    assert_allclose(rays.i, [1, 0, 1])
    assert_allclose(grad_power, [0.25, 0, 0.5])
    assert_allclose(grad_factor, [4, 0, 2])
