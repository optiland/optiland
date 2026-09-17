"""Shared Buchdahl algebra must retain shape, anchors and live gradients."""

from __future__ import annotations

import math

import pytest

import optiland.backend as be
from optiland.materials import AbbeMaterial, AbbeMaterialE
from optiland.materials.buchdahl import buchdahl_coordinate, evaluate_buchdahl

from .utils import assert_allclose


@pytest.mark.parametrize("order", [0, 3, 6])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_polynomial_orders_and_broadcasting(set_test_backend, order, precision):
    be.set_precision(precision)
    try:
        omega = be.array([[-0.125, 0.0, 0.25]])
        index = be.array([[1.5], [1.7]])
        coefficients = [math.comb(order, k) for k in range(1, order + 1)]
        result = evaluate_buchdahl(index, coefficients, omega)
        assert result.shape == (2, 3)
        # Binomial identity checks all powers independently of the evaluator.
        assert_allclose(result, index - 1 + (1 + omega) ** order, rtol=1e-6)
        assert_allclose(result[:, 1], index[:, 0])
    finally:
        be.set_precision("float64")


@pytest.mark.parametrize("reference,alpha", [(0.5875618, 2.5), (0.546074, 2.5),
                                           (0.5875618, 1.49152542373),
                                           (0.5875618, 1.11016949153)])
def test_coordinate_reference_and_inverse(set_test_backend, reference, alpha):
    assert_allclose(buchdahl_coordinate(reference, reference, alpha), 0)
    omega = be.array([[-0.1, 0.0], [0.1, 0.2]])
    wavelength = reference + omega / (1 - alpha * omega)
    assert_allclose(buchdahl_coordinate(wavelength, reference, alpha), omega)


@pytest.mark.parametrize("material,reference", [(AbbeMaterial, 0.5875618),
                                               (AbbeMaterialE, 0.546074)])
def test_existing_models_keep_live_reference_anchor(set_test_backend, material, reference):
    glass = material(1.6, 50)
    if material is AbbeMaterial:
        glass = material(1.6, 50, model="buchdahl")
    for index in (1.6, 1.7):
        glass.index = be.array([index])
        assert_allclose(glass.n(reference), index)


def test_six_term_live_parameters_and_repeated_backward(set_test_backend):
    if be.get_backend() != "torch":
        pytest.skip("Torch autograd")
    import torch

    wavelength = torch.tensor([0.45, 0.5875618, 0.8], dtype=torch.float64,
                              requires_grad=True)
    index = torch.nn.Parameter(torch.tensor(1.5, dtype=torch.float64))
    scale = torch.nn.Parameter(torch.tensor(0.01, dtype=torch.float64))
    alpha = 1.49152542373
    for _ in range(2):
        omega = buchdahl_coordinate(wavelength, 0.5875618, alpha)
        coefficients = [scale * math.comb(6, k) for k in range(1, 7)]
        result = evaluate_buchdahl(index, coefficients, omega)
        result.sum().backward()
        assert_allclose(index.grad, 3)
        assert_allclose(scale.grad, ((1 + omega) ** 6 - 1).sum())
        derivative = 6 * scale * (1 + omega) ** 5 / (1 + alpha * (wavelength - 0.5875618)) ** 2
        assert_allclose(wavelength.grad, derivative)
        with torch.no_grad():
            index += 0.1
            scale += 0.001
        index.grad = scale.grad = wavelength.grad = None
