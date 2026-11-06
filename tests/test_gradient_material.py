"""Unit tests for GradientMaterial."""

from __future__ import annotations

import numpy as np

from optiland import backend as be
from optiland.materials import GradientMaterial
from optiland.propagation.grin import GRINPropagation

from .utils import assert_allclose

COEFFICIENTS = {
    "n0": 1.6,
    "nr2": -0.02,
    "nr4": 3e-4,
    "nr6": -2e-6,
    "nz1": 0.01,
    "nz2": -1e-3,
    "nz3": 4e-5,
}


def _expected_index(x, y, z):
    c = COEFFICIENTS
    r2 = x**2 + y**2
    return (
        c["n0"] + c["nr2"] * r2 + c["nr4"] * r2**2 + c["nr6"] * r2**3
        + c["nz1"] * z + c["nz2"] * z**2 + c["nz3"] * z**3
    )  # fmt: skip


def test_index_follows_gradient_law(set_test_backend):
    material = GradientMaterial(**COEFFICIENTS)
    x, y, z = (
        be.array([0.0, 1.0, -2.0]),
        be.array([0.0, 0.5, 1.5]),
        be.array([0.0, 3.0, 7.0]),
    )
    n = material.n(0.55, x=x, y=y, z=z)
    assert_allclose(n, _expected_index(x, y, z))


def test_index_without_position_is_n0(set_test_backend):
    material = GradientMaterial(**COEFFICIENTS)
    assert_allclose(material.n(0.55), 1.6)
    assert_allclose(material.n(be.array([0.48, 0.55, 0.65])), be.full((3,), 1.6))
    assert_allclose(material.k(be.array([0.48, 0.55])), be.zeros(2))


def test_gradient_matches_finite_differences(set_test_backend):
    material = GradientMaterial(**COEFFICIENTS)
    x, y, z = be.array([0.7, -1.3]), be.array([-0.4, 2.1]), be.array([2.5, 6.0])
    n, dn_dx, dn_dy, dn_dz = material.index_and_gradient(x, y, z, 0.55)
    h = 1e-6
    assert_allclose(n, _expected_index(x, y, z))
    for analytic, step in ((dn_dx, (h, 0, 0)), (dn_dy, (0, h, 0)), (dn_dz, (0, 0, h))):
        plus = _expected_index(x + step[0], y + step[1], z + step[2])
        minus = _expected_index(x - step[0], y - step[1], z - step[2])
        assert_allclose(analytic, (plus - minus) / (2 * h), atol=1e-8)


def test_default_propagation_model_is_grin():
    assert isinstance(GradientMaterial().propagation_model, GRINPropagation)
    model = GRINPropagation(step_size=0.02)
    assert GradientMaterial(propagation_model=model).propagation_model is model


def test_to_dict_stores_coefficients():
    data = GradientMaterial(**COEFFICIENTS).to_dict()
    assert data["type"] == "GradientMaterial"
    for key, value in COEFFICIENTS.items():
        assert np.isclose(data[key], value)
