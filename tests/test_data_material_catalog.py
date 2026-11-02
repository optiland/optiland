"""Independent optical values and portable native material data."""

from __future__ import annotations

import json
import math
import subprocess
import sys

import numpy as np
import pytest

import optiland.backend as be
from optiland.materials import BaseMaterial, DataMaterial
from optiland.materials.dispersion import evaluate_formula
from tests.utils import assert_allclose


@pytest.mark.parametrize(
    "formula,coefficients,expected",
    [
        (
            "formula 2",
            [0, 1.2, 0.01, 0.1, 0.04, 0.5, 100],
            lambda w: np.sqrt(
                1
                + 1.2 * w * w / (w * w - 0.01)
                + 0.1 * w * w / (w * w - 0.04)
                + 0.5 * w * w / (w * w - 100)
            ),
        ),
        (
            "formula 3",
            [2.2, 0.01, 2, 0.002, -2, 0.0001, -4, 0.00001, -6, 0.000001, -8],
            lambda w: np.sqrt(
                2.2
                + 0.01 * w * w
                + 0.002 / w**2
                + 0.0001 / w**4
                + 0.00001 / w**6
                + 0.000001 / w**8
            ),
        ),
        (
            "formula 5",
            [1.4, 0.002, -1, 0.0001, -3.5],
            lambda w: 1.4 + 0.002 / w + 0.0001 / w**3.5,
        ),
        (
            "formula 7",
            [1.6, 0.003, 0.0001, 0.01, 0.001, 0.0001],
            lambda w: (
                1.6
                + 0.003 / (w * w - 0.028)
                + 0.0001 / (w * w - 0.028) ** 2
                + 0.01 * w**2
                + 0.001 * w**4
                + 0.0001 * w**6
            ),
        ),
    ],
)
def test_analytic_dispersion(formula, coefficients, expected, set_test_backend):
    material = DataMaterial.from_coefficients(formula, coefficients, name="synthetic")
    waves = np.array([[0.4, 0.55], [0.65, 0.9]])
    assert_allclose(material.n(be.array(waves)), be.array(expected(waves)))
    assert material.n(0.5).item() == pytest.approx(expected(0.5))
    assert_allclose(material.k(be.array(waves)), be.zeros_like(be.array(waves)))
    restored = BaseMaterial.from_dict(json.loads(json.dumps(material.to_dict())))
    assert_allclose(restored.n(be.array(waves)), material.n(be.array(waves)))


def transmission_material():
    return DataMaterial.from_coefficients(
        "formula 5",
        [1.5],
        name="table",
        extinction={
            "kind": "internal_transmittance",
            "wavelengths_um": [0.4, 0.5, 0.6, 0.8],
            "values": [0.7, 0.75, 0.82, 0.99],
            "thickness_mm": 5,
            "interpolation": "local_quadratic",
        },
    )


def test_transmission_interpolates_before_beer_law(set_test_backend):
    material = transmission_material()
    # First three points lie on T=.75 + .6*(w-.5) + (w-.5)^2.
    t = 0.75 + 0.6 * -0.05 + (-0.05) ** 2
    assert material.internal_transmittance(0.45, 5).item() == pytest.approx(t)
    assert material.internal_transmittance(0.45, 10).item() == pytest.approx(t * t)
    assert material.internal_transmittance(0.45, 0).item() == 1
    assert material.k(0.45).item() == pytest.approx(
        -math.log(t) * 0.45 / (4 * math.pi * 5000)
    )
    with pytest.raises(ValueError, match="outside"):
        material.k(0.9)


def test_transparent_table_stays_transparent_between_samples(set_test_backend):
    material = DataMaterial.from_coefficients(
        "formula 5",
        [1.5],
        name="transparent",
        extinction={
            "kind": "internal_transmittance",
            "wavelengths_um": [0.4, 0.5, 0.6, 0.8],
            "values": [1.0, 1.0, 1.0, 1.0],
            "thickness_mm": 5,
            "interpolation": "local_quadratic",
        },
    )
    waves = be.array(np.linspace(0.4, 0.8, 101))
    assert be.all(material.internal_transmittance(waves, 10) == 1)
    assert be.all(material.k(waves) == 0)


@pytest.mark.parametrize(
    "model,c",
    [
        ("constant", [2e-6]),
        ("corning", [1e-6, 2e-7, 3e-8, 4e-9]),
        ("schott", [1e-6, 2e-8, 3e-10, 4e-7, 5e-9, 0.2]),
    ],
)
def test_thermal_models(model, c, set_test_backend):
    material = DataMaterial.from_coefficients(
        "formula 5",
        [1.5],
        name="thermal",
        thermal={
            "kind": model,
            "coefficients": c,
            "reference_temperature_c": 20,
        },
    )
    w, dt = 0.55, 15
    if model == "constant":
        expected = 1.5 + c[0] * dt
    elif model == "corning":
        expected = 1.5 + dt * (c[0] + c[1] / w**2 + c[2] / w**4 + c[3] / w**6)
    else:

        def air(t):
            return 1 + (
                6432.8
                + 2949810 * w * w / (146 * w * w - 1)
                + 25540 * w * w / (41 * w * w - 1)
            ) * 1e-8 / (1 + (t - 15) * 0.0034785)

        n = 1.5 * air(20)
        dn = (
            (n * n - 1)
            / (2 * n)
            * dt
            * (
                c[0]
                + c[1] * dt
                + c[2] * dt * dt
                + (c[3] + c[4] * dt) / (w * w - c[5] ** 2)
            )
        )
        expected = (n + dn) / air(35)
    assert material.n(w, temperature=35).item() == pytest.approx(expected)
    assert material.n(w, temperature=20).item() == pytest.approx(1.5)
    with pytest.raises(ValueError, match="temperature"):
        material.n(w, temperature=float("nan"))


def test_embedded_material_has_no_external_dependency(tmp_path):
    path = tmp_path / "material.json"
    path.write_text(json.dumps(transmission_material().to_dict()))
    code = """
import json, sys
from optiland.materials import BaseMaterial
m = BaseMaterial.from_dict(json.load(open(sys.argv[1])))
print(m.n(.45).item(), m.internal_transmittance(.45, 10).item())
"""
    run = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert np.fromstring(run.stdout, sep=" ") == pytest.approx([1.5, 0.7225**2])


def test_data_is_owned_and_serialized_without_aliases():
    coeff = [1.5]
    metadata = {"provenance": [{"catalog": "synthetic"}]}
    material = DataMaterial.from_coefficients(
        "formula 5", coeff, name="owned", metadata=metadata
    )
    coeff[0] = 2
    metadata["provenance"].clear()
    saved = material.to_dict()
    saved["definition"]["dispersion"]["coefficients"][0] = 3
    saved["metadata"]["provenance"].clear()
    assert material.n(0.55).item() == 1.5
    assert material.metadata["provenance"] == [{"catalog": "synthetic"}]


@pytest.mark.parametrize("wave", [0, -1, float("nan"), float("inf"), 0.3, 0.9])
def test_rejects_invalid_wavelengths(wave):
    material = DataMaterial.from_coefficients(
        "formula 5", [1.5], name="bounded", wavelength_range=[0.4, 0.8]
    )
    with pytest.raises(ValueError, match="Wavelength"):
        material.n(wave)


@pytest.mark.parametrize(
    "formula,coefficients,extra,message",
    [
        ("formula 5", [float("nan")], {}, "finite"),
        ("unknown", [1.5], {}, "Unsupported"),
        ("formula 7", [1.5], {}, "Invalid coefficients"),
        ("formula 5", [1.5, 0.01], {}, "Invalid coefficients"),
        ("formula 5", [1.5], {"thermal": {"kind": "unknown"}}, "Unsupported thermal"),
        (
            "formula 5",
            [1.5],
            {
                "thermal": {
                    "kind": "constant",
                    "coefficients": [],
                    "reference_temperature_c": 20,
                }
            },
            "Invalid thermal",
        ),
        (
            "formula 5",
            [1.5],
            {
                "thermal": {
                    "kind": "constant",
                    "coefficients": [1e-6],
                    "reference_temperature_c": -273.15,
                }
            },
            "Invalid thermal",
        ),
    ],
)
def test_invalid_catalog_definitions_fail(formula, coefficients, extra, message):
    with pytest.raises(ValueError, match=message):
        DataMaterial.from_coefficients(formula, coefficients, name="invalid", **extra)


def test_unknown_shared_dispersion_formula_fails():
    with pytest.raises(ValueError, match="Unsupported"):
        evaluate_formula("unknown", [1.5], be.array([0.55]))


def test_generic_data_does_not_impose_a_positive_index_rule(set_test_backend):
    assert (
        DataMaterial.from_coefficients("formula 5", [-1], name="negative")
        .n(0.55)
        .item()
        == -1
    )
    material = DataMaterial.from_coefficients(
        "formula 5",
        [1.5],
        name="thermal",
        thermal={
            "kind": "constant",
            "coefficients": [-1],
            "reference_temperature_c": 20,
        },
    )
    # Native material definitions permit finite negative indices.
    assert material.n(0.55, temperature=30).item() == pytest.approx(-8.5)


@pytest.mark.parametrize("thickness", [-1, float("nan"), float("inf")])
def test_invalid_transmission_thickness_is_rejected(thickness):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        transmission_material().internal_transmittance(0.55, thickness)


def test_missing_transmission_is_lossless(set_test_backend):
    material = DataMaterial.from_coefficients("formula 5", [1.5], name="lossless")
    assert_allclose(material.internal_transmittance(be.array([0.4, 0.8]), 10), [1, 1])


def test_torch_wavelength_gradients():
    torch = pytest.importorskip("torch")
    be.set_backend("torch")
    try:
        wave = torch.tensor([0.45], dtype=torch.float64, requires_grad=True)
        material = DataMaterial.from_coefficients(
            "formula 5", [1.5, 0.002, -1], name="gradient"
        )
        material.n(wave).sum().backward()
        assert wave.grad.item() == pytest.approx(-0.002 / 0.45**2)
        wave.grad = None
        transmission_material().internal_transmittance(wave, 5).sum().backward()
        assert wave.grad.item() == pytest.approx(0.5)
    finally:
        be.set_backend("numpy")
