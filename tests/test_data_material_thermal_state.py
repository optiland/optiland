"""Thermal ownership, caller units and differentiation across shared evaluation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import yaml

import optiland.backend as be
from optiland.materials import BaseMaterial, DataMaterial, MaterialFile
from tests.utils import assert_allclose


def thermal(kind="constant", coefficients=None):
    return {
        "kind": kind,
        "coefficients": coefficients or [2e-6],
        "reference_temperature_c": 20,
    }


def test_shared_schott_matches_existing_file_with_independent_native_state(
    tmp_path, set_test_backend
):
    c = [1e-6, 2e-8, 3e-10, 4e-7, 5e-9, 0.2]
    model = thermal("schott", c)
    data = DataMaterial.from_coefficients("formula 5", [1.5, 0.001, -2], thermal=model)
    path = tmp_path / "thermal.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "DATA": [{"type": "formula 5", "coefficients": "1.5 .001 -2"}],
                "SPECS": {
                    "thermal_dispersion": [
                        {"type": "Schott", "coefficients": " ".join(map(str, c))}
                    ],
                    "temperature": "20 C",
                },
            }
        )
    )
    file = MaterialFile(str(path))
    waves = be.asarray([0.45, 0.6, 0.8])
    for temperature, pressure in [(0, 1), (35, 0.8), (35, 0)]:
        assert_allclose(
            data.n(waves, temperature=temperature, pressure=pressure),
            file.n(waves, temperature=temperature, pressure=pressure),
        )
    c[0] = 2
    assert data.definition.thermal.coefficients[0] == 1e-6
    with pytest.raises(FrozenInstanceError):
        data.definition.thermal.reference_temperature_c = 10
    restored = BaseMaterial.from_dict(data.to_dict())
    assert restored.definition == data.definition
    assert_allclose(restored.n(waves, temperature=40), data.n(waves, temperature=40))


@pytest.mark.parametrize(
    "temperature,pressure",
    [(float("nan"), 1), (-273.15, 1), (20, -1), (20, float("inf"))],
)
def test_environment_inputs_must_be_finite_and_physical(
    temperature, pressure, set_test_backend
):
    data = DataMaterial.from_coefficients("formula 5", [1.5], thermal=thermal())
    with pytest.raises(ValueError, match="temperature or pressure"):
        data.n(0.55, temperature=temperature, pressure=pressure)


def test_temperature_vectors_and_fresh_gradients(set_test_backend):
    data = DataMaterial.from_coefficients("formula 5", [1.5], thermal=thermal())
    temps = be.asarray([10.0, 20.0, 30.0])
    assert_allclose(data.n(0.55, temperature=temps), 1.5 + (temps - 20) * 2e-6)
    if be.get_backend() == "torch":
        import torch

        data.n(0.55, temperature=30.0)
        for _ in range(2):
            temp = torch.tensor([30.0], dtype=torch.float64, requires_grad=True)
            data.n(0.55, temperature=temp).sum().backward()
            torch.testing.assert_close(
                temp.grad, torch.tensor([2e-6], dtype=torch.float64)
            )


def test_measured_k_converts_to_internal_transmission_with_mm_units(set_test_backend):
    data = DataMaterial.from_samples(
        [0.4, 0.8],
        [1.6, 1.4],
        extinction={
            "kind": "tabulated_k",
            "wavelengths_um": [0.4, 0.8],
            "values": [1e-5, 2e-5],
        },
    )
    expected = np.exp(-4 * np.pi * 1.25e-5 * 2000 / 0.5)
    assert_allclose(data.internal_transmittance(0.5, 2), expected)
    assert_allclose(data.internal_transmittance(0.5, 0), 1)


def test_nonfinite_thermal_result_is_rejected(set_test_backend):
    data = DataMaterial.from_coefficients(
        "formula 5", [0.0], thermal=thermal("schott", [1e-6, 0, 0, 0, 0, 0.2])
    )
    with pytest.raises(ValueError, match="non-finite"):
        data.n(0.55, temperature=30)


@pytest.mark.parametrize("value", [None, [], {"bad": "type"}])
def test_scalar_definition_fields_report_validation_errors(value):
    for field in ["temperature", "thickness"]:
        kwargs = (
            {"thermal": {**thermal(), "reference_temperature_c": value}}
            if field == "temperature"
            else {
                "extinction": {
                    "kind": "internal_transmittance",
                    "wavelengths_um": [0.4, 0.6, 0.8],
                    "values": [0.7, 0.8, 0.9],
                    "thickness_mm": value,
                    "interpolation": "local_quadratic",
                }
            }
        )
        with pytest.raises(ValueError):
            DataMaterial.from_coefficients("formula 5", [1.5], **kwargs)


@pytest.mark.parametrize("field", ["temperature", "thickness"])
def test_owned_scalar_parameters_cannot_silently_detach_torch_graphs(field):
    torch = pytest.importorskip("torch")
    value = torch.tensor(20.0 if field == "temperature" else 5.0, requires_grad=True)
    kwargs = (
        {"thermal": {**thermal(), "reference_temperature_c": value}}
        if field == "temperature"
        else {
            "extinction": {
                "kind": "internal_transmittance",
                "wavelengths_um": [0.4, 0.6, 0.8],
                "values": [0.7, 0.8, 0.9],
                "thickness_mm": value,
                "interpolation": "local_quadratic",
            }
        }
    )
    with pytest.raises(ValueError):
        DataMaterial.from_coefficients("formula 5", [1.5], **kwargs)


def test_thermal_kind_must_be_a_string():
    with pytest.raises(ValueError, match="thermal"):
        DataMaterial.from_coefficients(
            "formula 5", [1.5], thermal={**thermal(), "kind": {}}
        )
