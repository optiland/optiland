"""Native transmission contracts are independent of any catalog reader."""

from __future__ import annotations

import json
import math

import pytest

import optiland.backend as be
from optiland.fileio import save_oslo_file
from optiland.materials import BaseMaterial, DataMaterial
from optiland.optic import Optic
from tests.utils import assert_allclose


def transmission(values, waves=(0.25, 0.5, 0.75), **updates):
    definition = {
        "kind": "internal_transmittance",
        "wavelengths_um": waves,
        "values": values,
        "thickness_mm": 5,
        "interpolation": "local_quadratic",
    }
    definition.update(updates)
    return DataMaterial.from_coefficients("formula 5", [1.5], extinction=definition)


def test_native_quadratic_tie_and_round_trip(set_test_backend):
    original = transmission([0.2, 0.4, 0.5, 0.9], [0.25, 0.5, 0.75, 1])
    restored = BaseMaterial.from_dict(json.loads(json.dumps(original.to_dict())))
    for material in (original, restored):
        # Equal third-node distances choose .25, with weights -1/8, 3/4, 3/8.
        assert_allclose(material.internal_transmittance(0.625, 5), 37 / 80)
        assert_allclose(
            material.k(0.625), -math.log(37 / 80) * 0.625 / (4 * math.pi * 5000)
        )


@pytest.mark.parametrize("values", [[0.95, 1, 1], [0.05, 0, 0]])
def test_native_quadratic_rejects_overshoot(values, set_test_backend):
    with pytest.raises(ValueError, match="Unphysical interpolated"):
        transmission(values).internal_transmittance(0.625, 5)


def test_native_zero_transmission_and_zero_thickness(set_test_backend):
    material = transmission([0, 0, 0])
    assert_allclose(material.internal_transmittance(be.array([0.3, 0.6]), 5), 0)
    assert_allclose(material.internal_transmittance(0.5, 0), 1)
    with pytest.raises(ValueError, match="Zero transmission"):
        material.k(0.5)


def test_native_precision_collapse_is_rejected(set_test_backend):
    material = transmission([0.8, 0.85, 0.9], [1, 1.000000001, 2])
    be.set_precision("float32")
    try:
        with pytest.raises(ValueError, match="coincide in active precision"):
            material.internal_transmittance(1, 5)
    finally:
        be.set_precision("float64")


@pytest.mark.parametrize(
    "updates",
    [
        {"values": [0.8, 1.1, 0.9]},
        {"values": [-0.1, 0.8, 0.9]},
        {"values": [float("nan"), 0.8, 0.9]},
        {"wavelengths_um": [0.5, 0.5, 0.8]},
        {"wavelengths_um": [0.5, 0.8], "values": [0.8, 0.9]},
        {"thickness_mm": 0},
        {"interpolation": "linear"},
    ],
)
def test_native_invalid_transmission_definitions_fail(updates):
    updates = dict(updates)
    values = updates.pop("values", [0.7, 0.8, 0.9])
    with pytest.raises(ValueError):
        transmission(values, **updates)


def test_oslo_export_refuses_to_drop_native_thermal_data(tmp_path, set_test_backend):
    material = DataMaterial.from_samples(
        [0.4, 0.8],
        [1.6, 1.4],
        thermal={
            "kind": "constant",
            "coefficients": [2e-6],
            "reference_temperature_c": 20,
        },
    )
    optic = Optic()
    optic.surfaces.add(index=0, thickness=be.inf)
    optic.surfaces.add(index=1, thickness=1, material=material, is_stop=True)
    optic.surfaces.add(index=2)
    optic.set_aperture(aperture_type="EPD", value=1)
    optic.fields.set_type("angle")
    optic.fields.add(y=0)
    optic.wavelengths.add(value=0.55, is_primary=True)
    destination = tmp_path / "protected.len"
    destination.write_text("preserve")
    with pytest.raises(NotImplementedError, match="thermal"):
        save_oslo_file(optic, destination)
    assert destination.read_text() == "preserve"
