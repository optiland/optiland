"""Catalog extensions retain the shared bounds policy and independent support."""

from __future__ import annotations

import json
import math

import pytest

import optiland.backend as be
from optiland.materials import BaseMaterial, DataMaterial
from tests.utils import assert_allclose


@pytest.mark.parametrize("source", ["samples", "coefficients"])
@pytest.mark.parametrize("bounds", ["raise", "clamp"])
def test_thermal_factories_preserve_bounds_on_native_reload(
    source, bounds, set_test_backend
):
    options = {
        "bounds": bounds,
        "thermal": {
            "kind": "constant",
            "coefficients": [2e-6],
            "reference_temperature_c": 20,
        },
        "extinction": {
            "kind": "tabulated_k",
            "wavelengths_um": [0.5, 0.7],
            "values": [0.01, 0.03],
        },
    }
    if source == "samples":
        original = DataMaterial.from_samples([0.4, 0.8], [1.6, 1.4], **options)
    else:
        original = DataMaterial.from_coefficients(
            "formula 5", [1.5], wavelength_range=(0.4, 0.8), **options
        )
    restored = BaseMaterial.from_dict(json.loads(json.dumps(original.to_dict())))
    for material in (original, restored):
        assert material.bounds == bounds
        assert_allclose(material.n(0.6, temperature=30), 1.5 + 2e-5)
        if source == "samples" and bounds == "clamp":
            assert_allclose(material.n(0.9, temperature=30), 1.4 + 2e-5)
        else:
            with pytest.raises(ValueError, match="range"):
                material.n(0.9, temperature=30)
        if bounds == "clamp":
            assert_allclose(material.k(0.45), 0.01)
            assert_allclose(
                material.internal_transmittance(0.45, 0.001),
                math.exp(-4 * math.pi * 0.01 / 0.45),
            )
        else:
            with pytest.raises(ValueError, match="range"):
                material.k(0.45)


@pytest.mark.parametrize("bounds", ["raise", "clamp"])
def test_transmission_support_stays_strict_with_clamped_index(bounds, set_test_backend):
    original = DataMaterial.from_samples(
        [0.4, 0.8],
        [1.6, 1.4],
        bounds=bounds,
        extinction={
            "kind": "internal_transmittance",
            "wavelengths_um": [0.5, 0.6, 0.7],
            "values": [0.7, 0.8, 0.9],
            "thickness_mm": 5,
            "interpolation": "local_quadratic",
        },
    )
    restored = BaseMaterial.from_dict(json.loads(json.dumps(original.to_dict())))
    for material in (original, restored):
        assert material.bounds == bounds
        assert material.spectral_range("k") == (0.5, 0.7)
        assert_allclose(
            material.internal_transmittance(be.asarray([0.5, 0.6, 0.7]), 5),
            [0.7, 0.8, 0.9],
        )
        for wavelength in (0.45, 0.75):
            with pytest.raises(ValueError, match="outside"):
                material.k(wavelength)
            with pytest.raises(ValueError, match="outside"):
                material.internal_transmittance(wavelength, 5)
