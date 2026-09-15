"""Property discovery and invalid inputs at shared material-data boundaries."""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.materials import BaseMaterial, DataMaterial
from optiland.materials.rii import decode_table
from optiland.materials.spectral import interpolate_linear


@pytest.mark.parametrize("name", [None, 3, {"catalog": "glass"}])
def test_material_labels_must_be_strings(name):
    with pytest.raises(ValueError, match="name must be a string"):
        DataMaterial.from_samples([0.4, 0.6], [1.5, 1.6], name=name)


def test_custom_material_has_a_label_without_a_catalog_identity():
    class CustomMaterial(BaseMaterial):
        def _calculate_n(self, wavelength, **kwargs):
            return 1.5

        def _calculate_k(self, wavelength, **kwargs):
            return 0.0

    material = CustomMaterial()
    assert material.display_name == "CustomMaterial"
    assert material.spectral_range() is None


@pytest.mark.parametrize("limits", [None, (0.4, 0.8)])
def test_formula_range_and_missing_extinction_are_independent(limits, set_test_backend):
    material = DataMaterial.from_coefficients(
        "formula 5", [1.5], wavelength_range=limits
    )
    assert material.spectral_range("n") == limits
    assert material.spectral_range("k") is None
    assert be.all(material.k(1.0) == 0)
    if limits is not None:
        with pytest.raises(ValueError, match="range"):
            material.n(1.0)


@pytest.mark.parametrize("waves,values", [([], []), ([0.4, 0.6], [1.5])])
def test_interpolation_rejects_unpaired_input(waves, values, set_test_backend):
    with pytest.raises(ValueError, match="paired samples"):
        interpolate_linear(be.asarray([0.5]), waves, values)


def test_interpolation_rejects_unknown_bounds_policy(set_test_backend):
    with pytest.raises(ValueError, match="bounds policy"):
        interpolate_linear(be.asarray([0.5]), [0.4, 0.6], [1.5, 1.6], bounds="guess")


def test_unknown_table_kind_does_not_fabricate_optical_data():
    assert decode_table({"type": "other measurement", "data": "0.5 1.5"}) == (
        None,
        None,
    )
