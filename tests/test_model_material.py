from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.materials import Material, ModelMaterial

from .utils import assert_allclose


def _scalar(value) -> float:
    """Return the first element of a backend result as a Python float."""
    return float(np.ravel(be.to_numpy(value))[0])


class TestModelMaterialAnchors:
    """The solved coefficients must reproduce the descriptors that built them."""

    def test_d_line_index_is_reproduced(self, set_test_backend):
        material = ModelMaterial(1.5168, 64.17, 0.0)
        assert_allclose(material.n(ModelMaterial.D_LINE), 1.5168, rtol=0, atol=1e-12)

    def test_abbe_number_is_reproduced(self, set_test_backend):
        material = ModelMaterial(1.5168, 64.17, 0.0)
        assert_allclose(material.abbe(), 64.17, rtol=0, atol=1e-9)

    def test_ir_reference_dispersion_is_reproduced(self, set_test_backend):
        material = ModelMaterial(1.5168, 64.17, 0.0, P_ref=0.60, lambda_ref=1.0)
        assert_allclose(
            material.partial_dispersion_ref(1.0), 0.60, rtol=0, atol=1e-9
        )

    def test_mid_band_matches_catalog_glass(self, set_test_backend):
        """Anchors alone cannot detect a wrong regression column order.

        The lower orders are solved so the F-C and g-F anchors hold exactly, so a
        shuffled regression matrix would still reproduce nd and vd. Comparing
        mid-band values against the catalog is what pins the term order.
        """
        material = ModelMaterial(1.5168, 64.17, 0.0)
        reference = Material("N-BK7")
        for wavelength in (0.50, 0.60, 0.70):
            assert_allclose(
                material.n(wavelength), reference.n(wavelength), rtol=0, atol=1e-3
            )


class TestModelMaterialState:
    def test_parameter_update_refreshes_coefficients(self, set_test_backend):
        material = ModelMaterial(1.5168, 64.17, 0.0)
        material.nd = 1.62

        assert_allclose(material.n(ModelMaterial.D_LINE), 1.62, rtol=0, atol=1e-12)
        assert_allclose(material.abbe(), 64.17, rtol=0, atol=1e-9)

    def test_repeated_evaluation_is_stable(self, set_test_backend):
        material = ModelMaterial(1.5168, 64.17, 0.0)
        first = material.n(0.55)
        second = material.n(0.55)

        assert_allclose(second, first, rtol=0, atol=0)

    def test_numpy_scalar_descriptors_are_accepted(self, set_test_backend):
        material = ModelMaterial(np.float32(1.5168), np.float32(64.17), np.float32(0.0))

        assert_allclose(material.n(0.5875618), 1.5168, rtol=0, atol=1e-5)

    def test_vectorized_matches_scalar(self, set_test_backend):
        material = ModelMaterial(1.5168, 64.17, 0.0)
        wavelengths = np.array([0.50, 0.5875618, 0.70])

        vectorized = material.n(wavelengths)
        expected = np.array([_scalar(material.n(w)) for w in wavelengths])

        assert_allclose(vectorized, expected, rtol=0, atol=1e-12)


class TestModelMaterialValidation:
    def test_wavelength_outside_model_range_raises(self, set_test_backend):
        material = ModelMaterial(1.5168, 64.17, 0.0)

        with pytest.raises(ValueError, match="Wavelength out of range"):
            material.n(0.30)
        with pytest.raises(ValueError, match="Wavelength out of range"):
            material.n(2.40)

    def test_unphysical_index_raises(self, set_test_backend):
        # Deliberately extreme descriptors: the polynomial dips below n = 1.
        material = ModelMaterial(1.0001, 10.0, 0.99)

        with pytest.raises(ValueError, match="unphysical"):
            material.n(0.365)

    def test_refractive_index_outside_glass_range_raises(self, set_test_backend):
        with pytest.raises(ValueError, match="Refractive index"):
            ModelMaterial(0.9, 64.17, 0.0)
        with pytest.raises(ValueError, match="Refractive index"):
            ModelMaterial(3.2, 64.17, 0.0)

    def test_abbe_number_outside_range_raises(self, set_test_backend):
        with pytest.raises(ValueError, match="Abbe"):
            ModelMaterial(1.5168, 5.0, 0.0)

    def test_partial_dispersion_pair_is_required(self, set_test_backend):
        with pytest.raises(ValueError, match="must be provided together"):
            ModelMaterial(1.5168, 64.17, 0.0, P_ref=0.60)

    def test_from_dict_requires_descriptors(self, set_test_backend):
        with pytest.raises(ValueError, match="Missing null required key"):
            ModelMaterial.from_dict({"nd": 1.5168, "vd": None, "dPgF": 0.0})


class TestModelMaterialContract:
    def test_regression_matrix_column_counts(self, set_test_backend):
        visible = np.load(str(ModelMaterial.REGRESSION_MATRIX_VISIBLE))
        infrared = np.load(str(ModelMaterial.REGRESSION_MATRIX_IR))

        assert visible.shape == (20, 4)
        assert infrared.shape == (35, 3)

    def test_extinction_is_zero(self, set_test_backend):
        material = ModelMaterial(1.5168, 64.17, 0.0)

        assert_allclose(material.k(0.55), 0.0, rtol=0, atol=0)

    def test_spectral_range_reports_model_interval(self, set_test_backend):
        material = ModelMaterial(1.5168, 64.17, 0.0)

        assert material.spectral_range("n") == (0.365, 2.30)
        assert material.spectral_range("k") is None

    def test_dict_round_trip(self, set_test_backend):
        material = ModelMaterial(1.5168, 64.17, 0.02, P_ref=0.60, lambda_ref=1.0)

        assert ModelMaterial.from_dict(material.to_dict()) == material
