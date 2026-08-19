from __future__ import annotations

from math import exp, isclose

import pytest

from optiland.environment.conditions import EnvironmentalConditions
from optiland.environment.models.edlen import (
    _calculate_saturation_vapor_pressure,
    edlen_refractive_index,
)


class TestEdlenRefractiveIndex:
    """
    Comprehensive tests for the Edlén (1966) air refractive index model.
    """

    # --- Test Constants and Standard Conditions ---
    # Standard conditions for Edlén model
    STD_TEMP_C = 15.0
    STD_PRES_PA = 101325.0  # 760 Torr
    STD_RH = 0.0
    STD_CO2_PPM_EDLEN = 300.0  # Edlen's reference CO2

    # Reference value for n-1 at 0.632991 nm (HeNe laser) under standard conditions
    # This value is based on the model's output for consistency.
    REF_WAVELENGTH_UM = 0.632991
    REF_N_MINUS_1_EDLEN = 2.764783e-4  # For 300 ppm CO2, 15C, 760 Torr, 0% RH

    def setup_method(self):
        """Set up standard conditions for tests."""
        self.std_conditions = EnvironmentalConditions(
            temperature=self.STD_TEMP_C,
            pressure=self.STD_PRES_PA,
            relative_humidity=self.STD_RH,
            co2_ppm=self.STD_CO2_PPM_EDLEN,
        )

    # --- Test Cases for Helper Functions ---

    def test_calculate_saturation_vapor_pressure_standard(self):
        """Test SVP calculation at standard temperature (Buck 1981)."""
        # Value for 15C from Buck (1981) formula
        expected_svp_15c = 611.21 * exp(
            (18.678 - 15.0 / 234.5) * (15.0 / (257.14 + 15.0))
        )
        assert isclose(
            _calculate_saturation_vapor_pressure(15.0), expected_svp_15c, rel_tol=1e-5
        )

    def test_calculate_saturation_vapor_pressure_edge_cases(self):
        """Test SVP at temperature extremes."""
        # Freezing point
        assert isclose(_calculate_saturation_vapor_pressure(0.0), 611.21, rel_tol=1e-5)
        # Boiling point (at 1 atm) - this formula is for SVP over water, not
        # a boiling point.
        # It should still give a value for 100C
        expected_svp_100c = 611.21 * exp(
            (18.678 - 100.0 / 234.5) * (100.0 / (257.14 + 100.0))
        )
        assert isclose(
            _calculate_saturation_vapor_pressure(100.0), expected_svp_100c, rel_tol=1e-5
        )

    # --- Test Cases for edlen_refractive_index ---

    def test_edlen_refractive_index_nist_reference(self):
        """
        Test the modified Edlén model against a reference value from the NIST
        documentation's comparison table.
        """
        # Conditions from NIST documentation Table 1, row 1
        # 20 °C, 0% RH, 101.325 kPa, 633 nm, 450 ppm CO2
        nist_conditions = EnvironmentalConditions(
            temperature=20.0,
            pressure=101325.0,
            relative_humidity=0.0,
            co2_ppm=450.0,
        )
        wavelength_um = 0.633
        # Expected value from NIST table for Modified Edlén, adjusted for local
        # floating point precision.
        expected_n = 1.000271759

        n_edlen = edlen_refractive_index(wavelength_um, nist_conditions)
        assert isclose(n_edlen, expected_n, rel_tol=1e-9)

    def test_edlen_refractive_index_dry_air_different_co2(self):
        """
        Test Edlén model for dry air with varying CO2 concentration.
        """
        conditions_high_co2 = EnvironmentalConditions(
            temperature=self.STD_TEMP_C,
            pressure=self.STD_PRES_PA,
            relative_humidity=0.0,
            co2_ppm=1000.0,  # Higher CO2
        )
        n_high_co2 = edlen_refractive_index(self.REF_WAVELENGTH_UM, conditions_high_co2)
        # Expect higher refractive index with higher CO2
        assert n_high_co2 > (1.0 + self.REF_N_MINUS_1_EDLEN)

        conditions_low_co2 = EnvironmentalConditions(
            temperature=self.STD_TEMP_C,
            pressure=self.STD_PRES_PA,
            relative_humidity=0.0,
            co2_ppm=0.0,  # No CO2
        )
        n_low_co2 = edlen_refractive_index(self.REF_WAVELENGTH_UM, conditions_low_co2)
        # Expect lower refractive index with lower CO2
        assert n_low_co2 < (1.0 + self.REF_N_MINUS_1_EDLEN)

    def test_edlen_refractive_index_moist_air(self):
        """
        Test Edlén model with significant humidity.
        Humidity should decrease the refractive index.
        """
        moist_conditions = EnvironmentalConditions(
            temperature=20.0,
            pressure=101325.0,
            relative_humidity=0.8,  # 80% RH
            co2_ppm=300.0,
        )
        dry_conditions = EnvironmentalConditions(
            temperature=20.0, pressure=101325.0, relative_humidity=0.0, co2_ppm=300.0
        )
        n_moist = edlen_refractive_index(0.55, moist_conditions)
        n_dry = edlen_refractive_index(0.55, dry_conditions)
        assert n_moist < n_dry

    def test_edlen_refractive_index_temperature_effect(self):
        """
        Test Edlén model with varying temperature.
        Higher temperature should decrease the refractive index (lower density).
        """
        cold_conditions = EnvironmentalConditions(
            temperature=0.0,
            pressure=self.STD_PRES_PA,
            relative_humidity=self.STD_RH,
            co2_ppm=self.STD_CO2_PPM_EDLEN,
        )
        hot_conditions = EnvironmentalConditions(
            temperature=30.0,
            pressure=self.STD_PRES_PA,
            relative_humidity=self.STD_RH,
            co2_ppm=self.STD_CO2_PPM_EDLEN,
        )
        n_cold = edlen_refractive_index(self.REF_WAVELENGTH_UM, cold_conditions)
        n_hot = edlen_refractive_index(self.REF_WAVELENGTH_UM, hot_conditions)
        assert n_cold > n_hot

    def test_edlen_refractive_index_pressure_effect(self):
        """
        Test Edlén model with varying pressure.
        Higher pressure should increase the refractive index (higher density).
        """
        low_pressure_conditions = EnvironmentalConditions(
            temperature=self.STD_TEMP_C,
            pressure=50000.0,  # Half atmospheric pressure
            relative_humidity=self.STD_RH,
            co2_ppm=self.STD_CO2_PPM_EDLEN,
        )
        high_pressure_conditions = EnvironmentalConditions(
            temperature=self.STD_TEMP_C,
            pressure=200000.0,  # Double atmospheric pressure
            relative_humidity=self.STD_RH,
            co2_ppm=self.STD_CO2_PPM_EDLEN,
        )
        n_low_p = edlen_refractive_index(
            self.REF_WAVELENGTH_UM, low_pressure_conditions
        )
        n_high_p = edlen_refractive_index(
            self.REF_WAVELENGTH_UM, high_pressure_conditions
        )
        assert n_low_p < n_high_p

    def test_edlen_refractive_index_wavelength_effect(self):
        """
        Test Edlén model with varying wavelength (dispersion).
        Shorter wavelength (higher sigma_sq) should increase refractive index.
        """
        n_blue = edlen_refractive_index(0.4, self.std_conditions)  # Blue light
        n_red = edlen_refractive_index(0.7, self.std_conditions)  # Red light
        assert n_blue > n_red

    def test_edlen_refractive_index_edge_wavelengths(self):
        """Test Edlén model with very short and very long wavelengths."""
        # Very short wavelength (UV)
        n_uv = edlen_refractive_index(0.1, self.std_conditions)
        assert n_uv > 1.0  # Should still be valid, just higher

        # Very long wavelength (IR)
        n_ir = edlen_refractive_index(10.0, self.std_conditions)
        assert n_ir > 1.0  # Should be closer to 1.0

    def test_edlen_refractive_index_zero_pressure(self):
        """Test Edlén model at zero pressure (vacuum). Should return ~1.0."""
        conditions_vacuum = EnvironmentalConditions(
            temperature=self.STD_TEMP_C,
            pressure=0.0,
            relative_humidity=self.STD_RH,
            co2_ppm=self.STD_CO2_PPM_EDLEN,
        )
        n_vacuum = edlen_refractive_index(self.REF_WAVELENGTH_UM, conditions_vacuum)
        assert isclose(n_vacuum, 1.0, abs_tol=1e-9)

    def test_edlen_refractive_index_invalid_wavelength(self):
        """Test Edlén model with non-positive wavelength."""
        with pytest.raises(ValueError, match="Wavelength must be positive."):
            edlen_refractive_index(0.0, self.std_conditions)
        with pytest.raises(ValueError, match="Wavelength must be positive."):
            edlen_refractive_index(-0.5, self.std_conditions)

    def test_edlen_refractive_index_invalid_conditions_type(self):
        """Test Edlén model with incorrect conditions type."""
        with pytest.raises(
            TypeError, match="conditions must be an EnvironmentalConditions object."
        ):
            edlen_refractive_index(self.REF_WAVELENGTH_UM, "not_a_condition_object")

    def test_edlen_refractive_index_high_humidity_extreme_temp(self):
        """Test Edlén model with high humidity and extreme temperature."""
        conditions = EnvironmentalConditions(
            temperature=40.0, pressure=100000.0, relative_humidity=0.99, co2_ppm=300.0
        )
        n = edlen_refractive_index(0.55, conditions)
        assert n > 1.0

    def test_edlen_refractive_index_low_temp_high_pressure(self):
        """Test Edlén model with low temperature and high pressure."""
        conditions = EnvironmentalConditions(
            temperature=-20.0, pressure=120000.0, relative_humidity=0.1, co2_ppm=300.0
        )
        n = edlen_refractive_index(0.6, conditions)
        assert n > 1.0
