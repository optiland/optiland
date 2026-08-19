from __future__ import annotations

from math import exp, isclose

import pytest

from optiland.environment.conditions import EnvironmentalConditions
from optiland.environment.models.birch_downs import (
    _calculate_saturation_vapor_pressure,
    _calculate_water_vapor_partial_pressure,
    birch_downs_refractive_index,
)


class TestBirchDownsRefractiveIndex:
    """
    Comprehensive tests for the Birch & Downs (1994) air refractive index model.
    """

    # --- Test Constants and Standard Conditions ---
    # Standard conditions for Birch & Downs model
    STD_TEMP_C = 15.0
    STD_PRES_PA = 101325.0
    STD_RH = 0.0
    STD_CO2_PPM_BD = 450.0  # B&D's reference CO2

    def setup_method(self):
        """Set up standard conditions for tests."""
        self.std_conditions = EnvironmentalConditions(
            temperature=self.STD_TEMP_C,
            pressure=self.STD_PRES_PA,
            relative_humidity=self.STD_RH,
            co2_ppm=self.STD_CO2_PPM_BD,
        )

    # --- Test Cases for Helper Functions ---

    def test_calculate_saturation_vapor_pressure_standard(self):
        """Test SVP calculation at standard temperature."""
        # Based on Ciddor's SVP formula used internally by B&D
        t_k_15c = 15.0 + 273.15
        A = 1.2378847e-5
        B = -1.9121316e-2
        C = 33.93711047
        D = -6.3431645e3
        expected_svp_15c = exp(A * t_k_15c**2 + B * t_k_15c + C + D / t_k_15c)
        assert isclose(
            _calculate_saturation_vapor_pressure(15.0), expected_svp_15c, rel_tol=1e-5
        )

    def test_calculate_water_vapor_partial_pressure_dry_air(self):
        """Test water vapor partial pressure for dry air (should be 0)."""
        conditions_dry = EnvironmentalConditions(
            temperature=20.0, pressure=101325.0, relative_humidity=0.0, co2_ppm=450.0
        )
        assert isclose(
            _calculate_water_vapor_partial_pressure(conditions_dry), 0.0, abs_tol=1e-12
        )

    def test_calculate_water_vapor_partial_pressure_moist_air(self):
        """Test water vapor partial pressure for moist air."""
        conditions_moist = EnvironmentalConditions(
            temperature=20.0, pressure=101325.0, relative_humidity=0.5, co2_ppm=450.0
        )
        svp_20c = _calculate_saturation_vapor_pressure(20.0)
        f_w = 1.00062 + (3.14e-8 * 101325.0) + (5.6e-7 * 20.0**2)
        expected_partial_pressure = 0.5 * f_w * svp_20c
        assert isclose(
            _calculate_water_vapor_partial_pressure(conditions_moist),
            expected_partial_pressure,
            rel_tol=1e-5,
        )

    # --- Test Cases for birch_downs_refractive_index ---

    def test_birch_downs_refractive_index_nist_reference(self):
        """
        Test the modified Birch & Downs model against a reference value from the
        NIST documentation's comparison table.
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
        # Expected value from NIST table for Modified Edlén (which is based on B&D)
        expected_n = 1.000271799

        n_bd = birch_downs_refractive_index(wavelength_um, nist_conditions)
        assert isclose(n_bd, expected_n, rel_tol=1e-9)

    def test_birch_downs_refractive_index_dry_air_different_co2(self):
        """
        Test B&D model for dry air with varying CO2 concentration.
        """
        wavelength_um = 0.633
        baseline_conditions = EnvironmentalConditions(
            temperature=self.STD_TEMP_C,
            pressure=self.STD_PRES_PA,
            relative_humidity=0.0,
            co2_ppm=self.STD_CO2_PPM_BD,
        )
        n_baseline = birch_downs_refractive_index(wavelength_um, baseline_conditions)

        conditions_high_co2 = EnvironmentalConditions(
            temperature=self.STD_TEMP_C,
            pressure=self.STD_PRES_PA,
            relative_humidity=0.0,
            co2_ppm=1000.0,  # Higher CO2
        )
        n_high_co2 = birch_downs_refractive_index(wavelength_um, conditions_high_co2)
        # Expect higher refractive index with higher CO2
        assert n_high_co2 > n_baseline

        conditions_low_co2 = EnvironmentalConditions(
            temperature=self.STD_TEMP_C,
            pressure=self.STD_PRES_PA,
            relative_humidity=0.0,
            co2_ppm=0.0,  # No CO2
        )
        n_low_co2 = birch_downs_refractive_index(wavelength_um, conditions_low_co2)
        # Expect lower refractive index with lower CO2
        assert n_low_co2 < n_baseline

    def test_birch_downs_refractive_index_moist_air(self):
        """
        Test B&D model with significant humidity.
        Humidity should decrease the refractive index.
        """
        moist_conditions = EnvironmentalConditions(
            temperature=20.0,
            pressure=101325.0,
            relative_humidity=0.8,  # 80% RH
            co2_ppm=450.0,
        )
        dry_conditions = EnvironmentalConditions(
            temperature=20.0, pressure=101325.0, relative_humidity=0.0, co2_ppm=450.0
        )
        n_moist = birch_downs_refractive_index(0.55, moist_conditions)
        n_dry = birch_downs_refractive_index(0.55, dry_conditions)
        assert n_moist < n_dry

    def test_birch_downs_refractive_index_temperature_effect(self):
        """
        Test B&D model with varying temperature.
        Higher temperature should decrease the refractive index (lower density).
        """
        cold_conditions = EnvironmentalConditions(
            temperature=0.0,
            pressure=self.STD_PRES_PA,
            relative_humidity=self.STD_RH,
            co2_ppm=self.STD_CO2_PPM_BD,
        )
        hot_conditions = EnvironmentalConditions(
            temperature=30.0,
            pressure=self.STD_PRES_PA,
            relative_humidity=self.STD_RH,
            co2_ppm=self.STD_CO2_PPM_BD,
        )
        wavelength_um = 0.633
        n_cold = birch_downs_refractive_index(wavelength_um, cold_conditions)
        n_hot = birch_downs_refractive_index(wavelength_um, hot_conditions)
        assert n_cold > n_hot

    def test_birch_downs_refractive_index_pressure_effect(self):
        """
        Test B&D model with varying pressure.
        Higher pressure should increase the refractive index (higher density).
        """
        low_pressure_conditions = EnvironmentalConditions(
            temperature=self.STD_TEMP_C,
            pressure=50000.0,  # Half atmospheric pressure
            relative_humidity=self.STD_RH,
            co2_ppm=self.STD_CO2_PPM_BD,
        )
        high_pressure_conditions = EnvironmentalConditions(
            temperature=self.STD_TEMP_C,
            pressure=200000.0,  # Double atmospheric pressure
            relative_humidity=self.STD_RH,
            co2_ppm=self.STD_CO2_PPM_BD,
        )
        wavelength_um = 0.633
        n_low_p = birch_downs_refractive_index(wavelength_um, low_pressure_conditions)
        n_high_p = birch_downs_refractive_index(wavelength_um, high_pressure_conditions)
        assert n_low_p < n_high_p

    def test_birch_downs_refractive_index_wavelength_effect(self):
        """
        Test B&D model with varying wavelength (dispersion).
        Shorter wavelength (higher sigma_sq) should increase refractive index.
        """
        n_blue = birch_downs_refractive_index(0.4, self.std_conditions)  # Blue light
        n_red = birch_downs_refractive_index(0.7, self.std_conditions)  # Red light
        assert n_blue > n_red

    def test_birch_downs_refractive_index_edge_wavelengths(self):
        """Test B&D model with very short and very long wavelengths."""
        # Very short wavelength (UV)
        n_uv = birch_downs_refractive_index(0.1, self.std_conditions)
        assert n_uv > 1.0  # Should still be valid, just higher

        # Very long wavelength (IR)
        n_ir = birch_downs_refractive_index(10.0, self.std_conditions)
        assert n_ir > 1.0  # Should be closer to 1.0

    def test_birch_downs_refractive_index_zero_pressure(self):
        """Test B&D model at zero pressure (vacuum). Should return ~1.0."""
        conditions_vacuum = EnvironmentalConditions(
            temperature=self.STD_TEMP_C,
            pressure=0.0,
            relative_humidity=self.STD_RH,
            co2_ppm=self.STD_CO2_PPM_BD,
        )
        wavelength_um = 0.633
        n_vacuum = birch_downs_refractive_index(wavelength_um, conditions_vacuum)
        assert isclose(n_vacuum, 1.0, abs_tol=1e-9)

    def test_birch_downs_refractive_index_invalid_wavelength(self):
        """Test B&D model with non-positive wavelength."""
        with pytest.raises(ValueError, match="Wavelength must be positive."):
            birch_downs_refractive_index(0.0, self.std_conditions)
        with pytest.raises(ValueError, match="Wavelength must be positive."):
            birch_downs_refractive_index(-0.5, self.std_conditions)

    def test_birch_downs_refractive_index_invalid_conditions_type(self):
        """Test B&D model with incorrect conditions type."""
        with pytest.raises(
            TypeError, match="conditions must be an EnvironmentalConditions object."
        ):
            wavelength_um = 0.633
            birch_downs_refractive_index(wavelength_um, "not_a_condition_object")

    def test_birch_downs_refractive_index_high_humidity_extreme_temp(self):
        """Test B&D model with high humidity and extreme temperature."""
        conditions = EnvironmentalConditions(
            temperature=40.0, pressure=100000.0, relative_humidity=0.99, co2_ppm=450.0
        )
        n = birch_downs_refractive_index(0.55, conditions)
        assert n > 1.0

    def test_birch_downs_refractive_index_low_temp_high_pressure(self):
        """Test B&D model with low temperature and high pressure."""
        conditions = EnvironmentalConditions(
            temperature=-20.0, pressure=120000.0, relative_humidity=0.1, co2_ppm=450.0
        )
        n = birch_downs_refractive_index(0.6, conditions)
        assert n > 1.0
