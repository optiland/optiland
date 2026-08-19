from __future__ import annotations

from math import isclose

import pytest

from optiland.environment.conditions import EnvironmentalConditions
from optiland.environment.models.kohlrausch import (
    ALPHA_T,
    T_REF_C,
    kohlrausch_refractive_index,
)


class TestKohlrauschRefractiveIndex:
    """
    Comprehensive tests for the Kohlrausch (1968) air refractive index model.
    """

    # --- Test Constants and Standard Conditions ---
    # Standard conditions for Kohlrausch model (dry air, no CO2 dependence)
    STD_TEMP_C = 15.0
    STD_PRES_PA = 101325.0

    # Reference value for Kohlrausch from Zemax OpticStudio documentation.
    # For 0.55 um, 15C, 101325 Pa, calculated from the authoritative formula.
    REF_WAVELENGTH_UM = 0.55
    REF_N_KOHLRAUSCH = 1.00271728

    def setup_method(self):
        """Set up standard conditions for tests."""
        self.std_conditions = EnvironmentalConditions(
            temperature=self.STD_TEMP_C,
            pressure=self.STD_PRES_PA,
            relative_humidity=0.0,  # Kohlrausch ignores humidity
            co2_ppm=400.0,  # Kohlrausch ignores CO2
        )

    # --- Test Cases for kohlrausch_refractive_index ---

    def test_kohlrausch_refractive_index_standard_conditions(self):
        """
        Test Kohlrausch model with standard conditions.
        Compares against a known reference value.
        """
        n_kohlrausch = kohlrausch_refractive_index(
            self.REF_WAVELENGTH_UM, self.std_conditions
        )
        assert isclose(n_kohlrausch, self.REF_N_KOHLRAUSCH, rel_tol=1e-8)

    def test_kohlrausch_refractive_index_temperature_effect(self):
        """
        Test Kohlrausch model with varying temperature.
        Higher temperature should decrease the refractive index.
        """
        cold_conditions = EnvironmentalConditions(
            temperature=0.0, pressure=self.STD_PRES_PA
        )
        hot_conditions = EnvironmentalConditions(
            temperature=30.0, pressure=self.STD_PRES_PA
        )
        n_cold = kohlrausch_refractive_index(self.REF_WAVELENGTH_UM, cold_conditions)
        n_hot = kohlrausch_refractive_index(self.REF_WAVELENGTH_UM, hot_conditions)
        assert n_cold > n_hot

    def test_kohlrausch_refractive_index_pressure_effect(self):
        """
        Test Kohlrausch model with varying pressure.
        Higher pressure should increase the refractive index.
        """
        low_pressure_conditions = EnvironmentalConditions(
            temperature=self.STD_TEMP_C,
            pressure=50000.0,  # Half atmospheric pressure
        )
        high_pressure_conditions = EnvironmentalConditions(
            temperature=self.STD_TEMP_C,
            pressure=200000.0,  # Double atmospheric pressure
        )
        n_low_p = kohlrausch_refractive_index(
            self.REF_WAVELENGTH_UM, low_pressure_conditions
        )
        n_high_p = kohlrausch_refractive_index(
            self.REF_WAVELENGTH_UM, high_pressure_conditions
        )
        assert n_low_p < n_high_p

    def test_kohlrausch_refractive_index_wavelength_effect(self):
        """
        Test Kohlrausch model with varying wavelength (dispersion).
        Shorter wavelength (higher sigma_sq) should increase refractive index.
        """
        n_blue = kohlrausch_refractive_index(0.4, self.std_conditions)  # Blue light
        n_red = kohlrausch_refractive_index(0.7, self.std_conditions)  # Red light
        assert n_blue > n_red

    def test_kohlrausch_refractive_index_edge_wavelengths(self):
        """Test Kohlrausch model with very short and very long wavelengths."""
        # Very short wavelength (UV)
        n_uv = kohlrausch_refractive_index(0.1, self.std_conditions)
        assert n_uv > 1.0  # Should still be valid, just higher

        # Very long wavelength (IR)
        n_ir = kohlrausch_refractive_index(10.0, self.std_conditions)
        assert n_ir > 1.0  # Should be closer to 1.0

    def test_kohlrausch_refractive_index_zero_pressure(self):
        """Test Kohlrausch model at zero pressure (vacuum). Should return ~1.0."""
        conditions_vacuum = EnvironmentalConditions(
            temperature=self.STD_TEMP_C, pressure=0.0
        )
        n_vacuum = kohlrausch_refractive_index(
            self.REF_WAVELENGTH_UM, conditions_vacuum
        )
        assert isclose(n_vacuum, 1.0, abs_tol=1e-9)

    def test_kohlrausch_refractive_index_invalid_wavelength_zero(self):
        """Test Kohlrausch model with zero wavelength."""
        with pytest.raises(ValueError, match="Wavelength must be non-zero."):
            kohlrausch_refractive_index(0.0, self.std_conditions)

    def test_kohlrausch_refractive_index_invalid_wavelength_negative(self):
        """Test model with negative wavelength (computes, but physically invalid)."""
        # The model computes with negative wavelength, but the result is not
        # physically meaningful. This test ensures it doesn't crash.
        n_neg_wavelength = kohlrausch_refractive_index(-0.5, self.std_conditions)
        assert isinstance(n_neg_wavelength, float)

    def test_kohlrausch_refractive_index_invalid_conditions_type(self):
        """Test Kohlrausch model with incorrect conditions type."""
        # Kohlrausch.py does not have an explicit type check.
        # It will likely raise an AttributeError.
        with pytest.raises(AttributeError):
            kohlrausch_refractive_index(
                self.REF_WAVELENGTH_UM, "not_a_condition_object"
            )

    def test_kohlrausch_refractive_index_extreme_low_temperature(self):
        """
        Test model with extremely low temperature causing a non-positive denominator.
        """
        # Calculate the temperature that makes 1 + (t_c - T_REF_C) * ALPHA_T <= 0
        # t_c - T_REF_C <= -1 / ALPHA_T
        # t_c <= T_REF_C - (1 / ALPHA_T)
        critical_temp = T_REF_C - (1.0 / ALPHA_T)
        conditions_extreme_cold = EnvironmentalConditions(
            temperature=critical_temp - 0.1,  # Just below critical
            pressure=self.STD_PRES_PA,
        )
        with pytest.raises(
            ValueError, match="Invalid temperature.*non-positive denominator."
        ):
            kohlrausch_refractive_index(self.REF_WAVELENGTH_UM, conditions_extreme_cold)

    def test_kohlrausch_refractive_index_high_temp_low_pressure(self):
        """Test Kohlrausch model with high temperature and low pressure."""
        conditions = EnvironmentalConditions(temperature=50.0, pressure=50000.0)
        n = kohlrausch_refractive_index(0.6, conditions)
        assert n > 1.0
