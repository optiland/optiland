from __future__ import annotations

from unittest.mock import patch

import pytest

from optiland.environment.conditions import EnvironmentalConditions
from optiland.environment.models.ciddor import (
    A0_Z,
    A1_Z,
    A2_Z,
    ALPHA_F,
    B0_Z,
    B1_Z,
    BETA_F,
    C0_Z,
    C1_Z,
    CO2_MOLAR_PPM,
    D_Z,
    E_Z,
    GAMMA_F,
    _calculate_compressibility,
    _calculate_enhancement_factor,
    _calculate_molar_mass_air,
    _calculate_saturation_vapor_pressure,
    ciddor_refractive_index,
)

from ..utils import assert_allclose


class TestCiddorRefractiveIndex:
    """
    Comprehensive tests for the Ciddor (1996) air refractive index model.
    """

    # --- Test Constants and Standard Conditions ---
    # Values from Ciddor (1996) paper for validation
    # Standard conditions for Ciddor model
    STD_TEMP_C = 15.0
    STD_PRES_PA = 101325.0
    STD_RH = 0.0
    STD_CO2_PPM = 450.0  # Ciddor's reference CO2 for refractivity

    # Reference value for n-1 at 0.6328 nm under standard conditions
    # This value is from the reference implementation on refractiveindex.info
    REF_WAVELENGTH_UM = 0.6328
    REF_N_MINUS_1_CIDDOR = 2.76534e-4  # For 450 ppm CO2, 15C, 101325 Pa, 0% RH

    def setup_method(self):
        """Set up standard conditions for tests."""
        self.std_conditions = EnvironmentalConditions(
            temperature=self.STD_TEMP_C,
            pressure=self.STD_PRES_PA,
            relative_humidity=self.STD_RH,
            co2_ppm=self.STD_CO2_PPM,
        )

    # --- Test Cases for Helper Functions ---

    def test_calculate_molar_mass_air_standard(self, set_test_backend):
        """Test molar mass calculation at standard CO2."""
        # Value from Ciddor (1996), text after Eq. (4) for 400 ppm
        expected_molar_mass = 1e-3 * (28.9635 + 12.011e-6 * (400.0 - CO2_MOLAR_PPM))
        assert_allclose(_calculate_molar_mass_air(400.0), expected_molar_mass)

    def test_calculate_molar_mass_air_different_co2(self, set_test_backend):
        """Test molar mass calculation with varying CO2."""
        expected_molar_mass_high_co2 = 1e-3 * (
            28.9635 + 12.011e-6 * (1000.0 - CO2_MOLAR_PPM)
        )
        assert_allclose(_calculate_molar_mass_air(1000.0), expected_molar_mass_high_co2)
        expected_molar_mass_low_co2 = 1e-3 * (
            28.9635 + 12.011e-6 * (0.0 - CO2_MOLAR_PPM)
        )
        assert_allclose(_calculate_molar_mass_air(0.0), expected_molar_mass_low_co2)

    def test_calculate_saturation_vapor_pressure_standard(self, set_test_backend):
        """Test SVP calculation at standard temperature."""
        # Value from Ciddor (1996), Table A1 for 20C
        expected_svp_20c = 2338.8  # Pa
        assert_allclose(
            _calculate_saturation_vapor_pressure(20.0), expected_svp_20c, rtol=1e-3
        )

    def test_calculate_saturation_vapor_pressure_edge_cases(self, set_test_backend):
        """Test SVP at temperature extremes."""
        # Freezing point
        assert_allclose(_calculate_saturation_vapor_pressure(0.0), 611.21262404)
        # Boiling point
        assert_allclose(_calculate_saturation_vapor_pressure(100.0), 101383.59550038)

    def test_calculate_saturation_vapor_pressure_no_where_backend(self):
        """Test SVP calculation fallback path for backends without `where`."""
        import optiland.backend as be_real

        class MockBackend:
            """A mock backend that has 'exp' but not 'where'."""

            @staticmethod
            def exp(x):
                return be_real.exp(x)

        with patch("optiland.environment.models.ciddor.be", MockBackend()):
            # Test case for t >= 0
            temp_pos = 20.0
            expected_pos = 2338.8
            assert_allclose(
                _calculate_saturation_vapor_pressure(temp_pos), expected_pos, rtol=1e-3
            )

            # Test case for t < 0
            temp_neg = -10.0
            expected_neg = 10.0 ** (-2663.5 / (temp_neg + 273.15) + 12.537)
            assert_allclose(
                _calculate_saturation_vapor_pressure(temp_neg), expected_neg
            )

    def test_calculate_enhancement_factor_standard(self, set_test_backend):
        """Test enhancement factor at standard conditions."""
        # Value from Ciddor (1996), Table A1 for 20C, 101325 Pa
        expected_f_20c_1atm = 1.00062 + (3.14e-8 * 101325.0) + (5.6e-7 * 20.0**2)
        assert_allclose(
            _calculate_enhancement_factor(101325.0, 20.0), expected_f_20c_1atm
        )

    def test_calculate_enhancement_factor_edge_cases(self, set_test_backend):
        """Test enhancement factor at pressure/temperature extremes."""
        # Zero pressure
        assert_allclose(_calculate_enhancement_factor(0.0, 0.0), ALPHA_F)
        # High pressure, high temperature
        high_p = 200000.0
        high_t = 50.0
        expected_f = ALPHA_F + BETA_F * high_p + GAMMA_F * high_t**2
        assert_allclose(_calculate_enhancement_factor(high_p, high_t), expected_f)

    def test_calculate_compressibility_standard(self, set_test_backend):
        """Test compressibility at standard conditions (dry air)."""
        # Value from Ciddor (1996), Table A1 for 20C, 101325 Pa, 0 mol fraction
        expected_z_20c_1atm_dry = (
            1.0
            - (101325.0 / 293.15) * (A0_Z + A1_Z * 20.0 + A2_Z * 20.0**2)
            + (101325.0 / 293.15) ** 2 * D_Z
        )
        assert_allclose(
            _calculate_compressibility(101325.0, 293.15, 0.0), expected_z_20c_1atm_dry
        )

    def test_calculate_compressibility_moist_air(self, set_test_backend):
        """Test compressibility with water vapor."""
        # Example with some molar fraction of H2O
        p_pa = 100000.0
        t_k = 290.0
        t_c = t_k - 273.15
        xw = 0.01  # 1% molar fraction
        expected_z = (
            1.0
            - (p_pa / t_k)
            * (
                A0_Z
                + A1_Z * t_c
                + A2_Z * t_c**2
                + (B0_Z + B1_Z * t_c) * xw
                + (C0_Z + C1_Z * t_c) * xw**2
            )
            + (p_pa / t_k) ** 2 * (D_Z + E_Z * xw**2)
        )
        assert_allclose(_calculate_compressibility(p_pa, t_k, xw), expected_z)

    # --- Test Cases for ciddor_refractive_index ---

    def test_ciddor_refractive_index_standard_conditions(self, set_test_backend):
        """
        Test Ciddor model with standard conditions and reference wavelength.
        Compares against Ciddor (1996) Table 1 value.
        """
        n_ciddor = ciddor_refractive_index(self.REF_WAVELENGTH_UM, self.std_conditions)
        assert_allclose(n_ciddor - 1.0, self.REF_N_MINUS_1_CIDDOR)

    def test_ciddor_refractive_index_dry_air_different_co2(self, set_test_backend):
        """
        Test Ciddor model for dry air with varying CO2 concentration.
        """
        conditions_high_co2 = EnvironmentalConditions(
            temperature=self.STD_TEMP_C,
            pressure=self.STD_PRES_PA,
            relative_humidity=0.0,
            co2_ppm=1000.0,  # Higher CO2
        )
        n_high_co2 = ciddor_refractive_index(
            self.REF_WAVELENGTH_UM, conditions_high_co2
        )
        # Expect higher refractive index with higher CO2
        assert n_high_co2 > (1.0 + self.REF_N_MINUS_1_CIDDOR)

        conditions_low_co2 = EnvironmentalConditions(
            temperature=self.STD_TEMP_C,
            pressure=self.STD_PRES_PA,
            relative_humidity=0.0,
            co2_ppm=0.0,  # No CO2
        )
        n_low_co2 = ciddor_refractive_index(self.REF_WAVELENGTH_UM, conditions_low_co2)
        # Expect lower refractive index with lower CO2
        assert n_low_co2 < (1.0 + self.REF_N_MINUS_1_CIDDOR)

    def test_ciddor_refractive_index_moist_air(self, set_test_backend):
        """
        Test Ciddor model with significant humidity.
        Humidity should decrease the refractive index.
        """
        moist_conditions = EnvironmentalConditions(
            temperature=20.0,
            pressure=101325.0,
            relative_humidity=0.8,  # 80% RH
            co2_ppm=400.0,
        )
        dry_conditions = EnvironmentalConditions(
            temperature=20.0, pressure=101325.0, relative_humidity=0.0, co2_ppm=400.0
        )
        n_moist = ciddor_refractive_index(0.55, moist_conditions)
        n_dry = ciddor_refractive_index(0.55, dry_conditions)
        assert n_moist < n_dry

    def test_ciddor_refractive_index_temperature_effect(self, set_test_backend):
        """
        Test Ciddor model with varying temperature.
        Higher temperature should decrease the refractive index (lower density).
        """
        cold_conditions = EnvironmentalConditions(
            temperature=0.0,
            pressure=self.STD_PRES_PA,
            relative_humidity=self.STD_RH,
            co2_ppm=self.STD_CO2_PPM,
        )
        hot_conditions = EnvironmentalConditions(
            temperature=30.0,
            pressure=self.STD_PRES_PA,
            relative_humidity=self.STD_RH,
            co2_ppm=self.STD_CO2_PPM,
        )
        n_cold = ciddor_refractive_index(self.REF_WAVELENGTH_UM, cold_conditions)
        n_hot = ciddor_refractive_index(self.REF_WAVELENGTH_UM, hot_conditions)
        assert n_cold > n_hot

    def test_ciddor_refractive_index_pressure_effect(self, set_test_backend):
        """
        Test Ciddor model with varying pressure.
        Higher pressure should increase the refractive index (higher density).
        """
        low_pressure_conditions = EnvironmentalConditions(
            temperature=self.STD_TEMP_C,
            pressure=50000.0,  # Half atmospheric pressure
            relative_humidity=self.STD_RH,
            co2_ppm=self.STD_CO2_PPM,
        )
        high_pressure_conditions = EnvironmentalConditions(
            temperature=self.STD_TEMP_C,
            pressure=200000.0,  # Double atmospheric pressure
            relative_humidity=self.STD_RH,
            co2_ppm=self.STD_CO2_PPM,
        )
        n_low_p = ciddor_refractive_index(
            self.REF_WAVELENGTH_UM, low_pressure_conditions
        )
        n_high_p = ciddor_refractive_index(
            self.REF_WAVELENGTH_UM, high_pressure_conditions
        )
        assert n_low_p < n_high_p

    def test_ciddor_refractive_index_wavelength_effect(self, set_test_backend):
        """
        Test Ciddor model with varying wavelength (dispersion).
        Shorter wavelength (higher sigma_sq) should increase refractive index.
        """
        n_blue = ciddor_refractive_index(0.4, self.std_conditions)  # Blue light
        n_red = ciddor_refractive_index(0.7, self.std_conditions)  # Red light
        assert n_blue > n_red

    def test_ciddor_refractive_index_edge_wavelengths(self, set_test_backend):
        """Test Ciddor model with very short and very long wavelengths."""
        # Very short wavelength (UV)
        n_uv = ciddor_refractive_index(0.1, self.std_conditions)
        assert n_uv > 1.0  # Should still be valid, just higher

        # Very long wavelength (IR)
        n_ir = ciddor_refractive_index(10.0, self.std_conditions)
        assert n_ir > 1.0  # Should be closer to 1.0

    def test_ciddor_refractive_index_zero_pressure(self, set_test_backend):
        """Test Ciddor model at zero pressure (vacuum). Should return ~1.0."""
        conditions_vacuum = EnvironmentalConditions(
            temperature=self.STD_TEMP_C,
            pressure=0.0,
            relative_humidity=self.STD_RH,
            co2_ppm=self.STD_CO2_PPM,
        )
        n_vacuum = ciddor_refractive_index(self.REF_WAVELENGTH_UM, conditions_vacuum)
        assert_allclose(n_vacuum, 1.0, atol=1e-9)

    def test_ciddor_refractive_index_invalid_wavelength(self, set_test_backend):
        """Test Ciddor model with non-positive wavelength."""
        with pytest.raises(ValueError, match="Wavelength must be positive."):
            ciddor_refractive_index(0.0, self.std_conditions)
        with pytest.raises(ValueError, match="Wavelength must be positive."):
            ciddor_refractive_index(-0.5, self.std_conditions)

    def test_ciddor_refractive_index_invalid_conditions_type(self, set_test_backend):
        """Test Ciddor model with incorrect conditions type."""
        with pytest.raises(
            TypeError, match="conditions must be an EnvironmentalConditions object."
        ):
            ciddor_refractive_index(self.REF_WAVELENGTH_UM, "not_a_condition_object")

    def test_ciddor_refractive_index_high_humidity_extreme_temp(self, set_test_backend):
        """Test Ciddor model with high humidity and extreme temperature."""
        conditions = EnvironmentalConditions(
            temperature=40.0, pressure=100000.0, relative_humidity=0.99, co2_ppm=400.0
        )
        n = ciddor_refractive_index(0.55, conditions)
        assert n > 1.0

    def test_ciddor_refractive_index_low_temp_high_pressure(self, set_test_backend):
        """Test Ciddor model with low temperature and high pressure."""
        conditions = EnvironmentalConditions(
            temperature=-20.0, pressure=120000.0, relative_humidity=0.1, co2_ppm=450.0
        )
        n = ciddor_refractive_index(0.6, conditions)
        assert n > 1.0

    def test_ciddor_refractive_index_consistency_with_example(self, set_test_backend):
        """Re-verify with the example provided in the module docstring."""
        conditions_example = EnvironmentalConditions(
            temperature=15.0, pressure=101325.0, relative_humidity=0.0, co2_ppm=400.0
        )
        # The example uses 0.55um, and the output is 1.000277641
        n_example = ciddor_refractive_index(0.55, conditions_example)
        assert_allclose(n_example, 1.000277641)
