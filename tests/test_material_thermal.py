from __future__ import annotations

import optiland.backend as be
from optiland import materials

from .utils import assert_allclose


class TestMaterialFileThermal:
    """
    Tests the temperature and pressure dependent calculations in MaterialFile
    using a real material (N-BK7) from the database.
    """

    def test_nair_calculation(self, set_test_backend):
        """
        Verifies the _nair method calculates the refractive index of air correctly.
        This test is independent of the material but needs an instance to call the method.
        """
        material = materials.Material("N-BK7")
        # Test parameters
        wavelength_um = 0.55
        temp_c = 25.0
        pressure_atm = 1.2

        # --- Manual Calculation ---
        w2 = wavelength_um**2
        n_ref_minus_1 = (
            6432.8 + (2949810 * w2) / (146 * w2 - 1) + (25540 * w2) / (41 * w2 - 1)
        ) * 1e-8
        air_thermal_coeff = 0.0034785
        air_ref_temp_c = 15.0
        denominator = 1.0 + (temp_c - air_ref_temp_c) * air_thermal_coeff
        expected_n_air = 1.0 + (n_ref_minus_1 * pressure_atm) / denominator

        # --- Comparison ---
        calculated_n_air = material._nair(wavelength_um, temp_c, pressure_atm)
        assert_allclose(expected_n_air, calculated_n_air)

    def test_no_correction_if_temp_is_none(self, set_test_backend):
        """
        Tests that if temperature is None, no correction is applied and the
        catalog refractive index is returned.
        """
        material = materials.Material("N-BK7")
        wavelength = 0.55
        # Expected value for N-BK7 at 0.55µm without corrections
        expected_n = 1.518519

        calculated_n = material.n(wavelength, temperature=None, pressure=None)
        assert_allclose(expected_n, calculated_n)

    def test_no_correction_if_no_thermal_data(self, set_test_backend):
        """
        Tests that if the material has no thermal coefficients, no correction
        is applied even if temperature is provided.
        """
        material = materials.Material("N-BK7")
        # Manually remove thermal data from the loaded material object
        material.thermdispcoef = []
        wavelength = 0.55
        # Expected value for N-BK7 at 0.55µm without corrections
        expected_n = 1.518519

        calculated_n = material.n(wavelength, temperature=30.0, pressure=1.2)
        assert_allclose(expected_n, calculated_n)

    def test_correction_with_temp_and_default_pressure(self, set_test_backend):
        """
        Tests that if pressure is None, it defaults to 1.0 atm.
        """
        material = materials.Material("N-BK7")
        wavelength = 0.55
        temp_c = 30.0

        # Calculate with explicit pressure of 1.0
        n_with_pressure = material.n(wavelength, temperature=temp_c, pressure=1.0)
        # Calculate with pressure=None
        n_with_none_pressure = material.n(wavelength, temperature=temp_c, pressure=None)

        assert n_with_pressure == n_with_none_pressure

    def test_correction_at_reference_temp(self, set_test_backend):
        """
        Tests that at the material's reference temperature (T == t0), the change
        in absolute index (dn_abs) is zero and the result is the same as no correction.
        """
        material = materials.Material("N-BK7")
        wavelength = 0.55
        # Use the material's reference temperature
        temp_c = material._t0  # 20.0 C
        pressure_atm = 1.0

        # Get the uncorrected value
        base_n = material.n(wavelength, temperature=None)
        # Get the value corrected at the reference temperature
        calculated_n = material.n(wavelength, temperature=temp_c, pressure=pressure_atm)

        # The result should be the same as the base refractive index
        assert_allclose(calculated_n, base_n)

    def test_full_correction_with_array_input(self, set_test_backend):
        """
        Verifies the full environmental correction with a backend array of wavelengths.
        """
        material = materials.Material("N-BK7")
        wavelengths = be.array([0.5, 0.55, 0.6])
        temp_c = 30.0
        pressure_atm = 1.2

        # Calculate for each wavelength individually
        n1 = material.n(wavelengths[0], temperature=temp_c, pressure=pressure_atm)
        n2 = material.n(wavelengths[1], temperature=temp_c, pressure=pressure_atm)
        n3 = material.n(wavelengths[2], temperature=temp_c, pressure=pressure_atm)
        expected_n_array = be.array(
            [be.to_numpy(n1).item(), be.to_numpy(n2).item(), be.to_numpy(n3).item()]
        )

        # Calculate with the array directly
        calculated_n_array = material.n(
            wavelengths, temperature=temp_c, pressure=pressure_atm
        )

        assert_allclose(calculated_n_array, expected_n_array)

    def test_full_correction_calculation(self, set_test_backend):
        """
        Verifies the full environmental correction calculation for N-BAF4 against a
        step-by-step manual calculation.
        """
        material = materials.Material("N-BAF4")
        # --- Test Parameters ---
        wavelength = 0.55
        temp_c = 30.0
        pressure_atm = 1.2

        # --- Manual Calculation Steps ---
        # 1. Air indices
        n_air_system = material._nair(wavelength, temp_c, pressure_atm)
        n_air_reference = material._nair(wavelength, material._t0, 1.0)

        # 2. Relative wavelength
        waverel = wavelength * n_air_system / n_air_reference

        # 3. Base relative index (using formula 2 for N-BAF4)
        c = material.coefficients
        wl2 = waverel**2
        n2 = (
            1.0
            + c[1] * wl2 / (wl2 - c[2])
            + c[3] * wl2 / (wl2 - c[4])
            + c[5] * wl2 / (wl2 - c[6])
        )
        base_relative_n = be.sqrt(n2)

        # 4. Absolute index at reference
        n_absolute_reference = base_relative_n * n_air_reference

        # 5. Change in absolute index (dn_abs)
        c_therm = material.thermdispcoef
        delta_t = temp_c - material._t0
        n_sq_minus_1 = n_absolute_reference**2 - 1.0
        two_n = 2.0 * n_absolute_reference
        term1 = c_therm[0] + c_therm[1] * delta_t + c_therm[2] * delta_t**2
        term2 = (c_therm[3] + c_therm[4] * delta_t) / (wavelength**2 - c_therm[5] ** 2)
        dn_abs = (n_sq_minus_1 / two_n) * (term1 + term2) * delta_t

        # 6. Corrected absolute index
        n_absolute_corrected = n_absolute_reference + dn_abs

        # 7. Final relative index
        expected_final_n = n_absolute_corrected / n_air_system
        # expected_final_n should be ~1.518490

        # --- Comparison ---
        calculated_n = material.n(wavelength, temperature=temp_c, pressure=pressure_atm)
        assert_allclose(calculated_n, expected_final_n)
