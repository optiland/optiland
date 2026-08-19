"""Edlén (1966) Air Refractive Index Model (with NIST Modification).

This module provides functions to calculate the refractive index of air based
on the seminal Edlén (1966) formulation. This implementation includes the
NIST-recommended temperature correction for the water vapor term, which
improves accuracy over a wider range of temperatures.

The original Edlén formulas require pressure in Torr (mmHg). This implementation
accepts environmental conditions in SI units (Pascals, Celsius) and performs
the necessary conversions internally.

References:
    - Edlén, B. (1966). The Refractive Index of Air. Metrologia, 2(2), 71-80.
    - Stone, J. A., & Zimmerman, J. H. (2001). Index of Refraction of Air
      (NIST Web Page). https://emtoolbox.nist.gov/Wavelength/Documentation.asp

Kramer Harrison, 2025
"""

from __future__ import annotations

import optiland.backend as be

from ..conditions import EnvironmentalConditions

# --- Model Constants from Edlén (1966) ---

# Standard air is defined at 15 °C, 760 Torr, and 0.03% (300 ppm) CO₂.
CO2_STD_PPM = 300.0

# Dispersion constants for standard air.
# Source: Edlén (1966), Eq. (1) [cite: 631, 632]
DISP_A = 8342.13
DISP_B = 2406030.0
DISP_C = 130.0
DISP_D = 15997.0
DISP_E = 38.9

# Density factor for standard air.
# Source: Edlén (1966), text preceding Eq. (12) [cite: 739]
DENSITY_FACTOR_STD = 720.775

# Thermal expansion coefficient for air.
# Source: Edlén (1966), Eq. (13) [cite: 743]
ALPHA_GAS = 0.0036710  # K⁻¹

# CO₂ correction factor.
# Source: Edlén (1966), Eq. (17)
# Converts from standard air (300 ppm) to air with x ppm CO₂.
# Factor is applied to (x - 0.0003), where x is fractional volume.
# For x_ppm, this is (x_ppm - 300) * 1e-6.
CO2_CORR_FACTOR = 0.540

# Water vapor correction constants.
# Source: Edlén (1966), Eq. (22) [cite: 848]
WATER_VAPOR_A = 5.722
WATER_VAPOR_B = 0.0457

# Unit Conversion
# Standard atmosphere (101325 Pa) is equivalent to 760 Torr.
TORR_TO_PA = 101325.0 / 760.0


def _calculate_saturation_vapor_pressure(temperature_c: float) -> float:
    """Calculates saturation vapor pressure of water in Pascals.

    Edlén's 1966 paper does not specify a formula, as it was common to use
    lookup tables. This implementation uses the Buck (1981) equation, a
    well-regarded approximation, to determine the saturation vapor pressure.

    Args:
        temperature_c: The temperature in degrees Celsius.

    Returns:
        The saturation vapor pressure in Pascals (Pa).
    """
    # Using the Buck (1981) equation for SVP over water.
    return 611.21 * be.exp(
        (18.678 - temperature_c / 234.5) * (temperature_c / (257.14 + temperature_c))
    )


def edlen_refractive_index(
    wavelength_um: float, conditions: EnvironmentalConditions
) -> float:
    """Calculates refractive index using the modified Edlén (1966) model.

    This implementation uses the Edlén (1966) formula with the NIST
    temperature correction for the water vapor term to improve accuracy.

    Args:
        wavelength_um: The wavelength of light in a vacuum, in micrometers (μm).
        conditions: An `EnvironmentalConditions` object containing the
            temperature, pressure, relative humidity, and CO₂ concentration.

    Returns:
        The phase refractive index of air (n).

    Raises:
        ValueError: If wavelength is not positive.
        TypeError: If conditions is not an `EnvironmentalConditions` object.

    Example:
        >>> from optiland.environment import EnvironmentalConditions
        >>> nist_conditions = EnvironmentalConditions(
        ...     temperature=20.0,
        ...     pressure=101325.0,
        ...     relative_humidity=0.0,
        ...     co2_ppm=450.0,
        ... )
        >>> n = edlen_refractive_index(0.633, nist_conditions)
        >>> print(f"Refractive index at 0.633 µm is {n:.8f}")
        Refractive index at 0.633 µm is 1.00027176
    """
    if not isinstance(conditions, EnvironmentalConditions):
        raise TypeError("conditions must be an EnvironmentalConditions object.")
    if not be.all(wavelength_um > 0):
        raise ValueError("Wavelength must be positive.")

    # --- 1. Calculate vacuum wavenumber and standard refractivity ---
    sigma_sq = (1.0 / wavelength_um) ** 2

    # Refractivity of standard air (n_s - 1) at 15°C, 760 Torr, 300 ppm CO₂.
    # Source: Edlén (1966), Eq. (1) [cite: 631, 632]
    n_s_minus_1 = 1.0e-8 * (
        DISP_A + DISP_B / (DISP_C - sigma_sq) + DISP_D / (DISP_E - sigma_sq)
    )

    # --- 2. Correct standard refractivity for actual CO₂ concentration ---
    # Source: Edlén (1966), Eq. (17)
    co2_factor = 1.0 + CO2_CORR_FACTOR * (conditions.co2_ppm - CO2_STD_PPM) * 1.0e-6
    n_s_corrected_minus_1 = n_s_minus_1 * co2_factor

    # --- 3. Correct for actual temperature and pressure (for dry air) ---
    p_torr = conditions.pressure / TORR_TO_PA
    t_c = conditions.temperature

    # Using the full formula for the density factor for dry air.
    # Source: Edlén (1966), Eq. (12)
    density_factor_actual = (
        p_torr * (1.0 + p_torr * (0.817 - 0.0133 * t_c) * 1.0e-6)
    ) / (1.0 + ALPHA_GAS * t_c)

    n_tp_minus_1 = n_s_corrected_minus_1 * (density_factor_actual / DENSITY_FACTOR_STD)

    # --- 4. Calculate and apply the correction for water vapor ---
    svp_pa = _calculate_saturation_vapor_pressure(t_c)
    f_pa = conditions.relative_humidity * svp_pa  # Partial pressure in Pa
    f_torr = f_pa / TORR_TO_PA

    # This term is the difference (n_moist - n_dry) and is added to the
    # dry air refractivity.
    # Source: Edlén (1966), Eq. (22) [cite: 461, 848], with NIST modification.
    water_vapor_correction_unscaled = (
        -f_torr * (WATER_VAPOR_A - WATER_VAPOR_B * sigma_sq) * 1.0e-8
    )
    # NIST modification for temperature dependence of water vapor term.
    temp_correction = 292.75 / (t_c + 273.15)
    water_vapor_correction = water_vapor_correction_unscaled * temp_correction

    # --- 5. Combine terms for the final refractive index ---
    n_final_minus_1 = n_tp_minus_1 + water_vapor_correction
    return 1.0 + n_final_minus_1
