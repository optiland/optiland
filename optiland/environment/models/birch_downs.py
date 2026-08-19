"""Birch and Downs Air Refractive Index Model (with NIST Modification)

This module provides a function to calculate the refractive index of air based on
the revised Edlén-style equation published by K. P. Birch and M. J. Downs in
1994. This implementation includes a temperature-dependent correction to the
water vapor term, as described in the NIST documentation for their "Modified
Edlén" equation, which is based on the Birch and Downs model.

The calculation is valid for standard air (N₂, O₂, Ar, CO₂) and can be
adjusted for varying temperature, pressure, humidity, and CO₂ concentration.

Example:
    >>> from optiland.environment import EnvironmentalConditions
    >>> conditions = EnvironmentalConditions(
    ...     temperature=20.0,
    ...     pressure=101325.0,
    ...     relative_humidity=0.5,
    ...     co2_ppm=450.0
    ... )
    >>> n = birch_downs_refractive_index(0.633, conditions)
    >>> print(f"Refractive index at 633 nm is {n:.8f}")
    Refractive index at 633 nm is 1.00027137

References:
    - Birch, K. P., & Downs, M. J. (1994). Correction to the Updated Edlén
      Equation for the Refractive Index of Air. Metrologia, 31(4), 315-316.
    - Stone, J. A., & Zimmerman, J. H. (2001). Index of Refraction of Air
      (NIST Web Page). https://emtoolbox.nist.gov/Wavelength/Documentation.asp

Kramer Harrison, 2025
"""

from __future__ import annotations

import optiland.backend as be

from ..conditions import EnvironmentalConditions

# --- Model Constants ---

# Standard environmental conditions for the dispersion formula.
# Source: Birch & Downs (1994).
P_STD_PA = 101325.0  # Standard pressure in Pascals.
T_STD_C = 15.0  # Standard temperature in degrees Celsius.
T_STD_K = 273.15 + T_STD_C  # Standard temperature in Kelvin.
CO2_STD_PPM = 450.0  # Standard CO₂ concentration in ppm for the 1994 revision.

# Dispersion constants for dry air at standard conditions (15°C, 101325 Pa,
# 450 ppm CO₂).
# Source: Birch & Downs (1994), Eq. (2).
# (n_s - 1) * 10^8 = A + B / (C - σ²) + D / (E - σ²)
DISPERSION_A = 8342.54
DISPERSION_B = 2406147.0
DISPERSION_C = 130.0
DISPERSION_D = 15998.0
DISPERSION_E = 38.9

# Water vapor refractivity constants.
# Source: Birch & Downs (1994), Eq. (3).
# n_tpf - n_tp = -f * [W1 - W2*σ²] * 10⁻¹⁰
WATER_VAPOR_A = 3.7345  # Unitless coefficient.
WATER_VAPOR_B = 0.0401  # In (μm⁻¹)^-2 or μm².

# CO₂ concentration correction factor.
# The 1994 Birch & Downs equation is baseline at 450 ppm. To adjust for other
# concentrations, a correction is needed. This multiplicative factor is derived
# from Ciddor (1996), which is a common and accepted practice.
# (n_adj - 1) = (n_450ppm - 1) * (1 + CO2_K * (x_ppm - 450))
CO2_CORRECTION_FACTOR = 0.534e-6  # In ppm⁻¹.


def _calculate_saturation_vapor_pressure(temperature_c: float) -> float:
    """Calculates the saturation vapor pressure of water in air.

    This uses a common approximation (based on the Ciddor (1996) paper's
    Appendix A) that is highly accurate for meteorological ranges.

    Args:
        temperature_c: The air temperature in degrees Celsius.

    Returns:
        The saturation vapor pressure in Pascals (Pa).
    """
    t_k = temperature_c + 273.15
    # Coefficients for the SVP formula.
    A = 1.2378847e-5  # K⁻²
    B = -1.9121316e-2  # K⁻¹
    C = 33.93711047
    D = -6.3431645e3  # K
    return be.exp(A * t_k**2 + B * t_k + C + D / t_k)


def _calculate_water_vapor_partial_pressure(
    conditions: EnvironmentalConditions,
) -> float:
    """Calculates the partial pressure of water vapor.

    This function includes the enhancement factor (f_w), which accounts for
    the non-ideal behavior of moist air. The formula for f_w is from
    Ciddor (1996).

    Args:
        conditions: The environmental conditions.

    Returns:
        The partial pressure of water vapor in Pascals (Pa).
    """
    saturation_pressure = _calculate_saturation_vapor_pressure(conditions.temperature)

    # Enhancement factor f_w for moist air.
    f_w = (
        1.00062 + (3.14e-8 * conditions.pressure) + (5.6e-7 * conditions.temperature**2)
    )

    return conditions.relative_humidity * f_w * saturation_pressure


def birch_downs_refractive_index(
    wavelength_um: float, conditions: EnvironmentalConditions
) -> float:
    """Calculates the refractive index of air using the Birch & Downs 1994 model.

    Args:
        wavelength_um: The wavelength of light in a vacuum, in micrometers (μm).
        conditions: An EnvironmentalConditions object containing the temperature,
            pressure, relative humidity, and CO₂ concentration.

    Returns:
        The refractive index of air (n).

    Raises:
        ValueError: If wavelength is not positive.
        TypeError: If conditions is not an EnvironmentalConditions object.
    """
    if not isinstance(conditions, EnvironmentalConditions):
        raise TypeError("conditions must be an EnvironmentalConditions object.")
    if not be.all(wavelength_um > 0):
        raise ValueError("Wavelength must be positive.")

    # 1. Calculate vacuum wavenumber (σ) in μm⁻¹.
    sigma_sq = (1.0 / wavelength_um) ** 2

    # 2. Calculate refractivity of standard dry air (n_s - 1) at 15 °C,
    # 101325 Pa, and 450 ppm CO₂ using Birch & Downs (1994), Eq. (2).
    n_s_minus_1_e8 = (
        DISPERSION_A
        + DISPERSION_B / (DISPERSION_C - sigma_sq)
        + DISPERSION_D / (DISPERSION_E - sigma_sq)
    )
    n_s_minus_1 = n_s_minus_1_e8 * 1.0e-8

    # 3. Correct the standard refractivity for the actual CO₂ concentration.
    # This adjusts from the standard 450 ppm to the measured value.
    co2_correction = 1.0 + CO2_CORRECTION_FACTOR * (conditions.co2_ppm - CO2_STD_PPM)
    n_as_minus_1 = n_s_minus_1 * co2_correction

    # 4. Convert the refractivity of dry air from standard conditions (n_as) to
    # the actual temperature and pressure (n_tp), using B&D (1994), Eq. (1).
    # This equation includes the non-ideal gas compressibility factor.
    t_c = conditions.temperature
    p_pa = conditions.pressure
    # The constant 96095.43 is from the original papers.
    density_term = (p_pa / 96095.43) * (
        (1 + 1e-8 * (0.601 - 0.00972 * t_c) * p_pa) / (1 + 0.003661 * t_c)
    )
    n_tp_minus_1 = n_as_minus_1 * density_term

    # 5. Calculate the correction due to water vapor (humidity).
    # This term is subtracted from the dry air refractivity.
    # Source: Birch & Downs (1994), Eq. (3), with NIST modification.
    f_pa = _calculate_water_vapor_partial_pressure(conditions)
    water_vapor_correction_unscaled = (
        -f_pa * (WATER_VAPOR_A - WATER_VAPOR_B * sigma_sq) * 1.0e-10
    )
    # NIST modification for temperature dependence of water vapor term.
    temp_correction = 292.75 / (t_c + 273.15)
    water_vapor_correction = water_vapor_correction_unscaled * temp_correction

    # 6. Calculate the final refractivity of moist air by applying the water
    # vapor correction to the dry air refractivity.
    # n_final - 1 = (n_tp - 1) + (n_tpf - n_tp)
    n_final_minus_1 = n_tp_minus_1 + water_vapor_correction

    return 1.0 + n_final_minus_1
