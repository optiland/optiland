"""Kohlrausch Air Refractive Index Model (Zemax Implementation).

This module provides a function to calculate the refractive index of air
according to the specific formula used by Zemax OpticStudio. This formula is
attributed to F. Kohlrausch, but the constants used are specific to the
Zemax implementation.

The model consists of a Sellmeier-type dispersion equation for a reference
condition and a linearized scaling factor to account for variations in
temperature and pressure. It is a model for dry air and does not explicitly
account for humidity or CO₂ variations.

References:
    - Ansys Optics. (2023). How OpticStudio calculates refractive index at
      arbitrary temperatures and pressures.
      https://optics.ansys.com/hc/en-us/articles/42661799783443-How-OpticStudio-calculates-refractive-index-at-arbitrary-temperatures-and-pressures

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..conditions import EnvironmentalConditions

# --- Model Constants from the Kohlrausch Formula ---

# Constants for the reference dispersion formula (n_ref), from Zemax documentation.
# This is a Sellmeier-2 formula written as:
# (n_ref - 1) * 10^5 = A + B / (C - σ²) + D / (E - σ²)
# where σ = 1/λ is the wavenumber in μm⁻¹.
DISP_A = 64.328
DISP_B = 29498.1
DISP_C = 146.0  # In μm⁻²
DISP_D = 25.54
DISP_E = 41.0  # In μm⁻²

# Reference and scaling constants for the pressure/temperature correction.
T_REF_C = 15.0  # Reference temperature in degrees Celsius.
P_STD_PA = 101325.0  # Standard atmospheric pressure in Pascals (1 atm).

# Linearized thermal coefficient for air, from Zemax documentation.
ALPHA_T = 0.00348  # In °C⁻¹


def kohlrausch_refractive_index(
    wavelength_um: float, conditions: EnvironmentalConditions
) -> float:
    """Calculates air refractive index using the Kohlrausch (Zemax) formula.

    This model ignores the humidity and CO₂ concentration from the conditions
    object, as it is designed for dry air.

    Args:
        wavelength_um: Wavelength of light in a vacuum, in micrometers (μm).
        conditions: An `EnvironmentalConditions` object. Only temperature and
            pressure are used.

    Returns:
        The refractive index of air (n).

    Raises:
        ValueError: If the wavelength is zero or if the temperature results in
            a non-positive denominator in the scaling term.

    Example:
        >>> from optiland.environment import EnvironmentalConditions
        >>> conditions_std = EnvironmentalConditions(
        ...     temperature=15.0,
        ...     pressure=101325.0
        ... )
        >>> n = kohlrausch_refractive_index(0.55, conditions_std)
        >>> print(f"Refractive index at 0.55 µm is {n:.8f}")
        Refractive index at 0.55 µm is 1.00271728
    """
    # --- 1. Calculate the reference refractivity (n_ref - 1) ---
    # The formula uses λ directly, but we convert it to the standard Sellmeier
    # form which uses wavenumber σ = 1/λ for clarity and robustness.
    try:
        sigma_sq = (1.0 / wavelength_um) ** 2
    except ZeroDivisionError as err:
        raise ValueError("Wavelength must be non-zero.") from err

    # Calculate (n_ref - 1) * 10^5
    n_ref_minus_1_e5 = (
        DISP_A + DISP_B / (DISP_C - sigma_sq) + DISP_D / (DISP_E - sigma_sq)
    )
    n_ref_minus_1 = n_ref_minus_1_e5 * 1.0e-5

    # --- 2. Apply temperature and pressure scaling ---
    t_c = conditions.temperature
    p_pa = conditions.pressure

    # P is relative pressure (dimensionless).
    relative_pressure = p_pa / P_STD_PA

    # Denominator of the scaling term.
    temp_scaling_denom = 1.0 + (t_c - T_REF_C) * ALPHA_T
    if temp_scaling_denom <= 0:
        raise ValueError(
            f"Invalid temperature {t_c}°C results in non-positive denominator."
        )

    # Final calculation for n_air
    n_air = 1.0 + (n_ref_minus_1 * relative_pressure) / temp_scaling_denom

    return n_air
