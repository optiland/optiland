"""Unified Interface for Air Refractive Index Models.

This module provides a high-level dispatcher function, `refractive_index_air`,
to conveniently access various models for calculating the refractive index of
air. This allows users to easily switch between models based on their specific
requirements for accuracy, environmental conditions, or consistency with other
tools.

The supported models include:

- Ciddor (1996): A highly accurate model for a wide range of wavelengths and
  atmospheric conditions.
- Edlén (1966): A widely used model, implemented here with the NIST
  temperature correction for the water vapor term to improve accuracy.
- Birch & Downs (1994): A revised Edlén-style equation, also implemented
  with the NIST temperature correction for the water vapor term.
- Kohlrausch: A model based on the formula used in Zemax OpticStudio.

Kramer Harrison, 2025
"""

from __future__ import annotations

from .conditions import EnvironmentalConditions
from .models.birch_downs import birch_downs_refractive_index
from .models.ciddor import ciddor_refractive_index
from .models.edlen import edlen_refractive_index
from .models.kohlrausch import kohlrausch_refractive_index


def refractive_index_air(
    wavelength_um: float, conditions: EnvironmentalConditions, model: str = "ciddor"
) -> float:
    """Calculates the refractive index of air using a specified model.

    This function acts as a dispatcher to various air refractive index models.
    The environmental conditions (temperature, pressure, humidity, CO2) are
    encapsulated in the `conditions` object.

    Args:
        wavelength_um: The wavelength of light in a vacuum, in micrometers (μm).
        conditions: An `EnvironmentalConditions` object containing the
            environmental parameters (temperature, pressure, humidity, CO2).
        model: The model to use for the calculation. Supported models are:
            'ciddor', 'edlen', 'birch_downs', 'kohlrausch'. This is
            case-insensitive. Defaults to 'ciddor'.

    Returns:
        The calculated refractive index of air.

    Raises:
        ValueError: If an unsupported model is specified.
        TypeError: If `conditions` is not an `EnvironmentalConditions` object.

    Example:
        >>> from optiland.environment import EnvironmentalConditions
        >>> conditions_std = EnvironmentalConditions(
        ...     temperature=15.0,
        ...     pressure=101325.0,
        ...     relative_humidity=0.0,
        ...     co2_ppm=400.0,
        ... )
        >>> n_ciddor = refractive_index_air(0.55, conditions_std, model="ciddor")
        >>> print(f"Refractive index at 0.55 µm is {n_ciddor:.8f}")
        Refractive index at 0.55 µm is 1.00027764
    """
    if not isinstance(conditions, EnvironmentalConditions):
        raise TypeError(
            "Input 'conditions' must be an instance of EnvironmentalConditions."
        )

    model_lower = model.lower()
    if model_lower == "ciddor":
        return ciddor_refractive_index(wavelength_um, conditions)
    elif model_lower == "edlen":
        return edlen_refractive_index(wavelength_um, conditions)
    elif model_lower == "birch_downs":
        return birch_downs_refractive_index(wavelength_um, conditions)
    elif model_lower == "kohlrausch":
        return kohlrausch_refractive_index(wavelength_um, conditions)
    else:
        raise ValueError(
            f"Unsupported air refractive index model: {model}. "
            "Supported models are: 'ciddor', 'edlen', 'birch_downs', 'kohlrausch'."
        )
