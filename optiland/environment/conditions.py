"""Dataclass for Environmental Conditions.

This module provides a convenient and type-safe dataclass,
`EnvironmentalConditions`, to encapsulate the environmental parameters required
for calculating the refractive index of air.

Using a dataclass ensures that all necessary parameters are provided in a
structured way, reducing the risk of errors and improving code readability.

Kramer Harrison, 2025
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnvironmentalConditions:
    """Stores environmental parameters for refractive index calculations.

    This dataclass holds all the necessary atmospheric conditions required by
    the various refractive index models.

    Attributes:
        pressure: Ambient air pressure in Pascals (Pa). Defaults to 101325.0 Pa,
            which is standard atmospheric pressure at sea level.
        temperature: Ambient air temperature in degrees Celsius (°C). Defaults
            to 15.0 °C, a common standard temperature for many models.
        relative_humidity: Relative humidity as a fraction (from 0.0 to 1.0).
            Defaults to 0.0 (dry air).
        co2_ppm: Carbon dioxide concentration in parts per million (ppm) by
            volume. Defaults to 400.0 ppm.
        wavelength: Optional. Wavelength of light in micrometers (μm). This
            attribute is not used by the core refractive index functions in
            this subpackage but is provided for convenience to store complete
            state information. Defaults to None.
    """

    pressure: float = 101325.0
    temperature: float = 15.0
    relative_humidity: float = 0.0
    co2_ppm: float = 400.0
    wavelength: float | None = None
