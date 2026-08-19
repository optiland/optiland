"""Unit tests for the EnvironmentalConditions dataclass."""

from __future__ import annotations

import pytest

from optiland.environment.conditions import EnvironmentalConditions


def test_environmental_conditions_defaults():
    """Test that EnvironmentalConditions initializes with correct default values."""
    conditions = EnvironmentalConditions()
    assert conditions.pressure == 101325.0
    assert conditions.temperature == 15.0
    assert conditions.relative_humidity == 0.0
    assert conditions.co2_ppm == 400.0
    assert conditions.wavelength is None


def test_environmental_conditions_custom_values():
    """Test that EnvironmentalConditions correctly assigns custom values."""
    conditions = EnvironmentalConditions(
        pressure=100000.0,
        temperature=20.0,
        relative_humidity=0.5,
        co2_ppm=350.0,
        wavelength=0.55,
    )
    assert conditions.pressure == 100000.0
    assert conditions.temperature == 20.0
    assert conditions.relative_humidity == 0.5
    assert conditions.co2_ppm == 350.0
    assert conditions.wavelength == 0.55


def test_environmental_conditions_partial_custom_values():
    """Test initializing with only some custom values, others default."""
    conditions = EnvironmentalConditions(temperature=25.0, co2_ppm=500.0)
    assert conditions.pressure == 101325.0  # Default
    assert conditions.temperature == 25.0  # Custom
    assert conditions.relative_humidity == 0.0  # Default
    assert conditions.co2_ppm == 500.0  # Custom
    assert conditions.wavelength is None  # Default


@pytest.mark.parametrize(
    "pressure, temperature, rh, co2, wl",
    [
        (101325, 15, 0.0, 400, None),
        (90000, 0, 0.25, 300, 0.633),
        (110000, 30, 1.0, 800, 1.55),
    ],
)
def test_environmental_conditions_parametrization(pressure, temperature, rh, co2, wl):
    """Test EnvironmentalConditions with various valid parameter sets."""
    conditions = EnvironmentalConditions(
        pressure=pressure,
        temperature=temperature,
        relative_humidity=rh,
        co2_ppm=co2,
        wavelength=wl,
    )
    assert conditions.pressure == pressure
    assert conditions.temperature == temperature
    assert conditions.relative_humidity == rh
    assert conditions.co2_ppm == co2
    assert conditions.wavelength == wl


def test_environmental_conditions_types():
    """Test that attributes maintain their types."""
    conditions = EnvironmentalConditions(
        pressure=100000.0,
        temperature=20.0,
        relative_humidity=0.5,
        co2_ppm=350.0,
        wavelength=0.55,
    )
    assert isinstance(conditions.pressure, float)
    assert isinstance(conditions.temperature, float)
    assert isinstance(conditions.relative_humidity, float)
    assert isinstance(conditions.co2_ppm, float)
    assert isinstance(conditions.wavelength, float)

    conditions_int = EnvironmentalConditions(
        pressure=100000,
        temperature=20,
        relative_humidity=0,  # will be float due to default
        co2_ppm=350,
    )
    # Dataclasses with type hints will try to convert or use as is.
    # Python's float() can take int.
    assert isinstance(conditions_int.pressure, int)  # Will be int if passed as int
    assert isinstance(conditions_int.temperature, int)  # Will be int if passed as int
    assert isinstance(
        conditions_int.relative_humidity, int
    )  # Will be int if passed as int
    assert isinstance(conditions_int.co2_ppm, int)  # Will be int if passed as int

    # Explicitly set RH to int to check
    conditions_int_rh = EnvironmentalConditions(relative_humidity=0)
    assert isinstance(conditions_int_rh.relative_humidity, int)

    conditions_none_wl = EnvironmentalConditions()
    assert conditions_none_wl.wavelength is None


def test_environmental_conditions_mutability():
    """Test that attributes can be mutated after instantiation."""
    conditions = EnvironmentalConditions()
    conditions.temperature = 25.0
    conditions.co2_ppm = 600.0
    assert conditions.temperature == 25.0
    assert conditions.co2_ppm == 600.0
    # Ensure others remain default
    assert conditions.pressure == 101325.0
    assert conditions.relative_humidity == 0.0
    assert conditions.wavelength is None
