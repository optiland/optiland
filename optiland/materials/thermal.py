"""Catalog thermal dispersion, with wavelengths in µm, Celsius and atmospheres."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.environment.conditions import EnvironmentalConditions
from optiland.environment.models.kohlrausch import P_STD_PA, kohlrausch_refractive_index

if TYPE_CHECKING:
    from collections.abc import Callable

    from optiland.materials.definition import ThermalModel


def catalog_air_index(wavelength_um: Any, temp_c: Any, pressure_atm: Any = 1.0) -> Any:
    """The existing catalog Edlén convention, independent of global environment.

    Pressure is in atmospheres, unlike the environment module's pascals. Keep
    this convention stable when applying catalog thermal coefficients.
    """
    conditions = EnvironmentalConditions(
        temperature=temp_c, pressure=pressure_atm * P_STD_PA
    )
    return kohlrausch_refractive_index(wavelength_um, conditions)


def schott_relative_index(
    base_relative_n: Any,
    wavelength_um: Any,
    coefficients: Any,
    reference_temperature_c: Any,
    temperature_c: Any,
    pressure_atm: Any,
) -> Any:
    """Apply absolute Schott dn then express the result relative to ambient air.

    The caller evaluates base_relative_n at the reference-relative wavelength.
    Separating that evaluation preserves each adapter's wavelength policy.
    """
    c, w = coefficients, wavelength_um
    absolute = base_relative_n * catalog_air_index(w, reference_temperature_c)
    dt = temperature_c - reference_temperature_c
    change = (
        (absolute**2 - 1)
        / (2 * absolute)
        * dt
        * (c[0] + c[1] * dt + c[2] * dt**2 + (c[3] + c[4] * dt) / (w**2 - c[5] ** 2))
    )
    return (absolute + change) / catalog_air_index(w, temperature_c, pressure_atm)


def thermal_index(
    dispersion: Callable[[Any], Any],
    wavelength: Any,
    model: ThermalModel,
    temperature: Any,
    pressure: Any,
) -> Any:
    """Evaluate an owned thermal definition without changing its air convention."""
    temp = be.asarray(temperature)
    pressure = be.asarray(1.0 if pressure is None else pressure)
    if (
        not be.all(be.isfinite(temp))
        or be.any(temp <= -273.15)
        or not be.all(be.isfinite(pressure))
        or be.any(pressure < 0)
    ):
        raise ValueError("Invalid material temperature or pressure")
    wave, t0 = wavelength, model.reference_temperature_c
    c = be.asarray(model.coefficients)
    air = catalog_air_index(wave, temp, pressure)
    if model.kind == "schott":
        reference_air = catalog_air_index(wave, t0)
        baseline = dispersion(wave * air / reference_air)
        value = schott_relative_index(baseline, wave, c, t0, temp, pressure)
    else:
        slope = c[0]
        if model.kind == "corning":
            slope = slope + sum(c[i] / wave ** (2 * i) for i in range(1, 4))
        value = (
            (dispersion(wave) + slope * (temp - t0))
            * catalog_air_index(wave, temp)
            / air
        )
    if not be.all(be.isfinite(value)):
        raise ValueError("Invalid thermal dispersion: non-finite index")
    return value
