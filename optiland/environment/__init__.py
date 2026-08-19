"""Environment Subpackage for Refractive Index Calculations.

This package provides a comprehensive toolkit for calculating the refractive
index of air. It features multiple well-established empirical models and a
unified interface for easy access and comparison.

The main entry point is the `refractive_index_air` function, which dispatches
to the selected model. Each model is implemented in its own module and can
also be accessed directly.

Example:
    To calculate the refractive index of air for a given wavelength and set of
    environmental conditions using the Ciddor model:

    >>> from optiland.environment import (
    ...     EnvironmentalConditions,
    ...     refractive_index_air,
    ... )
    >>> conditions = EnvironmentalConditions(
    ...     temperature=15.0,
    ...     pressure=101325.0,
    ...     relative_humidity=0.0,
    ...     co2_ppm=400.0,
    ... )
    >>> n = refractive_index_air(0.55, conditions, model="ciddor")
    >>> print(f"Refractive index at 0.55 µm is {n:.8f}")
    Refractive index at 0.55 µm is 1.00027764

Modules:
    conditions: Defines the `EnvironmentalConditions` dataclass for specifying
        temperature, pressure, humidity, and CO2 concentration.
    air_index: Provides the main `refractive_index_air` dispatcher function.
    models: A subpackage containing the individual model implementations:
        - ciddor: Implements the highly accurate Ciddor (1996) model.
        - edlen: Implements the Edlén (1966) model with the NIST temperature
          correction for the water vapor term.
        - birch_downs: Implements the Birch & Downs (1994) model, also with
          the NIST temperature correction for the water vapor term.
        - kohlrausch: Implements the Kohlrausch model as used in Zemax
          OpticStudio.

Key Exports:
    EnvironmentalConditions: Dataclass for specifying environmental parameters.
    refractive_index_air: Unified function to calculate air refractive index.
    ciddor_refractive_index: Direct access to the Ciddor model.
    edlen_refractive_index: Direct access to the modified Edlén model.
    birch_downs_refractive_index: Direct access to the modified Birch & Downs
        model.
    kohlrausch_refractive_index: Direct access to the Kohlrausch model.
"""

from __future__ import annotations

from .air_index import refractive_index_air
from .conditions import EnvironmentalConditions
from .models.birch_downs import birch_downs_refractive_index
from .models.ciddor import ciddor_refractive_index
from .models.edlen import edlen_refractive_index
from .models.kohlrausch import kohlrausch_refractive_index

__all__ = [
    "EnvironmentalConditions",
    "refractive_index_air",
    "ciddor_refractive_index",
    "kohlrausch_refractive_index",
    "edlen_refractive_index",
    "birch_downs_refractive_index",
]
