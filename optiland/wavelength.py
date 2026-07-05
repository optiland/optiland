"""Wavelength Module

This module defines the `Wavelength` and `WavelengthGroup` classes for
managing wavelengths in optical simulations. The `Wavelength` class represents
a single wavelength, allowing for its value to be defined in various units and
converted to microns for internal consistency. The `WavelengthGroup` class
manages collections of `Wavelength` objects, providing functionality to work
with multiple wavelengths simultaneously.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import optiland.backend as be

if TYPE_CHECKING:
    from collections.abc import Iterator

    from optiland._types import ScalarOrArray


class Wavelength:
    """Represents a wavelength value with support for unit conversion.

    Methods:
        _convert_to_um(): Converts the wavelength value to microns.
    """

    def __init__(
        self,
        value: ScalarOrArray,
        is_primary: bool = True,
        unit: str = "um",
        weight: float = 1.0,
    ):
        """Initializes a Wavelength instance

        Args:
            value (float): The value of the wavelength.
            is_primary (bool): Indicates whether the wavelength is a primary
                wavelength.
            unit (str): The unit of the wavelength value. Defaults to 'um'.
            weight (float): Non-negative relative importance scalar. A weight
                of 0.0 excludes the wavelength from optimization and weighted
                analysis. Defaults to 1.0.
        """
        self._value = value
        self.is_primary = is_primary
        self._unit = unit.lower()
        self._value_in_um = self._convert_to_um()
        self.weight = weight  # uses the validated setter

    @property
    def weight(self) -> float:
        """float: Non-negative relative importance scalar for this wavelength."""
        return self._weight

    @weight.setter
    def weight(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"Wavelength weight must be non-negative, got {value}.")
        self._weight = float(value)

    @property
    def value(self) -> float:
        """float: the value of the wavelength"""
        return self._value_in_um

    @property
    def unit(self) -> str:
        """str: the unit of the wavelength"""
        return "um"

    @unit.setter
    def unit(self, new_unit: str) -> None:
        """Sets the unit of the wavelength.

        Args:
            new_unit: The new unit to set for the wavelength.

        """
        self._unit = new_unit.lower()
        self._value_in_um = self._convert_to_um()

    def _convert_to_um(self) -> float:
        """Converts the wavelength value to micrometers (um) based on the
        current unit.

        Returns:
            float: The converted wavelength value in micrometers.

        Raises:
            ValueError: If the current unit is not supported for conversion to
                micrometers. Supported units: 'nm', 'um', 'mm', 'cm', 'm'.

        """
        unit_conversion = {"nm": 0.001, "um": 1, "mm": 1000, "cm": 10000, "m": 1000000}

        if self._unit in unit_conversion:
            conversion_factor = unit_conversion[self._unit]
            return float(self._value * conversion_factor)
        raise ValueError("Unsupported unit for conversion to microns.")

    def to_dict(self) -> dict[str, Any]:
        """Get a dictionary representation of the wavelength.

        Returns:
            A dictionary representation of the wavelength.

        """
        return {
            "value": self._value,
            "is_primary": self.is_primary,
            "unit": self._unit,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Wavelength:
        """Create a Wavelength instance from a dictionary representation.

        Args:
            data: A dictionary containing the wavelength data.

        Returns:
            A new Wavelength instance created from the data.

        """
        required_keys = {"value", "is_primary", "unit"}
        if not required_keys.issubset(data):
            missing = required_keys - data.keys()
            raise ValueError(f"Missing required keys: {missing}")

        return cls(
            value=data["value"],
            is_primary=data["is_primary"],
            unit=data["unit"],
            weight=data.get("weight", 1.0),
        )


class WavelengthGroup:
    """Represents a group of wavelengths, each with an optional weight.

    Attributes:
        wavelengths (list): A list of Wavelength objects.

    Methods:
        num_wavelengths(): Returns the number of wavelengths in the group.
        primary_index(): Returns the index of the primary wavelength.
        primary_wavelength(): Returns the primary wavelength.
        add_wavelength(value, is_primary=True, unit='um'): Adds a new
            wavelength to the group.
        get_wavelength(wavelength_number): Returns the value of a specific
            wavelength.
        get_wavelengths(): Returns a list of all the wavelength values in the
            group.

    """

    def __init__(self) -> None:
        """Initialize an empty WavelengthGroup."""
        self.wavelengths: list[Wavelength] = []

    @property
    def weights(self) -> tuple[float, ...]:
        """The weights of the wavelengths"""
        return tuple(wave.weight for wave in self.wavelengths)

    @property
    def num_wavelengths(self) -> int:
        """The number of wavelengths"""
        return len(self.wavelengths)

    def __getitem__(self, index: int) -> Wavelength:
        """Return the wavelength at *index*."""
        return self.wavelengths[index]

    def __iter__(self) -> Iterator[Wavelength]:
        """Iterate over the wavelengths in the group."""
        return iter(self.wavelengths)

    def __len__(self) -> int:
        """Return the number of wavelengths in the group."""
        return len(self.wavelengths)

    @property
    def primary_index(self) -> int:
        """The index of the primary wavelength

        raises:
            StopIteration: If no primary wavelength is found
        """
        return next(i for i, w in enumerate(self.wavelengths) if w.is_primary)

    @primary_index.setter
    def primary_index(self, index: int) -> None:
        """set the wavelength indexed by `index` as primary"""
        if not 0 <= index < len(self.wavelengths):
            raise ValueError("Index out of range")
        for idx, wavelength in enumerate(self.wavelengths):
            wavelength.is_primary = idx == index

    @property
    def primary_wavelength(self) -> Wavelength:
        """The primary wavelength"""
        return self.wavelengths[self.primary_index]

    def add(
        self,
        value: float,
        is_primary: bool = False,
        unit: str = "um",
        weight: float = 1.0,
    ) -> None:
        """Adds a new wavelength to the list of wavelengths.

        Args:
            value: The value of the wavelength.
            is_primary: Indicates if the wavelength is primary. Default is True.
            unit: The unit of the wavelength. Default is 'um'.
            weight: The weight of the wavelength. Default is 1.0.
        """
        if is_primary:
            for wavelength in self.wavelengths:
                wavelength.is_primary = False

        if self.num_wavelengths == 0:
            is_primary = True

        self.wavelengths.append(Wavelength(value, is_primary, unit, weight))

    def remove(self, index: int) -> None:
        """Remove a wavelength from the group.

        Args:
            index: The index of the wavelength to remove.
        """
        self.wavelengths.pop(index)

    def get_wavelength(self, wavelength_number: int) -> float:
        """Get the value of a specific wavelength.

        Args:
            wavelength_number: The index of the desired wavelength.

        Returns:
            The value of the specified wavelength.
        """
        return self.wavelengths[wavelength_number].value

    def get_wavelengths(self) -> list[float]:
        """Returns a list of wavelength values.

        Returns:
            A list of wavelength values.

        """
        return [wave.value for wave in self.wavelengths]

    def to_dict(self) -> dict[str, Any]:
        """Get a dictionary representation of the wavelength group.

        Returns:
            A dictionary representation of the wavelength group.

        """
        return {"wavelengths": [wave.to_dict() for wave in self.wavelengths]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WavelengthGroup:
        """Create a WavelengthGroup instance from a dictionary representation.

        Args:
            data: A dictionary containing the wavelength group data.

        Returns:
            A new WavelengthGroup instance created from the
                data.

        """
        if "wavelengths" not in data:
            raise ValueError('Missing required key: "wavelengths"')

        new_group = cls()
        for wave_data in data["wavelengths"]:
            new_group.add(**wave_data)

        return new_group


def add_wavelengths(
    wavelength_group: WavelengthGroup,
    min_value: float,
    max_value: float,
    num_wavelengths: int,
    unit: str = "um",
    *,
    sampling: str = "chebyshev",
    scale: str = "log",
) -> None:
    """Add new wavelengths corresponding to the geometrically-spaced Chebyshev nodes

    Args:
        min_value: Minimum wavelength value.
        max_value: Maximum wavelength value.
        num_wavelengths: The number of wavelengths to be added.
            Has to be an odd integer.
        unit: The unit of the wavelength. Default is 'um'.
        sampling: The sampling algorithm used. Defaults to 'chebyshev'.
            Currently supported options are:
                'chebyshev' - chebyshev nodes of the first type
                'uniform' - uniformly spaced nodes across the specified range
        scale: space in which the nodes are sampled. Defaults to 'log'.
            Currently supported options are:
                'log' - nodes are sampled in the logarithms of wavelength.
                'frequency' - nodes sampled in the frequency domain.
                'wavelength' - nodes sampled in the frequency domain. Not recommended.
    """
    if (
        not isinstance(num_wavelengths, int)
        or num_wavelengths % 2 == 0
        or num_wavelengths <= 0
    ):
        raise ValueError("num_wavelengths must be an odd positive integer")

    if min_value <= 0 or max_value <= 0:
        raise ValueError("min_value and max_value must be positive")

    scale = scale.lower()
    if scale in {"freq", "frequency"}:
        scale = "frequency"
    elif scale in {"wavelength"}:
        scale = "wavelength"
    elif scale in {"log", "logarithmic"}:
        scale = "log"
    else:
        raise ValueError(f"Unknown scale: {scale!r}")

    if scale == "frequency":
        power = -1.0
    elif scale == "wavelength":
        power = 1.0

    nodes = be.arange(1.0, num_wavelengths + 1.0)

    if sampling == "chebyshev":
        nodes = 0.5 * (1.0 - be.cos((2 * nodes - 1) * be.pi / (2 * num_wavelengths)))

    elif sampling == "uniform":
        nodes = (nodes - 0.5) / num_wavelengths

    if scale == "log":
        span = float(be.log2(be.array(max_value / min_value)))
        for i, node in enumerate(nodes):
            is_primary = i == num_wavelengths // 2
            value = min_value * 2 ** (span * node)
            wavelength_group.wavelengths.append(
                Wavelength(value, is_primary, unit, 1.0)
            )
    else:
        min_value = min_value**power
        max_value = max_value**power
        span = max_value - min_value
        for i, node in enumerate(nodes):
            is_primary = i == num_wavelengths // 2
            value = min_value + (span * node)
            wavelength_group.wavelengths.append(
                Wavelength(value ** (1.0 / power), is_primary, unit, 1.0)
            )
