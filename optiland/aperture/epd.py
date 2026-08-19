"""Entrance Pupil Diameter (EPD) Aperture

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.aperture.base import BaseSystemAperture

if TYPE_CHECKING:
    from optiland.paraxial import Paraxial


class EPDAperture(BaseSystemAperture):
    """Aperture specified directly as an entrance pupil diameter.

    The stored *value* is the entrance pupil diameter in lens units.

    Args:
        value: Entrance pupil diameter.

    """

    _ap_type_key = "EPD"

    def __init__(self, value: float) -> None:
        self._value = value

    @property
    def ap_type(self) -> str:
        return "EPD"

    @property
    def value(self) -> float:
        return self._value

    @property
    def supports_telecentric(self) -> bool:
        return False

    @property
    def is_scalable(self) -> bool:
        return True

    def compute_epd(self, paraxial: Paraxial, wavelength: float | None = None) -> float:
        """Return the stored EPD value directly.

        Args:
            paraxial: Unused for this aperture type.
            wavelength: Unused for this aperture type.

        Returns:
            The entrance pupil diameter.

        """
        return self._value

    def scale(self, factor: float) -> EPDAperture:
        """Return a new :class:`EPDAperture` with value scaled by *factor*.

        Args:
            factor: Multiplicative scale factor.

        Returns:
            A new EPDAperture instance.

        """
        return EPDAperture(self._value * factor)

    def to_dict(self) -> dict:
        return {"type": "EPD", "value": self._value}

    @classmethod
    def _from_dict(cls, data: dict) -> EPDAperture:
        return cls(data["value"])
