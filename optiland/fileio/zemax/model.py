"""Zemax Data Model

Defines ZemaxDataModel, the shared intermediate representation used by both
the Zemax reader (parser -> model) and writer (optic -> model) paths.

Kramer Harrison, 2024
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ZemaxDataModel:
    """Intermediate representation of a Zemax OpticStudio optical system.

    This dataclass is the stable lingua franca shared between the read and
    write paths. The reader populates it from a parsed .zmx file; the writer
    produces it from an Optic object.

    Attributes:
        name: Optional system name (from the NAME operand).
        aperture: Aperture information keyed by Optiland aperture type string
            (e.g. ``{"EPD": 8.0}``).
        fields: Field configuration including type, x/y coordinates and
            vignetting arrays.
        wavelengths: Wavelength data including values in microns, weights,
            and the primary wavelength index.
        surfaces: Mapping from surface index to a raw operand dict used by
            the encoder / converter.
        glass_catalogs: List of Zemax glass catalog names (GCAT operand).
        mode: Optical mode string; always ``"Sequential"`` for v1.
    """

    name: str | None = None
    aperture: dict[str, Any] = field(default_factory=dict)
    fields: dict[str, Any] = field(default_factory=dict)
    wavelengths: dict[str, Any] = field(
        default_factory=lambda: {"data": [], "weights": []}
    )
    surfaces: dict[int, dict[str, Any]] = field(default_factory=dict)
    glass_catalogs: list[str] | None = None
    mode: str = "Sequential"

    def to_dict(self) -> dict[str, Any]:
        """Return the data model as a plain dictionary.

        Returns:
            A plain dict representation suitable for use with
            ``ZemaxToOpticConverter``.
        """
        result: dict[str, Any] = {
            "name": self.name,
            "aperture": self.aperture,
            "fields": self.fields,
            "wavelengths": self.wavelengths,
            "surfaces": self.surfaces,
        }
        if self.glass_catalogs is not None:
            result["glass_catalogs"] = self.glass_catalogs
        return result
