"""CODE V Data Model

Defines CodeVDataModel, the shared intermediate representation used by both
the CODE V reader (parser -> model) and writer (optic -> model) paths.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CodeVDataModel:
    """Intermediate representation of a CODE V Sequential optical system.

    This dataclass is the stable lingua franca shared between the read and
    write paths. The reader populates it from a parsed .seq file; the writer
    produces it from an Optic object.

    Attributes:
        name: Optional system name (from the TITLE operand).
        aperture: Aperture information keyed by CODE V aperture command
            (e.g. ``{"EPD": 10.0}`` or ``{"FNO": 2.8}``).
        fields: Field configuration including type, x/y coordinates and
            optional weights.
        wavelengths: Wavelength data in microns, optional weights, and the
            0-based primary wavelength index.
        surfaces: Mapping from surface index to a raw operand dict used by
            the encoder / converter.
        radius_mode: If True, surface values are radii (RDM Y); if False,
            values are curvatures (RDM N). Defaults to True.
        units: Length units string; always ``"MM"`` for v1.
    """

    name: str | None = None
    aperture: dict[str, Any] = field(default_factory=dict)
    fields: dict[str, Any] = field(default_factory=dict)
    wavelengths: dict[str, Any] = field(default_factory=lambda: {"data": []})
    surfaces: dict[int, dict[str, Any]] = field(default_factory=dict)
    radius_mode: bool = True
    units: str = "MM"
    sto_surface_index: int | None = None  # explicit global stop surface (STO Sn)

    def to_dict(self) -> dict[str, Any]:
        """Return the data model as a plain dictionary.

        Returns:
            A plain dict representation suitable for use with
            ``CodeVToOpticConverter``.
        """
        return {
            "name": self.name,
            "aperture": self.aperture,
            "fields": self.fields,
            "wavelengths": self.wavelengths,
            "surfaces": self.surfaces,
            "radius_mode": self.radius_mode,
            "units": self.units,
            "sto_surface_index": self.sto_surface_index,
        }
