"""Optical materials built from owned data, with self-contained native persistence."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.materials.base import BaseMaterial
from optiland.materials.definition import (
    IndexTable,
    MaterialDefinition,
    definition_from_dict,
    definition_to_dict,
)
from optiland.materials.dispersion import evaluate_formula
from optiland.materials.spectral import checked_wavelengths, interpolate_linear

if TYPE_CHECKING:
    from optiland.propagation.base import BasePropagationModel


class DataMaterial(BaseMaterial):
    """Own sampled or analytic dispersion and optional measured extinction.

    Use ``from_samples`` or ``from_coefficients`` to construct optical data, or
    pass an explicit native definition. No file or registry is accessed.
    Definition records are immutable; create a replacement to change optical
    data. Wavelengths use micrometers and extinction k is dimensionless.

    Args:
        definition: Native dispersion and optional extinction objects.
        name: Descriptive label, never a catalog-lookup instruction.
        metadata: JSON-compatible descriptive provenance, copied on input/output.
        propagation_model: Optional registered propagation model.
    """

    def __init__(
        self,
        definition: dict[str, Any],
        *,
        name: str = "",
        metadata: dict[str, Any] | None = None,
        propagation_model: BasePropagationModel | None = None,
    ) -> None:
        super().__init__(propagation_model)
        if not isinstance(name, str):
            raise ValueError("Material name must be a string")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("Material metadata must be an object")
        self._definition = definition_from_dict(definition)
        self.name = name
        try:
            self._metadata = json.loads(json.dumps(metadata or {}, allow_nan=False))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "Material metadata must contain finite JSON data"
            ) from error

    @classmethod
    def from_samples(
        cls,
        wavelengths: Any,
        indices: Any,
        *,
        name: str = "",
        extinction: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        propagation_model: BasePropagationModel | None = None,
    ) -> DataMaterial:
        """Preserve paired samples with bounded linear interpolation of n and k."""
        return cls(
            {
                "dispersion": {
                    "kind": "tabulated",
                    "wavelengths_um": wavelengths,
                    "indices": indices,
                },
                "extinction": extinction,
            },
            name=name,
            metadata=metadata,
            propagation_model=propagation_model,
        )

    @classmethod
    def from_coefficients(
        cls,
        formula: str,
        coefficients: Any,
        *,
        name: str = "",
        wavelength_range: Any = None,
        extinction: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        propagation_model: BasePropagationModel | None = None,
    ) -> DataMaterial:
        """Preserve an analytic equation; optionally bound its wavelengths in µm."""
        return cls(
            {
                "dispersion": {
                    "kind": "formula",
                    "formula": formula,
                    "coefficients": coefficients,
                    "wavelength_range_um": wavelength_range,
                },
                "extinction": extinction,
            },
            name=name,
            metadata=metadata,
            propagation_model=propagation_model,
        )

    @property
    def definition(self) -> MaterialDefinition:
        """The read-only optical definition, independent of descriptive metadata."""
        return self._definition

    @property
    def metadata(self) -> dict[str, Any]:
        """An independent copy of optional provenance."""
        return deepcopy(self._metadata)

    @property
    def display_name(self) -> str:
        """Use the optional descriptive name without inventing catalog identity."""
        return self.name or "DataMaterial"

    def spectral_range(self, property_name: str = "n") -> tuple[float, float] | None:
        """Bounds of the requested property; absent extinction is unbounded zero."""
        super().spectral_range(property_name)
        data = (
            self.definition.dispersion
            if property_name == "n"
            else self.definition.extinction
        )
        if data is None:
            return None
        if hasattr(data, "wavelengths_um"):
            return data.wavelengths_um[0], data.wavelengths_um[-1]
        return data.wavelength_range_um

    def _cache_state(self) -> tuple:
        """The owned immutable definition is the complete optical state."""
        return (self._definition,)

    def _calculate_n(self, wavelength: Any, **kwargs: Any) -> Any:
        index = self.definition.dispersion
        if isinstance(index, IndexTable):
            wave = checked_wavelengths(wavelength)
            return interpolate_linear(wave, index.wavelengths_um, index.indices)
        wave = checked_wavelengths(wavelength, index.wavelength_range_um)
        result = evaluate_formula(index.formula, be.asarray(index.coefficients), wave)
        result = result + wave * 0  # Constant formulas retain the full query shape.
        if not be.all(be.isfinite(result)):
            raise ValueError(
                "Dispersion formula produced a non-finite refractive index"
            )
        return result

    def _calculate_k(self, wavelength: Any, **kwargs: Any) -> Any:
        wave = checked_wavelengths(wavelength)
        table = self.definition.extinction
        if table is None:
            return wave * 0
        return interpolate_linear(wave, table.wavelengths_um, table.values)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete definition inline with independent containers."""
        return {
            **super().to_dict(),
            "name": self.name,
            "definition": definition_to_dict(self.definition),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataMaterial:
        """Restore owned optical data; BaseMaterial restores propagation dispatch."""
        return cls(
            data["definition"], name=data.get("name", ""), metadata=data.get("metadata")
        )
