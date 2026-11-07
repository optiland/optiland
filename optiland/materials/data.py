"""Optical materials built from owned data, with self-contained native persistence."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.materials.base import BaseMaterial
from optiland.materials.definition import (
    IndexTable,
    MaterialDefinition,
    TransmissionTable,
    definition_from_dict,
    definition_to_dict,
)
from optiland.materials.dispersion import evaluate_formula
from optiland.materials.spectral import (
    BoundsPolicy,
    checked_wavelengths,
    extinction_to_transmission,
    interpolate_linear,
    interpolate_transmittance,
    transmission_to_extinction,
    validate_bounds,
)
from optiland.materials.thermal import thermal_index

if TYPE_CHECKING:
    from optiland.propagation.base import BasePropagationModel


class DataMaterial(BaseMaterial):
    """Own dispersion, optional extinction/transmission and thermal definitions.

    Use ``from_samples`` or ``from_coefficients`` to construct optical data, or
    pass an explicit native definition. No file or registry is accessed.
    Definition records are immutable; create a replacement to change optical
    data. Wavelengths use micrometers and extinction k is dimensionless.

    Args:
        definition: Native dispersion, optional extinction and thermal objects.
        name: Descriptive label, never a catalog-lookup instruction.
        metadata: JSON-compatible descriptive provenance, copied on input/output.
        propagation_model: Optional registered propagation model.
        bounds: Tabulated n/k queries outside their respective sample intervals
            raise by default. ``"clamp"`` holds the nearest endpoint value.
            Analytic formula validity limits remain enforced independently.
    """

    def __init__(
        self,
        definition: dict[str, Any],
        *,
        name: str = "",
        metadata: dict[str, Any] | None = None,
        propagation_model: BasePropagationModel | None = None,
        bounds: BoundsPolicy = "raise",
    ) -> None:
        super().__init__(propagation_model)
        self._bounds = validate_bounds(bounds)
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
        thermal: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        propagation_model: BasePropagationModel | None = None,
        bounds: BoundsPolicy = "raise",
    ) -> DataMaterial:
        """Preserve index samples with optional extinction and thermal data.

        Extinction uses a ``tabulated_k`` or ``internal_transmittance`` native
        object; thermal uses ``schott``, ``constant`` or ``corning``. Measured
        n/k tables use linear interpolation on independent grids, with optional
        endpoint clamping. Internal-transmission queries remain bounded.
        """
        return cls(
            {
                "dispersion": {
                    "kind": "tabulated",
                    "wavelengths_um": wavelengths,
                    "indices": indices,
                },
                "extinction": extinction,
                "thermal": thermal,
            },
            name=name,
            metadata=metadata,
            propagation_model=propagation_model,
            bounds=bounds,
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
        thermal: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        propagation_model: BasePropagationModel | None = None,
        bounds: BoundsPolicy = "raise",
    ) -> DataMaterial:
        """Preserve an analytic equation and optional extinction/thermal data.

        Wavelength limits use µm. Extinction and thermal objects have the same
        native definitions as ``from_samples``; coefficients remain analytic.
        ``bounds`` controls tabulated k only; supplied formula wavelength limits
        and internal-transmission support remain enforced independently.
        """
        return cls(
            {
                "dispersion": {
                    "kind": "formula",
                    "formula": formula,
                    "coefficients": coefficients,
                    "wavelength_range_um": wavelength_range,
                },
                "extinction": extinction,
                "thermal": thermal,
            },
            name=name,
            metadata=metadata,
            propagation_model=propagation_model,
            bounds=bounds,
        )

    @property
    def bounds(self) -> BoundsPolicy:
        """The read-only out-of-range policy for tabulated n and k."""
        return self._bounds

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
        """Track the owned definition and its table evaluation policy."""
        return (self._definition, self.bounds)

    def _calculate_n(self, wavelength: Any, **kwargs: Any) -> Any:
        wave = checked_wavelengths(wavelength)
        if self.definition.thermal is None or kwargs.get("temperature") is None:
            return self._dispersion(wave)
        return thermal_index(
            self._dispersion,
            wave,
            self.definition.thermal,
            kwargs["temperature"],
            kwargs.get("pressure"),
        )

    def _dispersion(self, wavelength: Any) -> Any:
        index = self.definition.dispersion
        if isinstance(index, IndexTable):
            wave = checked_wavelengths(wavelength)
            return interpolate_linear(
                wave, index.wavelengths_um, index.indices, bounds=self.bounds
            )
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
        if isinstance(table, TransmissionTable):
            transmission = interpolate_transmittance(
                wave, table.wavelengths_um, table.values
            )
            return transmission_to_extinction(transmission, wave, table.thickness_mm)
        return interpolate_linear(
            wave, table.wavelengths_um, table.values, bounds=self.bounds
        )

    def internal_transmittance(self, wavelength: Any, thickness_mm: float) -> Any:
        """Intensity transmission over a thickness in mm, excluding interfaces.

        For transmission tables, interpolate before scaling by Beer's law. Zero
        transmission remains representable here even though its k is not finite.
        """
        if not math.isfinite(thickness_mm) or thickness_mm < 0:
            raise ValueError("Transmission thickness must be finite and nonnegative")
        wave = checked_wavelengths(wavelength)
        table = self.definition.extinction
        if isinstance(table, TransmissionTable):
            transmission = interpolate_transmittance(
                wave, table.wavelengths_um, table.values
            )
            return transmission ** (thickness_mm / table.thickness_mm)
        return extinction_to_transmission(self.k(wave), wave, thickness_mm)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete definition inline with independent containers."""
        return {
            **super().to_dict(),
            "name": self.name,
            "definition": definition_to_dict(self.definition),
            "bounds": self.bounds,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataMaterial:
        """Restore owned optical data; BaseMaterial restores propagation dispatch."""
        return cls(
            data["definition"],
            name=data.get("name", ""),
            metadata=data.get("metadata"),
            bounds=data["bounds"],
        )
