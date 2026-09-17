"""Private immutable optical records and the native material-definition codec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from optiland.materials.dispersion import validate_formula
from optiland.materials.spectral import finite_values, paired_samples, wavelength_limits


@dataclass(frozen=True, slots=True)
class IndexTable:
    """Owned refractive-index samples at wavelengths in micrometers."""

    wavelengths_um: tuple[float, ...]
    indices: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FormulaDispersion:
    """Analytic refractiveindex.info coefficients and optional validity limits."""

    formula: str
    coefficients: tuple[float, ...]
    wavelength_range_um: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class ExtinctionTable:
    """Owned extinction-coefficient samples, on an independent wavelength grid."""

    wavelengths_um: tuple[float, ...]
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class MaterialDefinition:
    """A complete supported inline optical definition, without provenance or I/O."""

    dispersion: IndexTable | FormulaDispersion
    extinction: ExtinctionTable | None = None


def _fields(
    data: Any, required: set[str], optional: set[str] | frozenset[str] = frozenset()
) -> None:
    if not isinstance(data, dict):
        raise ValueError("Material definition components must be objects")
    missing = required - data.keys()
    unknown = data.keys() - required - optional
    if missing or unknown:
        raise ValueError(
            f"Invalid material definition fields: missing {sorted(missing)}, "
            f"unknown {sorted(unknown)}"
        )


def definition_from_dict(data: dict[str, Any]) -> MaterialDefinition:
    """Validate native optical data once and copy it into immutable records."""
    _fields(data, {"dispersion"}, {"extinction", "thermal"})
    if data.get("thermal") is not None:
        raise ValueError("Thermal data is not supported by this material feature")
    dispersion = data["dispersion"]
    if not isinstance(dispersion, dict):
        raise ValueError("Material dispersion must be an object")
    kind = dispersion.get("kind")
    if kind == "tabulated":
        _fields(dispersion, {"kind", "wavelengths_um", "indices"})
        waves, values = paired_samples(
            dispersion["wavelengths_um"], dispersion["indices"]
        )
        index = IndexTable(waves, values)
    elif kind == "formula":
        _fields(
            dispersion, {"kind", "formula", "coefficients"}, {"wavelength_range_um"}
        )
        formula = dispersion["formula"]
        coefficients = finite_values(
            dispersion["coefficients"], "Dispersion coefficients"
        )
        validate_formula(formula, coefficients)
        index = FormulaDispersion(
            formula,
            coefficients,
            wavelength_limits(dispersion.get("wavelength_range_um")),
        )
    else:
        raise ValueError(f"Unsupported dispersion kind: {kind!r}")
    extinction = data.get("extinction")
    table = None
    if extinction is not None:
        _fields(extinction, {"kind", "wavelengths_um", "values"})
        if extinction["kind"] != "tabulated_k":
            raise ValueError(f"Unsupported extinction kind: {extinction['kind']!r}")
        waves, values = paired_samples(
            extinction["wavelengths_um"], extinction["values"], nonnegative=True
        )
        table = ExtinctionTable(waves, values)
    return MaterialDefinition(index, table)


def definition_to_dict(definition: MaterialDefinition) -> dict[str, Any]:
    """Encode independent native containers without caches or backend state."""
    index = definition.dispersion
    if isinstance(index, IndexTable):
        dispersion = {
            "kind": "tabulated",
            "wavelengths_um": list(index.wavelengths_um),
            "indices": list(index.indices),
        }
    else:
        dispersion = {
            "kind": "formula",
            "formula": index.formula,
            "coefficients": list(index.coefficients),
            "wavelength_range_um": (
                list(index.wavelength_range_um) if index.wavelength_range_um else None
            ),
        }
    table = definition.extinction
    extinction = (
        None
        if table is None
        else {
            "kind": "tabulated_k",
            "wavelengths_um": list(table.wavelengths_um),
            "values": list(table.values),
        }
    )
    return {"dispersion": dispersion, "extinction": extinction, "thermal": None}
