"""Private immutable optical records and the native material-definition codec."""

from __future__ import annotations

import math
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
class TransmissionTable:
    """Internal transmission at a declared thickness in millimeters."""

    wavelengths_um: tuple[float, ...]
    values: tuple[float, ...]
    thickness_mm: float
    interpolation: str = "local_quadratic"


@dataclass(frozen=True, slots=True)
class ThermalModel:
    """Catalog thermal coefficients and a reference temperature in Celsius."""

    kind: str
    coefficients: tuple[float, ...]
    reference_temperature_c: float


@dataclass(frozen=True, slots=True)
class MaterialDefinition:
    """A complete supported inline optical definition, without provenance or I/O."""

    dispersion: IndexTable | FormulaDispersion
    extinction: ExtinctionTable | TransmissionTable | None = None
    thermal: ThermalModel | None = None


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
        _fields(
            extinction,
            {"kind", "wavelengths_um", "values"},
            {"thickness_mm", "interpolation"},
        )
        try:
            waves, values = paired_samples(
                extinction["wavelengths_um"], extinction["values"], nonnegative=True
            )
        except ValueError as error:
            label = (
                "transmission"
                if extinction["kind"] == "internal_transmittance"
                else "extinction"
            )
            raise ValueError(f"Invalid {label} table: {error}") from error
        if extinction["kind"] == "tabulated_k":
            _fields(extinction, {"kind", "wavelengths_um", "values"})
            table = ExtinctionTable(waves, values)
        elif extinction["kind"] == "internal_transmittance":
            _fields(
                extinction,
                {"kind", "wavelengths_um", "values", "thickness_mm", "interpolation"},
            )
            thickness = finite_values(
                [extinction["thickness_mm"]], "Transmission thickness"
            )[0]
            if (
                len(waves) < 3
                or any(value > 1 for value in values)
                or not math.isfinite(thickness)
                or thickness <= 0
                or extinction["interpolation"] != "local_quadratic"
            ):
                raise ValueError("Invalid local-quadratic transmission table")
            table = TransmissionTable(waves, values, thickness)
        else:
            raise ValueError(f"Unsupported extinction kind: {extinction['kind']!r}")
    thermal = data.get("thermal")
    model = None
    if thermal is not None:
        if (
            not isinstance(thermal, dict)
            or not isinstance(thermal.get("kind"), str)
            or thermal.get("kind")
            not in {
                "schott",
                "constant",
                "corning",
            }
        ):
            raise ValueError("Unsupported thermal model")
        _fields(thermal, {"kind", "coefficients", "reference_temperature_c"})
        c = finite_values(thermal["coefficients"], "Thermal coefficients")
        t0 = finite_values(
            [thermal["reference_temperature_c"]], "Thermal reference temperature"
        )[0]
        if (
            len(c) != {"schott": 6, "constant": 1, "corning": 4}[thermal["kind"]]
            or not math.isfinite(t0)
            or t0 <= -273.15
        ):
            raise ValueError("Invalid thermal coefficients or reference temperature")
        model = ThermalModel(thermal["kind"], c, t0)
    return MaterialDefinition(index, table, model)


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
            "kind": "internal_transmittance"
            if isinstance(table, TransmissionTable)
            else "tabulated_k",
            "wavelengths_um": list(table.wavelengths_um),
            "values": list(table.values),
        }
    )
    if isinstance(table, TransmissionTable):
        extinction.update(
            thickness_mm=table.thickness_mm, interpolation=table.interpolation
        )
    model = definition.thermal
    thermal = (
        None
        if model is None
        else {
            "kind": model.kind,
            "coefficients": list(model.coefficients),
            "reference_temperature_c": model.reference_temperature_c,
        }
    )
    return {"dispersion": dispersion, "extinction": extinction, "thermal": thermal}
