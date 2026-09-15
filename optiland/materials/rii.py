"""Decode refractiveindex.info spectral blocks into backend-independent records."""

from __future__ import annotations

from io import StringIO
from typing import Any

import numpy as np

from optiland.materials.definition import ExtinctionTable, FormulaDispersion, IndexTable
from optiland.materials.spectral import wavelength_limits


def decode_formula(block: dict[str, Any]) -> FormulaDispersion:
    """Parse a formula block; its evaluator validates supported coefficient arities."""
    limits = block.get("wavelength_range")
    return FormulaDispersion(
        block["type"],
        tuple(float(value) for value in block.get("coefficients", "").split()),
        wavelength_limits(limits.split()) if limits is not None else None,
    )


def decode_table(
    block: dict[str, Any],
) -> tuple[IndexTable | None, ExtinctionTable | None]:
    """Decode n, k or nk tables without imposing the owned-material bounds policy."""
    array = np.loadtxt(StringIO(block.get("data", "")))
    if array.ndim == 1:
        array = array.reshape((1, -1) if array.size else (0, 0))
    kind = block["type"]
    index, extinction = None, None
    if kind in {"tabulated n", "tabulated k", "tabulated nk"}:
        waves = tuple(float(value) for value in array[:, 0])
        values = tuple(float(value) for value in array[:, 1])
        if kind in {"tabulated n", "tabulated nk"}:
            index = IndexTable(waves, values)
        if kind == "tabulated k":
            extinction = ExtinctionTable(waves, values)
        elif kind == "tabulated nk":
            extinction = ExtinctionTable(
                waves, tuple(float(value) for value in array[:, 2])
            )
    return index, extinction
