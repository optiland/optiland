"""Shared refractiveindex.info dispersion equations, with wavelengths in µm."""

from __future__ import annotations

from typing import Any

import optiland.backend as be


def validate_formula(formula: str, coefficients: Any) -> None:
    """Validate a supported formula identifier and its coefficient arity."""
    if not isinstance(formula, str):
        raise ValueError("Dispersion formula must be a string identifier")
    count = len(coefficients)
    if formula in {"formula 1", "formula 2", "formula 3", "formula 5", "formula 6"}:
        valid = count >= 1 and count % 2 == 1
    elif formula == "formula 4":
        valid = count >= 9 and count % 2 == 1
    elif formula == "formula 7":
        valid = count >= 3
    elif formula == "formula 8":
        valid = count == 4
    elif formula == "formula 9":
        valid = count == 6
    else:
        raise ValueError(f"Unsupported dispersion formula: {formula!r}")
    if not valid:
        raise ValueError(f"Invalid coefficients for dispersion {formula}.")


def evaluate_formula(formula: str, coefficients: Any, wavelength: Any) -> Any:
    """Evaluate any of the nine supported equations without changing their model.

    Coefficients may be live backend arrays or plain data. The caller chooses
    wavelength bounds and relative/absolute index conventions; this function
    owns only the published algebra and retains coefficient/query gradients.
    """
    validate_formula(formula, coefficients)
    c, w = coefficients, wavelength
    if formula in {"formula 1", "formula 2"}:
        value = 1 + c[0]
        for i in range(1, len(c), 2):
            resonance = c[i + 1] ** 2 if formula == "formula 1" else c[i + 1]
            value = value + c[i] * w**2 / (w**2 - resonance)
        return be.sqrt(value)
    if formula in {"formula 3", "formula 5"}:
        value = c[0]
        for i in range(1, len(c), 2):
            value = value + c[i] * w ** c[i + 1]
        return be.sqrt(value) if formula == "formula 3" else value
    if formula == "formula 4":
        value = (
            c[0]
            + c[1] * w ** c[2] / (w**2 - c[3] ** c[4])
            + c[5] * w ** c[6] / (w**2 - c[7] ** c[8])
        )
        for i in range(9, len(c), 2):
            value = value + c[i] * w ** c[i + 1]
        return be.sqrt(value)
    if formula == "formula 6":
        value = 1 + c[0]
        for i in range(1, len(c), 2):
            value = value + c[i] / (c[i + 1] - w**-2)
        return value
    if formula == "formula 7":
        resonance = 1 / (w**2 - 0.028)
        value = c[0] + c[1] * resonance + c[2] * resonance**2
        for i in range(3, len(c)):
            value = value + c[i] * w ** (2 * (i - 2))
        return value
    if formula == "formula 8":
        value = c[0] + c[1] * w**2 / (w**2 - c[2]) + c[3] * w**2
        return be.sqrt((1 + 2 * value) / (1 - value))
    value = c[0] + c[1] / (w**2 - c[2]) + c[3] * (w - c[4]) / ((w - c[4]) ** 2 + c[5])
    return be.sqrt(value)
