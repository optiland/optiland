"""Buchdahl dispersion algebra shared by models with different fitted orders."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import optiland.backend as be

if TYPE_CHECKING:
    from collections.abc import Iterable


def buchdahl_coordinate(
    wavelength: Any, reference_wavelength: float, alpha: float
) -> Any:
    """Transform wavelength in µm to the model's Buchdahl coordinate.

    The model owns its reference wavelength, alpha (in inverse µm), valid
    wavelength interval and any pole checks. This kernel only evaluates the
    coordinate and preserves backend gradients.
    """
    delta = be.array(wavelength) - reference_wavelength
    return delta / (1 + alpha * delta)


def evaluate_buchdahl(index: Any, coefficients: Iterable[Any], omega: Any) -> Any:
    """Evaluate ``index + c1*omega + c2*omega**2 + ...``.

    Coefficients are ordered by increasing power, excluding the reference
    index. Inputs follow backend broadcasting rules; a coefficient may itself
    be an array of model parameters. No fitted values or computation graphs
    are retained. The caller derives coefficients from its live parameters.
    """
    omega = be.array(omega)
    value = index + be.zeros_like(omega)
    for power, coefficient in enumerate(coefficients, start=1):
        value = value + coefficient * omega**power
    return value
