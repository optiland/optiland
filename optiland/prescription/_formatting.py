"""Shared numeric and string formatting helpers for prescription output.

Kramer Harrison, 2026
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

_INF_THRESHOLD = 1e12


def fmt_float(value: float | None, decimals: int = 4, unit: str = "") -> str:
    """Format a float to a fixed number of decimal places.

    Args:
        value: The value to format, or None.
        decimals: Number of decimal places.
        unit: Optional unit suffix (space-separated).

    Returns:
        Formatted string, or "N/A" if value is None.
    """
    if value is None:
        return "N/A"
    try:
        import numpy as np

        v = float(np.asarray(value).flat[0])
    except (TypeError, ValueError, IndexError):
        return "N/A"
    result = f"{v:.{decimals}f}"
    if unit:
        result = f"{result} {unit}"
    return result


def fmt_infinity(value: float, decimals: int = 4, unit: str = "") -> str:
    """Format a float, returning "∞" for very large or infinite values.

    Args:
        value: The value to format.
        decimals: Number of decimal places for finite values.
        unit: Optional unit suffix.

    Returns:
        "∞" if the value is infinite or abs > 1e12, else formatted string.
    """
    try:
        import numpy as np

        v = float(np.asarray(value).flat[0])
    except (TypeError, ValueError, IndexError):
        return "N/A"
    if math.isinf(v) or abs(v) > _INF_THRESHOLD:
        return "∞"
    return fmt_float(v, decimals=decimals, unit=unit)


def safe_eval(
    fn: Callable[[], Any],
    decimals: int = 4,
    unit: str = "",
    infinity: bool = True,
) -> str:
    """Call fn() and format the result, returning "N/A" on any exception.

    Args:
        fn: Zero-argument callable returning a numeric value.
        decimals: Decimal places for formatting.
        unit: Optional unit suffix.
        infinity: If True, use fmt_infinity to detect ∞ values.

    Returns:
        Formatted string or "N/A".
    """
    try:
        import numpy as np

        value = fn()
        v = float(np.asarray(value).flat[0])
    except Exception:
        return "N/A"
    if infinity:
        return fmt_infinity(v, decimals=decimals, unit=unit)
    return fmt_float(v, decimals=decimals, unit=unit)
