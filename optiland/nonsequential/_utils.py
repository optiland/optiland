"""Shared array-dispatch utilities for the NSQ subpackage.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np


def get_xp(arr: np.ndarray):
    """Return numpy or cupy depending on array type."""
    try:
        import cupy  # type: ignore[import]

        if isinstance(arr, cupy.ndarray):
            return cupy
    except ImportError:
        pass
    return np


def to_numpy(arr: np.ndarray) -> np.ndarray:
    """Convert CuPy or NumPy array to NumPy."""
    try:
        import cupy  # type: ignore[import]

        if isinstance(arr, cupy.ndarray):
            return cupy.asnumpy(arr)
    except ImportError:
        pass
    return np.asarray(arr)
