"""Managed cores: SciPy solvers behind ``minimize()`` (private).

Kramer Harrison, 2026
"""

from __future__ import annotations

from .scipy_least_squares import LEAST_SQUARES_METHODS, run_scipy_least_squares
from .scipy_local import SCIPY_LOCAL_METHODS, run_scipy_local
from .torch_first_order import TORCH_METHODS, run_torch_first_order

__all__ = [
    "LEAST_SQUARES_METHODS",
    "SCIPY_LOCAL_METHODS",
    "TORCH_METHODS",
    "run_scipy_least_squares",
    "run_scipy_local",
    "run_torch_first_order",
]
