"""Hard constraints of an optimization problem (opt v2 M7). *Experimental.*

Declare them with ``OptimizationProblem.add_constraint(...)``.

Kramer Harrison, 2026
"""

from __future__ import annotations

from .constraint import (
    EQUALITY,
    INEQUALITY,
    Constraint,
    ConstraintRow,
    row_violations,
)
from .manager import ConstraintManager, row_report

__all__ = [
    "EQUALITY",
    "INEQUALITY",
    "Constraint",
    "ConstraintManager",
    "ConstraintRow",
    "row_report",
    "row_violations",
]
