"""Hard-constraint data layer for the optimization subpackage (§4.1).

``Constraint`` / ``ConstraintManager`` mirror ``Operand`` / ``OperandManager``
but describe **hard** equality / inequality constraints solved by the KKT
active-set controller (see :mod:`optiland.optimization.control.kkt`), kept in a
list separate from the merit operands.

Declare them on a problem with ``OptimizationProblem.add_constraint(...)``.

Kramer Harrison, 2026
"""

from __future__ import annotations

from .constraint import EQUALITY, INEQUALITY, Constraint, ConstraintRow
from .manager import ConstraintManager

__all__ = [
    "Constraint",
    "ConstraintRow",
    "ConstraintManager",
    "EQUALITY",
    "INEQUALITY",
]
