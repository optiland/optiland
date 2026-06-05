"""Constraint strategy subpackage — thin adapters over existing mechanisms.

Available strategies:
  - ``BoxBoundsStrategy``: reads ``var.bounds``; applies clamp + SciPy handoff.
  - ``ScipyNativeStrategy``: forwards bounds/constraints directly to SciPy.
  - ``CompositeStrategy``: chains multiple strategies.

For hard equality / inequality constraints solved with a true KKT active-set
step, declare them on the problem via ``OptimizationProblem.add_constraint``
(see :mod:`optiland.optimization.constraint`).  The legacy
``NullSpaceStrategy`` (project-then-restore) has been removed in favor of the
KKT path.

Kramer Harrison, 2026
"""

from __future__ import annotations

from .base import CompositeStrategy, ConstraintStrategy  # noqa: F401
from .bounds import BoxBoundsStrategy  # noqa: F401
from .scipy_native import ScipyNativeStrategy  # noqa: F401
