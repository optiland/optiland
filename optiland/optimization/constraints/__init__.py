"""Constraint strategy subpackage — thin adapters over existing mechanisms.

Strategies:
  - ``BoxBoundsStrategy``: reads ``var.bounds``; applies clamp + SciPy handoff.
  - ``ScipyNativeStrategy``: forwards bounds/constraints directly to SciPy.
  - ``CompositeStrategy``: chains multiple strategies.
  - ``NullSpaceStrategy``: null-space projection + Newton restoration for exact
    equality/inequality constraint enforcement.

Kramer Harrison, 2026
"""

from __future__ import annotations

from .base import CompositeStrategy, ConstraintStrategy  # noqa: F401
from .bounds import BoxBoundsStrategy  # noqa: F401
from .null_space import NullSpaceStrategy  # noqa: F401
from .scipy_native import ScipyNativeStrategy  # noqa: F401
