"""Constraint strategy subpackage — thin adapters over existing mechanisms.

Phase 1 strategies (D6):
  - ``BoxBoundsStrategy``: reads ``var.bounds``; applies clamp + SciPy handoff.
  - ``ScipyNativeStrategy``: forwards bounds/constraints directly to SciPy.
  - ``CompositeStrategy``: chains multiple strategies.

Phase 3 — exact CODE V-style projection:
  - ``NullSpaceStrategy``: null-space projection + Newton restoration.

Kramer Harrison, 2026
"""

from __future__ import annotations

from .base import CompositeStrategy, ConstraintStrategy  # noqa: F401
from .bounds import BoxBoundsStrategy  # noqa: F401
from .null_space import NullSpaceStrategy  # noqa: F401
from .scipy_native import ScipyNativeStrategy  # noqa: F401
