"""Constraint strategy subpackage — thin adapters over existing mechanisms.

Phase 1 strategies (D6):
  - ``BoxBoundsStrategy``: reads ``var.bounds``; applies clamp + SciPy handoff.
  - ``ScipyNativeStrategy``: forwards bounds/constraints directly to SciPy.
  - ``CompositeStrategy``: chains multiple strategies.

NullSpaceStrategy (exact CODE V-style projection) is Phase 3.

Kramer Harrison, 2026
"""

from __future__ import annotations

from .base import CompositeStrategy, ConstraintStrategy  # noqa: F401
from .bounds import BoxBoundsStrategy  # noqa: F401
from .scipy_native import ScipyNativeStrategy  # noqa: F401
