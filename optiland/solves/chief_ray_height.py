"""ChiefRayHeightSolve Module (Deprecated)

This module is deprecated and will be removed in v0.7.0.
Please use `optiland.solves.thickness` instead.
"""

from __future__ import annotations

import warnings

from optiland.solves.thickness import ChiefRayHeightThicknessSolve

warnings.warn(
    "The `optiland.solves.chief_ray_height` module is deprecated and will be "
    "removed in v0.7.0. Use `optiland.solves.thickness` instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Alias for backward compatibility
ChiefRayHeightSolve = ChiefRayHeightThicknessSolve
