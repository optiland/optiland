"""MarginalRayHeightSolve Module (Deprecated)

This module is deprecated and will be removed in v0.7.0.
Please use `optiland.solves.thickness` instead.
"""

from __future__ import annotations

import warnings

from optiland.solves.thickness import MarginalRayHeightThicknessSolve

warnings.warn(
    "The `optiland.solves.marginal_ray_height` module is deprecated and will be "
    "removed in v0.7.0. Use `optiland.solves.thickness` instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Alias for backward compatibility
MarginalRayHeightSolve = MarginalRayHeightThicknessSolve
