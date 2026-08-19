"""This package contains the mathematical basis for NURBS, adapted
for the Optiland backend.
"""

from __future__ import annotations

from .nurbs_fitting import (
    approximate_surface,
)
from .nurbs_geometry import NurbsGeometry

__all__ = ["NurbsGeometry", "approximate_surface"]
