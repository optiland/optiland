"""Pluggable coloring of drawn components and rays.

Kramer Harrison, 2026
"""

from __future__ import annotations

from optiland.visualization.coloring.components import (
    ComponentColorizer,
    ComponentRef,
    PropertyColorizer,
    ThemeColorizer,
    UniformColorizer,
)
from optiland.visualization.coloring.rays import (
    RayBundleRef,
    RayColorizer,
    TracedRays,
    UniformRayColorizer,
)
from optiland.visualization.coloring.scale import ColorScale

__all__ = [
    "ColorScale",
    "ComponentColorizer",
    "ComponentRef",
    "PropertyColorizer",
    "RayBundleRef",
    "RayColorizer",
    "ThemeColorizer",
    "TracedRays",
    "UniformColorizer",
    "UniformRayColorizer",
]
