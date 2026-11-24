"""Built-in tolerance types, registered in ``tolerance_registry`` on import."""

from __future__ import annotations

from optiland.tolerancing.registry import tolerance_registry
from optiland.tolerancing.types.surface import (
    ConicTolerance,
    FringesTolerance,
    RadiusTolerance,
    SurfaceDecenterTolerance,
    SurfaceTiltTolerance,
    ThicknessTolerance,
    VariableTolerance,
)

_BUILTIN = {
    "radius": RadiusTolerance,
    "fringes": FringesTolerance,
    "conic": ConicTolerance,
    "thickness": ThicknessTolerance,
    "surface_decenter": SurfaceDecenterTolerance,
    "surface_tilt": SurfaceTiltTolerance,
}

for _name, _cls in _BUILTIN.items():
    if _name not in tolerance_registry:
        tolerance_registry.register(_name, _cls)

__all__ = [
    "ConicTolerance",
    "FringesTolerance",
    "RadiusTolerance",
    "SurfaceDecenterTolerance",
    "SurfaceTiltTolerance",
    "ThicknessTolerance",
    "VariableTolerance",
]
