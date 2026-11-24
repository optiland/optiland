"""Built-in tolerance types, registered in ``tolerance_registry`` on import."""

from __future__ import annotations

from optiland.tolerancing.registry import tolerance_registry
from optiland.tolerancing.types.element import (
    ElementDecenterTolerance,
    ElementRollTolerance,
    ElementTiltTolerance,
    ElementTolerance,
)
from optiland.tolerancing.types.irregularity import IrregularityTolerance
from optiland.tolerancing.types.material import (
    AbbeTolerance,
    IndexTolerance,
    MaterialTolerance,
)
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
    "irregularity": IrregularityTolerance,
    "conic": ConicTolerance,
    "thickness": ThicknessTolerance,
    "surface_decenter": SurfaceDecenterTolerance,
    "surface_tilt": SurfaceTiltTolerance,
    "element_decenter": ElementDecenterTolerance,
    "element_tilt": ElementTiltTolerance,
    "element_roll": ElementRollTolerance,
    "index": IndexTolerance,
    "abbe": AbbeTolerance,
}

for _name, _cls in _BUILTIN.items():
    if _name not in tolerance_registry:
        tolerance_registry.register(_name, _cls)

__all__ = [
    "AbbeTolerance",
    "ConicTolerance",
    "ElementDecenterTolerance",
    "ElementRollTolerance",
    "ElementTiltTolerance",
    "ElementTolerance",
    "FringesTolerance",
    "IndexTolerance",
    "IrregularityTolerance",
    "MaterialTolerance",
    "RadiusTolerance",
    "SurfaceDecenterTolerance",
    "SurfaceTiltTolerance",
    "ThicknessTolerance",
    "VariableTolerance",
]
