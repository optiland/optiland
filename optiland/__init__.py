"""Open-source Optical Design in Python"""

from __future__ import annotations

from optiland.optimization.operand.operand import (
    operand_registry as operand_registry,
)
from optiland.surfaces.factories.geometry_factory import (
    GeometryFactory as GeometryFactory,
)
from optiland.surfaces.factories.interaction_model_factory import (
    InteractionModelFactory as InteractionModelFactory,
)
from optiland.surfaces.factories.surface_factory import (
    SurfaceFactory as SurfaceFactory,
)
from optiland.visualization.component_renderer import (
    ComponentRenderer as ComponentRenderer,
)

__version__ = "0.6.1"
