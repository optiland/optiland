"""NSQ components subpackage."""

from __future__ import annotations

from .absorbing import AbsorbingComponent
from .base import BaseComponent
from .compound import CompoundComponent
from .configs import (
    DoubletConfig,
    InteractionType,
    LensConfig,
    MirrorConfig,
    SurfaceConfig,
)
from .doublet import Doublet
from .lens import Lens
from .mirror import Mirror
from .reflective import ReflectiveComponent
from .refractive import RefractiveComponent
from .registry import ComponentRegistry
from .volume import NonWatertightVolumeError, Volume

__all__ = [
    "AbsorbingComponent",
    "BaseComponent",
    "CompoundComponent",
    "ComponentRegistry",
    "DoubletConfig",
    "Doublet",
    "InteractionType",
    "Lens",
    "LensConfig",
    "Mirror",
    "MirrorConfig",
    "NonWatertightVolumeError",
    "RefractiveComponent",
    "ReflectiveComponent",
    "SurfaceConfig",
    "Volume",
]
