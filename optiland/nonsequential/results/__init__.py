"""Results subpackage for Non-Sequential Raytracing."""

from __future__ import annotations

from .far_field_pattern import FarFieldPattern
from .irradiance_map import IrradianceMap
from .ray_database import RayDatabase
from .spectral_result import SpectralResult

__all__ = [
    "FarFieldPattern",
    "IrradianceMap",
    "RayDatabase",
    "SpectralResult",
]
