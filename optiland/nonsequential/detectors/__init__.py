"""Detectors subpackage for Non-Sequential Raytracing."""

from __future__ import annotations

from .base import BaseDetector
from .far_field import FarFieldDetector
from .irradiance import IrradianceDetector
from .ray_database import RayDatabaseDetector
from .spectral import SpectralDetector

__all__ = [
    "BaseDetector",
    "FarFieldDetector",
    "IrradianceDetector",
    "RayDatabaseDetector",
    "SpectralDetector",
]
