"""Detectors subpackage for Non-Sequential Raytracing."""

from __future__ import annotations

from .base import BaseDetector
from .configs import (
    FarFieldDetectorConfig,
    IrradianceDetectorConfig,
    RayDatabaseConfig,
    SpectralDetectorConfig,
)
from .far_field import FarFieldDetector
from .irradiance import IrradianceDetector
from .ray_database import RayDatabaseDetector
from .registry import DetectorRegistry
from .spectral import SpectralDetector

__all__ = [
    "BaseDetector",
    "DetectorRegistry",
    "FarFieldDetector",
    "FarFieldDetectorConfig",
    "IrradianceDetector",
    "IrradianceDetectorConfig",
    "RayDatabaseDetector",
    "RayDatabaseConfig",
    "SpectralDetector",
    "SpectralDetectorConfig",
]
