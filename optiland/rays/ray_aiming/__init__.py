"""
Ray Aiming Package

This package contains infrastructure and algorithms for aiming rays at the
stop surface in optical systems.

Kramer Harrison, 2025
"""

from __future__ import annotations

from optiland.rays.ray_aiming.base import BaseRayAimer
from optiland.rays.ray_aiming.iterative import IterativeRayAimer
from optiland.rays.ray_aiming.paraxial import ParaxialRayAimer
from optiland.rays.ray_aiming.registry import create_ray_aimer, register_aimer
from optiland.rays.ray_aiming.robust import RobustRayAimer

__all__ = [
    "BaseRayAimer",
    "ParaxialRayAimer",
    "IterativeRayAimer",
    "RobustRayAimer",
    "create_ray_aimer",
    "register_aimer",
]
