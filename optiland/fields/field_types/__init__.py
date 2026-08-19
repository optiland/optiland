"""Field Types Package

This package defines different field types for optical systems, including angle,
object height, paraxial image height, and real image height fields. Each field
type implements methods to calculate ray origins and paraxial object positions
based on the field definition.

Kramer Harrison, 2025
"""

from __future__ import annotations

from .angle import AngleField
from .base import BaseFieldDefinition
from .object_height import ObjectHeightField
from .paraxial_image_height import ParaxialImageHeightField
from .real_image_height import RealImageHeightField

__all__ = [
    "BaseFieldDefinition",
    "AngleField",
    "ObjectHeightField",
    "ParaxialImageHeightField",
    "RealImageHeightField",
]
