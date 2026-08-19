"""Prescription section classes."""

from __future__ import annotations

from optiland.prescription.sections.base import BaseSection
from optiland.prescription.sections.first_order import FirstOrderSection
from optiland.prescription.sections.seidel import SeidelAberrationSection
from optiland.prescription.sections.surface_table import (
    SurfaceGeometryTableSection,
    SurfaceMaterialTableSection,
)
from optiland.prescription.sections.system_overview import SystemOverviewSection

__all__ = [
    "BaseSection",
    "SystemOverviewSection",
    "FirstOrderSection",
    "SurfaceGeometryTableSection",
    "SurfaceMaterialTableSection",
    "SeidelAberrationSection",
]
