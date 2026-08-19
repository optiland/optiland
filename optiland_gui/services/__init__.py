"""Service classes for the Optiland GUI.

This package contains focused service classes extracted from ``OptilandConnector``
following the single-responsibility principle. Each service handles one domain of
the GUI's business logic. Services are plain Python classes (not QObject subclasses)
that receive a connector reference for signal emission and optic access.
"""

from __future__ import annotations

from optiland_gui.services.analysis_runner import AnalysisRunner
from optiland_gui.services.file_service import FileService
from optiland_gui.services.optimization_service import OptimizationService
from optiland_gui.services.surface_service import SurfaceService
from optiland_gui.services.system_service import SystemService

__all__ = [
    "AnalysisRunner",
    "FileService",
    "OptimizationService",
    "SurfaceService",
    "SystemService",
]
