from __future__ import annotations

from .base import BaseSolve
from .curvature import (
    ChiefRayAngleCurvatureSolve,
    CurvatureSolve,
    MarginalRayAngleCurvatureSolve,
)
from .factory import SolveFactory
from .quick_focus import QuickFocusSolve
from .solve_manager import SolveManager
from .thickness import (
    ChiefRayHeightThicknessSolve,
    MarginalRayHeightThicknessSolve,
    ThicknessSolve,
)

# Backwards compatibility aliases
RayHeightSolveBase = ThicknessSolve
MarginalRayHeightSolve = MarginalRayHeightThicknessSolve
ChiefRayHeightSolve = ChiefRayHeightThicknessSolve
MarginalRayAngleSolve = MarginalRayAngleCurvatureSolve
ChiefRayAngleSolve = ChiefRayAngleCurvatureSolve

__all__ = [
    "BaseSolve",
    "ThicknessSolve",
    "MarginalRayHeightThicknessSolve",
    "MarginalRayHeightSolve",
    "ChiefRayHeightThicknessSolve",
    "ChiefRayHeightSolve",
    "CurvatureSolve",
    "MarginalRayAngleCurvatureSolve",
    "MarginalRayAngleSolve",
    "ChiefRayAngleCurvatureSolve",
    "ChiefRayAngleSolve",
    "QuickFocusSolve",
    "SolveFactory",
    "SolveManager",
    "RayHeightSolveBase",
]
