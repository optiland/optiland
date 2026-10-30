"""Stopping criteria (experimental).

Kramer Harrison, 2026
"""

from __future__ import annotations

from .base import StoppingBase, StoppingCriterion
from .composite import AndCriterion, OrCriterion
from .criteria import CostTolerance, GradNormTolerance, MaxEvals, MaxIter, StepStall

__all__ = [
    "AndCriterion",
    "CostTolerance",
    "GradNormTolerance",
    "MaxEvals",
    "MaxIter",
    "OrCriterion",
    "StepStall",
    "StoppingBase",
    "StoppingCriterion",
]
