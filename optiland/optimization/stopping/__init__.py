"""Stopping criteria subpackage.

Kramer Harrison, 2026
"""

from __future__ import annotations

from .base import StoppingBase, StoppingCriterion  # noqa: F401
from .composite import AndCriterion, OrCriterion  # noqa: F401
from .criteria import (  # noqa: F401
    CostTolerance,
    GradNormTolerance,
    MaxEvals,
    MaxIter,
    RelImprovement,
    StepStall,
)
