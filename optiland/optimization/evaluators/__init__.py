"""Evaluator subpackage — differentiation-provenance adapters over OptimizationProblem.

Kramer Harrison, 2026
"""

from __future__ import annotations

import contextlib

from .base import EvalCapability, Evaluator  # noqa: F401
from .finite_difference import FiniteDiffEvaluator  # noqa: F401

with contextlib.suppress(ImportError, ModuleNotFoundError):
    from .autograd import AutogradEvaluator  # noqa: F401
