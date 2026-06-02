"""Native stepped optimizer subpackage.

Phase 1: TorchOptimizer (Adam / SGD).
Phase 2: LevenbergMarquardt, GaussNewton.

Kramer Harrison, 2026
"""

from __future__ import annotations

import contextlib

from .base import SteppedOptimizer  # noqa: F401

with contextlib.suppress(ImportError, ModuleNotFoundError):
    from .torch_opt import TorchOptimizer  # noqa: F401

from .least_squares import GaussNewton, LevenbergMarquardt  # noqa: F401
