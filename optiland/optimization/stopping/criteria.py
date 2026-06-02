"""Built-in stopping criteria.

Criteria are composable via ``&`` / ``|`` (see ``composite.py``).

Kramer Harrison, 2026
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import optiland.backend as be

from .base import StoppingBase

if TYPE_CHECKING:
    from optiland.optimization.state import OptimizationState


class MaxIter(StoppingBase):
    """Stop after ``n`` accepted iterations.

    Args:
        n: Maximum number of accepted steps (default 1000).
    """

    def __init__(self, n: int = 1000):
        self._n = n

    def should_stop(self, state: OptimizationState) -> tuple[bool, str | None]:
        if state.iteration >= self._n:
            return True, f"max_iter={self._n} reached"
        return False, None


class MaxEvals(StoppingBase):
    """Stop after ``n`` total value evaluations.

    Args:
        n: Maximum number of value evaluations.
    """

    def __init__(self, n: int):
        self._n = n

    def should_stop(self, state: OptimizationState) -> tuple[bool, str | None]:
        if state.n_value_evals >= self._n:
            return True, f"max_evals={self._n} reached"
        return False, None


class CostTolerance(StoppingBase):
    """Stop when the relative change in merit falls below ``tol``.

    Tracks the previous merit value and checks
    ``|prev - current| / max(|prev|, 1e-12) < tol``.

    Args:
        tol: Relative tolerance (default 1e-3).
    """

    def __init__(self, tol: float = 1e-3):
        self._tol = tol
        self._prev: float | None = None

    def reset(self, state: OptimizationState) -> None:
        v = state.value
        self._prev = float(be.to_numpy(v)) if hasattr(v, "item") else float(v)

    def should_stop(self, state: OptimizationState) -> tuple[bool, str | None]:
        # Never fire before any accepted step — at iteration 0 the optimizer
        # has not moved yet, so a zero relative change is expected and
        # meaningless.
        if state.iteration == 0:
            return False, None
        v = state.value
        curr = float(be.to_numpy(v)) if hasattr(v, "item") else float(v)
        if self._prev is None or math.isnan(curr):
            self._prev = curr
            return False, None
        denom = max(abs(self._prev), 1e-12)
        rel = abs(self._prev - curr) / denom
        self._prev = curr
        if rel < self._tol:
            return True, f"cost_tol={self._tol:.2e} (rel_change={rel:.2e})"
        return False, None


class GradNormTolerance(StoppingBase):
    """Stop when the gradient norm falls below ``tol``.

    Only active when ``state.gradient`` is not None.

    Args:
        tol: Absolute tolerance on the gradient L2 norm (default 1e-6).
    """

    def __init__(self, tol: float = 1e-6):
        self._tol = tol

    def should_stop(self, state: OptimizationState) -> tuple[bool, str | None]:
        if state.gradient is None:
            return False, None
        import numpy as np

        g = be.to_numpy(state.gradient)
        norm = float(np.linalg.norm(g))
        if norm < self._tol:
            return True, f"grad_norm={norm:.2e} < tol={self._tol:.2e}"
        return False, None


class StepStall(StoppingBase):
    """Stop when the step norm ``|Δx|`` falls below ``tol``.

    Only active when ``state.step_norm`` is not None.

    Args:
        tol: Absolute tolerance (default 1e-8).
    """

    def __init__(self, tol: float = 1e-8):
        self._tol = tol

    def should_stop(self, state: OptimizationState) -> tuple[bool, str | None]:
        if state.step_norm is None:
            return False, None
        if state.step_norm < self._tol:
            return True, f"step_stall: |Δx|={state.step_norm:.2e} < tol={self._tol:.2e}"
        return False, None


class RelImprovement(StoppingBase):
    """Stop when relative improvement over the initial value falls below ``tol``.

    Useful for capping the run once diminishing returns set in.

    Args:
        tol: Minimum acceptable relative improvement fraction (default 1e-4).
    """

    def __init__(self, tol: float = 1e-4):
        self._tol = tol
        self._initial: float | None = None

    def reset(self, state: OptimizationState) -> None:
        v = state.value
        self._initial = float(be.to_numpy(v)) if hasattr(v, "item") else float(v)

    def should_stop(self, state: OptimizationState) -> tuple[bool, str | None]:
        if self._initial is None or self._initial == 0.0:
            return False, None
        v = state.value
        curr = float(be.to_numpy(v)) if hasattr(v, "item") else float(v)
        rel = (self._initial - curr) / abs(self._initial)
        if rel < self._tol:
            return True, f"rel_improvement={rel:.2e} < tol={self._tol:.2e}"
        return False, None
