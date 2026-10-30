"""Built-in stopping criteria.

Compose with ``|`` (either) and ``&`` (both, latching). Experimental: part of
the opt v2 facade, subject to change until M10.

Kramer Harrison, 2026
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be

from .base import StoppingBase

if TYPE_CHECKING:
    from optiland.optimization.state import OptimizationState


class MaxIter(StoppingBase):
    """Stop after ``n`` accepted steps. *Experimental.*

    Args:
        n: Maximum number of accepted steps.
    """

    def __init__(self, n: int = 1000) -> None:
        self._n = n

    def should_stop(self, state: OptimizationState) -> tuple[bool, str | None]:
        """Stop once ``state.iteration >= n``."""
        if state.iteration >= self._n:
            return True, f"max_iter={self._n} reached"
        return False, None


class MaxEvals(StoppingBase):
    """Stop after ``n`` evaluations, Jacobian columns included. *Experimental.*

    Args:
        n: Maximum number of evaluations.
    """

    def __init__(self, n: int) -> None:
        self._n = n

    def should_stop(self, state: OptimizationState) -> tuple[bool, str | None]:
        """Stop once ``state.n_evals >= n``."""
        if state.n_evals >= self._n:
            return True, f"max_evals={self._n} reached"
        return False, None


class CostTolerance(StoppingBase):
    """Stop when the relative merit change of a step falls below ``tol``.

    Tests ``|prev - value| / max(|prev|, 1e-12) < tol``. It never fires before
    the first accepted step. *Experimental.*

    Args:
        tol: Relative tolerance.
    """

    def __init__(self, tol: float = 1e-3) -> None:
        self._tol = tol
        self._prev: float | None = None

    def reset(self, state: OptimizationState) -> None:
        """Remember the starting merit."""
        self._prev = float(state.value)

    def should_stop(self, state: OptimizationState) -> tuple[bool, str | None]:
        """Stop when the merit barely changed since the previous step."""
        if state.iteration == 0:
            return False, None
        value = float(state.value)
        prev, self._prev = self._prev, value
        if prev is None or math.isnan(value):
            return False, None
        rel = abs(prev - value) / max(abs(prev), 1e-12)
        if rel < self._tol:
            return True, f"cost_tol={self._tol:.2e} (rel_change={rel:.2e})"
        return False, None


class GradNormTolerance(StoppingBase):
    """Stop when the gradient L2 norm falls below ``tol``. *Experimental.*

    Inactive while the state carries no gradient.

    Args:
        tol: Absolute tolerance on the gradient norm.
    """

    def __init__(self, tol: float = 1e-6) -> None:
        self._tol = tol

    def should_stop(self, state: OptimizationState) -> tuple[bool, str | None]:
        """Stop when ``||gradient|| < tol``."""
        norm = state.grad_norm
        if norm is None:
            if state.gradient is None:
                return False, None
            norm = float(np.linalg.norm(be.to_numpy(state.gradient)))
        if norm < self._tol:
            return True, f"grad_norm={norm:.2e} < tol={self._tol:.2e}"
        return False, None


class StepStall(StoppingBase):
    """Stop when the step norm ``||dx||`` falls below ``tol``. *Experimental.*

    Inactive while the state carries no step.

    Args:
        tol: Absolute tolerance on the step norm.
    """

    def __init__(self, tol: float = 1e-8) -> None:
        self._tol = tol

    def should_stop(self, state: OptimizationState) -> tuple[bool, str | None]:
        """Stop when ``state.step_norm < tol``."""
        if state.step_norm is None:
            return False, None
        if state.step_norm < self._tol:
            return True, f"step_stall: |dx|={state.step_norm:.2e} < tol={self._tol:.2e}"
        return False, None
