"""BoxBoundsStrategy — thin reader of ``var.bounds`` (D6).

Reads bounds from ``variable.bounds`` (already in scaled space) and:
  - applies a clamp in ``apply_to_step`` for stepped optimizers.
  - forwards the bounds list to SciPy via ``to_scipy`` for managed optimizers.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from optiland.optimization.state import OptimizationState


class BoxBoundsStrategy:
    """Reads ``var.bounds`` and exposes them to optimizers.

    Args:
        variables: The variable list from ``OptimizationProblem.variables``.
    """

    def __init__(self, variables: Any = None):
        self._variables = variables
        self._bounds: list[tuple] = []

    def prepare(self, evaluator: Any, variables: Any) -> None:
        self._variables = variables
        self._bounds = [var.bounds for var in variables]

    def apply_to_step(self, x_proposed: Any, state: OptimizationState) -> Any:  # noqa: ARG002
        """Clamp ``x_proposed`` to variable bounds."""
        x = np.asarray(x_proposed, dtype=float)
        for i, (lo, hi) in enumerate(self._bounds):
            if lo is not None:
                x[i] = max(x[i], lo)
            if hi is not None:
                x[i] = min(x[i], hi)
        return x

    def to_scipy(self) -> list | None:
        """Return the bounds list in SciPy ``(lo, hi)`` format."""
        if not self._bounds:
            return None
        return [
            (lo if lo is not None else -np.inf, hi if hi is not None else np.inf)
            for lo, hi in self._bounds
        ]

    def is_feasible(self, x: Any, tol: float = 1e-8) -> bool:
        x = np.asarray(x, dtype=float)
        for i, (lo, hi) in enumerate(self._bounds):
            if lo is not None and x[i] < lo - tol:
                return False
            if hi is not None and x[i] > hi + tol:
                return False
        return True
