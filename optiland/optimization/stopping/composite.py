"""Composite stopping criteria, produced by ``|`` and ``&``.

Experimental: part of the opt v2 facade, subject to change until M10.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import StoppingBase, StoppingCriterion

if TYPE_CHECKING:
    from optiland.optimization.state import OptimizationState


class OrCriterion(StoppingBase):
    """Stop as soon as either criterion fires. *Experimental.*

    Both children are always evaluated, so each keeps its own bookkeeping.
    """

    def __init__(self, a: StoppingCriterion, b: StoppingCriterion) -> None:
        self._a = a
        self._b = b

    def reset(self, state: OptimizationState) -> None:
        """Reset both children."""
        self._a.reset(state)
        self._b.reset(state)

    def should_stop(self, state: OptimizationState) -> tuple[bool, str | None]:
        """Stop when either child stops; the first child's reason wins."""
        stop_a, reason_a = self._a.should_stop(state)
        stop_b, reason_b = self._b.should_stop(state)
        if stop_a:
            return True, reason_a
        if stop_b:
            return True, reason_b
        return False, None


class AndCriterion(StoppingBase):
    """Stop once both criteria have fired. *Experimental.*

    Each child latches: once it has fired, that stays remembered for the rest
    of the run even if it later reports "keep going". This makes
    ``MaxIter(n) & CostTolerance(tol)`` terminable, since ``MaxIter`` fires
    only once. :meth:`reset` clears the latches.
    """

    def __init__(self, a: StoppingCriterion, b: StoppingCriterion) -> None:
        self._a = a
        self._b = b
        self._fired_a: str | None = None
        self._fired_b: str | None = None

    def reset(self, state: OptimizationState) -> None:
        """Reset both children and clear the latches."""
        self._a.reset(state)
        self._b.reset(state)
        self._fired_a = None
        self._fired_b = None

    def should_stop(self, state: OptimizationState) -> tuple[bool, str | None]:
        """Stop on the step where the last outstanding child has fired."""
        stop_a, reason_a = self._a.should_stop(state)
        stop_b, reason_b = self._b.should_stop(state)
        if stop_a and self._fired_a is None:
            self._fired_a = reason_a or "criterion"
        if stop_b and self._fired_b is None:
            self._fired_b = reason_b or "criterion"
        if self._fired_a is not None and self._fired_b is not None:
            return True, f"({self._fired_a}) and ({self._fired_b})"
        return False, None
