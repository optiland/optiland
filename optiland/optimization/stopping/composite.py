"""Composite stopping criteria: And / Or.

Produced by the ``&`` and ``|`` operators on ``StoppingBase`` subclasses.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import StoppingBase, StoppingCriterion

if TYPE_CHECKING:
    from optiland.optimization.state import OptimizationState


class AndCriterion(StoppingBase):
    """Stop only when *both* criteria say stop.

    The ``|`` of two criteria stops when *either* fires; ``&`` requires both.
    This is used to combine a hard limit (``MaxIter``) with a convergence
    test so the run only stops when both have triggered simultaneously
    — prefer ``OrCriterion`` for the more common "stop on whichever fires
    first" pattern.
    """

    def __init__(self, a: StoppingCriterion, b: StoppingCriterion):
        self._a = a
        self._b = b

    def reset(self, state: OptimizationState) -> None:
        self._a.reset(state)
        self._b.reset(state)

    def should_stop(self, state: OptimizationState) -> tuple[bool, str | None]:
        sa, ra = self._a.should_stop(state)
        sb, rb = self._b.should_stop(state)
        if sa and sb:
            return True, f"({ra}) and ({rb})"
        return False, None


class OrCriterion(StoppingBase):
    """Stop when *either* criterion fires.

    Short-circuits: once the first criterion says stop, the second is still
    evaluated (for its side effects / reset tracking) but the result is
    dominated by the first.
    """

    def __init__(self, a: StoppingCriterion, b: StoppingCriterion):
        self._a = a
        self._b = b

    def reset(self, state: OptimizationState) -> None:
        self._a.reset(state)
        self._b.reset(state)

    def should_stop(self, state: OptimizationState) -> tuple[bool, str | None]:
        sa, ra = self._a.should_stop(state)
        sb, rb = self._b.should_stop(state)
        if sa:
            return True, ra
        if sb:
            return True, rb
        return False, None
