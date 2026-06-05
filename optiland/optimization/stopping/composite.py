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
    """Stop once *both* criteria have fired (latching).

    The ``|`` of two criteria stops when *either* fires; ``&`` requires both.
    Each child **latches**: once it reports stop, that decision is remembered
    for the rest of the run even if it later reports "keep going".  The
    composite stops on the iteration where the *last* outstanding child fires.

    This mirrors prysm's ``AllGovernor`` and makes patterns like
    ``MaxIter(n) & CostTolerance(tol)`` actually terminable: ``MaxIter`` fires
    exactly once (on the final iteration) and would otherwise never coincide
    with a separately-firing convergence test.  The latches are cleared by
    :meth:`reset`.

    Prefer ``OrCriterion`` for the more common "stop on whichever fires first"
    pattern.
    """

    def __init__(self, a: StoppingCriterion, b: StoppingCriterion):
        self._a = a
        self._b = b
        self._a_fired_reason: str | None = None
        self._b_fired_reason: str | None = None

    def reset(self, state: OptimizationState) -> None:
        self._a.reset(state)
        self._b.reset(state)
        self._a_fired_reason = None
        self._b_fired_reason = None

    def should_stop(self, state: OptimizationState) -> tuple[bool, str | None]:
        sa, ra = self._a.should_stop(state)
        sb, rb = self._b.should_stop(state)
        if sa and self._a_fired_reason is None:
            self._a_fired_reason = ra
        if sb and self._b_fired_reason is None:
            self._b_fired_reason = rb
        if self._a_fired_reason is not None and self._b_fired_reason is not None:
            return True, f"({self._a_fired_reason}) and ({self._b_fired_reason})"
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
