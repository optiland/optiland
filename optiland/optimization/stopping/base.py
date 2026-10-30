"""Stopping criterion protocol and composition base.

Experimental: part of the opt v2 facade, subject to change until M10.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from optiland.optimization.state import OptimizationState
    from optiland.optimization.stopping.composite import AndCriterion, OrCriterion


@runtime_checkable
class StoppingCriterion(Protocol):
    """Protocol of a stopping criterion. *Experimental.*

    ``should_stop`` is called once per accepted step, after the observers.
    """

    def reset(self, state: OptimizationState) -> None:
        """Clear internal state at the start of a run."""
        ...  # pragma: no cover

    def should_stop(self, state: OptimizationState) -> tuple[bool, str | None]:
        """Return ``(stop, reason)``."""
        ...  # pragma: no cover


class StoppingBase:
    """Base class that adds ``|`` and ``&`` composition and no-op defaults."""

    def reset(self, state: OptimizationState) -> None:  # noqa: ARG002
        """Clear internal state at the start of a run."""

    def should_stop(self, state: OptimizationState) -> tuple[bool, str | None]:  # noqa: ARG002
        """Return ``(False, None)``."""
        return False, None

    def __or__(self, other: StoppingCriterion) -> OrCriterion:
        from .composite import OrCriterion

        return OrCriterion(self, other)

    def __and__(self, other: StoppingCriterion) -> AndCriterion:
        from .composite import AndCriterion

        return AndCriterion(self, other)
