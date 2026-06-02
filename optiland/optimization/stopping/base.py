"""StoppingCriterion protocol and mixin base.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from optiland.optimization.state import OptimizationState
    from optiland.optimization.stopping.composite import AndCriterion, OrCriterion


@runtime_checkable
class StoppingCriterion(Protocol):
    """Protocol every stopping criterion must implement.

    ``should_stop`` is called once per **accepted** iteration by the driver.
    It must be free of side effects visible outside the criterion itself.
    """

    def reset(self, state: OptimizationState) -> None:
        """Reset internal state at the start of a new run."""
        ...  # pragma: no cover

    def should_stop(self, state: OptimizationState) -> tuple[bool, str | None]:
        """Return ``(stop, reason)``; ``reason`` may be None when not stopping."""
        ...  # pragma: no cover


class StoppingBase:
    """Concrete base that provides ``__and__`` / ``__or__`` composition and
    default no-op implementations."""

    def reset(self, state: OptimizationState) -> None:  # noqa: ARG002
        pass

    def should_stop(self, state: OptimizationState) -> tuple[bool, str | None]:  # noqa: ARG002
        return False, None

    def __and__(self, other: StoppingCriterion) -> AndCriterion:
        from .composite import AndCriterion

        return AndCriterion(self, other)

    def __or__(self, other: StoppingCriterion) -> OrCriterion:
        from .composite import OrCriterion

        return OrCriterion(self, other)
