"""CancelToken: thread-safe early stop.

Experimental: part of the opt v2 facade, subject to change until M10.

Kramer Harrison, 2026
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from optiland.optimization.stopping.base import StoppingBase
from optiland.optimization.stopping.composite import OrCriterion

if TYPE_CHECKING:
    from optiland.optimization.state import OptimizationState
    from optiland.optimization.stopping.base import StoppingCriterion


class CancelToken:
    """Thread-safe cancellation flag. *Experimental.*

    Call :meth:`cancel` from any thread (for example a GUI handler); the run
    stops cleanly at the next accepted step.

    Example:
        >>> token = CancelToken()
        >>> token.cancelled
        False
        >>> token.cancel()
        >>> token.cancelled
        True
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Request cancellation."""
        self._event.set()

    def reset(self) -> None:
        """Clear the flag so the token can be reused."""
        self._event.clear()

    @property
    def cancelled(self) -> bool:
        """True once cancellation has been requested."""
        return self._event.is_set()


class CancelCriterion(StoppingBase):
    """Stopping criterion that fires when a :class:`CancelToken` is set."""

    def __init__(self, token: CancelToken) -> None:
        self._token = token

    def should_stop(self, state: OptimizationState) -> tuple[bool, str | None]:  # noqa: ARG002
        """Stop when the token has been cancelled."""
        if self._token.cancelled:
            return True, "cancelled by user"
        return False, None


def with_cancel(
    stop: StoppingCriterion | None, token: CancelToken | None
) -> StoppingCriterion | None:
    """Combine a stopping criterion with a cancel token (either stops the run)."""
    if token is None:
        return stop
    cancel = CancelCriterion(token)
    return cancel if stop is None else OrCriterion(stop, cancel)
