"""CancelToken — thread-safe early-stop hook (D-GUI / MUTATE_TERMINATION).

A ``CancelToken`` can be set from any thread (e.g. a GUI button handler).
The driver checks it via a ``StoppingCriterion`` adapter at the end of each
accepted step, triggering a clean early stop without mid-iteration races.

Usage::

    token = CancelToken()
    result = minimize(problem, ..., cancel_token=token)

    # from another thread / GUI handler:
    token.cancel()

Kramer Harrison, 2026
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from optiland.optimization.stopping.base import StoppingBase

if TYPE_CHECKING:
    from optiland.optimization.state import OptimizationResult, OptimizationState


class CancelToken:
    """Thread-safe cancellation token.

    Set ``cancelled`` to True from any thread to request an early stop at
    the next accepted step boundary.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation."""
        with self._lock:
            self._cancelled = True

    def reset(self) -> None:
        """Clear the cancellation flag (reusable token)."""
        with self._lock:
            self._cancelled = False

    @property
    def cancelled(self) -> bool:
        """True if cancellation has been requested."""
        with self._lock:
            return self._cancelled


class _CancelCriterion(StoppingBase):
    """Internal adapter that exposes a CancelToken as a StoppingCriterion."""

    def __init__(self, token: CancelToken):
        self._token = token

    def should_stop(self, state: OptimizationState) -> tuple[bool, str | None]:
        if self._token.cancelled:
            return True, "cancelled by user"
        return False, None


class CancelObserver:
    """Observer that raises ``StopIteration`` in a SciPy callback when the
    token is set, enabling early exit from managed (SciPy) optimizers that
    support the modern ``StopIteration`` convention.

    This is attached automatically by ``ManagedDriver`` when a cancel_token
    is provided.
    """

    def __init__(self, token: CancelToken):
        self._token = token

    def on_start(self, state: OptimizationState) -> None:
        pass

    def on_step(self, state: OptimizationState) -> None:
        if self._token.cancelled:
            raise StopIteration("cancelled by user")

    def on_end(self, state: OptimizationState, result: OptimizationResult) -> None:
        pass
