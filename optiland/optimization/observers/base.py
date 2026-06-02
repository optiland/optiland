"""Observer protocol.

Observers are *read-only* by contract.  They must not mutate state and must
not perform expensive device synchronisation on every call (D9).

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from optiland.optimization.state import OptimizationResult, OptimizationState


@runtime_checkable
class Observer(Protocol):
    """Protocol for per-iteration observers.

    All three hooks are called exactly once per phase:
    ``on_start`` before the loop, ``on_step`` after each *accepted* step
    (D11), and ``on_end`` after the final result is built.
    """

    def on_start(self, state: OptimizationState) -> None: ...  # pragma: no cover

    def on_step(self, state: OptimizationState) -> None: ...  # pragma: no cover

    def on_end(
        self, state: OptimizationState, result: OptimizationResult
    ) -> None: ...  # pragma: no cover
