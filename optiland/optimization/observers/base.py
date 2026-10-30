"""Observer protocol.

Experimental: part of the opt v2 facade, subject to change until M10.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from optiland.optimization.state import OptimizationResult, OptimizationState


@runtime_checkable
class Observer(Protocol):
    """Read-only per-step consumer of an optimization run. *Experimental.*

    ``on_start`` runs once before the first step, ``on_step`` after every
    accepted step, and ``on_end`` once after the final result is built.
    Observers must not modify the state. Raising ``StopIteration`` from
    ``on_step`` ends the run cleanly.
    """

    def on_start(self, state: OptimizationState) -> None:
        """Called before the first step."""
        ...  # pragma: no cover

    def on_step(self, state: OptimizationState) -> None:
        """Called after every accepted step."""
        ...  # pragma: no cover

    def on_end(self, state: OptimizationState, result: OptimizationResult) -> None:
        """Called once with the final state and result."""
        ...  # pragma: no cover
