"""HistoryObserver: per-step records.

Experimental: part of the opt v2 facade, subject to change until M10.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.optimization.state import OptimizationResult, OptimizationState


class HistoryObserver:
    """Records the iteration, merit and step norm after every accepted step.

    *Experimental.*

    Attributes:
        records: One dict per step, with keys ``iteration``, ``value`` and
            (when known) ``step_norm``. The first record is the starting point.
    """

    def __init__(self) -> None:
        self.records: list[dict] = []

    def on_start(self, state: OptimizationState) -> None:
        """Reset the records and store the starting point."""
        self.records.clear()
        self.records.append({"iteration": 0, "value": float(state.value)})

    def on_step(self, state: OptimizationState) -> None:
        """Append the state after an accepted step."""
        record = {"iteration": state.iteration, "value": float(state.value)}
        if state.step_norm is not None:
            record["step_norm"] = float(state.step_norm)
        self.records.append(record)

    def on_end(self, state: OptimizationState, result: OptimizationResult) -> None:
        """Do nothing: the records are attached to the result by the driver."""


def find_history(observers: list | None) -> list[dict] | None:
    """Return the records of the first :class:`HistoryObserver`, else None."""
    for observer in observers or ():
        if isinstance(observer, HistoryObserver):
            return list(observer.records)
    return None
