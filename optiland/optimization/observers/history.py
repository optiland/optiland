"""HistoryObserver — records per-step snapshots.

Each accepted step appends a lightweight dict to ``self.records``.  The
list is attached to ``OptimizationResult.history`` at the end of the run.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.optimization.state import OptimizationResult, OptimizationState


def _to_float(v) -> float:
    if hasattr(v, "item"):
        return float(v.item())
    return float(v)


class HistoryObserver:
    """Records a snapshot after every accepted step.

    Attributes:
        records: List of ``dict`` with keys ``iteration``, ``value``,
            and optionally ``step_norm``.
    """

    def __init__(self):
        self.records: list[dict] = []

    def on_start(self, state: OptimizationState) -> None:
        self.records.clear()
        self.records.append({"iteration": 0, "value": _to_float(state.value)})

    def on_step(self, state: OptimizationState) -> None:
        rec = {
            "iteration": state.iteration,
            "value": _to_float(state.value),
        }
        if state.step_norm is not None:
            rec["step_norm"] = state.step_norm
        self.records.append(rec)

    def on_end(self, state: OptimizationState, result: OptimizationResult) -> None:
        pass  # records already attached via api.py
