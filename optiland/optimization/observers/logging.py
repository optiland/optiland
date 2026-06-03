"""Console and progress observers.

Both observers respect ``log_every`` to avoid printing on every iteration
(which would trigger device syncs).  Scalars are synced only when a
line is actually emitted.

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


class ConsoleObserver:
    """Prints a one-line summary every ``log_every`` accepted steps.

    Args:
        log_every: Print every Nth step (default 1, i.e. every step).
        prefix: Optional string prepended to each line.
    """

    def __init__(self, log_every: int = 1, prefix: str = ""):
        self._log_every = log_every
        self._prefix = prefix
        self._initial_value: float | None = None

    def on_start(self, state: OptimizationState) -> None:
        self._initial_value = _to_float(state.value)
        print(
            f"{self._prefix}Optimization started — "
            f"initial merit = {self._initial_value:.6g}"
        )

    def on_step(self, state: OptimizationState) -> None:
        if state.iteration % self._log_every != 0:
            return
        val = _to_float(state.value)
        print(f"{self._prefix}  iter {state.iteration:5d} | merit = {val:.6g}")

    def on_end(self, state: OptimizationState, result: OptimizationResult) -> None:
        pct = result.improvement_pct
        print(
            f"{self._prefix}Optimization ended  — "
            f"final merit = {result.value:.6g}  "
            f"({pct:+.2f}%)  [{result.status}]"
        )


class ProgressObserver:
    """Throttled observer suitable for GUI progress bars.

    Emits ``on_step`` only every ``log_every`` steps to keep the hot loop
    free of sync overhead.  The ``step_value`` property returns the
    last emitted merit as a Python float.

    Args:
        log_every: Emit every Nth accepted step (default 10).
    """

    def __init__(self, log_every: int = 10):
        self._log_every = log_every
        self.step_value: float | None = None
        self.step_iteration: int = 0

    def on_start(self, state: OptimizationState) -> None:
        self.step_value = _to_float(state.value)
        self.step_iteration = 0

    def on_step(self, state: OptimizationState) -> None:
        if state.iteration % self._log_every != 0:
            return
        self.step_value = _to_float(state.value)
        self.step_iteration = state.iteration

    def on_end(self, state: OptimizationState, result: OptimizationResult) -> None:
        self.step_value = result.value
        self.step_iteration = result.iterations
