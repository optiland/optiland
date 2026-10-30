"""ConsoleObserver: progress lines on stdout.

Experimental: part of the opt v2 facade, subject to change until M10.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.optimization.state import OptimizationResult, OptimizationState


class ConsoleObserver:
    """Prints one line every ``log_every`` accepted steps. *Experimental.*

    Args:
        log_every: Print every Nth step.
        prefix: Text put in front of each line.
    """

    def __init__(self, log_every: int = 1, prefix: str = "") -> None:
        self._log_every = max(int(log_every), 1)
        self._prefix = prefix
        self._initial = 0.0

    def on_start(self, state: OptimizationState) -> None:
        """Print the starting merit."""
        self._initial = float(state.value)
        print(f"{self._prefix}start: merit = {self._initial:.6g}")

    def on_step(self, state: OptimizationState) -> None:
        """Print the merit every ``log_every`` steps."""
        if state.iteration % self._log_every == 0:
            print(
                f"{self._prefix}iter {state.iteration:5d}: "
                f"merit = {float(state.value):.6g}"
            )

    def on_end(self, state: OptimizationState, result: OptimizationResult) -> None:  # noqa: ARG002
        """Print the final merit and why the run ended."""
        final = float(state.value)
        change = (
            100.0 * (final - self._initial) / self._initial if self._initial else 0.0
        )
        print(
            f"{self._prefix}end: merit = {final:.6g} ({change:+.2f}%) "
            f"after {state.iteration} steps, {state.n_evals} evaluations "
            f"[{state.stop_reason}]"
        )
