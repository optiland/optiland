"""CheckpointObserver: periodic state snapshots on disk.

Experimental: part of the opt v2 facade, subject to change until M10.

Kramer Harrison, 2026
"""

from __future__ import annotations

import contextlib
import copy
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.optimization.state import OptimizationResult, OptimizationState


class CheckpointObserver:
    """Pickles the :class:`OptimizationState` every ``every`` accepted steps.

    Files are named ``<prefix>_iter<iteration:06d>.pkl``; ``on_end`` writes one
    more with a ``_final`` suffix. A failed write never stops the run.
    *Experimental.*

    Args:
        directory: Target directory, created on start if missing.
        every: Steps between checkpoints.
        prefix: File name prefix.
        keep_last: Keep only the newest N files (0 keeps all).
    """

    def __init__(
        self,
        directory: str | Path,
        every: int = 10,
        prefix: str = "checkpoint",
        keep_last: int = 0,
    ) -> None:
        self._dir = Path(directory)
        self._every = max(int(every), 1)
        self._prefix = prefix
        self._keep_last = keep_last
        self._written: list[Path] = []

    def on_start(self, state: OptimizationState) -> None:  # noqa: ARG002
        """Create the checkpoint directory."""
        self._dir.mkdir(parents=True, exist_ok=True)

    def on_step(self, state: OptimizationState) -> None:
        """Write a checkpoint on every ``every``-th step."""
        if state.iteration > 0 and state.iteration % self._every == 0:
            self._save(state)

    def on_end(self, state: OptimizationState, result: OptimizationResult) -> None:  # noqa: ARG002
        """Write the final checkpoint."""
        self._save(state, suffix="_final")

    def _save(self, state: OptimizationState, suffix: str = "") -> None:
        path = self._dir / f"{self._prefix}_iter{state.iteration:06d}{suffix}.pkl"
        with contextlib.suppress(Exception):
            with path.open("wb") as handle:
                pickle.dump(copy.deepcopy(state), handle)
            self._written.append(path)
        while self._keep_last > 0 and len(self._written) > self._keep_last:
            with contextlib.suppress(FileNotFoundError):
                self._written.pop(0).unlink()
