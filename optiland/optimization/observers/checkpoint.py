"""CheckpointObserver — periodic state snapshots.

Writes a pickled copy of ``OptimizationState`` to disk every ``every``
accepted steps so that long runs can be resumed after interruption.

Files are written to ``directory`` with names::

    <prefix>_iter{iteration:06d}.pkl

A ``_final`` suffix is appended on ``on_end``.

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
    """Saves periodic optimization state snapshots to disk.

    Writes a pickled ``OptimizationState`` every ``every`` accepted steps.
    An additional ``_final`` snapshot is written in ``on_end``.

    Args:
        directory: Directory in which to write checkpoint files.  Created
            automatically if it does not exist.
        every: Number of accepted steps between checkpoints (default 10).
        prefix: Filename prefix (default ``"checkpoint"``).
        keep_last: If > 0, keep only the most recent N checkpoint files and
            delete older ones (default 0 = keep all).
    """

    def __init__(
        self,
        directory: str | Path,
        every: int = 10,
        prefix: str = "checkpoint",
        keep_last: int = 0,
    ) -> None:
        self._dir = Path(directory)
        self._every = every
        self._prefix = prefix
        self._keep_last = keep_last
        self._written: list[Path] = []

    # ------------------------------------------------------------------
    # Observer hooks
    # ------------------------------------------------------------------

    def on_start(self, state: OptimizationState) -> None:
        """Create the checkpoint directory if necessary."""
        self._dir.mkdir(parents=True, exist_ok=True)

    def on_step(self, state: OptimizationState) -> None:
        """Write a checkpoint if iteration is a multiple of ``every``."""
        if state.iteration > 0 and state.iteration % self._every == 0:
            self._save(state)

    def on_end(
        self,
        state: OptimizationState,
        result: OptimizationResult,  # noqa: ARG002
    ) -> None:
        """Write the final checkpoint."""
        self._save(state, suffix="_final")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save(self, state: OptimizationState, suffix: str = "") -> None:
        fname = self._dir / f"{self._prefix}_iter{state.iteration:06d}{suffix}.pkl"
        with contextlib.suppress(Exception):
            with fname.open("wb") as fh:
                pickle.dump(copy.deepcopy(state), fh)
            self._written.append(fname)
        if self._keep_last > 0:
            self._prune()

    def _prune(self) -> None:
        while len(self._written) > self._keep_last:
            oldest = self._written.pop(0)
            with contextlib.suppress(FileNotFoundError):
                oldest.unlink()
