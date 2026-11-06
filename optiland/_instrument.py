"""Private evaluation counters for tests and benchmarks.

Hot paths call :func:`count` with a fixed event name. While counting is
disabled (the default) the call is a single module-level boolean check, so it
is safe to leave in place permanently. Counting is switched on only through
the :func:`counting` context manager, which is meant for tests and the
benchmark harness, never for library users.

Events:
    ``trace``: :meth:`Optic.trace` calls.
    ``trace_generic``: :meth:`Optic.trace_generic` calls.
    ``paraxial_trace``: :meth:`Paraxial.trace` and :meth:`Paraxial.trace_generic`
        calls.
    ``material_n``: :meth:`BaseMaterial.n` calls that run the regular
        evaluation path. Calls answered by an evaluation-epoch memo
        (:mod:`optiland._epoch`) are not counted.

Kramer Harrison, 2026
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

EVENTS = ("trace", "trace_generic", "paraxial_trace", "material_n")

_enabled = False
_counts: dict[str, int] = dict.fromkeys(EVENTS, 0)

if TYPE_CHECKING:
    from collections.abc import Iterator


def count(event: str) -> None:
    """Record one occurrence of ``event`` if counting is enabled.

    Args:
        event: One of :data:`EVENTS`.
    """
    if _enabled:
        _counts[event] += 1


def snapshot() -> dict[str, int]:
    """Return a copy of the current counts."""
    return dict(_counts)


@contextmanager
def counting() -> Iterator[dict[str, int]]:
    """Enable counting from zero for the duration of the block.

    Yields:
        A live dict of event counts, updated as the block runs. It keeps its
        final values after the block exits. Nested use is not supported.

    Raises:
        RuntimeError: If counting is already active.
    """
    global _enabled, _counts
    if _enabled:
        raise RuntimeError("counting() is not re-entrant")
    previous = _counts
    live = dict.fromkeys(EVENTS, 0)
    _counts = live
    _enabled = True
    try:
        yield live
    finally:
        _enabled = False
        _counts = previous
