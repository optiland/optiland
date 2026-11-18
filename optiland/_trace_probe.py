"""Reports of the decisions that a trace takes for all rays of a call.

Most steps of a trace compute each ray from that ray alone. A few steps take
one decision for all rays of a call, for example whether any ray absorbs.
These steps report the decision here. Only the NumPy trace in parts
(``optiland.surfaces._chunked_trace``) reads the reports. It traces a large
bundle in parts, and the parts give the bits of the whole bundle only if
each part takes the decision that the whole bundle would take.

A step reports in one of two ways:

- ``note_decision(name, value)`` reports a decision whose outcome can change
  the bits of a ray. If the parts report different decisions, the whole
  bundle is traced.
- ``require_whole_bundle(reason)`` reports a decision that a part cannot
  take for the whole bundle, for example one that depends on the largest
  value of the call. The whole bundle is then traced.

The reader handles these two kinds only, so a new step that reports needs
no change in the reader. Outside ``recording()``, both functions return at
once.

This module is in the package root, like ``optiland._epoch``, because
``optiland.geometries`` and ``optiland.propagation`` cannot import
``optiland.surfaces`` without an import cycle.

Manuel Fragata Mendes, 2026
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterator

# The name of the reports of require_whole_bundle. It is not a string, so it
# is never equal to the name of a decision.
WHOLE_BUNDLE = object()

_REPORTS: contextvars.ContextVar[list[tuple[object, Hashable]] | None] = (
    contextvars.ContextVar("optiland_trace_probe", default=None)
)


@contextmanager
def recording() -> Iterator[list[tuple[object, Hashable]]]:
    """Record the reports of the steps that run inside the block.

    Yields:
        The reports, in the order of the calls. A decision is reported as
        ``(name, value)``, a request for the whole bundle as
        ``(WHOLE_BUNDLE, reason)``.
    """
    reports: list[tuple[object, Hashable]] = []
    token = _REPORTS.set(reports)
    try:
        yield reports
    finally:
        _REPORTS.reset(token)


def note_decision(name: str, value: Hashable) -> None:
    """Report a decision that is taken for all rays of the current call.

    Use it for a decision whose outcome can change the bits of a ray, so
    that each part must take the decision of the whole bundle.

    Args:
        name: The name of the decision.
        value: The outcome of the decision.
    """
    reports = _REPORTS.get()
    if reports is not None:
        reports.append((name, value))


def require_whole_bundle(reason: str) -> None:
    """Report that the current call must see all rays of the bundle.

    Use it for a decision that depends on all rays of the call, for example
    on their largest value, so that a part cannot take it for the bundle.

    Args:
        reason: Why the call must see all rays. It helps when debugging.
    """
    reports = _REPORTS.get()
    if reports is not None:
        reports.append((WHOLE_BUNDLE, reason))


def asks_for_whole_bundle(reports: list[tuple[object, Hashable]]) -> bool:
    """Return True if a step asked for the whole bundle.

    Args:
        reports: The reports of one part, from ``recording()``.

    Returns:
        True if ``require_whole_bundle`` was called for the part.
    """
    return any(name is WHOLE_BUNDLE for name, _ in reports)
