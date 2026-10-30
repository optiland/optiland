"""Trace Resume Hook Module

Private support for partial-trace finite differences (opt v2 PR4.1). The
finite-difference evaluator evaluates the baseline design inside
:func:`recording`, which makes every :meth:`RealRayTracer.trace_generic` call
store a :class:`~optiland.raytrace.snapshot.TraceSnapshot` and the recorded
surface data. It then evaluates each Jacobian column whose variable changes
only surface ``s`` (see
:func:`~optiland.raytrace.partial_trace.partial_trace_start`) inside
:func:`resuming`, which makes the same calls resume at ``s`` from the stored
state instead of tracing from the object.

A call is matched to its baseline trace by the optic, the wavelength and the
exact field and pupil coordinates, so the launched rays are the same rays.
Surfaces before ``s`` get back the data recorded by the matching baseline
trace, because other traces of the evaluation overwrite it in between. Any
call without a match traces in full.

Outside these context managers :func:`active` returns None and the tracer is
unchanged.

Kramer Harrison, 2026
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

import numpy as np

import optiland.backend as be

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterator

    from optiland.rays import RealRays
    from optiland.raytrace.real_ray_tracer import RealRayTracer
    from optiland.raytrace.snapshot import TraceSnapshot

# Per-surface attributes written by a trace (see ``Surface.reset``).
_SURFACE_ATTRS = ("x", "y", "z", "L", "M", "N", "intensity", "aoi", "opd", "u")


class TraceStore:
    """Snapshots of the traces of one baseline evaluation.

    Attributes:
        resumed: Number of traces resumed from this store.
    """

    def __init__(self) -> None:
        self._entries: dict[Hashable, tuple[TraceSnapshot, list[dict]]] = {}
        self.resumed = 0

    def __len__(self) -> int:
        return len(self._entries)

    def put(self, key: Hashable, snapshot: TraceSnapshot) -> None:
        """Store a snapshot with the surface data its trace just recorded."""
        records = [
            {
                name: getattr(surface, name)
                for name in _SURFACE_ATTRS
                if hasattr(surface, name)
            }
            for surface in snapshot.surfaces
        ]
        self._entries[key] = (snapshot, records)

    def get(self, key: Hashable) -> tuple[TraceSnapshot, list[dict]] | None:
        """The snapshot and surface data stored under ``key``, or None."""
        return self._entries.get(key)


class _Hook:
    """Active hook: record into a store, or resume from it at one surface."""

    __slots__ = ("optic_id", "start", "store")

    def __init__(self, store: TraceStore, optic_id: int | None, start: int | None):
        self.store = store
        self.optic_id = optic_id
        self.start = start

    def trace_generic(
        self,
        tracer: RealRayTracer,
        Hx: Any,
        Hy: Any,
        Px: Any,
        Py: Any,
        wavelength: Any,
        record: bool,
    ) -> RealRays | None:
        """Trace through the hook; None means the caller traces as usual."""
        key = _trace_key(tracer.optic, Hx, Hy, Px, Py, wavelength, record)
        if key is None:
            return None
        if self.start is None:
            rays, snapshot = tracer._trace(Hx, Hy, Px, Py, wavelength, record)
            self.store.put(key, snapshot)
            return rays
        if id(tracer.optic) != self.optic_id:
            return None
        entry = self.store.get(key)
        if entry is None:
            return None
        snapshot, records = entry
        surfaces = tracer.optic.surfaces.surfaces
        if not snapshot.matches(surfaces) or self.start >= len(snapshot):
            return None
        for surface, data in zip(surfaces[: self.start], records, strict=False):
            for name, value in data.items():
                setattr(surface, name, value)
        self.store.resumed += 1
        return tracer._trace_from(self.start, snapshot, record)


_hook: ContextVar[_Hook | None] = ContextVar("optiland_trace_resume", default=None)


def active() -> _Hook | None:
    """The hook active in this context, or None."""
    return _hook.get()


@contextmanager
def recording(store: TraceStore) -> Iterator[TraceStore]:
    """Record a snapshot of every ``trace_generic`` call into ``store``.

    Args:
        store: The store to fill.

    Yields:
        The store.
    """
    token = _hook.set(_Hook(store, None, None))
    try:
        yield store
    finally:
        _hook.reset(token)


@contextmanager
def resuming(store: TraceStore, optic: Any, start: int) -> Iterator[TraceStore]:
    """Resume matching ``trace_generic`` calls on ``optic`` at surface ``start``.

    Args:
        store: A store filled by :func:`recording` at the baseline design.
        optic: The only optic whose traces resume; others trace in full.
        start: The surface to resume at.

    Yields:
        The store.
    """
    token = _hook.set(_Hook(store, id(optic), start))
    try:
        yield store
    finally:
        _hook.reset(token)


def _array_key(value: Any) -> tuple:
    data = np.ascontiguousarray(be.to_numpy(value), dtype=float)
    return (data.shape, data.tobytes())


def _trace_key(
    optic: Any, Hx: Any, Hy: Any, Px: Any, Py: Any, wavelength: Any, record: bool
) -> Hashable | None:
    """Identity of one trace call, or None when it cannot be keyed."""
    try:
        return (
            id(optic),
            bool(record),
            _array_key(wavelength),
            _array_key(Hx),
            _array_key(Hy),
            _array_key(Px),
            _array_key(Py),
        )
    except (TypeError, ValueError):
        return None
