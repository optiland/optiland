"""Trace Snapshot Module

Private support for resuming a sequential real-ray trace part-way through the
system. A :class:`TraceSnapshot` stores the ray state *before* each surface of
one trace, so a later trace whose changes are confined to surface ``k`` and
beyond can restart at ``k`` from the stored state instead of from the object.

The snapshot is private API (opt v2 PR3.1). It is meant for the
finite-difference evaluator, which decides where a trace may resume with
:func:`optiland.raytrace.partial_trace.partial_trace_start`.

Kramer Harrison, 2026
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import optiland.backend as be

if TYPE_CHECKING:
    from collections.abc import Iterable

    from optiland.rays import RealRays
    from optiland.surfaces import Surface


def capture_rays(rays: RealRays) -> RealRays:
    """Return an independent copy of a ray bundle's state.

    Every array attribute (positions, direction cosines, OPD, intensity,
    wavelength and, for polarized rays, the polarization matrices) is copied,
    so later in-place updates of either bundle never reach the other. Scalar
    and immutable attributes are shared.

    Args:
        rays: The rays to copy.

    Returns:
        A ray bundle of the same type holding copies of the state of ``rays``.
    """
    state = copy.copy(rays)
    for name, value in vars(rays).items():
        if hasattr(value, "shape"):
            setattr(state, name, be.copy(value))
    return state


@dataclass(frozen=True)
class TraceSnapshot:
    """Ray state before each surface of one sequential trace.

    Attributes:
        states: ``states[k]`` is the ray bundle as it arrives at surface ``k``,
            in global coordinates, before surface ``k`` acts on it.
        surfaces: The surfaces traced, in order. A resume checks that the
            optic still holds these surface objects, so a snapshot never
            resumes on a system whose surface list was edited.
    """

    states: tuple[RealRays, ...]
    surfaces: tuple[Surface, ...]

    def __len__(self) -> int:
        """Number of surfaces recorded."""
        return len(self.states)

    def state_before(self, k: int) -> RealRays:
        """Return a fresh copy of the ray state before surface ``k``.

        Args:
            k: The surface index.

        Returns:
            A ray bundle that can be traced without changing the snapshot.
        """
        return capture_rays(self.states[k])

    def matches(self, surfaces: Iterable[Surface]) -> bool:
        """Whether ``surfaces`` holds exactly the surface objects recorded.

        Args:
            surfaces: The current surface sequence of the optic.

        Returns:
            True when the sequence has the same length and the same surface
            objects in the same order.
        """
        current = list(surfaces)
        return len(current) == len(self.surfaces) and all(
            a is b for a, b in zip(current, self.surfaces, strict=True)
        )
