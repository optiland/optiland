"""Element Tolerance Types Module

This module contains the tolerance types that move a group of consecutive
surfaces as one rigid body: element decenter, tilt and roll. Tilts and rolls
are about a pivot point (``RigidElement``); surfaces after the element do not
move.

Kramer Harrison, 2026
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from optiland.optimization.variable.element import RigidElement
from optiland.tolerancing.base import Phase
from optiland.tolerancing.registry import ToleranceType

if TYPE_CHECKING:
    from optiland.optic import Optic


class ElementTolerance(ToleranceType):
    """Rigid-body move of the surfaces ``first`` to ``last``.

    Each deviation is applied to the current surface positions, so changes
    applied earlier in a trial (such as a thickness tolerance inside the
    element) are kept. ``reset`` restores the positions from before the
    move exactly.

    Args:
        optic: The optic the tolerance applies to.
        first: The first surface of the element.
        last: The last surface of the element (inclusive).
        axis: The element axis of the move: ``"x"``, ``"y"`` or ``"z"``.
        pivot: The pivot point of tilts and rolls: ``"center"`` (default,
            midway between the first and last vertex), ``"first"``,
            ``"last"``, or the distance from the first vertex along the
            element axis, in lens units.

    Attributes:
        kind: ``"decenter"``, ``"tilt"`` or ``"roll"``.
        scale: Factor from deviation units to the move (degrees to radians
            for tilts and rolls).
    """

    phase = Phase.RIGID_BODY
    kind: str = ""
    scale: float = 1.0

    def __init__(
        self,
        optic: Optic,
        first: int,
        last: int,
        axis: str = "z",
        pivot: str | float = "center",
    ) -> None:
        super().__init__(optic)
        self.element = RigidElement(optic, first, last, pivot)
        self.axis = axis
        self._before: list[tuple] | None = None

    @property
    def label(self) -> str:
        """Return a short, human-readable name, e.g. ``"Element Tilt X S3-4"``."""
        axis = "" if self.kind == "roll" else f" {self.axis.upper()}"
        name = f"Element {self.kind.capitalize()}{axis}"
        return f"{name} S{self.element.first}-{self.element.last}"

    def apply(self, delta: float) -> None:
        """Move the element by ``delta`` from its position before the move."""
        if self._before is None:
            self._before = self.element.state()
        else:
            self.element.restore(self._before)
        self.element.move(self.kind, self.axis, self.scale * delta)

    def reset(self) -> None:
        """Restore the positions from before the move."""
        if self._before is not None:
            self.element.restore(self._before)
            self._before = None


def _check_axis(axis: str) -> str:
    if axis not in ("x", "y"):
        raise ValueError(f'axis must be "x" or "y", got {axis!r}.')
    return axis


class ElementDecenterTolerance(ElementTolerance):
    """Decenter of an element along its local ``axis`` ("x" or "y"), in lens
    units."""

    kind = "decenter"
    units = "lens units"

    def __init__(self, optic: Optic, first: int, last: int, axis: str) -> None:
        super().__init__(optic, first, last, axis=_check_axis(axis))


class ElementTiltTolerance(ElementTolerance):
    """Tilt of an element about its local ``axis`` ("x" or "y") through the
    pivot, in degrees."""

    kind = "tilt"
    units = "degrees"
    scale = math.pi / 180.0

    def __init__(
        self,
        optic: Optic,
        first: int,
        last: int,
        axis: str,
        pivot: str | float = "center",
    ) -> None:
        super().__init__(optic, first, last, axis=_check_axis(axis), pivot=pivot)


class ElementRollTolerance(ElementTolerance):
    """Roll of an element about its local z axis through the pivot, in
    degrees."""

    kind = "roll"
    units = "degrees"
    scale = math.pi / 180.0

    def __init__(
        self, optic: Optic, first: int, last: int, pivot: str | float = "center"
    ) -> None:
        super().__init__(optic, first, last, axis="z", pivot=pivot)
