"""Element Variable Module

This module contains variables that move a group of consecutive surfaces (an
element, or a group of elements) as one rigid body: a decenter, a tilt about a
pivot point, and a roll about the element axis. Surfaces after the element do
not move.

The axes are those of the element: the local x, y and z axes of its first
surface. For an element on the optical axis these are the global axes.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.optimization.variable.base import VariableBehavior, without_graph

if TYPE_CHECKING:
    from optiland._types import BEArray
    from optiland.optic import Optic
    from optiland.optimization.scaling.base import Scaler

Matrix = list[list["BEArray"]]
Vector = list["BEArray"]

_AXES = {"x": 0, "y": 1, "z": 2}


def _euler_matrix(rx, ry, rz) -> Matrix:
    """Return ``Rz @ Ry @ Rx``, the orientation of a coordinate system."""
    cx, sx = be.cos(rx), be.sin(rx)
    cy, sy = be.cos(ry), be.sin(ry)
    cz, sz = be.cos(rz), be.sin(rz)
    return [
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ]


def _euler_angles(m: Matrix) -> tuple[BEArray, BEArray, BEArray]:
    """Return ``(rx, ry, rz)`` such that ``_euler_matrix(rx, ry, rz) == m``.

    Exact for ``|ry| < 90°``.
    """
    ry = be.arcsin(-m[2][0])
    rx = be.arctan2(m[2][1], m[2][2])
    rz = be.arctan2(m[1][0], m[0][0])
    return rx, ry, rz


def _matmul(a: Matrix, b: Matrix) -> Matrix:
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)
    ]


def _apply(m: Matrix, v: Vector) -> Vector:
    return [sum(m[i][k] * v[k] for k in range(3)) for i in range(3)]


def _axis_rotation(u: Vector, angle) -> Matrix:
    """Return the rotation by ``angle`` about the unit vector ``u``
    (Rodrigues' formula)."""
    c, s = be.cos(angle), be.sin(angle)
    t = 1 - c
    x, y, z = u
    return [
        [c + t * x * x, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, c + t * y * y, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, c + t * z * z],
    ]


class RigidElement:
    """A group of consecutive surfaces moved as one rigid body.

    Args:
        optic: The optic that holds the surfaces.
        first: The first surface of the element.
        last: The last surface of the element (inclusive).
        pivot: The point that tilts and rolls are about: ``"center"``
            (default, midway between the first and last vertex), ``"first"``
            or ``"last"`` (that vertex), or a number: the distance from the
            first vertex along the element axis, in lens units.

    Raises:
        ValueError: If the surface range or the pivot is invalid, or if a
            surface is placed relative to another coordinate system.
    """

    def __init__(
        self,
        optic: Optic,
        first: int,
        last: int,
        pivot: str | float = "center",
    ) -> None:
        num_surfaces = optic.surfaces.num_surfaces
        if not 0 < first <= last < num_surfaces - 1:
            raise ValueError(
                f"Element surfaces must satisfy 0 < first <= last < "
                f"{num_surfaces - 1} (the object and image surfaces cannot "
                f"move), got first={first}, last={last}."
            )
        if isinstance(pivot, str) and pivot not in ("center", "first", "last"):
            raise ValueError(
                f'pivot must be "center", "first", "last" or a number, got {pivot!r}.'
            )
        self.optic = optic
        self.first = first
        self.last = last
        self.pivot = pivot
        for cs in self.coordinate_systems():
            if cs.reference_cs is not None:
                raise ValueError(
                    "Element moves are not supported for surfaces placed "
                    "relative to another coordinate system."
                )

    def coordinate_systems(self) -> list:
        """Return the coordinate systems of the element's surfaces."""
        surfaces = self.optic.surfaces
        return [surfaces[k].geometry.cs for k in range(self.first, self.last + 1)]

    def state(self) -> list[tuple]:
        """Return ``(x, y, z, rx, ry, rz)`` of every surface, to restore later."""
        return [
            (cs.x, cs.y, cs.z, cs.rx, cs.ry, cs.rz) for cs in self.coordinate_systems()
        ]

    def restore(self, state: list[tuple]) -> None:
        """Restore a state returned by ``state``."""
        for cs, (x, y, z, rx, ry, rz) in zip(
            self.coordinate_systems(), state, strict=True
        ):
            cs.x, cs.y, cs.z, cs.rx, cs.ry, cs.rz = x, y, z, rx, ry, rz

    def release_graph(self) -> None:
        """Drop the autograd graph held by the surface positions."""
        for cs in self.coordinate_systems():
            cs.x, cs.y, cs.z = (
                without_graph(cs.x),
                without_graph(cs.y),
                without_graph(cs.z),
            )
            cs.rx, cs.ry = without_graph(cs.rx), without_graph(cs.ry)
            cs.rz = without_graph(cs.rz)

    def axis(self, name: str) -> Vector:
        """Return the element's local ``"x"``, ``"y"`` or ``"z"`` axis in
        global coordinates."""
        cs = self.coordinate_systems()[0]
        column = _AXES[name]
        return [row[column] for row in _euler_matrix(cs.rx, cs.ry, cs.rz)]

    def pivot_point(self) -> Vector:
        """Return the pivot point in global coordinates."""
        systems = self.coordinate_systems()
        first = [systems[0].x, systems[0].y, systems[0].z]
        if self.pivot == "first":
            return first
        last = [systems[-1].x, systems[-1].y, systems[-1].z]
        if self.pivot == "last":
            return last
        if self.pivot == "center":
            return [(a + b) / 2 for a, b in zip(first, last, strict=True)]
        axis = self.axis("z")
        return [p + self.pivot * u for p, u in zip(first, axis, strict=True)]

    def decenter(self, axis: str, distance) -> None:
        """Move the element along its local ``axis`` ("x" or "y").

        Args:
            axis: The element axis to move along.
            distance: The distance, in lens units.
        """
        u = self.axis(axis)
        for cs in self.coordinate_systems():
            cs.x = cs.x + distance * u[0]
            cs.y = cs.y + distance * u[1]
            cs.z = cs.z + distance * u[2]

    def rotate(self, axis: str, angle) -> None:
        """Rotate the element about its local ``axis`` through the pivot.

        Args:
            axis: ``"x"`` or ``"y"`` for a tilt, ``"z"`` for a roll.
            angle: The rotation angle, in radians.
        """
        rotation = _axis_rotation(self.axis(axis), angle)
        p = self.pivot_point()
        for cs in self.coordinate_systems():
            offset = [cs.x - p[0], cs.y - p[1], cs.z - p[2]]
            x, y, z = (a + b for a, b in zip(p, _apply(rotation, offset), strict=True))
            orientation = _matmul(rotation, _euler_matrix(cs.rx, cs.ry, cs.rz))
            cs.x, cs.y, cs.z = x, y, z
            cs.rx, cs.ry, cs.rz = _euler_angles(orientation)

    def move(self, kind: str, axis: str, amount) -> None:
        """Apply a decenter (``kind="decenter"``) or a rotation
        (``kind="tilt"`` or ``"roll"``) of ``amount``."""
        if kind == "decenter":
            self.decenter(axis, amount)
        else:
            self.rotate(axis, amount)


class ElementVariableBehavior(VariableBehavior):
    """Variable that moves a group of surfaces as one rigid body.

    The value is the move from the state the variable was created in: zero
    at first. Each update applies the change from the previous value to the
    current surface positions, so other variables that act on the same
    surfaces (a thickness, for example) keep their effect.

    Args:
        optic: The optic that holds the surfaces.
        first: The first surface of the element.
        last: The last surface of the element (inclusive).
        scaler: The scaler for the variable. Defaults to IdentityScaler().
        pivot: The pivot of tilts and rolls; see ``RigidElement``.
        **kwargs: Additional keyword arguments.
    """

    kind: str = ""

    def __init__(
        self,
        optic: Optic,
        first: int,
        last: int,
        scaler: Scaler = None,
        pivot: str | float = "center",
        **kwargs,
    ) -> None:
        kwargs.pop("surface_number", None)
        super().__init__(optic, first, scaler=scaler, **kwargs)
        self.element = RigidElement(optic, first, last, pivot)
        self._value = be.array(0.0)

    def get_value(self):
        """Return the current move of the element."""
        return self._value

    def update_value(self, new_value) -> None:
        """Move the element to ``new_value``.

        Args:
            new_value: The new decenter (lens units) or angle (radians).
        """
        self.element.move(self.kind, self._axis, new_value - self._value)
        self._value = new_value

    def release_graph(self) -> None:
        """Drop the autograd graph of the last step from the element."""
        self._value = without_graph(self._value)
        self.element.release_graph()

    def _range(self) -> str:
        return f"Surfaces {self.element.first}-{self.element.last}"


def _check_axis(axis: str) -> str:
    if axis not in ("x", "y"):
        raise ValueError(f'axis must be "x" or "y", got {axis!r}.')
    return axis


class ElementDecenterVariable(ElementVariableBehavior):
    """Decenter of an element along its local ``axis`` ("x" or "y"), in lens
    units.

    Args:
        optic: The optic that holds the surfaces.
        first: The first surface of the element.
        last: The last surface of the element (inclusive).
        axis: ``"x"`` or ``"y"``.
        scaler: The scaler for the variable. Defaults to IdentityScaler().
        **kwargs: Additional keyword arguments.
    """

    kind = "decenter"

    def __init__(self, optic, first, last, axis, scaler=None, **kwargs) -> None:
        self._axis = _check_axis(axis)
        super().__init__(optic, first, last, scaler=scaler, **kwargs)

    def __str__(self) -> str:
        return f"Element Decenter {self._axis.upper()}, {self._range()}"


class ElementTiltVariable(ElementVariableBehavior):
    """Tilt of an element about its local ``axis`` ("x" or "y") through the
    pivot, in radians.

    Args:
        optic: The optic that holds the surfaces.
        first: The first surface of the element.
        last: The last surface of the element (inclusive).
        axis: ``"x"`` or ``"y"``.
        scaler: The scaler for the variable. Defaults to IdentityScaler().
        pivot: The pivot point; see ``RigidElement``. Defaults to
            ``"center"``.
        **kwargs: Additional keyword arguments.
    """

    kind = "tilt"

    def __init__(self, optic, first, last, axis, scaler=None, **kwargs) -> None:
        self._axis = _check_axis(axis)
        super().__init__(optic, first, last, scaler=scaler, **kwargs)

    def __str__(self) -> str:
        return f"Element Tilt {self._axis.upper()}, {self._range()}"


class ElementRollVariable(ElementVariableBehavior):
    """Roll of an element about its local z axis through the pivot, in
    radians.

    Args:
        optic: The optic that holds the surfaces.
        first: The first surface of the element.
        last: The last surface of the element (inclusive).
        scaler: The scaler for the variable. Defaults to IdentityScaler().
        pivot: The pivot point; see ``RigidElement``. Defaults to
            ``"center"``.
        **kwargs: Additional keyword arguments.
    """

    kind = "roll"
    _axis = "z"

    def __str__(self) -> str:
        return f"Element Roll, {self._range()}"
