"""Component Coloring Module

This module contains the strategies that choose the face color of drawn
components (lens elements, mirrors and standalone surfaces).

Kramer Harrison, 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from matplotlib.colors import to_rgba

import optiland.backend as be
from optiland.visualization.coloring.scale import RGBA, ColorScale

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from matplotlib.typing import ColorType

    from optiland.optic import Optic

ComponentKind = Literal["lens_element", "mirror", "surface"]


@dataclass(frozen=True)
class ComponentRef:
    """Identifies one drawn component by kind and position in ``optic.surfaces``.

    Attributes:
        kind (str): ``"lens_element"``, ``"mirror"`` or ``"surface"``.
            A lens element is one glass body bounded by surfaces ``(k, k + 1)``,
            so a cemented doublet holds two elements. A mirror or standalone
            surface (object, image, paraxial, phase) is ``(k,)``.
        surface_indices (tuple[int, ...]): Positions in ``optic.surfaces``
            (0 is the object surface).
    """

    kind: ComponentKind
    surface_indices: tuple[int, ...]


class ComponentColorizer(ABC):
    """Strategy that picks the face color of each drawn component."""

    def prepare(self, optic: Optic, refs: Sequence[ComponentRef]) -> None:  # noqa: B027
        """Hook called once per draw, before any :meth:`color_for` call.

        Args:
            optic (Optic): The system being drawn.
            refs (Sequence[ComponentRef]): Every component that will be drawn.
        """

    @abstractmethod
    def color_for(self, ref: ComponentRef) -> RGBA | None:
        """Return the face color of a component.

        Args:
            ref (ComponentRef): The component to color.

        Returns:
            RGBA | None: Color, or None to keep the default color.
        """

    @property
    def scale(self) -> ColorScale | None:
        """ColorScale | None: Resolved scale to show as a colorbar, if any."""
        return None


class ThemeColorizer(ComponentColorizer):
    """Keeps the theme's default color for every component."""

    def color_for(self, ref: ComponentRef) -> RGBA | None:
        return None


class UniformColorizer(ComponentColorizer):
    """Gives every component the same color.

    Args:
        color (ColorType): Anything ``matplotlib.colors.to_rgba`` accepts.
    """

    def __init__(self, color: ColorType) -> None:
        self._rgba = to_rgba(color)

    def color_for(self, ref: ComponentRef) -> RGBA | None:
        return self._rgba


class PropertyColorizer(ComponentColorizer):
    """Colors components by a scalar property mapped through a ColorScale.

    Args:
        value_fn (Callable[[Optic, ComponentRef], float | None]): Property of a
            component. Returning None or NaN keeps the default color. Values
            may be Python floats, NumPy scalars or Torch tensors.
        scale (ColorScale): Value-to-color mapping. Unset limits are taken from
            the evaluated values.
        kinds (tuple[str, ...]): Component kinds to color; other kinds keep the
            default color.
    """

    def __init__(
        self,
        value_fn: Callable[[Optic, ComponentRef], float | None],
        scale: ColorScale | None = None,
        kinds: tuple[ComponentKind, ...] = ("lens_element",),
    ) -> None:
        self._value_fn = value_fn
        self._requested_scale = scale if scale is not None else ColorScale()
        self._kinds = kinds
        self._colors: dict[ComponentRef, RGBA] = {}
        self._scale: ColorScale | None = None

    def prepare(self, optic: Optic, refs: Sequence[ComponentRef]) -> None:
        selected = [ref for ref in refs if ref.kind in self._kinds]
        self._colors = {}
        self._scale = None
        if not selected:
            return
        values = np.array([self._evaluate(optic, ref) for ref in selected])
        self._scale = self._requested_scale.resolved(values)
        for ref, row in zip(selected, self._scale.to_rgba(values), strict=True):
            if np.isfinite(row).all():
                self._colors[ref] = tuple(float(channel) for channel in row)

    def color_for(self, ref: ComponentRef) -> RGBA | None:
        return self._colors.get(ref)

    @property
    def scale(self) -> ColorScale | None:
        return self._scale

    def _evaluate(self, optic: Optic, ref: ComponentRef) -> float:
        """Evaluate the property as a detached Python float (NaN if absent)."""
        value = self._value_fn(optic, ref)
        if value is None:
            return float("nan")
        return np.asarray(be.to_numpy(value), dtype=float).item()
