"""Face Color Module

This module contains the mixin that lets drawn components carry colors resolved
by the visualization coloring strategies, without knowing about any strategy.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from optiland.visualization.coloring.scale import RGBA


class FaceColored:
    """Mixin for components whose face color can be overridden per element.

    An element is the unit a colorizer addresses: a lens has one per glass body,
    a mirror or standalone surface has exactly one. Colors are assigned by
    ``OpticalSystem``; a component with none assigned draws with its defaults.

    Attributes:
        component_kind (str): ``ComponentRef`` kind of the elements.
        face_colors (Sequence[RGBA | None]): Resolved color per element.
    """

    component_kind: str = "surface"
    face_colors: Sequence[RGBA | None] = ()

    def element_surfaces(self) -> list[tuple[Any, ...]]:
        """Surfaces bounding each element, in drawing order."""
        raise NotImplementedError

    def face_color(self, element: int = 0) -> RGBA | None:
        """Color override of an element.

        Args:
            element (int): Element position.

        Returns:
            RGBA | None: The color, or None when the default applies (nothing
            assigned, or an unusable color).
        """
        if element >= len(self.face_colors):
            return None
        color = self.face_colors[element]
        if color is None or np.isnan(color).any():
            return None
        return color
