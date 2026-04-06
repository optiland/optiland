"""Base renderer ABCs for NSQ visualization.

Kramer Harrison, 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from optiland.nonsequential.components.compound import CompoundComponent


class ComponentRenderer2D(ABC):
    """Abstract base for 2D compound-component renderers.

    Each concrete subclass knows how to draw one :class:`CompoundComponent`
    type onto a Matplotlib :class:`~matplotlib.axes.Axes`.
    """

    @abstractmethod
    def render(
        self,
        component: CompoundComponent,
        ax: Axes,
        theme=None,
        projection: str = "YZ",
    ) -> None:
        """Draw the component cross-section onto *ax*.

        Args:
            component: The compound component to render.
            ax: Matplotlib axes to draw on.
            theme: Optional theme object (colours, line-width, etc.).
            projection: Projection plane -- ``'YZ'``, ``'XZ'``, or ``'XY'``.
        """


class ComponentRenderer3D(ABC):
    """Abstract base for 3D compound-component renderers (VTK)."""

    @abstractmethod
    def render(
        self,
        component: CompoundComponent,
        renderer,
        theme=None,
    ) -> None:
        """Add VTK actors for *component* to *renderer*.

        Args:
            component: The compound component to render.
            renderer: VTK renderer to add actors to.
            theme: Optional theme object.
        """
