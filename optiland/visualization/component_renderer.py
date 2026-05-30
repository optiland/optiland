"""ComponentRenderer ABC

Defines the abstract base class for rendering a single optical component
in 2D (matplotlib) or 3D (VTK). Concrete renderers are registered with
OpticalSystem via OpticalSystem.register_component_renderer().

Kramer Harrison, 2026
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


class ComponentRenderer(abc.ABC):
    """Contract for rendering a single optical component in 2D or 3D.

    Concrete implementations are registered with OpticalSystem via
    ``OpticalSystem.register_component_renderer()``.

    Example:
        >>> class MyRenderer(ComponentRenderer):
        ...     def render_2d(self, ax, component_data):
        ...         z = component_data['z']
        ...         ax.axvline(z, color='red')
        ...
        >>> OpticalSystem.register_component_renderer('my_type', MyRenderer())
    """

    @abc.abstractmethod
    def render_2d(self, ax: plt.Axes, component_data: dict[str, Any]) -> None:
        """Draw this component onto a 2D matplotlib Axes.

        Args:
            ax: The target matplotlib Axes.
            component_data: Component parameters (e.g. thickness, radius, z).
        """

    def render_3d(self, renderer: Any, component_data: dict[str, Any]) -> None:  # noqa: B027
        """Draw this component in a 3D VTK renderer.

        Default is a no-op. Override for 3D-capable components.

        Args:
            renderer: VTK renderer instance.
            component_data: Component parameters.
        """
