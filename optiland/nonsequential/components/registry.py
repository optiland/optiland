"""ComponentRegistry for Non-Sequential Raytracing.

Maps compound component names to CompoundComponent objects and exposes a
flat surface list for the tracer.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.nonsequential.components.base import BaseComponent
    from optiland.nonsequential.components.compound import CompoundComponent


class ComponentRegistry:
    """Named registry of compound components.

    Stores :class:`~optiland.nonsequential.components.compound.CompoundComponent`
    objects keyed by name and provides the tracer with a flat list of
    :class:`~optiland.nonsequential.components.base.BaseComponent` surfaces.

    Attributes:
        _registry: Ordered dict mapping name -> CompoundComponent.
    """

    def __init__(self) -> None:
        """Initialize an empty ComponentRegistry."""
        self._registry: dict[str, CompoundComponent] = {}

    def add(self, name: str, component: CompoundComponent) -> None:
        """Add a compound component under the given name.

        Args:
            name: Unique identifier for the component.
            component: The compound component to register.

        Raises:
            KeyError: If a component with ``name`` already exists.
        """
        if name in self._registry:
            raise KeyError(f"Component '{name}' is already registered.")
        self._registry[name] = component

    def remove(self, name: str) -> None:
        """Remove a component by name.

        Args:
            name: Name of the component to remove.

        Raises:
            KeyError: If no component with ``name`` exists.
        """
        if name not in self._registry:
            raise KeyError(f"Component '{name}' not found.")
        del self._registry[name]

    def get(self, name: str) -> CompoundComponent:
        """Retrieve a compound component by name.

        Args:
            name: Name of the component.

        Returns:
            The registered compound component.

        Raises:
            KeyError: If no component with ``name`` exists.
        """
        if name not in self._registry:
            raise KeyError(f"Component '{name}' not found.")
        return self._registry[name]

    def __contains__(self, name: str) -> bool:
        """Check whether a component name is registered."""
        return name in self._registry

    @property
    def surfaces(self) -> list[BaseComponent]:
        """Flat list of all sub-surfaces in registration order.

        Returns:
            Concatenated list of sub-surfaces from all compound components.
        """
        return [s for c in self._registry.values() for s in c.surfaces]

    @property
    def compounds(self) -> list[CompoundComponent]:
        """Ordered list of all registered compound components.

        Returns:
            List of compound components in registration order.
        """
        return list(self._registry.values())
