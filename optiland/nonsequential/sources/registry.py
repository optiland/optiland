"""SourceRegistry for Non-Sequential Raytracing.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.nonsequential.sources.base import BaseNSQSource


class SourceRegistry:
    """Named registry of NSQ sources.

    Attributes:
        _registry: Ordered dict mapping name -> BaseNSQSource.
    """

    def __init__(self) -> None:
        """Initialize an empty SourceRegistry."""
        self._registry: dict[str, BaseNSQSource] = {}

    def add(self, name: str, source: BaseNSQSource) -> None:
        """Add a source.

        Args:
            name: Unique identifier.
            source: Source to register.

        Raises:
            KeyError: If a source with ``name`` already exists.
        """
        if name in self._registry:
            raise KeyError(f"Source '{name}' is already registered.")
        self._registry[name] = source

    def remove(self, name: str) -> None:
        """Remove a source by name.

        Args:
            name: Name of the source to remove.

        Raises:
            KeyError: If no source with ``name`` exists.
        """
        if name not in self._registry:
            raise KeyError(f"Source '{name}' not found.")
        del self._registry[name]

    def get(self, name: str) -> BaseNSQSource:
        """Retrieve a source by name.

        Args:
            name: Name of the source.

        Returns:
            The registered source.

        Raises:
            KeyError: If no source with ``name`` exists.
        """
        if name not in self._registry:
            raise KeyError(f"Source '{name}' not found.")
        return self._registry[name]

    def __contains__(self, name: str) -> bool:
        """Check whether a source name is registered."""
        return name in self._registry

    @property
    def sources(self) -> list[BaseNSQSource]:
        """Ordered list of all registered sources."""
        return list(self._registry.values())
