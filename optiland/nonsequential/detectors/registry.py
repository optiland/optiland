"""DetectorRegistry for Non-Sequential Raytracing.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.nonsequential.detectors.base import BaseDetector


class DetectorRegistry:
    """Named registry of NSQ detectors.

    Attributes:
        _registry: Ordered dict mapping name -> BaseDetector.
    """

    def __init__(self) -> None:
        """Initialize an empty DetectorRegistry."""
        self._registry: dict[str, BaseDetector] = {}

    def add(self, name: str, detector: BaseDetector) -> None:
        """Add a detector.

        Args:
            name: Unique identifier.
            detector: Detector to register.

        Raises:
            KeyError: If a detector with ``name`` already exists.
        """
        if name in self._registry:
            raise KeyError(f"Detector '{name}' is already registered.")
        self._registry[name] = detector

    def remove(self, name: str) -> None:
        """Remove a detector by name.

        Args:
            name: Name of the detector to remove.

        Raises:
            KeyError: If no detector with ``name`` exists.
        """
        if name not in self._registry:
            raise KeyError(f"Detector '{name}' not found.")
        del self._registry[name]

    def get(self, name: str) -> BaseDetector:
        """Retrieve a detector by name.

        Args:
            name: Name of the detector.

        Returns:
            The registered detector.

        Raises:
            KeyError: If no detector with ``name`` exists.
        """
        if name not in self._registry:
            raise KeyError(f"Detector '{name}' not found.")
        return self._registry[name]

    def __contains__(self, name: str) -> bool:
        """Check whether a detector name is registered."""
        return name in self._registry

    @property
    def detectors(self) -> list[BaseDetector]:
        """Ordered list of all registered detectors."""
        return list(self._registry.values())
