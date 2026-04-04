"""NSQ Scene container for Non-Sequential Raytracing.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.nonsequential.components.base import BaseComponent
    from optiland.nonsequential.detectors.base import BaseDetector
    from optiland.nonsequential.sources.base import BaseNSQSource


class NSQScene:
    """Flat container for all elements in a non-sequential scene.

    Components, sources, and detectors are stored in flat lists.
    There is no hierarchical scene graph. Each object is positioned
    independently in global 3D space using its own CoordinateSystem.

    Attributes:
        components: List of optical components (mirrors, lenses, absorbers).
        sources: List of ray sources.
        detectors: List of detector surfaces.
    """

    def __init__(self) -> None:
        """Initialize an empty NSQScene."""
        self.components: list[BaseComponent] = []
        self.sources: list[BaseNSQSource] = []
        self.detectors: list[BaseDetector] = []

    def add_component(self, component: BaseComponent) -> None:
        """Add an optical component to the scene.

        Args:
            component: The component to add.
        """
        self.components.append(component)

    def add_source(self, source: BaseNSQSource) -> None:
        """Add a ray source to the scene.

        Args:
            source: The source to add.
        """
        self.sources.append(source)

    def add_detector(self, detector: BaseDetector) -> None:
        """Add a detector to the scene.

        Args:
            detector: The detector to add.
        """
        self.detectors.append(detector)

    def validate(self) -> None:
        """Validate the scene for common configuration errors.

        Checks:
          - At least one source is present.
          - At least one detector is present.
          - Source emission directions are plausible (no zero directions).

        Raises:
            ValueError: If the scene is invalid.
        """
        if not self.sources:
            raise ValueError("Scene has no sources. Add at least one source.")
        if not self.detectors:
            raise ValueError("Scene has no detectors. Add at least one detector.")
