"""Abstract base classes for optic file readers and writers.

This module defines the extension contract for all file format readers and
writers. Future file formats (CODE V, OSLO, etc.) should subclass these ABCs.

Kramer Harrison, 2024
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.optic import Optic


class BaseOpticReader(ABC):
    """Abstract base for all optic file readers."""

    @abstractmethod
    def read(self, source: str) -> Optic:
        """Read an optic from a file path or URL.

        Args:
            source: Local file path or URL to the source file.

        Returns:
            A fully-configured Optic instance.
        """


class BaseOpticWriter(ABC):
    """Abstract base for all optic file writers."""

    @abstractmethod
    def write(self, optic: Optic, filepath: str) -> list[str]:
        """Write an optic to a file.

        Args:
            optic: The optic to export.
            filepath: Destination file path.

        Returns:
            A list of warning strings (empty if no warnings).
        """
