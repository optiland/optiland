"""BaseRenderer ABC for prescription renderers.

Kramer Harrison, 2026
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import os

    from optiland.prescription.document import Document


class BaseRenderer(abc.ABC):
    """Consumes a Document and produces output."""

    @abc.abstractmethod
    def render(self, document: Document) -> str:
        """Return the rendered document as a string.

        Args:
            document: The document model to render.

        Returns:
            Rendered string (plain text, rich markup, or file path for PDF).
        """
        ...

    @abc.abstractmethod
    def write(self, document: Document, path: str | os.PathLike) -> None:
        """Render and write to a file.

        Args:
            document: The document model to render.
            path: Output file path.
        """
        ...
