"""Output helpers shared by the prescription report classes.

Kramer Harrison, 2026
"""

from __future__ import annotations

import pathlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import os

    from optiland.prescription.document import Document
    from optiland.prescription.renderers.base import BaseRenderer


def timestamp() -> str:
    """Return the current UTC time in the report header format."""
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")


def print_document(document: Document) -> None:
    """Print a document to the console using ConsoleRenderer.

    Args:
        document: The document to print.
    """
    from optiland.prescription.renderers.console import ConsoleRenderer

    ConsoleRenderer().print(document)


def save_document(
    document: Document,
    path: str | os.PathLike,
    renderer: BaseRenderer | None = None,
) -> None:
    """Write a document to a file.

    Args:
        document: The document to write.
        path: Output file path. Extension determines the renderer when
            renderer is None: ``.pdf`` → PDFRenderer, anything else →
            PlainTextRenderer.
        renderer: Explicit renderer instance. Overrides extension inference.
    """
    path = pathlib.Path(path)
    if renderer is None:
        if path.suffix.lower() == ".pdf":
            from optiland.prescription.renderers.pdf import PDFRenderer

            renderer = PDFRenderer()
        else:
            from optiland.prescription.renderers.plain_text import (
                PlainTextRenderer,
            )

            renderer = PlainTextRenderer()
    renderer.write(document, path)
