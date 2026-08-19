"""Zemax File Exporter

Entry point for exporting an Optiland Optic to a Zemax OpticStudio .zmx file.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.fileio.base import BaseOpticWriter
from optiland.fileio.zemax.writer.encoder import ZemaxFileEncoder
from optiland.fileio.zemax.writer.formatter import OpticToZemaxConverter

if TYPE_CHECKING:
    from optiland.optic import Optic

# Re-export so zemax/__init__.py can import both from this module
__all__ = ["save_zemax_file", "ZemaxWriter", "OpticToZemaxConverter"]


def save_zemax_file(optic: Optic, filepath: str) -> None:
    """Export an Optic to a Zemax OpticStudio .zmx file.

    The file is written in UTF-16 LE encoding to match OpticStudio's default.
    Warnings are issued via Python's ``warnings`` module for:

    - Glasses with no Zemax catalog entry (written as MODEL glass).
    - Pickups or solves that cannot be represented (resolved values exported).

    Args:
        optic: The optic to export.
        filepath: Destination path (should end in ``.zmx``).

    Raises:
        NotImplementedError: If the optic contains a surface type not yet
            supported by the writer.
        ValueError: If a coordinate system cannot be decomposed into
            COORDBRK surfaces.
    """
    model = OpticToZemaxConverter(optic).convert()
    lines = ZemaxFileEncoder(model).encode()
    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


class ZemaxWriter(BaseOpticWriter):
    """BaseOpticWriter implementation for Zemax .zmx files.

    This thin wrapper around :func:`save_zemax_file` allows the Zemax writer
    to be used polymorphically via the BaseOpticWriter interface.
    """

    def write(self, optic: Optic, filepath: str) -> list[str]:
        """Write *optic* to a .zmx file at *filepath*.

        Args:
            optic: The optic to export.
            filepath: Destination path.

        Returns:
            An empty list (warnings are issued via the ``warnings`` module).
        """
        save_zemax_file(optic, filepath)
        return []
