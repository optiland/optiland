"""CODE V File Exporter

Entry point for exporting an Optiland Optic to a CODE V Sequential .seq file.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.fileio.base import BaseOpticWriter
from optiland.fileio.codev.writer.encoder import CodeVFileEncoder
from optiland.fileio.codev.writer.formatter import OpticToCodeVConverter

if TYPE_CHECKING:
    from optiland.optic import Optic

__all__ = ["save_codev_file", "CodeVWriter"]


def save_codev_file(optic: Optic, filepath: str) -> None:
    """Export an Optic to a CODE V Sequential .seq file.

    The file is written in UTF-8 encoding. Warnings are issued via Python's
    ``warnings`` module for:

    - Glasses with no CODE V catalog entry (written as fictitious Nd:Vd glass).
    - Pickups or solves that cannot be represented (resolved values exported).
    - Unsupported surface types (raises NotImplementedError).

    Args:
        optic: The optic to export.
        filepath: Destination path (should end in ``.seq``).

    Raises:
        NotImplementedError: If the optic contains a surface type not yet
            supported by the CODE V writer.
    """
    model = OpticToCodeVConverter(optic).convert()
    lines = CodeVFileEncoder(model).encode()
    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


class CodeVWriter(BaseOpticWriter):
    """BaseOpticWriter implementation for CODE V .seq files.

    This thin wrapper around :func:`save_codev_file` allows the CODE V writer
    to be used polymorphically via the BaseOpticWriter interface.
    """

    def write(self, optic: Optic, filepath: str) -> list[str]:
        """Write *optic* to a .seq file at *filepath*.

        Args:
            optic: The optic to export.
            filepath: Destination path.

        Returns:
            An empty list (warnings are issued via the ``warnings`` module).
        """
        save_codev_file(optic, filepath)
        return []
