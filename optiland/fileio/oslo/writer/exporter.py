"""OSLO File Exporter

Provides the OsloWriter and save_oslo_file entry point.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.fileio.base import BaseOpticWriter
from optiland.fileio.oslo.writer.encoder import OpticToOsloEncoder
from optiland.fileio.oslo.writer.formatter import OsloDataFormatter

if TYPE_CHECKING:
    from optiland.optic import Optic


class OsloWriter(BaseOpticWriter):
    """Exports an Optic object to an OSLO .len file.

    Args:
        None
    """

    def write(self, optic: Optic, filepath: str) -> list[str]:
        """Write an optic to an OSLO .len file.

        Args:
            optic: The optic to export.
            filepath: Destination file path.

        Returns:
            A list of warning strings (empty if no warnings).
        """
        encoder = OpticToOsloEncoder(optic)
        model = encoder.encode()

        formatter = OsloDataFormatter(model)
        content = formatter.format()

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return []


def save_oslo_file(optic: Optic, filepath: str) -> list[str]:
    """Save an Optic to an OSLO .len file.

    Args:
        optic: The optic to save.
        filepath: The destination path.

    Returns:
        A list of warnings.
    """
    return OsloWriter().write(optic, filepath)
