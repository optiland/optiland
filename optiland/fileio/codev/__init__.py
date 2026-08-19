"""CODE V Sequential file I/O for Optiland.

Provides reader and writer for CODE V Sequential (.seq) files.

Reader pipeline: .seq file -> CodeVDataParser -> CodeVDataModel
                 -> CodeVToOpticConverter -> Optic

Writer pipeline: Optic -> OpticToCodeVConverter -> CodeVDataModel
                 -> CodeVFileEncoder -> .seq file
"""

from __future__ import annotations

from optiland.fileio.codev.model import CodeVDataModel
from optiland.fileio.codev.reader.converter import CodeVToOpticConverter
from optiland.fileio.codev.surfaces import (
    BaseSurfaceHandler,
    get_handler,
    get_handler_for_optiland_type,
)
from optiland.fileio.codev.writer.exporter import CodeVWriter, save_codev_file

__all__ = [
    "CodeVDataModel",
    "CodeVToOpticConverter",
    "CodeVWriter",
    "save_codev_file",
    "BaseSurfaceHandler",
    "get_handler",
    "get_handler_for_optiland_type",
]
