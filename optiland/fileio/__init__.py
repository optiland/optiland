"""File input/output operations for Optiland.

This package handles loading and saving Optiland's native JSON format and
importing/exporting Zemax (.zmx) and CODE V Sequential (.seq) files.
"""

from __future__ import annotations

import warnings

from optiland.fileio.codev.reader.converter import CodeVToOpticConverter as _CodeVTC
from optiland.fileio.codev.writer.exporter import save_codev_file
from optiland.fileio.optiland_handler import (
    load_obj_from_json,
    load_optiland_file,
    save_obj_to_json,
    save_optiland_file,
)
from optiland.fileio.oslo.reader.converter import OsloToOpticConverter as _OsloTC
from optiland.fileio.oslo.writer.exporter import save_oslo_file
from optiland.fileio.zemax.reader.converter import ZemaxToOpticConverter as _NewZTC
from optiland.fileio.zemax.writer.exporter import save_zemax_file


def load_zemax_file(source: str):
    """Load a Zemax file and return an Optic object.

    Args:
        source: The path to a local .zmx file or a URL pointing to a .zmx file.

    Returns:
        An Optic object created from the Zemax file data.
    """
    return _NewZTC({}).read(source)


def load_codev_file(source: str):
    """Load a CODE V Sequential file and return an Optic object.

    Args:
        source: The path to a local .seq file.

    Returns:
        An Optic object created from the CODE V file data.
    """
    return _CodeVTC({}).read(source)


def load_oslo_file(source: str):
    """Load an OSLO .len file and return an Optic object.

    Args:
        source: The path to a local .len file.

    Returns:
        An Optic object created from the OSLO file data.
    """
    return _OsloTC().read(source)


# ---------------------------------------------------------------------------
# Deprecation shim — ZemaxToOpticConverter is now at optiland.fileio.zemax
# ---------------------------------------------------------------------------


def _make_deprecated_ztoc(*args, **kwargs):  # noqa: ANN002, ANN003
    warnings.warn(
        "Import ZemaxToOpticConverter from optiland.fileio.zemax instead. "
        "This shim will be removed in v0.8.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _NewZTC(*args, **kwargs)


# Expose as class-like callable; keep isinstance checks working by pointing
# to the real class.
ZemaxToOpticConverter = _NewZTC


__all__ = [
    # Zemax
    "save_zemax_file",
    "load_zemax_file",
    # CODE V
    "load_codev_file",
    "save_codev_file",
    # OSLO
    "load_oslo_file",
    "save_oslo_file",
    # Optiland JSON handler
    "load_obj_from_json",
    "save_obj_to_json",
    "load_optiland_file",
    "save_optiland_file",
    # Deprecated shim (kept for backward compat)
    "ZemaxToOpticConverter",
]
