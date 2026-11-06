"""CODE V glass names for Optiland catalog materials

Maps the catalog file an Optiland ``Material`` was resolved from to the glass
token CODE V expects on a surface line (``<NAME>_<CATALOG>``, for example
``NBK7_SCHOTT``).

The name is derived from ``material_data["filename"]`` and never from
``Material.name`` or ``Material.reference``: both are the user's search
string, so ``Material("SF5", "schott")`` is named ``SF5`` although it resolves
to the glass ``N-SF5``.
"""

from __future__ import annotations

import re
from typing import Any

from optiland.materials.material import Material

# Optiland catalog directory -> CODE V catalog suffix.
CODEV_CATALOGS: dict[str, str] = {
    "schott": "SCHOTT",
    "ohara": "OHARA",
    "hoya": "HOYA",
    "cdgm": "CDGM",
    "sumita": "SUMITA",
}

# (catalog directory, upper-case file stem) pairs that must not be named.
# Source: membership of the CODE V 2025.03 ``glass/*.xml`` catalogs, plus
# stems whose Optiland data differs from the CODE V glass of the same name.
# Such materials are written without a catalog name by the caller.
NOT_IN_CODEV: frozenset[tuple[str, str]] = frozenset(
    {
        ("schott", "LITHOTEC-CAF2"),
        ("ohara", "S-APL"),
        ("cdgm", "H-K9LAGT"),
        ("sumita", "K-FIR97UV"),
        ("schott", "SF5"),
        ("schott", "SF10"),
        ("schott", "SF11"),
        ("cdgm", "H-LAK67"),
        ("cdgm", "F4GTI"),
        ("cdgm", "H-ZLAF56B"),
    }
)

_NON_ALNUM = re.compile(r"[^A-Za-z0-9]")


def codev_name_for_filename(filename: str) -> tuple[str, str] | None:
    """Return the CODE V glass name for an Optiland catalog file path.

    Only files of the form ``glass/<catalog>/<stem>.yml`` in one of the
    catalogs of :data:`CODEV_CATALOGS` are named. Sub-folders
    (``glass/schott/misc/...``), other catalogs and the stems in
    :data:`NOT_IN_CODEV` are not.

    Args:
        filename: The ``material_data["filename"]`` of a ``Material``.

    Returns:
        ``(name, catalog)`` such as ``("NBK7", "SCHOTT")``, or ``None`` if the
        material has no verified CODE V name.
    """
    parts = filename.replace("\\", "/").split("/")
    if len(parts) != 3 or parts[0] != "glass" or not parts[2].endswith(".yml"):
        return None
    catalog = CODEV_CATALOGS.get(parts[1])
    stem = parts[2][: -len(".yml")]
    if catalog is None or (parts[1], stem.upper()) in NOT_IN_CODEV:
        return None
    name = _NON_ALNUM.sub("", stem).upper()
    return (name, catalog) if name else None


def codev_glass_name(mat: Any) -> tuple[str, str] | None:
    """Return the CODE V glass name of a material, if it has one.

    Args:
        mat: A surface material.

    Returns:
        ``(name, catalog)`` for a catalog ``Material`` with a verified CODE V
        name, otherwise ``None`` (every other material class, Hikari and other
        unsupported catalogs, custom files).
    """
    if not isinstance(mat, Material):
        return None
    data = getattr(mat, "material_data", None)
    filename = data.get("filename") if isinstance(data, dict) else None
    if not isinstance(filename, str):
        return None
    return codev_name_for_filename(filename)


def wavelength_outside_range(
    mat: Any, wavelengths: list[float]
) -> tuple[float, float, float] | None:
    """Return the first wavelength outside the data range of a catalog material.

    CODE V aborts the load when a system wavelength lies outside the range of a
    catalog glass, so such a glass cannot be named.

    Args:
        mat: A surface material.
        wavelengths: System wavelengths in um, in the order to be checked.

    Returns:
        ``(wavelength, min_wavelength, max_wavelength)`` in um for the first
        wavelength outside ``[min_wavelength, max_wavelength]`` (both ends are
        inside), or ``None`` if all are inside or the material records no range.
    """
    data = getattr(mat, "material_data", None)
    if not isinstance(data, dict):
        return None
    low = data.get("min_wavelength")
    high = data.get("max_wavelength")
    low = float("-inf") if low is None else float(low)
    high = float("inf") if high is None else float(high)
    for wavelength in wavelengths:
        if not low <= wavelength <= high:
            return wavelength, low, high
    return None
