"""Parsing helpers for catalog metadata and refractiveindex.info YAML payloads.

Split out of ``optiland.materials.registry`` — these are pure string/dict
parsing functions with no dependency on `MaterialRegistry` state.

Kramer Harrison, 2025
"""

from __future__ import annotations

import contextlib


def _levenshtein(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance between two strings."""
    rows, cols = len(s1) + 1, len(s2) + 1
    dist = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        dist[i][0] = i
    for j in range(1, cols):
        dist[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dist[i][j] = min(
                dist[i - 1][j] + 1,
                dist[i][j - 1] + 1,
                dist[i - 1][j - 1] + cost,
            )
    return dist[-1][-1]


def _catalog_dir_from_filename(filename: str, group: str = "") -> str:
    """Extract the manufacturer catalog name from a filename path.

    For glass entries whose path starts with ``glass/``, the manufacturer is
    always the second path segment (``glass/{manufacturer}/...``), regardless
    of nesting depth.  For all other entries the species name is the
    second-to-last segment.
    """
    parts = filename.replace("\\", "/").split("/")
    if group.lower() == "glass" and parts[0].lower() == "glass" and len(parts) >= 3:
        return parts[1]
    return parts[-2] if len(parts) >= 3 else ""


def _wavelength_range_from_data_points(raw: str) -> tuple[float, float] | None:
    """Min/max wavelength (µm) from a DATA entry's whitespace-table payload."""
    wls: list[float] = []
    for line in raw.strip().splitlines():
        parts = line.split()
        if parts:
            with contextlib.suppress(ValueError):
                wls.append(float(parts[0]))
    if not wls:
        return None
    return min(wls), max(wls)


def _wavelength_range_from_formula_field(wl_range: str) -> tuple[float, float] | None:
    """Min/max wavelength (µm) from a DATA entry's ``wavelength_range`` field."""
    parts = str(wl_range).split()
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def _extract_wavelength_range(data: dict) -> tuple[float, float]:
    """Extract min/max wavelength (µm) from a refractiveindex.info YAML payload."""
    for item in data.get("DATA", []):
        raw = item.get("data", "")
        if not raw:
            continue

        found = _wavelength_range_from_data_points(raw)
        if found is not None:
            return found

        # Formula entries may have a wavelength_range field
        wl_range = item.get("wavelength_range", "")
        if wl_range:
            found = _wavelength_range_from_formula_field(wl_range)
            if found is not None:
                return found

    return 0.0, 100.0
