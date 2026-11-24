"""Tolerance Presets Module

This module fills a tolerancing problem from a table of default tolerances, in
the way a shop's standard tolerances are applied to every surface, gap,
element and glass of a design.

Optiland ships no built-in grade tables: published "commercial / precision /
high precision" values differ between vendors and none is a standard (ISO
10110 defines the notation, not the values). Pass the table of your shop or
vendor.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

import optiland.backend as be

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from optiland.tolerancing.core import Tolerancing
    from optiland.tolerancing.tolerance import Tolerance

#: Where each built-in tolerance type is applied by ``add_default_tolerances``.
SCOPES: dict[str, str] = {
    "radius": "surface",
    "fringes": "surface",
    "conic": "surface",
    "irregularity": "surface",
    "surface_decenter": "surface",
    "surface_tilt": "surface",
    "thickness": "gap",
    "element_decenter": "element",
    "element_tilt": "element",
    "element_roll": "element",
    "index": "glass",
    "abbe": "glass",
}

#: Types that are added once per axis, x and y.
_PER_AXIS = {"surface_decenter", "surface_tilt", "element_decenter", "element_tilt"}

_LIMIT_KEYS = {"limit", "minus", "plus", "distribution"}


def _scalar(value) -> float:
    return float(np.ravel(be.to_numpy(value))[0])


def _is_air(material: Any, wavelength: float) -> bool:
    """Return True for a medium with index 1 (no refraction)."""
    return abs(_scalar(material.n(wavelength)) - 1.0) < 1e-9


def _optical_surfaces(tolerancing: Tolerancing, wavelength: float) -> list[int]:
    """Surfaces that refract or reflect, without the object and image."""
    surfaces = tolerancing.optic.surfaces
    numbers = []
    for k in range(1, surfaces.num_surfaces - 1):
        surface = surfaces[k]
        n_pre = _scalar(surface.material_pre.n(wavelength))
        n_post = _scalar(surface.material_post.n(wavelength))
        reflective = getattr(surface.interaction_model, "is_reflective", False)
        if reflective or abs(n_pre - n_post) > 1e-12:
            numbers.append(k)
    return numbers


def _elements(tolerancing: Tolerancing, wavelength: float) -> list[tuple[int, int]]:
    """Return ``(first, last)`` of each run of surfaces with glass between."""
    surfaces = tolerancing.optic.surfaces
    elements = []
    first = None
    for k in range(1, surfaces.num_surfaces - 1):
        in_glass = not _is_air(surfaces[k].material_post, wavelength)
        if in_glass and first is None:
            first = k
        elif not in_glass and first is not None:
            elements.append((first, k))
            first = None
    return elements


def _limits(entry: Any, name: str) -> dict[str, Any]:
    """Return ``add_tolerance`` keyword arguments for one table entry."""
    if isinstance(entry, int | float):
        return {"limit": float(entry)}
    if isinstance(entry, dict):
        unknown = set(entry) - _LIMIT_KEYS
        if unknown:
            raise ValueError(
                f"Unknown keys {sorted(unknown)} for {name!r}; use "
                f"{sorted(_LIMIT_KEYS)}."
            )
        return dict(entry)
    raise TypeError(
        f"Table entry for {name!r} must be a number or a dict, got "
        f"{type(entry).__name__}."
    )


def _targets(
    tolerancing: Tolerancing, scope: str, name: str, wavelength: float
) -> list[dict[str, Any]]:
    """Return the keyword arguments that place a type on every target."""
    optic = tolerancing.optic
    surfaces = optic.surfaces
    optical = _optical_surfaces(tolerancing, wavelength)
    if scope == "surface":
        targets = optical
        if name in ("radius", "conic"):
            # radius and conic tolerances need a curved surface
            targets = [
                k for k in optical if np.isfinite(_scalar(surfaces[k].geometry.radius))
            ]
        return [{"surface_number": k} for k in targets]
    if scope == "gap":
        # the gap in front of the image is usually the focus compensator
        last = surfaces.num_surfaces - 2
        return [
            {"surface_number": k}
            for k in range(1, last)
            if k in optical or k + 1 in optical
        ]
    if scope == "glass":
        return [
            {"surface_number": k}
            for k in range(1, surfaces.num_surfaces - 1)
            if not _is_air(surfaces[k].material_post, wavelength)
        ]
    if scope == "element":
        return [
            {"first": first, "last": last}
            for first, last in _elements(tolerancing, wavelength)
        ]
    raise ValueError(f"Unknown scope {scope!r}.")


def add_default_tolerances(
    tolerancing: Tolerancing,
    table: Mapping[str, Any],
    surfaces: Iterable[int] | None = None,
) -> list[Tolerance]:
    """Add the tolerances of a table to every matching part of the design.

    Each entry of ``table`` maps a tolerance type to its size: a number (a
    symmetric ``limit``) or a dict of ``add_tolerance`` arguments (``limit``,
    or ``minus`` and ``plus``, and optionally ``distribution``). Types are
    placed by scope:

    - surface types (``radius``, ``fringes``, ``conic``, ``irregularity``,
      ``surface_decenter``, ``surface_tilt``): every refracting or
      reflecting surface; ``radius`` and ``conic`` only on curved ones;
    - ``thickness``: every gap next to such a surface, except the gap in
      front of the image, which is usually the focus compensator;
    - element types (``element_decenter``, ``element_tilt``,
      ``element_roll``): every run of surfaces with glass between them;
    - ``index`` and ``abbe``: every surface followed by glass.

    Decenters and tilts are added once for x and once for y.

    Args:
        tolerancing: The tolerancing problem to add to.
        table: Tolerance type to size, e.g.
            ``{"fringes": 2, "thickness": 0.05, "surface_tilt": 0.02}``.
        surfaces: Restrict to these surface numbers. An element is kept when
            its first surface is listed. Defaults to all.

    Returns:
        The added tolerances, in the order they were added.

    Raises:
        ValueError: If a type has no known scope or an entry is malformed.
        TypeError: If an entry is neither a number nor a dict.
    """
    wavelength = _scalar(tolerancing.optic.wavelengths.primary_wavelength.value)
    keep = None if surfaces is None else set(surfaces)
    added = []
    for name, entry in table.items():
        if name not in SCOPES:
            raise ValueError(
                f"No default placement for tolerance type {name!r}; known "
                f"types: {', '.join(sorted(SCOPES))}. Add it with "
                "add_tolerance instead."
            )
        limits = _limits(entry, name)
        axes = ("x", "y") if name in _PER_AXIS else (None,)
        for target in _targets(tolerancing, SCOPES[name], name, wavelength):
            anchor = target.get("surface_number", target.get("first"))
            if keep is not None and anchor not in keep:
                continue
            for axis in axes:
                kwargs = dict(target) if axis is None else {**target, "axis": axis}
                added.append(tolerancing.add_tolerance(name, **limits, **kwargs))
    return added
