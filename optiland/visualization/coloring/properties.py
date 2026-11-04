"""Component Property Module

This module contains value functions that read a glass property of each lens
element, for use with :class:`PropertyColorizer`.

Kramer Harrison, 2026
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.materials.base import BaseMaterial

if TYPE_CHECKING:
    from collections.abc import Callable

    from optiland.optic import Optic
    from optiland.visualization.coloring.components import ComponentRef


def _element_material(optic: Optic, ref: ComponentRef) -> BaseMaterial | None:
    """Return the glass of a lens element, or None for any other component."""
    if ref.kind != "lens_element":
        return None
    return optic.surfaces[ref.surface_indices[0]].material_post


def _finite_float(value) -> float | None:
    """Convert a backend scalar to a detached Python float, or None if not finite."""
    result = float(np.asarray(be.to_numpy(value), dtype=float).item())
    return result if math.isfinite(result) else None


def abbe_number(optic: Optic, ref: ComponentRef) -> float | None:
    """Abbe number V_d of a lens element's glass.

    V_d is always computed from the glass's indices at the Fraunhofer d, F and
    C lines, so materials defined by a stored Abbe number in another convention
    (e.g. V_e for ``AbbeMaterialE``) are colored on the same scale as catalog
    glasses.

    Example:
        >>> from optiland.visualization.coloring import (
        ...     ColorScale, PropertyColorizer, abbe_number)
        >>> optic.draw(component_colorizer=PropertyColorizer(
        ...     abbe_number, ColorScale(cmap="coolwarm_r", label="Abbe number")))

    Args:
        optic (Optic): The system being drawn.
        ref (ComponentRef): The component to evaluate.

    Returns:
        float | None: V_d, or None for components that are not lens elements
        and for materials without dispersion (e.g. ``IdealMaterial``), which
        keep the default color.
    """
    material = _element_material(optic, ref)
    if material is None:
        return None
    # BaseMaterial.abbe is called explicitly because Abbe-defined materials
    # shadow the method with their stored Abbe number. A non-dispersive
    # material divides by zero here and is reported as None.
    with np.errstate(divide="ignore", invalid="ignore"):
        return _finite_float(BaseMaterial.abbe(material))


def refractive_index(
    wavelength: float | None = None,
) -> Callable[[Optic, ComponentRef], float | None]:
    """Build a value function giving each lens element's refractive index.

    Example:
        >>> from optiland.visualization.coloring import (
        ...     ColorScale, PropertyColorizer, refractive_index)
        >>> optic.draw(component_colorizer=PropertyColorizer(
        ...     refractive_index(0.5876), ColorScale(label="n_d")))

    Args:
        wavelength (float | None): Wavelength in microns. None means the
            optic's primary wavelength at the time of drawing.

    Returns:
        Callable[[Optic, ComponentRef], float | None]: Value function for
        :class:`PropertyColorizer` that returns the index, or None for
        components that are not lens elements.
    """

    def value_fn(optic: Optic, ref: ComponentRef) -> float | None:
        material = _element_material(optic, ref)
        if material is None:
            return None
        wl = optic.primary_wavelength if wavelength is None else wavelength
        return _finite_float(material.n(wl))

    return value_fn
