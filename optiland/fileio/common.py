"""Shared helpers for Optic-to-format writer converters.

Consolidates logic that was independently duplicated across the Zemax,
CODE V, and OSLO writer converters: air-material detection, field-type
classification, and Abbe-number computation for materials without a
native catalog entry in the target format.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.materials.ideal import IdealMaterial

if TYPE_CHECKING:
    from optiland.optic import Optic

# CIE standard wavelengths for Abbe number calculation (um)
WL_D = 0.5876  # helium d-line
WL_F = 0.4861  # hydrogen F-line
WL_C = 0.6563  # hydrogen C-line

# Map from Optiland field definition class name to field type string,
# shared by every writer's field-block conversion.
FIELD_CLASS_TO_TYPE: dict[str, str] = {
    "AngleField": "angle",
    "ObjectHeightField": "object_height",
    "ParaxialImageHeightField": "paraxial_image_height",
    "RealImageHeightField": "real_image_height",
}


def is_air(material: Any) -> bool:
    """Return True if *material* represents air (n ~= 1.0, non-absorbing)."""
    if material is None:
        return True
    if isinstance(material, str) and material.lower() in ("air", ""):
        return True
    if isinstance(material, IdealMaterial):
        n_val = float(be.atleast_1d(material.index)[0])
        return abs(n_val - 1.0) < 1e-6
    return False


def field_type_string(optic: Optic) -> str:
    """Derive the Optiland field type string from the optic's field definition."""
    fd = optic.fields.field_definition
    if fd is None:
        return "angle"
    return FIELD_CLASS_TO_TYPE.get(type(fd).__name__, "angle")


def compute_abbe_number(
    material: Any,
    n_eval_wavelength: float,
    fallback_nd: float = 1.5,
    fallback_vd: float = 64.17,
) -> tuple[float, float]:
    """Compute (Nd, Vd) for a material lacking a native catalog entry.

    Args:
        material: The material to evaluate.
        n_eval_wavelength: Wavelength (um) at which to report the
            refractive index ``Nd`` (typically the optic's primary
            wavelength). The Abbe number ``Vd`` is always computed from
            the CIE d/F/C wavelengths regardless of this value.
        fallback_nd: Value returned for Nd if evaluation raises.
        fallback_vd: Value returned for Vd if evaluation raises or the
            F-C dispersion is degenerate.

    Returns:
        A tuple of (Nd, Vd).
    """
    try:
        n_d = float(be.atleast_1d(be.array(material.n(n_eval_wavelength))).ravel()[0])
    except Exception:
        n_d = fallback_nd

    try:
        n_f = float(be.atleast_1d(be.array(material.n(WL_F))).ravel()[0])
        n_c = float(be.atleast_1d(be.array(material.n(WL_C))).ravel()[0])
        n_d_cie = float(be.atleast_1d(be.array(material.n(WL_D))).ravel()[0])
        denom = n_f - n_c
        v_d = 99.99 if abs(denom) < 1e-12 else (n_d_cie - 1.0) / denom
    except Exception:
        v_d = fallback_vd

    return n_d, v_d
