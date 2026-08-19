"""Surface geometry and material table sections.

Kramer Harrison, 2026
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from optiland.prescription._formatting import fmt_float, fmt_infinity, safe_eval
from optiland.prescription.document import Section, TableBlock, TextBlock
from optiland.prescription.sections.base import BaseSection

if TYPE_CHECKING:
    from optiland.optic.optic import Optic
    from optiland.surfaces.standard_surface import Surface

_FREEFORM_FOOTNOTE = (
    "† Freeform geometry: radius column shows geometry type. "
    "Full parameters available via optic.surfaces.surfaces[i].geometry."
)


def _geometry_type_tag(surface: Surface) -> str:
    geom = surface.geometry
    geom_name = type(geom).__name__
    # Sag count for Zernike-type
    if hasattr(geom, "num_terms"):
        return f"[{geom_name}({geom.num_terms})]"
    return f"[{geom_name}]"


def _radius_str(surface: Surface) -> tuple[str, bool]:
    """Return (radius_string, is_freeform)."""
    geom = surface.geometry
    if not hasattr(geom, "radius"):
        return _geometry_type_tag(surface), True
    try:
        r = float(geom.radius)
    except (TypeError, ValueError):
        return _geometry_type_tag(surface), True
    if math.isinf(r) or abs(r) > 1e12:
        return "∞", False
    return fmt_float(r), False


def _conic_str(surface: Surface) -> str:
    geom = surface.geometry
    if hasattr(geom, "k"):
        return fmt_float(geom.k)
    return "—"


class SurfaceGeometryTableSection(BaseSection):
    """Produces the 'Surface Data — Geometry' section."""

    def build(self, optic: Optic) -> Section:
        """Build the Surface Geometry table section.

        Args:
            optic: The optical system to describe.

        Returns:
            Section containing a TableBlock of surface geometry data.
        """
        surfaces = optic.surfaces.surfaces
        rows = []
        has_freeform = False

        for i, s in enumerate(surfaces):
            radius_str, is_freeform = _radius_str(s)
            if is_freeform:
                has_freeform = True
                radius_str = radius_str + " †"

            thick_str = fmt_infinity(s.thickness)
            conic_str = _conic_str(s)
            stop_str = "✓" if getattr(s, "is_stop", False) else ""
            comment = getattr(s, "comment", "") or ""
            surf_type = getattr(s, "surface_type", None) or type(s.geometry).__name__

            rows.append(
                [str(i), surf_type, radius_str, thick_str, conic_str, stop_str, comment]
            )

        table = TableBlock(
            headers=[
                "S#",
                "Type",
                "Radius (mm)",
                "Thickness (mm)",
                "Conic",
                "Stop",
                "Comment",
            ],
            rows=rows,
        )
        blocks = [table]
        if has_freeform:
            blocks.append(TextBlock(text=_FREEFORM_FOOTNOTE))

        return Section(title="Surface Data — Geometry", blocks=blocks)


class SurfaceMaterialTableSection(BaseSection):
    """Produces the 'Surface Data — Materials' section."""

    def build(self, optic: Optic) -> Section:
        """Build the Surface Materials table section.

        Args:
            optic: The optical system to describe.

        Returns:
            Section containing a TableBlock of surface material data.
        """
        surfaces = optic.surfaces.surfaces
        rows = []

        for i, s in enumerate(surfaces):
            mat_name = _material_name(s)
            nd_str = _nd_str(s)
            vd_str = _vd_str(s)
            semi_str = _semi_aperture_str(s)
            coat_str = _coating_str(s)
            rows.append([str(i), mat_name, nd_str, vd_str, semi_str, coat_str])

        table = TableBlock(
            headers=["S#", "Material", "nd", "Vd", "Semi-Diameter (mm)", "Coating"],
            rows=rows,
        )
        return Section(title="Surface Data — Materials", blocks=[table])


def _material_name(surface: Surface) -> str:
    im = surface.interaction_model
    if getattr(im, "is_reflective", False):
        return "MIRROR"
    mat = surface.material_post
    mat_type = type(mat).__name__
    if mat_type == "IdealMaterial":
        try:
            nd = float(mat.n(0.5876))
        except Exception:
            nd = 1.0
        if abs(nd - 1.0) < 1e-6:
            return "Air"
        return f"Ideal (n={nd:.4f})"
    name = getattr(mat, "name", None)
    if name:
        return str(name)
    return mat_type


def _nd_str(surface: Surface) -> str:
    im = surface.interaction_model
    if getattr(im, "is_reflective", False):
        return "—"
    mat = surface.material_post
    mat_type = type(mat).__name__
    if mat_type == "IdealMaterial":
        try:
            nd = float(mat.n(0.5876))
        except Exception:
            return "N/A"
        if abs(nd - 1.0) < 1e-6:
            return "1.0000"
        return fmt_float(nd)
    return safe_eval(lambda: mat.n(0.5876), infinity=False)


def _vd_str(surface: Surface) -> str:
    im = surface.interaction_model
    if getattr(im, "is_reflective", False):
        return "—"
    mat = surface.material_post
    mat_type = type(mat).__name__
    if mat_type == "IdealMaterial":
        return "—"
    if not hasattr(mat, "abbe"):
        return "—"
    return safe_eval(lambda: mat.abbe(), infinity=False)


def _semi_aperture_str(surface: Surface) -> str:
    val = getattr(surface, "semi_aperture", None)
    if val is None:
        return "—"
    return safe_eval(lambda: val, infinity=False)


def _coating_str(surface: Surface) -> str:
    try:
        coating = surface.interaction_model.coating
    except AttributeError:
        return "—"
    if coating is None:
        return "—"
    return type(coating).__name__
