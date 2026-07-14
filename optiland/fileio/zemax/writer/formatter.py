"""Optic to Zemax Converter

Converts an Optiland Optic object into a ZemaxDataModel. This is the mirror
of ZemaxToOpticConverter and is the first stage of the write pipeline.

Kramer Harrison, 2024
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.fileio.common import compute_abbe_number, field_type_string, is_air
from optiland.fileio.zemax.model import ZemaxDataModel
from optiland.fileio.zemax.surfaces import (
    CoordinateBreakSurfaceHandler,
    get_handler_for_optiland_type,
)
from optiland.materials.material import Material

if TYPE_CHECKING:
    from optiland.optic import Optic

# Map from Optiland aperture type to Zemax operand string
_AP_TYPE_TO_OPERAND: dict[str, str] = {
    "EPD": "ENPD",
    "imageFNO": "FNUM",
    "paraxialImageFNO": "PFIL",
    "objectNA": "OBNA",
    "float_by_stop_size": "FLOA",
}

# Map from Optiland field type string to Zemax FTYP integer
_FIELD_TYPE_TO_FTYP: dict[str, int] = {
    "angle": 0,
    "object_height": 1,
    "paraxial_image_height": 2,
    "real_image_height": 3,
}

# Map from Optiland geometry str() to Optiland surface type string.
# "Planar" (flat surface) maps to "standard" since Zemax encodes it as
# TYPE STANDARD with CURV 0.
_GEOM_STR_TO_TYPE: dict[str, str] = {
    "Planar": "standard",
    "Standard": "standard",
    "Even Asphere": "even_asphere",
    "Odd Asphere": "odd_asphere",
    "Toroidal": "toroidal",
}


class OpticToZemaxConverter:
    """Converts an Optic object to a ZemaxDataModel.

    This is the mirror of ZemaxToOpticConverter and constitutes the first
    stage of the write pipeline (Optic -> ZemaxDataModel -> text lines).

    Args:
        optic: The Optic to convert.
    """

    def __init__(self, optic: Optic):
        self._optic = optic

    def convert(self) -> ZemaxDataModel:
        """Build and return the ZemaxDataModel for the optic.

        Returns:
            A populated ZemaxDataModel ready for ZemaxFileEncoder.
        """
        model = ZemaxDataModel()
        model.name = self._optic.name
        self._convert_aperture(model)
        self._convert_fields(model)
        self._convert_wavelengths(model)
        self._warn_pickups_solves()
        self._convert_surfaces(model)
        return model

    # ------------------------------------------------------------------
    # Aperture
    # ------------------------------------------------------------------

    def _convert_aperture(self, model: ZemaxDataModel) -> None:
        ap = self._optic.aperture
        if ap is None:
            return
        operand = _AP_TYPE_TO_OPERAND.get(ap.ap_type)
        if operand is None:
            warnings.warn(
                f"Unknown aperture type '{ap.ap_type}'; skipping aperture export.",
                UserWarning,
                stacklevel=3,
            )
            return
        model.aperture[ap.ap_type] = float(ap.value)

    # ------------------------------------------------------------------
    # Fields
    # ------------------------------------------------------------------

    def _convert_fields(self, model: ZemaxDataModel) -> None:
        field_type = field_type_string(self._optic)
        ftyp_int = _FIELD_TYPE_TO_FTYP.get(field_type, 0)

        fields = self._optic.fields
        n = fields.num_fields

        x_vals = [float(f.x) for f in fields]
        y_vals = [float(f.y) for f in fields]

        # Vignetting — try vx/vy, fallback to zeros
        try:
            vcx = [float(f.vx) for f in fields]
            vcy = [float(f.vy) for f in fields]
        except AttributeError:
            vcx = [0.0] * n
            vcy = [0.0] * n

        model.fields = {
            "num_fields": n,
            "type": field_type,
            "ftyp_int": ftyp_int,
            "x": x_vals,
            "y": y_vals,
            "weights": [1.0] * n,
            "vignette_compress_x": vcx,
            "vignette_compress_y": vcy,
            "vignette_decenter_x": [0.0] * n,
            "vignette_decenter_y": [0.0] * n,
            "vignette_tangent_angle": [0.0] * n,
        }

    # ------------------------------------------------------------------
    # Wavelengths
    # ------------------------------------------------------------------

    def _convert_wavelengths(self, model: ZemaxDataModel) -> None:
        wls = self._optic.wavelengths
        data: list[float] = []
        primary_index = 0
        for i, w in enumerate(wls):
            data.append(float(w.value))
            if w.is_primary:
                primary_index = i

        model.wavelengths = {
            "data": data,
            "num_wavelengths": len(data),
            "primary_index": primary_index,
        }

    # ------------------------------------------------------------------
    # Pickups / Solves warning
    # ------------------------------------------------------------------

    def _warn_pickups_solves(self) -> None:
        pickups = list(self._optic.pickups.pickups)
        solves = list(self._optic.solves.solves)
        if pickups:
            warnings.warn(
                f"Optic has {len(pickups)} pickup(s) that cannot be represented "
                "in a .zmx file; resolved values will be exported instead.",
                UserWarning,
                stacklevel=3,
            )
        if solves:
            warnings.warn(
                f"Optic has {len(solves)} solve(s) that cannot be represented "
                "in a .zmx file; resolved values will be exported instead.",
                UserWarning,
                stacklevel=3,
            )

    # ------------------------------------------------------------------
    # Surfaces
    # ------------------------------------------------------------------

    def _convert_surfaces(self, model: ZemaxDataModel) -> None:
        """Iterate optic surfaces and populate model.surfaces.

        For surfaces with non-trivial coordinate systems (tilts/decenters),
        synthetic COORDBRK entries are inserted before and after.
        """
        glass_catalogs: list[str] = []
        output_idx = 0
        cb_handler = CoordinateBreakSurfaceHandler()

        for surface in self._optic.surfaces:
            optiland_type = self._resolve_geometry_type(surface, output_idx)
            cs_angles = self._coordinate_break_angles(surface.geometry.cs)

            if cs_angles is not None:
                model.surfaces[output_idx] = cb_handler.format_cs(
                    dx=float(surface.geometry.cs.x),
                    dy=float(surface.geometry.cs.y),
                    dz=0.0,
                    rx_deg=cs_angles[0],
                    ry_deg=cs_angles[1],
                    rz_deg=cs_angles[2],
                )
                output_idx += 1

            raw = self._encode_surface_body(
                surface, optiland_type, output_idx, glass_catalogs
            )
            model.surfaces[output_idx] = raw
            output_idx += 1

            if cs_angles is not None:
                cs = surface.geometry.cs
                model.surfaces[output_idx] = cb_handler.format_cs(
                    dx=-float(cs.x),
                    dy=-float(cs.y),
                    dz=0.0,
                    rx_deg=-cs_angles[0],
                    ry_deg=-cs_angles[1],
                    rz_deg=-cs_angles[2],
                )
                output_idx += 1

        if glass_catalogs:
            # Unique catalog names, preserving order
            model.glass_catalogs = list(dict.fromkeys(glass_catalogs))

    def _resolve_geometry_type(self, surface: Any, output_idx: int) -> str:
        """Map a surface's geometry to a Zemax-supported Optiland type string."""
        geom_str = str(surface.geometry)
        optiland_type = _GEOM_STR_TO_TYPE.get(geom_str)
        if optiland_type is None:
            raise NotImplementedError(
                f"Surface {output_idx}: geometry type '{geom_str}' "
                "is not supported by the Zemax writer."
            )
        return optiland_type

    def _coordinate_break_angles(self, cs: Any) -> tuple[float, float, float] | None:
        """Return (rx, ry, rz) in degrees if *cs* has a non-trivial transform."""
        has_tilt = any(
            abs(float(getattr(cs, attr, 0.0))) > 1e-12 for attr in ("rx", "ry", "rz")
        )
        has_decenter = any(
            abs(float(getattr(cs, attr, 0.0))) > 1e-12 for attr in ("x", "y")
        )
        if not (has_tilt or has_decenter):
            return None
        return (
            math.degrees(float(cs.rx)),
            math.degrees(float(cs.ry)),
            math.degrees(float(cs.rz)),
        )

    def _encode_surface_body(
        self,
        surface: Any,
        optiland_type: str,
        output_idx: int,
        glass_catalogs: list[str],
    ) -> dict[str, Any]:
        """Build the raw Zemax surface dict for a single optic surface."""
        handler = get_handler_for_optiland_type(optiland_type)
        raw = handler.format(surface)

        thickness = float(be.atleast_1d(be.array(surface.thickness)).ravel()[0])
        raw["DISZ"] = "INFINITY" if be.isinf(thickness) else thickness

        if surface.is_stop:
            raw["STOP"] = True

        # Semi-aperture (DIAM)
        # For float_by_stop_size aperture, the stop surface must carry DIAM
        ap = self._optic.aperture
        if surface.is_stop and ap is not None and ap.ap_type == "float_by_stop_size":
            raw["DIAM"] = float(ap.value)
        elif surface.semi_aperture is not None:
            raw["DIAM"] = float(surface.semi_aperture)

        # Physical aperture (CLAP)
        if surface.aperture is not None:
            raw["CLAP"] = surface.aperture

        # Glass
        glass_entry = self._format_glass(
            surface.material_post, output_idx, glass_catalogs
        )
        if glass_entry is not None:
            raw["GLAS"] = glass_entry

        return raw

    def _format_glass(
        self,
        mat: Any,
        surf_idx: int,
        glass_catalogs: list[str],
    ) -> dict[str, Any] | None:
        """Build a GLAS operand dict for a material.

        Args:
            mat: The surface material (post).
            surf_idx: Output surface index (used in warnings).
            glass_catalogs: Mutable list to accumulate catalog names.

        Returns:
            A dict with keys ``name`` and optionally ``catalog``,
            or None for air.
        """
        if is_air(mat):
            return None

        # Mirror
        if isinstance(mat, str) and mat.lower() == "mirror":
            return {"name": "MIRROR"}

        # Catalog glass (Material from glass catalog)
        if isinstance(mat, Material) and mat.reference:
            catalog = mat.reference.upper()
            glass_catalogs.append(catalog)
            return {"name": mat.name.upper(), "catalog": catalog}

        # Named glass without explicit reference — try to use name only
        if isinstance(mat, Material):
            return {"name": mat.name.upper()}

        # AbbeMaterial or any other material -> MODEL glass
        n_d, v_num = compute_abbe_number(mat, float(self._optic.primary_wavelength))

        mat_name = getattr(mat, "name", type(mat).__name__)
        warnings.warn(
            f"Surface {surf_idx}: glass '{mat_name}' has no Zemax catalog entry; "
            f"writing as MODEL glass (n={n_d:.6f}, V={v_num:.2f}). "
            "Round-trip fidelity is not guaranteed.",
            UserWarning,
            stacklevel=4,
        )
        return {"name": "MODEL", "n": n_d, "V": v_num}
