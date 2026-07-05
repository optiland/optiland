"""Optic to CODE V Converter

Converts an Optiland Optic object into a CodeVDataModel. This is the mirror
of CodeVToOpticConverter and is the first stage of the write pipeline.

Kramer Harrison, 2026
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.fileio.codev.model import CodeVDataModel
from optiland.fileio.codev.surfaces import get_handler_for_optiland_type
from optiland.fileio.common import compute_abbe_number, field_type_string, is_air
from optiland.materials.material import Material

if TYPE_CHECKING:
    from optiland.optic import Optic

# Map from Optiland geometry str() to surface type string
_GEOM_STR_TO_TYPE: dict[str, str] = {
    "Planar": "standard",
    "Standard": "standard",
    "Even Asphere": "even_asphere",
}


class OpticToCodeVConverter:
    """Converts an Optic object to a CodeVDataModel.

    This is the mirror of CodeVToOpticConverter and constitutes the first
    stage of the write pipeline (Optic -> CodeVDataModel -> text lines).

    Args:
        optic: The Optic to convert.
    """

    def __init__(self, optic: Optic):
        self._optic = optic

    def convert(self) -> CodeVDataModel:
        """Build and return the CodeVDataModel for the optic.

        Returns:
            A populated CodeVDataModel ready for CodeVFileEncoder.
        """
        model = CodeVDataModel()
        model.name = self._optic.name
        model.radius_mode = True  # always write radii
        self._convert_aperture(model)
        self._convert_fields(model)
        self._convert_wavelengths(model)
        self._warn_pickups_solves()
        self._convert_surfaces(model)
        return model

    # ------------------------------------------------------------------
    # Aperture
    # ------------------------------------------------------------------

    def _convert_aperture(self, model: CodeVDataModel) -> None:
        ap = self._optic.aperture
        if ap is None:
            return
        # Map Optiland aperture types to CODE V keys
        mapping = {
            "EPD": "EPD",
            "imageFNO": "FNO",
            "paraxialImageFNO": "FNO",
            "objectNA": "NAO",
            "imageNA": "NA",
            "float_by_stop_size": "EPD",
        }
        cv_key = mapping.get(ap.ap_type)
        if cv_key is None:
            warnings.warn(
                f"Unknown aperture type '{ap.ap_type}'; skipping aperture export.",
                UserWarning,
                stacklevel=3,
            )
            return
        model.aperture[cv_key] = float(ap.value)

    # ------------------------------------------------------------------
    # Fields
    # ------------------------------------------------------------------

    def _convert_fields(self, model: CodeVDataModel) -> None:
        field_type = field_type_string(self._optic)
        fields = self._optic.fields
        n = fields.num_fields

        x_vals = [float(f.x) for f in fields]
        y_vals = [float(f.y) for f in fields]

        try:
            weights = [float(f.weight) for f in fields]
        except AttributeError:
            weights = [1.0] * n

        model.fields = {
            "num_fields": n,
            "type": field_type,
            "x": x_vals,
            "y": y_vals,
            "weights": weights,
        }

    # ------------------------------------------------------------------
    # Wavelengths
    # ------------------------------------------------------------------

    def _convert_wavelengths(self, model: CodeVDataModel) -> None:
        wls = self._optic.wavelengths
        data: list[float] = []
        primary_index = 0
        for i, w in enumerate(wls):
            data.append(float(w.value))
            if w.is_primary:
                primary_index = i

        model.wavelengths = {
            "data": data,  # in um; encoder converts to nm
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
                "in a .seq file; resolved values will be exported instead.",
                UserWarning,
                stacklevel=3,
            )
        if solves:
            warnings.warn(
                f"Optic has {len(solves)} solve(s) that cannot be represented "
                "in a .seq file; resolved values will be exported instead.",
                UserWarning,
                stacklevel=3,
            )

    # ------------------------------------------------------------------
    # Surfaces
    # ------------------------------------------------------------------

    def _convert_surfaces(self, model: CodeVDataModel) -> None:
        """Iterate optic surfaces and populate model.surfaces.

        The ObjectSurface (index 0) is written as SO, the remaining real
        surfaces as S, and the last surface as SI.
        """
        all_surfaces = list(self._optic.surfaces)

        if not all_surfaces:
            return

        # Split: object surface (first), real surfaces (middle), image (last)
        obj_surface = all_surfaces[0]
        real_surfaces = all_surfaces[1:-1] if len(all_surfaces) > 2 else []

        output_idx = 0

        # Object surface (SO)
        obj_thickness = float(be.atleast_1d(be.array(obj_surface.thickness)).ravel()[0])
        model.surfaces[output_idx] = {
            "type": "object",
            "radius": 0.0,
            "thickness": obj_thickness,
        }
        output_idx += 1

        for surface in real_surfaces:
            model.surfaces[output_idx] = self._convert_real_surface(surface, output_idx)
            output_idx += 1

        # Image surface
        model.surfaces[output_idx] = {
            "type": "image",
            "radius": 0.0,
            "thickness": 0.0,
        }

    def _convert_real_surface(self, surface: Any, output_idx: int) -> dict[str, Any]:
        """Build the raw CODE V surface dict for a single non-object/image surface."""
        geom = surface.geometry
        geom_str = str(geom)
        optiland_type = _GEOM_STR_TO_TYPE.get(geom_str)

        if optiland_type is None:
            raise NotImplementedError(
                f"Surface {output_idx}: geometry type '{geom_str}' "
                "is not supported by the CODE V writer."
            )

        handler = get_handler_for_optiland_type(optiland_type)
        raw = handler.format(surface)

        thickness = float(be.atleast_1d(be.array(surface.thickness)).ravel()[0])
        raw["thickness"] = thickness
        raw["type"] = "standard"

        if surface.is_stop:
            raw["is_stop"] = True

        # Physical aperture
        if surface.aperture is not None:
            raw["aperture"] = surface.aperture

        # Glass — check reflective flag first (mirror)
        is_reflective = getattr(
            getattr(surface, "interaction_model", None), "is_reflective", False
        )
        glass_entry = self._format_glass(
            surface.material_post, output_idx, is_reflective
        )
        if glass_entry is not None:
            raw["glass"] = glass_entry

        raw.update(self._encode_decenters_tilts(geom.cs))
        return raw

    def _encode_decenters_tilts(self, cs: Any) -> dict[str, float]:
        """Encode a coordinate system's decenters/tilts as CODE V raw keys."""
        values = {
            "xde": float(getattr(cs, "x", 0.0)),
            "yde": float(getattr(cs, "y", 0.0)),
            "ade": math.degrees(float(getattr(cs, "rx", 0.0))),
            "bde": math.degrees(float(getattr(cs, "ry", 0.0))),
            "cde": math.degrees(float(getattr(cs, "rz", 0.0))),
        }
        return {k: v for k, v in values.items() if abs(v) > 1e-12}

    def _format_glass(
        self,
        mat: Any,
        surf_idx: int,
        is_reflective: bool = False,
    ) -> dict[str, Any] | None:
        """Build a glass specification dict for a material.

        Args:
            mat: The surface material (post).
            surf_idx: Output surface index (used in warnings).
            is_reflective: True if the surface is a reflective (mirror) surface.

        Returns:
            A dict with key ``"name"`` (and optionally ``"catalog"``),
            or None for air.
        """
        # Mirror surface — detected via interaction_model.is_reflective
        if is_reflective:
            return {"name": "REFL"}

        if is_air(mat):
            return None

        # Mirror material string fallback
        if isinstance(mat, str) and mat.lower() == "mirror":
            return {"name": "REFL"}

        # Catalog glass
        if isinstance(mat, Material) and mat.reference:
            catalog = mat.reference.upper()
            return {"name": mat.name.upper(), "catalog": catalog}

        if isinstance(mat, Material):
            return {"name": mat.name.upper()}

        # AbbeMaterial or unknown -> Nd:Vd fictitious glass
        n_d, v_d = compute_abbe_number(mat, float(self._optic.primary_wavelength))

        mat_name = getattr(mat, "name", type(mat).__name__)
        warnings.warn(
            f"Surface {surf_idx}: glass '{mat_name}' has no CODE V catalog entry; "
            f"writing as fictitious glass (Nd={n_d:.6f}, Vd={v_d:.2f}). "
            "Round-trip fidelity is not guaranteed.",
            UserWarning,
            stacklevel=4,
        )
        return {"nd": n_d, "vd": v_d}
