"""CODE V to Optic Converter

Converts a CodeVDataModel into an Optiland Optic object. This module also
provides the ``load_codev_file`` entry point for the reader path.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import Any

import optiland.backend as be
from optiland.fileio.base import BaseOpticReader
from optiland.fileio.codev.model import CodeVDataModel
from optiland.fileio.codev.reader.parser import CodeVDataParser
from optiland.optic import Optic

# Map from CODE V aperture key to Optiland aperture type
_APERTURE_KEY_MAP: dict[str, str] = {
    "EPD": "EPD",
    "FNO": "imageFNO",
    "NAO": "objectNA",
}


class CodeVToOpticConverter(BaseOpticReader):
    """Converts a CodeVDataModel into an Optic object.

    Also implements BaseOpticReader so that the full pipeline (file load ->
    parsing -> conversion) can be triggered via ``read()``.

    Args:
        codev_data: A plain dict or CodeVDataModel containing the CODE V
            optical system data.

    Attributes:
        data: The CODE V data as a plain dict.
        optic: The Optic instance built by :py:meth:`convert`.
    """

    def __init__(self, codev_data: dict[str, Any] | CodeVDataModel):
        if isinstance(codev_data, CodeVDataModel):
            self.data = codev_data.to_dict()
        else:
            self.data = dict(codev_data)
        self.optic: Optic | None = None

    # ------------------------------------------------------------------
    # BaseOpticReader
    # ------------------------------------------------------------------

    def read(self, source: str) -> Optic:
        """Read a CODE V .seq file and return a fully-configured Optic.

        Args:
            source: Local file path to a .seq file.

        Returns:
            A configured Optic instance.
        """
        data_model = CodeVDataParser(source).parse()
        self.data = data_model.to_dict()
        return self.convert()

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def convert(self) -> Optic:
        """Convert the stored CODE V data dict into an Optic object.

        Returns:
            The fully-configured Optic instance.
        """
        self.optic = Optic(self.data.get("name"))
        self._configure_surfaces()
        self._configure_aperture()
        self._configure_fields()
        self._configure_wavelengths()
        return self.optic

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _configure_surfaces(self) -> None:
        """Configure all surfaces on the optic.

        All surfaces from the data model — including object (SO) and image
        (SI) — are added sequentially. The Optiland surface factory
        automatically creates an ObjectSurface for index 0.

        CODE V XDE/YDE/ADE/BDE/CDE modifiers are mapped directly to the
        ``dx``/``dy``/``rx``/``ry``/``rz`` surface parameters so that
        thickness-based sequential propagation is preserved.
        """
        surfaces = self.data.get("surfaces", {})
        sto_index = self.data.get("sto_surface_index")

        # Ensure the first surface is an object-type surface.  Files that
        # contain only bare S lines (no SO / SI) need an implicit object.
        sorted_keys = sorted(surfaces.keys(), key=int)
        first_surf = surfaces[sorted_keys[0]] if sorted_keys else {}
        if first_surf.get("type", "standard") not in ("object",):
            implicit_obj: dict[str, Any] = {
                "type": "object",
                "radius": float(be.inf),
                "thickness": float(be.inf),
                "material": None,
                "is_stop": False,
                "conic": 0.0,
                "coefficients": [],
                "xde": 0.0,
                "yde": 0.0,
                "zde": 0.0,
                "ade": 0.0,
                "bde": 0.0,
                "cde": 0.0,
                "aperture": None,
            }
            # Renumber existing surfaces to make room for index 0
            new_surfaces: dict[int, dict[str, Any]] = {0: implicit_obj}
            for new_k, old_k in enumerate(sorted_keys, start=1):
                new_surfaces[new_k] = surfaces[old_k]
            surfaces = new_surfaces
            sorted_keys = sorted(surfaces.keys(), key=int)
            if sto_index is not None:
                sto_index += 1  # shift by 1 for the inserted object surface

        has_stop = any(sd.get("is_stop", False) for sd in surfaces.values())

        for surf_idx, idx in enumerate(sorted_keys):
            surf = surfaces[idx]
            # Apply explicit global stop reference (STO Sn)
            if sto_index is not None and surf_idx == sto_index:
                surf = dict(surf)
                surf["is_stop"] = True
                has_stop = True
            # Default stop: if no STO in file, surface 1 (first real surface)
            if not has_stop and surf_idx == 1:
                surf = dict(surf)
                surf["is_stop"] = True
            surface_params = self._build_surface_params(surf, surf_idx)
            self.optic.surfaces.add(**surface_params)

    def _build_surface_params(
        self, surf: dict[str, Any], surf_idx: int
    ) -> dict[str, Any]:
        """Build the kwargs dict for ``optic.surfaces.add()``.

        CODE V XDE/YDE/ZDE/ADE/BDE/CDE modifiers are mapped to Optiland's
        ``dx``/``dy``/``rx``/``ry``/``rz`` surface parameters so that the
        sequential propagation thickness is preserved.

        Args:
            surf: Raw surface dict from the parser.
            surf_idx: Sequential surface index within the optic.

        Returns:
            Keyword-argument dict for ``optic.surfaces.add()``.
        """
        # Object and image surfaces map to "standard" in Optiland's factory.
        # (The factory creates ObjectSurface for index 0 automatically.)
        surf_type_cv = surf.get("type", "standard")
        if surf_type_cv in ("object", "image"):
            optiland_type = "standard"
        else:
            profile = surf.get("profile", "SPH")
            if surf.get("coefficients"):
                profile = "ASP"
            optiland_type = "even_asphere" if profile == "ASP" else "standard"

        material = surf.get("material") or "air"

        thickness = surf.get("thickness", 0.0)
        # Large object distances (≥ 1e10) represent optical infinity in CODE V
        if surf_type_cv == "object" and abs(float(thickness)) >= 1e10:
            thickness = float(be.inf)

        params: dict[str, Any] = {
            "index": surf_idx,
            "surface_type": optiland_type,
            "radius": surf.get("radius", float(be.inf)),
            "conic": surf.get("conic", 0.0),
            "thickness": thickness,
            "is_stop": surf.get("is_stop", False),
            "material": material,
        }

        coefficients = surf.get("coefficients")
        if coefficients:
            params["coefficients"] = coefficients

        if surf.get("aperture") is not None:
            params["aperture"] = surf["aperture"]

        # Map CODE V surface decenters/tilts to Optiland surface parameters.
        xde = float(surf.get("xde", 0.0))
        yde = float(surf.get("yde", 0.0))
        ade_deg = float(surf.get("ade", 0.0))
        bde_deg = float(surf.get("bde", 0.0))
        cde_deg = float(surf.get("cde", 0.0))
        if xde or yde or ade_deg or bde_deg or cde_deg:
            params["dx"] = xde
            params["dy"] = yde
            params["rx"] = float(be.deg2rad(ade_deg))
            params["ry"] = float(be.deg2rad(bde_deg))
            params["rz"] = float(be.deg2rad(cde_deg))

        return params

    def _configure_aperture(self) -> None:
        """Configure the system aperture on the optic."""
        aperture_data = self.data.get("aperture", {})
        if not aperture_data:
            return

        if "NA" in aperture_data:
            raise NotImplementedError(
                "CODE V image-space numerical aperture (NA) is not yet "
                "supported in Optiland."
            )

        for cv_key, optiland_key in _APERTURE_KEY_MAP.items():
            if cv_key in aperture_data:
                self.optic.set_aperture(
                    aperture_type=optiland_key, value=float(aperture_data[cv_key])
                )
                return

        raise ValueError("No valid aperture type found in CODE V data.")

    def _configure_fields(self) -> None:
        """Configure the field group on the optic."""
        fields = self.data.get("fields", {})
        field_type = fields.get("type", "angle")
        self.optic.fields.set_type(field_type=field_type)

        field_x = fields.get("x", [0.0])
        field_y = fields.get("y", [0.0])

        for k in range(len(field_y)):
            x = field_x[k] if k < len(field_x) else 0.0
            y = field_y[k]
            self.optic.fields.add(x=float(x), y=float(y))

    def _configure_wavelengths(self) -> None:
        """Configure the wavelength group on the optic."""
        wl_data = self.data.get("wavelengths", {})
        primary_idx = wl_data.get("primary_index", 0)
        for idx, value in enumerate(wl_data.get("data", [])):
            self.optic.wavelengths.add(
                value=float(value), is_primary=(idx == primary_idx)
            )
