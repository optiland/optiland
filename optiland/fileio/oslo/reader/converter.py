"""OSLO to Optic Converter

Converts an OsloDataModel into an Optiland Optic object.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.fileio.base import BaseOpticReader
from optiland.fileio.oslo.reader.parser import OsloDataParser
from optiland.fileio.oslo.surfaces import get_handler
from optiland.materials import AbbeMaterial, IdealMaterial, Material
from optiland.optic import Optic

if TYPE_CHECKING:
    from optiland.fileio.oslo.model import OsloDataModel


class OsloToOpticConverter(BaseOpticReader):
    """Converts an OsloDataModel into an Optic object.

    Args:
        oslo_data: OsloDataModel containing the OSLO optical system data.
    """

    def __init__(self, oslo_data: OsloDataModel | None = None):
        self.data = oslo_data
        self.optic: Optic | None = None
        self.current_cs = CoordinateSystem()

    def read(self, source: str) -> Optic:
        """Read an OSLO file and return a fully-configured Optic.

        Args:
            source: Local file path to a .len file.

        Returns:
            A configured Optic instance.
        """
        self.data = OsloDataParser(source).parse()
        self.current_cs = CoordinateSystem()
        return self.convert()

    def convert(self) -> Optic:
        """Convert the stored OSLO data model into an Optic object.

        Returns:
            The fully-configured Optic instance.
        """
        if self.data is None:
            raise ValueError("No OSLO data to convert.")

        self.optic = Optic(self.data.name)
        self._configure_surfaces()
        self._configure_aperture()
        self._configure_fields()
        self._configure_wavelengths()
        return self.optic

    def _configure_surfaces(self) -> None:
        """Configure all surfaces on the optic."""
        # Check if any surface has decenters or tilts
        has_coord_transform = any(
            any(k in sd for k in ["DCX", "DCY", "DCZ", "TLA", "TLB", "TLC"])
            for sd in self.data.surfaces.values()
        )

        for idx in sorted(self.data.surfaces.keys()):
            surf_data = self.data.surfaces[idx]
            self._configure_surface(idx, surf_data, has_coord_transform)

    def _configure_surface(
        self, index: int, data: dict[str, Any], has_coord_transform: bool
    ) -> None:
        # Determine surface type
        # Default is standard. If AD, AE, AF, or AG are present, it's even_asphere.
        oslo_type = "standard"
        if any(k in data for k in ["AD", "AE", "AF", "AG"]):
            oslo_type = "even_asphere"

        handler = get_handler(oslo_type)
        surface_params = handler.parse(data)
        surface_params["index"] = index
        surface_params["is_stop"] = data.get("AST", False)
        surface_params["thickness"] = data.get("TH", 0.0)

        # Handle material
        material_raw = data.get("material", "AIR")
        surface_params["material"] = self._resolve_material(material_raw)

        # Handle aperture (AP is radius in OSLO)
        if "AP" in data:
            from optiland.physical_apertures import RadialAperture

            surface_params["aperture"] = RadialAperture(r_max=data["AP"])

        if has_coord_transform:
            # Resolve effective global position and orientation
            # OSLO decenters/tilts are applied to the surface.
            dx = data.get("DCX", 0.0)
            dy = data.get("DCY", 0.0)
            dz = data.get("DCZ", 0.0)
            rx = be.deg2rad(data.get("TLA", 0.0))
            ry = be.deg2rad(data.get("TLB", 0.0))
            rz = be.deg2rad(data.get("TLC", 0.0))

            if dx != 0 or dy != 0 or dz != 0 or rx != 0 or ry != 0 or rz != 0:
                # Apply transform to current CS
                self.current_cs = CoordinateSystem(
                    x=dx, y=dy, z=dz, rx=rx, ry=ry, rz=rz, reference_cs=self.current_cs
                )

            translation, _ = self.current_cs.get_effective_transform()
            rx_, ry_, rz_ = self.current_cs.get_effective_rotation_euler()

            surface_params.update(
                {
                    "x": float(translation[0]),
                    "y": float(translation[1]),
                    "z": float(translation[2]),
                    "rx": float(rx_),
                    "ry": float(ry_),
                    "rz": float(rz_),
                }
            )

            # Advance CS by thickness for next surface
            th = data.get("TH", 0.0)
            if not be.isinf(th):
                self.current_cs = CoordinateSystem(z=th, reference_cs=self.current_cs)
        else:
            # Standard sequential path, thickness handled by Optiland automatically
            pass

        self.optic.surfaces.add(**surface_params)

        # Handle PY solve (paraxial thickness solve)
        if "PY" in data:
            # PY 0.0 means adjust thickness to focal plane.
            # In Optiland, we can use a solve.
            # TODO: implement PY solve
            pass

    def _resolve_material(self, material_raw: str) -> Any:
        if material_raw == "AIR":
            return "air"
        if material_raw == "RFL":
            return "mirror"

        if material_raw.startswith("GLA "):
            rest = material_raw[4:].strip()
            parts = rest.split()
            if not parts:
                return "air"

            # Case 1: Catalog Glass (e.g. GLA BK7)
            if len(parts) == 1:
                try:
                    return Material(parts[0])
                except ValueError:
                    # Fallback to air or warning
                    return "air"

            # Case 2: Direct Indices (e.g. GLA 1.573 1.573 1.573)
            # Case 3: Modeled Glass (e.g. GLA MOD G1 1.6489 1.662...)
            try:
                if parts[0].upper() == "MOD":
                    # MOD <name> <nd> <n1> <n2>
                    nd = float(parts[2])
                    if len(parts) >= 5:
                        n1 = float(parts[3])
                        n2 = float(parts[4])
                        return self._create_abbe_material(nd, n1, n2)
                    return IdealMaterial(nd)
                else:
                    # <nd> <n1> <n2>
                    nd = float(parts[0])
                    if len(parts) >= 3:
                        n1 = float(parts[1])
                        n2 = float(parts[2])
                        return self._create_abbe_material(nd, n1, n2)
                    return IdealMaterial(nd)
            except (ValueError, IndexError):
                return "air"

        return "air"

    def _create_abbe_material(self, nd: float, n1: float, n2: float) -> Any:
        # Assuming n1 is F and n2 is C if standard wavelengths are used.
        # Abbe V = (n_d - 1) / (n_F - n_C)
        if n1 != n2:
            vd = (nd - 1.0) / (n1 - n2)
            if vd > 0:
                return AbbeMaterial(nd, vd)
        return IdealMaterial(nd)

    def _configure_aperture(self) -> None:
        aperture_data = self.data.aperture
        if "EPD" in aperture_data:
            self.optic.set_aperture("EPD", aperture_data["EPD"])

        if "FNO" in aperture_data:
            self.optic.set_aperture("imageFNO", aperture_data["FNO"])

        if "NAO" in aperture_data:
            self.optic.set_aperture("objectNA", aperture_data["NAO"])

    def _configure_fields(self) -> None:
        field_data = self.data.fields
        if not field_data:
            return

        field_type = field_data.get("type", "angle")
        self.optic.fields.set_type(field_type)

        y_coords = field_data.get("y", [0.0])
        for y in y_coords:
            self.optic.fields.add(y=y, x=0.0)

    def _configure_wavelengths(self) -> None:
        wl_data = self.data.wavelengths
        values = wl_data.get("values", [0.58756])
        weights = wl_data.get("weights", [1.0] * len(values))
        primary_idx = wl_data.get("primary_index", 0)

        # OSLO wavelengths are in microns.
        for idx, val in enumerate(values):
            is_primary = idx == primary_idx
            w = weights[idx] if idx < len(weights) else 1.0
            self.optic.wavelengths.add(value=val, is_primary=is_primary, weight=w)
