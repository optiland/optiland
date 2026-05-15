"""OSLO to Optic Converter

Converts an OsloDataModel into an Optiland Optic object.

Kramer Harrison, 2026
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.fileio.base import BaseOpticReader
from optiland.fileio.oslo.reader.parser import OsloDataParser
from optiland.fileio.oslo.surfaces import get_handler
from optiland.materials import AbbeMaterial, IdealMaterial, Material
from optiland.optic import Optic

# ---------------------------------------------------------------------------
# Fallback glass catalog for OSLO glass names not in the refractiveindex.info
# database.  Values are (nd, Vd) sourced from Schott, Ohara, and CDGM
# historical datasheets.  Used only when Material() lookup fails entirely.
# ---------------------------------------------------------------------------
_OSLO_GLASS_FALLBACK: dict[str, tuple[float, float]] = {
    # --- Barium crown / flint ---
    "BAF13": (1.6670, 48.64),
    "BAF53": (1.7012, 41.47),
    "BAFN10": (1.6700, 47.21),
    "BAFN11": (1.6670, 48.45),
    "BALF51": (1.6035, 60.56),
    "BALKN3": (1.5180, 58.90),
    # --- Barium dense flint ---
    "BASF10": (1.7076, 30.05),
    "BASF51": (1.6692, 45.00),
    "BASF52": (1.7432, 39.17),
    "BASF56": (1.7847, 43.97),
    # --- Borosilicate crown ---
    "BK3": (1.4970, 66.10),
    # --- Fluorite crown ---
    "FK1": (1.4678, 67.81),
    # --- Crown flint ---
    "KF3": (1.5143, 56.47),
    "KZF6": (1.5193, 41.87),
    "KZFN1": (1.5688, 45.43),
    "KZFSN5": (1.5979, 42.08),
    # --- Lanthanum flint ---
    "LAF13": (1.7880, 47.46),
    "LAFN24": (1.7040, 39.10),
    "LAFN28": (1.8052, 25.43),
    # --- Lanthanum crown ---
    "LAK28": (1.7880, 47.35),
    "LAK31": (1.6935, 53.20),
    "LAKN6": (1.6400, 60.10),
    "LAKN12": (1.6779, 55.30),
    "LAKN13": (1.6935, 53.20),
    "LAKN16": (1.7130, 53.83),
    # --- Lanthanum dense flint ---
    "LASF8": (1.7847, 26.09),
    "LASFN31": (1.8830, 40.78),
    # --- Light flint ---
    "LLF7": (1.5750, 41.47),
    # --- Dense flint ---
    "SF16": (1.6200, 36.35),
    "SF17": (1.6517, 33.82),
    "SF18": (1.7215, 29.24),
    "SF53": (1.7280, 28.68),
    "SF62": (1.5163, 64.06),
    "SF63": (1.5677, 42.84),
    # --- Crown glasses ---
    "SK19": (1.6667, 41.99),
    "SKN18": (1.6385, 55.45),
    # --- Dense crown ---
    "SSK51": (1.6030, 38.03),
    "SSKN5": (1.6582, 44.84),
    # --- Special / thallium-free ---
    "TIF1": (1.5860, 41.45),
    # --- Short-flint crown ---
    "ZKN7": (1.5082, 65.49),
}

# Manufacturer prefixes used in OSLO glass names that should be stripped before
# retrying the lookup (e.g. H_LAF2 → LAF2, O_PBH1 → PBH1).
_MANUFACTURER_PREFIXES = ("H_", "O_", "P_", "J_", "E_", "K_")

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
        self._py_surface_indices: list[int] = []

    def read(self, source: str) -> Optic:
        """Read an OSLO file and return a fully-configured Optic.

        Args:
            source: Local file path to a .len file.

        Returns:
            A configured Optic instance.
        """
        self.data = OsloDataParser(source).parse()
        self.current_cs = CoordinateSystem()
        self._py_surface_indices = []
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
        self._apply_py_solves()
        return self.optic

    def _configure_surfaces(self) -> None:
        """Configure all surfaces on the optic."""
        # Check if any surface has decenters or tilts
        has_coord_transform = any(
            any(k in sd for k in ["DCX", "DCY", "DCZ", "TLA", "TLB", "TLC"])
            for sd in self.data.surfaces.values()
        )

        # Determine if any surface is explicitly marked as the stop
        has_stop = any(sd.get("AST", False) for sd in self.data.surfaces.values())

        for idx in sorted(self.data.surfaces.keys()):
            surf_data = self.data.surfaces[idx]

            # OSLO convention: if no stop is specified, it's surface 1
            if not has_stop and idx == 1:
                surf_data["AST"] = True

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

        th = data.get("TH", 0.0)
        # OSLO uses 1e10 as "infinity"; allow for floating-point imprecision
        # (e.g., 9.9999999996e+09 appears in practice).
        if th >= 9.9e9:
            th = be.inf
        surface_params["thickness"] = th

        # Handle material
        material_raw = data.get("material", "AIR")
        surface_params["material"] = self._resolve_material(material_raw)

        # Handle aperture (AP is radius in OSLO).
        # Skip sentinel values like AP 4e9 which mean "infinite aperture".
        if "AP" in data and data["AP"] < 1e6:
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

        # Track surfaces with a PY (paraxial thickness) solve.
        # Actual solve is applied in _apply_py_solves() after full configuration.
        if "PY" in data:
            self._py_surface_indices.append(index)

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
                return self._resolve_catalog_glass(parts[0])

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

    def _resolve_catalog_glass(self, name: str) -> Any:
        """Resolve a single glass name to a Material.

        Resolution order:
        1. Direct lookup via Material() (uses the refractiveindex.info DB).
        2. Strip a known manufacturer prefix (H_, O_, …) and retry.
        3. Fall back to AbbeMaterial using a built-in OSLO glass table.
        4. Warn and return air as a last resort.
        """
        # Step 1 – direct DB lookup.
        try:
            return Material(name)
        except ValueError:
            pass

        # Step 2 – strip manufacturer prefix and retry.
        for prefix in _MANUFACTURER_PREFIXES:
            if name.upper().startswith(prefix):
                base = name[len(prefix) :]
                try:
                    return Material(base)
                except ValueError:
                    pass
                # Also check the fallback table for the base name.
                if base.upper() in _OSLO_GLASS_FALLBACK:
                    nd, vd = _OSLO_GLASS_FALLBACK[base.upper()]
                    return AbbeMaterial(nd, vd, model="buchdahl")
                break

        # Step 3 – built-in OSLO fallback table.
        entry = _OSLO_GLASS_FALLBACK.get(name.upper())
        if entry is not None:
            nd, vd = entry
            return AbbeMaterial(nd, vd, model="buchdahl")

        # Step 4 – cannot resolve; warn and fall back to air.
        warnings.warn(
            f"OSLO glass '{name}' could not be resolved and will be treated as "
            "air.  Add it to _OSLO_GLASS_FALLBACK in oslo/reader/converter.py "
            "if a catalog value is available.",
            UserWarning,
            stacklevel=4,
        )
        return "air"

    def _create_abbe_material(self, nd: float, n1: float, n2: float) -> Any:
        # n1 = nF (F line 0.48613 µm), n2 = nC (C line 0.65627 µm)
        # Abbe V = (nd - 1) / (nF - nC)
        if n1 != n2:
            vd = (nd - 1.0) / (n1 - n2)
            if vd > 0:
                return AbbeMaterial(nd, vd, model="buchdahl")
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
        y_coords = field_data.get("y", [0.0])

        # If object is at infinity, ObjectHeightField is invalid in Optiland.
        # OSLO often uses OBH even for infinite objects, encoding the angle.
        # OSLO OBH sign convention: negative means below axis - take abs().
        if field_type == "object_height" and be.isinf(self.optic.surfaces[0].thickness):
            field_type = "angle"
            y_coords = [abs(float(be.degrees(be.arctan(y / 1e10)))) for y in y_coords]
        elif field_type == "angle":
            # ANG is always positive in OSLO for the max half-angle.
            y_coords = [abs(y) for y in y_coords]

        self.optic.fields.set_type(field_type)

        # OSLO stores only the maximum field value.  Expand to three standard
        # field points (on-axis, 0.7×max, full field) for usable analysis.
        max_y = max((abs(y) for y in y_coords), default=0.0)
        fields_to_add = [0.0, round(0.7 * max_y, 8), max_y] if max_y > 0.0 else [0.0]

        for y in fields_to_add:
            self.optic.fields.add(y=y, x=0.0)

    def _apply_py_solves(self) -> None:
        """Apply paraxial thickness (PY) solves.

        PY 0.0 in an OSLO file means "set this surface's thickness so that
        the paraxial marginal ray focuses at the image plane."  We implement
        this via image_solve(), which drives F2() to zero by correctly setting
        the image distance.
        """
        if not self._py_surface_indices:
            return
        import contextlib

        with contextlib.suppress(Exception):
            self.optic.updater.image_solve()

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
