"""Zemax Data Parser

Parses a Zemax OpticStudio .zmx file into a ZemaxDataModel. The parser uses
a dispatch table of per-operand handler methods to process each line.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import Any

import optiland.backend as be
from optiland.fileio.zemax.model import ZemaxDataModel
from optiland.materials import AbbeMaterial, BaseMaterial, Material
from optiland.materials.material_spec import MatchPolicy
from optiland.physical_apertures import OffsetRadialAperture, RadialAperture

# Fraunhofer d-line (um), used to evaluate a candidate glass's index for
# comparison against the Nd recorded on a GLAS line.
_WL_D = 0.5875618


class ZemaxDataParser:
    """Parses a Zemax .zmx file into a ZemaxDataModel.

    Args:
        filename: Path to the .zmx file to parse.

    Attributes:
        filename: The file path being parsed.
        data_model: The ZemaxDataModel being populated during parsing.
    """

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.data_model = ZemaxDataModel()
        self._current_surf = -1
        self._current_surf_data: dict[str, Any] = {}
        self._current_aperture_offset = (0.0, 0.0)
        self._declared_field_count: int | None = None
        self._declared_wavelength_count: int | None = None
        self._raw_fields: dict[str, list[float]] = {}
        self._wavelength_slots: dict[int, tuple[float, float]] = {}
        self._primary_wavelength_slot: int | None = None
        self._read_config_data(["FTYP"])

        # Operand dispatch table — maps operand string to handler method
        self._operand_table = {
            "NAME": self._read_name,
            "FNUM": self._read_fno,
            "ENPD": self._read_epd,
            "OBNA": self._read_object_na,
            "FLOA": self._read_floating_stop,
            "FTYP": self._read_config_data,
            "XFLD": self._read_x_fields,
            "YFLD": self._read_y_fields,
            "XFLN": self._read_x_fields,
            "YFLN": self._read_y_fields,
            "WAVL": self._read_wavelength,
            "WAVM": self._read_wavelength,
            "PWAV": self._read_primary_wave,
            "SURF": self._read_surface,
            "TYPE": self._read_surf_type,
            "PARM": self._read_surface_parameter,
            "CURV": self._read_radius,
            "DISZ": self._read_thickness,
            "CONI": self._read_conic,
            "GLAS": self._read_glass,
            "STOP": self._read_stop,
            "DIAM": self._read_diameter,
            "MODE": self._read_mode,
            "GCAT": self._read_glass_catalog,
            "FWGN": self._read_field_weights,
            "VDXN": self._read_vignette_decenter_x,
            "VDYN": self._read_vignette_decenter_y,
            "VCXN": self._read_vignette_compress_x,
            "VCYN": self._read_vignette_compress_y,
            "VANN": self._read_vignette_tangent_angle,
            "CLAP": self._read_circular_aperture,
            "FLAP": self._read_floating_aperture,
            "OBDC": self._read_aperture_decenter,
        }

    def parse(self) -> ZemaxDataModel:
        """Read the Zemax file and extract optical data into a ZemaxDataModel.

        Tries UTF-16 LE, UTF-8, and ISO-8859-1 encodings in that order.

        Returns:
            A populated ZemaxDataModel.

        Raises:
            ValueError: If the file cannot be read or contains no aperture data.
        """
        encodings = ["utf-16", "utf-8", "iso-8859-1"]
        success = False
        for encoding in encodings:
            try:
                with open(self.filename, encoding=encoding) as fh:
                    for line in fh:
                        tokens = line.split()
                        if not tokens:
                            continue
                        operand = tokens[0]
                        if operand in self._operand_table:
                            self._operand_table[operand](tokens)
            except (UnicodeError, UnicodeDecodeError):
                continue

            if self.data_model.aperture:
                success = True
                break

        if not success:
            raise ValueError("Failed to read Zemax file.")

        self._finalize_fields()
        self._finalize_surface()
        return self.data_model

    # ------------------------------------------------------------------
    # Per-operand handlers
    # ------------------------------------------------------------------

    def _read_name(self, data: list[str]) -> None:
        self.data_model.name = " ".join(data[1:])

    def _read_fno(self, data: list[str]) -> None:
        if int(data[2]) == 0:
            self.data_model.aperture["imageFNO"] = float(data[1])
        elif int(data[2]) == 1:
            self.data_model.aperture["paraxialImageFNO"] = float(data[1])

    def _read_epd(self, data: list[str]) -> None:
        self.data_model.aperture["EPD"] = float(data[1])

    def _read_object_na(self, data: list[str]) -> None:
        if int(data[2]) == 0:
            self.data_model.aperture["objectNA"] = float(data[1])
        elif int(data[2]) == 1:
            self.data_model.aperture["object_cone_angle"] = float(data[1])

    def _read_floating_stop(self, data: list[str]) -> None:
        self.data_model.aperture["floating_stop"] = True

    def _read_config_data(self, data: list[str]) -> None:
        # Legacy ZEMAX files (e.g. VERS 6133) emit FTYP with only 1-2 tokens
        # where the current format has 8. Read defensively so a short FTYP
        # falls back to defaults instead of raising IndexError.
        def _safe_int(idx: int, default: int = 0) -> int:
            if idx >= len(data):
                return default
            tok = data[idx]
            if not tok:
                return default
            try:
                return int(tok)
            except ValueError:
                return default

        # Defaults describe legacy files; only declared non-negative counts
        # constrain incoming records. In particular, zero is authoritative.
        field_count = _safe_int(3, -1)
        wavelength_count = _safe_int(4, -1)
        self._declared_field_count = field_count if field_count >= 0 else None
        self._declared_wavelength_count = (
            wavelength_count if wavelength_count >= 0 else None
        )
        fields = self.data_model.fields
        fields["type"] = {
            0: "angle",
            1: "object_height",
            2: "paraxial_image_height",
            3: "real_image_height",
            4: "theodolite_angle",
        }.get(_safe_int(1, 0), "unsupported")
        fields["object_space_telecentric"] = _safe_int(2, 0) == 1
        fields["afocal_image_space"] = _safe_int(7, 0) == 1
        self._sync_fields()
        self._sync_wavelengths()

    def _read_field_column(self, key: str, data: list[str]) -> None:
        """Retain the latest full column so later declarations can expand it."""
        self._raw_fields[key] = [float(v) for v in data[1:]]
        self._sync_fields()

    def _sync_fields(self) -> None:
        """Publish bounded columns and align defaults only for omitted axes."""
        fields = self.data_model.fields
        for key, values in self._raw_fields.items():
            fields[key] = values[: self._declared_field_count]
        num_fields = max(
            (len(fields[axis]) for axis in ("x", "y") if axis in self._raw_fields),
            default=1,
        )
        if self._declared_field_count is not None:
            num_fields = min(num_fields, self._declared_field_count)
        for axis in ("x", "y"):
            if axis not in self._raw_fields:
                fields[axis] = [0.0] * num_fields
        fields["num_fields"] = (
            num_fields
            if self._declared_field_count is None
            else self._declared_field_count
        )

    def _read_x_fields(self, data: list[str]) -> None:
        self._read_field_column("x", data)

    def _read_y_fields(self, data: list[str]) -> None:
        self._read_field_column("y", data)

    def _read_wavelength(self, data: list[str]) -> None:
        """Read wavelength/weight pairs in microns, with indexed WAVM slots."""
        if data[0] == "WAVL":
            # Preserve historical token consumption, not the actual WAVL vector
            # layout. Parsing that layout and WWGT remains deferred.
            slot = max(self._wavelength_slots, default=0) + 1
        else:
            slot = int(data[1])
        if slot < 1:
            return
        val = float(data[2])
        weight = float(data[3]) if len(data) > 3 else 1.0
        self._wavelength_slots[slot] = (val, weight)
        self._sync_wavelengths()

    def _read_primary_wave(self, data: list[str]) -> None:
        self._primary_wavelength_slot = int(data[1])
        self._sync_wavelengths()

    def _sync_wavelengths(self) -> None:
        """Publish active slots in order without losing PWAV slot identity."""
        slots = [
            slot
            for slot in sorted(self._wavelength_slots)
            if self._declared_wavelength_count is None
            or slot <= self._declared_wavelength_count
        ]
        wavelengths = self.data_model.wavelengths
        wavelengths["data"] = [self._wavelength_slots[slot][0] for slot in slots]
        wavelengths["weights"] = [self._wavelength_slots[slot][1] for slot in slots]
        wavelengths["num_wavelengths"] = (
            len(slots)
            if self._declared_wavelength_count is None
            else self._declared_wavelength_count
        )
        if self._primary_wavelength_slot is not None:
            # A missing slot must not alias another record when holes compact.
            wavelengths["primary_index"] = (
                slots.index(self._primary_wavelength_slot)
                if self._primary_wavelength_slot in slots
                else None
            )

    def _read_surface(self, data: list[str]) -> None:
        if self._current_surf >= 0:
            self.data_model.surfaces[self._current_surf] = self._current_surf_data
        self._current_surf += 1
        self._current_aperture_offset = (0.0, 0.0)
        self._current_surf_data = {
            "type": "standard",
            "is_stop": False,
            "conic": 0.0,
            "material": "air",
            "aperture": None,
        }

    def _read_radius(self, data: list[str]) -> None:
        try:
            self._current_surf_data["radius"] = 1.0 / float(data[1])
        except ZeroDivisionError:
            self._current_surf_data["radius"] = be.inf

    def _read_thickness(self, data: list[str]) -> None:
        if data[1] == "INFINITY":
            self._current_surf_data["thickness"] = be.inf
        else:
            self._current_surf_data["thickness"] = float(data[1])

    def _read_conic(self, data: list[str]) -> None:
        self._current_surf_data["conic"] = float(data[1])

    def _read_glass(self, data: list[str]) -> None:
        material_name = data[1]
        if material_name.upper() == "MIRROR":
            self._current_surf_data["material"] = "mirror"
            return

        self._current_surf_data["material"] = material_name
        try:
            self._current_surf_data["index"] = float(data[4].replace(",", "."))
            self._current_surf_data["abbe"] = float(data[5].replace(",", "."))
        except IndexError:
            self._current_surf_data["index"] = None
            self._current_surf_data["abbe"] = None

        resolved = self._resolve_glass_by_catalog_and_index(material_name)

        if resolved is not None:
            self._current_surf_data["material"] = resolved
        else:
            # Try to resolve to a real Material from the glass catalog
            try:
                self._current_surf_data["material"] = Material(material_name)
            except ValueError:
                if self.data_model.glass_catalogs:
                    for mfg in self.data_model.glass_catalogs:
                        try:
                            self._current_surf_data["material"] = Material(
                                material_name, mfg.lower()
                            )
                            break
                        except ValueError:
                            continue

        # Fall back to AbbeMaterial if catalog lookup failed
        if not isinstance(self._current_surf_data["material"], BaseMaterial):
            self._current_surf_data["material"] = AbbeMaterial(
                self._current_surf_data["index"],
                self._current_surf_data["abbe"],
                model="buchdahl",
            )

    def _resolve_glass_by_catalog_and_index(
        self, material_name: str
    ) -> BaseMaterial | None:
        """Disambiguate a glass name against the file's declared GCAT catalogs.

        A bare ``Material(name)`` lookup silently returns a single "best
        match" without regard to which catalogs this file actually declares,
        so a name present in several catalogs (e.g. "F2" in both Schott and
        Hikari) can resolve to the wrong one on reload. When the GLAS line
        carries an Nd/Vd pair, use it to pick whichever catalog candidate is
        the closest match instead.

        Returns:
            The resolved material, or None if disambiguation isn't
            applicable (no declared catalogs, no Nd/Vd on the line, or no
            candidate found in any declared catalog).
        """
        index = self._current_surf_data.get("index")
        abbe = self._current_surf_data.get("abbe")
        if not self.data_model.glass_catalogs or index is None or index <= 1.0:
            return None

        candidates = []
        for mfg in self.data_model.glass_catalogs:
            try:
                candidates.append(
                    Material(material_name, mfg.lower(), match_policy=MatchPolicy.BEST)
                )
            except ValueError:
                continue

        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        def _scalar(value) -> float:
            return float(be.atleast_1d(be.array(value)).ravel()[0])

        def _distance(candidate: BaseMaterial) -> float:
            try:
                d_n = _scalar(candidate.n(_WL_D)) - index
                d_v = 0.0
                if abbe is not None:
                    d_v = (_scalar(candidate.abbe()) - abbe) / 100.0
                return d_n**2 + d_v**2
            except Exception:
                return float("inf")

        return min(candidates, key=_distance)

    def _read_stop(self, data: list[str]) -> None:
        self._current_surf_data["is_stop"] = True

    def _read_diameter(self, data: list[str]) -> None:
        self._current_surf_data["diameter"] = float(data[1])

    def _read_mode(self, data: list[str]) -> None:
        if data[1] != "SEQ":
            raise ValueError("Only sequential mode is supported.")

    def _read_glass_catalog(self, data: list[str]) -> None:
        self.data_model.glass_catalogs = data[1:]

    def _read_surf_type(self, data: list[str]) -> None:
        self._current_surf_data["type"] = {
            "STANDARD": "standard",
            "EVENASPH": "even_asphere",
            "ODDASPHE": "odd_asphere",
            "COORDBRK": "coordinate_break",
            "TOROIDAL": "toroidal",
        }.get(data[1], data[1].lower())

    def _read_surface_parameter(self, data: list[str]) -> None:
        key = f"param_{int(data[1]) - 1}"
        self._current_surf_data[key] = float(data[2])

    def _read_field_weights(self, data: list[str]) -> None:
        self._read_field_column("weights", data)

    def _read_vignette_decenter_x(self, data: list[str]) -> None:
        self._read_field_column("vignette_decenter_x", data)

    def _read_vignette_decenter_y(self, data: list[str]) -> None:
        self._read_field_column("vignette_decenter_y", data)

    def _read_vignette_compress_x(self, data: list[str]) -> None:
        self._read_field_column("vignette_compress_x", data)

    def _read_vignette_compress_y(self, data: list[str]) -> None:
        self._read_field_column("vignette_compress_y", data)

    def _read_vignette_tangent_angle(self, data: list[str]) -> None:
        self._read_field_column("vignette_tangent_angle", data)

    def _read_circular_aperture(self, data: list[str]) -> None:
        r_min = float(data[1])
        r_max = float(data[2])
        self._current_surf_data["aperture"] = self._make_circular_aperture(r_min, r_max)

    def _read_floating_aperture(self, data: list[str]) -> None:
        # FLAP stores the clear semi-diameter in its second field; a floating
        # aperture clips rays outside it just like a circular aperture.
        r_max = float(data[2])
        self._current_surf_data["aperture"] = self._make_circular_aperture(0.0, r_max)

    def _make_circular_aperture(self, r_min: float, r_max: float) -> RadialAperture:
        offset_x, offset_y = self._current_aperture_offset
        if offset_x == 0.0 and offset_y == 0.0:
            return RadialAperture(r_min=r_min, r_max=r_max)
        return OffsetRadialAperture(
            r_min=r_min,
            r_max=r_max,
            offset_x=offset_x,
            offset_y=offset_y,
        )

    def _read_aperture_decenter(self, data: list[str]) -> None:
        offset_x = float(data[1])
        offset_y = float(data[2])
        self._current_aperture_offset = (offset_x, offset_y)
        aperture = self._current_surf_data["aperture"]
        if aperture is None:
            return
        self._current_surf_data["aperture"] = self._make_circular_aperture(
            aperture.r_min, aperture.r_max
        )

    # ------------------------------------------------------------------
    # Finalizers
    # ------------------------------------------------------------------

    def _finalize_fields(self) -> None:
        """Deduplicate and sort fields by y-coordinate."""
        fields = self.data_model.fields
        if "x" not in fields or "y" not in fields:
            return

        keys = ["x", "y"]
        for extra in [
            "weights",
            "vignette_decenter_x",
            "vignette_decenter_y",
            "vignette_compress_x",
            "vignette_compress_y",
            "vignette_tangent_angle",
        ]:
            if extra in fields:
                keys.append(extra)

        zipped = list(zip(*(fields[k] for k in keys), strict=False))

        seen = set()
        unique = []
        for item in zipped:
            xy = item[:2]
            if xy not in seen:
                seen.add(xy)
                unique.append(item)

        sorted_items = sorted(unique, key=lambda it: it[1])

        if sorted_items:
            unzipped = list(zip(*sorted_items, strict=False))
            for i, k in enumerate(keys):
                fields[k] = list(unzipped[i])
        if self._declared_field_count is None:
            fields["num_fields"] = len(fields["x"])

    def _finalize_surface(self) -> None:
        """Flush the last in-progress surface into the model."""
        if self._current_surf >= 0:
            self.data_model.surfaces[self._current_surf] = self._current_surf_data
