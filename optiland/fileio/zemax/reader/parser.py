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
from optiland.physical_apertures import RadialAperture


class ZemaxDataParser:
    """Parses a Zemax .zmx file into a ZemaxDataModel.

    Args:
        filename: Path to the .zmx file to parse.

    Attributes:
        filename: The file path being parsed.
        data_model: The ZemaxDataModel being populated during parsing.
    """

    def __init__(self, filename: str):
        self.filename = filename
        self.data_model = ZemaxDataModel()
        self._current_surf = -1
        self._current_surf_data: dict[str, Any] = {}

        # Operand dispatch table — maps operand string to handler method
        self._operand_table = {
            "NAME": self._read_name,
            "FNUM": self._read_fno,
            "ENPD": self._read_epd,
            "OBNA": self._read_object_na,
            "FLOA": self._read_floating_stop,
            "FTYP": self._read_config_data,
            "XFLN": self._read_x_fields,
            "YFLN": self._read_y_fields,
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

        fields = self.data_model.fields
        fields["num_fields"] = _safe_int(3, 0)
        fields["type"] = {
            0: "angle",
            1: "object_height",
            2: "paraxial_image_height",
            3: "real_image_height",
            4: "theodolite_angle",
        }.get(_safe_int(1, 0), "unsupported")
        self.data_model.wavelengths["num_wavelengths"] = _safe_int(4, 0)
        fields["object_space_telecentric"] = _safe_int(2, 0) == 1
        fields["afocal_image_space"] = _safe_int(7, 0) == 1
        # Legacy files may omit XFLN/YFLN entirely (implying a single
        # on-axis field). Seed defaults so downstream code does not hit
        # KeyError 'x' / 'y'. Any real XFLN/YFLN lines that follow will
        # overwrite these.
        fields.setdefault("x", [0.0])
        fields.setdefault("y", [0.0])
        if fields["num_fields"] <= 0:
            fields["num_fields"] = 1

    def _read_x_fields(self, data: list[str]) -> None:
        n = self.data_model.fields["num_fields"]
        self.data_model.fields["x"] = [float(v) for v in data[1 : n + 1]]

    def _read_y_fields(self, data: list[str]) -> None:
        n = self.data_model.fields["num_fields"]
        self.data_model.fields["y"] = [float(v) for v in data[1 : n + 1]]

    def _read_wavelength(self, data: list[str]) -> None:
        val = float(data[2])
        weight = float(data[3]) if len(data) > 3 else 1.0
        if (
            len(self.data_model.wavelengths["data"])
            < self.data_model.wavelengths["num_wavelengths"]
        ):
            self.data_model.wavelengths["data"].append(val)
            self.data_model.wavelengths["weights"].append(weight)

    def _read_primary_wave(self, data: list[str]) -> None:
        self.data_model.wavelengths["primary_index"] = int(data[1]) - 1

    def _read_surface(self, data: list[str]) -> None:
        if self._current_surf >= 0:
            self.data_model.surfaces[self._current_surf] = self._current_surf_data
        self._current_surf += 1
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
                self._current_surf_data["index"], self._current_surf_data["abbe"]
            )

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
        n = self.data_model.fields["num_fields"]
        self.data_model.fields["weights"] = [float(v) for v in data[1 : n + 1]]

    def _read_vignette_decenter_x(self, data: list[str]) -> None:
        n = self.data_model.fields["num_fields"]
        self.data_model.fields["vignette_decenter_x"] = [
            float(v) for v in data[1 : n + 1]
        ]

    def _read_vignette_decenter_y(self, data: list[str]) -> None:
        n = self.data_model.fields["num_fields"]
        self.data_model.fields["vignette_decenter_y"] = [
            float(v) for v in data[1 : n + 1]
        ]

    def _read_vignette_compress_x(self, data: list[str]) -> None:
        n = self.data_model.fields["num_fields"]
        self.data_model.fields["vignette_compress_x"] = [
            float(v) for v in data[1 : n + 1]
        ]

    def _read_vignette_compress_y(self, data: list[str]) -> None:
        n = self.data_model.fields["num_fields"]
        self.data_model.fields["vignette_compress_y"] = [
            float(v) for v in data[1 : n + 1]
        ]

    def _read_vignette_tangent_angle(self, data: list[str]) -> None:
        n = self.data_model.fields["num_fields"]
        self.data_model.fields["vignette_tangent_angle"] = [
            float(v) for v in data[1 : n + 1]
        ]

    def _read_circular_aperture(self, data: list[str]) -> None:
        self._current_surf_data["aperture"] = RadialAperture(
            r_min=float(data[1]), r_max=float(data[2])
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

        if not sorted_items:
            return

        unzipped = list(zip(*sorted_items, strict=False))
        for i, k in enumerate(keys):
            fields[k] = list(unzipped[i])

    def _finalize_surface(self) -> None:
        """Flush the last in-progress surface into the model."""
        if self._current_surf >= 0:
            self.data_model.surfaces[self._current_surf] = self._current_surf_data
