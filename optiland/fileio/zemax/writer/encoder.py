"""Zemax File Encoder

Converts a ZemaxDataModel into a list of .zmx text lines in the order and
format expected by OpticStudio. The encoded lines are written to disk as
UTF-16 LE by save_zemax_file().

Kramer Harrison, 2024
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from optiland.fileio.zemax.model import ZemaxDataModel

# ---------------------------------------------------------------------------
# Aperture type -> Zemax operand
# ---------------------------------------------------------------------------
_AP_TYPE_TO_OPERAND: dict[str, str] = {
    "EPD": "ENPD",
    "imageFNO": "FNUM",
    "paraxialImageFNO": "PFIL",
    "objectNA": "OBNA",
    "float_by_stop_size": "FLOA",
}

_FIELD_TYPE_TO_FTYP: dict[str, int] = {
    "angle": 0,
    "object_height": 1,
    "paraxial_image_height": 2,
    "real_image_height": 3,
}


def _fmt(value: float) -> str:
    """Format a float in Zemax scientific notation."""
    return f"{value:.8E}"


def _fmt_vals(values: list[float]) -> str:
    """Format a list of floats separated by spaces."""
    return " ".join(_fmt(v) for v in values)


class ZemaxFileEncoder:
    """Encodes a ZemaxDataModel as a list of .zmx text lines.

    Args:
        model: The ZemaxDataModel to encode.
    """

    def __init__(self, model: ZemaxDataModel):
        self._model = model

    def encode(self) -> list[str]:
        """Produce the complete list of .zmx text lines.

        Returns:
            A list of strings, one per line of the output file.
        """
        lines: list[str] = []
        self._encode_header(lines)
        self._encode_surfaces(lines)
        return lines

    # ------------------------------------------------------------------
    # Header block
    # ------------------------------------------------------------------

    def _encode_header(self, lines: list[str]) -> None:
        lines.append("VERS 240000 3 0")
        lines.append("MODE SEQ")
        if self._model.name:
            lines.append(f"NAME {self._model.name}")
        else:
            lines.append("NAME")
        lines.append("NOTE 0")
        lines.append("UNIT MM X W X CM MR CPMM")

        # Aperture operand
        self._encode_aperture(lines)

        # Fields header
        self._encode_fields_header(lines)

        # Wavelengths
        self._encode_wavelengths(lines)

        # Glass catalogs
        if self._model.glass_catalogs:
            lines.append("GCAT " + " ".join(self._model.glass_catalogs))

        # Field coordinate arrays
        fields = self._model.fields
        n = fields.get("num_fields", 0)
        if n > 0:
            zeros = [0.0] * n
            ones = [1.0] * n

            def _arr(key: str, default: list[float]) -> str:
                return " ".join(_fmt(v) for v in fields.get(key, default))

            lines.append("XFLN " + _arr("x", zeros))
            lines.append("YFLN " + _arr("y", zeros))
            lines.append("FWGN " + _arr("weights", ones))
            lines.append("VDXN " + _arr("vignette_decenter_x", zeros))
            lines.append("VDYN " + _arr("vignette_decenter_y", zeros))
            lines.append("VCXN " + _arr("vignette_compress_x", zeros))
            lines.append("VCYN " + _arr("vignette_compress_y", zeros))
            lines.append("VANN " + _arr("vignette_tangent_angle", zeros))

    def _encode_aperture(self, lines: list[str]) -> None:
        ap = self._model.aperture
        if not ap:
            return
        for ap_type, operand in _AP_TYPE_TO_OPERAND.items():
            if ap_type in ap:
                if operand == "FLOA":
                    lines.append("FLOA")
                elif operand in ("FNUM", "PFIL"):
                    # FNUM has a second argument for paraxial/real flag
                    flag = 1 if operand == "PFIL" else 0
                    lines.append(f"{operand} {_fmt(ap[ap_type])} {flag}")
                elif operand == "OBNA":
                    lines.append(f"OBNA {_fmt(ap[ap_type])} 0")
                else:
                    lines.append(f"{operand} {_fmt(ap[ap_type])}")
                break

    def _encode_fields_header(self, lines: list[str]) -> None:
        fields = self._model.fields
        n = fields.get("num_fields", 0)
        ftyp_int = fields.get(
            "ftyp_int",
            _FIELD_TYPE_TO_FTYP.get(fields.get("type", "angle"), 0),
        )
        # FTYP <type> <telecentric> <num_fields> <num_wavelengths> 0 0 0
        num_wl = self._model.wavelengths.get("num_wavelengths", 1)
        lines.append(f"FTYP {ftyp_int} 0 {n} {num_wl} 0 0 0")

    def _encode_wavelengths(self, lines: list[str]) -> None:
        wl_data = self._model.wavelengths
        data = wl_data.get("data", [])
        primary_index = wl_data.get("primary_index", 0)
        for i, w in enumerate(data):
            lines.append(f"WAVM {i + 1} {_fmt(w)} 1")
        lines.append(f"PWAV {primary_index + 1}")

    # ------------------------------------------------------------------
    # Surface blocks
    # ------------------------------------------------------------------

    def _encode_surfaces(self, lines: list[str]) -> None:
        for idx in sorted(self._model.surfaces.keys()):
            raw = self._model.surfaces[idx]
            lines.append(f"SURF {idx}")
            self._encode_surface(lines, raw)

    def _encode_surface(self, lines: list[str], raw: dict[str, Any]) -> None:
        surf_type = raw.get("TYPE", "STANDARD")
        lines.append(f"  TYPE {surf_type}")

        if raw.get("STOP"):
            lines.append("  STOP")

        curv = raw.get("CURV", 0.0)
        lines.append(f"  CURV {_fmt(curv)}")
        lines.append("  HIDE 0")
        lines.append("  MIRR 2 1")
        lines.append("  SLAB 0")

        # Thickness
        disz = raw.get("DISZ", 0.0)
        if disz == "INFINITY" or (isinstance(disz, float) and math.isinf(disz)):
            lines.append("  DISZ INFINITY")
        else:
            lines.append(f"  DISZ {_fmt(float(disz))}")

        # Conic (omit if 0)
        coni = raw.get("CONI", 0.0)
        if coni is not None and abs(float(coni)) > 1e-16:
            lines.append(f"  CONI {_fmt(float(coni))}")

        # Glass
        glas = raw.get("GLAS")
        if glas is not None:
            lines.append(self._encode_glas(glas))

        # Diameter
        diam = raw.get("DIAM")
        if diam is not None:
            lines.append(f"  DIAM {_fmt(float(diam))}")

        # Physical aperture (CLAP)
        clap = raw.get("CLAP")
        if clap is not None:
            try:
                lines.append(
                    f"  CLAP {_fmt(float(clap.r_min))} {_fmt(float(clap.r_max))}"
                )
            except AttributeError:
                lines.append("  CLAP 0")

        # Parameters (PARM) — skip zeros
        for i in range(1, 17):
            key = f"PARM_{i}"
            if key in raw:
                val = float(raw[key])
                if abs(val) > 1e-16:
                    lines.append(f"  PARM {i} {_fmt(val)}")

    def _encode_glas(self, glas: dict[str, Any]) -> str:
        name = glas.get("name", "")
        if name == "MIRROR":
            return "  GLAS MIRROR 1 0 0 0"
        if "n" in glas and "V" in glas:
            # MODEL glass
            return f"  GLAS MODEL 1 0 {_fmt(glas['n'])} {_fmt(glas['V'])}"
        # Catalog glass
        return f"  GLAS {name} 1 0 0 0"
