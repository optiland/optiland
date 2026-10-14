"""Zemax File Encoder

Converts a ZemaxDataModel into a list of .zmx text lines in the order and
format expected by OpticStudio. The encoded lines are written to disk as
UTF-16 LE by save_zemax_file().

Kramer Harrison, 2024
"""

from __future__ import annotations

import math
import operator
from typing import TYPE_CHECKING, Any

from optiland.physical_apertures import (
    DifferenceAperture,
    EllipticalAperture,
    OffsetRadialAperture,
    RadialAperture,
    RectangularAperture,
    UnionAperture,
)

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
    """Format a float in Zemax scientific notation.

    Uses 17 significant digits, the IEEE-754 binary64 round-trip guarantee, so
    that ``save_zemax_file`` followed by ``load_zemax_file`` reproduces the
    original system bit-for-bit. The previous ``%.8E`` (9 significant digits)
    was lossy enough to matter: on ``HubbleTelescope`` (EFL ~5.76e4) a single
    save/load shifted real-ray image-surface intercepts by 1.8e-06 mm, which
    exceeds the 1e-06 mm agreement threshold used for cross-tool validation.
    """
    return f"{value:.16E}"


def _fmt_vals(values: list[float]) -> str:
    """Format a list of floats separated by spaces."""
    return " ".join(_fmt(v) for v in values)


def _offset(aperture: Any) -> tuple[float, float]:
    if isinstance(aperture, OffsetRadialAperture):
        return float(aperture.offset_x), float(aperture.offset_y)
    return 0.0, 0.0


def _encode_offset(lines: list[str], offset: tuple[float, float]) -> None:
    if offset != (0.0, 0.0):
        lines.append(f"  OBDC {_fmt(offset[0])} {_fmt(offset[1])}")


def _obscuration(aperture: Any) -> tuple[float, float, tuple[float, float]] | None:
    """The (r_min, r_max, offset) of an aperture the reader builds from OBSC.

    That is a radial aperture with no outer edge (passing r >= r_max), or the
    union of a disk of radius r_min with one, sharing the same centre.
    """

    def _outside(ap: Any) -> bool:
        return isinstance(ap, RadialAperture) and math.isinf(float(ap.r_max))

    if _outside(aperture):
        return 0.0, float(aperture.r_min), _offset(aperture)
    if isinstance(aperture, UnionAperture):
        inner, outer = aperture.a, aperture.b
        if (
            _outside(outer)
            and isinstance(inner, RadialAperture)
            and float(inner.r_min) == 0.0
            and math.isfinite(float(inner.r_max))
            and _offset(inner) == _offset(outer)
        ):
            return float(inner.r_max), float(outer.r_min), _offset(outer)
    return None


def _shape(aperture: Any) -> tuple[str, float, float, tuple[float, float]] | None:
    """The (keyword, x_half, y_half, offset) of a rectangle or ellipse.

    SQAP and ELAP pass rays inside the shape; SQOB and ELOB, which the reader
    builds as an unbounded aperture minus the shape, block them.
    """
    if type(aperture) is RectangularAperture:
        x_min, x_max = float(aperture.x_min), float(aperture.x_max)
        y_min, y_max = float(aperture.y_min), float(aperture.y_max)
        offset = ((x_min + x_max) / 2, (y_min + y_max) / 2)
        return "SQAP", (x_max - x_min) / 2, (y_max - y_min) / 2, offset
    if type(aperture) is EllipticalAperture:
        offset = (float(aperture.offset_x), float(aperture.offset_y))
        return "ELAP", float(aperture.a), float(aperture.b), offset
    if type(aperture) is DifferenceAperture:
        unbounded, hole = aperture.a, aperture.b
        if (
            type(unbounded) is RadialAperture
            and float(unbounded.r_min) == 0.0
            and math.isinf(float(unbounded.r_max))
        ):
            shape = _shape(hole)
            if shape is not None:
                keyword, x_half, y_half, offset = shape
                return keyword[:2] + "OB", x_half, y_half, offset
    return None


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

        Raises:
            ValueError: If nonempty wavelength data has an explicitly invalid
                primary index, including None, a non-integer, or an out-of-range value.
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
        num_wl = self._model.wavelengths.get(
            "num_wavelengths", len(self._model.wavelengths.get("data", []))
        )
        lines.append(f"FTYP {ftyp_int} 0 {n} {num_wl} 0 0 0")

    def _encode_wavelengths(self, lines: list[str]) -> None:
        wl_data = self._model.wavelengths
        data = wl_data.get("data", [])
        if len(data) == 0:
            return
        try:
            primary_index = operator.index(wl_data.get("primary_index", 0))
        except TypeError as exc:
            raise ValueError("Primary wavelength index must be an integer.") from exc
        if not 0 <= primary_index < len(data):
            raise ValueError("Primary wavelength is not present in the active data.")
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

        self._encode_thickness(lines, raw)
        self._encode_conic(lines, raw)
        self._encode_glass_line(lines, raw)
        self._encode_diameter(lines, raw)
        self._encode_physical_aperture(lines, raw)
        self._encode_parameters(lines, raw)

    def _encode_thickness(self, lines: list[str], raw: dict[str, Any]) -> None:
        disz = raw.get("DISZ", 0.0)
        if disz == "INFINITY" or (isinstance(disz, float) and math.isinf(disz)):
            lines.append("  DISZ INFINITY")
        else:
            lines.append(f"  DISZ {_fmt(float(disz))}")

    def _encode_conic(self, lines: list[str], raw: dict[str, Any]) -> None:
        coni = raw.get("CONI", 0.0)
        if coni is not None and abs(float(coni)) > 1e-16:
            lines.append(f"  CONI {_fmt(float(coni))}")

    def _encode_glass_line(self, lines: list[str], raw: dict[str, Any]) -> None:
        glas = raw.get("GLAS")
        if glas is not None:
            lines.append(self._encode_glas(glas))

    def _encode_diameter(self, lines: list[str], raw: dict[str, Any]) -> None:
        diam = raw.get("DIAM")
        if diam is not None:
            lines.append(f"  DIAM {_fmt(float(diam))}")

    def _encode_physical_aperture(self, lines: list[str], raw: dict[str, Any]) -> None:
        clap = raw.get("CLAP")
        if clap is None:
            return
        obscuration = _obscuration(clap)
        shape = _shape(clap)
        if shape is not None:
            keyword, x_half, y_half, offset = shape
            lines.append(f"  {keyword} {_fmt(x_half)} {_fmt(y_half)}")
            _encode_offset(lines, offset)
        elif obscuration is not None:
            r_min, r_max, offset = obscuration
            lines.append(f"  OBSC {_fmt(r_min)} {_fmt(r_max)}")
            _encode_offset(lines, offset)
        elif hasattr(clap, "r_min"):
            lines.append(f"  CLAP {_fmt(float(clap.r_min))} {_fmt(float(clap.r_max))}")
            _encode_offset(lines, _offset(clap))

    def _encode_parameters(self, lines: list[str], raw: dict[str, Any]) -> None:
        for i in range(1, 17):
            key = f"PARM_{i}"
            if key in raw:
                val = float(raw[key])
                if abs(val) > 1e-16:
                    lines.append(f"  PARM {i} {_fmt(val)}")

    def _encode_glas(self, glas: dict[str, Any]) -> str:
        name = glas.get("name", "")
        if name == "MIRROR":
            return "  GLAS MIRROR 0 0 0 0 0 0 0 0 0 0"
        if name == "MODEL" and "n" in glas and "V" in glas:
            # MODEL glass
            return f"  GLAS MODEL 1 0 {_fmt(glas['n'])} {_fmt(glas['V'])} 0 0 0 0 0 0"
        if "n" in glas and "V" in glas:
            # Catalog glass: record Nd/Vd so an ambiguous name (present in
            # multiple GCAT catalogs) can be disambiguated on reload.
            return f"  GLAS {name} 0 0 {_fmt(glas['n'])} {_fmt(glas['V'])} 0 0 0 0 0 0"
        # Catalog glass without index data (e.g. round-tripped from another format)
        return f"  GLAS {name} 0 0 0 0 0 0 0 0 0 0"
