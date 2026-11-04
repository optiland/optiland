"""CODE V File Encoder

Converts a CodeVDataModel into a list of .seq text lines in the order and
format expected by CODE V. The encoded lines are written to disk as UTF-8.

Kramer Harrison, 2026
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from optiland.fileio.codev.model import CodeVDataModel

# Ordered aspheric coefficient letter keys (4th–20th order)
_ASPH_KEYS = ("A", "B", "C", "D", "E", "F", "G", "H", "J")

# Map from field type string to (x-command, y-command) pair
_FIELD_TYPE_CMDS: dict[str, tuple[str, str]] = {
    "angle": ("XAN", "YAN"),
    "object_height": ("XOB", "YOB"),
    "paraxial_image_height": ("XIM", "YIM"),
    "real_image_height": ("XRI", "YRI"),
}

# Map from CODE V aperture key to command string
_AP_KEY_TO_CMD: dict[str, str] = {
    "EPD": "EPD",
    "FNO": "FNO",
    "NA": "NA",
    "NAO": "NAO",
}


def _fmt(value: float) -> str:
    """Format a float compactly, without losing any digit.

    The shortest ``%g`` form of at least 8 significant digits that reads back
    as exactly ``value`` is written (at most 17), so ``5.0`` is ``5``, ``0.1``
    is ``0.1``, ``1e10`` is ``1e+10`` and ``4910.01016`` keeps all its digits.
    A zero, including ``-0.0``, is written ``0``.

    Args:
        value: The float to format.

    Returns:
        A compact string representation that round-trips through ``float``.

    Raises:
        ValueError: If ``value`` is not finite (CODE V has no ``inf``/``nan``).
    """
    if not math.isfinite(value):
        raise ValueError(f"Cannot write non-finite value ({value!r}) to a CODE V file")
    value = float(value)
    if value == 0.0:
        return "0"
    for digits in range(8, 17):
        text = f"{value:.{digits}g}"
        if float(text) == value:
            return text
    return f"{value:.17g}"


def _um_to_nm(wavelength: float) -> float:
    """Convert a wavelength from micrometres to nanometres.

    The product is rounded to 9 decimals of a nanometre so that binary
    arithmetic noise (``0.4861 * 1000`` is ``486.09999999999997``) does not
    reach the file now that numbers are written at full precision.
    """
    return round(float(wavelength) * 1000.0, 9)


def _title(name: str | None) -> str:
    """Return a title CODE V keeps whole inside ``TITLE '...'``.

    A ``'`` ends the quoted title early, so it becomes ``"``; line breaks would
    end the command, so each becomes a space.
    """
    if not name:
        return ""
    return " ".join(str(name).splitlines()).replace("'", '"').strip()


def _check_numbers(where: str, mapping: dict[str, Any]) -> None:
    """Raise ``ValueError`` naming ``where`` and the key of a non-finite number.

    Lists and tuples of numbers are checked element-wise; other values
    (strings, flags, aperture objects) are ignored.
    """
    for key, value in mapping.items():
        items = value if isinstance(value, list | tuple) else [value]
        for item in items:
            if isinstance(item, bool) or not isinstance(item, int | float):
                continue
            if not math.isfinite(item):
                raise ValueError(f"{where}: non-finite {key} ({item!r})")


def _check_private_glass(model: CodeVDataModel, where: str, name: str) -> None:
    """Raise ``ValueError`` if a private glass has no usable index row."""
    data = model.wavelengths.get("data", [])
    row = model.private_glasses.get(name)
    if row is None or len(row) != len(data):
        raise ValueError(
            f"{where}: private glass '{name}' needs one index per wavelength "
            f"({len(data)}), found {None if row is None else len(row)}"
        )
    for n, i in zip(row, wavelength_write_order(model.wavelengths), strict=True):
        if not math.isfinite(n):
            raise ValueError(
                f"{where}: non-finite index ({n!r}) of private glass '{name}' "
                f"at {data[i] * 1000.0:.8g} nm"
            )


def _check_finite(model: CodeVDataModel) -> None:
    """Raise ``ValueError`` if any number in the model is not finite.

    The threshold filters used while writing (``abs(v) > tol``) are False for
    NaN, so a NaN would be dropped silently; this pass runs first so it is
    reported with its location instead.
    """
    _check_numbers("header", model.aperture)
    _check_numbers("header", model.wavelengths)
    _check_numbers("header", model.fields)
    for idx, raw in model.surfaces.items():
        where = "Object" if raw.get("type") == "object" else f"Surface {idx}"
        _check_numbers(where, raw)
        glass = raw.get("glass")
        if isinstance(glass, dict):
            _check_numbers(where, glass)
            if "private" in glass:
                _check_private_glass(model, where, glass["private"])


def wavelength_write_order(wl_data: dict[str, Any]) -> list[int]:
    """Return the order in which the wavelengths are written to the file.

    CODE V sorts ``WL`` into descending order and then applies ``REF`` and
    ``WTW`` by position in the sorted list, so the writer sorts first and
    permutes ``WTW`` and ``REF`` the same way. The sort is stable, so equal
    wavelengths keep their relative order.

    Args:
        wl_data: ``CodeVDataModel.wavelengths``; ``"data"`` is in um.

    Returns:
        Indices into ``wl_data["data"]``, longest wavelength first. The model
        itself is not reordered.
    """
    data = wl_data.get("data", [])
    return sorted(range(len(data)), key=lambda i: -data[i])


def _fmt_list(values: list[float]) -> str:
    """Format a list of floats separated by spaces.

    Args:
        values: List of floats to format.

    Returns:
        Space-separated string of formatted values.
    """
    return " ".join(_fmt(v) for v in values)


class CodeVFileEncoder:
    """Encodes a CodeVDataModel as a list of .seq text lines.

    Args:
        model: The CodeVDataModel to encode.
    """

    def __init__(self, model: CodeVDataModel):
        self._model = model

    def encode(self) -> list[str]:
        """Produce the complete list of .seq text lines.

        Returns:
            A list of strings, one per line of the output file.
        """
        _check_finite(self._model)
        lines: list[str] = []
        self._encode_header(lines)
        self._encode_surfaces(lines)
        lines.append("GO")
        return lines

    # ------------------------------------------------------------------
    # Header block
    # ------------------------------------------------------------------

    def _encode_header(self, lines: list[str]) -> None:
        lines.append("LEN NEW")
        lines.append("DIM M")
        lines.append("! Generated by Optiland")
        lines.append("RDM Y")
        title = _title(self._model.name)
        if title:
            lines.append(f"TITLE '{title}'")
        self._encode_aperture(lines)
        self._encode_wavelengths(lines)
        self._encode_fields(lines)
        self._encode_private_catalog(lines)
        lines.append("")  # blank separator before surfaces

    def _encode_aperture(self, lines: list[str]) -> None:
        ap = self._model.aperture
        if not ap:
            return
        if ap.get("TEL") == "Y":
            lines.append("TEL Y")
        for key, cmd in _AP_KEY_TO_CMD.items():
            if key in ap:
                lines.append(f"{cmd} {_fmt(ap[key])}")
                return

    def _encode_wavelengths(self, lines: list[str]) -> None:
        wl_data = self._model.wavelengths
        data = wl_data.get("data", [])
        if not data:
            return
        order = wavelength_write_order(wl_data)
        # Convert um -> nm
        lines.append("WL " + _fmt_list([_um_to_nm(data[i]) for i in order]))

        weights = wl_data.get("weights")
        if weights:
            if len(weights) != len(data):
                raise ValueError(
                    f"header: {len(weights)} wavelength weights for "
                    f"{len(data)} wavelengths"
                )
            lines.append("WTW " + _fmt_list([weights[i] for i in order]))
        else:
            lines.append("WTW " + " ".join(["1"] * len(data)))

        primary_index = wl_data.get("primary_index", 0)
        lines.append(f"REF {order.index(primary_index) + 1}")

    def _encode_fields(self, lines: list[str]) -> None:
        fields = self._model.fields
        field_type = fields.get("type", "angle")
        x_vals = fields.get("x", [])
        y_vals = fields.get("y", [])
        if not y_vals:
            return

        if field_type not in _FIELD_TYPE_CMDS:
            raise ValueError(f"header: unknown field type '{field_type}'")
        x_cmd, y_cmd = _FIELD_TYPE_CMDS[field_type]

        if x_vals and any(abs(v) > 1e-12 for v in x_vals):
            lines.append(f"{x_cmd} " + _fmt_list(x_vals))
        lines.append(f"{y_cmd} " + _fmt_list(y_vals))

        weights = fields.get("weights")
        if weights and any(abs(w - 1.0) > 1e-12 for w in weights):
            lines.append("WTF " + _fmt_list(weights))

    def _encode_private_catalog(self, lines: list[str]) -> None:
        """Write the ``PRV`` block of private glasses, if there are any."""
        glasses = self._model.private_glasses
        if not glasses:
            return
        wl_data = self._model.wavelengths
        order = wavelength_write_order(wl_data)
        lines.append("PRV")
        lines.append(
            "  PWL " + _fmt_list([_um_to_nm(wl_data["data"][i]) for i in order])
        )
        for name, indices in glasses.items():
            lines.append(f"  '{name}' " + " ".join(f"{n:.10g}" for n in indices))
        lines.append("END")

    # ------------------------------------------------------------------
    # Surface blocks
    # ------------------------------------------------------------------

    def _encode_surfaces(self, lines: list[str]) -> None:
        for idx in sorted(self._model.surfaces.keys()):
            raw = self._model.surfaces[idx]
            self._encode_surface(lines, raw)

    def _encode_surface(self, lines: list[str], raw: dict[str, Any]) -> None:
        lines.append(self._encode_surface_line(raw))

        if raw.get("is_stop"):
            lines.append("  STO")

        self._encode_profile(lines, raw)
        self._encode_conic(lines, raw)
        self._encode_asphere_coeffs(lines, raw)
        self._encode_decenters_tilts(lines, raw)
        self._encode_physical_aperture(lines, raw)

    def _encode_surface_line(self, raw: dict[str, Any]) -> str:
        surf_type = raw.get("type", "standard")
        r_str = _fmt(float(raw.get("radius", 0.0)))
        t_str = _fmt(float(raw.get("thickness", 0.0)))

        if surf_type == "object":
            return f"SO {r_str} {t_str}"
        if surf_type == "image":
            return f"SI {r_str} {t_str}"

        glass = raw.get("glass")
        glass_str = self._encode_glass_inline(glass) if glass else ""
        line = f"S  {r_str} {t_str}"
        if glass_str:
            line += f"  {glass_str}"
        return line

    @staticmethod
    def _conic(raw: dict[str, Any]) -> float:
        """Return the surface conic constant, or 0.0 if it is below the threshold."""
        conic = raw.get("conic", 0.0)
        if conic is None or not abs(float(conic)) > 1e-16:
            return 0.0
        return float(conic)

    @staticmethod
    def _letters(raw: dict[str, Any]) -> list[tuple[str, float]]:
        """Return the (letter, value) pairs of the non-zero aspheric coefficients."""
        if raw.get("profile", "SPH") != "ASP":
            return []
        pairs = (
            (key, float(raw.get(f"coeff_{i}", 0.0))) for i, key in enumerate(_ASPH_KEYS)
        )
        return [(key, val) for key, val in pairs if val != 0.0]

    def _encode_profile(self, lines: list[str], raw: dict[str, Any]) -> None:
        """Write the profile keyword CODE V needs before ``K`` and the letters.

        ``ASP`` when any aspheric letter is non-zero, else ``CON`` when there is a
        conic constant.
        """
        if self._letters(raw):
            lines.append("  ASP")
        elif self._conic(raw) != 0.0:
            lines.append("  CON")

    def _encode_conic(self, lines: list[str], raw: dict[str, Any]) -> None:
        conic = self._conic(raw)
        if conic != 0.0:
            lines.append(f"  K {_fmt(conic)}")

    def _encode_asphere_coeffs(self, lines: list[str], raw: dict[str, Any]) -> None:
        for key, val in self._letters(raw):
            lines.append(f"  {key} {_fmt(val)}")

    def _encode_decenters_tilts(self, lines: list[str], raw: dict[str, Any]) -> None:
        for cv_key, label in (
            ("xde", "XDE"),
            ("yde", "YDE"),
            ("zde", "ZDE"),
            ("ade", "ADE"),
            ("bde", "BDE"),
            ("cde", "CDE"),
        ):
            val = float(raw.get(cv_key, 0.0))
            if abs(val) > 1e-12:
                lines.append(f"  {label} {_fmt(val)}")

    def _encode_physical_aperture(self, lines: list[str], raw: dict[str, Any]) -> None:
        if "cir" in raw:
            lines.append(f"  CIR CLR {_fmt(raw['cir'])}")
        if "cir_obs" in raw:
            lines.append(f"  CIR OBS {_fmt(raw['cir_obs'])}")

    def _encode_glass_inline(self, glass: dict[str, Any]) -> str:
        """Build the glass string for the surface line.

        Args:
            glass: Glass specification dict from the formatter.

        Returns:
            Glass token string, e.g. ``"NBK7_SCHOTT"``, ``"REFL"`` or the quoted
            private glass name ``"'OPT1'"``.
        """
        if "private" in glass:
            return f"'{glass['private']}'"

        name = glass.get("name", "")

        if name == "REFL":
            return "REFL"

        catalog = glass.get("catalog")
        if catalog:
            return f"{name}_{catalog}"
        return name
