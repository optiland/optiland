"""OSLO Data Formatter

Formats an OsloDataModel into a .len file string.

Kramer Harrison, 2026
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from optiland.fileio.oslo.model import OsloDataModel


class OsloDataFormatter:
    """Formats an OsloDataModel into a .len file string.

    Args:
        model: The OsloDataModel to format.
    """

    def __init__(self, model: OsloDataModel):
        self.model = model

    def format(self) -> str:
        """Format the model.

        Returns:
            The complete OSLO .len file as a string.
        """
        lines: list[str] = []
        lines.append("// OSLO 5.00 0 0 0")
        lines.append(
            f'LEN NEW "{self.model.name}" '
            f"{self._fmt(self.model.scaling)} {self.model.num_surfaces}"
        )

        self._format_global_aperture(lines)

        # DES and UNI are always emitted (OSLO EDU convention)
        lines.append('DES "Optiland"')
        lines.append(f"UNI {self._fmt(self.model.units)}")

        self._format_field_commands(lines)
        self._format_notes(lines)

        for idx in sorted(self.model.surfaces.keys()):
            self._format_surface(lines, idx, self.model.surfaces[idx])

        self._format_wavelength_footer(lines)

        lines.append(f"END {self.model.num_surfaces}")
        lines.append("")

        return "\n".join(lines)

    def _format_global_aperture(self, lines: list[str]) -> None:
        if "EPD" in self.model.aperture:
            lines.append(f"EBR {self._fmt(self.model.aperture['EPD'] / 2.0)}")
        if "FNO" in self.model.aperture:
            lines.append(f"FNO {self._fmt(self.model.aperture['FNO'])}")
        if "NAO" in self.model.aperture:
            lines.append(f"NAO {self._fmt(self.model.aperture['NAO'])}")

    def _format_field_commands(self, lines: list[str]) -> None:
        if "y" not in self.model.fields:
            return
        field_type_cmd = (
            "OBH" if self.model.fields.get("type") == "object_height" else "ANG"
        )
        for y in self.model.fields["y"]:
            lines.append(f"{field_type_cmd} {self._fmt(y)}")

    def _format_notes(self, lines: list[str]) -> None:
        for cmd, content in self.model.notes.items():
            if cmd == "DES":
                continue  # already emitted explicitly above
            lines.append(f'{cmd} "{content}"')

    def _format_wavelength_footer(self, lines: list[str]) -> None:
        """Emit WV/WW footer lines (after surfaces, before END) per convention."""
        wl_data = self.model.wavelengths
        vals = wl_data.get("values")
        if not vals:
            return

        lines.append(f"WV {self._fmt_vals(vals)}")

        weights = wl_data.get("weights")
        if weights and len(weights) >= len(vals):
            lines.append(f"WW {self._fmt_vals(weights[: len(vals)])}")

    def _fmt_vals(self, values: list[float]) -> str:
        """Format up to the first 3 values, matching OSLO's WV/WW convention."""
        return " ".join(self._fmt(v) for v in values[:3])

    def _format_surface(
        self, lines: list[str], index: int, data: dict[str, Any]
    ) -> None:
        lines.append(data.get("material", "AIR"))

        self._format_surface_geometry(lines, data)
        self._format_surface_aspherics(lines, data)
        self._format_surface_decenter_tilt(lines, data)
        self._format_surface_solve_pickup(lines, data)

        if index < self.model.num_surfaces:
            lines.append("NXT")

    def _format_surface_geometry(self, lines: list[str], data: dict[str, Any]) -> None:
        if "RD" in data and not math.isinf(data["RD"]):
            lines.append(f"RD {self._fmt(data['RD'])}")

        if "TH" in data:
            th = 1e10 if math.isinf(data["TH"]) else data["TH"]
            lines.append(f"TH {self._fmt(th)}")  # 1e10 is OSLO's infinity convention

        if "AP" in data:
            lines.append(f"AP {self._fmt(data['AP'])}")

        if data.get("AST"):
            lines.append("AST")

        if "CC" in data and data["CC"] != 0:
            lines.append(f"CC {self._fmt(data['CC'])}")

    def _format_surface_aspherics(self, lines: list[str], data: dict[str, Any]) -> None:
        for key in ("AD", "AE", "AF", "AG"):
            if key in data:
                lines.append(f"{key} {self._fmt(data[key])}")

    def _format_surface_decenter_tilt(
        self, lines: list[str], data: dict[str, Any]
    ) -> None:
        for key in ("DCX", "DCY", "DCZ", "TLA", "TLB", "TLC"):
            if key in data and data[key] != 0:
                lines.append(f"{key} {self._fmt(data[key])}")

    def _format_surface_solve_pickup(
        self, lines: list[str], data: dict[str, Any]
    ) -> None:
        if "PY" in data:
            lines.append(f"PY {self._fmt(data['PY'])}")
        if "PK" in data:
            lines.append(f"PK {' '.join(str(it) for it in data['PK'])}")

    def _fmt(self, val: float) -> str:
        """Format float for OSLO using compact decimal notation (≤7 sig figs)."""
        if math.isinf(val):
            return "1.0e+10"
        return f"{val:.7g}"
