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
        lines = []

        # Comment header
        lines.append("// OSLO 5.00 0 0 0")

        # System declaration
        lines.append(
            f'LEN NEW "{self.model.name}" '
            f"{self._fmt(self.model.scaling)} {self.model.num_surfaces}"
        )

        # Global properties
        if "EPD" in self.model.aperture:
            lines.append(f"EBR {self._fmt(self.model.aperture['EPD'] / 2.0)}")

        if "FNO" in self.model.aperture:
            lines.append(f"FNO {self._fmt(self.model.aperture['FNO'])}")

        if "NAO" in self.model.aperture:
            lines.append(f"NAO {self._fmt(self.model.aperture['NAO'])}")

        # DES and UNI are always emitted (OSLO EDU convention)
        lines.append('DES "Optiland"')
        lines.append(f"UNI {self._fmt(self.model.units)}")

        if "y" in self.model.fields:
            field_type_cmd = (
                "OBH" if self.model.fields.get("type") == "object_height" else "ANG"
            )
            for y in self.model.fields["y"]:
                lines.append(f"{field_type_cmd} {self._fmt(y)}")

        for cmd, content in self.model.notes.items():
            if cmd == "DES":
                continue  # already emitted explicitly above
            lines.append(f'{cmd} "{content}"')

        # Surfaces
        for idx in sorted(self.model.surfaces.keys()):
            self._format_surface(lines, idx, self.model.surfaces[idx])

        # Wavelengths - written in the footer (after surfaces, before END),
        # matching OSLO .len file convention.
        wl_data = self.model.wavelengths
        if wl_data.get("values"):
            vals = wl_data["values"]
            if len(vals) >= 3:
                lines.append(
                    f"WV {self._fmt(vals[0])} {self._fmt(vals[1])} {self._fmt(vals[2])}"
                )
            elif len(vals) == 2:
                lines.append(f"WV {self._fmt(vals[0])} {self._fmt(vals[1])}")
            else:
                lines.append(f"WV {self._fmt(vals[0])}")

            if "weights" in wl_data and len(wl_data["weights"]) >= len(vals):
                w = wl_data["weights"]
                if len(vals) >= 3:
                    lines.append(
                        f"WW {self._fmt(w[0])} {self._fmt(w[1])} {self._fmt(w[2])}"
                    )
                elif len(vals) == 2:
                    lines.append(f"WW {self._fmt(w[0])} {self._fmt(w[1])}")
                else:
                    lines.append(f"WW {self._fmt(w[0])}")

        # End
        lines.append(f"END {self.model.num_surfaces}")
        lines.append("")

        return "\n".join(lines)

    def _format_surface(
        self, lines: list[str], index: int, data: dict[str, Any]
    ) -> None:
        # Medium
        mat = data.get("material", "AIR")
        lines.append(mat)

        # Geometry
        if "RD" in data and not math.isinf(data["RD"]):
            lines.append(f"RD {self._fmt(data['RD'])}")

        if "TH" in data:
            if math.isinf(data["TH"]):
                lines.append(f"TH {self._fmt(1e10)}")  # OSLO infinity representation
            else:
                lines.append(f"TH {self._fmt(data['TH'])}")

        if "AP" in data:
            lines.append(f"AP {self._fmt(data['AP'])}")

        if data.get("AST"):
            lines.append("AST")

        if "CC" in data and data["CC"] != 0:
            lines.append(f"CC {self._fmt(data['CC'])}")

        # Aspherics
        if "AD" in data:
            lines.append(f"AD {self._fmt(data['AD'])}")
        if "AE" in data:
            lines.append(f"AE {self._fmt(data['AE'])}")
        if "AF" in data:
            lines.append(f"AF {self._fmt(data['AF'])}")
        if "AG" in data:
            lines.append(f"AG {self._fmt(data['AG'])}")

        # Decenter/Tilt
        for k in ["DCX", "DCY", "DCZ", "TLA", "TLB", "TLC"]:
            if k in data and data[k] != 0:
                lines.append(f"{k} {self._fmt(data[k])}")

        # Solve/Pickup (optional/partial)
        if "PY" in data:
            lines.append(f"PY {self._fmt(data['PY'])}")

        if "PK" in data:
            lines.append(f"PK {' '.join(str(it) for it in data['PK'])}")

        if index < self.model.num_surfaces:
            lines.append("NXT")

    def _fmt(self, val: float) -> str:
        """Format float for OSLO using compact decimal notation (≤7 sig figs)."""
        if math.isinf(val):
            return "1.0e+10"
        return f"{val:.7g}"
