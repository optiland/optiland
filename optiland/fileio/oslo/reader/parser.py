"""OSLO Data Parser

Parses an OSLO .len file into an OsloDataModel.

Kramer Harrison, 2026
"""

from __future__ import annotations

import contextlib
import re
from typing import Any

from optiland.fileio.oslo.model import OsloDataModel


class OsloDataParser:
    """Parses an OSLO .len file into an OsloDataModel.

    Args:
        filename: Path to the .len file to parse.
    """

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.data_model = OsloDataModel()
        self._current_surf_idx = 0
        self._current_surf_data: dict[str, Any] = {}
        self._wavelength_values: list[float] = []
        self._wavelength_weights: list[float] = []

        # Command dispatch table
        self._dispatch_table = {
            "LEN": self._read_len,
            "EBR": self._read_ebr,
            "OBH": self._read_obh,
            "ANG": self._read_ang,
            "UNI": self._read_uni,
            "AIR": self._read_medium,
            "RFL": self._read_medium,
            "GLA": self._read_glass,
            "RD": self._read_rd,
            "TH": self._read_th,
            "AP": self._read_ap,
            "AST": self._read_ast,
            "CC": self._read_cc,
            "AD": self._read_coeff,
            "AE": self._read_coeff,
            "AF": self._read_coeff,
            "AG": self._read_coeff,
            "DCX": self._read_decenter,
            "DCY": self._read_decenter,
            "DCZ": self._read_decenter,
            "TLA": self._read_tilt,
            "TLB": self._read_tilt,
            "TLC": self._read_tilt,
            "WV": self._read_wv,
            "WV2": self._read_wv,
            "WV3": self._read_wv,
            "WW": self._read_ww,
            "NXT": self._read_nxt,
            "END": self._read_end,
            "PY": self._read_solve,
            "PK": self._read_pickup,
            "FNO": self._read_fno,
            "NAO": self._read_nao,
            "DES": self._read_des,
        }

    def parse(self) -> OsloDataModel:
        """Parse the OSLO file.

        Returns:
            A populated OsloDataModel.
        """
        with open(self.filename, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("//"):
                    continue

                # Handle quoted strings correctly
                tokens = self._tokenize(line)
                if not tokens:
                    continue

                cmd = tokens[0].upper()

                # Handle SNO1, SNO2, etc.
                if cmd.startswith("SNO"):
                    self._read_sno(tokens)
                    continue

                if cmd in self._dispatch_table:
                    self._dispatch_table[cmd](tokens)

        # Finalize wavelengths - deduplicate (inline WV before each GLA block
        # causes duplicates; the final WV+WW block defines system wavelengths
        # but shares the same values).
        seen: set[float] = set()
        unique_vals: list[float] = []
        for v in self._wavelength_values:
            if v not in seen:
                seen.add(v)
                unique_vals.append(v)
        weights = list(self._wavelength_weights)
        if weights and len(weights) < len(unique_vals):
            weights = weights + [1.0] * (len(unique_vals) - len(weights))
        self.data_model.wavelengths["values"] = unique_vals
        self.data_model.wavelengths["weights"] = weights[: len(unique_vals)]

        return self.data_model

    def _tokenize(self, line: str) -> list[str]:
        """Tokenize a line, respecting double quotes."""
        # This regex finds either strings in quotes or non-whitespace sequences
        return re.findall(r'"[^"]*"|\S+', line)

    def _read_len(self, tokens: list[str]) -> None:
        # LEN NEW "lens_name" <scaling> <total_surfaces>
        if len(tokens) >= 5 and tokens[1].upper() == "NEW":
            self.data_model.name = tokens[2].strip('"')
            self.data_model.scaling = float(tokens[3])
            self.data_model.num_surfaces = int(tokens[4])

    def _read_ebr(self, tokens: list[str]) -> None:
        # EBR <float> (Entrance Beam Radius)
        self.data_model.aperture["EPD"] = 2.0 * float(tokens[1])

    def _read_fno(self, tokens: list[str]) -> None:
        # FNO <float> (F-Number)
        self.data_model.aperture["FNO"] = float(tokens[1])

    def _read_nao(self, tokens: list[str]) -> None:
        # NAO <float> (Object NA)
        self.data_model.aperture["NAO"] = float(tokens[1])

    def _read_obh(self, tokens: list[str]) -> None:
        # OBH <float> (Object Height)
        self.data_model.fields["type"] = "object_height"
        if "y" not in self.data_model.fields:
            self.data_model.fields["y"] = []
        self.data_model.fields["y"].append(float(tokens[1]))

    def _read_ang(self, tokens: list[str]) -> None:
        # ANG <float> (Field Angle)
        self.data_model.fields["type"] = "angle"
        if "y" not in self.data_model.fields:
            self.data_model.fields["y"] = []
        self.data_model.fields["y"].append(float(tokens[1]))

    def _read_uni(self, tokens: list[str]) -> None:
        self.data_model.units = float(tokens[1])

    def _read_des(self, tokens: list[str]) -> None:
        self.data_model.notes["DES"] = " ".join(tokens[1:]).strip('"')

    def _read_sno(self, tokens: list[str]) -> None:
        cmd = tokens[0].upper()
        content = " ".join(tokens[1:]).strip('"')
        self.data_model.notes[cmd] = content

    def _read_medium(self, tokens: list[str]) -> None:
        # AIR or RFL
        self._current_surf_data["material"] = tokens[0].upper()

    def _read_glass(self, tokens: list[str]) -> None:
        # GLA <glass_def>
        # GLA BK7
        # GLA 1.573 1.573 1.573
        # GLA MOD G1 1.6489 1.662...
        self._current_surf_data["material"] = "GLA " + " ".join(tokens[1:])

    def _read_rd(self, tokens: list[str]) -> None:
        self._current_surf_data["RD"] = float(tokens[1])

    def _read_th(self, tokens: list[str]) -> None:
        self._current_surf_data["TH"] = float(tokens[1])

    def _read_ap(self, tokens: list[str]) -> None:
        self._current_surf_data["AP"] = float(tokens[1])

    def _read_ast(self, tokens: list[str]) -> None:
        self._current_surf_data["AST"] = True

    def _read_cc(self, tokens: list[str]) -> None:
        self._current_surf_data["CC"] = float(tokens[1])

    def _read_coeff(self, tokens: list[str]) -> None:
        cmd = tokens[0].upper()
        self._current_surf_data[cmd] = float(tokens[1])

    def _read_decenter(self, tokens: list[str]) -> None:
        cmd = tokens[0].upper()
        self._current_surf_data[cmd] = float(tokens[1])

    def _read_tilt(self, tokens: list[str]) -> None:
        cmd = tokens[0].upper()
        self._current_surf_data[cmd] = float(tokens[1])

    def _read_wv(self, tokens: list[str]) -> None:
        # WV <w1> [<w2> <w3>]
        # WV2 <w2>
        # WV3 <w3>
        for t in tokens[1:]:
            with contextlib.suppress(ValueError):
                self._wavelength_values.append(float(t))

    def _read_ww(self, tokens: list[str]) -> None:
        # WW <wt1> [<wt2> <wt3>]
        for t in tokens[1:]:
            with contextlib.suppress(ValueError):
                self._wavelength_weights.append(float(t))

    def _read_nxt(self, tokens: list[str]) -> None:
        # Save current surface and increment index
        self.data_model.surfaces[self._current_surf_idx] = self._current_surf_data
        self._current_surf_idx += 1
        self._current_surf_data = {}

    def _read_end(self, tokens: list[str]) -> None:
        if self._current_surf_data:
            self.data_model.surfaces[self._current_surf_idx] = self._current_surf_data
            self._current_surf_idx += 1
            self._current_surf_data = {}

    def _read_solve(self, tokens: list[str]) -> None:
        # TODO: Support more solves
        if tokens[0].upper() == "PY":
            self._current_surf_data["PY"] = float(tokens[1])

    def _read_pickup(self, tokens: list[str]) -> None:
        # TODO: Support standard pickups
        # PK <surf> <cmd>
        self._current_surf_data["PK"] = tokens[1:]
