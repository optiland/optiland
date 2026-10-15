"""OSLO Data Parser

Parses an OSLO .len file into an OsloDataModel.

Kramer Harrison, 2026
"""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import Any

from optiland.fileio.oslo.constants import DEFAULT_WAVELENGTHS_UM
from optiland.fileio.oslo.model import OsloDataModel, OsloDiagnostic
from optiland.fileio.oslo.reader.configurations import (
    configuration_spectrum,
    read_configuration_record,
)
from optiland.fileio.oslo.syntax import decode_text, tokenize

# Pickup variants constrain the same underlying surface property. Use one
# definition for validation, replacement by literal values/solves, and deletion.
_PICKUP_FAMILIES = {
    "CV": "CV",
    "CVM": "CV",
    "TH": "TH",
    "THM": "TH",
    "LN": "TH",
    "LNM": "TH",
    "AP": "AP",
    "GLA": "GLA",
    "TD": "TD",
    "TDM": "TD",
}


class OsloDataParser:
    """Parses an OSLO .len file into an OsloDataModel.

    Args:
        filename: Path to the .len file to parse.
    """

    def __init__(self, filename: str, *, strict: bool = False) -> None:
        self.filename = filename
        self.strict = strict
        self.data_model = OsloDataModel()
        self._current_surf_idx = 0
        self._current_surf_data: dict[str, Any] = {}
        self._coefficient_lines: dict[tuple[int, str], int] = {}
        self._wavelength_values = list(DEFAULT_WAVELENGTHS_UM)
        self._wavelength_weights: list[float] = []
        self._line = 0
        self._ended = False
        self._field_table = False
        self._ignore_footer = False
        self._seen_len = False
        self._configuration_table = False
        self._seen_cfg = False

        # Command dispatch table
        self._dispatch_table = {
            "LEN": self._read_len,
            "EBR": self._read_system_aperture,
            "OBH": self._read_field,
            "ANG": self._read_field,
            "GIH": self._read_field,
            "UNI": self._read_uni,
            "AIR": self._read_medium,
            "RFL": self._read_medium,
            "RFH": self._read_medium,
            "AIF": self._read_medium,
            "GLA": self._read_glass,
            "GLF": self._read_glass,
            "RD": self._read_rd,
            "RDF": self._read_rd,
            "CV": self._read_cv,
            "CVF": self._read_cv,
            "CVX": self._read_coeff,
            "RDX": self._read_rdx,
            "TH": self._read_th,
            "THF": self._read_th,
            "AP": self._read_ap,
            "APF": self._read_ap,
            "APN": self._read_apn,
            "APCK": self._read_apck,
            "APK": self._read_aperture_pickup,
            "ATP": self._read_special_aperture,
            "AAC": self._read_special_aperture,
            "AAN": self._read_special_aperture,
            "AGN": self._read_special_aperture,
            "AX1": self._read_special_aperture,
            "AX2": self._read_special_aperture,
            "AY1": self._read_special_aperture,
            "AY2": self._read_special_aperture,
            "ASP": self._read_asp,
            "AST": self._read_ast,
            "CC": self._read_coeff,
            "AD": self._read_coeff,
            "AE": self._read_coeff,
            "AF": self._read_coeff,
            "AG": self._read_coeff,
            "DCX": self._read_coeff,
            "DCY": self._read_coeff,
            "DCZ": self._read_coeff,
            "DT": self._read_coeff,
            "GC": self._read_coeff,
            "RCO": self._read_return,
            "BEN": self._read_flag,
            "TOX": self._read_coeff,
            "TOY": self._read_coeff,
            "TOZ": self._read_coeff,
            "TLA": self._read_coeff,
            "TLB": self._read_coeff,
            "TLC": self._read_coeff,
            "WV": self._read_spectrum,
            "WW": self._read_spectrum,
            "NXT": self._read_nxt,
            "GTO": self._read_gto,
            "END": self._read_end,
            "PY": self._read_solve,
            "PYC": self._read_solve,
            "PU": self._read_solve,
            "PUC": self._read_solve,
            "EC": self._read_solve,
            "PK": self._read_pickup,
            "FNO": self._read_system_aperture,
            "NAO": self._read_system_aperture,
            "NAP": self._read_system_aperture,
            "PUK": self._read_system_aperture,
            "TELE": self._read_tele,
            "DES": self._read_sno,
            "PFL": self._read_paraxial,
            "PFM": self._read_coeff,
            "GSP": self._read_coeff,
            "GOR": self._read_coeff,
            "TCE": self._read_coeff,
            "NOT": self._read_note,
            "LMO": self._read_group,
        }
        for cmd in (
            "ATD",
            "CXD",
            "APD",
            "GCD",
            "RCD",
            "BED",
            "PFD",
            "TDD",
            "CSD",
            "TSD",
        ):
            self._dispatch_table[cmd] = self._read_delete
        for entry in "ABCD":
            for kind in "AP":
                self._dispatch_table[f"J{kind}{entry}"] = self._read_jones

    def parse(self) -> OsloDataModel:
        """Parse the OSLO file.

        Returns:
            A populated OsloDataModel.
        """
        self.__init__(self.filename, strict=self.strict)
        raw = Path(self.filename).read_bytes()
        try:
            source = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            source = raw.decode("cp1252")
        for self._line, line in enumerate(source.splitlines(), 1):
            try:
                for statement in self._statements(line):
                    tokens = tokenize(statement)
                    if not tokens:
                        continue
                    cmd = tokens[0].upper()
                    tokens[0] = cmd
                    if self._ended:
                        self._read_footer(tokens)
                    elif re.fullmatch(r"SNO\d+", cmd):
                        self._read_sno(tokens)
                    elif re.fullmatch(r"W[VW][1-9]\d*", cmd):
                        self._validate_numbers(tokens)
                        self._read_spectrum(tokens)
                    elif re.fullmatch(r"AS\d+", cmd):
                        self._validate_numbers(tokens)
                        self._read_coeff(tokens)
                    elif re.fullmatch(r"AV[XY][1-4]", cmd):
                        self._validate_numbers(tokens)
                        self._read_special_aperture(tokens)
                    elif cmd in self._dispatch_table:
                        self._validate_numbers(tokens)
                        self._validate_command(tokens)
                        self._dispatch_table[cmd](tokens)
                    elif cmd in {
                        "DRW",
                        "LDP",
                        "CBK",
                        "ELMDF1",
                        "ELMDF2",
                        "BDI",
                        "BDD",
                        "VX",
                        "PF",
                        "LMN",
                        "LME",
                    }:
                        continue  # Drawing-only data, without optical effects.
                    else:
                        self._unsupported(cmd)
            except (ValueError, IndexError) as exc:
                raise ValueError(f"{self.filename}:{self._line}: {exc}") from exc
        if self._configuration_table:
            raise ValueError(f"{self.filename}: unterminated CFG table")
        if not self._ended:
            self._read_end(["END"])
        if not self._seen_len:
            raise ValueError(f"{self.filename}: missing LEN NEW prescription")
        if set(self.data_model.surfaces) != set(
            range(self.data_model.num_surfaces + 1)
        ):
            raise ValueError(f"{self.filename}: surface records do not match LEN count")
        for index, data in self.data_model.surfaces.items():
            coefficient = next((k for k in data if re.fullmatch(r"AS\d+", k)), None)
            if data.get("ASP", "ADO") == "ADO" and coefficient is not None:
                self._current_surf_idx = index
                self._unsupported(
                    "ASn",
                    "general coefficients require ASP ASR/ARA/ASX",
                    line=self._coefficient_lines[index, coefficient],
                )

        values = self._wavelength_values
        weights = self._wavelength_weights + [1.0] * len(values)
        self.data_model.wavelengths["values"] = values
        self.data_model.wavelengths["weights"] = weights[: len(values)]
        if not any(self.data_model.wavelengths["weights"]):
            raise ValueError(f"{self.filename}: wavelength weights cannot all be zero")
        if weights[0] == 0:
            raise ValueError(
                f"{self.filename}: OSLO primary wavelength weight must be positive"
            )

        for configuration in self.data_model.configurations.values():
            configuration_spectrum(self.data_model, configuration)
        return self.data_model

    def _unsupported(
        self,
        command: str,
        message: str = "unsupported command",
        *,
        line: int | None = None,
    ) -> None:
        line = self._line if line is None else line
        diagnostic = OsloDiagnostic(command, line, self._current_surf_idx, message)
        self.data_model.diagnostics.append(diagnostic)
        detail = (
            f"{self.filename}:{line}: OSLO {command} at surface "
            f"{self._current_surf_idx}: {message}; import may be incomplete"
        )
        if self.strict:
            raise ValueError(detail)
        warnings.warn(detail, UserWarning, stacklevel=3)

    @staticmethod
    def _validate_numbers(tokens: list[str]) -> None:
        for token in tokens[1:]:
            try:
                value = float(token)
            except ValueError:
                continue
            if not math.isfinite(value):
                raise ValueError(f"{tokens[0]} requires finite numeric input")

    def _validate_command(self, tokens: list[str]) -> None:
        cmd = tokens[0]
        scalar = {
            "EBR",
            "FNO",
            "NAO",
            "NAP",
            "PUK",
            "UNI",
            "ANG",
            "OBH",
            "GIH",
            "RD",
            "RDF",
            "CV",
            "CVF",
            "TH",
            "THF",
            "CC",
            "CVX",
            "RDX",
            "AD",
            "AE",
            "AF",
            "AG",
            "DCX",
            "DCY",
            "DCZ",
            "TLA",
            "TLB",
            "TLC",
            "DT",
            "GC",
            "TOX",
            "TOY",
            "TOZ",
            "APN",
            "PFL",
            "PFM",
            "GSP",
            "GOR",
            "TCE",
            "PY",
            "PYC",
            "PU",
            "PUC",
            "EC",
            "GTO",
            "TELE",
            "APCK",
        }
        if cmd in scalar and len(tokens) != 2:
            raise ValueError(f"{cmd} requires exactly one argument")
        if (
            cmd
            in {
                "AIR",
                "AIF",
                "RFL",
                "RFH",
                "AST",
                "BEN",
                "NXT",
                "ATD",
                "CXD",
                "APD",
                "GCD",
                "RCD",
                "BED",
                "PFD",
                "TDD",
                "CSD",
                "TSD",
            }
            and len(tokens) != 1
        ):
            raise ValueError(f"{cmd} does not take arguments")
        if cmd == "RCO" and len(tokens) not in {1, 2}:
            raise ValueError("RCO takes at most one surface reference")
        if cmd == "ASP":
            if len(tokens) not in {2, 3}:
                raise ValueError("ASP expects a type and optional coefficient count")
            if len(tokens) == 3 and not 0 <= int(tokens[2]) <= 256:
                raise ValueError("ASP coefficient count must be between 0 and 256")
        if cmd in {"EBR", "FNO", "NAO", "NAP"} and float(tokens[1]) <= 0:
            raise ValueError(f"{cmd} must be positive")
        if cmd == "DT" and float(tokens[1]) not in {-1, 1}:
            raise ValueError("DT must be -1 or 1")
        if cmd in {"GC", "GOR"} and not float(tokens[1]).is_integer():
            raise ValueError(f"{cmd} requires an integer")
        if cmd in {"GLA", "GLF"} and (
            len(tokens) < 2 or (tokens[1].upper() == "MOD" and len(tokens) < 3)
        ):
            raise ValueError("GLA requires a glass name or index data")

    @staticmethod
    def _statements(line: str) -> list[str]:
        """Split commands and comments outside quoted, backslash-escaped text."""
        statements = []
        start = 0
        quoted = escaped = False
        for i, char in enumerate(line):
            if escaped:
                escaped = False
                continue
            if char == "\\" and quoted:
                escaped = True
            elif char == '"':
                quoted = not quoted
            elif not quoted:
                if line[i : i + 2] == "//":
                    line = line[:i]
                    break
                if char == ";":
                    statements.append(line[start:i])
                    start = i + 1
        if quoted:
            raise ValueError("unterminated quoted string")
        statements.append(line[start:])
        return statements

    def _read_len(self, tokens: list[str]) -> None:
        # LEN NEW "lens_name" <scaling> <total_surfaces>
        if self._seen_len or len(tokens) != 5 or tokens[1].upper() != "NEW":
            raise ValueError('LEN expects NEW "name" scaling surface-count')
        self._seen_len = True
        self.data_model.name = decode_text(tokens[2])
        self.data_model.scaling = float(tokens[3])
        self.data_model.num_surfaces = int(tokens[4])
        if not 1 <= self.data_model.num_surfaces <= 10000:
            raise ValueError("LEN surface count must be between 1 and 10000")

    def _read_system_aperture(self, tokens: list[str]) -> None:
        """The latest aperture specification replaces the previous one."""
        command, value = tokens[0], float(tokens[1])
        if command == "EBR":
            # The shared model represents OSLO's beam radius as a diameter.
            command, value = "EPD", 2 * value
        elif command == "PUK":
            value = abs(value)
        self.data_model.aperture = {command: value}

    def _read_field(self, tokens: list[str]) -> None:
        """Read ANG half-angle, OBH object height or GIH Gaussian image height."""
        self.data_model.fields = {
            "type": {
                "ANG": "angle",
                "OBH": "object_height",
                "GIH": "gaussian_image_height",
            }[tokens[0]],
            "y": [float(tokens[1])],
        }

    def _read_footer(self, tokens: list[str]) -> None:
        """Read declarative field data without executing analysis or CCL blocks."""
        cmd = tokens[0]
        if self._ignore_footer:
            return
        if self._configuration_table:
            if cmd == "CFG":
                raise ValueError("nested CFG table")
            if cmd == "END":
                if len(tokens) != 1:
                    raise ValueError("configuration END takes no arguments")
                self._configuration_table = False
            elif not read_configuration_record(self.data_model, tokens):
                self._unsupported(cmd, "configuration override is not mapped")
            return
        if cmd == "CFG":
            if [t.upper() for t in tokens] != ["CFG", "NEW"]:
                self._ignore_footer = True
                self._unsupported(cmd, "only declarative CFG NEW is imported")
                return
            if self._seen_cfg:
                raise ValueError("multiple CFG tables are not supported")
            self._seen_cfg = self._configuration_table = True
            self._field_table = False
        elif cmd in {"CFWT", "CFAC"}:
            read_configuration_record(self.data_model, tokens)
        elif cmd == "LEN":
            self._ignore_footer = True
            self._unsupported(cmd, "additional configurations are not imported")
        elif cmd == "RST":
            self._field_table = len(tokens) == 2 and tokens[1].upper() == "NEW"
            if self._field_table:
                self.data_model.fields["points"] = {}
        elif cmd == "END":
            self._field_table = False
        elif cmd == "F" and self._field_table:
            self._validate_numbers(tokens)
            if len(tokens) not in {12, 14}:
                raise ValueError("F requires an index and ten field-table values")
            index = int(tokens[1])
            values = [float(v) for v in tokens[2:]]
            if index < 1 or values[9] < 0:
                raise ValueError("F requires a positive index and nonnegative weight")
            if any(values[2:5]) or any(values[10:]):
                self._unsupported(
                    "F",
                    "field depth, reference-ray aiming or extended flags "
                    "are not mapped",
                )
            ymin, ymax, xmin, xmax = values[5:9]
            if not (ymin < ymax and xmin < xmax):
                raise ValueError("F pupil bounds must be increasing")
            if ymin != -ymax or xmin != -xmax:
                self._unsupported("F", "asymmetric field pupil bounds are not mapped")
                ymin, ymax, xmin, xmax = -1, 1, -1, 1
            self.data_model.fields["points"][index] = {
                "y": values[0],
                "x": values[1],
                "weight": values[9],
                "vy": 1 - ymax,
                "vx": 1 - xmax,
            }

    def _read_tele(self, tokens: list[str]) -> None:
        if tokens[1].upper() not in {"ON", "OFF", "0", "1"}:
            raise ValueError("TELE expects ON or OFF")
        self.data_model.settings["telecentric"] = tokens[1].upper() in {"ON", "1"}

    def _read_uni(self, tokens: list[str]) -> None:
        self.data_model.units = float(tokens[1])
        if self.data_model.units <= 0:
            raise ValueError("UNI must be positive (millimeters per lens unit)")

    def _read_sno(self, tokens: list[str]) -> None:
        cmd = tokens[0].upper()
        content = decode_text(" ".join(tokens[1:]))
        self.data_model.notes[cmd] = content

    def _read_medium(self, tokens: list[str]) -> None:
        # AIR or RFL
        cmd = tokens[0].upper()
        self._clear_constraint("GLA")
        self._current_surf_data["material"] = {"AIF": "AIR", "RFH": "RFL"}.get(cmd, cmd)

    def _read_glass(self, tokens: list[str]) -> None:
        # GLA <glass_def>
        # GLA BK7
        # GLA 1.573 1.573 1.573
        # GLA MOD G1 1.6489 1.662...
        self._clear_constraint("GLA")
        self._current_surf_data["material"] = "GLA " + " ".join(tokens[1:])
        self._current_surf_data["glass_wavelengths"] = list(self._wavelength_values)

    def _read_paraxial(self, tokens: list[str]) -> None:
        self._current_surf_data["PFL"] = float(tokens[1])

    def _read_note(self, tokens: list[str]) -> None:
        self._current_surf_data["note"] = decode_text(" ".join(tokens[1:]))

    def _read_group(self, tokens: list[str]) -> None:
        if tokens[1].upper() not in {"EGR", "ELE"}:
            self._unsupported("LMO", "non-sequential groups are not mapped")

    def _read_delete(self, tokens: list[str]) -> None:
        data = self._current_surf_data
        keys = {
            "ATD": [
                k
                for k in data
                if k in {"CC", "AD", "AE", "AF", "AG", "ASP"}
                or re.fullmatch(r"AS\d+", k)
            ],
            "CXD": ["CVX"],
            "APD": ["APN", "special_apertures", "aperture_pickups"],
            "GCD": ["GC"],
            "RCD": ["RCO"],
            "BED": ["BEN"],
            "PFD": ["PFL", "PFM"],
            "TDD": [
                "DCX",
                "DCY",
                "DCZ",
                "TLA",
                "TLB",
                "TLC",
                "DT",
                "TOX",
                "TOY",
                "TOZ",
                "GC",
                "RCO",
                "BEN",
            ],
            "CSD": [],  # Solves and pickups are cleared together below.
            "TSD": [],
        }[tokens[0]]
        for key in keys:
            data.pop(key, None)
        family = {"CSD": "CV", "TSD": "TH", "TDD": "TD"}.get(tokens[0])
        if family:
            self._clear_constraint(family)

    def _read_rd(self, tokens: list[str]) -> None:
        self._clear_constraint("CV")
        self._current_surf_data["RD"] = float(tokens[1]) or math.inf

    def _read_cv(self, tokens: list[str]) -> None:
        self._clear_constraint("CV")
        curvature = float(tokens[1])
        self._current_surf_data["RD"] = 1 / curvature if curvature else math.inf

    def _read_rdx(self, tokens: list[str]) -> None:
        radius = float(tokens[1])
        self._current_surf_data["CVX"] = 1 / radius if radius else 0.0

    def _read_asp(self, tokens: list[str]) -> None:
        kind = {"0": "ADO", "1": "ASR", "2": "ASX"}.get(tokens[1], tokens[1].upper())
        self._current_surf_data["ASP"] = kind
        if kind not in {"ADO", "ASR", "ASX", "ARA"}:
            self._unsupported("ASP", f"asphere type {kind} is not mapped")

    def _read_th(self, tokens: list[str]) -> None:
        self._clear_constraint("TH")
        self._current_surf_data["TH"] = float(tokens[1])

    def _read_ap(self, tokens: list[str]) -> None:
        self._clear_constraint("AP")
        index = 2 if tokens[1].upper() in {"CHK", "UNC"} else 1
        if len(tokens) != index + 1:
            raise ValueError("AP expects an optional CHK/UNC flag and one radius")
        value = float(tokens[index])
        if value < 0:
            raise ValueError("AP radius must be nonnegative")
        self._current_surf_data["AP"] = value
        self._current_surf_data["aperture_checked"] = tokens[1].upper() == "CHK"

    def _read_apck(self, tokens: list[str]) -> None:
        flag = tokens[1].upper()
        if flag not in {"ON", "OFF", "1", "0"}:
            raise ValueError("APCK expects ON or OFF")
        self.data_model.settings["aperture_check"] = flag in {"ON", "1"}

    def _read_apn(self, tokens: list[str]) -> None:
        count = int(tokens[1])
        if not 0 <= count <= 256:
            raise ValueError("APN count must be between 0 and 256")
        self._current_surf_data["APN"] = count
        self._current_surf_data["special_apertures"] = {}
        self._current_surf_data.pop("aperture_pickups", None)

    def _read_special_aperture(self, tokens: list[str]) -> None:
        if len(tokens) != 3:
            raise ValueError(f"{tokens[0]} expects an aperture identifier and value")
        cmd, identifier = tokens[:2]
        identifier = identifier.upper()
        value = float(tokens[2])
        if cmd in {"ATP", "AAC", "AGN"} and not value.is_integer():
            raise ValueError(f"{cmd} requires an integer")
        if cmd == "AAC" and value not in {1, 2, 4}:
            self._unsupported(
                cmd,
                f"aperture action {value} is not mapped",
            )
        self._current_surf_data.setdefault("special_apertures", {}).setdefault(
            identifier, {}
        )[cmd] = value

    def _read_ast(self, tokens: list[str]) -> None:
        for data in self.data_model.surfaces.values():
            data.pop("AST", None)
        self._current_surf_data["AST"] = True

    def _read_coeff(self, tokens: list[str]) -> None:
        if len(tokens) != 2:
            raise ValueError(f"{tokens[0]} expects one coefficient")
        cmd = tokens[0].upper()
        self._current_surf_data[cmd] = float(tokens[1])
        if re.fullmatch(r"AS\d+", cmd):
            self._coefficient_lines[self._current_surf_idx, cmd] = self._line

    def _read_return(self, tokens: list[str]) -> None:
        # Legacy saved prescriptions encode the default local undo as RCO 0
        # (e.g. Lambda Research EyeModel_LB.len and io61ac.len).
        self._current_surf_data["RCO"] = (
            (int(tokens[1]) or self._current_surf_idx)
            if len(tokens) > 1
            else self._current_surf_idx
        )

    def _read_flag(self, tokens: list[str]) -> None:
        self._current_surf_data[tokens[0]] = True

    def _read_jones(self, tokens: list[str]) -> None:
        if len(tokens) != 2:
            raise ValueError(f"{tokens[0]} requires exactly one argument")
        value = float(tokens[1])
        if tokens[0].startswith("JA") and value < 0:
            raise ValueError(f"{tokens[0]} amplitude must be nonnegative")
        self._current_surf_data[tokens[0]] = value

    def _read_spectrum(self, tokens: list[str]) -> None:
        cmd = tokens[0]
        values = [float(t) for t in tokens[1:]]
        wavelength = cmd.startswith("WV")
        if not values or any(v <= 0 if wavelength else v < 0 for v in values):
            raise ValueError(f"{cmd} requires positive wavelengths/nonnegative weights")
        attr = "_wavelength_values" if wavelength else "_wavelength_weights"
        if len(cmd) == 2:
            setattr(self, attr, values)
            return
        index = int(cmd[2:]) - 1
        if index > 1000 or len(values) != 1:
            raise ValueError(f"{cmd} requires one value and a bounded wavelength index")
        target = getattr(self, attr)
        if wavelength and index > len(target):
            raise ValueError(f"{cmd} leaves undefined wavelength slots")
        target.extend([1.0] * max(0, index + 1 - len(target)))
        target[index] = values[0]

    def _read_nxt(self, tokens: list[str]) -> None:
        # Save current surface and increment index
        self.data_model.surfaces[self._current_surf_idx] = self._current_surf_data
        self._current_surf_idx += 1
        if self._current_surf_idx > self.data_model.num_surfaces:
            raise ValueError("NXT exceeds LEN surface count")
        self._current_surf_data = self.data_model.surfaces.get(
            self._current_surf_idx, {}
        )

    def _read_gto(self, tokens: list[str]) -> None:
        self.data_model.surfaces[self._current_surf_idx] = self._current_surf_data
        index = int(tokens[1])
        if not 0 <= index <= self.data_model.num_surfaces:
            raise ValueError("GTO surface is outside the declared lens")
        self._current_surf_idx = index
        self._current_surf_data = self.data_model.surfaces.get(index, {})

    def _read_end(self, tokens: list[str]) -> None:
        if len(tokens) > 2 or (
            len(tokens) == 2 and int(tokens[1]) != self.data_model.num_surfaces
        ):
            raise ValueError("END surface count differs from LEN")
        self.data_model.surfaces[self._current_surf_idx] = self._current_surf_data
        self._ended = True

    def _read_solve(self, tokens: list[str]) -> None:
        self._clear_constraint("CV" if tokens[0] in {"PU", "PUC"} else "TH")
        self._current_surf_data[tokens[0]] = float(tokens[1])

    def _clear_constraint(self, kind: str) -> None:
        family = _PICKUP_FAMILIES[kind]
        data = self._current_surf_data
        if "pickups" in data:
            data["pickups"] = [
                p for p in data["pickups"] if _PICKUP_FAMILIES[p[0].upper()] != family
            ]
        for key in {"CV": ("PU", "PUC"), "TH": ("PY", "PYC", "EC")}.get(family, ()):
            data.pop(key, None)

    def _read_pickup(self, tokens: list[str]) -> None:
        if len(tokens) < 3:
            raise ValueError("PK requires a pickup type and preceding source")
        kind = tokens[1].upper()
        if kind not in _PICKUP_FAMILIES:
            self._unsupported("PK", f"pickup type {tokens[1]} is not mapped")
            return
        expected = {
            "LN": {4, 5},
            "LNM": {4, 5},
            "CV": {3, 4},
            "CVM": {3, 4},
            "TH": {3, 4},
            "THM": {3, 4},
        }.get(kind, {3})
        if len(tokens) not in expected:
            raise ValueError(f"PK {kind} has an invalid number of arguments")
        int(tokens[2])
        if kind in {"LN", "LNM"}:
            int(tokens[3])
        self._clear_constraint(kind)
        self._current_surf_data.setdefault("pickups", []).append(tokens[1:])

    def _read_aperture_pickup(self, tokens: list[str]) -> None:
        if len(tokens) != 4:
            raise ValueError(
                "APK expects target aperture, source surface and source aperture"
            )
        self._current_surf_data.setdefault("aperture_pickups", []).append(
            [tokens[1].upper(), tokens[2], tokens[3].upper()]
        )
