"""CODE V Data Parser

Parses a CODE V Sequential (.seq) file into a CodeVDataModel. The parser uses
a state-machine approach: global commands are consumed first, then surface
blocks are parsed line by line.

Kramer Harrison, 2026
"""

from __future__ import annotations

import contextlib
import re
import warnings
from typing import Any

import optiland.backend as be
from optiland.fileio.codev.model import CodeVDataModel
from optiland.materials import AbbeMaterial, BaseMaterial, Material
from optiland.physical_apertures import RadialAperture

# Ordered aspheric coefficient letter keys (4th-20th order)
_ASPH_KEYS = ("A", "B", "C", "D", "E", "F", "G", "H", "J")

# Tokens that start a surface definition line
_SURFACE_STARTERS = frozenset({"SO", "SI", "S", "STO"})

# Map from field command to field type string and axis
_FIELD_CMD_MAP: dict[str, tuple[str, str]] = {
    "XAN": ("angle", "x"),
    "YAN": ("angle", "y"),
    "XOB": ("object_height", "x"),
    "YOB": ("object_height", "y"),
    "XIM": ("paraxial_image_height", "x"),
    "YIM": ("paraxial_image_height", "y"),
}


def _is_surface_line(tokens: list[str]) -> bool:
    """Return True if this token list starts a new surface definition.

    Args:
        tokens: Non-empty list of tokens from a single (preprocessed) line.

    Returns:
        True if the first token is a surface starter keyword.
    """
    return tokens[0] in _SURFACE_STARTERS


class CodeVDataParser:
    """Parses a CODE V .seq file into a CodeVDataModel.

    Args:
        filename: Path to the .seq file to parse.

    Attributes:
        filename: The file path being parsed.
        data_model: The CodeVDataModel being populated during parsing.
    """

    def __init__(self, filename: str):
        self.filename = filename
        self.data_model = CodeVDataModel()
        self._current_surf: int = -1
        self._current_surf_data: dict[str, Any] = {}
        self._in_prv_block: bool = False

        # Global-section dispatch table
        self._global_table: dict[str, Any] = {
            "TITLE": self._read_title,
            "TIT": self._read_title,
            "INI": lambda t: None,  # designer initials - ignored
            "DIM": self._read_dim,
            "RDM": self._read_rdm,
            "EPD": self._read_epd,
            "FNO": self._read_fno,
            "NA": self._read_na,
            "NAO": self._read_nao,
            "WL": self._read_wl,
            "WTW": self._read_wtw,
            "REF": self._read_ref,
            "CWL": lambda t: None,  # coating ref wavelength — ignored
            "XAN": self._read_field_cmd,
            "YAN": self._read_field_cmd,
            "XOB": self._read_field_cmd,
            "YOB": self._read_field_cmd,
            "XIM": self._read_field_cmd,
            "YIM": self._read_field_cmd,
            "WTF": self._read_wtf,
            "VUX": lambda t: None,
            "VLX": lambda t: None,
            "VUY": lambda t: None,
            "VLY": lambda t: None,
            "TEM": lambda t: None,
            "PRE": lambda t: None,
            "PRV": self._read_prv,
            "END": self._read_end,
        }

        # Surface-modifier dispatch table
        self._surf_modifier_table: dict[str, Any] = {
            "STO": self._read_sto,
            "SLB": lambda t: None,  # surface label — ignored
            "K": self._read_conic,
            "XDE": self._read_xde,
            "YDE": self._read_yde,
            "ZDE": self._read_zde,
            "ADE": self._read_ade,
            "BDE": self._read_bde,
            "CDE": self._read_cde,
            "DAR": lambda t: None,
            "BEN": lambda t: None,
            "REV": lambda t: None,
            "CIR": self._read_cir,
            "REX": lambda t: None,
            "REY": lambda t: None,
            "THC": lambda t: None,
            "CCY": lambda t: None,
            "PIM": lambda t: None,
            "SPH": lambda t: None,
            "CON": lambda t: None,
            "ASP": lambda t: None,
            "CYL": lambda t: None,
            "DIF": lambda t: None,
            "HWL": lambda t: None,
            "HOR": lambda t: None,
            "HCT": lambda t: None,
            "HCO": lambda t: None,
        }
        # Aspheric coefficient keys
        for _key in _ASPH_KEYS:
            self._surf_modifier_table[_key] = self._read_asph_coeff

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def parse(self) -> CodeVDataModel:
        """Read the CODE V file and extract optical data into a CodeVDataModel.

        Returns:
            A populated CodeVDataModel.

        Raises:
            ValueError: If the file cannot be read or no aperture data found.
        """
        lines = self._load_and_preprocess()
        for tokens in lines:
            self._dispatch(tokens)

        self._finalize_surface()
        self._finalize_fields()
        return self.data_model

    # ------------------------------------------------------------------
    # File loading and preprocessing
    # ------------------------------------------------------------------

    def _load_and_preprocess(self) -> list[list[str]]:
        """Load the file, strip comments, join continuations, tokenize.

        Semicolons are treated as command separators (as in CODE V), so a
        single physical line may yield multiple logical token lists.

        Returns:
            List of token lists (one list per logical command).
        """
        try:
            with open(self.filename, encoding="utf-8") as fh:
                raw_lines = fh.readlines()
        except UnicodeDecodeError:
            with open(self.filename, encoding="latin-1") as fh:
                raw_lines = fh.readlines()

        result: list[list[str]] = []
        continuation = ""
        for line in raw_lines:
            # Strip inline comments
            if "!" in line:
                line = line[: line.index("!")]
            line = line.rstrip()
            if not line:
                if continuation:
                    self._tokenize_logical_line(continuation, result)
                    continuation = ""
                continue
            if line.endswith("&"):
                continuation += " " + line[:-1]
            else:
                full = (continuation + " " + line).strip()
                continuation = ""
                self._tokenize_logical_line(full, result)

        if continuation:
            self._tokenize_logical_line(continuation, result)

        return result

    def _tokenize_logical_line(self, line: str, result: list[list[str]]) -> None:
        """Split a logical line on semicolons and append token lists.

        Double-quoted string literals (e.g. version strings) are stripped
        before splitting so they cannot introduce spurious semicolons.

        Args:
            line: A fully-joined logical line (continuations already merged).
            result: Output list to append token lists to.
        """
        # Remove double-quoted string literals (e.g. "VERSION: 10.7 ...")
        line = re.sub(r'"[^"]*"', "", line)
        for segment in line.split(";"):
            tokens = segment.split()
            if tokens:
                result.append(tokens)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, tokens: list[str]) -> None:
        """Route a token list to the appropriate handler.

        Args:
            tokens: Non-empty list of tokens from one logical line.
        """
        if self._in_prv_block:
            if tokens[0].upper() == "END":
                self._in_prv_block = False
            return  # skip private catalog data

        cmd = tokens[0].upper()

        # Surface-starting keywords always start a new surface context
        if cmd in ("SO", "SI") or (
            cmd == "S" and (len(tokens) == 1 or _looks_like_float(tokens[1]))
        ):
            self._flush_surface()
            self._start_surface(tokens)
            return

        # STO may also appear as a standalone surface keyword (stop surface)
        if cmd == "STO" and self._current_surf < 0:
            # Treat as a standard surface with no geometry token
            self._flush_surface()
            self._start_surface(["S", "0.0", "0.0"])
            self._current_surf_data["is_stop"] = True
            return

        # Within a surface context, check modifier table first
        if self._current_surf >= 0 and cmd in self._surf_modifier_table:
            self._surf_modifier_table[cmd](tokens)
            return

        # Global table
        if cmd in self._global_table:
            self._global_table[cmd](tokens)

    # ------------------------------------------------------------------
    # Surface lifecycle
    # ------------------------------------------------------------------

    def _flush_surface(self) -> None:
        """Store the in-progress surface dict into the data model."""
        if self._current_surf >= 0:
            self.data_model.surfaces[self._current_surf] = self._current_surf_data

    def _start_surface(self, tokens: list[str]) -> None:
        """Begin a new surface from a surface-definition token list.

        Args:
            tokens: Tokens for the surface line,
                e.g. ``["S", "50.0", "5.0", "N-BK7_SCHOTT"]``.
        """
        self._current_surf += 1
        cmd = tokens[0].upper()

        surf_type = {"SO": "object", "SI": "image"}.get(cmd, "standard")
        self._current_surf_data = {
            "type": surf_type,
            "radius": float(be.inf),
            "thickness": 0.0,
            "material": None,
            "is_stop": False,
            "conic": 0.0,
            "coefficients": [],
            "xde": 0.0,
            "yde": 0.0,
            "zde": 0.0,
            "ade": 0.0,
            "bde": 0.0,
            "cde": 0.0,
            "aperture": None,
        }

        # Parse radius and thickness from inline tokens
        try:
            r_raw = float(tokens[1]) if len(tokens) > 1 else 0.0
            self._current_surf_data["radius"] = self._convert_radius(r_raw)
        except (ValueError, IndexError):
            pass

        with contextlib.suppress(ValueError, IndexError):
            self._current_surf_data["thickness"] = float(tokens[2])

        # Optional glass token
        if len(tokens) > 3:
            glass_token = tokens[3]
            self._current_surf_data["material"] = self._parse_glass(glass_token)

    def _convert_radius(self, val: float) -> float:
        """Convert a raw file radius/curvature value to a radius.

        Handles the RDM flag: if radius_mode=True values are already radii;
        if False values are curvatures.

        Args:
            val: Raw value from the file.

        Returns:
            Radius in lens units, or ``float(be.inf)`` for planar.
        """
        if val == 0.0:
            return float(be.inf)
        if self.data_model.radius_mode:
            return float(val)
        # curvature mode: radius = 1/curvature
        return 1.0 / float(val)

    # ------------------------------------------------------------------
    # Global handlers
    # ------------------------------------------------------------------

    def _read_title(self, tokens: list[str]) -> None:
        raw = " ".join(tokens[1:]).strip().strip("'\"")
        self.data_model.name = raw or None

    def _read_dim(self, tokens: list[str]) -> None:
        mapping = {"M": "MM", "C": "CM", "I": "IN"}
        self.data_model.units = mapping.get(tokens[1].upper(), "MM")

    def _read_rdm(self, tokens: list[str]) -> None:
        self.data_model.radius_mode = len(tokens) < 2 or tokens[1].upper() != "N"

    def _read_epd(self, tokens: list[str]) -> None:
        self.data_model.aperture["EPD"] = float(tokens[1])

    def _read_fno(self, tokens: list[str]) -> None:
        self.data_model.aperture["FNO"] = float(tokens[1])

    def _read_na(self, tokens: list[str]) -> None:
        self.data_model.aperture["NA"] = float(tokens[1])

    def _read_nao(self, tokens: list[str]) -> None:
        self.data_model.aperture["NAO"] = float(tokens[1])

    def _read_wl(self, tokens: list[str]) -> None:
        # Wavelengths in nm -> convert to µm
        self.data_model.wavelengths["data"] = [float(v) / 1000.0 for v in tokens[1:]]

    def _read_wtw(self, tokens: list[str]) -> None:
        self.data_model.wavelengths["weights"] = [float(v) for v in tokens[1:]]

    def _read_ref(self, tokens: list[str]) -> None:
        # 1-based -> 0-based
        self.data_model.wavelengths["primary_index"] = int(tokens[1]) - 1

    def _read_field_cmd(self, tokens: list[str]) -> None:
        cmd = tokens[0].upper()
        field_type, axis = _FIELD_CMD_MAP[cmd]
        if "type" not in self.data_model.fields:
            self.data_model.fields["type"] = field_type
        self.data_model.fields[axis] = [float(v) for v in tokens[1:]]

    def _read_wtf(self, tokens: list[str]) -> None:
        self.data_model.fields["weights"] = [float(v) for v in tokens[1:]]

    def _read_prv(self, tokens: list[str]) -> None:
        warnings.warn(
            "Private glass catalog (PRV block) is not supported; "
            "glasses defined in PRV will be resolved as AbbeMaterial if possible.",
            UserWarning,
            stacklevel=2,
        )
        self._in_prv_block = True

    def _read_end(self, tokens: list[str]) -> None:
        self._in_prv_block = False

    # ------------------------------------------------------------------
    # Surface modifier handlers
    # ------------------------------------------------------------------

    def _read_sto(self, tokens: list[str]) -> None:
        # STO Sn — global cross-reference: stop is at the Nth surface label
        if len(tokens) > 1 and re.match(r"^[Ss]\d+$", tokens[1]):
            self.data_model.sto_surface_index = int(tokens[1][1:])
        else:
            self._current_surf_data["is_stop"] = True

    def _read_conic(self, tokens: list[str]) -> None:
        self._current_surf_data["conic"] = float(tokens[1])

    def _read_xde(self, tokens: list[str]) -> None:
        self._current_surf_data["xde"] = float(tokens[1])

    def _read_yde(self, tokens: list[str]) -> None:
        self._current_surf_data["yde"] = float(tokens[1])

    def _read_zde(self, tokens: list[str]) -> None:
        self._current_surf_data["zde"] = float(tokens[1])

    def _read_ade(self, tokens: list[str]) -> None:
        self._current_surf_data["ade"] = float(tokens[1])

    def _read_bde(self, tokens: list[str]) -> None:
        self._current_surf_data["bde"] = float(tokens[1])

    def _read_cde(self, tokens: list[str]) -> None:
        self._current_surf_data["cde"] = float(tokens[1])

    def _read_asph_coeff(self, tokens: list[str]) -> None:
        key_letter = tokens[0].upper()
        idx = _ASPH_KEYS.index(key_letter)
        coeffs = self._current_surf_data["coefficients"]
        # Extend list if necessary
        while len(coeffs) <= idx:
            coeffs.append(0.0)
        coeffs[idx] = float(tokens[1])
        # Mark surface as aspheric
        self._current_surf_data["profile"] = "ASP"

    def _read_cir(self, tokens: list[str]) -> None:
        # CIR [CLR|OBS|EDG] <radius>
        # Find the numeric token
        for tok in tokens[1:]:
            try:
                r = float(tok)
                self._current_surf_data["aperture"] = RadialAperture(r_min=0.0, r_max=r)
                return
            except ValueError:
                continue

    # ------------------------------------------------------------------
    # Glass parsing
    # ------------------------------------------------------------------

    def _parse_glass(self, token: str) -> BaseMaterial | str | None:
        """Parse a CODE V glass specification token.

        Handles: ``REFL``, ``<name>_<catalog>``, ``<Nd>:<Vd>``,
        6-digit ``NNNNVV``, legacy ``NNN.VVV`` decimal code,
        and bare glass name.  CODE V omits hyphens from Schott/Ohara
        glass names (e.g. ``NBK7`` for ``N-BK7``); this method tries
        inserting the hyphen when the direct lookup fails.

        Args:
            token: Glass specification string from the surface line.

        Returns:
            A ``BaseMaterial``, ``"mirror"`` string, or ``None`` for air.
        """
        token = token.strip("'\"")
        if not token or token.upper() in ("AIR", ""):
            return None

        upper = token.upper()

        if upper == "REFL":
            return "mirror"

        # <Nd>:<Vd> fictitious glass
        if ":" in token:
            try:
                nd_str, vd_str = token.split(":", 1)
                return AbbeMaterial(float(nd_str), float(vd_str))
            except (ValueError, TypeError):
                pass

        # Legacy NNN.VVV decimal glass code (e.g. 569.631 -> Nd=1.569, Vd=63.1)
        # Also handles extended-precision forms like 517000.520000
        if re.match(r"^\d+\.\d+$", token):
            try:
                int_str, dec_str = token.split(".", 1)
                nd = 1.0 + int(int_str[:3]) / 1000.0
                vd = int(dec_str[:3].ljust(3, "0")) / 10.0
                if 1.0 < nd < 4.0 and 0.0 < vd < 200.0:
                    return AbbeMaterial(nd, vd)
            except (ValueError, IndexError):
                pass

        # 6-digit glass code: NNN VVV -> Nd = 1 + NNN/1000, Vd = VVV/10
        # e.g. 516800 -> Nd=1.516, Vd=80.0
        if len(token) == 6 and token.isdigit():
            try:
                nd = 1.0 + int(token[:3]) / 1000.0
                vd = int(token[3:]) / 10.0
                return AbbeMaterial(nd, vd)
            except ValueError:
                pass

        # <name>_<catalog> format
        if "_" in token:
            parts = token.rsplit("_", 1)
            name, catalog = parts[0], parts[1]
            catalog_lower = catalog.lower()
            for candidate in _glass_name_candidates(name):
                try:
                    return Material(candidate, catalog_lower)
                except ValueError:
                    pass
            # Try without catalog
            for candidate in _glass_name_candidates(name):
                try:
                    return Material(candidate)
                except ValueError:
                    pass

        # Bare glass name — try catalog lookup with hyphen normalization
        for candidate in _glass_name_candidates(upper):
            try:
                return Material(candidate)
            except ValueError:
                pass

        warnings.warn(
            f"Glass '{token}' could not be resolved; treating as air.",
            UserWarning,
            stacklevel=2,
        )
        return None

    # ------------------------------------------------------------------
    # Finalizers
    # ------------------------------------------------------------------

    def _finalize_surface(self) -> None:
        """Flush the last in-progress surface into the model."""
        if self._current_surf >= 0:
            self.data_model.surfaces[self._current_surf] = self._current_surf_data

    def _finalize_fields(self) -> None:
        """Ensure x and y field arrays are consistent, defaulting x to zeros."""
        fields = self.data_model.fields
        if "y" in fields and "x" not in fields:
            fields["x"] = [0.0] * len(fields["y"])
        elif "x" in fields and "y" not in fields:
            fields["y"] = [0.0] * len(fields["x"])


def _glass_name_candidates(name: str) -> list[str]:
    """Return name variants to try when resolving a CODE V glass name.

    CODE V omits hyphens from glass names that contain them in catalog
    databases (e.g. ``NBK7`` instead of ``N-BK7``).  This function
    generates candidates by trying to re-insert a hyphen after the first
    character when the name starts with a single letter that is commonly
    used as a prefix (N, S, P, Q, E, H, L, M, K, F, G, C).

    Args:
        name: Raw glass name token (may be upper-cased).

    Returns:
        List of candidate name strings to try in order.
    """
    candidates: list[str] = [name]
    _HYPHEN_PREFIXES = frozenset("NSPQEHLMKFGC")
    if len(name) > 2 and name[0].upper() in _HYPHEN_PREFIXES and name[1].isalpha():
        candidates.append(f"{name[0]}-{name[1:]}")
    return candidates


def _looks_like_float(s: str) -> bool:
    """Return True if *s* looks like a float literal.

    Args:
        s: String to test.

    Returns:
        True if ``s`` can be parsed as a float.
    """
    try:
        float(s)
        return True
    except ValueError:
        return False
