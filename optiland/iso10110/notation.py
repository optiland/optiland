"""ISO 10110 Notation Models

Typed, validated containers for per-surface and per-element fabrication
specifications. Built on pydantic (``extra="forbid"``) so a typo'd or
renamed keyword argument raises immediately rather than being silently
dropped. All fields default to None / empty string; unspecified fields are
rendered as the placeholder dash (—) in the drawing.

ISO 10110 part mapping
----------------------
0/  Stress birefringence          → ElementSpec.birefringence
1/  Bubbles and inclusions        → ElementSpec.bubbles
2/  Inhomogeneity and striae      → ElementSpec.homogeneity_grade / nh_class /
                                    striae_grade / striae_density / striae_shadowgraph
3/  Surface form error            → SurfaceSpec.sag_tolerance / irregularity /
                                    rotationally_symmetric / test_wavelength
4/  Centration tolerance          → SurfaceSpec.centration / centration_decentration
5/  Surface imperfections         → SurfaceSpec.imperfections / scratches /
                                    edge_chips / coating_imperfections
6/  Laser damage threshold        → SurfaceSpec.laser_damage
7/  Surface treatment / coatings  → SurfaceSpec.coating
8/  Surface texture               → SurfaceSpec.roughness
13/ Transmitted wavefront         → SurfaceSpec.wavefront_deformation
15/ Assembly surface imperfections→ SurfaceSpec.assembly_imperfections

Bernhard Lutzer, 2026
"""

from __future__ import annotations

import re
import warnings

from pydantic import BaseModel, ConfigDict, model_validator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Valid NH homogeneity class designators.
#: Preferred designators per ISO 10110-18:2019 Table 2 are NH100, NH040,
#: NH010, NH004, NH002, NH001.  The older series (NH40, NH20 …) is retained
#: for backward compatibility with legacy drawings.
NH_CLASSES: frozenset[str] = frozenset(
    {
        # ISO 10110-18:2019 Table 2 (preferred)
        "NH100",
        "NH040",
        "NH010",
        "NH004",
        "NH002",
        "NH001",
        # Legacy designators (backward compatibility)
        "NH40",
        "NH20",
        "NH10",
        "NH5",
        "NH2",
        "NH1",
        "NH0.5",
        "NH0.2",
        "NH0.1",
    }
)

#: Map legacy integer homogeneity grade (0–5) to ISO 10110-18:2019 NH class.
_LEGACY_NH_MAP: dict[int, str] = {
    0: "NH100",
    1: "NH040",
    2: "NH010",
    3: "NH004",
    4: "NH002",
    5: "NH001",
}

#: Valid ISO 10110-8:2019 Table 1 polish grade designators.
#: P1 ≤ 8 nm, P2 ≤ 4 nm, P3 ≤ 2 nm, P4 ≤ 1 nm (Rq over 0.002–1.0 mm band).
_POLISH_GRADES: frozenset[str] = frozenset({"P1", "P2", "P3", "P4"})

#: Valid grating-type designators per ISO 10110-16:2017 Table 5.
#: LG = linear grating, CG = concentric grating, CGH = computer-generated
#: hologram, TG = transmission, RG = reflection, AG = amplitude grating,
#: SG = surface-relief grating, IG = index grating.
GRATING_TYPES: frozenset[str] = frozenset(
    {
        "LG",
        "CG",
        "CGH",
        "TG",
        "RG",
        "AG",
        "SG",
        "IG",
    }
)

#: Renard R5 series base values (each multiplied by 10^n covers all decades).
_R5_BASE = (1.0, 1.6, 2.5, 4.0, 6.3)


def _in_r5(value: float, rtol: float = 0.05) -> bool:
    """Return True when *value* is within *rtol* of a Renard R5 preferred number."""
    if value <= 0:
        return False
    import math

    decade = 10 ** math.floor(math.log10(value))
    scaled = value / decade
    return any(abs(scaled - b) / b <= rtol for b in _R5_BASE)


def _r5_advisory(spec_str: str, field_name: str) -> None:
    """Emit one advisory warning when any ``×A`` size in *spec_str* is not R5.

    Args:
        spec_str:   ISO 10110-7/3 notation string, e.g. ``"2×0.04"``.
        field_name: Human-readable field name used in the warning text.
    """
    for m in re.finditer(r"×(\d+(?:\.\d+)?)", _fix_mul(spec_str)):
        v = float(m.group(1))
        if v > 0 and not _in_r5(v):
            warnings.warn(
                f"{field_name}: size {v} is not a Renard R5 preferred number "
                "(1.0, 1.6, 2.5, 4.0, 6.3 × 10^n). See ISO 10110-7.",
                stacklevel=4,
            )
            break  # one warning per field


def round_to_r5(value: float) -> float:
    """Round *value* up to the nearest Renard R5 preferred number.

    The R5 series is 1.0, 1.6, 2.5, 4.0, 6.3 and their decade multiples.
    Rounding is always upward (conservative — never relaxes a tolerance).

    Args:
        value: Positive number to round.

    Returns:
        The smallest R5 preferred number ≥ *value*.

    Raises:
        ValueError: If *value* is not positive.

    Examples::

        round_to_r5(0.05)  → 0.063
        round_to_r5(0.10)  → 0.10   (already R5)
        round_to_r5(0.12)  → 0.16
        round_to_r5(1.8)   → 2.5
    """
    import math

    if value <= 0:
        raise ValueError(f"value must be > 0, got {value}")
    decade = 10 ** math.floor(math.log10(value))
    scaled = value / decade
    for b in sorted(_R5_BASE):
        if scaled <= b * (1 + 1e-9):  # 1e-9 tolerance for exact matches
            return round(b * decade, 10)
    # scaled > 6.3 → next decade
    return round(_R5_BASE[0] * decade * 10, 10)


def _fix_mul(s: str) -> str:
    """Replace ASCII ``x`` separator with ISO multiplication sign × (U+00D7)."""
    return re.sub(r"(\d)x(\d)", r"\1×\2", s)


# ---------------------------------------------------------------------------
# Surface-level specification
# ---------------------------------------------------------------------------


class SurfaceSpec(BaseModel):
    """ISO 10110 fabrication specifications for a single optical surface.

    3/ Surface form (ISO 10110-5)
    -----------------------------
    Format: ``3/ A(B/C)``

    - *sag_tolerance* (A): maximum sag error from test plate in fringes.
      Use ``None`` (rendered as ``—``) when the radius tolerance is given
      as a dimension on the drawing.
    - *irregularity* (B): p-v irregularity in fringes, placed in parentheses.
    - *rotationally_symmetric* (C): rotationally-symmetric component of the
      irregularity (optional second value after ``/`` inside the parentheses).
    - *test_wavelength*: test wavelength in nm.  When set and ≠ 546.07 nm the
      drawn ``3/`` code appends ``@ λ=<value> nm`` per ISO 10110-5.

    4/ Centering (ISO 10110-6)
    --------------------------
    - *centration*: wedge angle tolerance in arc minutes.
    - *centration_decentration*: optional lateral decentration δ in mm.

    5/ Surface imperfections (ISO 10110-7)
    ---------------------------------------
    Format: ``5/ NxA; CN'xA'; LN"xA"; EA'"``

    - *imperfections*: base dig specification string, e.g. ``"2×0.04"``.
    - *coating_imperfections*: coating digs, C-prefix, e.g. ``"1×0.016"``.
    - *scratches*: long scratches, L-prefix, e.g. ``"1×0.001"``
      (N scratches × width in mm).
    - *edge_chips*: edge-chip protrusion, E-prefix, e.g. ``"0.5"`` (mm).

    6/ Laser damage threshold (ISO 10110-17)
    ----------------------------------------
    - *laser_damage*: free-form laser damage threshold string per ISO 10110-17,
      e.g. ``"H 20J/cm²; 10ns; 1064nm"``.

    7/ Coating (ISO 10110-9 / ISO 9211)
    ------------------------------------
    - *coating*: free-form coating callout, e.g. ``"AR 400-700 nm R<0.5%"``.

    Dimensions
    ----------
    - *r_tolerance*: ± radius tolerance in mm shown next to R on the drawing.
    - *ca_diameter*: clear aperture diameter in mm (the usable optical zone,
      usually smaller than the physical edge diameter).
    - *ca_tolerance*: tolerance string for the CA, e.g. ``"+0.1/-0"`` or
      ``"±0.1"``.

    8/ Surface texture (ISO 10110-8:2019)
    ---------------------------------------
    - *roughness*: surface finish designator string (ISO 10110-8:2019 §5).
      Valid forms:

      - ``"P"``  — polished, no quantitative spec (§5.3.1)
      - ``"P1"`` / ``"P2"`` / ``"P3"`` / ``"P4"`` — polish grade (§5.3.2,
        Table 1): P1 ≤ 8 nm, P2 ≤ 4 nm, P3 ≤ 2 nm, P4 ≤ 1 nm Rq over
        the spatial band 0.002–1.0 mm.
      - ``"G"``  — matte/ground surface (§5.2); Rq in μm may follow, e.g.
        ``"G / 0.0025-0.8 / Rq 2"``
      - ``"-1/Rq 0.002"``  — polished with upper spatial-band limit 1 mm
        and Rq ≤ 0.002 μm (§5.3.3)
      - ``"0.002-1/Rq 0.002"``  — polished with full spatial band (§5.3.3)
    """

    model_config = ConfigDict(extra="forbid")

    # ── 3/ surface form ──────────────────────────────────────────────────
    sag_tolerance: float | None = None  # A: test-plate sag error
    irregularity: float | None = None  # B: p-v irregularity
    rotationally_symmetric: float | None = None  # C: rotationally-symmetric term
    test_wavelength: float | None = None  # test λ in nm (546.07 nm = default)
    form_unit: str = "fr"  # unit: "fr" (fringes) or "nm" (nanometres)

    # ── 4/ centering ─────────────────────────────────────────────────────
    centration: float | None = None  # σ in arc minutes
    centration_decentration: float | None = None  # δ lateral decentration (mm)

    # ── 5/ surface imperfections ─────────────────────────────────────────
    imperfections: str | None = None  # base NxA
    coating_imperfections: str | None = None  # C prefix, e.g. "1×0.016"
    scratches: str | None = None  # L prefix, e.g. "1×0.001"
    edge_chips: str | None = None  # E prefix, e.g. "0.5"

    # ── 7/ coating ───────────────────────────────────────────────────────
    coating: str | None = None

    # ── Dimensions ───────────────────────────────────────────────────────
    r_tolerance: str | float | None = None  # ± tol on R (mm): "±0.05" or bare 0.05
    ca_diameter: float | None = None  # clear aperture Ø (mm)
    ca_tolerance: str | float | None = None  # e.g. "+0.1/-0", "±0.05", or bare 0.05

    # ── 6/ laser damage threshold (ISO 10110-17) ─────────────────────────
    laser_damage: str | None = None  # e.g. "H 20J/cm²; 10ns; 1064nm"

    # ── 13/ transmitted wavefront deformation (ISO 10110-14) ─────────────
    wavefront_deformation: str | None = None  # e.g. "0.1(0.05)" (A(B) waves)

    # ── 15/ assembly surface imperfections (ISO 10110-7) ─────────────────
    assembly_imperfections: str | None = None  # e.g. "2×0.04" (cemented/contacted)

    # ── 8/ surface texture ───────────────────────────────────────────────
    roughness: str | None = None  # ISO 10110-8:2019: "P", "P2", "G", "-1/Rq 0.002", …

    # ── Chamfer (ISO 10110-12) ────────────────────────────────────────────
    chamfer: str | None = None  # width range e.g. "0.2–0.5" mm
    chamfer_angle: float | None = None  # included angle in degrees (default 45°)

    # ── Sharp edge (§5.9.5.2) ────────────────────────────────────────────
    sharp_edge: bool = False  # True → "0" symbol; edge must stay sharp

    # ── Pyramid error / prismatic tolerance (ISO 10110-6 §5.3) ──────────
    pyramid_error: float | None = (
        None  # maximum permissible prismatic deviation σ' (arc min)
    )

    # ── Grating type (ISO 10110-16 Table 5) ──────────────────────────────
    # Applies to diffractive/grating surfaces.  Placed at the start of the
    # surface specification column per ISO 10110-16 §5.3.
    # Valid values: LG (linear), CG (concentric), CGH (computer-generated
    # hologram), TG (transmission), RG (reflection), AG (amplitude),
    # SG (surface relief), IG (index grating).
    grating_type: str = "LG"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    # Numeric string coercion (e.g. values arriving as strings from YAML) is
    # handled automatically by pydantic's field typing; this validator only
    # covers ISO-specific business rules that span more than one field.

    @model_validator(mode="after")
    def _validate(self) -> SurfaceSpec:
        if self.centration is not None and self.centration < 0:
            raise ValueError(f"centration must be ≥ 0, got {self.centration}")
        if self.ca_diameter is not None and self.ca_diameter <= 0:
            raise ValueError(f"ca_diameter must be > 0, got {self.ca_diameter}")

        # ISO 10110-6: the lateral decentration δ appears inside σ'(δ); without σ
        # the parenthesis is absent and δ is silently dropped.
        if self.centration_decentration is not None and self.centration is None:
            warnings.warn(
                "centration_decentration (δ) is set but centration (σ) is not. "
                "Per ISO 10110-6 the decentration appears only inside '4/ σ′(δ)'; "
                "without σ, δ will not be rendered. "
                "Set centration as well to include the decentration value.",
                stacklevel=3,
            )

        # ISO 10110-5 §4.2: the rotationally-symmetric irregularity component C
        # is the second element inside the parenthesis (A(B/C)); B must be set.
        # Without B the parenthesis is absent and C is silently dropped.
        if self.rotationally_symmetric is not None and self.irregularity is None:
            warnings.warn(
                "rotationally_symmetric (C) is set but irregularity (B) is not. "
                "Per ISO 10110-5 §4.2 the C value appears only inside the '(B/C)' "
                "parenthesis; without B, C will not be rendered in the 3/ code. "
                "Set irregularity as well to include the RSI component.",
                stacklevel=3,
            )
        # ISO 10110-5 §4.2: the RSI component C cannot exceed total irregularity B,
        # because C is the rotationally-symmetric *part* of B.
        if (
            self.rotationally_symmetric is not None
            and self.irregularity is not None
            and self.rotationally_symmetric > self.irregularity
        ):
            warnings.warn(
                f"rotationally_symmetric (C={self.rotationally_symmetric}) exceeds "
                f"irregularity (B={self.irregularity}). "
                "Per ISO 10110-5 §4.2, C is the RSI component of B and must be ≤ B.",
                stacklevel=3,
            )
        if self.form_unit not in ("fr", "nm"):
            raise ValueError(
                f"form_unit must be 'fr' (fringes) or 'nm' (nanometres), "
                f"got {self.form_unit!r}"
            )

        # Normalise ASCII 'x' → '×' in all imperfection string fields so that
        # YAML round-trips always use the ISO multiplication sign (U+00D7).
        for _attr in (
            "imperfections",
            "coating_imperfections",
            "scratches",
            "assembly_imperfections",
        ):
            _v = getattr(self, _attr)
            if _v is not None:
                setattr(self, _attr, _fix_mul(_v))

        # Renard R5 advisory for all ×A size fields (ISO 10110-7 / ISO 10110-3)
        for _field, _fname in (
            (self.imperfections, "5/ imperfections"),
            (self.coating_imperfections, "5/ coating_imperfections (C prefix)"),
            (self.scratches, "5/ scratches (L prefix)"),
            (self.assembly_imperfections, "15/ assembly_imperfections"),
        ):
            if _field is not None:
                _r5_advisory(_field, _fname)

        # Advisory for invalid polish grade (ISO 10110-8:2019 §4.3.3 — only P1–P4)
        if self.roughness is not None:
            _pg = re.match(r"^P(\d+)\b", self.roughness.strip())
            if _pg:
                grade_str = f"P{_pg.group(1)}"
                if grade_str not in _POLISH_GRADES:
                    warnings.warn(
                        f"'{grade_str}' is not a valid ISO 10110-8:2019 polish grade. "
                        "Only P1, P2, P3, P4 are defined in Table 1. "
                        f"Got {grade_str}.",
                        stacklevel=3,
                    )

        # Advisory for unknown grating type (ISO 10110-16:2017 Table 5)
        if self.grating_type and self.grating_type not in GRATING_TYPES:
            warnings.warn(
                f"grating_type '{self.grating_type}' is not a standard ISO 10110-16 "
                f"Table 5 designator.  Known types: {sorted(GRATING_TYPES)}.",
                stacklevel=3,
            )

        # Advisory for 13/ wavefront deformation format (ISO 10110-14)
        # Expected: A  or  A(B)  or  A(B/C)  where A, B, C are non-negative numbers.
        if self.wavefront_deformation is not None:
            _wf_pat = re.compile(
                r"^\d+(?:\.\d+)?"
                r"(?:\(\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?\))?$"
            )
            if not _wf_pat.match(self.wavefront_deformation.strip()):
                warnings.warn(
                    f"13/ wavefront_deformation '{self.wavefront_deformation}' "
                    "does not match the ISO 10110-14 format A(B/C) where A, B, "
                    "C are non-negative numbers "
                    "(e.g. '0.25', '0.25(0.1)', '0.25(0.1/0.05)').",
                    stacklevel=3,
                )

        return self

    # ------------------------------------------------------------------
    # ISO formatted strings
    # ------------------------------------------------------------------

    def fmt_form_error(self) -> str:
        """Return the ISO ``3/`` code string.

        Format: ``3/ A(B/C)``  where A = sag error from test plate,
        B = p-v irregularity, C = rotationally-symmetric component.

        Units follow *form_unit*: ``"fr"`` (fringes, default) or ``"nm"``
        (nanometres, preferred by ISO 10110-5 since 2016 revision).  When
        ``"nm"`` each numeric value is suffixed with `` nm``.

        A is omitted entirely when *sag_tolerance* is ``None`` (radius
        tolerance is given as a dimension on the drawing).  The parenthesised
        B/C group is omitted when neither irregularity nor RSI is given.

        Per ISO 10110-5:2016 §4.3, when fringe units are used the wavelength
        **must always** be stated: the string appends ``; λ=<value> nm``.
        If *test_wavelength* is ``None``, the ISO standard green-mercury
        reference wavelength 546.07 nm is assumed.  For nm units no wavelength
        annotation is needed.

        Examples (fringes)::

            sag=2, irr=0.5, rsi=0.25               → "3/ 2(0.5/0.25); λ=546.1 nm"
            irr=0.5, test_wavelength=632.8          → "3/ (0.5); λ=632.8 nm"
            nothing set                             → "3/ —"

        Examples (nm)::

            form_unit="nm", sag=50, irr=20  → "3/ 50 nm(20 nm)"
        """
        has_a = self.sag_tolerance is not None
        has_b = self.irregularity is not None
        has_c = self.rotationally_symmetric is not None

        if not has_a and not has_b:
            return "3/ —"

        use_nm = self.form_unit == "nm"
        u = " nm" if use_nm else ""

        def _fmtv(v) -> str:
            # Show integer values without decimal point for cleanliness
            return (f"{int(v)}" if float(v) == int(float(v)) else str(v)) + u

        bc = ""
        if has_b:
            inner = (
                f"{_fmtv(self.irregularity)}/{_fmtv(self.rotationally_symmetric)}"
                if has_c
                else _fmtv(self.irregularity)
            )
            bc = f"({inner})"

        a = _fmtv(self.sag_tolerance) if has_a else ""
        result = f"3/ {a}{bc}"

        # ISO 10110-5:2016 §4.3 — fringe units must always carry the wavelength.
        # nm units are absolute and need no wavelength annotation.
        if not use_nm:
            wl = self.test_wavelength if self.test_wavelength is not None else 546.07
            result += f"; \u03bb={wl:.1f} nm"

        return result

    def fmt_centration(self) -> str:
        """Return the ISO ``4/`` code string.

        Format: ``4/ σ'(δ)``  where σ = tilt tolerance in arc-minutes and
        δ = lateral decentration in mm (optional, per ISO 10110-6).
        """
        if self.centration is None:
            return "4/ —"
        v = self.centration
        v_str = str(int(v)) if v == int(v) else str(v)
        base = f"4/ {v_str}\u2032"  # U+2032 prime for arc-minutes
        if self.centration_decentration is not None:
            d = self.centration_decentration
            d_str = str(int(d)) if d == int(d) else str(d)
            base += f"({d_str})"
        return base

    def fmt_imperfections(self) -> str:
        """Return the ISO ``5/`` code string with optional L, E, C extensions.

        Any ASCII ``x`` separator between count and size (e.g. ``2x0.04``) is
        normalised to the ISO-standard multiplication sign ``×`` (U+00D7).
        """
        parts: list[str] = []
        if self.imperfections is not None:
            parts.append(_fix_mul(self.imperfections))
        if self.coating_imperfections is not None:
            parts.append(f"C{_fix_mul(self.coating_imperfections)}")
        if self.scratches is not None:
            parts.append(f"L{_fix_mul(self.scratches)}")
        if self.edge_chips is not None:
            parts.append(f"E{self.edge_chips}")
        if not parts:
            return "5/ —"
        return "5/ " + "; ".join(parts)

    def fmt_coating(self) -> str:
        """Return the ISO ``7/`` coating designation string (ISO 10110-9).

        Returns ``"7/ —"`` when no coating is specified so the field is never
        omitted from the drawing (ISO 10110 requires all codes to appear).
        No ``λ`` prefix — ISO 10110-9 / ISO 9211 notation does not use one.
        """
        if self.coating is None:
            return "7/ —"
        return f"7/ {self.coating}"

    def fmt_laser_damage(self) -> str:
        """Return the ISO ``6/`` laser damage threshold string (ISO 10110-17).

        Returns ``"6/ —"`` when not specified.
        """
        if self.laser_damage is None:
            return "6/ —"
        return f"6/ {self.laser_damage}"

    def fmt_wavefront(self) -> str:
        """Return the ISO ``13/`` transmitted wavefront deformation string (10110-14).

        Format: ``13/ A(B)``  where A = p-v wavefront error in waves,
        B = irregularity component (optional, placed in parentheses).

        Returns ``"13/ —"`` when not specified.
        """
        if self.wavefront_deformation is None:
            return "13/ —"
        return f"13/ {self.wavefront_deformation}"

    def fmt_assembly_imperfections(self) -> str:
        """Return the ISO ``15/`` assembly surface imperfections string (ISO 10110-7).

        Applies to cemented or contacted surfaces evaluated as a sub-assembly.
        Returns ``"15/ —"`` when not specified.
        """
        if self.assembly_imperfections is None:
            return "15/ —"
        return f"15/ {_fix_mul(self.assembly_imperfections)}"

    def fmt_pyramid_error(self) -> str:
        """Return the pyramid-error ``pyr σ'`` string (ISO 10110-6 §5.3).

        Pyramid error (prismatic tolerance) constrains the maximum permissible
        deviation of the surface normal from the mechanical reference axis, stated
        in arc minutes.  When not specified, returns ``""`` (empty — not a numbered
        ISO code; shown inline with the centration row when present).

        Example::

            pyramid_error=2.5  →  "pyr 2.5′"
        """
        if self.pyramid_error is None:
            return ""
        v = self.pyramid_error
        v_str = str(int(v)) if v == int(v) else str(v)
        return f"pyr {v_str}\u2032"  # U+2032 prime for arc-minutes

    def fmt_roughness(self) -> str:
        """Return the ISO ``8/`` surface texture code string (ISO 10110-8:2019).

        Unlike the other ``fmt_*`` methods this returns ``""`` (empty string)
        when *roughness* is ``None``, because the 8/ code is optional — not
        every surface needs a texture specification.

        Valid *roughness* designators (ISO 10110-8:2019 §5):

        - ``"P"``   — polished, no quantitative spec (§5.3.1).
        - ``"P1"`` / ``"P2"`` / ``"P3"`` / ``"P4"`` — polish grade (§5.3.2,
          Table 1): P1 ≤ 8 nm, P2 ≤ 4 nm, P3 ≤ 2 nm, P4 ≤ 1 nm Rq over
          the spatial band 0.002–1.0 mm.
        - ``"G"``   — matte/ground surface (§5.2).
        - ``"-1/Rq 0.002"``  — polished with upper spatial-band limit 1 mm
          and Rq ≤ 0.002 μm (§5.3.3).
        - ``"0.002-1/Rq 0.002"``  — polished with full spatial band (§5.3.3).

        Examples::

            roughness="P"              → "8/ P"
            roughness="P2"             → "8/ P2"
            roughness="-1/Rq 0.002"   → "8/ -1/Rq 0.002"
            roughness=None             → ""
        """
        if self.roughness is None:
            return ""
        return f"8/ {self.roughness}"

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: dict) -> SurfaceSpec:
        # Tolerant of stale/unknown keys from previously-saved YAML files —
        # unlike direct API calls (set_surface/set_element), which forbid
        # unknown keywords so typos raise immediately.
        return cls(**{k: v for k, v in d.items() if k in cls.model_fields})


# ---------------------------------------------------------------------------
# Element-level specification
# ---------------------------------------------------------------------------


class ElementSpec(BaseModel):
    """ISO 10110 fabrication specifications for a lens element (bulk glass).

    Bulk-glass codes
    ----------------
    - *birefringence*: maximum stress birefringence in nm/cm (ISO 10110-2, ``0/``).
    - *bubbles*: bubble-and-inclusion string, e.g. ``"1×0.16"`` → ``1/ 1×0.16``
      (ISO 10110-3).
    - *homogeneity_grade*: legacy refractive-index homogeneity class 0–5 (ISO 10110-4).
      Prefer *nh_class* for contemporary drawings.
    - *nh_class*: contemporary homogeneity designator per ISO 10110-18:2019,
      e.g. ``"NH040"``.  Preferred values (ISO 10110-18:2019 Table 2):
      NH100, NH040, NH010, NH004, NH002, NH001.
      Legacy values also accepted: NH40, NH20, NH10, NH5, NH2, NH1, NH0.5,
      NH0.2, NH0.1.  When set, takes precedence over *homogeneity_grade*
      in the ``2/`` code.
    - *striae_grade*: legacy striae class string (e.g. ``"A"``), kept for
      backward compatibility.  Prefer *striae_density* or *striae_shadowgraph*.
    - *striae_density*: striae density class 1–5 (ISO 10110-4, density method).
    - *striae_shadowgraph*: striae shadowgraph class A–D (ISO 10110-4,
      shadowgraph method).

    Dimensions
    ----------
    - *diameter*: physical edge diameter of the element in mm.
    - *diameter_tolerance*: symmetric ± tolerance on the physical diameter (mm).
    - *ct_tolerance*: symmetric ± tolerance on the centre thickness (mm).

    Title-block fields
    ------------------
    - *part_number*, *revision*, *drawn_by*, *approved_by*, *notes*.
    """

    model_config = ConfigDict(extra="forbid")

    # ── Bulk glass codes ─────────────────────────────────────────────────
    birefringence: float | None = None
    bubbles: str | None = None
    # Homogeneity — legacy int (0-5) or contemporary NH class
    homogeneity_grade: int | None = None
    nh_class: str | None = None
    # Striae — legacy string or typed density/shadowgraph fields
    striae_grade: str | None = None  # kept for backward compat
    striae_density: int | None = None  # 1–5 (density method)
    striae_shadowgraph: str | None = None  # A–D (shadowgraph method)

    # ── Title block ───────────────────────────────────────────────────────
    part_number: str = ""
    revision: str = "A"
    drawn_by: str = ""
    approved_by: str = ""
    notes: str = ""

    # ── Dimensions ───────────────────────────────────────────────────────
    diameter: float | None = None  # physical edge Ø (mm)
    diameter_tolerance: str | float | None = None  # ± tol on Ø: "±0.05" or bare 0.05
    ct_tolerance: str | float | None = None  # ± tol on CT: "±0.10" or bare 0.10

    # ── Matched-pair assembly (ISO 10110-1 §5.11.3) ──────────────────────
    matched_pair: bool = False  # True → "M" marker on CT dimension

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    # Numeric string coercion (e.g. values arriving as strings from YAML) is
    # handled automatically by pydantic's field typing; this validator only
    # covers ISO-specific business rules that span more than one field.

    @model_validator(mode="after")
    def _validate(self) -> ElementSpec:
        if self.birefringence is not None and self.birefringence < 0:
            raise ValueError(f"birefringence must be ≥ 0, got {self.birefringence}")

        # Normalise ASCII 'x' → '×' in bubbles string (ISO multiplication sign)
        if self.bubbles is not None:
            self.bubbles = _fix_mul(self.bubbles)
            _r5_advisory(self.bubbles, "1/ bubbles")

        if self.nh_class is not None and self.nh_class not in NH_CLASSES:
            raise ValueError(
                f"nh_class '{self.nh_class}' is not a valid NH designator. "
                f"Valid values: {sorted(NH_CLASSES)}"
            )

        sd = self.striae_density
        if sd is not None and not (1 <= sd <= 5):
            raise ValueError(f"striae_density must be 1–5, got {sd}")

        if self.striae_shadowgraph is not None:
            if self.striae_shadowgraph.upper() not in {"A", "B", "C", "D"}:
                raise ValueError(
                    f"striae_shadowgraph must be A–D, got {self.striae_shadowgraph}"
                )
            self.striae_shadowgraph = self.striae_shadowgraph.upper()

        hg = self.homogeneity_grade
        if hg is not None and not (0 <= hg <= 5):
            raise ValueError(f"homogeneity_grade must be 0–5, got {hg}")

        return self

    # ------------------------------------------------------------------
    # ISO formatted strings
    # ------------------------------------------------------------------

    def fmt_birefringence(self) -> str:
        """Return ``0/ A`` where A is max stress birefringence in nm/cm.

        Per ISO 10110-2 §5.2: ``0/ A`` where A is the OPD in nm/cm.
        Integer-valued specs are shown without a trailing ``.0``
        (e.g. ``0/ 5 nm/cm`` rather than ``0/ 5.0 nm/cm``).
        """
        if self.birefringence is None:
            return "0/ —"
        v = self.birefringence
        v_str = str(int(v)) if v == int(v) else str(v)
        return f"0/ {v_str} nm/cm"

    def fmt_bubbles(self) -> str:
        if self.bubbles is None:
            return "1/ —"
        return "1/ " + _fix_mul(self.bubbles)

    def fmt_homogeneity(self) -> str:
        """Return the ISO ``2/`` code string.

        Homogeneity part: uses *nh_class* (e.g. ``NH20``) when set, falls back
        to legacy *homogeneity_grade* integer (0–5).

        Striae part: uses *striae_density* (1–5) or *striae_shadowgraph* (A–D)
        when set; falls back to legacy *striae_grade* string.

        Format: ``2/ <homogeneity>; <striae>``
        """
        # Homogeneity half
        if self.nh_class is not None:
            h = self.nh_class
        elif self.homogeneity_grade is not None:
            # Map legacy integer grade to ISO 10110-18:2019 NH class designator
            h = _LEGACY_NH_MAP.get(self.homogeneity_grade, str(self.homogeneity_grade))
        else:
            h = None

        # Striae half — typed fields take priority over legacy string
        if self.striae_density is not None:
            s = str(self.striae_density)
        elif self.striae_shadowgraph is not None:
            s = self.striae_shadowgraph
        elif self.striae_grade is not None:
            s = self.striae_grade
        else:
            s = None

        if h is None and s is None:
            return "2/ —"
        h_str = h if h is not None else "—"
        s_str = s if s is not None else "—"
        return f"2/ {h_str}; {s_str}"

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: dict) -> ElementSpec:
        # Tolerant of stale/unknown keys from previously-saved YAML files —
        # unlike direct API calls (set_element/set_title), which forbid
        # unknown keywords so typos raise immediately.
        return cls(**{k: v for k, v in d.items() if k in cls.model_fields})
