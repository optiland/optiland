"""ISO 10110 shared geometry, tolerance-formatting, and layout constants.

Everything here is format-agnostic: coordinate mapping (:class:`_Geo`),
tolerance string parsing, ISO 10110 spec-table row assembly, and layout
constants shared by both the matplotlib and DXF renderers (see
``_base_renderer.py``, ``_mpl_renderer.py``, ``_dxf_renderer.py``).
"""

from __future__ import annotations

import math

import numpy as np

import optiland.backend as be

PAPER_SIZES: dict[str, tuple[float, float]] = {
    "A5": (148.0, 210.0),
    "A4": (210.0, 297.0),
    "A3": (297.0, 420.0),
    "A2": (420.0, 594.0),
    "A1": (594.0, 841.0),
    "LETTER": (215.9, 279.4),
    "LEGAL": (215.9, 355.6),
    "TABLOID": (279.4, 431.8),
}


def _paper_wh(paper: str, orientation: str) -> tuple[float, float]:
    key = paper.upper()
    if key not in PAPER_SIZES:
        raise ValueError(f"Unknown paper '{paper}'. Choose from {list(PAPER_SIZES)}")
    short, long = PAPER_SIZES[key]
    if orientation.lower() == "portrait":
        return short, long
    if orientation.lower() == "landscape":
        return long, short
    raise ValueError("orientation must be 'portrait' or 'landscape'")


BORDER = 10.0  # inner border margin
TITLE_H = 55.0  # title-block strip height
MAT_W = 110.0  # material-table cell width inside title block

# Shared layout heights used by both renderers
_SPEC_H = 72.0  # ISO specification table height (up to 12 rows per surface column)
_TTL_R1_H = 15.0  # title-block main row height
_TTL_R2_H = 13.0  # title-block notes / standards row height
_TTL_H = _TTL_R1_H + _TTL_R2_H  # 28 mm total
_BOT_H = _SPEC_H + _TTL_H  # 83 mm total bottom section


_SYM_TRI = [(0.000000, 0.000000), (-19.255406, 33.351343), (19.255404, 33.351343)]
_SYM_ARM = [
    (0.000000, 0.000000),
    (12.894160, 22.281294),
    (25.779242, 44.654052),
    (38.596664, 67.080966),
]
_SYM_BAR = [(38.147170, 66.827652), (61.857127, 66.827652)]
_SYM_SC = 0.115  # scale factor: SVG units → mm


def _sym_transform(vx, vy, cos_t, sin_t, pts):
    """Rotate + scale + translate SVG symbol points to sheet mm."""
    return [
        (
            vx + _SYM_SC * (x * cos_t - y * sin_t),
            vy + _SYM_SC * (x * sin_t + y * cos_t),
        )
        for x, y in pts
    ]


# Encircled-λ symbol for coating rows in spec table (ISO 10110-9)
_LAMBDA_R = 1.5  # circle radius in sheet mm


def _mat_display_name(mat) -> str:
    """Return ISO 10110 glass name with catalog/manufacturer in parentheses.

    Prefers the explicitly-set *reference* attribute; falls back to the
    ``reference`` key in ``material_data`` (populated from the database CSV).
    The catalog name is omitted when it is identical to the glass name
    (case-insensitive) to avoid redundant output like ``F2 (F2)``.
    """
    name = getattr(mat, "name", mat.__class__.__name__)
    ref: str | None = getattr(mat, "reference", None)
    if ref is None:
        md = getattr(mat, "material_data", None)
        if isinstance(md, dict):
            ref = md.get("reference")
    if ref and ref.lower() != name.lower():
        return f"{name} ({ref})"
    return name


# ── Shared spec-table assembly (ISO 10110-1 Table 1 code order) ─────────────
#
# _MatplotlibRenderer and _DxfRenderer both build the same per-surface and
# per-material spec-table rows, just with different text formatting (mathtext
# vs. plain). The two helpers below hold the "which codes, in what order"
# logic exactly once, so a standards revision touching table order/contents
# is a single edit instead of two parallel ones (see ISO10110_improvements.md
# §U1 commit history — this duplication has already cost double-editing more
# than once).


def _numbered_code_rows(sspec, si: int, n: int) -> tuple[list[str], str, int]:
    """Return the numbered ISO 10110 code rows for one surface column.

    Covers 3/, 4/, 5/, 6/, 7/, 8/, 13/, 15/, and the pyramid-error row, in
    ISO 10110-1 Table 1 numerical order.

    Args:
        sspec: The surface's :class:`~optiland.iso10110.notation.SurfaceSpec`.
        si:    0-based index of this surface within the element.
        n:     Total number of surfaces in the element (for cement detection).

    Returns:
        A ``(lines, coating_str, coat_row_idx)`` tuple:

        - *lines*: row strings, in order. Includes a blank placeholder at
          *coat_row_idx* where the 7/ coating text belongs — the caller
          renders *coating_str* there itself, alongside the encircled-λ
          symbol (ISO 10110-9 §5.11.2), since that symbol is drawn very
          differently in matplotlib vs. DXF.
        - *coating_str*: the 7/ coating text (may be empty).
        - *coat_row_idx*: index into *lines* of the coating placeholder.
    """
    lines: list[str] = [
        sspec.fmt_form_error(),  # 3/
        sspec.fmt_centration(),  # 4/
        sspec.fmt_imperfections(),  # 5/
        sspec.fmt_laser_damage(),  # 6/
    ]
    # 7/ coating — after 6/ per ISO 10110-1 Table 1 numerical order
    coating_str = sspec.fmt_coating()
    coat_row_idx = len(lines)
    lines.append("")  # placeholder; caller draws coating_str + encircled-λ here
    # 8/ surface texture — after 7/ and before 13/ per ISO Table 1 order
    rough = sspec.fmt_roughness()
    if rough:
        lines.append(rough)
    lines.append(sspec.fmt_wavefront())  # 13/
    # 15/ assembly surface imperfections (ISO 10110-7 §5.5): relevant only for
    # cement interfaces. For outer surfaces, show the code only when the user
    # has explicitly set a value (not the default dash).
    is_cement_surf = 0 < si < n - 1
    if is_cement_surf or sspec.assembly_imperfections is not None:
        lines.append(sspec.fmt_assembly_imperfections())  # 15/
    # pyr σ' — pyramid error / prismatic tolerance (ISO 10110-6 §5.3)
    pyr = sspec.fmt_pyramid_error()
    if pyr:
        lines.append(pyr)
    return lines, coating_str, coat_row_idx


def _material_code_rows(comp_espec) -> list[str]:
    """Return the 0/, 1/, 2/ bulk-glass quality rows for one material column.

    Args:
        comp_espec: :class:`~optiland.iso10110.notation.ElementSpec` for this
            glass component (see ``DrawingSpec.get_material_spec``).
    """
    return [
        comp_espec.fmt_birefringence(),  # 0/
        comp_espec.fmt_bubbles(),  # 1/
        comp_espec.fmt_homogeneity(),  # 2/
    ]


# ── Geometry helpers ─────────────────────────────────────────────────────────


def _to_float(v) -> float:
    return float(np.asarray(be.to_numpy(be.array(v))))


def _surf_z(surface) -> float:
    return _to_float(surface.geometry.cs.z)


def _surf_r(surface) -> float:
    try:
        return _to_float(surface.geometry.radius)
    except AttributeError:
        return math.inf


def _parse_tol(tol) -> tuple[str, str]:
    """Return (hi_str, lo_str) from a tolerance specification."""
    MINUS = "\u2212"
    if tol is None:
        return "", ""
    s = str(tol).strip()
    if s.startswith("±"):
        v = s[1:]
        return f"+{v}", f"{MINUS}{v}"
    if "/" in s:
        hi, lo = s.split("/", 1)
        lo = lo.strip().lstrip("-")
        return hi.strip(), f"{MINUS}{lo}"
    try:
        v = abs(float(s))
        return f"+{v:.3f}", f"{MINUS}{v:.3f}"
    except ValueError:
        return s, ""


def _tol_math(tol) -> str:
    """Return a mathtext ``^{}_{}`  tolerance string, or ``""``."""
    if tol is None:
        return ""
    s = str(tol).strip()
    if s.startswith("±"):
        v = s[1:]
        return f"^{{+{v}}}_{{-{v}}}"
    if "/" in s:
        hi, lo = s.split("/", 1)
        lo = lo.strip().lstrip("-\u2212")
        return f"^{{{hi.strip()}}}_{{-{lo}}}"
    try:
        v = abs(float(s))
        return f"^{{+{v:.3f}}}_{{-{v:.3f}}}"
    except ValueError:
        return ""


def _surface_header_lines(
    surf,
    sspec,
    is_rear: bool = False,
    plain_text: bool = False,
) -> list[str]:
    """Return ISO 10110-1 Table 1 header lines for *surf* in the spec table.

    Covers standard, aspheric, cylindrical, toroidal, biconic, general-surface,
    and grating geometries per §5.9.2 / Table 1.

    Args:
        surf:       Surface object with a ``geometry`` attribute.
        sspec:      :class:`~optiland.iso10110.notation.SurfaceSpec` for this surface.
        is_rear:    ``True`` for the exit surface of the element (CX/CC sense flips).
        plain_text: When ``True`` the tolerance is formatted as ``+hi/-lo`` plain
                    text instead of matplotlib mathtext ``^{hi}_{lo}``.  Use for
                    DXF output.
    """
    from optiland.geometries import (
        BiconicGeometry,
        EvenAsphere,
        GridSagGeometry,
        NurbsGeometry,
        OddAsphere,
        PlaneGrating,
        PolynomialGeometry,
        StandardGratingGeometry,
        ToroidalGeometry,
        ZernikePolynomialGeometry,
    )
    from optiland.geometries.chebyshev import ChebyshevPolynomialGeometry
    from optiland.geometries.forbes import (
        ForbesQ2dGeometry,
        ForbesQNormalSlopeGeometry,
    )

    g = surf.geometry
    r_val = _surf_r(surf)

    # ── Grating (ISO 10110-1 §5.9.8 + ISO 10110-16 §5.3) ────────────────
    # First line: ISO 10110-16 Table 5 type symbol (LG, CG, CGH …) from
    # sspec.grating_type; second line: ISO 10110-1 surface identifier GRAT;
    # then (for curved substrates) R and κ, then grating frequency (lines/mm),
    # diffraction order, and groove orientation angle φ per §5.3 / Table 2.
    if isinstance(g, (PlaneGrating, StandardGratingGeometry)):
        gt = sspec.grating_type if sspec.grating_type else "LG"
        lines = [gt, "GRAT"]
        # Curved substrate (StandardGratingGeometry only): show R and κ so
        # the drawing conveys both the substrate shape and the grating params.
        if isinstance(g, StandardGratingGeometry):
            try:
                r_sub = _to_float(g.radius)
                if math.isinf(r_sub):
                    lines.append("R \u221e")  # ∞
                else:
                    r_sub_s = f"{r_sub:+.3f}"
                    if plain_text:
                        tol_p = _tol_plain(sspec.r_tolerance)
                        lines.append(f"R {r_sub_s}{tol_p}" if tol_p else f"R {r_sub_s}")
                    else:
                        tol_m = _tol_math(sspec.r_tolerance)
                        lines.append(
                            f"R ${r_sub_s}{tol_m}$" if tol_m else f"R {r_sub_s}"
                        )
                k_sub = _to_float(g.k)
                if k_sub != 0.0:
                    lines.append(f"\u03ba = {k_sub:+.4g}")  # κ = …
            except AttributeError:
                pass
        # ISO 10110-16 §5.3: grating frequency and diffraction order.
        # grating_period is in µm (both PlaneGrating and StandardGratingGeometry).
        try:
            period_um = _to_float(g.grating_period)  # µm
            freq_lpmm = 1000.0 / period_um  # lines/mm
            order = int(round(_to_float(g.grating_order)))
            lines.append(f"{freq_lpmm:.1f} l/mm")
            lines.append(f"m = {order:+d}" if order != 0 else "m = 0")
        except (AttributeError, ZeroDivisionError):
            pass
        # ISO 10110-16 Table 2: groove orientation angle φ (radians → degrees).
        # Omitted when zero (default: grooves perpendicular to x-axis).
        try:
            phi_rad = _to_float(g.groove_orientation_angle)
            phi_deg = math.degrees(phi_rad)
            if abs(phi_deg) > 1e-6:
                lines.append(f"\u03c6 = {phi_deg:.1f}\u00b0")  # φ = …°
        except AttributeError:
            pass
        return lines

    # ── General / freeform surfaces (GS) — non-rotationally-symmetric ────
    # ForbesQ2dGeometry and ChebyshevPolynomialGeometry have 2-D departures
    # from the conic base and are therefore freeform surfaces, not ASPH.
    if isinstance(
        g,
        (
            PolynomialGeometry,
            ZernikePolynomialGeometry,
            GridSagGeometry,
            NurbsGeometry,
            ForbesQ2dGeometry,
            ChebyshevPolynomialGeometry,
        ),
    ):
        return ["GS"]

    # ── Axially-symmetric aspheres (ASPH) ─────────────────────────────────
    if isinstance(g, (EvenAsphere, OddAsphere, ForbesQNormalSlopeGeometry)):
        r_str = "∞" if math.isinf(r_val) else f"{r_val:+.3f}"
        # Vertex radius — include r_tolerance when set (ISO 10110-12 §5.3.1)
        if plain_text:
            tol_p = _tol_plain(sspec.r_tolerance)
            r_line = f"R {r_str}{tol_p}" if tol_p else f"R {r_str}"
        else:
            tol_m = _tol_math(sspec.r_tolerance)
            r_line = f"R ${r_str}{tol_m}$" if tol_m else f"R {r_str}"
        lines: list[str] = ["ASPH", r_line]
        # Conic constant — ISO 10110-12 §4.3.1: κ shall be given when non-zero.
        # Omit for κ = 0 (spherical base — no conic contribution to declare).
        try:
            k_val = _to_float(g.k)
            if k_val != 0.0:
                lines.append(f"\u03ba = {k_val:+.4g}")  # κ = …
        except AttributeError:
            pass
        # Polynomial coefficients — ISO 10110-12 notation
        # EvenAsphere: C_i * r^{2i}, coefficients[i] → ISO A_{2(i+1)}
        # OddAsphere:  C_i * r^i,    coefficients[i] → ISO A_{i+1}
        is_odd = isinstance(g, OddAsphere)
        try:
            for i, ci in enumerate(g.coefficients):
                c_val = _to_float(ci)
                if c_val != 0.0:
                    exp = (i + 1) if is_odd else (2 * (i + 1))
                    lines.append(f"A{exp} = {c_val:+.4e}")
        except AttributeError:
            pass
        # Forbes Q polynomial coefficients: radial_terms dict {order m: a_m}
        # ISO 10110-12 has no dedicated Forbes notation; show ρmax + non-zero a_m.
        try:
            rt = g.radial_terms
            if rt:
                nr = _to_float(g.norm_radius)
                lines.append(f"\u03c1max = {nr:.3g}")  # ρmax
                for m in sorted(rt.keys()):
                    a_val = _to_float(rt[m])
                    if a_val != 0.0:
                        lines.append(f"a{m} = {a_val:+.4e}")
        except AttributeError:
            pass
        return lines

    # ── Toroidal ─────────────────────────────────────────────────────────
    if isinstance(g, ToroidalGeometry):
        r_rot = _to_float(g.R_rot)
        r_yz = _to_float(g.R_yz)
        if math.isinf(r_rot):
            # One infinite radius of curvature → cylindrical surface
            r_yz_s = "∞" if math.isinf(r_yz) else f"{r_yz:+.3f}"
            if math.isinf(r_yz):
                return ["CYL", f"R {r_yz_s}"]
            if plain_text:
                tol_p = _tol_plain(sspec.r_tolerance)
                lines = ["CYL", f"R {r_yz_s}{tol_p}" if tol_p else f"R {r_yz_s}"]
            else:
                tol_m = _tol_math(sspec.r_tolerance)
                lines = ["CYL", f"R ${r_yz_s}{tol_m}$" if tol_m else f"R {r_yz_s}"]
            # CYL base-curve conic constant κ (ISO 10110-12): show when non-zero
            try:
                k_cyl = _to_float(g.k_yz)
                if k_cyl != 0.0:
                    lines.append(f"\u03ba = {k_cyl:+.4g}")
            except AttributeError:
                pass
            return lines
        r_rot_s = f"{r_rot:+.3f}"
        r_yz_s = "∞" if math.isinf(r_yz) else f"{r_yz:+.3f}"
        lines = ["TOROID", f"Rx {r_rot_s}", f"Ry {r_yz_s}"]
        # TOROID base-curve conic constant κ (ISO 10110-12): show when non-zero
        try:
            k_tor = _to_float(g.k_yz)
            if k_tor != 0.0:
                lines.append(f"\u03ba = {k_tor:+.4g}")
        except AttributeError:
            pass
        return lines

    # ── Biconic ───────────────────────────────────────────────────────────
    if isinstance(g, BiconicGeometry):
        rx = _to_float(g.Rx)
        ry = _to_float(g.Ry)
        rx_s = "∞" if math.isinf(rx) else f"{rx:+.3f}"
        ry_s = "∞" if math.isinf(ry) else f"{ry:+.3f}"
        lines = ["BICONIC", f"Rx {rx_s}", f"Ry {ry_s}"]
        # ISO 10110-12 conic constants: show κx and κy when non-zero
        try:
            kx = _to_float(g.kx)
            if kx != 0.0:
                lines.append(f"\u03bax = {kx:+.4g}")
        except AttributeError:
            pass
        try:
            ky = _to_float(g.ky)
            if ky != 0.0:
                lines.append(f"\u03bay = {ky:+.4g}")
        except AttributeError:
            pass
        return lines

    # ── Standard: sphere or flat plane ───────────────────────────────────
    if math.isinf(r_val):
        return ["R ∞"]

    r_str = f"{r_val:+.3f}"
    # CX = convex towards the incoming light side; for the rear surface the
    # convention flips because we label from the exit side.
    cx_cc = ("CX" if r_val < 0 else "CC") if is_rear else ("CX" if r_val > 0 else "CC")
    if plain_text:
        tol_p = _tol_plain(sspec.r_tolerance)
        return [f"R {r_str}{tol_p} {cx_cc}"] if tol_p else [f"R {r_str} {cx_cc}"]
    tol_m = _tol_math(sspec.r_tolerance)
    return [f"R ${r_str}{tol_m}$ {cx_cc}"] if tol_m else [f"R {r_str} {cx_cc}"]


def _tol_plain(tol) -> str:
    """Plain-text tolerance suffix for DXF (no mathtext)."""
    if tol is None:
        return ""
    hi, lo = _parse_tol(tol)
    if not hi:
        return ""
    lo = lo.replace("\u2212", "-")
    return f" {hi}/{lo}"


def _sag_curve(surface, sa: float, n: int = 200) -> np.ndarray:
    y = np.linspace(-sa, sa, n)
    sag = np.asarray(be.to_numpy(surface.geometry.sag(be.zeros(n), be.array(y))))
    return np.column_stack([_surf_z(surface) + sag, y])


def _nice_scale(raw: float) -> float:
    prev = 0.25
    for s in [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 50]:
        if s > raw:
            return prev
        prev = s
    return 50.0


# ── ISO 128-50 short-long-short glass hatch helper ───────────────────────────


class _Geo:
    """Maps optical mm → sheet mm (origin = bottom-left of paper)."""

    def __init__(
        self,
        element,
        spec,
        pw: float,
        ph: float,
        bottom_h: float = _BOT_H,
        cal_x: float = 22.5,
        cal_y: float = 30.0,
        border_margin: float = BORDER,
    ) -> None:
        self.element = element
        self.spec = spec
        self.pw, self.ph = pw, ph
        self._bottom_h = bottom_h
        self._cal_x = cal_x
        self._cal_y = cal_y
        self._border = border_margin
        self._build()

    def _build(self):
        surfs = self.element.surfaces
        optic = self.spec.optic

        self.sa = [self.element.semi_aperture(i, optic) for i in range(len(surfs))]
        self.sa_max = max(self.sa)

        curves = [_sag_curve(s, sa) for s, sa in zip(surfs, self.sa, strict=True)]
        self.opt_z_min = min(c[:, 0].min() for c in curves)
        self.opt_z_max = max(c[:, 0].max() for c in curves)
        z_span = max(self.opt_z_max - self.opt_z_min, 0.5)
        y_span = 2.0 * self.sa_max

        aw = (self.pw - 2 * self._border) - 2 * self._cal_x
        ah = (self.ph - 2 * self._border - self._bottom_h) - 2 * self._cal_y

        raw = min(aw / z_span, ah / y_span)
        self.scale = _nice_scale(raw)

        cx = self.pw / 2.0
        cy = (
            self._border
            + self._bottom_h
            + (self.ph - 2 * self._border - self._bottom_h) / 2.0
        )

        self._ox = cx - ((self.opt_z_min + self.opt_z_max) / 2.0) * self.scale
        self._oy = cy

    def pt(self, z: float, y: float) -> tuple[float, float]:
        return (self._ox + z * self.scale, self._oy + y * self.scale)

    def curve(self, surf_idx: int, n: int = 200) -> np.ndarray:
        return self.curve_at(surf_idx, self.sa[surf_idx], n)

    def rim(self, surf_idx: int, sign: float) -> tuple[float, float]:
        return self.rim_at(surf_idx, sign, self.sa[surf_idx])

    def curve_at(self, surf_idx: int, sa_val: float, n: int = 200) -> np.ndarray:
        """Like :meth:`curve`, but for an explicit semi-aperture rather than
        the surface's own fitted ``self.sa[surf_idx]`` — used by both
        renderers to draw the shared outer envelope (``sa_max``) across all
        surfaces of an element, and for individual cement interfaces.
        """
        raw = _sag_curve(self.element.surfaces[surf_idx], sa_val, n)
        return np.column_stack(
            [
                self._ox + raw[:, 0] * self.scale,
                self._oy + raw[:, 1] * self.scale,
            ]
        )

    def rim_at(self, surf_idx: int, sign: float, sa_val: float) -> tuple[float, float]:
        """Like :meth:`rim`, for an explicit semi-aperture (see :meth:`curve_at`)."""
        surf = self.element.surfaces[surf_idx]
        sag = _to_float(surf.geometry.sag(be.zeros(1), be.array([sa_val]))[0])
        return self.pt(_surf_z(surf) + sag, sign * sa_val)

    def scale_label(self) -> str:
        sc = self.scale
        if sc >= 1:
            return f"{int(sc) if sc == int(sc) else sc}:1"
        return f"1:{int(round(1 / sc))}"


# ── ISO 10110-11 general tolerance defaults ──────────────────────────────────


def _iso10110_11_defaults(element_size: float) -> dict:
    """Return ISO 10110-11:2019 Table 1 default tolerances for *element_size*.

    Args:
        element_size: Largest dimension of the element in mm.  For lenses this
            is the physical outer diameter; for prisms use the diagonal.
            ISO 10110-11 §4.1: "The default tolerances are determined by the
            largest dimension of the element."
    """
    diagonal = element_size  # backward-compat alias inside the function
    if diagonal <= 10:
        return dict(
            diameter=0.1,
            thickness=0.1,
            form_error="3/ 5",
            centration="4/ 3\u2032",
            imperfections="5/ 2\u00d70.04",
        )
    if diagonal <= 30:
        return dict(
            diameter=0.1,
            thickness=0.15,
            form_error="3/ 5",
            centration="4/ 3\u2032",
            imperfections="5/ 2\u00d70.063",
        )
    if diagonal <= 100:
        return dict(
            diameter=0.2,
            thickness=0.2,
            form_error="3/ 5",
            centration="4/ 5\u2032",
            imperfections="5/ 3\u00d70.10",
        )
    return dict(
        diameter=0.5,
        thickness=0.5,
        form_error="3/ 10",
        centration="4/ 5\u2032",
        imperfections="5/ 5\u00d70.16",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# _BaseRenderer — shared layout logic, expressed via format-specific primitives
# ═══════════════════════════════════════════════════════════════════════════════
