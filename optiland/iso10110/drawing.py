"""ISO 10110 Element Drawing Generator

Rendering architecture
----------------------
Two isolated renderer classes handle format-specific output
(HarrisonKramer's architectural feedback):

* :class:`_MatplotlibRenderer` — PDF / PNG via matplotlib.
* :class:`_DxfRenderer`        — DXF via ezdxf.

:class:`ElementDrawing` is a thin public coordinator that delegates to
whichever renderer ``save_pdf`` / ``save_dxf`` / ``save_png`` / ``show``
require.  Neither renderer class is part of the public API.

Paper / orientation
-------------------
``paper`` : "A5","A4","A3","A2","A1","LETTER","LEGAL","TABLOID"
``orientation`` : "portrait" (default) | "landscape"

Bernhard Lutzer, 2026
"""

from __future__ import annotations

import math
from datetime import date
from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.iso10110.style import DrawingStyle

if TYPE_CHECKING:
    from pathlib import Path

    import ezdxf


# ezdxf (the "manufacturing" extra) is imported lazily, only where DXF output
# is actually produced (ElementDrawing.generate(), dxf_to_png()) — not here at
# module scope — so that `import optiland.iso10110` and PDF/PNG-only usage
# (save_pdf, save_png, show) work without ezdxf installed. _DxfRenderer itself
# only operates on an already-constructed ezdxf document/modelspace passed in
# by the caller, so it needs no import of its own either.


# ── Paper sizes (short_mm, long_mm) ─────────────────────────────────────────
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


# ── Layout constants (mm, independent of paper size) ────────────────────────
BORDER = 10.0  # inner border margin
TITLE_H = 55.0  # title-block strip height
MAT_W = 110.0  # material-table cell width inside title block

# Shared layout heights used by both renderers
_SPEC_H = 72.0  # ISO specification table height (up to 12 rows per surface column)
_TTL_R1_H = 15.0  # title-block main row height
_TTL_R2_H = 13.0  # title-block notes / standards row height
_TTL_H = _TTL_R1_H + _TTL_R2_H  # 28 mm total
_BOT_H = _SPEC_H + _TTL_H  # 83 mm total bottom section

# DXF layers
L_BORDER = "BORDER"
L_OUTLINE = "OUTLINE"
L_AXIS = "AXIS"
L_CEMENT = "CEMENT"
L_DIMS = "DIMS"
L_CALLOUT = "CALLOUT"
L_TABLE = "TABLE"

# Callout symbol geometry (SVG normalised units → sheet mm via _SYM_SC)
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


# ── Material display helpers ─────────────────────────────────────────────────


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


def _dxf_arrowhead(
    msp,
    tip: tuple[float, float],
    toward: tuple[float, float],
    layer: str,
    arr_len: float = 2.5,
    arr_w: float = 0.8,
) -> None:
    """Draw a filled DXF arrowhead (SOLID triangle) at *tip*, pointing away
    from *toward*.

    The arrow tip is at *tip*; the base is placed ``arr_len`` mm behind it in the
    direction toward → tip.  The base is ``2 × arr_w`` mm wide.

    Args:
        msp:     ezdxf modelspace.
        tip:     Tip of the arrowhead (x, y).
        toward:  Reference point on the dimension line body side (the vector
                 ``toward → tip`` gives the arrow direction).
        layer:   DXF layer for the SOLID entity.
        arr_len: Length of the arrowhead in mm (default 2.5).
        arr_w:   Half-width of the arrowhead base in mm (default 0.8).
    """
    import math as _m

    dx, dy = tip[0] - toward[0], tip[1] - toward[1]
    dist = _m.hypot(dx, dy)
    if dist < 1e-9:
        return
    ux, uy = dx / dist, dy / dist  # unit vector: toward → tip
    px, py = -uy, ux  # perpendicular unit vector
    # Base centre: arr_len behind the tip
    bx = tip[0] - arr_len * ux
    by = tip[1] - arr_len * uy
    p1 = (bx + arr_w * px, by + arr_w * py)
    p2 = (bx - arr_w * px, by - arr_w * py)
    # DXF SOLID: 4 vertices; p2 repeated so the quad degenerates to a triangle.
    # ezdxf vertex order for SOLID is 1, 2, 4, 3 → pass [tip, p1, p2, p2].
    msp.add_solid([tip, p1, p2, p2], dxfattribs={"layer": layer})


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


def _draw_glass_hatch(
    ax,
    outline: np.ndarray,
    direction: int = 1,
    hsp: float = 3.0,
    color: str = "#6090b0",
) -> None:
    """Draw ISO 128-50 long-short-long optical-glass hatch clipped to *outline*.

    The linestyle ``(0, (7, 2, 2, 2))`` produces the long-short-long (CENTER)
    pattern per ISO 128-50, with alternating 45°/135° directions for each
    component of a cemented lens (ISO 128-50 §4.2.3).

    Args:
        ax:        Matplotlib axes.
        outline:   (N, 2) array of sheet-mm vertices (closed polygon).
        direction: +1 for 45° hatch, -1 for 135° hatch (alternating components).
        hsp:       Line spacing in mm.
        color:     Hatch colour.
    """
    import matplotlib.patches as mpatches
    from matplotlib.path import Path as MPath

    codes = [MPath.MOVETO] + [MPath.LINETO] * (len(outline) - 1) + [MPath.CLOSEPOLY]
    clip_patch = mpatches.PathPatch(
        MPath(np.vstack([outline, outline[:1]]), codes), transform=ax.transData
    )

    xmn = outline[:, 0].min()
    xmx = outline[:, 0].max()
    ymn = outline[:, 1].min()
    ymx = outline[:, 1].max()
    x0 = xmn - hsp
    x1 = xmx + hsp

    if direction >= 0:  # 45°: y = x + c
        c_range = np.arange(ymn - xmx - hsp, ymx - xmn + hsp, hsp)
        for c in c_range:
            (ln,) = ax.plot(
                [x0, x1],
                [x0 + c, x1 + c],
                color=color,
                lw=0.4,
                zorder=3,
                linestyle=(0, (7, 2, 2, 2)),
                solid_capstyle="butt",
            )
            ln.set_clip_path(clip_patch)
    else:  # 135°: y = –x + c
        c_range = np.arange(ymn + xmn - hsp, ymx + xmx + hsp, hsp)
        for c in c_range:
            (ln,) = ax.plot(
                [x0, x1],
                [c - x0, c - x1],
                color=color,
                lw=0.4,
                zorder=3,
                linestyle=(0, (7, 2, 2, 2)),
                solid_capstyle="butt",
            )
            ln.set_clip_path(clip_patch)


# ── _Geo : optical → sheet coordinate mapping ────────────────────────────────


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


class _BaseRenderer:
    """Shared ISO 10110 layout logic for both output-format renderers.

    :class:`_MatplotlibRenderer` and :class:`_DxfRenderer` both subclass this
    and implement the small set of format-specific primitives below
    (``_prim_*`` line/circle/text drawing, plus ``_tol_fmt``/``_fmt_line``
    text-formatting hooks); layout expressed purely in terms of those
    primitives — e.g. the ISO spec table — lives here once instead of being
    duplicated per format. A future renderer (e.g. SVG) only needs to
    implement the primitives, not re-derive the layout.

    Concrete subclasses must set ``self._geo``, ``self.element``,
    ``self.spec``, ``self.style``, ``self.pw``/``self.ph`` in ``__init__``
    (as both already do), and must set the primitives' actual draw target
    (``self._ax`` for matplotlib, ``self._msp`` for DXF) before calling any
    shared layout method.
    """

    # ── primitives (implemented per format) ────────────────────────────────
    #
    # ``role`` selects the format-specific styling (matplotlib line
    # weight/zorder/color, or DXF layer) for that element — it is not a
    # layout decision, so it never affects position/content, only how a
    # primitive draws itself.

    def _prim_rect(self, x: float, y: float, w: float, h: float, *, role: str) -> None:
        """Unfilled axis-aligned rectangle outline."""
        raise NotImplementedError

    def _prim_hline(self, x0: float, x1: float, y: float, *, role: str) -> None:
        raise NotImplementedError

    def _prim_vline(self, x: float, y0: float, y1: float, *, role: str) -> None:
        raise NotImplementedError

    def _prim_line(
        self, p1: tuple[float, float], p2: tuple[float, float], *, role: str
    ) -> None:
        """Arbitrary two-point line (unlike ``_prim_hline``/``_prim_vline``,
        may carry a non-solid linestyle depending on *role*)."""
        raise NotImplementedError

    def _prim_dim_arrow(
        self, p1: tuple[float, float], p2: tuple[float, float], *, role: str
    ) -> None:
        """Double-headed dimension arrow from *p1* to *p2* (witness lines at
        the endpoints are drawn separately, via ``_prim_vline``/``_prim_hline``)."""
        raise NotImplementedError

    def _prim_polygon(self, points: list[tuple[float, float]], *, role: str) -> None:
        """Closed, unfilled polygon outline."""
        raise NotImplementedError

    def _prim_curve(self, points: list[tuple[float, float]], *, role: str) -> None:
        """Open curve through *points* (start, ...control points..., end).

        Matplotlib renders it as a cubic Bezier through all points; DXF (no
        native Bezier TEXT/LINE primitive here) draws a straight line from
        the first to the last point — a pre-existing per-format rendering
        difference, not a shared layout decision.
        """
        raise NotImplementedError

    def _prim_text(
        self, pos: tuple[float, float], s: str, *, ha: str, va: str, role: str
    ) -> None:
        raise NotImplementedError

    def _prim_circle(self, center: tuple[float, float], r: float, *, role: str) -> None:
        raise NotImplementedError

    def _tol_fmt(self, tol) -> str:
        """Format a tolerance string in this renderer's text representation
        (matplotlib mathtext vs. DXF plain text)."""
        raise NotImplementedError

    def _fmt_line(self, s: str) -> str:
        """Format-specific post-processing of a spec-table text line.

        Identity for matplotlib; DXF strips mathtext markers left over from
        ``_surface_header_lines(..., plain_text=True)`` and replaces the
        infinity symbol for broader DXF-viewer compatibility.
        """
        return s

    # ── shared layout ───────────────────────────────────────────────────────

    def _draw_borders(self) -> None:
        pw, ph = self.pw, self.ph
        self._prim_rect(0, 0, pw, ph, role="sheet_border_outer")
        self._prim_rect(
            self.style.border_margin,
            self.style.border_margin,
            pw - 2 * self.style.border_margin,
            ph - 2 * self.style.border_margin,
            role="sheet_border_inner",
        )

    def _draw_axes(self) -> None:
        geo = self._geo
        s = geo.scale
        surfs = self.element.surfaces
        n = len(surfs)

        mg = self._axis_margin_mm() / s
        ax_x0, ax_y0 = geo.pt(geo.opt_z_min - mg, 0)
        ax_x1, _ = geo.pt(geo.opt_z_max + mg, 0)

        # Optical axis — ISO 128 long-dash double-dot (PHANTOM line)
        self._prim_line((ax_x0, ax_y0), (ax_x1, ax_y0), role="optical_axis")

        if self.rotational_axis_y == 0.0:
            # Axes coincide — label once as "opt./rot. axis"
            self._prim_text(
                (ax_x0 - 1.0, ax_y0),
                "opt./rot. axis",
                ha="right",
                va="center",
                role="axis_label",
            )
        else:
            # Separate rotational axis — ISO 128 long-dash single-dot (DASHDOT)
            self._prim_text(
                (ax_x0 - 1.0, ax_y0),
                "opt. axis",
                ha="right",
                va="center",
                role="axis_label",
            )
            rot_x0, rot_y0 = geo.pt(geo.opt_z_min - mg, self.rotational_axis_y)
            rot_x1, _ = geo.pt(geo.opt_z_max + mg, self.rotational_axis_y)
            self._prim_line((rot_x0, rot_y0), (rot_x1, rot_y0), role="rotational_axis")
            self._prim_text(
                (rot_x0 - 1.0, rot_y0),
                "rot. axis",
                ha="right",
                va="center",
                role="axis_label",
            )

        # Surface labels for cement interfaces below the axis
        # (front / rear labels are placed near their callout symbols below)
        for si in range(1, n - 1):
            lbl_x = geo.pt(_surf_z(surfs[si]), 0)[0]
            self._prim_text(
                (lbl_x, ax_y0 - 2.5),
                f"S{si + 1}",
                ha="center",
                va="top",
                role="axis_surface_label",
            )

    def _draw_sharp_edge_symbols(self) -> None:
        # A "0" near the surface vertex (§5.9.5.2) signals that no protective
        # chamfer is permitted; the edge must remain sharp.
        geo = self._geo
        surfs = self.element.surfaces
        for si, surf in enumerate(surfs):
            s_idx = self.element.surface_indices[si]
            sspec = self.spec.get_surface_spec(s_idx)
            if not sspec.sharp_edge:
                continue
            vx_ax, vy_ax = geo.pt(_surf_z(surf), 0)
            self._prim_text(
                (vx_ax, vy_ax - 3.5),
                "0",
                ha="center",
                va="top",
                role="sharp_edge_symbol",
            )

    def _draw_aperture_brackets(self) -> None:
        # Effective aperture brackets (ISO 10110-11) — always shown.
        geo = self._geo
        surfs = self.element.surfaces
        n = len(surfs)
        for si, surf in enumerate(surfs):
            if si not in (0, n - 1):
                continue
            s_idx = self.element.surface_indices[si]
            sspec = self.spec.get_surface_spec(s_idx)
            # Fall back to geometric semi-aperture when not explicitly set
            ca_d = (
                sspec.ca_diameter if sspec.ca_diameter is not None else 2.0 * geo.sa[si]
            )
            ca_r = ca_d / 2.0
            sag_ca = _to_float(surf.geometry.sag(be.zeros(1), be.array([ca_r]))[0])
            ca_x = geo.pt(_surf_z(surf) + sag_ca, 0)[0]
            ca_y_top = geo.pt(0, ca_r)[1]
            ca_y_bot = geo.pt(0, -ca_r)[1]
            tick_len = 3.0
            side = -1 if si == 0 else 1
            # §5.6: test-zone boundary must be thin solid line (01.1)
            tick_x1 = ca_x + side * tick_len
            self._prim_hline(ca_x, tick_x1, ca_y_top, role="aperture_bracket")
            self._prim_hline(ca_x, tick_x1, ca_y_bot, role="aperture_bracket")
            self._prim_vline(tick_x1, ca_y_bot, ca_y_top, role="aperture_bracket")
            label_x = tick_x1 + side * 1.0
            label_ha = "right" if si == 0 else "left"
            self._prim_text(
                (label_x, (ca_y_top + ca_y_bot) / 2),
                f"Øₑ {ca_d:.2f}",
                ha=label_ha,
                va="center",
                role="aperture_bracket_label",
            )

    def _draw_surface_finish_callouts(self, sa_e: float) -> None:
        geo = self._geo
        surfs = self.element.surfaces
        n = len(surfs)

        sym_fracs = {0: 0.85, n - 1: 0.65}

        for si, frac in sym_fracs.items():
            surf = surfs[si]
            y_pos = frac * sa_e
            sag_v = _to_float(surf.geometry.sag(be.zeros(1), be.array([y_pos]))[0])
            vx, vy = geo.pt(_surf_z(surf) + sag_v, y_pos)

            r_val = _surf_r(surf)
            if math.isinf(r_val):
                nx, ny = (-1.0, 0.0) if si == 0 else (1.0, 0.0)
            else:
                abs_r = abs(r_val)
                cos_a = math.sqrt(max(abs_r**2 - y_pos**2, 0.0)) / abs_r
                sin_a = math.copysign(1.0, r_val) * y_pos / abs_r
                nx, ny = (-cos_a, sin_a) if si == 0 else (cos_a, -sin_a)

            theta = math.atan2(-nx, ny)
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)

            def _tp(pts, _vx=vx, _vy=vy, _ct=cos_t, _st=sin_t):
                return _sym_transform(_vx, _vy, _ct, _st, pts)

            # Triangle
            self._prim_polygon(_tp(_SYM_TRI), role="surface_finish_symbol")

            # Bezier arm (matplotlib) / straight-line approximation (DXF)
            self._prim_curve(_tp(_SYM_ARM), role="surface_finish_symbol")

            # Horizontal bar
            bar_pts = _tp(_SYM_BAR)
            self._prim_line(bar_pts[0], bar_pts[1], role="surface_finish_symbol")

            # Surface label next to bar end
            bx, by = bar_pts[1]
            self._prim_text(
                (bx + 1.5 * cos_t, by + 1.5 * sin_t),
                f"S{si + 1}",
                ha="center",
                va="center",
                role="surface_finish_label",
            )

    def _draw_spec_table(self) -> None:
        pw = self.pw
        geo = self._geo
        surfs = self.element.surfaces
        n = len(surfs)

        n_comps = self.element.num_components
        mats = self.element.glass_materials
        n_cols = n + n_comps

        tbl_y0 = self.style.border_margin + _TTL_H
        tbl_y1 = tbl_y0 + _SPEC_H
        tbl_x0 = self.style.border_margin
        tbl_x1 = pw - self.style.border_margin
        tbl_w = tbl_x1 - tbl_x0
        col_w = tbl_w / n_cols
        H_HDR = 7.0

        self._prim_rect(tbl_x0, tbl_y0, tbl_w, _SPEC_H, role="table")
        for ci in range(1, n_cols):
            xd = tbl_x0 + ci * col_w
            self._prim_vline(xd, tbl_y0, tbl_y1, role="table")
        self._prim_hline(tbl_x0, tbl_x1, tbl_y1 - H_HDR, role="table")

        def _col_hdr(ci: int, txt: str) -> None:
            cx = tbl_x0 + (ci + 0.5) * col_w
            self._prim_text(
                (cx, tbl_y1 - H_HDR / 2),
                txt,
                ha="center",
                va="center",
                role="table_header",
            )

        def _col_body(ci: int, lines: list[str]) -> None:
            x0 = tbl_x0 + ci * col_w + 2.0
            body = _SPEC_H - H_HDR
            sp = min(body / max(len(lines), 1), 6.5)
            for j, txt in enumerate(lines):
                self._prim_text(
                    (x0, tbl_y1 - H_HDR - 2.0 - j * sp),
                    txt,
                    ha="left",
                    va="top",
                    role="table_body",
                )

        # Surface columns (even indices)
        for si in range(n):
            ci = si * 2
            s_idx = self.element.surface_indices[si]
            sspec = self.spec.get_surface_spec(s_idx)
            if si == 0:
                hdr = "SURFACE 1 (FRONT)"
            elif si == n - 1:
                hdr = f"SURFACE {n} (REAR)"
            else:
                hdr = f"SURFACE {si + 1} (CEMENT)"
            _col_hdr(ci, hdr)

            # ── Row 1: surface type + radius (ISO Table 1 order) ─────────
            lines: list[str] = [
                self._fmt_line(ln)
                for ln in _surface_header_lines(
                    surfs[si], sspec, is_rear=(si == n - 1), plain_text=self._plain_text
                )
            ]

            # ── Row 2: Øe ────────────────────────────────────────────────
            _ca_d = (
                sspec.ca_diameter if sspec.ca_diameter is not None else 2.0 * geo.sa[si]
            )
            _ca_tol_m = (
                self._tol_fmt(sspec.ca_tolerance)
                if sspec.ca_diameter is not None
                else ""
            )
            lines.append(
                f"Øₑ ${_ca_d:.2f}{_ca_tol_m}$" if _ca_tol_m else f"Øₑ {_ca_d:.2f}"
            )

            # ── Row 3: Schutzfase (protective chamfer, §5.9.5.4) ────────
            if sspec.chamfer is not None:
                ang = sspec.chamfer_angle if sspec.chamfer_angle is not None else 45
                lines.append(f"Schutzfase {sspec.chamfer} × {ang}°")

            # ── Numbered codes in ISO Table 1 order: 3/, 4/, 5/, 6/, 7/, 8/ ──
            _base_len = len(lines)
            _code_lines, coating_str, _coat_idx = _numbered_code_rows(sspec, si, n)
            lines += _code_lines
            coat_row_idx = _base_len + _coat_idx

            _col_body(ci, lines)

            # Encircled-λ at the coating row (ISO 10110-9, §5.11.2)
            if coating_str:
                _sp_coat = min((_SPEC_H - H_HDR) / max(len(lines), 1), 6.5)
                _coat_y = tbl_y1 - H_HDR - 2.0 - coat_row_idx * _sp_coat
                _coat_x = tbl_x0 + ci * col_w + 2.0
                self._prim_circle(
                    (_coat_x + _LAMBDA_R, _coat_y - _LAMBDA_R),
                    _LAMBDA_R,
                    role="lambda_symbol",
                )
                self._prim_text(
                    (_coat_x + _LAMBDA_R, _coat_y - _LAMBDA_R),
                    "λ",
                    ha="center",
                    va="center",
                    role="lambda_symbol",
                )
                self._prim_text(
                    (_coat_x + 2 * _LAMBDA_R + 1.0, _coat_y),
                    coating_str,
                    ha="left",
                    va="top",
                    role="table_body_coating",
                )

        # Material columns (odd indices)
        for mi in range(n_comps):
            ci = mi * 2 + 1
            mat = mats[mi]
            display_name = _mat_display_name(mat)
            try:
                nd = float(np.asarray(be.to_numpy(be.array(mat.n(0.5876)))).flat[0])
                vd = float(np.asarray(be.to_numpy(be.array(mat.abbe()))).flat[0])
                # §5.10.1: n and ν must state the reference wavelength
                lines = [
                    display_name,
                    f"$n_d$ = {nd:.4f}  (587.6 nm)"
                    if not self._plain_text
                    else f"nd = {nd:.4f}  (587.6 nm)",
                    f"$\\nu_d$ = {vd:.2f}  (587.6 nm)"
                    if not self._plain_text
                    else f"νd = {vd:.2f}  (587.6 nm)",
                ]
            except Exception:
                lines = [display_name]
            # 0/, 1/, 2/ — per ISO 10110-1 Table 1; each glass column has its own
            # quality spec (cemented doublets may specify different grades per glass).
            comp_espec = self.spec.get_material_spec(self.element_index, mi)
            lines += _material_code_rows(comp_espec)
            _col_hdr(ci, "MATERIAL" if n_comps == 1 else f"GLASS {mi + 1}")
            _col_body(ci, lines)

    def _draw_title_block(self, phys_d: float) -> None:
        pw = self.pw
        geo = self._geo
        espec = self.spec.get_element_spec(self.element_index)

        ttl_x0 = self.style.border_margin
        ttl_x1 = pw - self.style.border_margin
        ttl_w = ttl_x1 - ttl_x0
        r1_y0 = self.style.border_margin + _TTL_R2_H
        r1_y1 = self.style.border_margin + _TTL_H
        r2_y0 = self.style.border_margin
        r2_y1 = r1_y0

        self._prim_rect(ttl_x0, r2_y0, ttl_w, _TTL_H, role="title")
        self._prim_hline(ttl_x0, ttl_x1, r1_y0, role="title")

        fracs1 = [0.34, 0.14, 0.14, 0.14, 0.12, 0.06, 0.06]
        xs1 = [ttl_x0]
        for f in fracs1:
            xs1.append(xs1[-1] + f * ttl_w)
        for xi in xs1[1:-1]:
            self._prim_vline(xi, r1_y0, r1_y1, role="title")

        def _tc(x0: float, x1: float, lbl: str, val: str) -> None:
            cx = (x0 + x1) / 2
            self._prim_text(
                (cx, r1_y0 + _TTL_R1_H * 0.25),
                lbl,
                ha="center",
                va="center",
                role="title_field_label",
            )
            self._prim_text(
                (cx, r1_y0 + _TTL_R1_H * 0.70),
                val,
                ha="center",
                va="center",
                role="title_field_value",
            )

        proj = self.spec.project_name or "OPTICAL ELEMENT"
        org = self.spec.organisation
        pn = espec.part_number or f"ELEM-{self.element_index + 1:03d}"
        proj_display = f"{org} / {proj}   {pn}" if org else f"{proj}   {pn}"
        sheet_num = f"{self.element_index + 1} / {self.total_sheets}"

        _tc(xs1[0], xs1[1], "ORGANISATION / PROJECT / PART NO.", proj_display)
        _tc(xs1[1], xs1[2], "DRAWN BY", espec.drawn_by or "—")
        _tc(xs1[2], xs1[3], "APPROVED", espec.approved_by or "—")
        _tc(xs1[3], xs1[4], "DATE", str(date.today()))
        _tc(xs1[4], xs1[5], "SCALE", geo.scale_label())
        _tc(xs1[5], xs1[6], "SHEET", sheet_num)
        _tc(xs1[6], xs1[7], "REV", espec.revision)

        notes_split = ttl_x0 + 0.60 * ttl_w
        self._prim_vline(notes_split, r2_y0, r2_y1, role="title")
        cy_r2 = (r2_y0 + r2_y1) / 2
        self._prim_text(
            (ttl_x0 + 2.0, r2_y1 - 1.5),
            "NOTES",
            ha="left",
            va="top",
            role="title_label_muted",
        )
        if espec.notes:
            self._prim_text(
                (ttl_x0 + 2.0, cy_r2),
                espec.notes,
                ha="left",
                va="center",
                role="title_notes_value",
            )

        std_w = ttl_x1 - notes_split
        sym_zone = notes_split + std_w * 0.60
        std_text_cx = (notes_split + sym_zone) / 2
        # ISO 10110-11 §4.1: defaults are based on the largest dimension of
        # the element — for a lens that is its physical outer diameter.
        defs = _iso10110_11_defaults(phys_d)
        self._prim_text(
            (std_text_cx, r2_y1 - 1.5),
            "DIM. IN mm",
            ha="center",
            va="top",
            role="title_label_muted",
        )
        self._prim_text(
            (std_text_cx, cy_r2),
            f"GENERAL TOL. PER ISO 10110-11\n"
            f"Ø±{defs['diameter']:.2f}  CT±{defs['thickness']:.2f}  "
            f"{defs['form_error']}  {defs['centration']}  {defs['imperfections']}",
            ha="center",
            va="center",
            role="title_general_tol",
        )

        # First-angle projection symbol
        sym_cy = cy_r2
        sym_R = _TTL_R2_H * 0.28
        sym_r = sym_R * 0.60
        sym_h = sym_R * 1.8
        sym_gap = sym_R * 0.5
        total_sym_w = 2 * sym_R + sym_gap + sym_h
        sym_cx_ref = (sym_zone + ttl_x1) / 2
        x_circ_c = sym_cx_ref - total_sym_w / 2 + sym_R
        x_cone_l = x_circ_c + sym_R + sym_gap
        x_cone_r = x_cone_l + sym_h
        self._prim_circle((x_circ_c, sym_cy), sym_R, role="projection_symbol")
        self._prim_circle((x_circ_c, sym_cy), sym_r, role="projection_symbol")
        self._prim_polygon(
            [
                (x_cone_l, sym_cy + sym_R),
                (x_cone_r, sym_cy + sym_r),
                (x_cone_r, sym_cy - sym_r),
                (x_cone_l, sym_cy - sym_R),
            ],
            role="projection_symbol",
        )
        self._prim_line(
            (x_circ_c - sym_R * 1.3, sym_cy),
            (x_cone_r + sym_R * 0.5, sym_cy),
            role="projection_axis",
        )

    def _draw_dimension_lines(self, sa_e: float, top_f, top_r, bot_r, bot_f) -> float:
        geo = self._geo
        s = geo.scale
        espec = self.spec.get_element_spec(self.element_index)
        surfs = self.element.surfaces

        # Centre thickness (below lens)
        z0 = _surf_z(surfs[0])
        zN = _surf_z(surfs[-1])
        ct = abs(zN - z0)
        dy = geo.pt(0, -(sa_e + 8.0 / s))[1]
        ey = geo.pt(0, -(sa_e + 1.5 / s))[1]
        p0x = geo.pt(z0, 0)[0]
        p1x = geo.pt(zN, 0)[0]
        self._prim_vline(p0x, ey, dy, role="dims")
        self._prim_vline(p1x, ey, dy, role="dims")
        self._prim_dim_arrow((p0x, dy), (p1x, dy), role="dims")
        ct_tol_m = self._tol_fmt(espec.ct_tolerance)
        _ct_m_marker = " M" if getattr(espec, "matched_pair", False) else ""
        ct_txt = self._fmt_dim_text(f"{ct:.2f}", ct_tol_m) + _ct_m_marker
        self._prim_text(
            ((p0x + p1x) / 2, dy - 1.5), ct_txt, ha="center", va="top", role="dim_ct"
        )

        # Physical outer diameter (right side)
        phys_d = espec.diameter if espec.diameter is not None else 2.0 * sa_e
        dz = geo.opt_z_max + 10.0 / s
        dax = geo.pt(dz, 0)[0]
        self._prim_hline(top_r[0], dax + 2, top_r[1], role="dims")
        self._prim_hline(bot_r[0], dax + 2, bot_r[1], role="dims")
        # The OD arrow's xy/xytext happen to be in the opposite order from
        # the CT/ET/component arrows in the original matplotlib code (a
        # pre-existing inconsistency, not a semantic difference) — preserved
        # exactly via the distinct "dims_od" role.
        self._prim_dim_arrow((dax, top_r[1]), (dax, bot_r[1]), role="dims_od")
        d_tol_m = self._tol_fmt(espec.diameter_tolerance)
        d_txt = "Ø " + self._fmt_dim_text(f"{phys_d:.2f}", d_tol_m)
        self._prim_text(
            (dax + 2.5, (top_r[1] + bot_r[1]) / 2),
            d_txt,
            ha="left",
            va="center",
            role="dim_od",
        )

        # Edge thickness (above lens, reference dimension in brackets)
        z1r = _surf_z(surfs[0]) + _to_float(
            surfs[0].geometry.sag(be.zeros(1), be.array([sa_e]))[0]
        )
        zNr = _surf_z(surfs[-1]) + _to_float(
            surfs[-1].geometry.sag(be.zeros(1), be.array([sa_e]))[0]
        )
        et = abs(zNr - z1r)
        et_dy = geo.pt(0, sa_e + 8.0 / s)[1]
        et_ey = geo.pt(0, sa_e + 1.5 / s)[1]
        et_x0 = top_f[0]
        et_x1 = top_r[0]
        self._prim_vline(et_x0, et_ey, et_dy, role="dims")
        self._prim_vline(et_x1, et_ey, et_dy, role="dims")
        self._prim_dim_arrow((et_x0, et_dy), (et_x1, et_dy), role="dims")
        self._prim_text(
            ((et_x0 + et_x1) / 2, et_dy + 1.5),
            f"({et:.2f})",
            ha="center",
            va="bottom",
            role="dim_et",
        )

        # Component thicknesses (cemented doublet)
        if self.element.is_cemented:
            th_y = geo.pt(0, sa_e + 8.0 / s)[1]
            eth_y = geo.pt(0, sa_e + 1.5 / s)[1]
            for ci, th in enumerate(
                self.element.component_thicknesses(self.spec.optic)
            ):
                za = _surf_z(surfs[ci])
                zb = _surf_z(surfs[ci + 1])
                xa = geo.pt(za, 0)[0]
                xb = geo.pt(zb, 0)[0]
                self._prim_vline(xa, eth_y, th_y, role="dims")
                self._prim_vline(xb, eth_y, th_y, role="dims")
                self._prim_dim_arrow((xa, th_y), (xb, th_y), role="dims")
                self._prim_text(
                    ((xa + xb) / 2, th_y + 2),
                    f"t{ci + 1} = {th:.2f}",
                    ha="center",
                    va="bottom",
                    role="dim_component",
                )

        return phys_d

    def _fmt_dim_text(self, value: str, tol_m: str) -> str:
        """Wrap *value* + tolerance markup identically to how the matplotlib
        mathtext / DXF plain-text tolerance strings are assembled elsewhere."""
        raise NotImplementedError

    def _axis_margin_mm(self) -> float:
        """How far the optical/rotational axis line extends past the lens
        outline, in optical mm — a pre-existing per-format tuning difference
        (10mm matplotlib, 15mm DXF), not a shared layout decision."""
        raise NotImplementedError

    def _draw_reference_annotation(self) -> None:
        # ISO 10110 reference + λ annotation (§4, mandatory since 2019),
        # placed at bottom-left of the drawing field per Annex A examples.
        ref_wl = getattr(self.spec, "reference_wavelength", 546.07)
        self._prim_text(
            (self.style.border_margin + 2.0, self.style.border_margin + _BOT_H + 3.0),
            f"Ang. nach ISO 10110; λ = {ref_wl:.2f} nm",
            ha="left",
            va="bottom",
            role="reference_note",
        )

    def _draw_efl_annotation(self) -> None:
        # f' (EFL) annotation — shown in all ISO Annex A examples.
        try:
            efl = float(self.spec.optic.paraxial.f2())
        except Exception:
            return
        self._prim_text(
            (self.style.border_margin + 2.0, self.ph - self.style.border_margin - 5.0),
            f"f′ = {efl:.2f} mm",
            ha="left",
            va="top",
            role="efl_annotation",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# _MatplotlibRenderer — isolated matplotlib / PDF / PNG rendering
# ═══════════════════════════════════════════════════════════════════════════════


class _MatplotlibRenderer(_BaseRenderer):
    """Renders a single element ISO 10110 drawing to a matplotlib Figure.

    Isolated from file-I/O and DXF logic per HarrisonKramer's architectural
    feedback: "moving renderers to their own isolated classes will make things
    much more maintainable."

    Args:
        element:       :class:`~optiland.iso10110.elements.LensElement`
        spec:          :class:`~optiland.iso10110.spec.DrawingSpec`
        element_index: 0-based element number within the drawing set.
        total_sheets:  Total number of drawing sheets (for X/Y sheet numbering).
        pw, ph:        Paper width and height in mm.
        rotational_axis_y: y-offset (optical mm) of the rotational axis.
            ``0.0`` (default) = coincides with optical axis and is shown combined.
            Non-zero = separate rotational-axis line is drawn at that offset.
        style:         :class:`~optiland.iso10110.style.DrawingStyle` font-size
            scale factors and border margin. Defaults to ``DrawingStyle()``.
    """

    def __init__(
        self,
        element,
        spec,
        element_index: int,
        total_sheets: int,
        pw: float,
        ph: float,
        rotational_axis_y: float = 0.0,
        style: DrawingStyle | None = None,
    ) -> None:
        self.element = element
        self.spec = spec
        self.element_index = element_index
        self.total_sheets = total_sheets
        self.pw = pw
        self.ph = ph
        self.rotational_axis_y = rotational_axis_y
        self.style = style if style is not None else DrawingStyle()
        self._geo = _Geo(
            element,
            spec,
            pw,
            ph,
            bottom_h=_BOT_H,
            cal_x=12.0,
            cal_y=18.0,
            border_margin=self.style.border_margin,
        )
        self._ax = None

    # ── primitives (drawn onto self._ax, set at the top of render()) ────────

    _plain_text = False

    # (pos, fontsize_base, style_scale_attr, zorder, kwargs)
    _TEXT_ROLE_STYLE: dict[str, tuple] = {
        "table_header": (6.5, "table_header_scale", 8, {"fontweight": "bold"}),
        "table_body": (7.5, "table_body_scale", 8, {}),
        "table_body_coating": (7.5, "table_body_scale", 9, {}),
        "lambda_symbol": (5.0, "lambda_symbol_scale", 9, {"fontstyle": "italic"}),
        "title_field_label": (5.5, "title_label_scale", 8, {}),
        "title_field_value": (7.5, "title_value_scale", 8, {"fontweight": "bold"}),
        "title_label_muted": (5.5, "title_label_scale", 8, {"color": "#555555"}),
        "title_notes_value": (6.5, "title_notes_scale", 8, {}),
        "title_general_tol": (5.0, "title_general_tol_scale", 8, {}),
        # Dimension-line labels didn't specify an explicit zorder originally;
        # 3 reproduces matplotlib's own default Text zorder exactly.
        "dim_ct": (9, "dimension_scale", 8, {}),
        "dim_od": (9, "dimension_scale", 8, {}),
        "dim_et": (8, "dimension_scale", 8, {}),
        "dim_component": (7, "component_dimension_scale", 3, {}),
        "reference_note": (6.0, "reference_note_scale", 8, {"fontstyle": "italic"}),
        "efl_annotation": (7.5, "efl_annotation_scale", 8, {"fontweight": "bold"}),
        "axis_label": (5.5, "axis_label_scale", 2, {}),
        "axis_surface_label": (6.0, "surface_label_scale", 5, {"fontweight": "bold"}),
        "sharp_edge_symbol": (
            8.0,
            "symbol_annotation_scale",
            7,
            {"fontweight": "bold"},
        ),
        "aperture_bracket_label": (5.5, "axis_label_scale", 6, {}),
        "surface_finish_label": (6.0, "surface_label_scale", 7, {"fontweight": "bold"}),
    }

    # role -> (lw, zorder)
    _PANEL_ROLE_STYLE = {
        "table": (0.5, 6),
        "title": (0.5, 6),
        "sheet_border_outer": (0.4, 10),
        "sheet_border_inner": (0.8, 10),
    }
    # "dims" didn't specify an explicit zorder originally; 2 reproduces
    # matplotlib's own default Line2D zorder exactly.
    _PANEL_LINE_ROLE_STYLE = {
        "table": (0.4, 7),
        "title": (0.4, 7),
        "dims": (0.5, 2),
        "aperture_bracket": (0.4, 6),
    }
    # role -> lw
    _CIRCLE_ROLE_LW = {"lambda_symbol": 0.5, "projection_symbol": 0.7}

    def _prim_rect(self, x: float, y: float, w: float, h: float, *, role: str) -> None:
        import matplotlib.patches as mpatches

        lw, zorder = self._PANEL_ROLE_STYLE[role]
        self._ax.add_patch(
            mpatches.Rectangle(
                (x, y), w, h, lw=lw, ec="black", fc="none", zorder=zorder
            )
        )

    def _prim_hline(self, x0: float, x1: float, y: float, *, role: str) -> None:
        lw, zorder = self._PANEL_LINE_ROLE_STYLE[role]
        self._ax.plot([x0, x1], [y, y], "k-", lw=lw, zorder=zorder)

    def _prim_vline(self, x: float, y0: float, y1: float, *, role: str) -> None:
        lw, zorder = self._PANEL_LINE_ROLE_STYLE[role]
        self._ax.plot([x, x], [y0, y1], "k-", lw=lw, zorder=zorder)

    # role -> (linestyle, zorder)
    _LINE_ROLE_STYLE = {
        "projection_axis": ((0, (4, 1.5, 1, 1.5)), 9),
        "optical_axis": ((0, (8, 2, 1, 2, 1, 2)), 1),  # ISO 128 PHANTOM
        "rotational_axis": ((0, (8, 2, 1, 2)), 1),  # ISO 128 DASHDOT
    }

    def _prim_line(
        self, p1: tuple[float, float], p2: tuple[float, float], *, role: str
    ) -> None:
        if role == "surface_finish_symbol":
            # Matches the original bar's own PathPatch/MPath construction
            # (kept distinct from the plain ax.plot used for axis lines, to
            # avoid changing its exact vector output).
            import matplotlib.patches as mpatches
            from matplotlib.path import Path as MPath

            self._ax.add_patch(
                mpatches.PathPatch(
                    MPath([p1, p2], [MPath.MOVETO, MPath.LINETO]),
                    fc="none",
                    ec="black",
                    lw=0.7,
                    zorder=6,
                )
            )
            return
        linestyle, zorder = self._LINE_ROLE_STYLE[role]
        self._ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color="black",
            lw=0.5,
            linestyle=linestyle,
            zorder=zorder,
        )

    # role -> (lw, zorder)
    _POLYGON_ROLE_STYLE = {
        "projection_symbol": (0.7, 9),
        "surface_finish_symbol": (0.7, 6),
    }

    def _prim_polygon(self, points: list[tuple[float, float]], *, role: str) -> None:
        import matplotlib.patches as mpatches

        lw, zorder = self._POLYGON_ROLE_STYLE[role]
        if role == "surface_finish_symbol":
            # Matches the original triangle's own PathPatch/MPath construction
            # (kept distinct from the plain mpatches.Polygon used for the
            # projection symbol, to avoid changing its exact vector output).
            from matplotlib.path import Path as MPath

            self._ax.add_patch(
                mpatches.PathPatch(
                    MPath(
                        [*points, points[0]],
                        [MPath.MOVETO]
                        + [MPath.LINETO] * (len(points) - 1)
                        + [MPath.CLOSEPOLY],
                    ),
                    fc="none",
                    ec="black",
                    lw=lw,
                    zorder=zorder,
                )
            )
            return
        self._ax.add_patch(
            mpatches.Polygon(
                points, closed=True, fill=False, ec="black", lw=lw, zorder=zorder
            )
        )

    def _prim_curve(self, points: list[tuple[float, float]], *, role: str) -> None:
        import matplotlib.patches as mpatches
        from matplotlib.path import Path as MPath

        lw, zorder = self._POLYGON_ROLE_STYLE[role]
        self._ax.add_patch(
            mpatches.PathPatch(
                MPath(points, [MPath.MOVETO] + [MPath.CURVE4] * (len(points) - 1)),
                fc="none",
                ec="black",
                lw=lw,
                zorder=zorder,
            )
        )

    def _prim_text(
        self, pos: tuple[float, float], s: str, *, ha: str, va: str, role: str
    ) -> None:
        base, scale_attr, zorder, extra = self._TEXT_ROLE_STYLE[role]
        self._ax.text(
            pos[0],
            pos[1],
            s,
            ha=ha,
            va=va,
            fontsize=base * getattr(self.style, scale_attr),
            zorder=zorder,
            **extra,
        )

    def _prim_circle(self, center: tuple[float, float], r: float, *, role: str) -> None:
        import matplotlib.patches as mpatches

        self._ax.add_patch(
            mpatches.Circle(
                center,
                r,
                fill=False,
                ec="black",
                lw=self._CIRCLE_ROLE_LW[role],
                zorder=9,
            )
        )

    def _tol_fmt(self, tol) -> str:
        return _tol_math(tol)

    def _fmt_dim_text(self, value: str, tol_m: str) -> str:
        return f"${value}{tol_m}$" if tol_m else value

    def _axis_margin_mm(self) -> float:
        return 10.0

    def _prim_dim_arrow(
        self, p1: tuple[float, float], p2: tuple[float, float], *, role: str
    ) -> None:
        xy, xytext = (p1, p2) if role == "dims_od" else (p2, p1)
        self._ax.annotate(
            "",
            xy=xy,
            xytext=xytext,
            arrowprops=dict(arrowstyle="<->", color="black", lw=0.6, mutation_scale=7),
        )

    # ── main render ─────────────────────────────────────────────────────────

    def render(self, figsize: tuple | None = None):
        """Build and return a matplotlib Figure for this element."""
        import matplotlib.pyplot as plt

        pw, ph = self.pw, self.ph

        if figsize is None:
            figsize = (pw / 25.4, ph / 25.4)

        fig = plt.figure(figsize=figsize, facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, pw)
        ax.set_ylim(0, ph)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor("white")
        self._ax = ax

        self._draw_borders()
        sa_e, top_f, top_r, bot_r, bot_f = self._draw_lens_outline(ax)
        self._draw_axes()
        self._draw_sharp_edge_symbols()
        self._draw_aperture_brackets()
        self._draw_surface_finish_callouts(sa_e)
        phys_d = self._draw_dimension_lines(sa_e, top_f, top_r, bot_r, bot_f)
        self._draw_spec_table()
        self._draw_reference_annotation()
        self._draw_efl_annotation()
        self._draw_title_block(phys_d)

        return fig

    # ── render sections ────────────────────────────────────────────────────

    def _draw_lens_outline(
        self, ax
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        geo = self._geo
        n = len(self.element.surfaces)

        # ── Lens cross-section ────────────────────────────────────────────
        sa_e = geo.sa_max
        front = geo.curve_at(0, sa_e)
        rear = geo.curve_at(n - 1, sa_e)
        top_f = geo.rim_at(0, +1, sa_e)
        top_r = geo.rim_at(n - 1, +1, sa_e)
        bot_r = geo.rim_at(n - 1, -1, sa_e)
        bot_f = geo.rim_at(0, -1, sa_e)

        outline = np.vstack([front, [top_r], rear[::-1], [bot_f]])

        # ISO 128-50 optical-glass fill
        ax.fill(
            outline[:, 0],
            outline[:, 1],
            facecolor="#ddeeff",
            edgecolor="none",
            linewidth=0,
            zorder=2,
        )

        # ISO 128-50 short-long-short hatch; alternating direction for
        # cemented assemblies (§5.2: "components must be hatched in
        # alternating directions")
        n_comps = self.element.num_components
        if n_comps == 1:
            _draw_glass_hatch(ax, outline, direction=+1)
        else:
            for _k in range(n_comps):
                _cf = geo.curve_at(_k, sa_e)
                _cr = geo.curve_at(_k + 1, sa_e)
                _tr = geo.rim_at(_k + 1, +1, sa_e)
                _bl = geo.rim_at(_k, -1, sa_e)
                _comp_outline = np.vstack([_cf, [_tr], _cr[::-1], [_bl]])
                ax.fill(
                    _comp_outline[:, 0],
                    _comp_outline[:, 1],
                    facecolor="#ddeeff",
                    edgecolor="none",
                    linewidth=0,
                    zorder=2,
                )
                _draw_glass_hatch(
                    ax, _comp_outline, direction=(+1 if _k % 2 == 0 else -1)
                )

        closed = np.vstack([outline, outline[0]])
        ax.plot(closed[:, 0], closed[:, 1], "k-", lw=1.5, zorder=4)

        for i in range(1, n - 1):  # cement interfaces
            cp = geo.curve(i)
            ax.plot(cp[:, 0], cp[:, 1], "k-", lw=0.8, zorder=5)

        return sa_e, top_f, top_r, bot_r, bot_f


# ═══════════════════════════════════════════════════════════════════════════════
# _DxfRenderer — isolated ezdxf / DXF rendering
# ═══════════════════════════════════════════════════════════════════════════════


class _DxfRenderer(_BaseRenderer):
    """Renders a single element ISO 10110 drawing into an ezdxf document.

    Isolated from matplotlib logic per HarrisonKramer's architectural feedback.

    Args:
        style: :class:`~optiland.iso10110.style.DrawingStyle` font-size scale
            factors and border margin. Defaults to ``DrawingStyle()``.
    """

    def __init__(
        self,
        element,
        spec,
        element_index: int,
        total_sheets: int,
        pw: float,
        ph: float,
        rotational_axis_y: float = 0.0,
        style: DrawingStyle | None = None,
    ) -> None:
        self.element = element
        self.spec = spec
        self.element_index = element_index
        self.total_sheets = total_sheets
        self.pw = pw
        self.ph = ph
        self.rotational_axis_y = rotational_axis_y
        self.style = style if style is not None else DrawingStyle()
        self._geo = _Geo(
            element,
            spec,
            pw,
            ph,
            bottom_h=_BOT_H,
            cal_x=12.0,
            cal_y=18.0,
            border_margin=self.style.border_margin,
        )
        self._msp = None

    def render(self, doc) -> None:
        """Populate *doc* (an ezdxf Drawing) with the element drawing."""
        self._setup_layers(doc)
        msp = doc.modelspace()
        self._msp = msp
        self._draw_borders()
        self._draw_reference_annotation()
        self._draw_efl_annotation()
        espec = self.spec.get_element_spec(self.element_index)
        phys_d = (
            espec.diameter if espec.diameter is not None else 2.0 * self._geo.sa_max
        )
        self._draw_title_block(phys_d)
        self._draw_spec_table()
        self._draw_axes()
        self._dxf_lens(msp)
        n = len(self.element.surfaces)
        sa_e = self._geo.sa_max
        top_f = self._geo.rim_at(0, +1, sa_e)
        top_r = self._geo.rim_at(n - 1, +1, sa_e)
        # Pre-existing DXF-specific quirk (predates this refactor, preserved
        # exactly): the OD dimension's bottom witness line is anchored to the
        # *front* surface's rim here, unlike the matplotlib renderer which
        # uses the rear surface for both top and bottom.
        bot_r = self._geo.rim_at(0, -1, sa_e)
        bot_f = self._geo.rim_at(0, -1, sa_e)
        self._draw_dimension_lines(sa_e, top_f, top_r, bot_r, bot_f)
        self._draw_aperture_brackets()
        self._draw_surface_finish_callouts(sa_e)
        self._draw_sharp_edge_symbols()

    # ── primitives ──────────────────────────────────────────────────────────

    _plain_text = True

    # role -> DXF layer, for the panel (rect/line) and circle primitives
    _PANEL_ROLE_LAYER = {
        "table": L_TABLE,
        "title": L_BORDER,
        "dims": L_DIMS,
        "sheet_border_outer": L_BORDER,
        "sheet_border_inner": L_BORDER,
        "aperture_bracket": L_DIMS,
    }
    # role -> explicit lineweight override (None = use the layer's own default)
    _PANEL_ROLE_LINEWEIGHT = {"sheet_border_inner": 50}
    _CIRCLE_ROLE_LAYER = {"lambda_symbol": L_TABLE, "projection_symbol": L_BORDER}

    def _prim_rect(self, x: float, y: float, w: float, h: float, *, role: str) -> None:
        dxfattribs = {"layer": self._PANEL_ROLE_LAYER[role]}
        lw = self._PANEL_ROLE_LINEWEIGHT.get(role)
        if lw is not None:
            dxfattribs["lineweight"] = lw
        self._msp.add_lwpolyline(
            [(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
            close=True,
            dxfattribs=dxfattribs,
        )

    def _prim_hline(self, x0: float, x1: float, y: float, *, role: str) -> None:
        self._msp.add_line(
            (x0, y), (x1, y), dxfattribs={"layer": self._PANEL_ROLE_LAYER[role]}
        )

    def _prim_vline(self, x: float, y0: float, y1: float, *, role: str) -> None:
        self._msp.add_line(
            (x, y0), (x, y1), dxfattribs={"layer": self._PANEL_ROLE_LAYER[role]}
        )

    def _prim_line(
        self, p1: tuple[float, float], p2: tuple[float, float], *, role: str
    ) -> None:
        if role == "surface_finish_symbol":
            self._msp.add_line(p1, p2, dxfattribs={"layer": L_CALLOUT})
            return
        if role == "rotational_axis":
            # Visually distinct from the optical axis (dash-double-dot) so the
            # two lines can be told apart when offset — ISO 128 §3.2 type 04.1.
            lt_rot = "DASHDOT"
            if lt_rot not in self._msp.doc.linetypes:
                self._msp.doc.linetypes.add(lt_rot, pattern=[8.0, -2.0, 1.0, -2.0])
            self._msp.add_line(p1, p2, dxfattribs={"layer": L_AXIS, "linetype": lt_rot})
            return
        # "projection_axis" / "optical_axis": rely on the AXIS layer's own
        # "OPTICAL" dash-dot linetype rather than a per-entity dash pattern.
        self._msp.add_line(p1, p2, dxfattribs={"layer": L_AXIS})

    _POLYGON_ROLE_LAYER = {
        "projection_symbol": L_BORDER,
        "surface_finish_symbol": L_CALLOUT,
    }

    def _prim_polygon(self, points: list[tuple[float, float]], *, role: str) -> None:
        self._msp.add_lwpolyline(
            points, close=True, dxfattribs={"layer": self._POLYGON_ROLE_LAYER[role]}
        )

    def _prim_curve(self, points: list[tuple[float, float]], *, role: str) -> None:
        # No native Bezier here: straight line from the first to the last
        # point, matching the DXF renderer's original approximation.
        self._msp.add_line(points[0], points[-1], dxfattribs={"layer": L_CALLOUT})

    _HALIGN = {"left": 0, "center": 4, "right": 2}
    _VALIGN = {"bottom": 1, "center": 2, "top": 3}

    # role -> (height_base, style_scale_attr, layer)
    _TEXT_ROLE_STYLE: dict[str, tuple] = {
        "table_header": (2.0, "table_header_scale", L_TABLE),
        "table_body": (2.0, "table_body_scale", L_TABLE),
        "table_body_coating": (2.0, "table_body_scale", L_TABLE),
        "lambda_symbol": (_LAMBDA_R * 1.4, "lambda_symbol_scale", L_TABLE),
        "title_field_label": (1.8, "title_label_scale", L_BORDER),
        "title_field_value": (2.5, "title_value_scale", L_BORDER),
        "title_label_muted": (1.8, "title_label_scale", L_BORDER),
        "title_notes_value": (2.0, "title_notes_scale", L_BORDER),
        "title_general_tol": (1.8, "title_general_tol_scale", L_BORDER),
        "dim_ct": (2.5, "dimension_scale", L_DIMS),
        "dim_od": (2.5, "dimension_scale", L_DIMS),
        "dim_et": (2.5, "dimension_scale", L_DIMS),
        "dim_component": (2.0, "component_dimension_scale", L_DIMS),
        "reference_note": (1.8, "reference_note_scale", L_BORDER),
        "efl_annotation": (2.5, "efl_annotation_scale", L_BORDER),
        "axis_label": (2.0, "axis_label_scale", L_AXIS),
        "axis_surface_label": (2.0, "surface_label_scale", L_AXIS),
        "sharp_edge_symbol": (3.0, "symbol_annotation_scale", L_CALLOUT),
        "aperture_bracket_label": (2.0, "axis_label_scale", L_DIMS),
        "surface_finish_label": (2.0, "surface_label_scale", L_CALLOUT),
    }

    def _prim_text(
        self, pos: tuple[float, float], s: str, *, ha: str, va: str, role: str
    ) -> None:
        height_base, scale_attr, layer = self._TEXT_ROLE_STYLE[role]
        height = height_base * getattr(self.style, scale_attr)

        if role == "title_general_tol":
            # DXF TEXT has no multi-line support: split on the shared "\n"
            # into two single-line entities, offset ±1.2mm around the same
            # anchor the matplotlib renderer centers its one multi-line
            # text block on.
            line1, line2 = s.split("\n")
            for line, dy, line_va in ((line1, 1.2, "bottom"), (line2, -1.2, "top")):
                p = (pos[0], pos[1] + dy)
                self._msp.add_text(
                    line,
                    dxfattribs={
                        "layer": layer,
                        "height": height,
                        "insert": p,
                        "halign": 4,
                        "valign": self._VALIGN[line_va],
                        "align_point": p,
                    },
                )
            return

        # The coating text is DXF-vertically-centered on the lambda circle
        # rather than top-aligned to the row like the matplotlib renderer —
        # a deliberate per-format tuning difference, not shared layout.
        if role == "table_body_coating":
            pos = (pos[0], pos[1] - _LAMBDA_R)
            va = "center"

        # Dimension-line labels sit further from their dimension line in DXF
        # than in matplotlib (larger absolute text height needs more
        # clearance) — per-format offsets tuned independently, not shared
        # layout, applied on top of the shared anchor position.
        if role == "dim_ct":
            pos = (pos[0], pos[1] - 2.5)
        elif role == "dim_od":
            pos = (pos[0] + 0.5, pos[1])
        elif role == "dim_et":
            pos = (pos[0], pos[1] + 0.5)

        halign = self._HALIGN[ha]
        valign = self._VALIGN[va]
        dxfattribs = {
            "layer": layer,
            "height": height,
            "insert": pos,
            "halign": halign,
            "valign": valign,
        }
        # ezdxf requires align_point for non-left alignment; the aperture
        # bracket label always carries one in the original code even for its
        # left-aligned case (a pre-existing inconsistency, preserved as-is).
        if halign != 0 or role == "aperture_bracket_label":
            dxfattribs["align_point"] = pos
        self._msp.add_text(s, dxfattribs=dxfattribs)

    def _prim_circle(self, center: tuple[float, float], r: float, *, role: str) -> None:
        self._msp.add_circle(
            center, r, dxfattribs={"layer": self._CIRCLE_ROLE_LAYER[role]}
        )

    def _tol_fmt(self, tol) -> str:
        return _tol_plain(tol)

    def _fmt_line(self, s: str) -> str:
        # DXF R2010: strip mathtext markers and replace ∞ with "inf" for broad
        # viewer compatibility. κ (U+03BA) and other BMP characters are left
        # intact — ezdxf encodes them correctly in R2010 TEXT entities.
        return s.replace("∞", "inf").replace("$", "")

    def _fmt_dim_text(self, value: str, tol_m: str) -> str:
        return f"{value}{tol_m}"

    def _axis_margin_mm(self) -> float:
        return 15.0

    def _prim_dim_arrow(
        self, p1: tuple[float, float], p2: tuple[float, float], *, role: str
    ) -> None:
        self._msp.add_line(p1, p2, dxfattribs={"layer": L_DIMS})
        _dxf_arrowhead(self._msp, p1, p2, L_DIMS)
        _dxf_arrowhead(self._msp, p2, p1, L_DIMS)

    # ── layers ──────────────────────────────────────────────────────────────

    def _setup_layers(self, doc):
        lays = doc.layers

        def _a(name, color, lw):
            if name not in lays:
                lays.add(name, color=color, lineweight=lw)

        _a(L_BORDER, 7, 25)
        _a(L_OUTLINE, 7, 50)
        _a(L_AXIS, 8, 13)
        _a(L_CEMENT, 7, 13)
        _a(L_DIMS, 7, 13)
        _a(L_CALLOUT, 7, 13)
        _a(L_TABLE, 7, 13)

        # Optical axis: ISO 128-20 dash-double-dot linetype matching mpl renderer.
        # "OPTICAL" is a private name to avoid conflict with the standard AutoCAD
        # "CENTER" (dash-single-dot) definition.
        lt = "OPTICAL"
        if lt not in doc.linetypes:
            doc.linetypes.add(lt, pattern=[8.0, -2.0, 1.0, -2.0, 1.0, -2.0])
        lays.get(L_AXIS).dxf.linetype = lt

    # ── optical axis ────────────────────────────────────────────────────────

    # ── lens outline ────────────────────────────────────────────────────────

    def _dxf_lens(self, msp):
        geo = self._geo
        surfs = self.element.surfaces
        n = len(surfs)
        sa_e = geo.sa_max

        def _dxf_curve_sa(si, sa_val):
            return geo.curve_at(si, sa_val).tolist()

        front = _dxf_curve_sa(0, sa_e)
        rear = _dxf_curve_sa(n - 1, sa_e)
        top_r = geo.rim_at(n - 1, +1, sa_e)
        bot_f = geo.rim_at(0, -1, sa_e)
        outline = front + [top_r] + list(reversed(rear)) + [bot_f]
        msp.add_lwpolyline(
            outline, close=True, dxfattribs={"layer": L_OUTLINE, "lineweight": 50}
        )
        for i in range(1, n - 1):
            msp.add_lwpolyline(
                _dxf_curve_sa(i, geo.sa[i]),
                dxfattribs={"layer": L_CEMENT, "lineweight": 25},
            )

        # ISO 128-50 optical glass hatch — alternating 45° / 135° per component
        # (ISO 128-50 §4.2.3: adjacent components must be hatched in opposite
        # directions so cement interfaces are visually distinguishable).
        n_comps = self.element.num_components
        try:
            for _k in range(n_comps):
                _cf = _dxf_curve_sa(_k, sa_e)
                _cr = _dxf_curve_sa(_k + 1, sa_e)
                _tr = geo.rim_at(_k + 1, +1, sa_e)
                _bl = geo.rim_at(_k, -1, sa_e)
                _comp_outline = _cf + [_tr] + list(reversed(_cr)) + [_bl]
                _h = msp.add_hatch(color=150, dxfattribs={"layer": L_OUTLINE})
                _angle = 0 if _k % 2 == 0 else 90  # 45° or 135° (ANSI31 base is 45°)
                _h.set_pattern_fill("ANSI31", scale=1.0, angle=_angle)
                _h.paths.add_polyline_path(_comp_outline, is_closed=True)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# ElementDrawing — thin public coordinator
# ═══════════════════════════════════════════════════════════════════════════════


class ElementDrawing:
    """ISO 10110 fabrication drawing for one lens element.

    Delegates all rendering to :class:`_MatplotlibRenderer` (PDF/PNG) and
    :class:`_DxfRenderer` (DXF).  Rendering logic is fully isolated in those
    classes; this class handles only file-I/O and public API.

    Args:
        element:       :class:`~optiland.iso10110.elements.LensElement`
        spec:          :class:`~optiland.iso10110.spec.DrawingSpec`
        element_index: 0-based element number.
        paper:         Paper size string (default ``"A4"``).
        orientation:   ``"portrait"`` (default) or ``"landscape"``.
        total_sheets:  Total sheet count for X/Y numbering.
        rotational_axis_y: y-offset (optical mm) of the rotational axis when
            it differs from the optical axis.  Default ``0.0`` = coincident.
        style:         :class:`~optiland.iso10110.style.DrawingStyle` font-size
            scale factors and border margin. Defaults to ``DrawingStyle()``,
            reproducing the built-in appearance exactly.
    """

    def __init__(
        self,
        element,
        spec,
        element_index: int = 0,
        paper: str = "A4",
        orientation: str = "portrait",
        total_sheets: int = 1,
        rotational_axis_y: float = 0.0,
        style: DrawingStyle | None = None,
    ) -> None:
        self.element = element
        self.spec = spec
        self.element_index = element_index
        self.paper = paper.upper()
        self.orientation = orientation.lower()
        self.total_sheets = total_sheets
        self.rotational_axis_y = rotational_axis_y
        self.style = style if style is not None else DrawingStyle()
        self._pw, self._ph = _paper_wh(paper, orientation)
        self._doc = None

    # ── public ──────────────────────────────────────────────────────────────

    def generate(self) -> ezdxf.document.Drawing:
        """Build DXF document (for save_dxf)."""
        try:
            import ezdxf
            from ezdxf.math import Vec3 as _DxfVec3
        except ImportError as exc:
            raise ImportError(
                "ezdxf is required for DXF output. Install it with the "
                "'manufacturing' extra: pip install optiland[manufacturing]"
            ) from exc

        doc = ezdxf.new("R2010", units=4)
        doc.header["$INSUNITS"] = 4
        renderer = _DxfRenderer(
            self.element,
            self.spec,
            self.element_index,
            self.total_sheets,
            self._pw,
            self._ph,
            rotational_axis_y=self.rotational_axis_y,
            style=self.style,
        )
        renderer.render(doc)
        # ezdxf.write() calls update_extents() which overwrites $EXTMIN/$EXTMAX
        # from modelspace.dxf.extmin/extmax — and Vec3(0,0,0) is falsy, so the
        # origin corner is never written.  Override the method on this instance
        # so the paper-boundary extents always survive to the saved file.
        _pw, _ph = self._pw, self._ph

        def _paper_extents():
            doc.header["$EXTMIN"] = _DxfVec3(0.0, 0.0, 0.0)
            doc.header["$EXTMAX"] = _DxfVec3(_pw, _ph, 0.0)
            active = doc.active_layout()
            doc.header["$PEXTMIN"] = active.dxf.extmin
            doc.header["$PEXTMAX"] = active.dxf.extmax

        doc.update_extents = _paper_extents
        self._doc = doc
        return doc

    def save_dxf(self, path: str | Path) -> None:
        """Save native DXF file."""
        if self._doc is None:
            self.generate()
        self._doc.saveas(str(path))

    def save_pdf(self, path: str | Path) -> None:
        """Save clean PDF via matplotlib."""
        import matplotlib.pyplot as plt

        fig = self._mpl_figure()
        fig.savefig(
            str(path), dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        plt.close(fig)

    def save_png(self, path: str | Path, dpi: int = 200) -> None:
        """Save PNG via matplotlib."""
        import matplotlib.pyplot as plt

        fig = self._mpl_figure()
        fig.savefig(str(path), dpi=dpi, facecolor="white", edgecolor="none")
        plt.close(fig)

    def show(self, figsize: tuple | None = None) -> None:
        """Display inline."""
        import matplotlib.pyplot as plt

        fig = self._mpl_figure(figsize=figsize)
        plt.show()
        plt.close(fig)

    def _mpl_figure(self, figsize: tuple | None = None):
        """Delegate to _MatplotlibRenderer (kept for backward compatibility)."""
        renderer = _MatplotlibRenderer(
            self.element,
            self.spec,
            self.element_index,
            self.total_sheets,
            self._pw,
            self._ph,
            rotational_axis_y=self.rotational_axis_y,
            style=self.style,
        )
        return renderer.render(figsize)


# ── DXF → PNG converter ───────────────────────────────────────────────────────


def dxf_to_png(
    dxf_path: str | Path,
    png_path: str | Path,
    dpi: int = 200,
) -> None:
    """Render a DXF file to PNG using ezdxf with a white background."""
    import ezdxf
    import matplotlib.pyplot as plt
    from ezdxf.addons.drawing import Frontend, RenderContext
    from ezdxf.addons.drawing.config import BackgroundPolicy, Configuration
    from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

    doc = ezdxf.readfile(str(dxf_path))
    pw, ph = 210.0, 297.0
    try:
        extmin = doc.header["$EXTMIN"]
        extmax = doc.header["$EXTMAX"]
        pw = abs(extmax[0] - extmin[0]) or pw
        ph = abs(extmax[1] - extmin[1]) or ph
    except Exception:
        pass

    fig = plt.figure(figsize=(pw / 25.4, ph / 25.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("white")

    cfg = Configuration.defaults().with_changes(
        background_policy=BackgroundPolicy.WHITE
    )
    ctx = RenderContext(doc)
    backend = MatplotlibBackend(ax)
    Frontend(ctx, backend, config=cfg).draw_layout(doc.modelspace(), finalize=True)
    ax.set_axis_off()
    fig.savefig(
        str(png_path), dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    plt.close(fig)


# ── convenience factory ───────────────────────────────────────────────────────


def draw_element(
    element,
    spec,
    element_index: int = 0,
    paper: str = "A4",
    orientation: str = "portrait",
    style: DrawingStyle | None = None,
) -> ElementDrawing:
    """Build and generate an :class:`ElementDrawing` for *element* in one call."""
    d = ElementDrawing(
        element,
        spec,
        element_index,
        paper=paper,
        orientation=orientation,
        style=style,
    )
    d.generate()
    return d
