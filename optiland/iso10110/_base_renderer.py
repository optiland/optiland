"""Shared ISO 10110 layout logic, expressed via format-specific primitives.

:class:`_BaseRenderer` is subclassed by both
:class:`~optiland.iso10110._mpl_renderer._MatplotlibRenderer` and
:class:`~optiland.iso10110._dxf_renderer._DxfRenderer`.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np

import optiland.backend as be
from optiland.iso10110._geometry import (
    _BOT_H,
    _LAMBDA_R,
    _SPEC_H,
    _SYM_ARM,
    _SYM_BAR,
    _SYM_TRI,
    _TTL_H,
    _TTL_R1_H,
    _TTL_R2_H,
    _iso10110_11_defaults,
    _mat_display_name,
    _material_code_rows,
    _numbered_code_rows,
    _surf_r,
    _surf_z,
    _surface_header_lines,
    _sym_transform,
    _to_float,
)


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
