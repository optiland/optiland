"""ISO 10110 element drawing — DXF (ezdxf) renderer."""

from __future__ import annotations

from optiland.iso10110._base_renderer import _BaseRenderer
from optiland.iso10110._geometry import _BOT_H, _LAMBDA_R, _Geo, _tol_plain
from optiland.iso10110.style import DrawingStyle

# DXF layers
L_BORDER = "BORDER"
L_OUTLINE = "OUTLINE"
L_AXIS = "AXIS"
L_CEMENT = "CEMENT"
L_DIMS = "DIMS"
L_CALLOUT = "CALLOUT"
L_TABLE = "TABLE"


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
