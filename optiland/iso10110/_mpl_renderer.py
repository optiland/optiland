"""ISO 10110 element drawing — matplotlib (PDF/PNG) renderer."""

from __future__ import annotations

import numpy as np

from optiland.iso10110._base_renderer import _BaseRenderer
from optiland.iso10110._geometry import _BOT_H, _Geo, _tol_math
from optiland.iso10110.style import DrawingStyle


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
