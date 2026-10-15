"""Measured GUI layout for compatible retained two-dimensional analysis grids."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose
from typing import TYPE_CHECKING, Any

from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.layout_engine import LayoutEngine
from matplotlib.text import Text

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.axis import Axis
    from matplotlib.backend_bases import RendererBase
    from matplotlib.figure import Figure
    from matplotlib.legend import Legend


@dataclass
class _AxesLayout:
    axes: Axes
    position: tuple[float, float, float, float]
    column: int
    row: int
    xlabel_visible: bool
    labelbottom: bool
    last_position: tuple[float, float, float, float]
    last_xlabel_visible: bool
    last_labelbottom: bool
    allow_shared: bool = True
    hidden_by_layout: bool = False
    tick_visibility: dict[Text, bool] = field(default_factory=dict)
    offset_visible: bool = True


def _same(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return len(left) == len(right) and all(
        isclose(a, b, rel_tol=0, abs_tol=1e-8) for a, b in zip(left, right, strict=True)
    )


def _text_key(text: Text) -> tuple[Any, ...]:
    return (
        text.get_text(),
        str(text.get_fontproperties()),
        text.get_rotation(),
        text.get_visible(),
        text.get_ha(),
        text.get_va(),
    )


def _coordinate_key(axis: Axis) -> tuple[Any, ...]:
    locations = tuple(axis.get_majorticklocs())
    formatter = axis.get_major_formatter()
    return locations, tuple(formatter.format_ticks(locations)), formatter.get_offset()


def _axis_key(axis: Axis) -> tuple[Any, ...]:
    keys = []
    for which in ("major", "minor"):
        major = which == "major"
        locations = tuple(
            axis.get_majorticklocs() if major else axis.get_minorticklocs()
        )
        formatter = axis.get_major_formatter() if major else axis.get_minor_formatter()
        ticks = (
            axis.get_major_ticks(len(locations))
            if major
            else axis.get_minor_ticks(len(locations))
        )
        keys.append(
            (
                locations,
                tuple(formatter.format_ticks(locations)),
                formatter.get_offset(),
                tuple(
                    (_text_key(tick.label1)[1:], _text_key(tick.label2)[1:])
                    for tick in ticks
                ),
                tuple(
                    axis.get_tick_params(which=which).get(name)
                    for name in ("pad", "labelbottom", "labeltop")
                ),
            )
        )
    return (
        axis.labelpad,
        _text_key(axis.get_offset_text())[1:],
        tuple(keys),
    )


def _title_key(axes: Axes, *, points: bool = False) -> tuple[Any, ...]:
    origin = axes.transAxes.transform((0, 0))
    return tuple(
        (
            _text_key(text),
            tuple(
                round(float(v) / (axes.figure.dpi if points else 1), 8)
                for v in text.get_transform().transform((0, 0)) - origin
            ),
        )
        for text in axes.get_children()
        if isinstance(text, Text) and text not in axes.texts
    )


def _tick_span(axis: Axis, renderer: RendererBase) -> float:
    """Measure the active axis span needed by neighboring painted tick labels."""
    horizontal = axis is axis.axes.xaxis
    dimension = 0 if horizontal else 1
    span = axis.axes.bbox.width if horizontal else axis.axes.bbox.height
    if span <= 0:
        return 0.0
    limits = axis.get_view_interval()
    required = 0.0
    for label_name in ("label1", "label2"):
        entries = []
        for tick in (*axis.get_major_ticks(), *axis.get_minor_ticks()):
            text = getattr(tick, label_name)
            if (
                not text.get_visible()
                or not text.get_text()
                or not min(limits) <= tick.get_loc() <= max(limits)
            ):
                continue
            anchor = text.get_transform().transform(text.get_position())[dimension]
            box = text.get_window_extent(renderer)
            lower, upper = (box.x0, box.x1) if horizontal else (box.y0, box.y1)
            entries.append((float(anchor), float(lower), float(upper)))
        entries.sort()
        for left, right in zip(entries[:-1], entries[1:], strict=True):
            distance = right[0] - left[0]
            if distance > 0:
                required = max(
                    required,
                    (left[2] - left[0] + right[0] - right[1]) * span / distance,
                )
    return required


class AnalysisFigureLayout(LayoutEngine):
    """Reflow measured decorations, sharing only identical column coordinates.

    The prepared manual rectangles must describe a complete aligned grid.
    Insets, overlapping/twin axes, colorbars, nonlinear scales and figure text
    keep their scientific layout. No font, data, limit or locator is changed.
    This public Matplotlib layout engine also runs before vector and alternate
    DPI export; unchanged geometry/style uses its cached layout.
    """

    _adjust_compatible = True
    _colorbar_gridspec = False
    _PAD_POINTS = 4.0

    def __init__(self, figure: Figure) -> None:
        super().__init__()
        positions = [tuple(ax.get_position(original=True).bounds) for ax in figure.axes]
        columns = sorted(set((p[0], p[2]) for p in positions))
        rows = sorted(set((p[1], p[3]) for p in positions), reverse=True)
        self._columns = columns
        self._rows = rows
        self._axes = [
            _AxesLayout(
                ax,
                position,
                columns.index((position[0], position[2])),
                rows.index((position[1], position[3])),
                ax.xaxis.label.get_visible(),
                bool(ax.xaxis.get_tick_params().get("labelbottom", True)),
                position,
                ax.xaxis.label.get_visible(),
                bool(ax.xaxis.get_tick_params().get("labelbottom", True)),
            )
            for ax, position in zip(figure.axes, positions, strict=True)
        ]
        self._legend: Legend | None = figure.legends[0] if figure.legends else None
        self._legends = tuple(figure.legends)
        self._legend_anchor = (
            tuple(
                self._legend.get_bbox_to_anchor()
                .transformed(figure.transFigure.inverted())
                .bounds
            )
            if self._legend
            else None
        )
        self._last_anchor = self._legend_anchor
        self._last_dimensions: tuple[Any, ...] | None = None
        self._last_legend_box: tuple[float, ...] | None = None
        self._last_legend_text: tuple[Any, ...] = ()
        self._cache_key: tuple[Any, ...] | None = None
        self._disabled = False
        self.minimum_height_inches = 0.0
        self.minimum_width_inches = 0.0
        self._minimum_style_key: tuple[Any, ...] | None = None

    @classmethod
    def for_analysis(
        cls, figure: Figure, prepared: dict[str, Any]
    ) -> AnalysisFigureLayout | None:
        """Return a policy only when captured rectangles prove a supported grid."""
        axes = figure.axes
        if (
            len(axes) < 2
            or len(axes) != len(prepared["axes"])
            or figure.texts
            or len(figure.legends) > 1
            or any(data["colorbar"] or not data["visible"] for data in prepared["axes"])
            or any(
                ax.name != "rectilinear"
                or ax.child_axes
                or ax.get_xscale() != "linear"
                or ax.get_yscale() != "linear"
                or ax.get_adjustable() != "box"
                for ax in axes
            )
        ):
            return None
        positions = [tuple(ax.get_position(original=True).bounds) for ax in axes]
        columns = sorted(set((p[0], p[2]) for p in positions))
        rows = sorted(set((p[1], p[3]) for p in positions))
        if (
            len(columns) * len(rows) != len(positions)
            or len(set(positions)) != len(positions)
            or any(
                a[0] + a[1] > b[0] + 1e-8
                for a, b in zip(columns[:-1], columns[1:], strict=True)
            )
            or any(
                a[0] + a[1] > b[0] + 1e-8
                for a, b in zip(rows[:-1], rows[1:], strict=True)
            )
        ):
            return None
        return cls(figure)

    def set(self, **kwargs: Any) -> None:
        """Keep this bounded GUI policy free of undocumented layout options."""
        if kwargs:
            raise TypeError("AnalysisFigureLayout has no configurable parameters")

    def navigation_restored(self) -> None:
        """Accept rectangles restored by the toolbar's public history actions."""
        if not self._disabled:
            for state in self._axes:
                state.last_position = tuple(
                    state.axes.get_position(original=True).bounds
                )
            self._cache_key = None

    def _restore_labels(self) -> None:
        for state in self._axes:
            if state.hidden_by_layout:
                state.axes.xaxis.label.set_visible(state.xlabel_visible)
                state.axes.tick_params(axis="x", labelbottom=state.labelbottom)
                for text, visible in state.tick_visibility.items():
                    text.set_visible(visible)
                state.axes.xaxis.get_offset_text().set_visible(state.offset_visible)
                state.hidden_by_layout = False
                state.tick_visibility.clear()

    def _restore(self, figure: Figure) -> None:
        self._restore_labels()
        for state in self._axes:
            state.axes.set_position(state.position)
        if self._legend and self._legend_anchor:
            self._legend.set_bbox_to_anchor(self._legend_anchor, figure.transFigure)

    def _remember(self, figure: Figure, renderer: RendererBase) -> None:
        for state in self._axes:
            state.last_position = tuple(state.axes.get_position(original=True).bounds)
            state.last_xlabel_visible = state.axes.xaxis.label.get_visible()
            state.last_labelbottom = bool(
                state.axes.xaxis.get_tick_params().get("labelbottom", True)
            )
        if self._legend:
            self._last_anchor = tuple(
                self._legend.get_bbox_to_anchor()
                .transformed(figure.transFigure.inverted())
                .bounds
            )
            self._last_legend_box = tuple(
                self._legend.get_window_extent(renderer).bounds
            )
            self._last_legend_text = tuple(
                _text_key(text) for text in self._legend.get_texts()
            )
        self._last_dimensions = (tuple(figure.bbox.bounds), figure.dpi)

    def _key(self, figure: Figure, renderer: RendererBase) -> tuple[Any, ...]:
        # Axis.draw computes decoration positions in display pixels. They are
        # outputs of layout, so including them would invalidate every draw.
        return (
            tuple(figure.bbox.bounds),
            figure.dpi,
            tuple(
                (
                    tuple(s.axes.get_xlim()),
                    tuple(s.axes.get_ylim()),
                    s.axes.get_aspect(),
                    s.axes.get_data_ratio(),
                    _text_key(s.axes.xaxis.label),
                    _text_key(s.axes.yaxis.label),
                    _title_key(s.axes),
                    _axis_key(s.axes.xaxis),
                    _axis_key(s.axes.yaxis),
                    s.xlabel_visible,
                    s.labelbottom,
                    s.allow_shared,
                )
                for s in self._axes
            ),
            tuple(_text_key(t) for t in self._legend.get_texts())
            if self._legend
            else (),
            tuple(self._legend.get_window_extent(renderer).bounds)
            if self._legend
            else (),
        )

    def _height_style_key(self) -> tuple[Any, ...]:
        """Typography/spacing inputs independent of automatic tick counts.

        A changed public style gets a fresh chance to fit the viewport. Tick
        visibility changed by coordinate consolidation is represented by the
        original intent, so expanding the canvas cannot clear its own minimum.
        """
        result = []
        for state in self._axes:
            axes = state.axes
            axis_styles = []
            for axis in (axes.xaxis, axes.yaxis):
                tick_styles = []
                for ticks in (axis.get_major_ticks(), axis.get_minor_ticks()):
                    for label in ("label1", "label2"):
                        styles = set()
                        for tick in ticks:
                            text = getattr(tick, label)
                            visible = state.tick_visibility.get(
                                text, text.get_visible()
                            )
                            style = _text_key(text)
                            styles.add(
                                (style[1], style[2], visible, style[4], style[5])
                            )
                        tick_styles.append(frozenset(styles))
                text = _text_key(axis.label)
                visible = state.xlabel_visible if axis is axes.xaxis else text[3]
                offset = _text_key(axis.get_offset_text())
                offset_visible = (
                    state.offset_visible
                    if axis is axes.xaxis and state.hidden_by_layout
                    else offset[3]
                )
                axis_styles.append(
                    (
                        text[:3] + (visible,) + text[4:],
                        axis.labelpad,
                        offset[1:3] + (offset_visible,) + offset[4:],
                        tuple(tick_styles),
                        id(axis.get_major_locator()),
                        id(axis.get_major_formatter()),
                        id(axis.get_minor_locator()),
                        id(axis.get_minor_formatter()),
                    )
                )
            result.append(
                (
                    _title_key(axes, points=True),
                    axes.get_aspect(),
                    axes.get_data_ratio(),
                    tuple(axis_styles),
                )
            )
        return tuple(result)

    @staticmethod
    def _renderer(figure: Figure) -> RendererBase:
        get_renderer = getattr(figure.canvas, "get_renderer", None)
        if get_renderer:
            return get_renderer()
        # Vector canvases expose no renderer getter. Agg supplies public text
        # metrics at the export figure's actual dimensions and point/DPI scale.
        return RendererAgg(figure.bbox.width, figure.bbox.height, figure.dpi)

    def execute(self, figure: Figure) -> None:
        if self._disabled:
            return
        if (
            tuple(figure.axes) != tuple(s.axes for s in self._axes)
            or tuple(figure.legends) != self._legends
            or figure.texts
        ):
            self._restore_labels()
            self.minimum_height_inches = 0.0
            self.minimum_width_inches = 0.0
            self._disabled = True
            return
        if any(
            not _same(tuple(s.axes.get_position(original=True).bounds), s.last_position)
            for s in self._axes
        ):
            # A public manual position edit owns geometry from this point on.
            self._restore_labels()
            self.minimum_height_inches = 0.0
            self.minimum_width_inches = 0.0
            self._disabled = True
            return
        for state in self._axes:
            visible = state.axes.xaxis.label.get_visible()
            labelbottom = bool(
                state.axes.xaxis.get_tick_params().get("labelbottom", True)
            )
            if (
                visible != state.last_xlabel_visible
                or labelbottom != state.last_labelbottom
            ):
                state.xlabel_visible = visible
                state.labelbottom = labelbottom
                state.allow_shared = False
            if state.hidden_by_layout:
                if state.axes.xaxis.get_offset_text().get_visible():
                    state.offset_visible = True
                    state.allow_shared = False
                for text in state.tick_visibility:
                    if text.get_visible():
                        state.tick_visibility[text] = True
                        state.allow_shared = False
        renderer = self._renderer(figure)
        display = not figure.canvas.is_saving()
        if self._legend and self._last_legend_box is not None:
            anchor = tuple(
                self._legend.get_bbox_to_anchor()
                .transformed(figure.transFigure.inverted())
                .bounds
            )
            changed_anchor = not _same(anchor, self._last_anchor)
            dimensions = (tuple(figure.bbox.bounds), figure.dpi)
            changed_text = (
                tuple(_text_key(t) for t in self._legend.get_texts())
                != self._last_legend_text
            )
            legend_box = tuple(self._legend.get_window_extent(renderer).bounds)
            old_bounds, old_dpi = self._last_dimensions
            # A point anchor's legend offset reflects public placement/column
            # changes even when the canvas was resized in the same draw.
            offset = (
                (legend_box[0] - anchor[0] * figure.bbox.width) / figure.dpi,
                (legend_box[1] - anchor[1] * figure.bbox.height) / figure.dpi,
            )
            old_offset = (
                (self._last_legend_box[0] - self._last_anchor[0] * old_bounds[2])
                / old_dpi,
                (self._last_legend_box[1] - self._last_anchor[1] * old_bounds[3])
                / old_dpi,
            )
            if figure.dpi == old_dpi:
                changed_placement = not _same(offset, old_offset)
            else:
                # Font rasterization changes intrinsic legend dimensions
                # across DPI/backends. Anchor fractions still distinguish its
                # placement without treating those metrics as a manual edit.
                fractions = (
                    (legend_box[0] - anchor[0] * figure.bbox.width) / legend_box[2],
                    (legend_box[1] - anchor[1] * figure.bbox.height) / legend_box[3],
                )
                old_fractions = (
                    (self._last_legend_box[0] - self._last_anchor[0] * old_bounds[2])
                    / self._last_legend_box[2],
                    (self._last_legend_box[1] - self._last_anchor[1] * old_bounds[3])
                    / self._last_legend_box[3],
                )
                changed_placement = any(
                    abs(a - b) > 0.1
                    for a, b in zip(fractions, old_fractions, strict=True)
                )
            changed_style = (
                changed_text
                or changed_placement
                or dimensions == self._last_dimensions
                and (
                    not _same(
                        tuple(self._legend.get_window_extent(renderer).bounds),
                        self._last_legend_box,
                    )
                    or tuple(_text_key(t) for t in self._legend.get_texts())
                    != self._last_legend_text
                )
            )
            if changed_anchor or changed_style:
                # Public manual legend placement/style owns its layout.
                self._restore_labels()
                self.minimum_height_inches = 0.0
                self.minimum_width_inches = 0.0
                self._disabled = True
                return
        if any(not state.axes.get_visible() for state in self._axes):
            # Hidden axes have no tight bounding box. Keep the original layout
            # until the complete grid is visible again, then measure afresh.
            self._restore(figure)
            self.minimum_height_inches = 0.0
            self.minimum_width_inches = 0.0
            self._remember(figure, renderer)
            self._cache_key = None
            return
        key = self._key(figure, renderer)
        if key == self._cache_key:
            return
        if (
            display
            and (self.minimum_height_inches or self.minimum_width_inches)
            and self._height_style_key() != self._minimum_style_key
        ):
            self.minimum_height_inches = 0.0
            self.minimum_width_inches = 0.0
        self._restore(figure)
        width, height = figure.bbox.width, figure.bbox.height
        padding = self._PAD_POINTS * figure.dpi / 72
        boxes = [state.axes.get_tightbbox(renderer) for state in self._axes]
        clipped = any(
            box.x0 < 0 or box.y0 < 0 or box.x1 > width or box.y1 > height
            for box in boxes
        )
        overlap = any(
            a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1 :]
        )
        legend_box = self._legend.get_window_extent(renderer) if self._legend else None
        if legend_box:
            clipped |= (
                legend_box.x0 < 0
                or legend_box.y0 < 0
                or legend_box.x1 > width
                or legend_box.y1 > height
            )
            overlap |= any(legend_box.overlaps(box) for box in boxes)
        if (
            not clipped
            and not overlap
            and not any(
                _tick_span(state.axes.xaxis, renderer) > state.axes.bbox.width + 0.5
                or _tick_span(state.axes.yaxis, renderer) > state.axes.bbox.height + 0.5
                for state in self._axes
            )
        ):
            self._remember(figure, renderer)
            self._cache_key = key
            return
        if self._legend_anchor and self._legend_anchor[1] + self._legend_anchor[
            3
        ] > min(state.position[1] for state in self._axes):
            # An intentionally internal/manual figure legend keeps its position.
            self._remember(figure, renderer)
            self._cache_key = key
            return
        bottom_legend = legend_box.height + padding if legend_box else 0.0
        if display and legend_box and legend_box.width + 2 * padding > width:
            self.minimum_width_inches = (legend_box.width + 2 * padding) / figure.dpi
        full_row_padding = sum(
            max(
                s.axes.bbox.y0 - box.y0
                for s, box in zip(self._axes, boxes, strict=True)
                if s.row == r
            )
            + max(
                box.y1 - s.axes.bbox.y1
                for s, box in zip(self._axes, boxes, strict=True)
                if s.row == r
            )
            for r in range(len(self._rows))
        )
        full_data_height = (
            height - full_row_padding - bottom_legend - padding * (len(self._rows) + 1)
        )
        text_sized_plot_height = (
            2
            * max(
                max(s.axes.title.get_fontsize(), s.axes.xaxis.label.get_fontsize())
                for s in self._axes
            )
            * figure.dpi
            / 72
        )
        needs_shared_coordinates = (
            full_data_height / len(self._rows) < text_sized_plot_height
        )
        for column in range(len(self._columns)):
            states = [s for s in self._axes if s.column == column]
            first = states[0].axes
            if (
                needs_shared_coordinates
                and len(states) > 1
                and all(
                    s.axes.get_xscale() == "linear"
                    and _same(tuple(s.axes.get_xlim()), tuple(first.get_xlim()))
                    and s.axes.get_xlabel() == first.get_xlabel()
                    and _coordinate_key(s.axes.xaxis) == _coordinate_key(first.xaxis)
                    and s.labelbottom
                    and s.allow_shared
                    and s.xlabel_visible
                    for s in states
                )
            ):
                for state in states:
                    if state.row < len(self._rows) - 1:
                        state.tick_visibility = {
                            tick.label1: tick.label1.get_visible()
                            for tick in state.axes.xaxis.get_major_ticks()
                        }
                        state.hidden_by_layout = True
                        state.offset_visible = (
                            state.axes.xaxis.get_offset_text().get_visible()
                        )
                        state.axes.xaxis.label.set_visible(False)
                        state.axes.tick_params(axis="x", labelbottom=False)
        # Reuse the measured rectangles when no labels were consolidated. A
        # second measurement is needed only after changing their visibility.
        if any(state.hidden_by_layout for state in self._axes):
            boxes = [state.axes.get_tightbbox(renderer) for state in self._axes]
        for iteration in range(3):
            previous = [
                tuple(state.axes.get_position(original=True).bounds)
                for state in self._axes
            ]
            if iteration:
                boxes = [state.axes.get_tightbbox(renderer) for state in self._axes]
            pads = [
                (
                    max(0, state.axes.bbox.x0 - box.x0),
                    max(0, box.x1 - state.axes.bbox.x1),
                    max(0, state.axes.bbox.y0 - box.y0),
                    max(0, box.y1 - state.axes.bbox.y1),
                )
                for state, box in zip(self._axes, boxes, strict=True)
            ]
            left = [
                max(pads[i][0] for i, s in enumerate(self._axes) if s.column == c)
                for c in range(len(self._columns))
            ]
            right = [
                max(pads[i][1] for i, s in enumerate(self._axes) if s.column == c)
                for c in range(len(self._columns))
            ]
            bottom = [
                max(pads[i][2] for i, s in enumerate(self._axes) if s.row == r)
                for r in range(len(self._rows))
            ]
            top = [
                max(pads[i][3] for i, s in enumerate(self._axes) if s.row == r)
                for r in range(len(self._rows))
            ]
            data_width = width - sum(left) - sum(right) - padding * (len(left) + 1)
            data_height = (
                height
                - sum(bottom)
                - sum(top)
                - bottom_legend
                - padding * (len(bottom) + 1)
            )
            # A positive but tiny data rectangle can still collide with its
            # own tick labels and field title. Reserve a text-sized plot area
            # as well as measured decorations; preserve the grid's proportions.
            row_weights = [r[1] for r in self._rows]
            column_weights = [c[1] for c in self._columns]
            tick_floors = []
            for state in self._axes:
                axes = state.axes
                tick_width = _tick_span(axes.xaxis, renderer)
                tick_height = _tick_span(axes.yaxis, renderer)
                if axes.get_aspect() != "auto":
                    ratio = float(axes.get_aspect()) * axes.get_data_ratio()
                    tick_width, tick_height = (
                        max(tick_width, tick_height / ratio),
                        max(tick_height, tick_width * ratio),
                    )
                tick_floors.append((tick_width, tick_height))
            row_floors = [
                max(
                    text_sized_plot_height,
                    max(
                        tick_floors[i][1]
                        for i, s in enumerate(self._axes)
                        if s.row == row
                    ),
                )
                for row in range(len(self._rows))
            ]
            required_height = max(
                floor * sum(row_weights) / weight
                for floor, weight in zip(row_floors, row_weights, strict=True)
            )
            column_floors = [
                max(
                    text_sized_plot_height,
                    max(
                        tick_floors[i][0]
                        for i, s in enumerate(self._axes)
                        if s.column == column
                    ),
                    max(
                        text.get_window_extent(renderer).width
                        for state in self._axes
                        if state.column == column
                        for text in state.axes.get_children()
                        if isinstance(text, Text) and text not in state.axes.texts
                    )
                    - left[column]
                    - right[column],
                )
                for column in range(len(self._columns))
            ]
            required_width = max(
                floor * sum(column_weights) / weight
                for floor, weight in zip(column_floors, column_weights, strict=True)
            )
            if display and data_height < required_height:
                self.minimum_height_inches = (
                    height - data_height + required_height
                ) / figure.dpi
            if display and data_width < required_width:
                self.minimum_width_inches = max(
                    self.minimum_width_inches,
                    (width - data_width + required_width) / figure.dpi,
                )
            if data_width <= 0 or data_height <= 0:
                self._restore(figure)
                break
            widths = [data_width * w / sum(column_weights) for w in column_weights]
            heights = [data_height * h / sum(row_weights) for h in row_weights]
            xs = [padding + left[0]]
            for c in range(1, len(widths)):
                xs.append(xs[-1] + widths[c - 1] + right[c - 1] + padding + left[c])
            ys = [height - padding - top[0] - heights[0]]
            for r in range(1, len(heights)):
                ys.append(ys[-1] - bottom[r - 1] - padding - top[r] - heights[r])
            for state in self._axes:
                c, r = state.column, state.row
                state.axes.set_position(
                    (
                        xs[c] / width,
                        ys[r] / height,
                        widths[c] / width,
                        heights[r] / height,
                    )
                )
                state.axes.apply_aspect()
            if all(
                _same(before, tuple(state.axes.get_position(original=True).bounds))
                for before, state in zip(previous, self._axes, strict=True)
            ):
                break
        if legend_box and self._legend and self._legend_anchor:
            x, y, w, h = self._legend_anchor
            dx = (width - legend_box.width) / 2 - legend_box.x0
            dy = padding - legend_box.y0
            self._legend.set_bbox_to_anchor(
                (x + dx / width, y + dy / height, w, h), figure.transFigure
            )
        self._remember(figure, renderer)
        self._cache_key = self._key(figure, renderer)
        if display and (self.minimum_height_inches or self.minimum_width_inches):
            # Record final typography after restoration and tick materialization.
            self._minimum_style_key = self._height_style_key()
