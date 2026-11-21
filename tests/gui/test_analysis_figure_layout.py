"""Compact retained plots remain readable without changing scientific state."""

from __future__ import annotations

import hashlib
import io
import pickle
import struct
import threading
from pathlib import Path

import numpy as np
import pytest
from matplotlib import get_data_path
from matplotlib.figure import Figure
from matplotlib.ticker import FixedFormatter, FixedLocator, MaxNLocator
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtGui import QFont, QFontDatabase, QFontInfo, QFontMetrics, QRawFont
from shiboken6 import isValid

from optiland.samples.objectives import CookeTriplet, DoubleGauss
from optiland_gui.analysis_panel import AnalysisPanel, CustomMatplotlibToolbar
from optiland_gui.services.analysis_plots import draw_analysis_plot
from optiland_gui.services.analysis_worker import prepare_analysis
from optiland_gui.services.job_records import OpticSnapshot
from optiland_gui.widgets import analysis_figure_layout as figure_layout
from optiland_gui.widgets.analysis_figure_layout import AnalysisFigureLayout
from optiland_gui.widgets.retained_analysis_canvas import RetainedAnalysisCanvas
from tests.gui.analysis_job_fakes import connector_for
from tests.gui.test_calculation_jobs import wait_for


@pytest.fixture
def plot_panel_font(qapp):
    """Use a required portable font for Qt geometry, restoring global state."""
    path = Path(get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    assert path.is_file(), f"Required plot panel font is missing: {path}"
    font_id = QFontDatabase.addApplicationFont(str(path))
    assert font_id >= 0, f"Qt could not load the plot panel font: {path}"
    previous = QFont(qapp.font())
    try:
        families = QFontDatabase.applicationFontFamilies(font_id)
        assert families
        font = QFont(families[0], 9)
        assert QFontInfo(font).family() == families[0]
        assert QRawFont.fromFont(font).isValid()
        assert QFontMetrics(font).horizontalAdvance("Field 123") > 0
        qapp.setFont(font)
        yield
    finally:
        qapp.setFont(previous)
        assert QFontDatabase.removeApplicationFont(font_id)


@pytest.fixture(scope="module")
def optical_plots():
    plots = {}
    for optic_type, name, arguments in (
        (CookeTriplet, "Ray Fan", {"num_points": 128}),
        (DoubleGauss, "Spot Diagram", {"num_rings": 6}),
    ):
        plots[name] = prepare_analysis(
            OpticSnapshot.capture(optic_type()),
            {
                "name": name,
                "constructor_args": arguments,
                "view_args": {},
                "theme": "dark",
            },
            lambda *_args, **_kwargs: None,
            threading.Event(),
        )["plot"]
    return plots


@pytest.fixture
def canvases(qapp):
    result = []

    def make(prepared=None, theme="light", width=921, height=279, dpi=100):
        figure = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        canvas = RetainedAnalysisCanvas(figure)
        result.append(canvas)
        if prepared is not None:
            draw_analysis_plot(figure, prepared, theme)
            canvas.set_analysis_layout(prepared)
        return canvas

    yield make
    for canvas in result:
        if isValid(canvas):
            canvas.close()
            canvas.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def decorations(figure):
    """Only tick labels inside limits are actually painted by Matplotlib."""
    result = []
    for axes in figure.axes:
        result.extend([axes.title, axes.xaxis.label, axes.yaxis.label])
        for axis, limits in (
            (axes.xaxis, axes.get_xlim()),
            (axes.yaxis, axes.get_ylim()),
        ):
            result.append(axis.get_offset_text())
            result.extend(
                label
                for tick in axis.get_major_ticks()
                if min(limits) - 1e-8 <= tick.get_loc() <= max(limits) + 1e-8
                for label in (tick.label1, tick.label2)
            )
    return [text for text in result if text.get_visible() and text.get_text()] + list(
        figure.legends
    )


def assert_figure_readable(figure, renderer):
    boxes = [text.get_window_extent(renderer) for text in decorations(figure)]
    width, height = figure.bbox.width, figure.bbox.height
    for box in boxes:
        assert box.x0 >= -0.5
        assert box.y0 >= -0.5
        assert box.x1 <= width + 0.5
        assert box.y1 <= height + 0.5
    for index, left in enumerate(boxes):
        for right in boxes[index + 1 :]:
            assert (
                min(left.x1, right.x1) - max(left.x0, right.x0) <= 0.5
                or min(left.y1, right.y1) - max(left.y0, right.y0) <= 0.5
            )


def assert_readable(canvas):
    assert_figure_readable(canvas.figure, canvas.get_renderer())


def scientific_state(figure):
    return pickle.dumps(
        [
            (
                axes.get_xlim(),
                axes.get_ylim(),
                axes.get_title(),
                axes.get_xlabel(),
                axes.get_ylabel(),
                str(axes.title.get_fontproperties()),
                str(axes.xaxis.label.get_fontproperties()),
                str(axes.yaxis.label.get_fontproperties()),
                [line.get_xydata() for line in axes.lines],
                [collection.get_offsets() for collection in axes.collections],
            )
            for axes in figure.axes
        ],
        protocol=5,
    )


def positions(figure):
    return [tuple(axes.get_position(original=True).bounds) for axes in figure.axes]


def install_grid(canvas, rows=2, columns=2):
    for row in range(rows):
        for column in range(columns):
            axes = canvas.figure.add_axes(
                (0.14 + column * 0.42, 0.57 - row * 0.39, 0.3, 0.27)
            )
            axes.plot([-1, 0, 1], [-1, 0, 1])
            axes.set(
                xlabel="Pupil coordinate", ylabel="Aberration", title=f"Field {row}"
            )
    prepared = {
        "axes": [{"visible": True, "colorbar": False} for _ in canvas.figure.axes]
    }
    canvas.set_analysis_layout(prepared)
    return prepared


@pytest.mark.parametrize("name", ["Ray Fan", "Spot Diagram"])
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_real_compact_plots_preserve_state_and_remove_collisions(
    canvases, optical_plots, name, theme
):
    prepared = optical_plots[name]
    original_payload = pickle.dumps(prepared, protocol=5)
    canvas = canvases(prepared, theme)
    figure = canvas.figure
    before = scientific_state(figure)
    artists = [(tuple(ax.lines), tuple(ax.collections)) for ax in figure.axes]
    locators = [
        (ax.xaxis.get_major_locator(), ax.yaxis.get_major_locator())
        for ax in figure.axes
    ]
    legend_text = [text.get_text() for text in figure.legends[0].get_texts()]
    canvas.draw()
    assert_readable(canvas)
    assert scientific_state(figure) == before
    assert pickle.dumps(prepared, protocol=5) == original_payload
    assert [(tuple(ax.lines), tuple(ax.collections)) for ax in figure.axes] == artists
    assert [
        (ax.xaxis.get_major_locator(), ax.yaxis.get_major_locator())
        for ax in figure.axes
    ] == locators
    assert [text.get_text() for text in figure.legends[0].get_texts()] == legend_text
    assert all(
        ax.title.get_visible() and ax.yaxis.label.get_visible() for ax in figure.axes
    )
    if name == "Ray Fan":
        assert [ax.xaxis.label.get_visible() for ax in figure.axes] == [False] * 4 + [
            True
        ] * 2


@pytest.mark.parametrize("name", ["Ray Fan", "Spot Diagram"])
def test_resize_cycles_restore_labels_without_layout_drift(
    canvases, optical_plots, name
):
    canvas = canvases(optical_plots[name])
    figure = canvas.figure
    canvas.draw()
    compact = positions(figure)
    before = scientific_state(figure)
    for _ in range(3):
        figure.set_size_inches(9.21, 6, forward=False)
        canvas.draw()
        assert_readable(canvas)
        assert all(ax.xaxis.label.get_visible() for ax in figure.axes)
        assert all(ax.xaxis.get_tick_params()["labelbottom"] for ax in figure.axes)
        figure.set_size_inches(9.21, 2.79, forward=False)
        canvas.draw()
        assert_readable(canvas)
        np.testing.assert_allclose(positions(figure), compact, atol=1e-8, rtol=0)
    assert scientific_state(figure) == before


def test_clipped_fan_avoids_redundant_tick_acceptance_measurements(
    canvases, optical_plots, monkeypatch
):
    canvas = canvases(optical_plots["Ray Fan"])
    before = scientific_state(canvas.figure)
    calls = []
    original = figure_layout._tick_span

    def measured(axis, renderer):
        calls.append(axis)
        return original(axis, renderer)

    monkeypatch.setattr(figure_layout, "_tick_span", measured)
    canvas.draw()
    # The fixed compact Fan previously made 48 tick-span measurements. Its
    # required solver work fits within 36; clipping already rejects the
    # original geometry before the additional 12 acceptance measurements.
    assert 0 < len(calls) <= 36
    assert_readable(canvas)
    assert scientific_state(canvas.figure) == before
    calls.clear()
    canvas.draw()
    assert calls == []


@pytest.mark.parametrize("which", ["major", "minor"])
@pytest.mark.parametrize("reverse", [False, True])
def test_fitting_grid_still_reflows_colliding_coordinate_ticks(
    canvases, which, reverse
):
    canvas = canvases(width=1000, height=450)
    for column in range(2):
        axes = canvas.figure.add_axes((0.14 + 0.5 * column, 0.4, 0.12, 0.27))
        axes.plot([-1, 1], [-1, 1])
        axes.set(
            xlim=(1, -1) if reverse else (-1, 1),
            ylim=(-1, 1),
            aspect="equal",
            title=f"Field {column}",
        )
        locator = FixedLocator([-0.5, 0.5])
        formatter = FixedFormatter(["negative coordinate", "positive coordinate"])
        if which == "major":
            axes.xaxis.set_major_locator(locator)
            axes.xaxis.set_major_formatter(formatter)
        else:
            axes.xaxis.set_major_locator(FixedLocator([]))
            axes.xaxis.set_minor_locator(locator)
            axes.xaxis.set_minor_formatter(formatter)
        axes.tick_params(axis="x", which=which, labelbottom=True)
        if reverse:
            axes.tick_params(axis="x", which=which, labelrotation=30)
            for text in axes.get_xticklabels(minor=which == "minor"):
                text.set_ha("left")
    canvas.draw()
    renderer = canvas.get_renderer()
    boxes = [ax.get_tightbbox(renderer) for ax in canvas.figure.axes]
    assert all(
        box.x0 >= 0 and box.x1 <= 1000 and box.y0 >= 0 and box.y1 <= 450
        for box in boxes
    )
    assert not boxes[0].overlaps(boxes[1])
    ticks = canvas.figure.axes[0].get_xticklabels(minor=which == "minor")
    assert (
        ticks[0]
        .get_window_extent(renderer)
        .overlaps(ticks[1].get_window_extent(renderer))
    )
    before = scientific_state(canvas.figure)
    original_positions = positions(canvas.figure)
    prepared = {
        "axes": [{"visible": True, "colorbar": False} for _ in canvas.figure.axes]
    }
    canvas.set_analysis_layout(prepared)
    canvas.draw()
    assert positions(canvas.figure) != original_positions
    assert_readable(canvas)
    assert scientific_state(canvas.figure) == before
    for axes in canvas.figure.axes:
        assert axes.get_aspect() == 1
        labels = axes.get_xticklabels(minor=which == "minor")
        assert [label.get_text() for label in labels] == formatter.seq
        # The general decoration helper covers major ticks; explicitly check
        # this coordinate presentation so minor labels get the same guarantee.
        final_boxes = [
            label.get_window_extent(canvas.get_renderer()) for label in labels
        ]
        assert all(
            box.x0 >= -0.5 and box.x1 <= 1000.5 and box.y0 >= -0.5 and box.y1 <= 450.5
            for box in final_boxes
        )
        left, right = final_boxes
        assert (
            min(left.x1, right.x1) - max(left.x0, right.x0) <= 0.5
            or min(left.y1, right.y1) - max(left.y0, right.y0) <= 0.5
        )


def test_minimum_typography_is_recorded_once_after_the_fan_solve(
    canvases, optical_plots, monkeypatch
):
    # Leave enough room to paint all labels, while requiring a measured
    # minimum on supported Matplotlib versions with different text metrics.
    canvas = canvases(optical_plots["Ray Fan"], height=250)
    engine = canvas.figure.get_layout_engine()
    before = scientific_state(canvas.figure)
    calls = []
    original = engine._height_style_key

    def measured():
        calls.append(True)
        return original()

    monkeypatch.setattr(engine, "_height_style_key", measured)
    canvas.draw()
    assert engine.minimum_height_inches or engine.minimum_width_inches
    # Multiple solve passes must retain only the final typography for the
    # next resize. There is no incoming minimum yet.
    assert len(calls) == 1
    assert_readable(canvas)
    assert scientific_state(canvas.figure) == before
    calls.clear()
    canvas.draw()
    assert calls == []


def test_minimum_typography_follows_small_grid_label_restoration(canvases, monkeypatch):
    canvas = canvases(width=60, height=60)
    install_grid(canvas)
    engine = canvas.figure.get_layout_engine()
    before = scientific_state(canvas.figure)
    original_positions = positions(canvas.figure)
    snapshots = []
    original = engine._height_style_key

    def measured():
        snapshots.append(
            [
                (
                    ax.xaxis.label.get_visible(),
                    ax.xaxis.get_tick_params()["labelbottom"],
                )
                for ax in canvas.figure.axes
            ]
        )
        return original()

    monkeypatch.setattr(engine, "_height_style_key", measured)
    canvas.draw()
    assert engine.minimum_height_inches > 0
    assert snapshots == [[(True, True)] * 4]
    assert positions(canvas.figure) == original_positions
    assert scientific_state(canvas.figure) == before


def test_public_font_reduction_remeasures_minimum_and_matches_fresh_layout(
    canvases, monkeypatch
):
    canvas = canvases(width=220, height=164)
    install_grid(canvas)
    # Establish overflow on both dimensions explicitly rather than depending
    # on the Matplotlib version's default font metrics.
    for axes in canvas.figure.axes:
        axes.title.set_fontsize(12)
        axes.xaxis.label.set_fontsize(12)
        axes.yaxis.label.set_fontsize(12)
        axes.tick_params(labelsize=12)
    canvas.draw()
    engine = canvas.figure.get_layout_engine()
    before_minimum = (engine.minimum_width_inches, engine.minimum_height_inches)
    assert all(value > 0 for value in before_minimum)
    fresh = canvases(width=220, height=164)
    install_grid(fresh)
    for candidate in (canvas, fresh):
        for axes in candidate.figure.axes:
            axes.title.set_fontsize(6)
            axes.xaxis.label.set_fontsize(6)
            axes.yaxis.label.set_fontsize(6)
            axes.tick_params(labelsize=6)
    before = scientific_state(canvas.figure)
    calls = []
    original = engine._height_style_key

    def measured():
        calls.append(True)
        return original()

    monkeypatch.setattr(engine, "_height_style_key", measured)
    canvas.draw()
    fresh.draw()
    # The incoming fingerprint invalidates the old minimum; the smaller public
    # fonts now fit, so no outgoing minimum snapshot is needed.
    assert len(calls) == 1
    after_minimum = (engine.minimum_width_inches, engine.minimum_height_inches)
    assert after_minimum == (0, 0)
    fresh_engine = fresh.figure.get_layout_engine()
    assert after_minimum == (
        fresh_engine.minimum_width_inches,
        fresh_engine.minimum_height_inches,
    )
    assert positions(canvas.figure) == positions(fresh.figure)
    assert (
        hashlib.sha256(canvas.buffer_rgba()).digest()
        == hashlib.sha256(fresh.buffer_rgba()).digest()
    )
    assert scientific_state(canvas.figure) == before


def test_unchanged_draw_uses_cached_geometry(canvases, optical_plots, monkeypatch):
    canvas = canvases(optical_plots["Ray Fan"])
    canvas.draw()
    before = positions(canvas.figure)
    calls = []
    for axes in canvas.figure.axes:
        original = axes.get_tightbbox

        def measured(*args, original=original, **kwargs):
            calls.append(True)
            return original(*args, **kwargs)

        monkeypatch.setattr(axes, "get_tightbbox", measured)
    canvas.draw()
    assert calls == []
    assert positions(canvas.figure) == before
    assert_readable(canvas)


def test_independent_zoom_restores_column_coordinates(canvases, optical_plots):
    canvas = canvases(optical_plots["Ray Fan"])
    canvas.draw()
    axes = canvas.figure.axes
    axes[0].set_xlim(-0.2, 0.6)
    canvas.draw()
    assert axes[0].get_xlim() == (-0.2, 0.6)
    assert all(axes[index].xaxis.label.get_visible() for index in (0, 2, 4))
    assert all(
        axes[index].xaxis.get_tick_params()["labelbottom"] for index in (0, 2, 4)
    )
    assert not axes[1].xaxis.label.get_visible()


def test_manual_tick_text_fonts_and_locator_parameters_are_preserved(canvases):
    canvas = canvases(height=450)
    install_grid(canvas)
    canvas.draw()
    axes = canvas.figure.axes[0]
    locator = FixedLocator([-1, 0, 1])
    formatter = FixedFormatter(["lower pupil", "center", "upper pupil"])
    axes.xaxis.set_major_locator(locator)
    axes.xaxis.set_major_formatter(formatter)
    axes.tick_params(axis="x", labelrotation=35)
    axes.set_xlabel("Manually named coordinate", fontfamily="DejaVu Serif", fontsize=13)
    axes.set_title("Manually titled field", fontfamily="DejaVu Serif", fontsize=14)
    canvas.draw()
    assert axes.xaxis.get_major_locator() is locator
    assert axes.xaxis.get_major_formatter() is formatter
    assert [text.get_text() for text in axes.get_xticklabels()] == formatter.seq
    assert all(text.get_rotation() == 35 for text in axes.get_xticklabels())
    assert axes.xaxis.label.get_fontfamily() == ["DejaVu Serif"]
    assert axes.xaxis.label.get_fontsize() == 13
    assert axes.get_title() == "Manually titled field"
    automatic = MaxNLocator(nbins=5)
    axes.yaxis.set_major_locator(automatic)
    canvas.draw()
    automatic.set_params(nbins=2)
    expected = automatic.tick_values(*axes.get_ylim())
    canvas.draw()
    assert axes.yaxis.get_major_locator() is automatic
    np.testing.assert_array_equal(axes.get_yticks(), expected)


@pytest.mark.parametrize(
    "edit", ["labelpad", "titlepad", "lefttitle", "offset", "minor"]
)
def test_public_spacing_and_decoration_styles_invalidate_cached_layout(
    canvases, monkeypatch, edit
):
    canvas = canvases(height=420)
    install_grid(canvas)
    axes = canvas.figure.axes[0]
    axes.xaxis.set_minor_locator(FixedLocator([-0.3, 0.3]))
    axes.xaxis.set_minor_formatter(FixedFormatter(["left", "right"]))
    canvas.draw()
    calls = []
    original = axes.get_tightbbox

    def measured(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(axes, "get_tightbbox", measured)
    if edit == "labelpad":
        axes.xaxis.labelpad = 12
    elif edit == "titlepad":
        axes.set_title(axes.get_title(), pad=15)
    elif edit == "lefttitle":
        axes.set_title("Side title", loc="left", fontsize=14)
    elif edit == "offset":
        axes.xaxis.get_offset_text().set_fontsize(15)
    else:
        axes.xaxis.get_minor_ticks()[0].label1.set_fontsize(15)
    canvas.draw()
    assert calls
    if edit == "labelpad":
        assert axes.xaxis.labelpad == 12
    elif edit == "lefttitle":
        assert axes.get_title(loc="left") == "Side title"
    elif edit == "offset":
        assert axes.xaxis.get_offset_text().get_fontsize() == 15
    elif edit == "minor":
        assert axes.xaxis.get_minor_ticks()[0].label1.get_fontsize() == 15
    calls.clear()
    canvas.draw()
    assert calls == []


def test_different_manual_coordinate_presentation_is_never_consolidated(
    canvases, optical_plots
):
    canvas = canvases(optical_plots["Ray Fan"])
    axes = canvas.figure.axes[0]
    labels = ["negative pupil", "center", "positive pupil"]
    axes.xaxis.set_major_locator(FixedLocator([-1, 0, 1]))
    axes.xaxis.set_major_formatter(FixedFormatter(labels))
    canvas.draw()
    assert axes.xaxis.label.get_visible()
    assert [label.get_text() for label in axes.get_xticklabels()] == labels
    assert all(label.get_visible() for label in axes.get_xticklabels())


def test_manually_restored_offset_visibility_retains_coordinate_ownership(canvases):
    canvas = canvases(width=500, height=200)
    install_grid(canvas, rows=2, columns=1)
    for axes in canvas.figure.axes:
        axes.set_xlim(1e6, 1e6 + 2)
    canvas.draw()
    top = canvas.figure.axes[0]
    offset = top.xaxis.get_offset_text()
    assert offset.get_text() == "+1e6"
    assert not offset.get_visible()
    offset.set_visible(True)
    canvas.draw()
    assert offset.get_visible()
    assert top.xaxis.label.get_visible()
    assert top.get_xlim() == (1e6, 1e6 + 2)


def test_manually_restored_individual_tick_visibility_is_preserved(
    canvases, optical_plots
):
    canvas = canvases(optical_plots["Ray Fan"])
    canvas.draw()
    top = canvas.figure.axes[0]
    tick = top.xaxis.get_major_ticks()[1].label1
    assert not tick.get_visible()
    tick.set_visible(True)
    canvas.draw()
    assert tick.get_visible()
    assert top.xaxis.label.get_visible()
    canvas.figure.set_size_inches(9.21, 6.2, forward=False)
    canvas.draw()
    assert tick.get_visible()


def test_manual_visibility_is_not_reset(canvases, optical_plots):
    canvas = canvases(optical_plots["Ray Fan"])
    canvas.draw()
    axes = canvas.figure.axes
    axes[4].xaxis.label.set_visible(False)
    axes[4].tick_params(axis="x", labelbottom=False)
    axes[1].xaxis.label.set_visible(True)
    canvas.draw()
    canvas.figure.set_size_inches(9.21, 6, forward=False)
    canvas.draw()
    assert not axes[4].xaxis.label.get_visible()
    assert not axes[4].xaxis.get_tick_params()["labelbottom"]
    assert axes[1].xaxis.label.get_visible()


def test_manual_axes_position_is_owned_by_user(canvases, optical_plots):
    canvas = canvases(optical_plots["Ray Fan"])
    canvas.draw()
    axes = canvas.figure.axes[0]
    manual = (0.2, 0.7, 0.3, 0.2)
    axes.set_position(manual)
    canvas.draw()
    np.testing.assert_allclose(axes.get_position(original=True).bounds, manual)
    assert all(ax.xaxis.label.get_visible() for ax in canvas.figure.axes)
    canvas.figure.set_size_inches(9.21, 6, forward=False)
    canvas.draw()
    np.testing.assert_allclose(axes.get_position(original=True).bounds, manual)


def test_hiding_and_reshowing_axes_restores_and_reflows_coordinates(
    canvases, optical_plots
):
    canvas = canvases(optical_plots["Ray Fan"])
    canvas.draw()
    axes = canvas.figure.axes[0]
    before = scientific_state(canvas.figure)
    compact = positions(canvas.figure)
    axes.set_visible(False)
    canvas.draw()
    assert all(ax.xaxis.label.get_visible() for ax in canvas.figure.axes)
    axes.set_visible(True)
    canvas.draw()
    assert_readable(canvas)
    np.testing.assert_allclose(positions(canvas.figure), compact, atol=1e-8, rtol=0)
    assert scientific_state(canvas.figure) == before


def test_manual_topology_change_restores_owned_coordinates(canvases, optical_plots):
    canvas = canvases(optical_plots["Ray Fan"])
    canvas.draw()
    axes = canvas.figure.axes[0]
    axes.twinx()
    canvas.draw()
    assert all(ax.xaxis.label.get_visible() for ax in canvas.figure.axes)
    assert all(ax.xaxis.get_tick_params()["labelbottom"] for ax in canvas.figure.axes)


@pytest.mark.parametrize("edit", ["anchor", "location", "columns", "font"])
def test_manual_legend_settings_are_owned_by_user(canvases, optical_plots, edit):
    canvas = canvases(optical_plots["Ray Fan"])
    canvas.draw()
    legend = canvas.figure.legends[0]
    if edit == "anchor":
        legend.set_bbox_to_anchor((0.5, 0.5), canvas.figure.transFigure)
    elif edit == "location":
        legend.set_loc("upper left")
    elif edit == "columns":
        legend.set_ncols(1)
    else:
        legend.get_texts()[0].set_fontfamily("DejaVu Serif")
        legend.get_texts()[0].set_fontsize(15)
    anchor = (
        legend.get_bbox_to_anchor()
        .transformed(canvas.figure.transFigure.inverted())
        .bounds
    )
    before = positions(canvas.figure)
    canvas.draw()
    np.testing.assert_allclose(
        legend.get_bbox_to_anchor()
        .transformed(canvas.figure.transFigure.inverted())
        .bounds,
        anchor,
    )
    assert positions(canvas.figure) == before
    if edit == "font":
        assert legend.get_texts()[0].get_fontsize() == 15


@pytest.mark.parametrize("edit", ["location", "font"])
def test_legend_edit_and_resize_in_same_draw_preserve_manual_ownership(
    canvases, optical_plots, edit
):
    canvas = canvases(optical_plots["Ray Fan"], height=164)
    canvas.draw()
    engine = canvas.figure.get_layout_engine()
    assert engine.minimum_height_inches > 0
    legend = canvas.figure.legends[0]
    if edit == "location":
        legend.set_loc("upper left")
    else:
        legend.get_texts()[0].set_fontsize(15)
    anchor = (
        legend.get_bbox_to_anchor()
        .transformed(canvas.figure.transFigure.inverted())
        .bounds
    )
    canvas.figure.set_size_inches(8, 3, forward=False)
    canvas.draw()
    np.testing.assert_allclose(
        legend.get_bbox_to_anchor()
        .transformed(canvas.figure.transFigure.inverted())
        .bounds,
        anchor,
    )
    assert engine.minimum_height_inches == 0
    assert engine.minimum_width_inches == 0
    assert all(ax.xaxis.label.get_visible() for ax in canvas.figure.axes)


@pytest.mark.parametrize("edit", ["add_legend", "replace_legend", "figure_text"])
def test_new_figure_decorations_restore_owned_labels_and_yield_layout(
    canvases, optical_plots, edit
):
    canvas = canvases(optical_plots["Ray Fan"], height=164)
    canvas.draw()
    engine = canvas.figure.get_layout_engine()
    assert engine.minimum_height_inches > 0
    if edit == "figure_text":
        canvas.figure.text(0.5, 0.5, "Manual annotation")
    else:
        if edit == "replace_legend":
            canvas.figure.legends[0].remove()
        canvas.figure.legend(canvas.figure.axes[0].lines[:1], ["Manual series"])
    canvas.draw()
    assert all(ax.xaxis.label.get_visible() for ax in canvas.figure.axes)
    assert engine.minimum_height_inches == 0
    assert engine.minimum_width_inches == 0


@pytest.mark.parametrize(
    "kind",
    [
        "colorbar",
        "twin",
        "inset",
        "3d",
        "log",
        "incomplete",
        "text",
        "hidden",
        "legends",
        "adjustable",
        "overlap_rows",
        "overlap_columns",
        "count",
    ],
)
def test_unsupported_scientific_layouts_remain_unchanged(canvases, kind):
    canvas = canvases()
    figure = canvas.figure
    prepared = install_grid(canvas)
    if kind == "colorbar":
        prepared["axes"][0]["colorbar"] = True
    elif kind == "twin":
        figure.axes[0].twinx()
        prepared["axes"].append({"visible": True, "colorbar": False})
    elif kind == "inset":
        figure.axes[0].inset_axes((0.4, 0.4, 0.3, 0.3))
    elif kind == "3d":
        figure.delaxes(figure.axes[0])
        figure.add_axes((0.14, 0.57, 0.3, 0.27), projection="3d")
    elif kind == "log":
        figure.axes[0].set_xscale("log")
    elif kind == "incomplete":
        figure.delaxes(figure.axes[-1])
        prepared["axes"].pop()
    elif kind == "text":
        figure.text(0.5, 0.5, "Scientific annotation")
    elif kind == "hidden":
        prepared["axes"][0]["visible"] = False
    elif kind == "legends":
        figure.legend(figure.axes[0].lines, ["One"])
        figure.legend(figure.axes[1].lines, ["Two"])
    elif kind == "adjustable":
        figure.axes[0].set_adjustable("datalim")
    elif kind == "count":
        prepared["axes"].pop()
    elif kind == "overlap_rows":
        for axes in figure.axes[2:]:
            x, _, width, height = axes.get_position().bounds
            axes.set_position((x, 0.4, width, height))
    elif kind == "overlap_columns":
        for axes in figure.axes[1::2]:
            _, y, width, height = axes.get_position().bounds
            axes.set_position((0.3, y, width, height))
    before = positions(figure)
    canvas.set_analysis_layout(prepared)
    assert figure.get_layout_engine() is None
    canvas.draw()
    np.testing.assert_allclose(positions(figure), before)


@pytest.mark.parametrize(
    "format,dpi", [("png", 100), ("png", 200), ("svg", 100), ("pdf", 100)]
)
def test_exports_reflow_at_export_dimensions_and_restore_display(
    canvases, optical_plots, qapp, monkeypatch, format, dpi
):
    canvas = canvases(optical_plots["Ray Fan"], height=250)
    canvas.resize(921, 250)
    canvas.show()
    qapp.processEvents()
    canvas.draw()
    display_dpi = canvas.figure.dpi
    display_size = canvas.figure.get_size_inches() * display_dpi
    display_pixels = tuple(int(value) for value in display_size)
    engine = canvas.figure.get_layout_engine()
    minimum = (engine.minimum_width_inches, engine.minimum_height_inches)
    assert any(value > 0 for value in minimum)
    fingerprint = engine._minimum_style_key
    style_calls = []
    original = engine._height_style_key

    def measured():
        style_calls.append(canvas.figure.canvas.is_saving())
        return original()

    monkeypatch.setattr(engine, "_height_style_key", measured)
    before = scientific_state(canvas.figure)
    original_positions = positions(canvas.figure)
    draw_sizes = []

    def record_draw(event):
        draw_sizes.append(
            (tuple(event.canvas.figure.bbox.bounds), event.canvas.figure.dpi)
        )
        assert_figure_readable(event.canvas.figure, event.renderer)

    canvas.mpl_connect("draw_event", record_draw)
    output = io.BytesIO()
    canvas.figure.savefig(output, format=format, dpi=dpi)
    assert style_calls == []
    assert engine._minimum_style_key is fingerprint
    assert (engine.minimum_width_inches, engine.minimum_height_inches) == minimum
    assert len(output.getvalue()) > 1000
    assert draw_sizes
    if format == "png":
        assert struct.unpack(">II", output.getvalue()[16:24]) == (
            round(921 * dpi / 100),
            round(250 * dpi / 100),
        )
        assert any(size[1] == dpi for size in draw_sizes)
    else:
        assert any(size[1] == 72 for size in draw_sizes)
    wait_for(
        qapp,
        lambda: (int(canvas.renderer.width), int(canvas.renderer.height))
        == display_pixels,
    )
    qapp.processEvents()
    assert not any(style_calls)
    assert canvas.figure.canvas is canvas
    assert canvas.figure.dpi == display_dpi
    assert canvas.buffer_rgba().shape[:2] == display_pixels[::-1]
    np.testing.assert_allclose(
        canvas.figure.get_size_inches() * canvas.figure.dpi,
        display_size,
        rtol=0,
        atol=1e-8,
    )
    assert_readable(canvas)
    np.testing.assert_allclose(
        positions(canvas.figure), original_positions, atol=1e-8, rtol=0
    )
    assert scientific_state(canvas.figure) == before
    image = hashlib.sha256(canvas.buffer_rgba()).digest()
    canvas.draw()
    assert hashlib.sha256(canvas.buffer_rgba()).digest() == image


@pytest.mark.parametrize("ratio", [1.5, 2])
@pytest.mark.parametrize("name", ["Ray Fan", "Spot Diagram"])
def test_layout_uses_physical_point_metrics_at_scaled_display_dpi(
    canvases, optical_plots, ratio, name
):
    canvas = canvases(
        optical_plots[name], width=921 * ratio, height=279 * ratio, dpi=100 * ratio
    )
    before = scientific_state(canvas.figure)
    canvas.draw()
    assert_readable(canvas)
    assert scientific_state(canvas.figure) == before


def test_roomy_uncrowded_grid_keeps_scientific_rectangles(canvases):
    canvas = canvases(width=1200, height=1000)
    install_grid(canvas)
    before = positions(canvas.figure)
    canvas.draw()
    assert positions(canvas.figure) == before
    assert_readable(canvas)


def test_empty_canvas_uses_normal_matplotlib_layout(canvases):
    canvas = canvases()
    canvas.set_analysis_layout({"axes": []})
    canvas.draw()
    assert canvas.figure.get_layout_engine() is None


def test_policy_has_no_runtime_configuration(canvases):
    canvas = canvases()
    install_grid(canvas)
    engine = canvas.figure.get_layout_engine()
    assert isinstance(engine, AnalysisFigureLayout)
    engine.set()
    with pytest.raises(TypeError, match="no configurable parameters"):
        engine.set(fontsize=6)


def test_analysis_panel_installs_layout_without_rerunning_optics(
    qapp, optical_plots, plot_panel_font
):
    connector = connector_for(CookeTriplet())
    panel = AnalysisPanel(connector)
    try:
        page = panel._execute_analysis(
            panel._analysis_class_map["Ray Fan"], "Ray Fan", {"num_points": 128}, {}
        )
        panel.switch_plot_page(0)
        connector.calculation_jobs.complete(
            page["page_id"],
            {
                "plot": optical_plots["Ray Fan"],
                "summary": "Completed",
                "size_bytes": 100,
            },
        )
        retained = panel.active_mpl_canvas_widget
        assert isinstance(retained.figure.get_layout_engine(), AnalysisFigureLayout)
        prepared = page["prepared"]
        payload = pickle.dumps(prepared, protocol=5)
        jobs = len(connector.calculation_jobs.submissions)
        settings = pickle.dumps(page["constructor_args_used"], protocol=5)
        for height in (279, 620, 279):
            retained.figure.set_size_inches(9.21, height / 100, forward=False)
            retained.draw()
            assert_readable(retained)
            assert panel.active_mpl_canvas_widget is retained
            assert page["prepared"] is prepared
        for theme in ("light", "dark"):
            panel.update_theme(theme)
            canvas = panel.active_mpl_canvas_widget
            assert isinstance(canvas.figure.get_layout_engine(), AnalysisFigureLayout)
            canvas.figure.set_size_inches(9.21, 2.79, forward=False)
            canvas.draw()
            assert_readable(canvas)
        assert len(connector.calculation_jobs.submissions) == jobs
        assert pickle.dumps(page["prepared"], protocol=5) == payload
        assert pickle.dumps(page["constructor_args_used"], protocol=5) == settings
    finally:
        panel.close()
        panel.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_preexisting_manual_tick_visibility_survives_shared_coordinate_layout(
    canvases, optical_plots
):
    canvas = canvases(optical_plots["Ray Fan"])
    axes = canvas.figure.axes[0]
    ticks = axes.xaxis.get_major_ticks()
    ticks[0].label1.set_visible(False)
    canvas.draw()
    assert not axes.xaxis.label.get_visible()
    canvas.figure.set_size_inches(9.21, 6.2, forward=False)
    canvas.draw()
    assert not ticks[0].label1.get_visible()
    assert all(tick.label1.get_visible() for tick in ticks[1:])


def test_intentionally_internal_legend_keeps_its_placement(canvases):
    canvas = canvases()
    prepared = install_grid(canvas)
    handles = [canvas.figure.axes[0].lines[0]]
    legend = canvas.figure.legend(
        handles, ["Scientific series"], loc="center", bbox_to_anchor=(0.5, 0.5)
    )
    canvas.set_analysis_layout(prepared)
    original = positions(canvas.figure)
    anchor = (
        legend.get_bbox_to_anchor()
        .transformed(canvas.figure.transFigure.inverted())
        .bounds
    )
    canvas.draw()
    assert positions(canvas.figure) == original
    np.testing.assert_allclose(
        legend.get_bbox_to_anchor()
        .transformed(canvas.figure.transFigure.inverted())
        .bounds,
        anchor,
    )


def test_impossibly_small_grid_retains_all_scientific_decorations(canvases):
    canvas = canvases(width=60, height=60)
    install_grid(canvas)
    before = scientific_state(canvas.figure)
    original = positions(canvas.figure)
    canvas.draw()
    assert scientific_state(canvas.figure) == before
    assert positions(canvas.figure) == original
    assert all(ax.xaxis.label.get_visible() for ax in canvas.figure.axes)
    assert all(
        ax.title.get_visible() and ax.yaxis.label.get_visible()
        for ax in canvas.figure.axes
    )


def test_real_toolbar_history_reflows_restored_compact_view(canvases, optical_plots):
    canvas = canvases(optical_plots["Ray Fan"])
    toolbar = CustomMatplotlibToolbar(canvas, canvas)
    canvas.draw()
    initial_limits = [(ax.get_xlim(), ax.get_ylim()) for ax in canvas.figure.axes]
    toolbar.push_current()
    canvas.figure.axes[0].set_xlim(-0.2, 0.6)
    canvas.draw()
    toolbar.push_current()
    toolbar.back()
    canvas.draw()
    assert [
        (ax.get_xlim(), ax.get_ylim()) for ax in canvas.figure.axes
    ] == initial_limits
    assert_readable(canvas)
    assert not canvas.figure.axes[0].xaxis.label.get_visible()
    toolbar.forward()
    canvas.draw()
    assert canvas.figure.axes[0].get_xlim() == (-0.2, 0.6)
    assert canvas.figure.axes[0].xaxis.label.get_visible()
    toolbar.home()
    canvas.draw()
    assert_readable(canvas)
    canvas.figure.set_size_inches(9.21, 6.2, forward=False)
    canvas.draw()
    assert all(ax.xaxis.label.get_visible() for ax in canvas.figure.axes)
    canvas.figure.set_size_inches(9.21, 2.79, forward=False)
    canvas.draw()
    assert_readable(canvas)


def test_manual_legend_anchor_is_preserved_when_an_axis_is_hidden(
    canvases, optical_plots
):
    canvas = canvases(optical_plots["Ray Fan"])
    canvas.draw()
    legend = canvas.figure.legends[0]
    legend.set_bbox_to_anchor((0.5, 0.5), canvas.figure.transFigure)
    anchor = (
        legend.get_bbox_to_anchor()
        .transformed(canvas.figure.transFigure.inverted())
        .bounds
    )
    canvas.figure.axes[0].set_visible(False)
    canvas.draw()
    np.testing.assert_allclose(
        legend.get_bbox_to_anchor()
        .transformed(canvas.figure.transFigure.inverted())
        .bounds,
        anchor,
    )
