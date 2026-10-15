"""Measured overflow keeps compact dock plots inspectable at normal font sizes."""

from __future__ import annotations

import io
import struct
from pathlib import Path

import pytest
from matplotlib.backend_bases import MouseEvent
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtTest import QTest

from optiland_gui.widgets.analysis_canvas_viewport import AnalysisCanvasViewport
from tests.gui import test_analysis_figure_layout as layout_tests
from tests.gui.test_analysis_figure_layout import (
    assert_readable,
    install_grid,
    scientific_state,
)
from tests.gui.test_calculation_jobs import wait_for

canvases = layout_tests.canvases
optical_plots = layout_tests.optical_plots


@pytest.fixture
def viewports(canvases, qapp):
    result = []

    def make(prepared=None, theme="dark", width=573, height=164):
        canvas = canvases(prepared, theme, width=width, height=height)
        if prepared is None:
            install_grid(canvas)
        viewport = AnalysisCanvasViewport(canvas)
        styles = Path(__file__).parents[2] / "optiland_gui" / "resources" / "styles"
        viewport.setStyleSheet(
            (styles / f"{theme}_theme.qss").read_text(encoding="utf-8")
            + (styles / "sidebar.qss").read_text(encoding="utf-8")
        )
        result.append(viewport)
        viewport.resize(width, height)
        viewport.show()
        settle(qapp, canvas)
        return viewport, canvas

    yield make
    for viewport in result:
        viewport.close()
        viewport.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def settle(qapp, canvas):
    wait_for(qapp, lambda: hasattr(canvas, "renderer"))
    # Child-size and scrollbar changes can queue another public draw_idle.
    # Wait for the actual renderer and geometry to stabilize together.
    QTest.qWait(500)
    wait_for(
        qapp,
        lambda: (
            not canvas._draw_pending
            and not canvas._resize_followup.isActive()
            and not canvas._resize_quiet.isActive()
            and not canvas._resize_progress.isActive()
            and abs(canvas.renderer.width - canvas.width() * canvas.device_pixel_ratio)
            < 1e-8
            and abs(
                canvas.renderer.height - canvas.height() * canvas.device_pixel_ratio
            )
            < 1e-8
        ),
    )
    QTest.qWait(100)


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize(
    "width,height", [(573, 164), (573, 240), (573, 260), (430, 164), (220, 164)]
)
def test_actual_dpr_viewport_keeps_all_fields_readable_and_settles(
    viewports, optical_plots, qapp, monkeypatch, theme, width, height
):
    viewport, canvas = viewports(optical_plots["Ray Fan"], theme, width, height)
    before = scientific_state(canvas.figure)
    assert_readable(canvas)
    assert viewport.width() == width and viewport.height() == height
    assert all(
        ax.title.get_visible() and ax.yaxis.label.get_visible()
        for ax in canvas.figure.axes
    )
    assert viewport.verticalScrollBar().maximum() > 0
    if width == 220:
        assert viewport.horizontalScrollBar().maximum() > 0
    calls = []
    draw_cid = canvas.mpl_connect("draw_event", lambda _event: calls.append(True))
    QTest.qWait(350)
    assert calls == []
    bbox_calls = []
    for axes in canvas.figure.axes:
        original = axes.get_tightbbox

        def measured(*args, original=original, **kwargs):
            bbox_calls.append(True)
            return original(*args, **kwargs)

        monkeypatch.setattr(axes, "get_tightbbox", measured)
    canvas.draw()
    assert bbox_calls == []
    assert scientific_state(canvas.figure) == before
    canvas.mpl_disconnect(draw_cid)
    viewport.verticalScrollBar().setValue(viewport.verticalScrollBar().maximum())
    assert viewport.verticalScrollBar().value() > 0


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("width", [573, 430, 220])
def test_native_sized_spot_viewport_keeps_square_axes_and_tick_labels_readable(
    viewports, optical_plots, qapp, monkeypatch, theme, width
):
    viewport, canvas = viewports(optical_plots["Spot Diagram"], theme, width)
    before = scientific_state(canvas.figure)
    locators = [
        (ax.xaxis.get_major_locator(), ax.yaxis.get_major_locator())
        for ax in canvas.figure.axes
    ]
    assert_readable(canvas)
    assert viewport.verticalScrollBar().maximum() > 0
    for axes in canvas.figure.axes:
        assert axes.get_aspect() == 1
        assert abs(axes.bbox.width - axes.bbox.height) < 1e-8
    draws = []
    canvas.mpl_connect("draw_event", lambda _event: draws.append(True))
    QTest.qWait(350)
    assert draws == []
    bbox_calls = []
    for axes in canvas.figure.axes:
        original = axes.get_tightbbox

        def measured(*args, original=original, **kwargs):
            bbox_calls.append(True)
            return original(*args, **kwargs)

        monkeypatch.setattr(axes, "get_tightbbox", measured)
    canvas.draw()
    assert bbox_calls == []
    assert_readable(canvas)
    assert scientific_state(canvas.figure) == before
    assert [
        (ax.xaxis.get_major_locator(), ax.yaxis.get_major_locator())
        for ax in canvas.figure.axes
    ] == locators


def test_spot_zoom_and_public_aspect_changes_remeasure_overflow(
    viewports, optical_plots, qapp
):
    viewport, canvas = viewports(optical_plots["Spot Diagram"])
    axes = canvas.figure.axes[0]
    limits = axes.get_xlim()
    axes.set_xlim(limits[0] / 2, limits[1] / 2)
    canvas.draw_idle()
    settle(qapp, canvas)
    assert axes.get_xlim() == (limits[0] / 2, limits[1] / 2)
    assert axes.get_aspect() == 1
    assert_readable(canvas)
    axes.set_aspect(0.5)
    canvas.draw_idle()
    settle(qapp, canvas)
    assert axes.get_aspect() == 0.5
    assert abs(axes.bbox.height / axes.bbox.width - 0.5 * axes.get_data_ratio()) < 1e-8
    assert_readable(canvas)
    for ax in canvas.figure.axes:
        ax.set_aspect("auto")
    canvas.draw_idle()
    settle(qapp, canvas)
    assert all(ax.get_aspect() == "auto" for ax in canvas.figure.axes)
    assert_readable(canvas)
    viewport.resize(1200, 620)
    settle(qapp, canvas)
    assert canvas.minimumSize().height() == 0
    assert canvas.minimumSize().width() == 0
    assert_readable(canvas)


def test_overflow_keyboard_access_does_not_steal_plot_wheel_zoom(
    viewports, optical_plots, qapp
):
    from optiland_gui.gui_plot_utils import handle_matplotlib_scroll_zoom

    viewport, canvas = viewports(optical_plots["Ray Fan"])
    assert viewport.accessibleName() and viewport.accessibleDescription()
    viewport.setFocus()
    QTest.keyClick(viewport, Qt.Key.Key_PageDown)
    assert viewport.verticalScrollBar().value() > 0
    viewport.verticalScrollBar().setValue(0)
    QTest.keyClick(viewport, Qt.Key.Key_Down)
    assert viewport.verticalScrollBar().value() > 0
    scroll_before = viewport.verticalScrollBar().value()
    axes = canvas.figure.axes[0]
    limits = axes.get_xlim()
    canvas.mpl_connect("scroll_event", handle_matplotlib_scroll_zoom)
    x, y = axes.transData.transform((0, 0))
    event = MouseEvent("scroll_event", canvas, x, y, button="up", step=1)
    canvas.callbacks.process("scroll_event", event)
    qapp.processEvents()
    assert axes.get_xlim() != limits
    assert viewport.verticalScrollBar().value() == scroll_before


@pytest.mark.parametrize(
    "format,dpi",
    [("png", "figure"), ("png", 200), ("svg", "figure"), ("pdf", "figure")],
)
def test_export_covers_full_scrolled_figure_and_restores_viewport(
    viewports, optical_plots, qapp, format, dpi
):
    viewport, canvas = viewports(optical_plots["Ray Fan"], width=220)
    before = scientific_state(canvas.figure)
    size = (canvas.width(), canvas.height())
    minima = (canvas.minimumWidth(), canvas.minimumHeight())
    viewport.verticalScrollBar().setValue(20)
    output = io.BytesIO()
    canvas.figure.savefig(output, format=format, dpi=dpi)
    settle(qapp, canvas)
    assert len(output.getvalue()) > 1000
    if format == "png":
        # Matplotlib's "figure" export DPI is the original point scale,
        # independent of a display canvas's device-pixel-ratio adjustment.
        export_dpi = 100 if dpi == "figure" else dpi
        assert struct.unpack(">II", output.getvalue()[16:24]) == (
            int(canvas.figure.get_figwidth() * export_dpi),
            int(canvas.figure.get_figheight() * export_dpi),
        )
    assert (canvas.width(), canvas.height()) == size
    assert (canvas.minimumWidth(), canvas.minimumHeight()) == minima
    assert viewport.verticalScrollBar().value() == 20
    assert scientific_state(canvas.figure) == before
    assert_readable(canvas)


def test_overflow_resets_for_room_style_and_manual_geometry(
    viewports, optical_plots, qapp
):
    viewport, canvas = viewports(optical_plots["Ray Fan"])
    viewport.resize(921, 620)
    settle(qapp, canvas)
    assert viewport.verticalScrollBar().maximum() == 0
    assert viewport.horizontalScrollBar().maximum() == 0
    assert canvas.minimumHeight() == 0
    viewport.resize(573, 200)
    settle(qapp, canvas)
    assert_readable(canvas)
    assert canvas.minimumHeight() > 0
    for axes in canvas.figure.axes:
        axes.title.set_fontsize(4)
        axes.xaxis.label.set_fontsize(4)
        axes.yaxis.label.set_fontsize(4)
        axes.tick_params(labelsize=4)
    canvas.draw_idle()
    settle(qapp, canvas)
    assert canvas.minimumHeight() == 0
    assert viewport.verticalScrollBar().maximum() == 0
    # A public manual rectangle edit owns layout and releases any imposed size.
    viewport.resize(220, 90)
    settle(qapp, canvas)
    canvas.figure.axes[0].set_position((0.2, 0.7, 0.3, 0.2))
    canvas.draw_idle()
    settle(qapp, canvas)
    assert canvas.minimumHeight() == 0 and canvas.minimumWidth() == 0


def test_initial_long_bottom_legend_reserves_horizontal_overflow(viewports, qapp):
    viewport, canvas = viewports(width=220)
    # Exercise ordinary cheap resize handling regardless of shared-host load.
    # Keep resize deferral disabled even if optical tests or coverage are busy.
    canvas._EXPENSIVE_DRAW_MS = float("inf")
    # Reinstall using an initial scientific legend rather than a later manual edit.
    legend = canvas.figure.legend(
        canvas.figure.axes[0].lines,
        ["Full scientific series description with wavelength and field details"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
    )
    canvas.set_analysis_layout({"axes": [{"visible": True, "colorbar": False}] * 4})
    canvas.draw_idle()
    settle(qapp, canvas)
    assert viewport.horizontalScrollBar().maximum() > 0
    box = legend.get_window_extent(canvas.get_renderer())
    assert box.x0 >= 0 and box.x1 <= canvas.figure.bbox.width
    assert (
        abs(canvas.figure.bbox.width - canvas.width() * canvas.device_pixel_ratio)
        < 1e-8
    )
    assert (
        abs(canvas.figure.bbox.height - canvas.height() * canvas.device_pixel_ratio)
        < 1e-8
    )
    draws = []
    canvas.mpl_connect("draw_event", lambda _event: draws.append(True))
    QTest.qWait(350)
    assert draws == []


@pytest.mark.parametrize("replacement", [None, "tight"])
def test_public_layout_replacement_releases_viewport_minimum(
    viewports, optical_plots, qapp, replacement
):
    viewport, canvas = viewports(optical_plots["Ray Fan"], width=220)
    assert canvas.minimumHeight() > 0 and canvas.minimumWidth() > 0
    canvas.figure.set_layout_engine(replacement)
    canvas.draw_idle()
    settle(qapp, canvas)
    assert canvas.minimumSize().width() == 0
    assert canvas.minimumSize().height() == 0
    assert viewport.horizontalScrollBar().maximum() == 0
    assert viewport.verticalScrollBar().maximum() == 0


def test_closing_viewport_closes_child_canvas_and_stops_resize_draws(
    viewports, optical_plots, qapp
):
    viewport, canvas = viewports(optical_plots["Ray Fan"])
    draws = []
    canvas.mpl_connect("draw_event", lambda _event: draws.append(True))
    canvas.resize(600, 300)
    viewport.close()
    QTest.qWait(350)
    assert not canvas.isVisible()
    assert draws == []


def test_queued_minimum_does_not_resize_hidden_plot(viewports, qapp):
    viewport, canvas = viewports()
    before = canvas.minimumSize()
    engine = canvas.figure.get_layout_engine()
    engine.minimum_height_inches += 2
    viewport._minimum_changed.emit()
    viewport.hide()
    qapp.processEvents()
    assert canvas.minimumSize() == before
    viewport.show()
    settle(qapp, canvas)
    assert_readable(canvas)


def test_plain_canvas_viewport_resize_leaves_layout_owner_unchanged(canvases, qapp):
    canvas = canvases()
    canvas.figure.subplots().plot([0, 1], [0, 1])
    viewport = AnalysisCanvasViewport(canvas)
    viewport.resize(400, 300)
    viewport.show()
    settle(qapp, canvas)
    viewport.resize(450, 350)
    settle(qapp, canvas)
    assert canvas.figure.get_layout_engine() is None
    assert canvas.minimumWidth() == canvas.minimumHeight() == 0
    viewport.close()
    viewport.deleteLater()
