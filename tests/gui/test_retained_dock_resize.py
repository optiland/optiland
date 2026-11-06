"""Retained plots coalesce costly dock resizes without delaying plot actions."""

from __future__ import annotations

import hashlib
import io
import pickle
import struct
import threading
import time

import numpy as np
import pytest
from matplotlib.artist import Artist
from matplotlib.figure import Figure
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QPointF, Qt, QTimer
from PySide6.QtGui import QFont, QFontDatabase, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDockWidget, QMainWindow, QWidget
from shiboken6 import isValid

from optiland.samples.objectives import CookeTriplet, DoubleGauss
from optiland_gui.analysis_panel import AnalysisPanel
from optiland_gui.services.analysis_worker import prepare_analysis
from optiland_gui.services.job_records import OpticSnapshot
from optiland_gui.widgets.retained_analysis_canvas import RetainedAnalysisCanvas
from tests.gui.analysis_job_fakes import connector_for
from tests.gui.test_calculation_jobs import wait_for


class CostlyArtist(Artist):
    """Deterministically exercise the costly-render policy on any test CPU."""

    def draw(self, renderer):
        time.sleep(0.06)


def final_pixels_match(canvas):
    width, height = canvas.get_width_height(physical=True)
    return (
        (int(canvas.renderer.width), int(canvas.renderer.height)) == (width, height)
        and canvas.buffer_rgba().shape[:2] == (height, width)
        and np.allclose(
            canvas.figure.get_size_inches() * canvas.figure.dpi,
            np.array([canvas.width(), canvas.height()]) * canvas.device_pixel_ratio,
            rtol=0,
            atol=1e-8,
        )
    )


@pytest.fixture
def costly_canvas(qapp):
    figure = Figure(figsize=(6, 4), dpi=100)
    axes = figure.subplots()
    axes.plot([0, 1], [0, 1])
    figure.add_artist(CostlyArtist())
    canvas = RetainedAnalysisCanvas(figure)
    canvas.resize(600, 400)
    canvas.show()
    QTest.qWait(180)
    yield canvas
    if isValid(canvas):
        canvas.close()
        canvas.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.fixture
def populated_dock_tree(qapp):
    if not QFontDatabase.families():
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
        qapp.setFont(QFont("Segoe UI", 9))
    window = QMainWindow()
    window.setDockNestingEnabled(True)
    center = QWidget()
    center.setFixedWidth(0)
    window.setCentralWidget(center)
    retained, docks = [], []
    for optic_type, name, constructor_args in (
        (CookeTriplet, "Ray Fan", {"num_points": 128}),
        (DoubleGauss, "Spot Diagram", {"num_rings": 6}),
    ):
        optic = optic_type()
        connector = connector_for(optic)
        prepared = prepare_analysis(
            OpticSnapshot.capture(optic),
            {
                "name": name,
                "constructor_args": constructor_args,
                "view_args": {},
                "theme": "dark",
            },
            lambda *_args, **_kwargs: None,
            threading.Event(),
        )
        panel = AnalysisPanel(connector)
        page = panel._execute_analysis(
            panel._analysis_class_map[name], name, constructor_args, {}
        )
        panel.switch_plot_page(0)
        connector.calculation_jobs.complete(page["page_id"], prepared)
        # Force the costly branch independent of rendering speed on CI.
        panel.active_mpl_canvas_widget.figure.add_artist(CostlyArtist())
        dock = QDockWidget(name, window)
        dock.setWidget(panel)
        window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        retained.append((panel, connector, page))
        docks.append(dock)
    window.splitDockWidget(docks[0], docks[1], Qt.Orientation.Horizontal)
    window.resize(1500, 900)
    window.show()
    QTest.qWait(350)
    yield window, docks, retained
    window.close()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_populated_splitter_burst_preserves_zoom_data_and_final_image(
    qapp, populated_dock_tree
):
    window, docks, retained = populated_dock_tree
    canvases, before, draws = [], [], [[], []]
    for index, (panel, connector, page) in enumerate(retained):
        canvas = panel.active_mpl_canvas_widget
        canvases.append(canvas)
        for axes in canvas.figure.axes:
            left, right = axes.get_xlim()
            axes.set_xlim(left + (right - left) * 0.1, right - (right - left) * 0.1)
        canvas.draw()
        before.append(
            (
                canvas.figure,
                page["prepared"],
                pickle.dumps(page["prepared"], protocol=5),
                len(connector.calculation_jobs.submissions),
                [(axis.get_xlim(), axis.get_ylim()) for axis in canvas.figure.axes],
            )
        )
        canvas.mpl_connect(
            "draw_event", lambda _event, index=index: draws[index].append(True)
        )
    requests = 0
    driver = QTimer(window)
    driver.setInterval(16)

    def request_resize():
        nonlocal requests
        window.resizeDocks(
            docks, [750 + requests * 8, 750 - requests * 8], Qt.Orientation.Horizontal
        )
        requests += 1
        if requests == 24:
            driver.stop()

    driver.timeout.connect(request_resize)
    driver.start()
    wait_for(qapp, lambda: requests == 24)
    wait_for(qapp, lambda: all(final_pixels_match(canvas) for canvas in canvases))
    QTest.qWait(180)
    # The old immediate-per-resize canvas produces 23 renders per panel.
    assert all(0 < len(events) < 12 for events in draws)
    for canvas, (panel, connector, page), state in zip(
        canvases, retained, before, strict=True
    ):
        figure, payload, data, jobs, limits = state
        assert panel.active_mpl_canvas_widget is canvas
        assert canvas.figure is figure
        assert page["prepared"] is payload
        assert pickle.dumps(page["prepared"], protocol=5) == data
        assert len(connector.calculation_jobs.submissions) == jobs
        assert [(axis.get_xlim(), axis.get_ylim()) for axis in figure.axes] == limits
        assert final_pixels_match(canvas)
        np.testing.assert_allclose(
            figure.get_size_inches() * figure.dpi,
            np.array([canvas.width(), canvas.height()]) * canvas.device_pixel_ratio,
            rtol=0,
            atol=1e-8,
        )
        final_image = hashlib.sha256(canvas.buffer_rgba()).digest()
        canvas.draw()
        assert hashlib.sha256(canvas.buffer_rgba()).digest() == final_image


@pytest.mark.parametrize("draw_method", ["draw", "draw_idle"])
def test_explicit_draw_cancels_resize_deferral(qapp, costly_canvas, draw_method):
    canvas = costly_canvas
    draws = []
    canvas.mpl_connect("draw_event", lambda _event: draws.append(True))
    canvas.resize(800, 530)
    qapp.processEvents()
    assert not draws
    getattr(canvas, draw_method)()
    wait_for(qapp, lambda: bool(draws))
    assert final_pixels_match(canvas)
    completed = len(draws)
    QTest.qWait(320)
    assert len(draws) == completed


def test_public_resize_callback_can_request_immediate_draw(qapp, costly_canvas):
    canvas = costly_canvas
    callbacks, draws = [], []
    canvas.mpl_connect("draw_event", lambda _event: draws.append(True))

    def on_resize(event):
        callbacks.append((event.width, event.height))
        canvas.draw()

    canvas.mpl_connect("resize_event", on_resize)
    canvas.resize(830, 550)
    qapp.processEvents()
    assert callbacks == [(830, 550)]
    assert len(draws) == 1
    assert final_pixels_match(canvas)
    QTest.qWait(320)
    assert len(draws) == 1


def test_resize_callback_changing_canvas_size_finishes_at_latest_geometry(
    qapp, costly_canvas
):
    canvas = costly_canvas
    callbacks = []

    def on_resize(event):
        callbacks.append((event.width, event.height))
        if len(callbacks) == 1:
            canvas.resize(910, 610)

    canvas.mpl_connect("resize_event", on_resize)
    canvas.resize(830, 550)
    qapp.processEvents()
    wait_for(qapp, lambda: final_pixels_match(canvas))
    assert canvas.width() == 910
    assert canvas.height() == 610
    assert callbacks[-1] == (910, 610)
    assert not canvas._handling_resize


def test_resize_without_figure_leaves_renderer_and_timers_untouched(
    qapp, costly_canvas
):
    canvas = costly_canvas
    figure = canvas.figure
    before = canvas.buffer_rgba().tobytes()
    canvas.figure = None
    try:
        canvas.resize(830, 550)
        assert not canvas._handling_resize
        assert not canvas._resize_followup.isActive()
        assert canvas.buffer_rgba().tobytes() == before
    finally:
        canvas.figure = figure
        canvas.resize(840, 560)
    wait_for(qapp, lambda: final_pixels_match(canvas))


def test_sustained_resize_requests_periodic_progress(qapp, costly_canvas):
    canvas = costly_canvas
    draws, requests = [], []
    canvas.mpl_connect("draw_event", lambda _event: draws.append(len(requests)))
    driver = QTimer(canvas)
    driver.setInterval(16)

    def resize():
        requests.append(True)
        canvas.resize(600 + len(requests) * 3, 400 + len(requests))
        if len(requests) == 35:
            driver.stop()

    driver.timeout.connect(resize)
    driver.start()
    wait_for(qapp, lambda: len(requests) == 35)
    wait_for(qapp, lambda: final_pixels_match(canvas))
    assert draws
    assert any(count < 35 for count in draws)
    assert len(draws) < len(requests) // 2


@pytest.mark.parametrize("input_kind", ["click", "wheel", "move"])
def test_plot_coordinates_are_current_before_pending_resize_input(
    qapp, costly_canvas, input_kind
):
    canvas = costly_canvas
    draws, positions = [], []
    canvas.mpl_connect("draw_event", lambda _event: draws.append(True))
    event_name = {
        "click": "button_press_event",
        "wheel": "scroll_event",
        "move": "motion_notify_event",
    }[input_kind]
    canvas.mpl_connect(
        event_name,
        lambda event: positions.append((event.xdata, event.ydata, len(draws))),
    )
    canvas.resize(1000, 650)
    qapp.processEvents()
    assert not draws
    axes = canvas.figure.axes[0]
    limits = axes.get_xlim(), axes.get_ylim()
    # Use final normalized axes geometry to avoid relying on stale renderer data.
    position = axes.get_position()
    ratio = canvas.device_pixel_ratio
    width, height = canvas.get_width_height(physical=True)
    px = width * (position.x0 + position.width * 0.8)
    py = height * (position.y0 + position.height * 0.6)
    point = QPoint(round(px / ratio), round((height - py) / ratio))
    if input_kind == "click":
        QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=point)
    elif input_kind == "move":
        QTest.mouseMove(canvas, point)
    else:
        event = QWheelEvent(
            QPointF(point),
            QPointF(canvas.mapToGlobal(point)),
            QPoint(),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        QApplication.sendEvent(canvas, event)
    assert positions
    x, y, rendered_before_input = positions[-1]
    expected_x = limits[0][0] + (limits[0][1] - limits[0][0]) * 0.8
    expected_y = limits[1][0] + (limits[1][1] - limits[1][0]) * 0.6
    assert x == pytest.approx(expected_x, abs=0.003)
    assert y == pytest.approx(expected_y, abs=0.003)
    assert rendered_before_input == 1
    assert final_pixels_match(canvas)


def test_scaled_preview_owns_rgba_memory_and_fills_current_rect(qapp, costly_canvas):
    canvas = costly_canvas
    background = canvas.grab().toImage().pixelColor(5, 5).name()
    rendered_before = canvas.buffer_rgba().tobytes()
    canvas.resize(1000, 650)
    qapp.processEvents()
    image = canvas.grab().toImage()
    assert image.deviceIndependentSize().toSize() == canvas.size()
    # The complete retained figure, including its right/bottom background,
    # fills the new rectangle rather than leaving clipped old renderer storage.
    assert image.pixelColor(image.width() - 3, image.height() - 3).name() == background
    assert canvas.buffer_rgba().tobytes() == rendered_before
    # Mutating the old renderer memory cannot mutate the owned preview copy.
    rgba = canvas.buffer_rgba()
    np.asarray(rgba)[:] = [255, 0, 255, 255]
    assert canvas.grab().toImage().pixelColor(5, 5).name() == background
    wait_for(qapp, lambda: final_pixels_match(canvas))
    assert canvas.grab().toImage().pixelColor(5, 5).name() == background


def test_pending_resize_uses_physical_pixels_with_dpr(qapp, monkeypatch):
    # A deterministic DPR seam check, not a physical mixed-monitor simulation.
    monkeypatch.setattr(
        RetainedAnalysisCanvas, "device_pixel_ratio", property(lambda _canvas: 2.0)
    )
    figure = Figure(figsize=(6, 4), dpi=200)
    figure.subplots().plot([0, 1], [0, 1])
    figure.add_artist(CostlyArtist())
    canvas = RetainedAnalysisCanvas(figure)
    canvas.resize(600, 400)
    canvas.show()
    QTest.qWait(200)
    canvas.resize(800, 530)
    qapp.processEvents()
    assert canvas.get_width_height(physical=True) == (1600, 1060)
    wait_for(qapp, lambda: final_pixels_match(canvas))
    np.testing.assert_allclose(
        canvas.figure.get_size_inches() * canvas.figure.dpi, [1600, 1060]
    )
    canvas.close()
    canvas.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_export_during_pending_resize_uses_complete_current_figure(qapp, costly_canvas):
    canvas = costly_canvas
    canvas.resize(810, 540)
    qapp.processEvents()
    exported = io.BytesIO()
    canvas.print_figure(exported, format="png", dpi="figure", bbox_inches=None)
    assert struct.unpack(">II", exported.getvalue()[16:24]) == (810, 540)
    wait_for(qapp, lambda: final_pixels_match(canvas))
    assert canvas.figure.axes[0].get_xlim() == pytest.approx((-0.05, 1.05))


@pytest.mark.parametrize(
    "export_format, export_dpi", [("svg", 100), ("pdf", 100), ("png", 250)]
)
def test_export_restores_current_onscreen_renderer_after_export_canvas(
    qapp, costly_canvas, export_format, export_dpi
):
    canvas = costly_canvas
    canvas.resize(810, 540)
    qapp.processEvents()
    # Consume the preview paint before exporting: no native file dialog or
    # unrelated expose event should be required to finish the on-screen plot.
    canvas.grab()
    exported = io.BytesIO()
    canvas.print_figure(exported, format=export_format, dpi=export_dpi)
    wait_for(qapp, lambda: final_pixels_match(canvas))
    ratio = canvas.device_pixel_ratio
    assert canvas.figure.dpi == 100 * ratio
    assert canvas.get_width_height(physical=True) == (
        round(810 * ratio), round(540 * ratio)
    )
    final_image = hashlib.sha256(canvas.buffer_rgba()).digest()
    canvas.draw()
    assert hashlib.sha256(canvas.buffer_rgba()).digest() == final_image


def test_failed_export_does_not_strand_pending_resize(qapp, costly_canvas):
    canvas = costly_canvas
    canvas.resize(810, 540)
    qapp.processEvents()
    with pytest.raises(ValueError, match="not supported"):
        canvas.print_figure(io.BytesIO(), format="unsupported")
    wait_for(qapp, lambda: final_pixels_match(canvas))


def test_explicit_blit_finishes_pending_resize_before_copy(qapp, costly_canvas):
    canvas = costly_canvas
    draws = []
    canvas.mpl_connect("draw_event", lambda _event: draws.append(True))
    canvas.resize(810, 540)
    qapp.processEvents()
    assert not draws
    canvas.blit()
    assert draws == [True]
    assert final_pixels_match(canvas)
    QTest.qWait(320)
    assert draws == [True]


def test_ordinary_blit_updates_artist_without_full_redraw(qapp, costly_canvas):
    canvas = costly_canvas
    axes = canvas.figure.axes[0]
    (marker,) = axes.plot([0.25], [0.7], "o", color="red", markersize=20, animated=True)
    canvas.draw()
    background = canvas.copy_from_bbox(axes.bbox)
    draws = []
    canvas.mpl_connect("draw_event", lambda _event: draws.append(True))

    def pixel_at(x, y):
        pixels = np.asarray(canvas.buffer_rgba())
        px, py = axes.transData.transform((x, y))
        return pixels[pixels.shape[0] - 1 - round(py), round(px), :3].copy()

    axes.draw_artist(marker)
    canvas.blit(axes.bbox)
    np.testing.assert_array_equal(pixel_at(0.25, 0.7), [255, 0, 0])
    canvas.restore_region(background)
    marker.set_data([0.75], [0.3])
    axes.draw_artist(marker)
    canvas.blit(axes.bbox)
    np.testing.assert_array_equal(pixel_at(0.75, 0.3), [255, 0, 0])
    assert not np.array_equal(pixel_at(0.25, 0.7), [255, 0, 0])
    QTest.qWait(320)
    assert not draws
    np.testing.assert_array_equal(pixel_at(0.75, 0.3), [255, 0, 0])


def test_empty_and_cheap_plot_keeps_native_idle_resize_path(qapp):
    figure = Figure(figsize=(6, 4), dpi=100)
    canvas = RetainedAnalysisCanvas(figure)
    canvas.show()
    QTest.qWait(50)
    draws = []
    canvas.mpl_connect("draw_event", lambda _event: draws.append(True))
    canvas.resize(810, 540)
    qapp.processEvents()
    assert draws
    assert final_pixels_match(canvas)
    completed = len(draws)
    QTest.qWait(320)
    assert len(draws) == completed
    canvas.close()
    canvas.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_hiding_pending_resize_then_reopening_renders_current_size(qapp, costly_canvas):
    canvas = costly_canvas
    draws = []
    canvas.mpl_connect("draw_event", lambda _event: draws.append(True))
    canvas.resize(800, 530)
    qapp.processEvents()
    assert not draws
    canvas.hide()
    QTest.qWait(320)
    assert not draws
    canvas.resize(900, 600)
    canvas.show()
    wait_for(qapp, lambda: final_pixels_match(canvas))
    assert draws


def test_deleting_pending_canvas_cancels_owned_resize_callbacks(qapp, costly_canvas):
    canvas = costly_canvas
    draws = []
    canvas.mpl_connect("draw_event", lambda _event: draws.append(True))
    canvas.resize(800, 530)
    qapp.processEvents()
    assert not draws
    canvas.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not isValid(canvas)
    QTest.qWait(320)
    assert not draws
