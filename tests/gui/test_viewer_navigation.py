"""A navigation gesture has one owner and preserves the view until zoom release."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.backend_bases import MouseButton, MouseEvent
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from optiland_gui.optiland_connector import OptilandConnector
from optiland_gui.viewer_panel import MatplotlibViewer
from tests.gui.test_calculation_jobs import wait_for


@pytest.fixture(scope="module")
def navigation_connector(qapp):
    connector = OptilandConnector()
    yield connector
    connector.calculation_jobs.shutdown()
    wait_for(qapp, lambda: connector.calculation_jobs._process is None)
    connector.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.fixture
def viewer(qapp, minimal_optic, navigation_connector):
    navigation_connector.load_optic_from_object(minimal_optic)
    widget = MatplotlibViewer(navigation_connector)
    widget.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    widget.resize(800, 600)
    widget.show()
    wait_for(qapp, lambda: widget.layout_job.data is not None)
    assert widget.layout_job.label.text() == "Layout is up to date."
    widget.ax.set_aspect("auto")
    widget.ax.set_xlim(0, 10)
    widget.ax.set_ylim(0, 10)
    widget.canvas.draw()
    yield widget
    widget.layout_job.cancel()
    widget.close()
    wait_for(qapp, lambda: not navigation_connector.calculation_jobs.running)
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def limits(viewer):
    return np.array([viewer.ax.get_xlim(), viewer.ax.get_ylim()])


def emit(viewer, name, pixel, button=MouseButton.LEFT):
    kwargs = {"buttons": {button}} if name == "motion_notify_event" else {}
    event = MouseEvent(name, viewer.canvas, *pixel, button=button, **kwargs)
    viewer.canvas.callbacks.process(name, event)
    return event


@pytest.mark.parametrize("start,end", [((3, 3), (7, 7)), ((7, 7), (3, 3))])
def test_rectangle_zoom_changes_limits_only_on_release(viewer, start, end):
    before = limits(viewer)
    start_pixel, end_pixel = viewer.ax.transData.transform([start, end])
    viewer.toolbar.zoom()
    emit(viewer, "motion_notify_event", start_pixel)
    cursor = viewer.canvas.cursor().shape()
    emit(viewer, "button_press_event", start_pixel)
    assert not viewer._is_panning
    emit(viewer, "motion_notify_event", end_pixel)
    np.testing.assert_array_equal(limits(viewer), before)
    emit(viewer, "button_release_event", end_pixel)
    assert not np.array_equal(limits(viewer), before)
    assert viewer.canvas.cursor().shape() == cursor
    after = limits(viewer)
    viewer.toolbar.back()
    np.testing.assert_allclose(limits(viewer), before)
    viewer.toolbar.forward()
    np.testing.assert_allclose(limits(viewer), after)


def test_toolbar_pan_does_not_also_start_custom_pan(viewer):
    viewer.toolbar.pan()
    start, end = viewer.ax.transData.transform([(3, 3), (4, 4)])
    before = limits(viewer)
    emit(viewer, "button_press_event", start)
    assert not viewer._is_panning
    emit(viewer, "motion_notify_event", end)
    emit(viewer, "button_release_event", end)
    assert not np.array_equal(limits(viewer), before)
    viewer.toolbar.back()
    np.testing.assert_allclose(limits(viewer), before)


def test_default_pan_has_history_and_stops_on_outside_release(viewer):
    before = limits(viewer)
    start, end = viewer.ax.transData.transform([(3, 3), (4, 4)])
    emit(viewer, "button_press_event", start)
    assert viewer._is_panning
    emit(viewer, "motion_notify_event", end)
    emit(viewer, "button_release_event", (-10, -10))
    assert not viewer._is_panning
    after = limits(viewer)
    assert not np.array_equal(after, before)
    emit(viewer, "motion_notify_event", start)
    np.testing.assert_array_equal(limits(viewer), after)
    viewer.toolbar.back()
    np.testing.assert_allclose(limits(viewer), before)
    viewer.toolbar.forward()
    np.testing.assert_allclose(limits(viewer), after)


def test_other_widget_lock_prevents_custom_pan(viewer):
    owner = object()
    before = limits(viewer)
    start, end = viewer.ax.transData.transform([(3, 3), (7, 7)])
    viewer.canvas.widgetlock(owner)
    try:
        emit(viewer, "button_press_event", start)
        emit(viewer, "motion_notify_event", end)
        emit(viewer, "button_release_event", end)
        assert not viewer._is_panning
        np.testing.assert_array_equal(limits(viewer), before)
    finally:
        viewer.canvas.widgetlock.release(owner)


@pytest.mark.parametrize(
    "interrupt",
    ["zoom", "lock", "focus_out", "hide", "deactivate", "escape", "request", "theme"],
)
def test_interrupted_default_drag_cannot_keep_panning(viewer, interrupt):
    start, end = viewer.ax.transData.transform([(3, 3), (7, 7)])
    before = limits(viewer)
    emit(viewer, "button_press_event", start)
    assert viewer._is_panning
    owner = object()
    if interrupt == "zoom":
        viewer.toolbar.zoom()
    elif interrupt == "lock":
        viewer.canvas.widgetlock(owner)
    elif interrupt == "escape":
        QApplication.sendEvent(
            viewer.canvas, QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier)
        )
    elif interrupt == "request":
        viewer.plot_optic(preserve_zoom=True)
    elif interrupt == "theme":
        serial = viewer.connector.calculation_jobs._serial
        viewer.update_theme("light")
        assert viewer.connector.calculation_jobs._serial == serial
    else:
        event_type = {
            "focus_out": QEvent.Type.FocusOut,
            "hide": QEvent.Type.Hide,
            "deactivate": QEvent.Type.WindowDeactivate,
        }[interrupt]
        QApplication.sendEvent(viewer.canvas, QEvent(event_type))
    try:
        emit(viewer, "motion_notify_event", end)
        assert not viewer._is_panning
        np.testing.assert_array_equal(limits(viewer), before)
    finally:
        if interrupt == "lock":
            viewer.canvas.widgetlock.release(owner)
    emit(viewer, "button_release_event", end)


def test_click_without_drag_and_missing_coordinates_do_not_move_view(viewer):
    pixel = viewer.ax.transData.transform((3, 3))
    before = limits(viewer)
    viewer.toolbar.zoom()
    emit(viewer, "button_press_event", pixel)
    emit(viewer, "button_release_event", pixel)
    np.testing.assert_array_equal(limits(viewer), before)
    viewer.on_mouse_move_on_plot(
        SimpleNamespace(inaxes=viewer.ax, xdata=None, ydata=None)
    )
    assert not viewer.cursor_coord_label.isVisible()


def test_navigation_does_not_replot_optic(viewer, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Navigation must not recalculate the optical plot")

    monkeypatch.setattr(viewer, "plot_optic", unexpected)
    serial = viewer.connector.calculation_jobs._serial
    start, end = viewer.ax.transData.transform([(3, 3), (7, 7)])
    viewer.toolbar.zoom()
    emit(viewer, "button_press_event", start)
    emit(viewer, "motion_notify_event", end)
    emit(viewer, "button_release_event", end)
    viewer.toolbar.back()
    viewer.toolbar.forward()
    assert viewer.connector.calculation_jobs._serial == serial


def test_async_scene_presentation_finishes_drag_started_while_calculating(viewer, qapp):
    previous_data = viewer.layout_job.data
    serial = viewer.connector.calculation_jobs._serial
    viewer.num_rays_spinbox.setValue(5)
    viewer.plot_optic(preserve_zoom=True)
    assert viewer.connector.calculation_jobs._serial == serial + 1
    assert viewer.layout_job.data is previous_data  # Last good scene stays visible.
    before = limits(viewer)
    start, end = viewer.ax.transData.transform([(3, 3), (4, 4)])
    emit(viewer, "button_press_event", start)
    emit(viewer, "motion_notify_event", end)
    assert viewer._is_panning
    dragged = limits(viewer)
    assert not np.array_equal(dragged, before)

    wait_for(qapp, lambda: viewer.layout_job.data is not previous_data)
    assert not viewer._is_panning
    assert viewer._pan_start_x is None and viewer._pan_start_y is None
    np.testing.assert_allclose(limits(viewer), dragged)
    emit(viewer, "motion_notify_event", start)
    np.testing.assert_allclose(limits(viewer), dragged)
    emit(viewer, "button_release_event", start)
    viewer.toolbar.back()
    np.testing.assert_allclose(limits(viewer), before)
    viewer.toolbar.forward()
    np.testing.assert_allclose(limits(viewer), dragged)
    assert viewer.connector.calculation_jobs._serial == serial + 1


def test_default_pan_is_anchored_through_multiple_motion_events(viewer):
    before = limits(viewer)
    start, middle, end = viewer.ax.transData.transform([(3, 3), (4, 4), (5, 5)])
    emit(viewer, "button_press_event", start)
    emit(viewer, "motion_notify_event", middle)
    emit(viewer, "motion_notify_event", end)
    emit(viewer, "button_release_event", end)
    np.testing.assert_allclose(limits(viewer), before - 2)
    viewer.toolbar.back()
    np.testing.assert_allclose(limits(viewer), before)
    viewer.toolbar.forward()
    np.testing.assert_allclose(limits(viewer), before - 2)
    viewer.toolbar.home()
    np.testing.assert_allclose(limits(viewer), before)


def test_scroll_zoom_and_coordinate_readout_keep_working(viewer):
    pixel = viewer.ax.transData.transform((3, 3))
    emit(viewer, "motion_notify_event", pixel)
    assert "3.000" in viewer.cursor_coord_label.text()
    before = limits(viewer)
    viewer.canvas.callbacks.process(
        "scroll_event", MouseEvent("scroll_event", viewer.canvas, *pixel, step=1)
    )
    np.testing.assert_allclose(np.diff(limits(viewer)), np.diff(before) / 1.1)
    viewer.canvas.callbacks.process(
        "scroll_event", MouseEvent("scroll_event", viewer.canvas, *pixel, step=-1)
    )
    np.testing.assert_allclose(limits(viewer), before, atol=1e-12)


def test_native_qt_rectangle_drag(viewer, qapp):
    if qapp.platformName() in ("offscreen", "minimal"):
        pytest.skip("Requires the native Qt window platform")
    viewer.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    viewer.resize(800, 600)
    viewer.show()
    qapp.processEvents()
    viewer.canvas.draw()
    before = limits(viewer)
    start, end = viewer.ax.transData.transform([(3, 3), (7, 7)])
    ratio = viewer.canvas.device_pixel_ratio

    def qt_point(pixel):
        return QPoint(
            round(pixel[0] / ratio),
            round(viewer.canvas.height() - pixel[1] / ratio),
        )

    viewer.toolbar.zoom()
    QTest.mousePress(viewer.canvas, Qt.MouseButton.LeftButton, pos=qt_point(start))
    QTest.mouseMove(viewer.canvas, qt_point(end))
    np.testing.assert_array_equal(limits(viewer), before)
    assert not viewer._is_panning
    QTest.mouseRelease(viewer.canvas, Qt.MouseButton.LeftButton, pos=qt_point(end))
    assert not np.array_equal(limits(viewer), before)
