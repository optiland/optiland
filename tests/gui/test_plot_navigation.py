"""Shared 2D gestures preserve scale, history, tool ownership and Qt themes."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

from optiland_gui.widgets.plot_navigation import PlotNavigationToolbar


@pytest.fixture
def plot(qapp):
    widget = QWidget()
    widget.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    canvas = FigureCanvasQTAgg(Figure())
    ax = canvas.figure.add_subplot()
    toolbar = PlotNavigationToolbar(canvas, widget)
    layout = QVBoxLayout(widget)
    layout.addWidget(toolbar)
    layout.addWidget(canvas)
    widget.resize(800, 600)
    widget.show()
    qapp.processEvents()
    ax.set(xlim=(0, 10), ylim=(0, 10))
    canvas.draw()
    yield SimpleNamespace(widget=widget, canvas=canvas, ax=ax, toolbar=toolbar)
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def emit(plot, name, pixel, button=MouseButton.RIGHT, **kwargs):
    if name == "motion_notify_event":
        kwargs.setdefault("buttons", {button})
    event = MouseEvent(name, plot.canvas, *pixel, button=button, **kwargs)
    plot.canvas.callbacks.process(name, event)
    return event


def limits(plot):
    return np.array([plot.ax.get_xlim(), plot.ax.get_ylim()])


@pytest.mark.parametrize("button", [MouseButton.RIGHT, MouseButton.MIDDLE])
@pytest.mark.parametrize("mode", [None, "pan", "zoom"])
def test_direct_pan_keeps_scale_and_one_history_entry(plot, button, mode):
    if mode:
        getattr(plot.toolbar, mode)()
    initial = limits(plot)
    start, middle, end = plot.ax.transData.transform([(3, 3), (4, 4), (6, 6)])
    emit(plot, "button_press_event", start, button)
    emit(plot, "motion_notify_event", middle, button)
    emit(plot, "motion_notify_event", end, button)
    emit(plot, "button_release_event", (-10, -10), button)
    moved = limits(plot)
    assert not np.array_equal(initial, moved)
    np.testing.assert_allclose(np.diff(moved), np.diff(initial))
    assert len(plot.toolbar._nav_stack) == 2
    assert plot.canvas.widgetlock.locked() == bool(mode)
    plot.toolbar.back()
    np.testing.assert_allclose(limits(plot), initial)
    plot.toolbar.forward()
    np.testing.assert_allclose(limits(plot), moved)


def test_left_drag_is_reserved_for_selection_unless_a_tool_is_selected(plot):
    initial = limits(plot)
    start, end = plot.ax.transData.transform([(3, 3), (6, 6)])
    for event, pixel in [
        ("button_press_event", start),
        ("motion_notify_event", end),
        ("button_release_event", end),
    ]:
        emit(plot, event, pixel, MouseButton.LEFT)
    np.testing.assert_array_equal(limits(plot), initial)
    assert not plot.toolbar.is_dragging


def test_click_focus_routes_escape_to_the_plot_tool(plot, qapp):
    plot.widget.activateWindow()
    qapp.processEvents()
    plot.canvas.clearFocus()
    plot.toolbar.zoom()
    QTest.mouseClick(plot.canvas, Qt.MouseButton.LeftButton, pos=QPoint(200, 200))
    assert qapp.focusWidget() is plot.canvas
    QTest.keyClick(qapp.focusWidget(), Qt.Key.Key_Escape)
    assert not plot.toolbar.mode and not plot.toolbar.is_dragging


@pytest.mark.parametrize("mode", [None, "pan", "zoom"])
def test_wheel_works_in_armed_tools_but_cannot_interrupt_a_drag(plot, mode):
    if mode:
        getattr(plot.toolbar, mode)()
    pixel = plot.ax.transData.transform((3, 3))
    initial = limits(plot)
    emit(plot, "scroll_event", pixel, step=1)
    np.testing.assert_allclose(np.diff(limits(plot)), np.diff(initial) / 1.2)
    after = limits(plot)
    button = MouseButton.LEFT if mode else MouseButton.RIGHT
    emit(plot, "button_press_event", pixel, button)
    emit(plot, "scroll_event", pixel, step=-1)
    np.testing.assert_array_equal(limits(plot), after)
    QApplication.sendEvent(
        plot.canvas, QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier)
    )
    assert not plot.toolbar.is_dragging and not plot.toolbar.mode
    assert not plot.canvas.widgetlock.locked()


@pytest.mark.parametrize("scale", ["linear", "log", "symlog"])
@pytest.mark.parametrize("inverted", [False, True])
def test_wheel_anchor_and_pan_use_axis_scale(plot, scale, inverted):
    plot.ax.set(xscale=scale, yscale=scale, xlim=(1, 100), ylim=(1, 100))
    if inverted:
        plot.ax.invert_xaxis()
    plot.canvas.draw()
    initial = limits(plot)
    pixel = plot.ax.transData.transform((10, 10))
    event = emit(plot, "scroll_event", pixel, step=2)
    np.testing.assert_allclose(
        plot.ax.transData.transform((event.xdata, event.ydata)), pixel
    )
    assert plot.ax.xaxis_inverted() == inverted
    plot.toolbar.back()
    np.testing.assert_allclose(limits(plot), initial)
    plot.canvas.draw()
    start, end = plot.ax.transData.transform([(10, 10), (20, 20)])
    emit(plot, "button_press_event", start)
    emit(plot, "motion_notify_event", end)
    emit(plot, "button_release_event", end)
    assert not np.array_equal(limits(plot), initial)
    for axis, before, after in zip(
        (plot.ax.xaxis, plot.ax.yaxis), initial, limits(plot), strict=True
    ):
        transform = axis.get_transform()
        np.testing.assert_allclose(
            np.diff(transform.transform(before)), np.diff(transform.transform(after))
        )


@pytest.mark.parametrize(
    "interrupt", ["escape", "focus_out", "hide", "deactivate", "update", "pan"]
)
def test_interrupted_rectangle_clears_feedback_without_applying_view(
    plot, monkeypatch, interrupt
):
    rectangles = []
    monkeypatch.setattr(plot.canvas, "drawRectangle", rectangles.append)
    before = limits(plot)
    start, end = plot.ax.transData.transform([(3, 3), (7, 7)])
    plot.toolbar.zoom()
    emit(plot, "button_press_event", start, MouseButton.LEFT)
    emit(plot, "motion_notify_event", end, MouseButton.LEFT)
    assert rectangles[-1] is not None
    motion_callback = plot.toolbar._zoom_info.cid
    history_length = len(plot.toolbar._nav_stack)

    if interrupt in ("update", "pan"):
        getattr(plot.toolbar, interrupt)()
    elif interrupt == "escape":
        QApplication.sendEvent(
            plot.canvas, QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier)
        )
    else:
        event_type = {
            "focus_out": QEvent.Type.FocusOut,
            "hide": QEvent.Type.Hide,
            "deactivate": QEvent.Type.WindowDeactivate,
        }[interrupt]
        QApplication.sendEvent(plot.canvas, QEvent(event_type))

    assert not plot.toolbar.is_dragging
    assert rectangles[-1] is None
    assert motion_callback not in plot.canvas.callbacks.callbacks["motion_notify_event"]
    emit(plot, "motion_notify_event", end, MouseButton.LEFT)
    emit(plot, "button_release_event", end, MouseButton.LEFT)
    np.testing.assert_array_equal(limits(plot), before)
    assert len(plot.toolbar._nav_stack) == (0 if interrupt == "update" else history_length)


@pytest.mark.parametrize(
    "interrupt", ["update", "pan", "zoom", "home", "back", "forward", "lost_buttons"]
)
def test_interrupt_finishes_drag_and_releases_its_lock(plot, interrupt):
    start, end = plot.ax.transData.transform([(3, 3), (5, 5)])
    emit(plot, "button_press_event", start)
    emit(plot, "motion_notify_event", end)
    if interrupt == "lost_buttons":
        emit(plot, "motion_notify_event", end, buttons=set())
    else:
        getattr(plot.toolbar, interrupt)()
    assert not plot.toolbar.is_dragging
    assert plot.canvas.widgetlock.locked() == (interrupt in ("pan", "zoom"))
    after = limits(plot)
    emit(plot, "motion_notify_event", start)
    np.testing.assert_array_equal(limits(plot), after)


def test_second_button_does_not_replace_active_gesture(plot):
    start, end = plot.ax.transData.transform([(3, 3), (5, 5)])
    emit(plot, "button_press_event", start)
    emit(plot, "button_press_event", end, MouseButton.MIDDLE)
    emit(plot, "button_release_event", end, MouseButton.MIDDLE)
    assert plot.toolbar.is_dragging
    emit(plot, "button_release_event", start)
    assert not plot.toolbar.is_dragging
    assert len(plot.toolbar._nav_stack) == 1  # A click doesn't add a pan.
    plot.toolbar.zoom()
    emit(plot, "button_press_event", start, MouseButton.LEFT)
    emit(plot, "button_press_event", end)
    emit(plot, "button_release_event", end)
    assert plot.toolbar._zoom_info is not None
    emit(plot, "button_release_event", end, MouseButton.LEFT)
    assert not plot.toolbar.is_dragging


def test_lock_and_non_navigable_axes_block_wheel_and_pan(plot):
    initial = limits(plot)
    pixel = plot.ax.transData.transform((5, 5))
    owner = object()
    plot.canvas.widgetlock(owner)
    emit(plot, "scroll_event", pixel, step=1)
    emit(plot, "button_press_event", pixel)
    assert plot.canvas.widgetlock.isowner(owner)
    plot.canvas.widgetlock.release(owner)
    plot.ax.set_navigate(False)
    emit(plot, "scroll_event", pixel, step=1)
    emit(plot, "button_press_event", pixel)
    emit(plot, "button_press_event", (-10, -10))
    assert not plot.toolbar.is_dragging
    np.testing.assert_array_equal(limits(plot), initial)


def test_twinned_axes_pan_together(plot):
    twin = plot.ax.twinx()
    twin.set_ylim(100, 200)
    plot.canvas.draw()
    initial = limits(plot)
    pixel = twin.transData.transform((4, 140))
    emit(plot, "button_press_event", pixel)
    emit(plot, "motion_notify_event", pixel + (40, 40))
    emit(plot, "button_release_event", pixel + (40, 40))
    assert twin.get_ylim() != (100, 200)
    assert plot.ax.get_ylim() != tuple(initial[1])
    np.testing.assert_allclose(np.diff(twin.get_ylim()), 100)
    np.testing.assert_allclose(np.diff(limits(plot)), np.diff(initial))


def test_repeated_wheel_zoom_keeps_finite_span_and_rejects_collapsed_view(plot):
    event = SimpleNamespace(inaxes=plot.ax, xdata=5.0, ydata=5.0, step=10)
    for _ in range(200):
        plot.toolbar._on_scroll(event)
    before = limits(plot)
    history_length = len(plot.toolbar._nav_stack)
    assert np.isfinite(before).all()
    assert np.all(np.diff(before) > 0)
    # Once the smallest useful span is reached, another wheel event must
    # preserve both limits and history rather than creating a collapsed view.
    plot.toolbar._on_scroll(event)
    np.testing.assert_array_equal(limits(plot), before)
    assert len(plot.toolbar._nav_stack) == history_length
    for step in (float("nan"), 0):
        event.step = step
        plot.toolbar._on_scroll(event)
    event.step, event.xdata = 1, float("inf")
    plot.toolbar._on_scroll(event)
    np.testing.assert_array_equal(limits(plot), before)
    assert len(plot.toolbar._nav_stack) == history_length


def test_escape_without_navigation_is_available_to_parent_widgets(plot):
    before = limits(plot)
    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.NoModifier)
    assert not plot.toolbar.eventFilter(plot.canvas, event)
    assert not plot.toolbar.mode and not plot.toolbar.is_dragging
    assert not plot.canvas.widgetlock.locked()
    np.testing.assert_array_equal(limits(plot), before)


@pytest.mark.parametrize("bounds", [(1e-99, 1e99), (1e-300, 1e-200)])
def test_extreme_log_zoom_cannot_overflow_or_underflow(plot, bounds):
    plot.ax.set_xscale("log")
    plot.ax.set_xlim(bounds)
    initial = limits(plot)
    center = 10 ** np.mean(np.log10(bounds))
    pixel = plot.ax.transData.transform((center, 5))
    emit(plot, "scroll_event", pixel, step=-10)
    np.testing.assert_array_equal(limits(plot), initial)
    assert len(plot.toolbar._nav_stack) == 0


def test_3d_axes_keep_their_existing_navigation(plot):
    plot.canvas.figure.clear()
    axes = plot.canvas.figure.add_subplot(projection="3d")
    plot.canvas.draw()
    pixel = axes.bbox.get_points().mean(axis=0)
    initial = axes.get_xlim(), axes.get_ylim(), axes.get_zlim()
    emit(plot, "scroll_event", pixel, step=1)
    assert (axes.get_xlim(), axes.get_ylim(), axes.get_zlim()) == initial
    emit(plot, "button_press_event", pixel)
    assert plot.toolbar._pointer_pan is None
    emit(plot, "button_release_event", pixel)


@pytest.mark.parametrize(
    "button", [Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton]
)
def test_native_qt_direct_pan(plot, qapp, button):
    if qapp.platformName() in ("offscreen", "minimal"):
        pytest.skip("Requires native Qt mouse dispatch")
    initial = limits(plot)
    start, end = plot.ax.transData.transform([(3, 3), (5, 5)])
    ratio = plot.canvas.device_pixel_ratio

    def point(pixel):
        return QPoint(
            round(pixel[0] / ratio), round(plot.canvas.height() - pixel[1] / ratio)
        )

    QTest.mousePress(plot.canvas, button, pos=point(start))
    QTest.mouseMove(plot.canvas, point(end))
    QTest.mouseRelease(plot.canvas, button, pos=point(end))
    assert not plot.toolbar.is_dragging
    assert not np.array_equal(limits(plot), initial)
    np.testing.assert_allclose(np.diff(limits(plot)), np.diff(initial))
