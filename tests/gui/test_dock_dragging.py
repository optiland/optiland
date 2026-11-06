"""Panel title gestures follow Qt's threshold and docking lifecycle."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QElapsedTimer, QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDockWidget, QMainWindow, QWidget

from optiland_gui.widgets.custom_dock_widget import CustomDockWidget


@pytest.fixture
def dock_window(qapp):
    window = QMainWindow()
    window.setAnimated(False)
    window.setDockNestingEnabled(True)
    window.setCentralWidget(QWidget())
    window.setGeometry(40, 40, 680, 400)
    dock = CustomDockWidget("Panel", window)
    dock.setObjectName("TestPanel")
    content = QWidget()
    content.setMinimumSize(120, 90)
    dock.setWidget(content)
    window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
    window.show()
    qapp.processEvents()
    yield window, dock
    window.close()
    window.deleteLater()
    qapp.processEvents()


def move_to(title, global_position):
    QTest.mouseMove(title, title.mapFromGlobal(global_position))
    QApplication.processEvents()


@pytest.mark.parametrize("floating", [False, True])
def test_click_jitter_does_not_move_or_undock_panel(qapp, dock_window, floating):
    _, dock = dock_window
    dock.setFloating(floating)
    qapp.processEvents()
    title = dock.titleBarWidget()
    start = QPoint(20, title.height() // 2)
    global_start = title.mapToGlobal(start)
    before = dock.geometry()
    QTest.mousePress(title, Qt.MouseButton.LeftButton, pos=start)
    move_to(title, global_start + QPoint(qapp.startDragDistance() - 1, 0))
    assert dock.isFloating() == floating
    assert dock.geometry() == before
    QTest.mouseRelease(
        title, Qt.MouseButton.LeftButton, pos=title.mapFromGlobal(global_start)
    )
    qapp.processEvents()
    assert dock.isFloating() == floating


def test_undock_drag_keeps_grab_point_in_title_and_releases(qapp, dock_window):
    window, dock = dock_window
    title = dock.titleBarWidget()
    start = QPoint(20, title.height() // 2)
    global_start = title.mapToGlobal(start)
    destination = global_start + QPoint(180, window.height() + 40)
    QTest.mousePress(title, Qt.MouseButton.LeftButton, pos=start)
    move_to(title, destination)
    assert dock.isFloating()
    # No application-side move may displace the title away from Qt's grab point.
    assert title.rect().contains(title.mapFromGlobal(destination))
    QTest.mouseRelease(
        title, Qt.MouseButton.LeftButton, pos=title.mapFromGlobal(destination)
    )
    qapp.processEvents()
    assert dock.isFloating()
    settled = dock.geometry()
    move_to(title, destination + QPoint(40, 25))
    assert dock.geometry() == settled


def test_double_click_floats_and_redocks_with_feature_flags(qapp, dock_window):
    _, dock = dock_window
    title = dock.titleBarWidget()
    point = QPoint(20, title.height() // 2)
    QTest.mouseDClick(title, Qt.MouseButton.LeftButton, pos=point)
    qapp.processEvents()
    assert dock.isFloating()
    QTest.mouseDClick(title, Qt.MouseButton.LeftButton, pos=point)
    qapp.processEvents()
    assert not dock.isFloating()
    dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable)
    QTest.mouseDClick(title, Qt.MouseButton.LeftButton, pos=point)
    qapp.processEvents()
    assert not dock.isFloating()


def test_title_label_does_not_obstruct_gesture_hit_area(dock_window):
    _, dock = dock_window
    title = dock.titleBarWidget()
    label = title.title_label
    global_point = label.mapToGlobal(label.rect().center())
    assert QApplication.widgetAt(global_point) is title


@pytest.mark.parametrize(
    "area",
    [
        Qt.DockWidgetArea.LeftDockWidgetArea,
        Qt.DockWidgetArea.RightDockWidgetArea,
        Qt.DockWidgetArea.TopDockWidgetArea,
        Qt.DockWidgetArea.BottomDockWidgetArea,
    ],
)
def test_one_press_can_undock_and_snap_to_each_side(qapp, dock_window, area):
    window, dock = dock_window
    title = dock.titleBarWidget()
    start = QPoint(20, title.height() // 2)
    global_start = title.mapToGlobal(start)
    targets = {
        Qt.DockWidgetArea.LeftDockWidgetArea: QPoint(10, window.height() // 2),
        Qt.DockWidgetArea.RightDockWidgetArea: QPoint(
            window.width() - 10, window.height() // 2
        ),
        Qt.DockWidgetArea.TopDockWidgetArea: QPoint(window.width() // 2, 10),
        Qt.DockWidgetArea.BottomDockWidgetArea: QPoint(
            window.width() // 2, window.height() - 10
        ),
    }
    QTest.mousePress(title, Qt.MouseButton.LeftButton, pos=start)
    move_to(title, global_start + QPoint(180, window.height() + 40))
    assert dock.isFloating()
    destination = window.mapToGlobal(targets[area])
    move_to(title, destination)
    QTest.mouseRelease(
        title, Qt.MouseButton.LeftButton, pos=title.mapFromGlobal(destination)
    )
    # Qt completes plugging a floating dock through its event loop.
    timer = QElapsedTimer()
    timer.start()
    while dock.isFloating() and timer.elapsed() < 1000:
        QTest.qWait(10)
    assert not dock.isFloating()
    assert window.dockWidgetArea(dock) == area


def test_disallowed_side_does_not_capture_floating_panel(qapp, dock_window):
    window, dock = dock_window
    dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)
    dock.setFloating(True)
    dock.move(window.mapToGlobal(QPoint(150, window.height() + 40)))
    qapp.processEvents()
    title = dock.titleBarWidget()
    start = QPoint(20, title.height() // 2)
    QTest.mousePress(title, Qt.MouseButton.LeftButton, pos=start)
    destination = window.mapToGlobal(QPoint(window.width() - 10, window.height() // 2))
    move_to(title, destination)
    QTest.mouseRelease(
        title, Qt.MouseButton.LeftButton, pos=title.mapFromGlobal(destination)
    )
    QTest.qWait(250)
    assert dock.isFloating()
