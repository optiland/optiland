"""Custom panel titles retain Qt's floating edge and corner resize behavior."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMainWindow, QWidget

from optiland_gui.widgets.custom_dock_widget import CustomDockWidget


@pytest.fixture
def floating_panel(qapp):
    window = QMainWindow()
    window.resize(700, 450)
    dock = CustomDockWidget("Panel", window)
    content = QWidget()
    content.setMinimumSize(160, 100)
    dock.setWidget(content)
    window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
    window.show()
    dock.setFloating(True)
    dock.setMaximumSize(350, 300)
    dock.setGeometry(200, 200, 260, 210)
    qapp.processEvents()
    yield dock
    window.close()
    window.deleteLater()
    qapp.processEvents()


@pytest.mark.parametrize(
    "edge",
    [
        "left",
        "right",
        "top",
        "bottom",
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
    ],
)
@pytest.mark.parametrize("motion", ["grow", "minimum", "maximum"])
def test_edges_follow_pointer_clamp_and_anchor_opposite_boundary(
    qapp, floating_panel, edge, motion
):
    dock = floating_panel
    before = dock.geometry()
    x = (
        1
        if "left" in edge
        else dock.width() - 2
        if "right" in edge
        else dock.width() // 2
    )
    y = (
        1
        if "top" in edge
        else dock.height() - 2
        if "bottom" in edge
        else dock.height() // 2
    )
    point = QPoint(x, y)
    global_start = dock.mapToGlobal(point)
    distance = -1000 if motion == "minimum" else 1000 if motion == "maximum" else 30
    dx = -distance if "left" in edge else distance if "right" in edge else 0
    dy = -distance if "top" in edge else distance if "bottom" in edge else 0
    destination = global_start + QPoint(dx, dy)
    expected = QRect(before)
    if dx:
        width = max(
            dock.minimumWidth(), min(dock.maximumWidth(), before.width() + distance)
        )
        expected.setWidth(width)
        if "left" in edge:
            expected.moveRight(before.right())
    if dy:
        height = max(
            dock.minimumHeight(), min(dock.maximumHeight(), before.height() + distance)
        )
        expected.setHeight(height)
        if "top" in edge:
            expected.moveBottom(before.bottom())
    QTest.mousePress(dock, Qt.MouseButton.LeftButton, pos=point)
    QTest.mouseMove(dock, dock.mapFromGlobal(destination))
    qapp.processEvents()
    assert dock.geometry() == expected
    QTest.mouseRelease(
        dock, Qt.MouseButton.LeftButton, pos=dock.mapFromGlobal(destination)
    )
    qapp.processEvents()
    assert dock.geometry() == expected
    assert dock.isFloating()
    QTest.mouseMove(dock, dock.mapFromGlobal(destination + QPoint(40, 25)))
    qapp.processEvents()
    assert dock.geometry() == expected
