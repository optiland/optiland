"""A title drag can place a panel relative to populated sibling content."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QElapsedTimer, QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QTextEdit

from optiland_gui.widgets.custom_dock_widget import CustomDockWidget
from tests.gui.test_dock_dragging import dock_window as dock_window
from tests.gui.test_dock_dragging import move_to


@pytest.mark.parametrize("placement", ["tab", "right", "bottom"])
def test_drag_places_panel_in_sibling_tabs_or_splits(qapp, dock_window, placement):
    window, source = dock_window
    target = CustomDockWidget("Results", window)
    target.setObjectName("ResultsDock")
    content = QTextEdit()
    content.setPlainText("Retained analysis results\n" * 20)
    content.setMinimumSize(120, 90)
    target.setWidget(content)
    window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, target)
    window.resizeDocks([source, target], [240, 300], Qt.Orientation.Horizontal)
    qapp.processEvents()
    title = source.titleBarWidget()
    start = QPoint(20, title.height() // 2)
    global_start = title.mapToGlobal(start)
    QTest.mousePress(title, Qt.MouseButton.LeftButton, pos=start)
    move_to(title, global_start + QPoint(180, window.height() + 40))
    assert source.isFloating()
    if placement == "right":
        local_target = QPoint(target.width() - 10, target.height() // 2)
    elif placement == "bottom":
        local_target = QPoint(target.width() // 2, target.height() - 10)
    else:
        local_target = target.rect().center()
    destination = target.mapToGlobal(local_target)
    move_to(title, destination)
    QTest.mouseRelease(
        title, Qt.MouseButton.LeftButton, pos=title.mapFromGlobal(destination)
    )
    timer = QElapsedTimer()
    timer.start()
    while source.isFloating() and timer.elapsed() < 1000:
        QTest.qWait(10)
    assert not source.isFloating()
    if placement == "tab":
        assert source in window.tabifiedDockWidgets(target)
    elif placement == "right":
        assert source.x() > target.geometry().right()
        assert source.y() == target.y()
    else:
        assert source.y() > target.geometry().bottom()
        assert source.x() == target.x()
    assert target.widget() is content
    assert content.toPlainText() == "Retained analysis results\n" * 20
    state = window.saveState()
    source.setFloating(True)
    assert window.restoreState(state)
    qapp.processEvents()
    assert not source.isFloating()
    if placement == "tab":
        assert source in window.tabifiedDockWidgets(target)
    assert target.widget() is content
