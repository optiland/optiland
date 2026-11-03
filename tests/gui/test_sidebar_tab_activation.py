"""Sidebar selection must activate dock tabs before frame grips are restacked."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import shiboken6
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDockWidget, QLabel, QToolButton

from optiland_gui.panel_manager import PanelManager
from optiland_gui.widgets.custom_dock_widget import CustomDockWidget
from optiland_gui.widgets.frameless_window import FramelessWindow
from optiland_gui.widgets.sidebar import SidebarWidget


@pytest.fixture(params=[QDockWidget, CustomDockWidget])
def tabbed_frame(qapp, request):
    window = FramelessWindow()
    window.resize(1000, 700)
    window.setDockNestingEnabled(True)
    sidebar = SidebarWidget()
    side_dock = QDockWidget("Sidebar", window)
    side_dock.setObjectName("Sidebar")
    side_dock.setWidget(sidebar)
    window.addDockWidget(Qt.LeftDockWidgetArea, side_dock)
    docks = []
    for index in range(3):
        # Duplicate/translated titles must not be used to identify a tab.
        dock = request.param("Panel", window)
        dock.setObjectName(f"Panel{index}")
        dock.setWidget(QLabel(f"Content {index}"))
        window.addDockWidget(Qt.RightDockWidgetArea, dock)
        docks.append(dock)
    for dock in docks[1:]:
        window.tabifyDockWidget(docks[0], dock)
    manager = SimpleNamespace(
        analysis_dock=docks[0],
        optimization_dock=docks[1],
        lens_editor_dock=docks[2],
        terminal_dock=docks[2],
    )
    sidebar.menuSelected.connect(
        lambda name: PanelManager.on_sidebar_menu_selected(manager, name)
    )
    window.show()
    qapp.processEvents()
    yield window, sidebar, docks
    if shiboken6.isValid(window):
        window.close()
        window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()
    assert qapp.mouseButtons() == Qt.NoButton


def assert_front(window, selected, others):
    assert not selected.visibleRegion().isEmpty()
    assert window.rect().intersects(selected.geometry())
    for dock in others:
        assert dock.visibleRegion().isEmpty()
        assert not window.rect().intersects(dock.geometry())


@pytest.mark.parametrize("layout", ["initial", "restore", "rebuild"])
def test_sidebar_click_activates_tab_and_preserves_resize_grips(
    qapp, tabbed_frame, layout
):
    window, sidebar, docks = tabbed_frame
    if layout == "restore":
        saved = window.saveState()
        docks[0].setFloating(True)
        assert window.restoreState(saved)
    elif layout == "rebuild":
        for dock in docks:
            window.removeDockWidget(dock)
            window.addDockWidget(Qt.RightDockWidgetArea, dock)
        for dock in docks[1:]:
            window.tabifyDockWidget(docks[0], dock)
    for dock in docks:
        dock.show()
    qapp.processEvents()
    for name, selected in (
        ("analysis", docks[0]),
        ("optimization", docks[1]),
        ("analysis", docks[0]),
    ):
        button = sidebar.findChild(QToolButton, f"sidebar-btn-{name}")
        QTest.mouseClick(button, Qt.LeftButton)
        # Native tab activation is synchronous, not a delayed retry of selection.
        assert_front(window, selected, [dock for dock in docks if dock is not selected])
        qapp.processEvents()
        assert_front(window, selected, [dock for dock in docks if dock is not selected])
        for grip in window._grips.values():
            assert window.childAt(grip.geometry().center()) is grip
        assert not window._observing_gesture


def test_plain_raise_and_hidden_tab_activation(tabbed_frame, qapp):
    window, _, docks = tabbed_frame
    for dock in docks:
        dock.hide()
        dock.show()
        dock.raise_()
        assert_front(window, dock, [other for other in docks if other is not dock])
        qapp.processEvents()
        assert_front(window, dock, [other for other in docks if other is not dock])


def test_queued_grip_update_is_owned_by_window(tabbed_frame, qapp):
    window, _, docks = tabbed_frame
    docks[0].raise_()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()
    assert not shiboken6.isValid(window)


def test_sidebar_selection_keeps_a_floating_panel_floating(tabbed_frame, qapp):
    window, sidebar, docks = tabbed_frame
    selected = docks[0]
    selected.setFloating(True)
    qapp.processEvents()
    bounds = selected.geometry()
    selected.hide()
    QTest.mouseClick(
        sidebar.findChild(QToolButton, "sidebar-btn-analysis"), Qt.LeftButton
    )
    qapp.processEvents()
    assert selected.isFloating()
    assert not selected.visibleRegion().isEmpty()
    assert selected.geometry() == bounds
    assert window.tabifiedDockWidgets(docks[1]) == [docks[2]]
    assert not window._observing_gesture
