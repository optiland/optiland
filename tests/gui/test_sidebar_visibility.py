"""Sidebar visibility and remembered width use native Qt actions."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QSettings, Qt
from PySide6.QtGui import QIcon
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QDockWidget,
    QMainWindow,
    QMenu,
    QTextEdit,
    QToolBar,
    QToolButton,
    QWidget,
)

from optiland_gui.action_manager import ActionManager
from optiland_gui.config import THEME_DARK_PATH, THEME_LIGHT_PATH
from optiland_gui.main_window import MainWindow
from optiland_gui.optiland_connector import OptilandConnector
from optiland_gui.panel_manager import PanelManager
from optiland_gui.widgets.sidebar import SidebarDockWidget, SidebarWidget


class SidebarWindow(MainWindow):
    def __init__(self, settings_path, connector):
        QMainWindow.__init__(self)
        self.resize(1200, 850)
        self.setAnimated(False)
        self.setDockNestingEnabled(True)
        self.setCentralWidget(QWidget())
        self.settings = QSettings(str(settings_path), QSettings.Format.IniFormat)
        self.next_save_slot_index = self.settings.value(
            "Layouts/NextSaveSlot", 1, type=int
        )
        self.connector = connector
        self.iface = SimpleNamespace()
        self.toast_manager = SimpleNamespace(notify=MagicMock())
        manager = self.panel_manager = PanelManager(self, connector)
        manager.sidebar = SidebarDockWidget(self)
        manager.sidebar.setWidget(SidebarWidget(self))
        self.addDockWidget(Qt.LeftDockWidgetArea, manager.sidebar)
        manager.all_docks = [manager.sidebar]
        dock = QDockWidget("Retained results", self)
        dock.setObjectName("RetainedResults")
        dock.setWidget(QTextEdit("Unsaved draft and retained results"))
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        manager.all_docks.append(dock)
        self.action_manager = ActionManager(self, connector)
        self.action_manager._create_view_actions()
        self.action_manager._create_layout_actions()
        self.action_manager._create_file_actions()
        self.action_manager._create_theme_actions()

    showEvent = QMainWindow.showEvent
    resizeEvent = QMainWindow.resizeEvent
    closeEvent = QMainWindow.closeEvent


@pytest.fixture
def workspace(qapp, tmp_path):
    connector = OptilandConnector()
    window = SidebarWindow(tmp_path / "sidebar.ini", connector)
    window.show()
    qapp.processEvents()
    yield window
    window.close()
    window.deleteLater()
    connector.calculation_jobs.shutdown()
    connector.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def contents(window):
    return [
        (dock.widget(), dock.widget().toPlainText())
        for dock in window.panel_manager.all_docks
        if isinstance(dock.widget(), QTextEdit)
    ]


@pytest.mark.parametrize("theme_path", [THEME_DARK_PATH, THEME_LIGHT_PATH])
def test_leftmost_toolbar_sidebar_action_syncs_menu_keyboard_and_native_state(
    qapp, workspace, theme_path
):
    sidebar = workspace.panel_manager.sidebar
    toolbar = QToolBar(workspace)
    workspace._populate_quick_actions_toolbar(toolbar)
    workspace.addToolBar(toolbar)
    workspace.current_theme_path = theme_path
    workspace.load_stylesheets()
    action = sidebar.toggleViewAction()
    menu = QMenu("View", workspace)
    menu.addAction(action)
    button = toolbar.widgetForAction(action)
    qapp.processEvents()
    before = contents(workspace)
    assert toolbar.actions()[0] is action
    assert button.defaultAction() is menu.actions()[0]
    assert button.toolButtonStyle() == Qt.ToolButtonIconOnly
    assert button.accessibleName() == "Show/hide navigation sidebar"
    theme = "dark" if theme_path == THEME_DARK_PATH else "light"
    assert (
        button.icon().pixmap(24, 24).toImage()
        == QIcon(f":/icons/{theme}/menu.svg").pixmap(24, 24).toImage()
    )
    assert (
        button.geometry().right()
        < toolbar.widgetForAction(workspace.action_manager.get_action("new")).x()
    )
    QTest.mouseClick(button, Qt.LeftButton)
    qapp.processEvents()
    assert sidebar.isHidden() and not action.isChecked() and not button.isChecked()
    hidden = workspace.saveState()
    button.setFocus()
    QTest.keyClick(button, Qt.Key_Space)
    qapp.processEvents()
    assert sidebar.isVisible() and action.isChecked() and button.isChecked()
    visible = workspace.saveState()
    assert workspace.restoreState(hidden)
    qapp.processEvents()
    assert sidebar.isHidden() and not button.isChecked()
    assert workspace.restoreState(visible)
    qapp.processEvents()
    assert sidebar.isVisible() and button.isChecked()
    action.trigger()
    qapp.processEvents()
    assert sidebar.isHidden() and not button.isChecked()
    workspace.current_theme_path = (
        THEME_LIGHT_PATH if theme_path == THEME_DARK_PATH else THEME_DARK_PATH
    )
    old = button.icon().cacheKey()
    workspace.load_stylesheets()
    assert button.icon().cacheKey() != old
    assert sidebar.isHidden() and contents(workspace) == before


def test_sidebar_navigation_starts_at_top_and_survives_width_changes(qapp):
    sidebar = SidebarWidget()
    sidebar.resize(60, 600)
    sidebar.show()
    qapp.processEvents()
    buttons = sidebar.findChildren(QToolButton)
    assert len(buttons) == 7
    assert sidebar.layout().itemAt(0).widget() is buttons[0]
    assert buttons[0].y() == sidebar.layout().contentsMargins().top()
    assert all(button.toolButtonStyle() == Qt.ToolButtonIconOnly for button in buttons)
    selected = []
    sidebar.menuSelected.connect(selected.append)
    analysis = sidebar.findChild(QToolButton, "sidebar-btn-analysis")
    QTest.mouseClick(analysis, Qt.LeftButton)
    sidebar.resize(150, 600)
    qapp.processEvents()
    assert selected == ["analysis"] and analysis.isChecked()
    assert analysis.text() == "Analysis"
    assert all(
        button.toolButtonStyle() == Qt.ToolButtonTextUnderIcon for button in buttons
    )
    sidebar.close()
    sidebar.deleteLater()


@pytest.mark.parametrize("theme_path", [THEME_DARK_PATH, THEME_LIGHT_PATH])
@pytest.mark.parametrize("width", [60, 75, 110, 150])
def test_sidebar_toggle_retains_resized_width_and_navigation(
    qapp, workspace, theme_path, width
):
    sidebar = workspace.panel_manager.sidebar
    sidebar.setWidget(SidebarWidget(workspace))
    workspace.action_manager._create_file_actions()
    toolbar = QToolBar(workspace)
    workspace._populate_quick_actions_toolbar(toolbar)
    workspace.addToolBar(toolbar)
    workspace.current_theme_path = theme_path
    workspace.load_stylesheets()
    # Establish an exact user width through Qt's layout, then release limits
    # just as a splitter drag does; testing only a fixed 60px dock hid the bug.
    sidebar.setFixedWidth(width)
    workspace.resizeDocks([sidebar], [width], Qt.Horizontal)
    workspace.layout().activate()
    sidebar.setMinimumWidth(60)
    sidebar.setMaximumWidth(150)
    qapp.processEvents()
    before = sidebar.width()
    collapsed = sidebar.widget()._is_collapsed
    contents_before = contents(workspace)
    button = toolbar.widgetForAction(sidebar.toggleViewAction())
    assert before == width
    for _ in range(3):
        QTest.mouseClick(button, Qt.LeftButton)
        qapp.processEvents()
        assert sidebar.isHidden()
        QTest.mouseClick(button, Qt.LeftButton)
        qapp.processEvents()
        assert sidebar.isVisible() and sidebar.width() == before
        assert sidebar.widget()._is_collapsed == collapsed
        assert sidebar.minimumWidth() == 60 and sidebar.maximumWidth() == 150
    # A resize after showing must update the width remembered on the next hide.
    other_width = 75 if width > 80 else 130
    sidebar.setFixedWidth(other_width)
    workspace.resizeDocks([sidebar], [other_width], Qt.Horizontal)
    workspace.layout().activate()
    sidebar.setMinimumWidth(60)
    sidebar.setMaximumWidth(150)
    qapp.processEvents()
    assert sidebar.width() == other_width
    sidebar.hide()  # Programmatic hide uses the same memory as the View menu.
    sidebar.toggleViewAction().trigger()
    qapp.processEvents()
    assert sidebar.width() == other_width
    assert contents(workspace) == contents_before


def test_sidebar_width_follows_native_restore_and_new_limits(qapp, workspace):
    states = {}
    sidebar = workspace.panel_manager.sidebar
    sidebar.setWidget(SidebarWidget(workspace))
    for slot, width in ((1, 110), (2, 75)):
        sidebar.setFixedWidth(width)
        workspace.resizeDocks([sidebar], [width], Qt.Horizontal)
        workspace.layout().activate()
        sidebar.setMinimumWidth(60)
        sidebar.setMaximumWidth(150)
        qapp.processEvents()
        states[slot] = workspace.saveState()
    for slot in (1, 2, 1):
        sidebar.hide()
        assert workspace.restoreState(states[slot])
        qapp.processEvents()
        assert sidebar.isVisible()
        restored_width = sidebar.width()
        sidebar.toggleViewAction().trigger()
        sidebar.toggleViewAction().trigger()
        qapp.processEvents()
        assert sidebar.width() == restored_width
    sidebar.setFixedWidth(110)
    workspace.resizeDocks([sidebar], [110], Qt.Horizontal)
    workspace.layout().activate()
    sidebar.setMinimumWidth(60)
    sidebar.setMaximumWidth(150)
    qapp.processEvents()
    sidebar.hide()
    sidebar.setMaximumWidth(90)
    sidebar.toggleViewAction().trigger()
    qapp.processEvents()
    assert sidebar.width() == 90
    assert sidebar.minimumWidth() == 60 and sidebar.maximumWidth() == 90


def test_sidebar_width_ignores_parent_hide_and_floating_geometry(qapp, workspace):
    sidebar = workspace.panel_manager.sidebar
    sidebar.setWidget(SidebarWidget(workspace))
    sidebar.setFixedWidth(110)
    workspace.resizeDocks([sidebar], [110], Qt.Horizontal)
    workspace.layout().activate()
    sidebar.setMinimumWidth(60)
    sidebar.setMaximumWidth(150)
    qapp.processEvents()
    sidebar.hide()
    workspace.hide()
    workspace.show()
    sidebar.toggleViewAction().trigger()
    qapp.processEvents()
    assert sidebar.width() == 110
    sidebar.setFloating(True)
    sidebar.resize(140, 500)
    qapp.processEvents()
    geometry = sidebar.geometry()
    limits = sidebar.minimumWidth(), sidebar.maximumWidth()
    sidebar.toggleViewAction().trigger()
    sidebar.toggleViewAction().trigger()
    qapp.processEvents()
    assert sidebar.isFloating() and sidebar.geometry() == geometry
    assert (sidebar.minimumWidth(), sidebar.maximumWidth()) == limits


def test_initially_hidden_sidebar_can_show_without_remembered_width(qapp):
    window = QMainWindow()
    sidebar = SidebarDockWidget(window)
    sidebar.setWidget(SidebarWidget(window))
    window.addDockWidget(Qt.LeftDockWidgetArea, sidebar)
    sidebar.toggleViewAction().trigger()
    window.show()
    qapp.processEvents()
    assert sidebar.isVisible()
    assert sidebar.minimumWidth() < sidebar.maximumWidth()
    window.close()
    window.deleteLater()


def test_production_sidebar_enables_native_visibility_action(
    qapp, workspace, monkeypatch
):
    # Construct the production docks, stubbing unrelated viewers and kernel.
    # A default QDockWidget fixture would conceal Qt's non-closable restriction.
    import optiland_gui.panel_manager as panels

    def content(*args, **kwargs):
        return QWidget()

    def viewer(*args):
        widget = QWidget()
        widget.viewer2D = SimpleNamespace(set_interaction_state=MagicMock())
        widget.viewer3D = None
        return widget

    monkeypatch.setattr(panels, "ViewerPanel", viewer)
    for name in (
        "LensEditor",
        "SystemPropertiesPanel",
        "AnalysisPanel",
        "OptimizationPanel",
        "PythonTerminalWidget",
    ):
        monkeypatch.setattr(panels, name, content)
    window = QMainWindow()
    window.iface = object()
    manager = PanelManager(window, workspace.connector)
    manager.create_all_panels(window)
    window.addDockWidget(Qt.LeftDockWidgetArea, manager.sidebar)
    window.show()
    qapp.processEvents()
    action = manager.sidebar.toggleViewAction()
    assert action.isEnabled() and action.isChecked()
    assert manager.sidebar.titleBarWidget().findChildren(QToolButton) == []
    action.trigger()
    qapp.processEvents()
    assert manager.sidebar.isHidden() and not action.isChecked()
    action.trigger()
    qapp.processEvents()
    assert manager.sidebar.isVisible() and action.isChecked()
    window.close()
    window.deleteLater()
