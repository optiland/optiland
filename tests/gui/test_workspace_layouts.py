"""Workspace commands change Qt topology while retaining documents and panels."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import QByteArray, QCoreApplication, QEvent, QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QDockWidget,
    QMainWindow,
    QMenu,
    QMenuBar,
    QTextEdit,
    QWidget,
)

from optiland_gui.action_manager import ActionManager
from optiland_gui.main_window import MainWindow
from optiland_gui.optiland_connector import OptilandConnector
from optiland_gui.panel_manager import PanelManager
from optiland_gui.widgets.command_palette import CommandRegistry
from optiland_gui.widgets.python_terminal import PythonTerminalWidget


class ConsoleContent(QTextEdit):
    def _style_sheet_changed(self):
        pass


class ScriptContent(PythonTerminalWidget):
    """Use the actual nested UI construction without starting a Python kernel."""

    def __init__(self, parent):
        QWidget.__init__(self, parent)
        self.current_theme = "dark"
        self.untitled_script_counter = 0
        self.kernel_manager = self.kernel_client = None
        self._define_themes()
        self._setup_ui()


class WorkspaceWindow(MainWindow):
    def __init__(self, settings_path, connector):
        QMainWindow.__init__(self)
        self.resize(1400, 850)
        self.setAnimated(False)
        self.setDockNestingEnabled(True)
        self.setCentralWidget(QWidget())
        self.settings = QSettings(str(settings_path), QSettings.Format.IniFormat)
        self.next_save_slot_index = 1
        self.toast_manager = SimpleNamespace(notify=MagicMock())
        self.connector = connector
        manager = self.panel_manager = PanelManager(self, connector)
        names = (
            ("sidebar", "SidebarDockWidget"),
            ("viewer_dock", "ViewerDock"),
            ("lens_editor_dock", "LensEditorDock"),
            ("system_properties_dock", "SystemPropertiesDock"),
            ("analysis_dock", "AnalysisPanelDock"),
            ("optimization_dock", "OptimizationDock"),
            ("terminal_dock", "TerminalDock"),
        )
        for attribute, name in names:
            dock = QDockWidget(name, self)
            dock.setObjectName(name)
            content = QTextEdit()
            content.setPlainText(f"Retained {name} draft and results")
            content.setMinimumSize(120, 90)
            dock.setWidget(content)
            setattr(manager, attribute, dock)
            manager.all_docks.append(dock)
        manager.sidebar.setFixedWidth(60)
        manager.python_terminal = ScriptContent(self)
        manager.terminal_dock.setWidget(manager.python_terminal)
        manager.python_terminal.script_tabs.widget(0).setPlainText(
            "optic.name = 'draft'"
        )
        manager.viewer_panel = SimpleNamespace(
            viewer2D=SimpleNamespace(plot_optic=MagicMock()),
            viewer3D=SimpleNamespace(render_optic=MagicMock()),
        )
        self.action_manager = ActionManager(self, connector)
        self.action_manager._create_view_actions()
        self.action_manager._create_layout_actions()
        view = self.menuBar().addMenu("&View")
        self._populate_workspace_menu(view.addMenu("&Workspace"))
        manager.setup_default_layout()

    showEvent = QMainWindow.showEvent
    resizeEvent = QMainWindow.resizeEvent
    closeEvent = QMainWindow.closeEvent


@pytest.fixture
def workspace(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "optiland_gui.widgets.python_terminal.RichJupyterWidget", ConsoleContent
    )
    connector = OptilandConnector()
    window = WorkspaceWindow(tmp_path / "workspaces.ini", connector)
    window.show()
    qapp.processEvents()
    window.panel_manager.equalize_workspace()
    qapp.processEvents()
    yield window
    window.close()
    window.deleteLater()
    connector.calculation_jobs.shutdown()
    connector.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def contents(window):
    manager = window.panel_manager
    return {
        dock.objectName(): (
            dock.widget(),
            dock.widget().minimumSize(),
            dock.widget().maximumSize(),
            dock.widget().toPlainText()
            if isinstance(dock.widget(), QTextEdit)
            else None,
        )
        for dock in manager.all_docks
    } | {
        "draft": manager.python_terminal.script_tabs.widget(0).toPlainText(),
        "document": window.connector.get_optic().to_dict(),
        "revision": window.connector.document_state.edit_token,
    }


def topology(window, docks):
    return {
        dock.objectName(): (
            window.dockWidgetArea(dock),
            dock.isFloating(),
            dock.isHidden(),
            sorted(item.objectName() for item in window.tabifiedDockWidgets(dock)),
        )
        for dock in docks
    }


@pytest.mark.parametrize("arrangement", ["two_panes", "four_panes"])
def test_presets_recover_floating_hidden_tabs_without_replacing_contents(
    qapp, workspace, arrangement
):
    manager = workspace.panel_manager
    manager.system_properties_dock.hide()
    manager.optimization_dock.setFloating(True)
    manager.viewer_dock.setFloating(True)
    terminal = manager.python_terminal
    terminal.editor_dock.setFloating(True)
    terminal.console_dock.hide()
    before = contents(workspace)
    commits = []
    workspace.connector.document_state.committed.connect(commits.append)
    for _ in range(3):
        manager.arrange_workspace(arrangement)
        qapp.processEvents()
        manager.equalize_workspace()
        qapp.processEvents()
        assert all(
            not dock.isFloating() and not dock.isHidden() for dock in manager.all_docks
        )
        assert all(
            not dock.isFloating() and not dock.isHidden()
            for dock in (terminal.editor_dock, terminal.console_dock)
        )
        assert not terminal.dock_area.tabifiedDockWidgets(terminal.editor_dock)
        assert contents(workspace) == before
        assert not commits
        right_tabs = workspace.tabifiedDockWidgets(manager.analysis_dock)
        assert manager.system_properties_dock in right_tabs
        assert manager.optimization_dock in right_tabs
        if arrangement == "two_panes":
            assert manager.viewer_dock in workspace.tabifiedDockWidgets(
                manager.lens_editor_dock
            )
            assert manager.terminal_dock in right_tabs
            assert manager.viewer_dock.x() < manager.analysis_dock.x()
        else:
            assert not workspace.tabifiedDockWidgets(manager.lens_editor_dock)
            assert not workspace.tabifiedDockWidgets(manager.viewer_dock)
            assert (
                manager.viewer_dock.y() > manager.lens_editor_dock.geometry().bottom()
            )
            assert manager.terminal_dock.y() > manager.analysis_dock.geometry().bottom()
            assert (
                manager.analysis_dock.x() > manager.lens_editor_dock.geometry().right()
            )


def test_invalid_preset_and_slots_do_not_mutate_layout(workspace):
    state, geometry = workspace.saveState(), workspace.geometry()
    with pytest.raises(ValueError, match="Unknown workspace arrangement"):
        workspace.panel_manager.arrange_workspace("unknown")
    for callback in (
        workspace.save_named_layout_slot,
        workspace.rename_layout_slot,
        workspace._load_layout_from_slot,
    ):
        with pytest.raises(ValueError, match="Unknown workspace slot"):
            callback(3)
    assert workspace.saveState() == state
    assert workspace.geometry() == geometry


@pytest.mark.parametrize("arrangement", ["two_panes", "four_panes"])
def test_equalize_balances_unconstrained_visible_split_geometry(
    qapp, workspace, arrangement
):
    manager = workspace.panel_manager
    manager.arrange_workspace(arrangement)
    qapp.processEvents()
    left = (
        manager.viewer_dock if arrangement == "two_panes" else manager.lens_editor_dock
    )
    right = manager.analysis_dock
    workspace.resizeDocks([left, right], [250, 850], Qt.Horizontal)
    if arrangement == "four_panes":
        workspace.resizeDocks(
            [manager.lens_editor_dock, manager.viewer_dock], [150, 550], Qt.Vertical
        )
    qapp.processEvents()
    state = topology(workspace, manager.all_docks)
    manager.equalize_workspace()
    qapp.processEvents()
    assert abs(left.width() - right.width()) <= 8
    if arrangement == "four_panes":
        assert (
            abs(manager.lens_editor_dock.height() - manager.viewer_dock.height()) <= 35
        )
    assert topology(workspace, manager.all_docks) == state


def test_equalize_preserves_constraints_and_ignores_hidden_floating_panels(
    qapp, workspace
):
    manager = workspace.panel_manager
    manager.lens_editor_dock.setMaximumWidth(260)
    manager.analysis_dock.setMinimumWidth(400)
    manager.optimization_dock.hide()
    manager.system_properties_dock.setFloating(True)
    terminal = manager.python_terminal
    terminal.console_dock.setFloating(True)
    qapp.processEvents()
    limits = contents(workspace)
    state = topology(workspace, manager.all_docks)
    floating_geometry = manager.system_properties_dock.geometry()
    inner_floating_geometry = terminal.console_dock.geometry()
    manager.equalize_workspace()
    qapp.processEvents()
    assert manager.lens_editor_dock.width() <= 260
    assert manager.analysis_dock.width() >= 400
    assert manager.lens_editor_dock.maximumWidth() == 260
    assert manager.analysis_dock.minimumWidth() == 400
    assert manager.system_properties_dock.geometry() == floating_geometry
    assert terminal.console_dock.geometry() == inner_floating_geometry
    assert topology(workspace, manager.all_docks) == state
    assert contents(workspace) == limits


def test_script_dock_identities_and_outer_inner_states_restore_repeatedly(
    qapp, workspace
):
    manager = workspace.panel_manager
    terminal = manager.python_terminal
    assert terminal.dock_area.objectName() == "ScriptTerminalDockArea"
    assert terminal.editor_dock.objectName() == "ScriptEditorDock"
    assert terminal.console_dock.objectName() == "ScriptConsoleDock"
    manager.system_properties_dock.hide()
    manager.viewer_dock.setFloating(True)
    terminal.dock_area.tabifyDockWidget(terminal.editor_dock, terminal.console_dock)
    terminal.editor_dock.hide()
    qapp.processEvents()
    before = contents(workspace)
    outer = topology(workspace, manager.all_docks)
    inner = topology(terminal.dock_area, [terminal.editor_dock, terminal.console_dock])
    workspace.action_manager.get_action("save_layout_1").trigger()
    assert isinstance(
        workspace.settings.value("Layouts/Config1TerminalState"), QByteArray
    )
    assert workspace.next_save_slot_index == 1
    for _ in range(3):
        workspace.arrange_two_panes()
        terminal.editor_dock.show()
        terminal.console_dock.setFloating(True)
        qapp.processEvents()
        assert workspace._load_layout_from_slot(1)
        qapp.processEvents()
        assert topology(workspace, manager.all_docks) == outer
        assert (
            topology(terminal.dock_area, [terminal.editor_dock, terminal.console_dock])
            == inner
        )
        assert contents(workspace) == before


def test_supported_old_outer_slot_restores_without_changing_nested_state(
    qapp, workspace
):
    manager = workspace.panel_manager
    terminal = manager.python_terminal
    manager.viewer_dock.hide()
    outer = workspace.saveState()
    workspace.settings.setValue("Layouts/Config2Geometry", workspace.saveGeometry())
    workspace.settings.setValue("Layouts/Config2State", outer)
    manager.viewer_dock.show()
    terminal.editor_dock.hide()
    terminal.console_dock.setFloating(True)
    qapp.processEvents()
    inner = topology(terminal.dock_area, [terminal.editor_dock, terminal.console_dock])
    assert workspace._load_layout_from_slot(2)
    qapp.processEvents()
    assert manager.viewer_dock.isHidden()
    assert (
        topology(terminal.dock_area, [terminal.editor_dock, terminal.console_dock])
        == inner
    )


@pytest.mark.parametrize("failure", ["geometry", "outer", "inner", "type", "missing"])
def test_failed_restore_keeps_working_outer_inner_state_and_geometry(
    qapp, workspace, failure
):
    workspace.save_named_layout_slot(1)
    workspace.arrange_two_panes()
    terminal = workspace.panel_manager.python_terminal
    terminal.console_dock.setFloating(True)
    terminal.editor_dock.hide()
    workspace.panel_manager.viewer_dock.setFloating(True)
    workspace.move(30, 40)
    qapp.processEvents()
    geometry = workspace.geometry()
    outer = topology(workspace, workspace.panel_manager.all_docks)
    inner = topology(terminal.dock_area, [terminal.editor_dock, terminal.console_dock])
    before = contents(workspace)
    keys = {
        "geometry": "Geometry",
        "outer": "State",
        "inner": "TerminalState",
        "type": "TerminalState",
        "missing": "State",
    }
    key = f"Layouts/Config1{keys[failure]}"
    if failure == "missing":
        workspace.settings.remove(key)
    else:
        workspace.settings.setValue(
            key, "wrong type" if failure == "type" else QByteArray(b"invalid")
        )
    assert not workspace._load_layout_from_slot(1)
    qapp.processEvents()
    assert workspace.geometry() == geometry
    assert topology(workspace, workspace.panel_manager.all_docks) == outer
    assert (
        topology(terminal.dock_area, [terminal.editor_dock, terminal.console_dock])
        == inner
    )
    assert contents(workspace) == before


def test_workspace_commands_are_reachable_from_main_menu_and_palette(workspace):
    # Populate the real application entry points, not only the workspace helper.
    for key in (
        "new", "open", "save", "save_as", "import_zemax", "import_codev",
        "export_zemax", "export_codev", "exit", "undo", "redo", "dark_theme",
        "light_theme", "about",
    ):
        workspace.action_manager._create_action(key, key)
    bar = QMenuBar(workspace)
    workspace._populate_main_menu_bar(bar)
    view_action = next(a for a in bar.actions() if a.text() == "&View")
    view = view_action.menu()
    workspace_action = next(a for a in view.actions() if a.text() == "&Workspace")
    menu = workspace_action.menu()
    assert workspace.action_manager.get_action("arrange_two_panes") in menu.actions()
    workspace.command_registry = CommandRegistry()
    workspace.panel_manager.analysis_panel = SimpleNamespace(_analysis_class_map={})
    workspace._register_palette_commands()
    workspace.command_registry.find("Arrange Two Panes").callback()
    manager = workspace.panel_manager
    assert manager.viewer_dock in workspace.tabifiedDockWidgets(manager.lens_editor_dock)
    workspace.command_registry.find("Save Workspace 1").callback()
    assert workspace.settings.contains("Layouts/Config1TerminalState")


def test_hiding_script_pane_keeps_outer_terminal_visible(qapp, workspace):
    manager = workspace.panel_manager
    manager.terminal_dock.show()
    manager.python_terminal.editor_dock.show()
    manager.python_terminal.editor_dock.toggleViewAction().trigger()
    qapp.processEvents()
    assert manager.python_terminal.editor_dock.isHidden()
    assert not manager.terminal_dock.isHidden()


def test_initial_layout_tolerates_unavailable_viewers(workspace):
    workspace.panel_manager.viewer_panel.viewer2D = None
    workspace.panel_manager.viewer_panel.viewer3D = None
    before = contents(workspace)
    workspace._setup_layout()
    assert contents(workspace) == before


@pytest.mark.parametrize("saved", [False, True])
def test_restore_without_notifications_keeps_return_value(workspace, saved):
    if saved:
        workspace.save_named_layout_slot(1)
    workspace.toast_manager = None
    assert workspace._load_layout_from_slot(1) is saved
    workspace.settings.setValue("Layouts/Config1Geometry", "invalid")
    workspace.settings.setValue("Layouts/Config1State", QByteArray())
    assert not workspace._load_layout_from_slot(1)


def test_named_explicit_and_alternating_slots_update_actions_and_persist(
    workspace, monkeypatch
):
    next_save = workspace.action_manager.get_action("save_layout")
    assert next_save.text() == "Save to Next Workspace (1)"
    monkeypatch.setattr(
        "optiland_gui.main_window.QInputDialog.getText",
        lambda *args, **kwargs: ("  Optical Review  ", True),
    )
    workspace.action_manager.get_action("rename_layout_1").trigger()
    assert workspace._workspace_name(1) == "Optical Review"
    assert workspace._workspace_slot_menus[1].title() == "&1: Optical Review"
    assert (
        workspace.action_manager.get_action("save_layout_1").text()
        == "Save to Optical Review"
    )
    assert not workspace.action_manager.get_action("load_layout_1").isEnabled()
    workspace.action_manager.get_action("save_layout_1").trigger()
    assert next_save.text() == "Save to Next Workspace (1)"
    assert workspace.action_manager.get_action("load_layout_1").isEnabled()
    assert (
        workspace.action_manager.get_action("load_layout_1").text()
        == "Restore Optical Review"
    )
    workspace.save_layout_slot()
    assert workspace.next_save_slot_index == 2
    assert next_save.text() == "Save to Next Workspace (2)"
    assert "workspace 2" in workspace.action_manager.get_action("save_layout").toolTip()
    workspace.save_layout_slot()
    assert workspace.next_save_slot_index == 1
    assert next_save.text() == "Save to Next Workspace (1)"
    assert workspace.action_manager.get_action("load_layout_2").isEnabled()
    workspace.settings.sync()
    stored = QSettings(workspace.settings.fileName(), QSettings.Format.IniFormat)
    assert stored.value("Layouts/Config1Name") == "Optical Review"
    assert isinstance(stored.value("Layouts/Config2TerminalState"), QByteArray)


@pytest.mark.parametrize("name", ["Ray & Fan", "&File"])
def test_workspace_names_are_literal_menu_text_with_no_new_mnemonics(
    workspace, monkeypatch, name
):
    monkeypatch.setattr(
        "optiland_gui.main_window.QInputDialog.getText", lambda *a, **k: (name, True)
    )
    workspace.rename_layout_slot(1)
    escaped = name.replace("&", "&&")
    assert workspace._workspace_name(1) == name
    assert workspace._workspace_slot_menus[1].title() == f"&1: {escaped}"
    action = workspace.action_manager.get_action("load_layout_1")
    assert action.text() == f"Restore {escaped}"
    assert action.toolTip() == f"Restore workspace 1: {name}"
    assert (
        workspace.action_manager.get_action("save_layout_1").text()
        == f"Save to {escaped}"
    )
    assert (
        workspace.action_manager.get_action("rename_layout_1").text()
        == f"Rename {escaped}..."
    )


def test_workspace_palette_tracks_literal_names_and_runs_existing_actions(
    workspace, monkeypatch
):
    workspace.command_registry = CommandRegistry()
    workspace._register_workspace_commands()
    monkeypatch.setattr(
        "optiland_gui.main_window.QInputDialog.getText",
        lambda *a, **k: ("Ray & Fan", True),
    )
    workspace.rename_layout_slot(1)
    command = workspace.command_registry.find("Save Workspace 1")
    assert command.description.endswith("Ray & Fan")
    assert command.keywords == ["Ray & Fan"]
    command.callback()
    assert workspace.settings.contains("Layouts/Config1TerminalState")
    assert len(workspace.command_registry.all_commands()) == 7


@pytest.mark.parametrize("mode", ["maximized", "fullscreen"])
@pytest.mark.parametrize("failure", ["outer", "inner"])
def test_failed_restore_preserves_window_state_and_original_normal_geometry(
    qapp, workspace, mode, failure
):
    workspace.save_named_layout_slot(1)
    normal_geometry = workspace.geometry()
    if mode == "maximized":
        workspace.showMaximized()
    else:
        workspace.showFullScreen()
    qapp.processEvents()
    window_state = workspace.windowState()
    key = "State" if failure == "outer" else "TerminalState"
    workspace.settings.setValue(f"Layouts/Config1{key}", QByteArray(b"invalid"))
    assert not workspace._load_layout_from_slot(1)
    qapp.processEvents()
    assert workspace.windowState() == window_state
    workspace.showNormal()
    qapp.processEvents()
    assert workspace.geometry() == normal_geometry


@pytest.mark.parametrize("answer", [("", True), ("ignored", False)])
def test_cancelled_or_blank_workspace_name_keeps_saved_name(
    workspace, monkeypatch, answer
):
    monkeypatch.setattr(
        "optiland_gui.main_window.QInputDialog.getText", lambda *a, **k: answer
    )
    workspace.rename_layout_slot(1)
    assert workspace._workspace_name(1) == "Workspace 1"


def test_workspace_menu_keyboard_and_nested_recovery_remain_accessible(qapp, workspace):
    manager = workspace.panel_manager
    menu = QMenu(workspace)
    workspace._populate_workspace_menu(menu)
    menu.popup(workspace.mapToGlobal(workspace.rect().center()))
    qapp.processEvents()
    QTest.keyClick(menu, Qt.Key_T)
    qapp.processEvents()
    assert manager.viewer_dock in workspace.tabifiedDockWidgets(
        manager.lens_editor_dock
    )
    manager.terminal_dock.hide()
    manager.python_terminal.editor_dock.hide()
    manager.python_terminal.editor_dock.toggleViewAction().trigger()
    qapp.processEvents()
    assert not manager.terminal_dock.isHidden()
    assert not manager.python_terminal.editor_dock.isHidden()
    assert manager.python_terminal.editor_dock.isVisible()


def test_initial_plot_is_requested_once_and_layout_recovery_never_recalculates(
    qapp, workspace
):
    panel = workspace.panel_manager.viewer_panel
    before = contents(workspace)
    workspace._setup_layout()
    assert panel.viewer2D.plot_optic.call_count == 1
    assert panel.viewer3D.render_optic.call_count == 1
    workspace.reset_windows_action()
    workspace.arrange_two_panes()
    workspace.arrange_four_panes()
    workspace.equalize_panes()
    workspace.save_named_layout_slot(1)
    assert workspace._load_layout_from_slot(1)
    qapp.processEvents()
    assert panel.viewer2D.plot_optic.call_count == 1
    assert panel.viewer3D.render_optic.call_count == 1
    assert contents(workspace) == before
