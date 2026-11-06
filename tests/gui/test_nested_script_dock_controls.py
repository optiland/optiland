"""Nested Script Editor and Console use the same native dock controls."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QEnterEvent, QFont, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDockWidget, QMainWindow, QTextEdit, QWidget

from optiland_gui.config import THEME_DARK_PATH, THEME_LIGHT_PATH
from optiland_gui.widgets.custom_dock_title_bar import CustomDockTitleBar
from optiland_gui.widgets.python_terminal import PythonTerminalWidget


class ConsoleContent(QTextEdit):
    def _style_sheet_changed(self):
        pass


class ScriptContent(PythonTerminalWidget):
    """Construct real editor/nested docks without starting an unrelated kernel."""

    def __init__(self, parent):
        QWidget.__init__(self, parent)
        self.current_theme = "dark"
        self.untitled_script_counter = 0
        self.kernel_manager = self.kernel_client = None
        self._define_themes()
        self._setup_ui()


@pytest.fixture
def controls(qapp, monkeypatch):
    previous_style = qapp.styleSheet()
    font = Path("C:/Windows/Fonts/segoeui.ttf")
    if not QFontDatabase.families() and font.exists():
        assert QFontDatabase.addApplicationFont(str(font)) >= 0
        qapp.setFont(QFont("Segoe UI", 9))
    monkeypatch.setattr(
        "optiland_gui.widgets.python_terminal.RichJupyterWidget", ConsoleContent
    )
    window = QMainWindow()
    terminal = ScriptContent(window)
    window.setCentralWidget(terminal)
    window.resize(1050, 540)
    window.show()
    window.activateWindow()
    qapp.processEvents()
    yield window, None, None, terminal
    window.close()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()
    qapp.setStyleSheet(previous_style)


def set_theme(qapp, controls, theme):
    path = THEME_DARK_PATH if theme == "dark" else THEME_LIGHT_PATH
    qapp.setStyleSheet(Path(path).read_text())
    controls[3].set_theme(theme)
    qapp.processEvents()


def hover(qapp, button, entered):
    """Route enter/leave events; offscreen has no physical cursor crossings."""
    event = (
        QEnterEvent(QPointF(button.rect().center()), QPointF(), QPointF())
        if entered
        else QEvent(QEvent.Type.Leave)
    )
    qapp.sendEvent(button, event)
    qapp.processEvents()


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_script_title_controls_keep_drafts_geometry_and_native_qt_state(
    qapp, controls, theme
):
    _, _, _, terminal = controls
    editor = terminal.script_tabs.widget(0)
    editor.setPlainText("optic.name = 'unsaved draft'")
    docks = [terminal.editor_dock, terminal.console_dock]
    set_theme(qapp, controls, theme)
    topology = [
        (terminal.dock_area.dockWidgetArea(dock), dock.isFloating(), dock.isHidden())
        for dock in docks
    ]
    for selected_theme in (theme, "dark" if theme == "light" else "light", theme):
        set_theme(qapp, controls, selected_theme)
    assert topology == [
        (terminal.dock_area.dockWidgetArea(dock), dock.isFloating(), dock.isHidden())
        for dock in docks
    ]
    assert editor.toPlainText() == "optic.name = 'unsaved draft'"
    assert terminal.script_tabs.widget(0) is editor
    for dock in docks:
        title = dock.titleBarWidget()
        assert isinstance(title, CustomDockTitleBar)
        assert title.title_label.text() == dock.windowTitle()
        for button in (title.undock_btn, title.close_btn):
            assert button.size().width() == 26
            assert button.size().height() == 26
            assert title.rect().contains(button.geometry())
            assert not button.icon().isNull()
            assert button.focusPolicy() == Qt.FocusPolicy.StrongFocus
        title.undock_btn.setFocus(Qt.FocusReason.TabFocusReason)
        QTest.keyClick(title.undock_btn, Qt.Key.Key_Space)
        qapp.processEvents()
        assert dock.isFloating()
        assert title.undock_btn.accessibleName() == "Dock panel"
        assert title.undock_btn.hasFocus()
        QTest.keyClick(title.undock_btn, Qt.Key.Key_Space)
        qapp.processEvents()
        assert not dock.isFloating()
        assert title.undock_btn.accessibleName() == "Float panel"
        title.close_btn.setFocus(Qt.FocusReason.TabFocusReason)
        QTest.keyClick(title.close_btn, Qt.Key.Key_Space)
        assert dock.isHidden()
        dock.toggleViewAction().trigger()
        assert not dock.isHidden()
    assert editor.toPlainText() == "optic.name = 'unsaved draft'"


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_script_header_hover_and_keyboard_focus_render_in_complete_window(
    qapp, controls, theme
):
    window, _, _, terminal = controls
    set_theme(qapp, controls, theme)
    title = terminal.editor_dock.titleBarWidget()
    button = title.undock_btn
    window.setFocus()
    QTest.mouseMove(window, QPoint(window.width() - 2, window.height() - 2))
    hover(qapp, button, False)
    qapp.processEvents()
    normal = button.grab().toImage()
    QTest.mouseMove(button, button.rect().center())
    hover(qapp, button, True)
    qapp.processEvents()
    hovered = button.grab().toImage()
    assert hovered != normal
    button.setFocus(Qt.FocusReason.TabFocusReason)
    qapp.processEvents()
    focused = button.grab().toImage()
    assert focused != hovered
    image = window.grab().toImage()
    point = title.mapTo(window, QPoint(title.width() // 2, 2))
    assert image.pixelColor(point).name() == (
        "#2e2e2e" if theme == "dark" else "#dee2e6"
    )


def test_terminal_theme_leaves_an_unrelated_native_child_title_untouched(
    qapp, controls
):
    terminal = controls[3]
    dock = QDockWidget("Native tool", terminal.dock_area)
    dock.setWidget(QTextEdit())
    terminal.dock_area.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
    for theme in ("light", "dark", "light"):
        set_theme(qapp, controls, theme)
    assert dock.titleBarWidget() is None
    assert dock.styleSheet() == ""
