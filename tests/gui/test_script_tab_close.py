"""Native script-tab close targets remain centered and select the correct tab."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QPointF, Qt
from PySide6.QtGui import QEnterEvent, QFont, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QMainWindow,
    QMessageBox,
    QStyle,
    QStyleOptionTab,
    QTabBar,
    QTextEdit,
    QWidget,
)

from optiland_gui.config import THEME_DARK_PATH, THEME_LIGHT_PATH
from optiland_gui.resources import resources_rc  # noqa: F401
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
@pytest.mark.parametrize("direction", [Qt.LeftToRight, Qt.RightToLeft])
def test_script_tab_close_is_padded_and_closes_the_reordered_script(
    qapp, controls, monkeypatch, theme, direction
):
    window, _, _, terminal = controls
    tabs = terminal.script_tabs
    tabs.setLayoutDirection(direction)
    draft = tabs.widget(0)
    draft.setPlainText("optic.name = 'retain this draft'")
    other = terminal._create_new_tab("print('discard this script')")
    tabs.setTabText(1, "secondary_with_a_long_title.py")
    warning = []
    replies = iter((QMessageBox.Discard, QMessageBox.Cancel))

    def answer_close(*args):
        warning.append(args[2])
        return next(replies)

    monkeypatch.setattr(QMessageBox, "warning", answer_close)
    bar = tabs.tabBar()
    close_side = QTabBar.ButtonPosition(
        bar.style().styleHint(QStyle.SH_TabBar_CloseButtonPosition, None, bar)
    )
    buttons = [bar.tabButton(index, close_side) for index in range(tabs.count())]
    for selected_theme in (theme, "dark" if theme == "light" else "light", theme):
        set_theme(qapp, controls, selected_theme)
    for selected in (0, 1):
        tabs.setCurrentIndex(selected)
        qapp.processEvents()
        for index, button in enumerate(buttons):
            tab = bar.tabRect(index)
            assert tab.contains(button.geometry())
            assert abs(button.geometry().center().y() - tab.center().y()) <= 1
            option = QStyleOptionTab()
            bar.initStyleOption(option, index)
            text = bar.style().subElementRect(QStyle.SE_TabBarTabText, option, bar)
            assert not text.intersects(button.geometry())
            # Long titles scroll inactive close controls out of the viewport.
            if index != selected:
                continue
            assert button.isVisible()
            hover(qapp, button, False)
            image = button.grab().toImage()
            foreground = "#e0e0e0" if theme == "dark" else "#343a40"
            pixels = [
                (x, y)
                for y in range(image.height())
                for x in range(image.width())
                if image.pixelColor(x, y).name() == foreground
            ]
            assert pixels, Counter(
                image.pixelColor(x, y).name()
                for y in range(image.height())
                for x in range(image.width())
            ).most_common(6)
            scale = image.devicePixelRatio()
            left, right = min(x for x, _ in pixels), max(x for x, _ in pixels)
            assert left / scale >= 7
            assert (image.width() - 1 - right) / scale >= 7
            assert abs((left + right) / 2 - (image.width() - 1) / 2) <= scale
            geometry = button.geometry()
            hover(qapp, button, True)
            hovered = button.grab().toImage()
            assert hovered != image
            assert button.geometry() == geometry
    bar.moveTab(1, 0)
    tabs.setCurrentIndex(0)
    qapp.processEvents()
    close = bar.tabButton(0, close_side)
    assert close is buttons[1]
    assert qapp.widgetAt(close.mapToGlobal(close.rect().center())) is close
    QTest.mouseClick(close, Qt.LeftButton)
    assert tabs.indexOf(other) == -1
    assert tabs.count() == 1
    assert tabs.widget(0) is draft
    assert draft.toPlainText() == "optic.name = 'retain this draft'"
    assert len(warning) == 1
    assert "secondary_with_a_long_title.py" in warning[0]
    close = bar.tabButton(0, close_side)
    QTest.mouseClick(close, Qt.LeftButton)
    assert len(warning) == 2
    assert "Untitled-1" in warning[1]
    assert tabs.widget(0) is draft
    assert draft.toPlainText() == "optic.name = 'retain this draft'"
