"""Dock and document tabs consistently sit above the content they select."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QDockWidget,
    QMainWindow,
    QTabBar,
    QTabWidget,
    QTextEdit,
    QWidget,
)

from optiland_gui.config import DEFAULT_TAB_POSITION
from optiland_gui.main_window import MainWindow
from optiland_gui.optiland_connector import OptilandConnector
from optiland_gui.optimization_panel import OptimizationPanel
from optiland_gui.widgets.python_terminal import PythonTerminalWidget


class PlacementConsole(QTextEdit):
    def _style_sheet_changed(self):
        pass


class PlacementTerminal(PythonTerminalWidget):
    """Build real terminal panes and tabs without starting an embedded kernel."""

    def __init__(self, parent):
        QWidget.__init__(self, parent)
        self.current_theme = "dark"
        self.untitled_script_counter = 0
        self.kernel_manager = self.kernel_client = None
        self._define_themes()
        self._setup_ui()


class PlacementWindow(MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self._configure_window()
        self.resize(1100, 750)

    showEvent = QMainWindow.showEvent
    resizeEvent = QMainWindow.resizeEvent
    closeEvent = QMainWindow.closeEvent


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_outer_nested_and_script_tabs_stay_above_retained_contents(
    qapp, monkeypatch, theme
):
    monkeypatch.setattr(
        "optiland_gui.widgets.python_terminal.RichJupyterWidget", PlacementConsole
    )
    window = PlacementWindow()
    terminal = PlacementTerminal(window)
    window.setCentralWidget(terminal)
    outer = []
    for name in ("Analysis", "System Properties"):
        dock = QDockWidget(name, window)
        dock.setObjectName(name)
        dock.setWidget(QTextEdit(f"Retained {name}"))
        window.addDockWidget(Qt.RightDockWidgetArea, dock)
        outer.append(dock)
    window.tabifyDockWidget(*outer)
    outer[0].raise_()
    terminal.dock_area.tabifyDockWidget(terminal.editor_dock, terminal.console_dock)
    terminal.editor_dock.raise_()
    terminal.script_tabs.widget(0).setPlainText("optic.name = 'keep this draft'")
    styles = Path(__file__).parents[2] / "optiland_gui/resources/styles"
    window.setStyleSheet((styles / f"{theme}_theme.qss").read_text())
    window.show()
    QTest.qWait(20)
    assert QTabWidget.TabPosition.North == DEFAULT_TAB_POSITION
    for dock_window, labels, active in (
        (window, {"Analysis", "System Properties"}, outer[0]),
        (terminal.dock_area, {"Script Editor", "Console"}, terminal.editor_dock),
    ):
        for area in (
            Qt.LeftDockWidgetArea,
            Qt.RightDockWidgetArea,
            Qt.TopDockWidgetArea,
            Qt.BottomDockWidgetArea,
        ):
            assert dock_window.tabPosition(area) == QTabWidget.TabPosition.North
        bars = [
            bar
            for bar in dock_window.findChildren(QTabBar)
            if {bar.tabText(index) for index in range(bar.count())} == labels
        ]
        assert len(bars) == 1
        bar = bars[0]
        assert bar.isVisible()
        assert bar.shape() == QTabBar.Shape.RoundedNorth
        assert (
            bar.mapTo(dock_window, QPoint(0, bar.height())).y()
            <= active.mapTo(dock_window, QPoint()).y()
        )
    assert terminal.script_tabs.tabPosition() == QTabWidget.TabPosition.North
    assert (
        terminal.script_tabs.widget(0).toPlainText() == "optic.name = 'keep this draft'"
    )
    assert outer[0].widget().toPlainText() == "Retained Analysis"
    window.close()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_optimization_tabs_use_shared_presentation_default(qapp):
    connector = OptilandConnector()
    panel = OptimizationPanel(connector)
    assert panel._tabs.tabPosition() == DEFAULT_TAB_POSITION
    assert [panel._tabs.tabText(index) for index in range(panel._tabs.count())] == [
        "Variables",
        "Operands",
        "Optimizer",
    ]
    panel.close()
    panel.deleteLater()
    connector.calculation_jobs.shutdown()
    connector.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
