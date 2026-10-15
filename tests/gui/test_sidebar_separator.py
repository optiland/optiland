"""The dock-only workspace has a single usable native sidebar separator."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QPointF, QRect, Qt
from PySide6.QtGui import QHoverEvent, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDockWidget, QMainWindow, QStyle, QTextEdit, QWidget

from optiland_gui.main_window import MainWindow
from optiland_gui.panel_manager import PanelManager


class SeparatorWindow(MainWindow):
    """Use production layout construction without optical workers or a kernel."""

    def __init__(self):
        QMainWindow.__init__(self)
        self.resize(1400, 850)
        self.setAnimated(False)
        manager = self.panel_manager = PanelManager(self, None)
        for attribute in (
            "sidebar",
            "viewer_dock",
            "lens_editor_dock",
            "system_properties_dock",
            "analysis_dock",
            "optimization_dock",
            "terminal_dock",
        ):
            dock = QDockWidget(attribute, self)
            dock.setObjectName(attribute)
            editor = QTextEdit(f"Retained {attribute} draft")
            editor.setMinimumSize(120, 90)
            dock.setWidget(editor)
            setattr(manager, attribute, dock)
            manager.all_docks.append(dock)
        manager.sidebar.setWidget(QWidget())
        manager.sidebar.setTitleBarWidget(QWidget())
        manager.sidebar.setMinimumWidth(60)
        manager.sidebar.setMaximumWidth(150)
        manager.viewer_panel = SimpleNamespace(
            viewer2D=SimpleNamespace(plot_optic=MagicMock()),
            viewer3D=SimpleNamespace(render_optic=MagicMock()),
        )

    showEvent = QMainWindow.showEvent
    resizeEvent = QMainWindow.resizeEvent
    closeEvent = QMainWindow.closeEvent
    mouseMoveEvent = QMainWindow.mouseMoveEvent
    mousePressEvent = QMainWindow.mousePressEvent
    mouseReleaseEvent = QMainWindow.mouseReleaseEvent


@pytest.fixture(params=["light", "dark"])
def separator_window(qapp, request):
    window = SeparatorWindow()
    styles = Path(__file__).parents[2] / "optiland_gui/resources/styles"
    window.setStyleSheet((styles / f"{request.param}_theme.qss").read_text())
    window._setup_layout()
    window.show()
    window.resizeDocks([window.panel_manager.sidebar], [100], Qt.Horizontal)
    qapp.processEvents()
    yield window
    window.close()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def assert_single_separator(window):
    sidebar = window.panel_manager.sidebar
    left = window.panel_manager.lens_editor_dock
    right = next(
        dock
        for dock in (
            window.panel_manager.analysis_dock,
            window.panel_manager.system_properties_dock,
            window.panel_manager.optimization_dock,
        )
        if window.rect().intersects(QRect(dock.mapTo(window, QPoint()), dock.size()))
    )
    separator = window.style().pixelMetric(
        QStyle.PM_DockWidgetSeparatorExtent, None, window
    )
    assert window.centralWidget() is None
    assert left.x() - sidebar.geometry().right() - 1 == separator
    assert (
        sidebar.width() + left.width() + right.width() >= window.width() - 2 * separator
    )
    assert window.width() - right.geometry().right() - 1 <= separator
    return separator


def test_default_layout_uses_full_width_and_retains_dock_contents(
    qapp, separator_window
):
    window = separator_window
    manager = window.panel_manager
    before = [
        (dock.widget(), dock.widget().toPlainText()) for dock in manager.all_docks[1:]
    ]
    for width in (1400, 1200, 1400):
        window.resize(width, 850)
        qapp.processEvents()
        assert_single_separator(window)
        assert [
            (dock.widget(), dock.widget().toPlainText())
            for dock in manager.all_docks[1:]
        ] == before
    assert manager.viewer_panel.viewer2D.plot_optic.call_count == 1
    assert manager.viewer_panel.viewer3D.render_optic.call_count == 1


def test_separator_routes_held_pointer_resize_and_native_layout_restore(
    qapp, separator_window
):
    window = separator_window
    sidebar = window.panel_manager.sidebar
    separator = assert_single_separator(window)
    point = QPoint(sidebar.geometry().right() + 1 + separator // 2, sidebar.y() + 100)
    QTest.mouseMove(window, point)
    QCoreApplication.sendEvent(
        window,
        QHoverEvent(
            QEvent.HoverMove,
            QPointF(point),
            QPointF(window.mapToGlobal(point)),
            QPointF(),
        ),
    )
    qapp.processEvents()
    assert window.cursor().shape() == Qt.SplitHCursor
    width = sidebar.width()
    QTest.mousePress(window, Qt.LeftButton, pos=point)
    destination = point - QPoint(20, 0)
    QCoreApplication.sendEvent(
        window,
        QMouseEvent(
            QEvent.MouseMove,
            QPointF(destination),
            QPointF(window.mapToGlobal(destination)),
            Qt.NoButton,
            Qt.LeftButton,
            Qt.NoModifier,
        ),
    )
    qapp.processEvents()
    QTest.mouseRelease(window, Qt.LeftButton, pos=destination)
    assert 60 <= sidebar.width() < width
    assert_single_separator(window)
    state = window.saveState()
    sidebar.setFloating(True)
    qapp.processEvents()
    assert sidebar.isFloating()
    assert window.restoreState(state)
    qapp.processEvents()
    assert not sidebar.isFloating()
    assert_single_separator(window)
