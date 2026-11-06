"""Dock controls and native separators expose practical, accessible hit targets."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QAccessible, QHoverEvent, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QLineEdit,
    QMainWindow,
    QStyleFactory,
    QToolButton,
    QWidget,
)

from optiland_gui.widgets.custom_dock_widget import CustomDockWidget


@pytest.fixture(params=["fusion", "windows11"])
def dock_style(qapp, request):
    style_names = QStyleFactory.keys()
    names = {name.lower(): name for name in style_names}
    if request.param not in names:
        pytest.skip(f"Qt style {request.param} is unavailable")
    previous = qapp.style().objectName()
    qapp.setStyle(names[request.param])
    yield qapp
    qapp.setStyle(previous)


@pytest.fixture(params=["light", "dark"])
def panel_window(dock_style, request):
    theme = (
        Path(__file__).parents[2]
        / "optiland_gui/resources/styles"
        / f"{request.param}_theme.qss"
    )
    window = QMainWindow()
    window.setStyleSheet(theme.read_text(encoding="utf-8"))
    window.setAnimated(False)
    window.setDockNestingEnabled(True)
    window.setCentralWidget(QWidget())
    window.resize(800, 520)
    dock = CustomDockWidget("Analysis panel", window)
    dock.setObjectName("AnalysisPanel")
    content = QWidget()
    content.setMinimumSize(120, 80)
    dock.setWidget(content)
    window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
    window.show()
    window.activateWindow()
    dock_style.processEvents()
    yield window, dock, request.param
    window.close()
    window.deleteLater()
    dock_style.processEvents()


def test_header_controls_are_large_and_have_accessible_actions(qapp, panel_window):
    _, dock, _ = panel_window
    title = dock.titleBarWidget()
    assert len(title.findChildren(QToolButton)) == 2
    assert title.height() <= 36
    assert title.toolTip() == "Analysis panel"
    for button, name in (
        (title.undock_btn, "Float panel"),
        (title.close_btn, "Close panel"),
    ):
        assert button.width() >= 24
        assert button.height() >= 24
        assert not button.icon().isNull()
        assert button.toolTip() == name
        assert button.accessibleName() == name
        interface = QAccessible.queryAccessibleInterface(button)
        assert interface.text(QAccessible.Text.Name) == name
        assert interface.text(QAccessible.Text.Description)
        assert button.focusPolicy() == Qt.FocusPolicy.StrongFocus


def test_button_corners_and_keyboard_float_redock_and_close(qapp, panel_window):
    _, dock, _ = panel_window
    title = dock.titleBarWidget()
    button = title.undock_btn
    for point in (
        QPoint(1, 1),
        button.rect().bottomRight() - QPoint(1, 1),
    ):
        assert QApplication.widgetAt(button.mapToGlobal(point)) is button
        QTest.mouseClick(button, Qt.MouseButton.LeftButton, pos=point)
        qapp.processEvents()
        assert dock.isFloating()
        assert button.toolTip() == "Dock panel"
        assert button.accessibleName() == "Dock panel"
        QTest.mouseClick(button, Qt.MouseButton.LeftButton, pos=point)
        qapp.processEvents()
        assert not dock.isFloating()
        assert button.toolTip() == "Float panel"

    button.setFocus()
    qapp.processEvents()
    assert button.hasFocus()
    QTest.keyClick(button, Qt.Key.Key_Space)
    qapp.processEvents()
    assert dock.isFloating()
    assert button.hasFocus()
    QTest.keyClick(button, Qt.Key.Key_Space)
    qapp.processEvents()
    assert not dock.isFloating()
    assert button.hasFocus()
    title.close_btn.setFocus()
    qapp.processEvents()
    assert title.close_btn.hasFocus()
    QTest.keyClick(title.close_btn, Qt.Key.Key_Space)
    qapp.processEvents()
    assert not dock.isVisible()


def test_close_button_corner_hides_only_its_panel(qapp, panel_window):
    window, dock, _ = panel_window
    button = dock.titleBarWidget().close_btn
    QTest.mouseClick(button, Qt.MouseButton.LeftButton, pos=QPoint(1, 1))
    qapp.processEvents()
    assert not dock.isVisible()
    assert window.isVisible()


def test_programmatic_redock_preserves_editor_focus(qapp, panel_window):
    window, dock, _ = panel_window
    dock.setFloating(True)
    qapp.processEvents()
    button = dock.titleBarWidget().undock_btn

    # Redocking in the background must not steal focus from the active editor.
    main_editor = QLineEdit(window.centralWidget())
    main_editor.show()
    window.activateWindow()
    main_editor.setFocus()
    qapp.processEvents()
    assert qapp.activeWindow() is window
    assert qapp.focusWidget() is main_editor
    button.click()
    qapp.processEvents()
    assert not dock.isFloating()
    assert qapp.activeWindow() is window
    assert qapp.focusWidget() is main_editor
    assert not button.hasFocus()
    QTest.keyClicks(qapp.focusWidget(), "main editor")
    assert main_editor.text() == "main editor"


@pytest.mark.parametrize("activation", ["mouse", "keyboard"])
def test_explicit_redock_selects_returned_panel_tab(qapp, panel_window, activation):
    window, dock, _ = panel_window
    peer = CustomDockWidget("Notes", window)
    peer.setObjectName("NotesPanel")
    editor = QLineEdit("Keep this draft", peer)
    peer.setWidget(editor)
    window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, peer)
    window.tabifyDockWidget(dock, peer)
    dock.raise_()
    qapp.processEvents()
    assert not dock.visibleRegion().isEmpty()
    button = dock.titleBarWidget().undock_btn

    def activate_button():
        if activation == "mouse":
            QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        else:
            button.setFocus()
            QTest.keyClick(button, Qt.Key.Key_Space)
        qapp.processEvents()

    activate_button()
    assert dock.isFloating()
    assert not peer.visibleRegion().isEmpty()
    activate_button()
    assert not dock.isFloating()
    assert window.tabifiedDockWidgets(dock) == [peer]
    assert not dock.visibleRegion().isEmpty()
    assert peer.visibleRegion().isEmpty()
    assert editor.text() == "Keep this draft"
    if activation == "keyboard":
        assert button.hasFocus()


def test_controls_follow_feature_restrictions_and_title(qapp, panel_window):
    _, dock, _ = panel_window
    title = dock.titleBarWidget()
    dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
    qapp.processEvents()
    assert not title.undock_btn.isVisible()
    assert not title.close_btn.isVisible()
    dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable)
    qapp.processEvents()
    assert title.close_btn.isVisible()
    assert not title.undock_btn.isVisible()
    dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetFloatable)
    qapp.processEvents()
    assert title.undock_btn.isVisible()
    assert not title.close_btn.isVisible()
    dock.setWindowTitle("Changed panel title")
    assert title.title_label.text() == "Changed panel title"
    assert title.toolTip() == "Changed panel title"
    dock.setFloating(True)
    qapp.processEvents()
    assert title.undock_btn.accessibleName() == "Dock panel"


def test_long_title_preserves_button_hit_areas(qapp, panel_window):
    _, dock, _ = panel_window
    dock.setFloating(True)
    dock.setWindowTitle("Analysis panel with a very long description " * 6)
    dock.resize(180, 150)
    qapp.processEvents()
    title = dock.titleBarWidget()
    assert title.title_label.width() > 0
    assert title.rect().contains(title.close_btn.geometry())
    assert title.rect().contains(title.undock_btn.geometry())
    assert not title.title_label.geometry().intersects(title.undock_btn.geometry())
    assert title.toolTip() == dock.windowTitle()


def test_header_uses_a_theme_matched_background(qapp, panel_window):
    _, dock, theme = panel_window
    title = dock.titleBarWidget()
    image = title.grab().toImage()
    background = image.pixelColor(2, title.height() // 2)
    assert (background.lightness() > 180) is (theme == "light")


def test_action_glyphs_keep_contrast_after_theme_switch(qapp, panel_window):
    window, dock, theme = panel_window
    for current in (theme, "dark" if theme == "light" else "light"):
        theme_path = (
            Path(__file__).parents[2]
            / "optiland_gui/resources/styles"
            / f"{current}_theme.qss"
        )
        window.setStyleSheet(theme_path.read_text(encoding="utf-8"))
        qapp.processEvents()
        for button in (
            dock.titleBarWidget().undock_btn,
            dock.titleBarWidget().close_btn,
        ):
            image = button.icon().pixmap(16, 16).toImage()
            values = [
                image.pixelColor(x, y).lightness()
                for x in range(image.width())
                for y in range(image.height())
                if image.pixelColor(x, y).alpha() > 127
            ]
            assert values, "The action needs a visible glyph"
            lightness = sum(values) / len(values)
            assert (lightness > 180) is (current == "dark")


@pytest.mark.parametrize(
    "orientation", [Qt.Orientation.Horizontal, Qt.Orientation.Vertical]
)
def test_native_separator_has_practical_width_and_resizes_neighbors(
    qapp, panel_window, orientation
):
    window, first, _ = panel_window
    second = CustomDockWidget("Second panel", window)
    second.setObjectName("SecondPanel")
    content = QWidget()
    content.setMinimumSize(120, 80)
    second.setWidget(content)
    window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, second)
    window.splitDockWidget(first, second, orientation)
    qapp.processEvents()
    # Leave both neighbors room to shrink; fresh docks start at their minima.
    window.resizeDocks([first, second], [240, 240], orientation)
    qapp.processEvents()
    if orientation == Qt.Orientation.Horizontal:
        gap = second.x() - first.geometry().right() - 1
        point = QPoint(first.geometry().right() + 1 + gap // 2, first.y() + 100)
        delta = QPoint(30, 0)
        before = first.width(), second.width()
    else:
        gap = second.y() - first.geometry().bottom() - 1
        point = QPoint(
            first.x() + first.width() // 2, first.geometry().bottom() + 1 + gap // 2
        )
        delta = QPoint(0, 30)
        before = first.height(), second.height()
    assert 6 <= gap <= 8
    QTest.mouseMove(window, point)
    # Explicit hover delivery keeps offscreen event routing deterministic.
    QApplication.sendEvent(
        window,
        QHoverEvent(
            QEvent.Type.HoverMove,
            QPointF(point),
            QPointF(window.mapToGlobal(point)),
            QPointF(),
        ),
    )
    qapp.processEvents()
    expected_cursor = (
        Qt.CursorShape.SplitHCursor
        if orientation == Qt.Orientation.Horizontal
        else Qt.CursorShape.SplitVCursor
    )
    assert window.cursor().shape() == expected_cursor
    QTest.mousePress(window, Qt.MouseButton.LeftButton, pos=point)
    # QTest.mouseMove reports no buttons on the offscreen Windows plugin.
    # Send the same held-button event a real native separator gesture receives.
    QApplication.sendEvent(
        window,
        QMouseEvent(
            QEvent.Type.MouseMove,
            QPointF(point + delta),
            QPointF(window.mapToGlobal(point + delta)),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        ),
    )
    qapp.processEvents()
    QTest.mouseRelease(window, Qt.MouseButton.LeftButton, pos=point + delta)
    qapp.processEvents()
    after = (
        (first.width(), second.width())
        if orientation == Qt.Orientation.Horizontal
        else (first.height(), second.height())
    )
    assert after[0] > before[0]
    assert after[1] < before[1]
    assert sum(after) == sum(before)
    assert not first.isFloating()
    assert not second.isFloating()
