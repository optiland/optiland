"""Sidebar hover uses the entire button, including streams without Qt Enter."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QObject, QPoint, QPointF, Qt
from PySide6.QtGui import QHoverEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QMainWindow,
    QTextEdit,
    QToolButton,
    QWidget,
)

from optiland_gui.config import SIDEBAR_QSS_PATH, THEME_DARK_PATH, THEME_LIGHT_PATH
from optiland_gui.resources import resources_rc  # noqa: F401
from optiland_gui.widgets.sidebar import SidebarWidget


class PaintObserver(QObject):
    def __init__(self, button):
        super().__init__(button)
        self.count = 0
        button.installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Paint:
            self.count += 1
        return False


@pytest.fixture(params=[THEME_DARK_PATH, THEME_LIGHT_PATH], ids=["dark", "light"])
def sidebar_frame(qapp, request):
    window = QMainWindow()
    window.resize(900, 800)
    window.setCentralWidget(QTextEdit())
    dock = QDockWidget("Navigation", window)
    dock.setTitleBarWidget(QWidget())
    sidebar = SidebarWidget(window)
    dock.setWidget(sidebar)
    window.addDockWidget(Qt.LeftDockWidgetArea, dock)
    window.setStyleSheet(
        Path(request.param).read_text(encoding="utf-8")
        + "\n"
        + Path(SIDEBAR_QSS_PATH).read_text(encoding="utf-8")
    )
    sidebar.update_icons("dark" if request.param == THEME_DARK_PATH else "light")
    window.show()
    qapp.processEvents()
    yield window, dock, sidebar, request.param
    window.close()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def live_background(window, button):
    """Read the existing backing store; QWidget.grab would force a repaint."""
    origin = button.mapTo(window, QPoint())
    image = (
        window.screen()
        .grabWindow(
            window.winId(), origin.x(), origin.y(), button.width(), button.height()
        )
        .toImage()
    )
    assert not image.isNull()
    ratio = image.devicePixelRatio()
    return image.pixelColor(round((button.width() - 10) * ratio), round(10 * ratio))


def hover_paths(button):
    x, y = button.width() // 2, button.height() // 2
    return {
        "top": [QPoint(x, v) for v in range(-5, y + 1)],
        "bottom": [QPoint(x, v) for v in range(button.height() + 5, y - 1, -1)],
        "left": [QPoint(v, y) for v in range(-5, x + 1)],
        "right": [QPoint(v, y) for v in range(button.width() + 5, x - 1, -1)],
    }


@pytest.mark.parametrize("width", [60, 150])
@pytest.mark.parametrize("selected", [False, True])
def test_icon_paths_repaint_without_forced_grabs(qapp, sidebar_frame, width, selected):
    window, dock, sidebar, theme = sidebar_frame
    dock.setFixedWidth(width)
    qapp.processEvents()
    expected = "#5a5a5a" if theme == THEME_DARK_PATH else "#dee2e6"
    outside = QPoint(window.width() - 20, window.height() // 2)
    navigation, wip = [], []
    sidebar.menuSelected.connect(navigation.append)
    sidebar.showWipMessage.connect(wip.append)

    def move_outside():
        # Use a distinct first coordinate across successive test windows too.
        QTest.mouseMove(window.windowHandle(), outside + QPoint(2, 0))
        QTest.mouseMove(window.windowHandle(), outside)
        QTest.qWait(20)

    for item in sidebar._buttons_list:
        if selected and item["name"] in sidebar._wip_buttons:
            continue
        button = item["widget"]
        # Select a different implemented button for an unselected target.
        alternate = sidebar._buttons_list[-1 if item["name"] == "design" else 1]
        alternate["widget"].setChecked(True)
        if selected:
            button.setChecked(True)
        qapp.processEvents()
        geometry = button.geometry()
        observer = PaintObserver(button)
        for direction, points in hover_paths(button).items():
            move_outside()
            baseline = live_background(window, button)
            observer.count = 0
            origin = button.mapTo(window, QPoint())
            for point in points:
                QTest.mouseMove(window.windowHandle(), origin + point)
                qapp.processEvents()
            QTest.qWait(20)
            assert observer.count > 0, direction
            assert live_background(window, button).name() == expected, direction
            assert button.geometry() == geometry and button.isChecked() == selected
            move_outside()
            assert live_background(window, button) == baseline, direction
        assert navigation == [] and wip == []


@pytest.mark.parametrize("width", [60, 150])
@pytest.mark.parametrize("selected", [False, True])
def test_hover_move_without_enter_covers_icon_and_clears_on_exit_and_hide(
    qapp, sidebar_frame, width, selected
):
    window, dock, sidebar, theme = sidebar_frame
    dock.setFixedWidth(width)
    qapp.processEvents()
    button = sidebar._buttons_list[-1]["widget"]
    button.setChecked(selected)
    QTest.mouseMove(
        window.windowHandle(), QPoint(window.width() - 20, window.height() // 2)
    )
    QTest.qWait(20)
    baseline = live_background(window, button)
    expected = "#5a5a5a" if theme == THEME_DARK_PATH else "#dee2e6"
    navigation = []
    sidebar.menuSelected.connect(navigation.append)
    geometry = button.geometry()

    def send_hover(kind, point):
        # Reproduce the Windows trace: valid local HoverMove, absent Enter and
        # absent native underMouse state. Do not repair Qt's flag in the test.
        button.setAttribute(Qt.WidgetAttribute.WA_UnderMouse, False)
        QApplication.sendEvent(
            button,
            QHoverEvent(
                kind,
                QPointF(point),
                QPointF(button.mapToGlobal(point)),
                QPointF(-5, -5),
                Qt.NoModifier,
            ),
        )
        qapp.processEvents()
        assert not button.underMouse()

    observer = PaintObserver(button)
    for direction, points in hover_paths(button).items():
        observer.count = 0
        for point in points:
            send_hover(QEvent.Type.HoverMove, point)
        QTest.qWait(20)
        assert observer.count > 0, direction
        assert live_background(window, button).name() == expected, direction
        assert button.geometry() == geometry and button.isChecked() == selected
        send_hover(QEvent.Type.HoverLeave, QPoint(-5, -5))
        QTest.qWait(20)
        assert live_background(window, button) == baseline
    send_hover(QEvent.Type.HoverEnter, button.rect().center())
    QTest.qWait(20)
    assert live_background(window, button).name() == expected
    # A hide must clear feedback even without HoverLeave.
    button.hide()
    button.show()
    QTest.qWait(20)
    assert live_background(window, button) == baseline
    assert navigation == []
    # Paint-only hover tracking must preserve Qt keyboard and mouse activation.
    button.setFocus()
    QTest.keyClick(button, Qt.Key_Space)
    assert navigation == ["scripts"] and button.isChecked()
    QTest.mouseClick(sidebar._buttons_list[1]["widget"], Qt.LeftButton)
    assert navigation == ["scripts", "design"]
    assert sidebar._buttons_list[1]["widget"].isChecked()


@pytest.mark.parametrize("width", [60, 150])
def test_sidebar_hover_crossings_work_for_selected_and_unselected_buttons(
    qapp, sidebar_frame, width
):
    window, dock, sidebar, theme_path = sidebar_frame
    dock.setFixedWidth(width)
    qapp.processEvents()
    buttons = sidebar.findChildren(QToolButton)
    navigation, wip = [], []
    sidebar.menuSelected.connect(navigation.append)
    sidebar.showWipMessage.connect(wip.append)
    hover_color = "#5a5a5a" if theme_path == THEME_DARK_PATH else "#dee2e6"
    selected_border = "#007acc" if theme_path == THEME_DARK_PATH else "#6c757d"
    outside = QPoint(window.width() - 20, window.height() // 2)

    def move_to(button):
        QTest.mouseMove(
            window.windowHandle(), button.mapTo(window, button.rect().center())
        )
        qapp.processEvents()

    for target in buttons:
        name = target.objectName().removeprefix("sidebar-btn-")
        states = (False, True) if name not in sidebar._wip_buttons else (False,)
        for selected in states:
            # Select a different implemented panel before the unselected case.
            alternate = buttons[1] if name != "design" else buttons[-1]
            QTest.mouseClick(alternate, Qt.LeftButton)
            if selected:
                QTest.mouseClick(target, Qt.LeftButton)
            assert target.isChecked() == selected
            QTest.mouseMove(window.windowHandle(), outside)
            qapp.processEvents()
            normal = target.grab().toImage()
            geometry = target.geometry()
            # Start at the adjacent button, then cross its edge and the gap.
            index = buttons.index(target)
            previous = buttons[index - 1] if index else buttons[1]
            move_to(previous)
            assert previous.underMouse() and not target.underMouse()
            counts = len(navigation), len(wip)
            move_to(target)
            assert target.underMouse() and not previous.underMouse()
            hovered = target.grab().toImage()
            assert hovered != normal
            assert hovered.pixelColor(width // 2, 8).name() == hover_color
            assert target.geometry() == geometry and target.isChecked() == selected
            if selected:
                assert (
                    hovered.pixelColor(1, target.height() // 2).name()
                    == selected_border
                )
            move_to(previous)
            assert not target.underMouse()
            assert target.grab().toImage() == normal
            assert (len(navigation), len(wip)) == counts
    # WIP clicks still preserve the last selected panel and emit one notice.
    QTest.mouseClick(buttons[-1], Qt.LeftButton)
    QTest.mouseClick(buttons[-2], Qt.LeftButton)
    assert len(wip) == 1 and buttons[-1].isChecked()
    assert navigation[-1] == "scripts"


def test_disabled_sidebar_button_keeps_disabled_feedback_and_cannot_activate(
    qapp, sidebar_frame
):
    window, _, sidebar, _ = sidebar_frame
    button = sidebar._buttons_list[-1]["widget"]
    navigation = []
    sidebar.menuSelected.connect(navigation.append)
    outside = QPoint(window.width() - 20, window.height() // 2)
    QTest.mouseMove(window.windowHandle(), outside)
    button.setEnabled(False)
    QTest.qWait(20)
    normal = live_background(window, button)
    QTest.mouseMove(window.windowHandle(), button.mapTo(window, button.rect().center()))
    QTest.qWait(20)
    assert live_background(window, button) == normal
    QTest.mouseClick(button, Qt.LeftButton)
    QTest.keyClick(button, Qt.Key_Space)
    assert navigation == [] and not button.isChecked()
    QTest.mouseMove(window.windowHandle(), outside)
    button.setEnabled(True)
    QTest.mouseMove(window.windowHandle(), button.mapTo(window, button.rect().center()))
    QTest.qWait(20)
    assert live_background(window, button) != normal
    QTest.mouseClick(button, Qt.LeftButton)
    assert navigation == ["scripts"] and button.isChecked()
