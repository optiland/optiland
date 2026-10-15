"""Window controls describe their next action across state and theme changes."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMenuBar

from optiland_gui.main_window import MainWindow
from optiland_gui.widgets.custom_title_bar import CustomTitleBar
from optiland_gui.widgets.frameless_window import FramelessWindow


class TitleBarWindow(MainWindow):
    """Exercise the real window state handlers without workers or a kernel."""

    def __init__(self):
        FramelessWindow.__init__(self)
        self.toast_manager = None
        self.command_palette = None
        self.custom_title_bar_widget = CustomTitleBar(QMenuBar(), self)
        self.custom_title_bar_widget.maximize_restore_requested.connect(
            self._handle_maximize_restore
        )
        self.setCentralWidget(self.custom_title_bar_widget)
        self.resize(800, 500)

    def closeEvent(self, event):
        event.accept()

    def showEvent(self, event):
        FramelessWindow.showEvent(self, event)


@pytest.fixture
def title_window(qapp):
    window = TitleBarWindow()
    window.show()
    qapp.processEvents()
    yield window
    window.close()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()


def assert_window_action(window, theme):
    maximized = window.isMaximized()
    button = window.custom_title_bar_widget.maximize_button
    assert button.isChecked() is maximized
    assert button.toolTip() == ("Restore" if maximized else "Maximize")
    expected = QIcon(
        f":/icons/{theme}/{'maximize_restore' if maximized else 'maximize'}.svg"
    )
    assert not expected.isNull()
    assert not button.icon().isNull()
    size = QSize(24, 24)
    image = button.icon().pixmap(size).toImage()
    assert image == expected.pixmap(size).toImage()
    if not maximized:
        # A hollow, single rectangle, rather than the offset Restore windows.
        assert image.pixelColor(12, 4).alpha() > 0
        assert image.pixelColor(4, 12).alpha() > 0
        assert image.pixelColor(12, 12).alpha() == 0
        assert image.pixelColor(2, 12).alpha() == 0


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_window_icons_follow_button_and_external_state_changes(
    qapp, title_window, theme
):
    title = title_window.custom_title_bar_widget
    assert_window_action(title_window, "dark")
    title.update_theme_icons(theme)
    assert_window_action(title_window, theme)
    title.maximize_button.click()
    qapp.processEvents()
    assert title_window.isMaximized()
    assert_window_action(title_window, theme)
    title_window.showNormal()
    qapp.processEvents()
    assert not title_window.isMaximized()
    assert_window_action(title_window, theme)
    title_window.showMaximized()
    qapp.processEvents()
    assert_window_action(title_window, theme)
    title.maximize_button.click()
    qapp.processEvents()
    assert not title_window.isMaximized()
    assert_window_action(title_window, theme)


@pytest.mark.parametrize("maximized", [False, True])
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_theme_switch_preserves_window_action(qapp, title_window, maximized, theme):
    if maximized:
        title_window.showMaximized()
        qapp.processEvents()
    title = title_window.custom_title_bar_widget
    for selected in (theme, "light" if theme == "dark" else "dark", theme):
        title.update_theme_icons(selected)
        assert title_window.isMaximized() is maximized
        assert_window_action(title_window, selected)
