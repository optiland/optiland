"""Custom title surfaces paint correctly inside the application's toolbar."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QMainWindow, QMenuBar, QToolBar, QWidget

from optiland_gui.resources import resources_rc  # noqa: F401
from optiland_gui.widgets.custom_title_bar import CustomTitleBar


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_title_background_is_painted_in_the_composed_toolbar_window(qapp, theme):
    window = QMainWindow()
    window.resize(1400, 400)
    window.setCentralWidget(QWidget())
    title = CustomTitleBar(QMenuBar(), window)
    title.project_label.clear()
    toolbar = QToolBar(window)
    toolbar.setObjectName("CustomTitleBarToolbar")
    toolbar.addWidget(title)
    window.addToolBar(toolbar)
    styles = Path(__file__).parents[2] / "optiland_gui/resources/styles"
    window.show()
    for selected in (theme, "light" if theme == "dark" else "dark", theme):
        window.setStyleSheet((styles / f"{selected}_theme.qss").read_text())
        title.update_theme_icons(selected)
        qapp.processEvents()
        point = title.mapTo(window, title.project_label.geometry().center())
        expected = "#2e2e2e" if selected == "dark" else "#e9ecef"
        image = window.grab().toImage()
        ratio = image.devicePixelRatio()
        assert (
            image.pixelColor(round(point.x() * ratio), round(point.y() * ratio)).name()
            == expected
        )
        assert title.testAttribute(Qt.WA_StyledBackground)
        foreground = title.title_label.palette().color(QPalette.WindowText).name()
        assert foreground == ("#e0e0e0" if selected == "dark" else "#212529")
        assert not title.minimize_button.icon().isNull()
        assert not title.maximize_button.icon().isNull()
        assert not title.close_button.icon().isNull()
        border = title.mapTo(
            window,
            QPoint(title.project_label.geometry().center().x(), title.height() - 1),
        )
        # Sample a complete physical pixel inside the 1-logical-pixel border;
        # its inner edge may cover half a pixel at fractional scaling.
        border_color = image.pixelColor(
            round(border.x() * ratio), int((border.y() + 1) * ratio) - 1
        )
        assert border_color.name() == ("#4a4a4a" if selected == "dark" else "#ced4da")
        if selected == "dark":
            edge = toolbar.mapTo(
                window, QPoint(toolbar.width() - 3, toolbar.height() // 2)
            )
            color = image.pixelColor(round(edge.x() * ratio), round(edge.y() * ratio))
            # Native toolbar edges may shade the stylesheet surface by one level.
            assert all(abs(channel - 46) <= 2 for channel in color.getRgb()[:3])
    window.close()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
