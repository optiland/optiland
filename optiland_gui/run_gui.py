"""Entry point for launching the Optiland GUI application.

This module initializes the :class:`~PySide6.QtWidgets.QApplication` and the
:class:`~optiland_gui.main_window.MainWindow`, starting the event loop to run
the graphical user interface for Optiland.

Authors:
    Manuel Fragata Mendes, 2025
"""

from __future__ import annotations

import ctypes
import os
import sys

from PySide6.QtCore import QLocale, QSize, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen

from .config import APPLICATION_NAME, OPTILAND_ICON_PATH, ORGANIZATION_NAME
from .main_window import MainWindow
from .resources import resources_rc  # noqa: F401


def main() -> None:
    """Application entry point."""
    if sys.platform == "win32":
        myappid = f"{ORGANIZATION_NAME}.{APPLICATION_NAME}.1.0"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    # Force XCB (X11) on Linux to avoid BadWindow / X_ConfigureWindow crashes
    # caused by incompatibility between the VTK/OpenGL render pipeline and the
    # Qt Wayland backend.  This must be set before QApplication is created.
    if sys.platform.startswith("linux"):
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(OPTILAND_ICON_PATH))
    QLocale.setDefault(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))

    original_pixmap = QPixmap(":/images/logo.png")
    desired_size = QSize(700, 400)
    scaled_pixmap = original_pixmap.scaled(
        desired_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
    )

    # Create and show splash screen
    splash = QSplashScreen(scaled_pixmap)
    splash.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
    splash.setEnabled(False)
    splash.showMessage(
        "<h3>Initializing application...</h3>",
        Qt.AlignBottom | Qt.AlignHCenter,
        Qt.white,
    )
    splash.show()
    app.processEvents()

    # Initialize the main window while splash is visible.  The time taken
    # here is the actual loading time the user experiences.
    window = MainWindow()
    window.show()

    # Close the splash screen once the main window is ready.
    splash.finish(window)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
