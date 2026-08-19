"""Provides a custom title bar for the main application window.

This module defines :class:`CustomTitleBar`, a :class:`~PySide6.QtWidgets.QWidget`
that replaces the default window title bar to allow custom styling and additional
widgets — specifically the main menu bar and a project-name label.

Author: Manuel Fragata Mendes, 2025
"""

from __future__ import annotations

import webbrowser

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenuBar,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QToolButton,
    QWidget,
)

_GITHUB_URL = "https://github.com/optiland/optiland"
_HELP_URL = "https://optiland.readthedocs.io/en/latest/index.html"

_BTN_SIZE = QSize(30, 30)


class CustomTitleBar(QWidget):
    """A custom title bar for the main application window.

    Hosts the application name label, the main menu bar, quick-access tool
    buttons (settings, GitHub, help), and custom-styled minimize, maximize,
    and close buttons.  Window dragging is implemented via mouse event handlers.

    Signals:
        minimize_requested: Emitted when the minimize button is clicked.
        maximize_restore_requested: Emitted when the maximize/restore button
            is clicked.
        close_requested: Emitted when the close button is clicked.
        settings_requested: Emitted when the settings button is clicked.

    Args:
        main_menu_bar_instance: The :class:`~PySide6.QtWidgets.QMenuBar` to
            embed in the title bar.
        parent: Optional parent widget.
    """

    minimize_requested = Signal()
    maximize_restore_requested = Signal()
    close_requested = Signal()
    settings_requested = Signal()

    def __init__(
        self, main_menu_bar_instance: QMenuBar, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.current_theme = "dark"
        self.setObjectName("CustomTitleBar")
        self.setFixedHeight(40)
        self.setAutoFillBackground(True)

        self._mouse_move_offset = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 5, 0)
        layout.setSpacing(10)

        self.title_label = QLabel("Optiland")
        self.title_label.setObjectName("TitleBarOptilandLabel")
        self.title_label.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred
        )
        layout.addWidget(self.title_label)

        self.main_menu_bar = main_menu_bar_instance
        if self.main_menu_bar:
            self.main_menu_bar.setObjectName("TitleBarMenuBar")
            self.main_menu_bar.setSizePolicy(
                QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred
            )
            layout.addWidget(self.main_menu_bar, 0, Qt.AlignmentFlag.AlignCenter)

        # Vertical separator between menu and tool buttons
        separator = QFrame()
        separator.setObjectName("TitleBarSeparator")
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)

        # Quick-access tool buttons (settings, GitHub, documentation)
        self.settings_button = QToolButton()
        self.settings_button.setObjectName("TitleBarSettingsButton")
        self.settings_button.setFixedSize(_BTN_SIZE)
        self.settings_button.setToolTip("Settings")
        self.settings_button.clicked.connect(self.settings_requested.emit)
        layout.addWidget(self.settings_button)
        layout.addSpacing(-10)

        self.github_button = QToolButton()
        self.github_button.setObjectName("TitleBarGitHubButton")
        self.github_button.setFixedSize(_BTN_SIZE)
        self.github_button.setToolTip("Open GitHub Page")
        self.github_button.clicked.connect(self._open_github_url)
        layout.addWidget(self.github_button)
        layout.addSpacing(-10)

        self.help_button = QToolButton()
        self.help_button.setObjectName("TitleBarHelpButton")
        self.help_button.setFixedSize(_BTN_SIZE)
        self.help_button.setToolTip("Open Documentation")
        self.help_button.clicked.connect(self._open_help_url)
        layout.addWidget(self.help_button)

        layout.addStretch(1)

        self.project_label = QLabel("Current Project: UnnamedProject.opds")
        self.project_label.setObjectName("TitleBarProjectLabel")
        self.project_label.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred
        )
        layout.addWidget(self.project_label)

        layout.addSpacerItem(
            QSpacerItem(20, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)
        )

        # Window management buttons (minimize, maximize/restore, close)
        self.minimize_button = QPushButton()
        self.minimize_button.setObjectName("TitleBarMinimizeButton")
        self.minimize_button.setFixedSize(_BTN_SIZE)
        self.minimize_button.setToolTip("Minimize")
        self.minimize_button.clicked.connect(self.minimize_requested.emit)
        layout.addWidget(self.minimize_button)

        self.maximize_button = QPushButton()
        self.maximize_button.setObjectName("TitleBarMaximizeButton")
        self.maximize_button.setFixedSize(_BTN_SIZE)
        self.maximize_button.setCheckable(True)
        self.maximize_button.setToolTip("Maximize")
        self.maximize_button.clicked.connect(self.maximize_restore_requested.emit)
        layout.addWidget(self.maximize_button)

        self.close_button = QPushButton()
        self.close_button.setObjectName("TitleBarCloseButton")
        self.close_button.setFixedSize(_BTN_SIZE)
        self.close_button.setToolTip("Close")
        self.close_button.clicked.connect(self.close_requested.emit)
        layout.addWidget(self.close_button)

        self.update_theme_icons()

    def update_theme_icons(self, theme: str = "dark") -> None:
        """Update all button icons to match *theme*.

        Args:
            theme: The theme name (``"dark"`` or ``"light"``).
        """
        self.current_theme = theme
        self.minimize_button.setIcon(QIcon(f":/icons/{theme}/minimize.svg"))
        self.close_button.setIcon(QIcon(f":/icons/{theme}/close.svg"))
        self.maximize_button.setIcon(QIcon(f":/icons/{theme}/maximize_restore.svg"))
        self.settings_button.setIcon(QIcon(f":/icons/{theme}/settings.svg"))
        self.github_button.setIcon(QIcon(f":/icons/{theme}/brand_github.svg"))
        self.help_button.setIcon(QIcon(f":/icons/{theme}/help.svg"))

    def set_project_name(self, name: str) -> None:
        """Set the displayed project name label.

        Args:
            name: The project name.  Falls back to ``"UnnamedProject.opds"``
                if empty.
        """
        if not name:
            name = "UnnamedProject.opds"
        self.project_label.setText(f"Current Project: {name}")

    def update_maximize_button_state(self, is_maximized: bool) -> None:
        """Update the visual state and tooltip of the maximize/restore button.

        Args:
            is_maximized: ``True`` if the window is currently maximized.
        """
        self.maximize_button.setChecked(is_maximized)
        self.maximize_button.setToolTip("Restore" if is_maximized else "Maximize")

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        """Begin a window drag on left-button press."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._mouse_move_offset = (
                event.globalPosition().toPoint()
                - self.window().frameGeometry().topLeft()
            )
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        """Move the window while the left button is held."""
        if (
            self._mouse_move_offset is not None
            and event.buttons() == Qt.MouseButton.LeftButton
        ):
            new_window_pos = event.globalPosition().toPoint() - self._mouse_move_offset
            self.window().move(new_window_pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001
        """End the window drag on mouse release."""
        self._mouse_move_offset = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: ANN001
        """Toggle maximize/restore on double-click."""
        if event.button() == Qt.LeftButton:
            self.maximize_restore_requested.emit()

    def _open_github_url(self) -> None:
        """Open the Optiland GitHub repository in the default browser."""
        webbrowser.open(_GITHUB_URL)

    def _open_help_url(self) -> None:
        """Open the Optiland documentation in the default browser."""
        webbrowser.open(_HELP_URL)
