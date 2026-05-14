"""Provides the main sidebar navigation widget for the application.

This module defines :class:`SidebarWidget`, which is a
:class:`~PySide6.QtWidgets.QWidget` that serves as the primary navigation
control for the Optiland GUI.  It features a series of tool buttons for
selecting different application panels (e.g. Design, Analysis) and
automatically collapses to an icon-only view when resized below a threshold.

Author: Manuel Fragata Mendes, 2025
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QResizeEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

COLLAPSE_THRESHOLD_WIDTH = 80
SIDEBAR_MIN_WIDTH = 60
SIDEBAR_MAX_WIDTH = 150


class SidebarWidget(QWidget):
    """A collapsible sidebar widget for main panel navigation.

    Displays a vertical list of tool buttons that emit a signal when a menu
    item is selected.  Automatically collapses to an icon-only mode when its
    width falls below :data:`COLLAPSE_THRESHOLD_WIDTH`, and links to external
    resources via the title bar's buttons.

    Signals:
        menuSelected (str): Emitted when a navigation button is clicked,
            carrying the internal name of the selected menu.
        showWipMessage (str): Emitted when a work-in-progress button is
            clicked, carrying a description to show in a toast notification.

    Args:
        parent: Optional parent widget.
    """

    menuSelected = Signal(str)
    showWipMessage = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Buttons that are not yet implemented
        self._wip_buttons = [
            "dash",
            "materials",
            "tolerancing",
        ]
        self._last_checked_button: QToolButton | None = None
        self.setObjectName("SidebarWidget")
        self.setMinimumWidth(SIDEBAR_MIN_WIDTH)
        self.setMaximumWidth(SIDEBAR_MAX_WIDTH)

        self._is_collapsed = False
        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)

        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(5, 10, 5, 10)
        self._main_layout.setSpacing(5)

        self.title_label = QLabel("|||")
        self.title_label.setObjectName("SidebarTitleLabel")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setFixedHeight(30)
        self.title_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._main_layout.addWidget(self.title_label)

        self._buttons_list: list[dict] = []
        self.current_theme = "dark"

        button_definitions = [
            ("dash", "Dash", "dash.svg"),
            ("design", "Design", "design.svg"),
            ("analysis", "Analysis", "analysis.svg"),
            ("optimization", "Optimization", "optimization.svg"),
            ("materials", "Materials", "materials.svg"),
            ("tolerancing", "Tolerancing", "tolerancing.svg"),
            ("scripts", "Scripts", "terminal.svg"),
        ]

        for name, text, icon_filename in button_definitions:
            button = QToolButton()
            button.setObjectName(f"sidebar-btn-{name}")
            button.setText(text)
            button.setIconSize(QSize(24, 24))
            button.setCheckable(True)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.setFixedHeight(65)

            self._main_layout.addWidget(button)
            self._button_group.addButton(button)
            self._buttons_list.append(
                {
                    "widget": button,
                    "name": name,
                    "text": text,
                    "icon_filename": icon_filename,
                }
            )
            button.clicked.connect(self._handle_button_click)

        self._main_layout.addStretch(1)
        self.update_icons()

        # Pre-select the "design" button as the default navigation target
        design_item = next(
            (item for item in self._buttons_list if item["name"] == "design"), None
        )
        if design_item:
            design_item["widget"].setChecked(True)
            self._last_checked_button = design_item["widget"]

    def _handle_button_click(self) -> None:
        """Handle clicks on navigation buttons.

        Shows a WIP toast for unimplemented features; emits
        :attr:`menuSelected` for implemented ones.
        """
        checked_button = self._button_group.checkedButton()
        if not checked_button:
            return

        button_name = next(
            (b["name"] for b in self._buttons_list if b["widget"] == checked_button),
            None,
        )

        if button_name in self._wip_buttons:
            self.showWipMessage.emit(
                "This feature is currently under development. "
                "Stay tuned for updates to the GUI!"
            )
            checked_button.setChecked(False)
            if self._last_checked_button:
                self._last_checked_button.setChecked(True)
            return

        if button_name:
            self.menuSelected.emit(button_name)
            self._last_checked_button = checked_button

    def set_collapsed(self, collapsed: bool) -> None:
        """Set the collapsed state of the sidebar.

        In the collapsed state buttons show only icons; in the expanded state
        they show icons and text below them.

        Args:
            collapsed: ``True`` to collapse to icon-only mode.
        """
        if self._is_collapsed == collapsed:
            return

        self._is_collapsed = collapsed
        if collapsed:
            self.title_label.setText("|||")
            for item in self._buttons_list:
                item["widget"].setText("")
                item["widget"].setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
                item["widget"].setToolTip(item["text"])
        else:
            self.title_label.setText("|||")
            for item in self._buttons_list:
                item["widget"].setText(item["text"])
                item["widget"].setToolButtonStyle(
                    Qt.ToolButtonStyle.ToolButtonTextUnderIcon
                )
                item["widget"].setToolTip("")
        self.updateGeometry()

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Automatically collapse or expand based on available width."""
        super().resizeEvent(event)
        current_width = event.size().width()

        if current_width <= COLLAPSE_THRESHOLD_WIDTH and not self._is_collapsed:
            self.set_collapsed(True)
        elif current_width > COLLAPSE_THRESHOLD_WIDTH and self._is_collapsed:
            self.set_collapsed(False)

    def force_set_collapse_state(self, collapse: bool) -> None:
        """Force a specific collapse state, bypassing the width check.

        Args:
            collapse: The desired collapse state.
        """
        self.set_collapsed(collapse)

    def update_icons(self, theme: str = "dark") -> None:
        """Update all button icons to match *theme*.

        Args:
            theme: The theme name (``"dark"`` or ``"light"``).
        """
        self.current_theme = theme
        for item in self._buttons_list:
            icon_path = f":/icons/{self.current_theme}/{item['icon_filename']}"
            item["widget"].setIcon(QIcon(icon_path))
