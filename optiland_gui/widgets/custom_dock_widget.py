"""Provides a QDockWidget with a custom title bar."""

from __future__ import annotations

from PySide6.QtWidgets import QDockWidget, QWidget

from .custom_dock_title_bar import CustomDockTitleBar


class CustomDockWidget(QDockWidget):
    """A :class:`~PySide6.QtWidgets.QDockWidget` with a custom title bar.

    Wraps the standard dock widget and installs a
    :class:`CustomDockTitleBar` automatically.

    Args:
        title: The display title for the dock.
        parent: Optional parent widget.
    """

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self.setTitleBarWidget(CustomDockTitleBar(self, title))

    def setWidget(self, widget: QWidget) -> None:
        """Set the content widget while preserving the custom title bar.

        Args:
            widget: The content widget to embed.
        """
        current_title_bar = self.titleBarWidget()
        super().setWidget(widget)
        if not isinstance(self.titleBarWidget(), CustomDockTitleBar):
            self.setTitleBarWidget(current_title_bar)
