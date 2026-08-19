"""Provides a custom title bar for QDockWidgets.

Author: Manuel Fragata Mendes, 2025
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStyle,
    QStyleOption,
    QWidget,
)


class CustomDockTitleBar(QWidget):
    """A custom title bar for :class:`~PySide6.QtWidgets.QDockWidget` instances.

    Provides macOS-style traffic-light buttons (hide, float/dock, close) and
    supports dragging the dock widget while it is floating.

    Args:
        parent_dock: The dock widget that owns this title bar.
        title: Display text shown in the title bar label.
    """

    def __init__(self, parent_dock: QDockWidget, title: str = "") -> None:
        super().__init__(parent_dock)
        self.setObjectName("CustomDockTitleBar")
        self.dock_widget = parent_dock

        # Initialise drag-tracking state
        self._mouse_press_pos = None
        self._window_pos_before_move = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 3, 8, 3)
        layout.setSpacing(2)

        self.title_label = QLabel(title)
        layout.addWidget(self.title_label)
        layout.addStretch()

        self.minimize_btn = QPushButton(self)
        self.minimize_btn.setObjectName("DockMinimizeButton")
        self.minimize_btn.setFixedSize(10, 10)
        self.minimize_btn.setToolTip("Hide")
        self.minimize_btn.clicked.connect(parent_dock.toggleViewAction().trigger)

        self.undock_btn = QPushButton(self)
        self.undock_btn.setObjectName("DockUndockButton")
        self.undock_btn.setFixedSize(10, 10)
        self.undock_btn.setToolTip("Float/Dock")
        self.undock_btn.clicked.connect(
            lambda: parent_dock.setFloating(not parent_dock.isFloating())
        )

        self.close_btn = QPushButton(self)
        self.close_btn.setObjectName("DockCloseButton")
        self.close_btn.setFixedSize(10, 10)
        self.close_btn.setToolTip("Close")
        self.close_btn.clicked.connect(parent_dock.close)

        layout.addWidget(self.minimize_btn)
        layout.addWidget(self.undock_btn)
        layout.addWidget(self.close_btn)

    def paintEvent(self, event) -> None:  # noqa: ANN001
        """Draw the background according to the active stylesheet."""
        opt = QStyleOption()
        opt.initFrom(self)
        painter = QPainter(self)
        self.style().drawPrimitive(
            QStyle.PrimitiveElement.PE_Widget, opt, painter, self
        )
        super().paintEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        """Begin a drag operation when the left button is pressed over
        a floating dock."""
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.dock_widget.isFloating()
        ):
            self._mouse_press_pos = event.globalPosition().toPoint()
            self._window_pos_before_move = self.dock_widget.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        """Move the floating dock widget as the mouse is dragged."""
        if (
            self.dock_widget.isFloating()
            and event.buttons() == Qt.MouseButton.LeftButton
            and self._mouse_press_pos is not None
        ):
            delta = event.globalPosition().toPoint() - self._mouse_press_pos
            self.dock_widget.move(self._window_pos_before_move + delta)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001
        """End the drag operation on mouse release."""
        self._mouse_press_pos = None
        super().mouseReleaseEvent(event)
