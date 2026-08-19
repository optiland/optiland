"""
Provides a frameless window base class for the Optiland GUI.

This module defines `FramelessWindow`, a QMainWindow subclass that implements
custom window decorations, including a custom title bar and logic for resizing,
moving, and snapping the window (AeroSnap-like behavior).

Author: Manuel Fragata Mendes, 2025
Refactored by: Jules, 2025
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, QRect, Qt
from PySide6.QtWidgets import QApplication, QMainWindow


class FramelessWindow(QMainWindow):
    """A base class for creating a frameless main window with custom controls.

    This class handles the logic for moving, resizing, and snapping the window,
    which is necessary when using the `Qt.FramelessWindowHint`. It also provides
    hooks for a custom title bar.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setMouseTracking(True)

        self.grip_size = 8
        self.is_resizing = False
        self.resize_area = None
        self.is_moving = False
        self.drag_position = QPoint()
        self.start_geometry = None
        self.start_pos = None

        # This attribute should be set by the subclass
        self.custom_title_bar_widget = None

    def mousePressEvent(self, event: QEvent):
        """Handle mouse press events for window dragging and resizing."""
        if event.button() == Qt.LeftButton:
            cursor_pos = event.position().toPoint()
            self.resize_area = self._get_resize_area(cursor_pos)

            if self.resize_area:
                self.is_resizing = True
            elif self.custom_title_bar_widget:
                titlebar_rect = self.custom_title_bar_widget.rect()
                global_pos = self.custom_title_bar_widget.mapToGlobal(QPoint(0, 0))
                window_pos = self.mapFromGlobal(global_pos)
                titlebar_rect.moveTo(window_pos)

                if titlebar_rect.contains(cursor_pos) and not self.isMaximized():
                    self.is_moving = True
                    self.drag_position = (
                        event.globalPosition().toPoint()
                        - self.frameGeometry().topLeft()
                    )

            self.start_geometry = self.geometry()
            self.start_pos = event.globalPosition().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QEvent):
        """Handle mouse move events for window dragging, resizing, and AeroSnap."""
        cursor_pos = event.position().toPoint()
        global_pos = event.globalPosition().toPoint()

        self.updateCursorShape(cursor_pos)

        if self.is_resizing and not self.isMaximized():
            diff = global_pos - self.start_pos
            new_geometry = QRect(self.start_geometry)

            if self.resize_area in ["top_left", "top", "top_right"]:
                new_geometry.setTop(self.start_geometry.top() + diff.y())
            if self.resize_area in ["bottom_left", "bottom", "bottom_right"]:
                new_geometry.setBottom(self.start_geometry.bottom() + diff.y())
            if self.resize_area in ["top_left", "left", "bottom_left"]:
                new_geometry.setLeft(self.start_geometry.left() + diff.x())
            if self.resize_area in ["top_right", "right", "bottom_right"]:
                new_geometry.setRight(self.start_geometry.right() + diff.x())

            if (
                new_geometry.width() >= self.minimumWidth()
                and new_geometry.height() >= self.minimumHeight()
            ):
                self.setGeometry(new_geometry)

        elif self.is_moving and not self.isMaximized():
            self.move(global_pos - self.drag_position)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QEvent):
        """Handle mouse release events and implement AeroSnap actions."""
        if event.button() == Qt.LeftButton:
            if self.is_moving and not self.isMaximized():
                screen = self.screen()
                if not screen:
                    screen = QApplication.primaryScreen()
                screen_geometry = screen.availableGeometry()
                global_pos = event.globalPosition().toPoint()

                at_top_edge = global_pos.y() <= screen_geometry.top() + 5
                at_left_edge = global_pos.x() <= screen_geometry.left() + 5
                at_right_edge = global_pos.x() >= screen_geometry.right() - 5

                if at_top_edge:
                    self.showMaximized()
                elif at_left_edge:
                    self.showNormal()
                    new_geometry = QRect(
                        screen_geometry.left(),
                        screen_geometry.top(),
                        screen_geometry.width() // 2,
                        screen_geometry.height(),
                    )
                    self.setGeometry(new_geometry)
                elif at_right_edge:
                    self.showNormal()
                    new_geometry = QRect(
                        screen_geometry.left() + screen_geometry.width() // 2,
                        screen_geometry.top(),
                        screen_geometry.width() // 2,
                        screen_geometry.height(),
                    )
                    self.setGeometry(new_geometry)

            self.is_moving = False
            self.is_resizing = False
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    def _get_resize_area(self, pos: QPoint) -> str | None:
        """Determines which resize area the mouse cursor is in."""
        rect = self.rect()
        grip = self.grip_size

        if QRect(0, 0, grip, grip).contains(pos):
            return "top_left"
        if QRect(rect.width() - grip, 0, grip, grip).contains(pos):
            return "top_right"
        if QRect(0, rect.height() - grip, grip, grip).contains(pos):
            return "bottom_left"
        if QRect(rect.width() - grip, rect.height() - grip, grip, grip).contains(pos):
            return "bottom_right"
        if QRect(grip, 0, rect.width() - 2 * grip, grip).contains(pos):
            return "top"
        if QRect(0, grip, grip, rect.height() - 2 * grip).contains(pos):
            return "left"
        if QRect(rect.width() - grip, grip, grip, rect.height() - 2 * grip).contains(
            pos
        ):
            return "right"
        if QRect(grip, rect.height() - grip, rect.width() - 2 * grip, grip).contains(
            pos
        ):
            return "bottom"
        return None

    def updateCursorShape(self, pos: QPoint):
        """Update the cursor shape based on the mouse position."""
        if self.isMaximized():
            self.setCursor(Qt.ArrowCursor)
            return

        resize_area = self._get_resize_area(pos)

        if resize_area in ("top_left", "bottom_right"):
            self.setCursor(Qt.SizeFDiagCursor)
        elif resize_area in ("top_right", "bottom_left"):
            self.setCursor(Qt.SizeBDiagCursor)
        elif resize_area in ("top", "bottom"):
            self.setCursor(Qt.SizeVerCursor)
        elif resize_area in ("left", "right"):
            self.setCursor(Qt.SizeHorCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def keyPressEvent(self, event: QEvent):
        """Handle key events for window management shortcuts."""
        if event.modifiers() & Qt.ControlModifier:
            screen = self.screen()
            if not screen:
                screen = QApplication.primaryScreen()
            screen_geometry = screen.availableGeometry()

            if event.key() == Qt.Key_Left:
                self.showNormal()
                self.setGeometry(
                    QRect(
                        screen_geometry.left(),
                        screen_geometry.top(),
                        screen_geometry.width() // 2,
                        screen_geometry.height(),
                    )
                )
            elif event.key() == Qt.Key_Right:
                self.showNormal()
                self.setGeometry(
                    QRect(
                        screen_geometry.left() + screen_geometry.width() // 2,
                        screen_geometry.top(),
                        screen_geometry.width() // 2,
                        screen_geometry.height(),
                    )
                )
            elif event.key() == Qt.Key_Up:
                self.showMaximized()
            elif event.key() == Qt.Key_Down:
                if self.isMaximized():
                    self.showNormal()
                else:
                    self.showMinimized()
        super().keyPressEvent(event)
