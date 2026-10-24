"""Frameless main-window gestures with native window-manager handoff.

Qt owns supported system moves/resizes. The fallback owns one captured gesture,
uses logical coordinates, and respects the window's live layout constraints.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, Qt
from PySide6.QtGui import QGuiApplication, QKeyEvent, QMouseEvent, QScreen
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QMainWindow,
    QMenuBar,
    QWidget,
)

_EDGES = {
    "top_left": Qt.Edge.TopEdge | Qt.Edge.LeftEdge,
    "top": Qt.Edge.TopEdge,
    "top_right": Qt.Edge.TopEdge | Qt.Edge.RightEdge,
    "left": Qt.Edge.LeftEdge,
    "right": Qt.Edge.RightEdge,
    "bottom_left": Qt.Edge.BottomEdge | Qt.Edge.LeftEdge,
    "bottom": Qt.Edge.BottomEdge,
    "bottom_right": Qt.Edge.BottomEdge | Qt.Edge.RightEdge,
}
_CURSORS = {
    "top_left": Qt.CursorShape.SizeFDiagCursor,
    "bottom_right": Qt.CursorShape.SizeFDiagCursor,
    "top_right": Qt.CursorShape.SizeBDiagCursor,
    "bottom_left": Qt.CursorShape.SizeBDiagCursor,
    "top": Qt.CursorShape.SizeVerCursor,
    "bottom": Qt.CursorShape.SizeVerCursor,
    "left": Qt.CursorShape.SizeHorCursor,
    "right": Qt.CursorShape.SizeHorCursor,
}


class FramelessWindow(QMainWindow):
    """One gesture owner for the main frame, independent of panel docking."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        self.setMouseTracking(True)
        self.grip_size = 8
        self.custom_title_bar_widget = None
        self.is_resizing = False
        self.resize_area = None
        self.is_moving = False
        self.drag_position = QPoint()
        self.start_geometry = None
        self.start_pos = None
        self._gesture = None
        self._was_maximized = False
        self._normal_geometry = None
        self._restoring = False
        self._observing_gesture = False
        self._owns_mouse_grab = False
        self._gesture_serial = 0
        self._grips = {}
        for area in _EDGES:
            grip = QWidget(self)
            grip.setObjectName(f"FrameGrip_{area}")
            grip.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
            grip.setMouseTracking(True)
            grip.setCursor(_CURSORS[area])
            self._grips[area] = grip

    def _observe_gesture(self, enabled: bool) -> None:
        """Observe application input only while Python owns a gesture."""
        if enabled == self._observing_gesture:
            return
        self._observing_gesture = enabled
        app = QApplication.instance()
        if enabled:
            app.installEventFilter(self)
        else:
            app.removeEventFilter(self)

    def event(self, event: QEvent) -> bool:
        """Keep frame lifecycle local, including close overrides in subclasses."""
        if hasattr(self, "_grips"):
            kind = event.type()
            if kind == QEvent.Type.ChildPolished:
                child = event.child()
                if isinstance(child, QWidget) and child not in self._grips.values():
                    child.installEventFilter(self)
            if (
                kind
                in (
                    QEvent.Type.Close,
                    QEvent.Type.Hide,
                    QEvent.Type.WindowDeactivate,
                    QEvent.Type.UngrabMouse,
                    QEvent.Type.WindowStateChange,
                )
                and not self._restoring
            ):
                self._finish_gesture()
            if kind in (
                QEvent.Type.Show,
                QEvent.Type.LayoutRequest,
                QEvent.Type.WindowStateChange,
            ):
                self._update_grips()
        return super().event(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Accept only frame-edge or title-background presses propagated to us."""
        if event.button() == Qt.MouseButton.LeftButton and self._press(event):
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Toggle the main title once without intercepting dock or editor input."""
        if event.button() == Qt.MouseButton.LeftButton and self._is_title_position(
            self.mapFromGlobal(event.globalPosition().toPoint())
        ):
            self._finish_gesture()
            self.custom_title_bar_widget.maximize_restore_requested.emit()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Restore legacy window commands after focused widgets handle input."""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier and event.key() in (
            Qt.Key.Key_Left,
            Qt.Key.Key_Right,
            Qt.Key.Key_Up,
            Qt.Key.Key_Down,
        ):
            if self._gesture in ("native_move", "native_resize"):
                event.ignore()
                return
            self._finish_gesture(cancel=True)
            screen = self.screen() or QGuiApplication.primaryScreen()
            if screen is not None:
                if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right):
                    self._place_on_side(screen, right=event.key() == Qt.Key.Key_Right)
                elif event.key() == Qt.Key.Key_Up:
                    self.showMaximized()
                elif self.isMaximized():
                    self.showNormal()
                else:
                    self.showMinimized()
            event.accept()
        else:
            super().keyPressEvent(event)

    def _size_limits(self):
        minimum = self.minimumSize().expandedTo(self.minimumSizeHint())
        maximum = self.maximumSize()
        # A contradictory layout hint must not override an explicit maximum.
        minimum.setWidth(min(minimum.width(), maximum.width()))
        minimum.setHeight(min(minimum.height(), maximum.height()))
        return minimum, maximum

    def _area_rects(self):
        grip, width, height = self.grip_size, self.width(), self.height()
        return {
            "top_left": QRect(0, 0, grip, grip),
            "top": QRect(grip, 0, max(0, width - 2 * grip), grip),
            "top_right": QRect(width - grip, 0, grip, grip),
            "left": QRect(0, grip, grip, max(0, height - 2 * grip)),
            "right": QRect(width - grip, grip, grip, max(0, height - 2 * grip)),
            "bottom_left": QRect(0, height - grip, grip, grip),
            "bottom": QRect(grip, height - grip, max(0, width - 2 * grip), grip),
            "bottom_right": QRect(width - grip, height - grip, grip, grip),
        }

    def _resize_edges(self, area):
        if self.isMaximized() or self.isFullScreen():
            return Qt.Edge(0)
        edges = _EDGES[area]
        minimum, maximum = self._size_limits()
        if minimum.width() >= maximum.width():
            edges &= ~(Qt.Edge.LeftEdge | Qt.Edge.RightEdge)
        if minimum.height() >= maximum.height():
            edges &= ~(Qt.Edge.TopEdge | Qt.Edge.BottomEdge)
        return edges

    def _update_grips(self):
        for area, rectangle in self._area_rects().items():
            grip = self._grips[area]
            grip.setGeometry(rectangle)
            edges = self._resize_edges(area)
            grip.setVisible(bool(edges))
            if edges:
                actual_area = next(
                    key for key, value in _EDGES.items() if value == edges
                )
                grip.setCursor(_CURSORS[actual_area])
                grip.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_grips"):
            self._update_grips()

    def _get_resize_area(self, pos: QPoint) -> str | None:
        for area, rectangle in self._area_rects().items():
            if rectangle.contains(pos):
                edges = self._resize_edges(area)
                return next(
                    (key for key, value in _EDGES.items() if value == edges), None
                )
        return None

    def updateCursorShape(self, pos: QPoint):
        """Keep the public frame cursor helper aligned with enabled grips."""
        self.setCursor(
            _CURSORS.get(self._get_resize_area(pos), Qt.CursorShape.ArrowCursor)
        )

    def _is_title_position(self, pos):
        title = self.custom_title_bar_widget
        if title is None or not title.isVisible():
            return False
        local = title.mapFrom(self, pos)
        if not title.rect().contains(local):
            return False
        child = self.childAt(pos)
        while child is not None and child is not title:
            if (
                isinstance(child, (QAbstractButton, QMenuBar))
                or child.focusPolicy() != Qt.FocusPolicy.NoFocus
            ):
                return False
            child = child.parentWidget()
        return child is title

    def _start_system_move(self):
        handle = self.windowHandle()
        return handle is not None and handle.startSystemMove()

    def _start_system_resize(self, edges):
        handle = self.windowHandle()
        return handle is not None and handle.startSystemResize(edges)

    def _finish_gesture(self, *, cancel=False):
        previous = self._gesture
        original = self.start_geometry
        maximized = self._was_maximized
        normal = self._normal_geometry
        self._gesture = None
        self._gesture_serial += 1
        self._observe_gesture(False)
        self.start_pos = None
        self.start_geometry = None
        self._normal_geometry = None
        self.resize_area = None
        self.is_moving = False
        self.is_resizing = False
        captured = self._owns_mouse_grab
        self._owns_mouse_grab = False
        if captured and QWidget.mouseGrabber() is self:
            self.releaseMouse()
        self.unsetCursor()
        if cancel and previous in ("move", "resize"):
            if maximized:
                self.setGeometry(normal)
                self.showMaximized()
            else:
                self.setGeometry(original)

    def _press(self, event):
        if not self.isVisible() or self.isFullScreen():
            return False
        pos = self.mapFromGlobal(event.globalPosition().toPoint())
        area = self._get_resize_area(pos)
        if area is None and not self._is_title_position(pos):
            return False
        self._finish_gesture()
        self.start_pos = event.globalPosition().toPoint()
        self.start_geometry = self.geometry()
        self._was_maximized = self.isMaximized()
        self._normal_geometry = self.normalGeometry() if self._was_maximized else None
        self.drag_position = self.start_pos - self.pos()
        if area:
            self.resize_area = area
            self.is_resizing = True
            self._handoff("resize")
        else:
            self._gesture = "pending_move"
            self._observe_gesture(True)
        return True

    def _handoff(self, operation: str) -> None:
        """Retire local tracking on native acceptance; capture only on refusal."""
        self._gesture = f"native_{operation}"
        serial = self._gesture_serial
        self._observe_gesture(False)
        accepted = (
            self._start_system_move()
            if operation == "move"
            else self._start_system_resize(_EDGES[self.resize_area])
        )
        if serial != self._gesture_serial:
            return
        if accepted:
            self._finish_gesture()
        else:
            self._gesture = operation
            self._observe_gesture(True)
            if operation == "resize":
                self.setCursor(_CURSORS[self.resize_area])
            self._owns_mouse_grab = True
            self.grabMouse()

    def _begin_move(self, global_pos):
        if self._was_maximized:
            ratio = self.drag_position.x() / max(1, self.width())
            vertical_offset = self.drag_position.y()
            normal = self._normal_geometry
            self._restoring = True
            self.showNormal()
            self.setGeometry(normal)
            self.drag_position = QPoint(round(ratio * self.width()), vertical_offset)
            self.move(global_pos - self.drag_position)
            self._restoring = False
        self.is_moving = True
        self._handoff("move")

    def _resize_fallback(self, global_pos):
        difference = global_pos - self.start_pos
        rectangle = QRect(self.start_geometry)
        edges = _EDGES[self.resize_area]
        minimum, maximum = self._size_limits()
        if edges & (Qt.Edge.LeftEdge | Qt.Edge.RightEdge):
            direction = -1 if edges & Qt.Edge.LeftEdge else 1
            width = max(
                minimum.width(),
                min(maximum.width(), rectangle.width() + direction * difference.x()),
            )
            rectangle.setWidth(width)
            if edges & Qt.Edge.LeftEdge:
                rectangle.moveRight(self.start_geometry.right())
        if edges & (Qt.Edge.TopEdge | Qt.Edge.BottomEdge):
            direction = -1 if edges & Qt.Edge.TopEdge else 1
            height = max(
                minimum.height(),
                min(maximum.height(), rectangle.height() + direction * difference.y()),
            )
            rectangle.setHeight(height)
            if edges & Qt.Edge.TopEdge:
                rectangle.moveBottom(self.start_geometry.bottom())
        self.setGeometry(rectangle)

    def _snap_fallback(self, global_pos):
        screen = (
            QGuiApplication.screenAt(global_pos)
            or self.screen()
            or QGuiApplication.primaryScreen()
        )
        if screen is None:
            return
        available = screen.availableGeometry()
        if abs(global_pos.y() - available.top()) <= 5:
            if screen is not self.screen():
                self.setScreen(screen)
                # Selecting a virtual-desktop screen does not move the window.
                # Keep the normal size, with its center on the pointer's screen.
                self.move(available.center() - self.rect().center())
            self.showMaximized()
        elif (
            abs(global_pos.x() - available.left()) <= 5
            or abs(global_pos.x() - available.right()) <= 5
        ):
            self._place_on_side(
                screen, right=abs(global_pos.x() - available.left()) > 5
            )

    def _place_on_side(self, screen: QScreen, *, right: bool) -> None:
        """Use the same constrained available-area placement for keys and drags."""
        self.showNormal()
        available = screen.availableGeometry()
        minimum, maximum = self._size_limits()
        half = (
            available.width() - available.width() // 2
            if right
            else available.width() // 2
        )
        width = max(minimum.width(), min(maximum.width(), half))
        height = max(minimum.height(), min(maximum.height(), available.height()))
        x = available.right() - width + 1 if right else available.left()
        self.setGeometry(x, available.top(), width, height)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Watch direct-child stacking locally and captured input temporarily."""
        if not isinstance(watched, QWidget):
            return False
        kind = event.type()
        if (
            watched not in self._grips.values()
            and watched.parent() is self
            and kind
            in (
                QEvent.Type.Show,
                QEvent.Type.LayoutRequest,
                QEvent.Type.ZOrderChange,
            )
        ):
            self._update_grips()
        if kind == QEvent.Type.ParentChange and watched.parent() is not self:
            watched.removeEventFilter(self)
        if not self._observing_gesture:
            return False
        belongs = watched.window() is self
        if (
            self._gesture
            and not belongs
            and kind
            in (
                QEvent.Type.MouseButtonPress,
                QEvent.Type.MouseButtonDblClick,
                QEvent.Type.MouseMove,
                QEvent.Type.MouseButtonRelease,
                QEvent.Type.KeyPress,
                QEvent.Type.KeyRelease,
            )
        ):
            # Captured frame input targets this window. Foreign input means
            # ownership ended; leave its delivery and geometry untouched.
            self._finish_gesture()
            return False
        if (
            self._gesture
            and kind == QEvent.Type.KeyPress
            and event.key() == Qt.Key.Key_Escape
        ):
            self._finish_gesture(cancel=True)
            return True
        if kind in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonDblClick):
            # Let the actual receiver decide whether this is a new frame gesture.
            self._finish_gesture()
        elif kind == QEvent.Type.MouseMove and self._gesture:
            if not event.buttons() & Qt.MouseButton.LeftButton:
                self._finish_gesture()
                return False
            global_pos = event.globalPosition().toPoint()
            if self._gesture == "pending_move":
                if (
                    global_pos - self.start_pos
                ).manhattanLength() < QApplication.startDragDistance():
                    return True
                self._begin_move(global_pos)
            if self._gesture == "resize":
                self._resize_fallback(global_pos)
            elif self._gesture == "move":
                self.move(global_pos - self.drag_position)
            return self._gesture in ("pending_move", "move", "resize")
        elif (
            kind == QEvent.Type.MouseButtonRelease
            and self._gesture
            and event.button() == Qt.MouseButton.LeftButton
        ):
            snap = self._gesture == "move"
            global_pos = event.globalPosition().toPoint()
            self._finish_gesture()
            if snap:
                self._snap_fallback(global_pos)
            return True
        return False
