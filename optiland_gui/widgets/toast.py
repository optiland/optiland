"""Toast notification system for the Optiland GUI.

Provides :class:`ToastWidget` (a single notification card) and
:class:`ToastManager` (the singleton stack manager owned by ``MainWindow``).

Design notes (from SPEC §1):
- All toasts are passive (information only, no action buttons).
- Auto-dismiss after 7 s except Error-level toasts (persist until clicked).
- Max 3 toasts visible; oldest is evicted immediately when a 4th arrives.
- Position: bottom-right of the main window with 16 px margin.
- Enter: slide in from right + fade in over 200 ms (OutCubic).
- Exit: slide out to right + fade out over 150 ms (InCubic).

Author: Manuel Fragata Mendes, 2025
"""

from __future__ import annotations

from PySide6.QtCore import (
    QEasingCurve,
    QParallelAnimationGroup,
    QPoint,
    QPropertyAnimation,
    Qt,
    QTimer,
)
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

_SEVERITY_ACCENT: dict[str, str] = {
    "success": "#4CAF50",
    "info": "#007ACC",
    "warning": "#FF9800",
    "error": "#F44336",
}

_SEVERITY_ICON: dict[str, str] = {
    "success": "✓",
    "info": "ℹ",
    "warning": "⚠",
    "error": "✕",
}

_AUTO_DISMISS_MS = 7_000
_MAX_VISIBLE = 3
_TOAST_WIDTH = 320
_TOAST_MARGIN = 16
_TOAST_SPACING = 8
_ENTER_DURATION = 200
_EXIT_DURATION = 150
_SHIFT_DURATION = 200


class ToastWidget(QWidget):
    """A single toast notification card.

    Args:
        message: The primary message text.
        severity: One of ``"success"``, ``"info"``, ``"warning"``, ``"error"``.
        sub_message: Optional secondary line shown below the main message.
        parent: The parent widget (should be the main window).
    """

    def __init__(
        self,
        message: str,
        severity: str = "info",
        sub_message: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.severity = severity
        self._dismissed = False

        self.setFixedWidth(_TOAST_WIDTH)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.SubWindow)

        self._build_ui(message, severity, sub_message)
        self._apply_shadow()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self, message: str, severity: str, sub_message: str | None) -> None:
        accent = _SEVERITY_ACCENT.get(severity, "#007ACC")
        icon_char = _SEVERITY_ICON.get(severity, "ℹ")

        # Outer container (rounded + background)
        outer = QWidget(self)
        outer.setObjectName("ToastOuter")
        outer.setAttribute(Qt.WA_StyledBackground, True)
        outer.setStyleSheet(
            f"""
            QWidget#ToastOuter {{
                background-color: #2D2D2D;
                border-radius: 8px;
                border-left: 4px solid {accent};
            }}
            """
        )

        outer_layout = QHBoxLayout(outer)
        outer_layout.setContentsMargins(12, 10, 12, 10)
        outer_layout.setSpacing(10)

        # Icon
        icon_label = QLabel(icon_char)
        icon_label.setStyleSheet(
            f"color: {accent}; font-size: 16px; background: transparent;"
        )
        icon_label.setFixedWidth(20)
        icon_label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        outer_layout.addWidget(icon_label)

        # Text block
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        msg_label = QLabel(message)
        msg_label.setWordWrap(True)
        msg_label.setStyleSheet(
            "color: #E0E0E0; font-size: 13px; background: transparent;"
        )
        msg_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        text_layout.addWidget(msg_label)

        if sub_message:
            sub_label = QLabel(sub_message)
            sub_label.setWordWrap(True)
            sub_label.setStyleSheet(
                "color: #AAAAAA; font-size: 11px; background: transparent;"
            )
            text_layout.addWidget(sub_label)

        outer_layout.addLayout(text_layout, 1)

        # Root layout
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(outer)

        self.adjustSize()

    def _apply_shadow(self) -> None:
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, int(0.6 * 255)))
        self.setGraphicsEffect(shadow)

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        """Dismiss the toast when the user clicks anywhere on it."""
        if self.parent() and hasattr(self.parent(), "_toast_manager"):
            self.parent()._toast_manager._dismiss(self)
        else:
            self.hide()
        super().mousePressEvent(event)


class ToastManager:
    """Manages the stack of :class:`ToastWidget` instances shown in the main window.

    Owned by ``MainWindow``.  Call :meth:`notify` from anywhere to show a toast.

    Args:
        parent_window: The :class:`~PySide6.QtWidgets.QMainWindow` that owns
            this manager.  Toasts are reparented to it so they float over
            the content area.
    """

    def __init__(self, parent_window: QWidget) -> None:
        self._parent = parent_window
        self._stack: list[ToastWidget] = []  # oldest first
        # Tag the parent so ToastWidget.mousePressEvent can reach us
        self._parent._toast_manager = self  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def notify(
        self,
        message: str,
        severity: str = "info",
        sub_message: str | None = None,
    ) -> None:
        """Show a toast notification.

        Args:
            message: Primary notification text.
            severity: ``"success"``, ``"info"``, ``"warning"``, or ``"error"``.
            sub_message: Optional secondary line.
        """
        # Evict oldest if at capacity
        if len(self._stack) >= _MAX_VISIBLE:
            self._dismiss(self._stack[0], animated=False)

        toast = ToastWidget(message, severity, sub_message, parent=self._parent)
        toast.adjustSize()
        self._stack.append(toast)

        # Position off-screen to the right initially
        target_pos = self._target_pos(len(self._stack) - 1, toast)
        start_pos = QPoint(target_pos.x() + _TOAST_WIDTH + 40, target_pos.y())
        toast.move(start_pos)
        toast.show()
        toast.raise_()

        # Shift existing toasts up
        self._restack_animated(skip_last=True)

        # Enter animation
        self._animate_enter(toast, start_pos, target_pos)

        # Auto-dismiss timer (errors do not auto-dismiss)
        if severity != "error":
            timer = QTimer(toast)
            timer.setSingleShot(True)
            timer.setInterval(_AUTO_DISMISS_MS)
            timer.timeout.connect(lambda t=toast: self._dismiss(t))
            timer.start()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _target_pos(self, stack_index: int, toast: ToastWidget) -> QPoint:
        """Compute the bottom-right target position for a given stack slot."""
        pw = self._parent.width()
        ph = self._parent.height()
        th = toast.sizeHint().height() if toast.sizeHint().height() > 10 else 70
        # Compute bottom-to-top
        total_below = 0
        for i in range(stack_index + 1, len(self._stack)):
            t = self._stack[i]
            total_below += t.sizeHint().height() + _TOAST_SPACING
        y = ph - _TOAST_MARGIN - th - total_below
        x = pw - _TOAST_WIDTH - _TOAST_MARGIN
        return QPoint(x, y)

    def _restack(self) -> None:
        """Reposition all toasts without animation."""
        for i, toast in enumerate(self._stack):
            pos = self._target_pos(i, toast)
            toast.move(pos)

    def _restack_animated(self, skip_last: bool = False) -> None:
        """Animate all existing toasts to their new positions."""
        count = len(self._stack)
        end = count - 1 if skip_last else count
        for i in range(end):
            toast = self._stack[i]
            target = self._target_pos(i, toast)
            anim = QPropertyAnimation(toast, b"pos", toast)
            anim.setDuration(_SHIFT_DURATION)
            anim.setEndValue(target)
            anim.setEasingCurve(QEasingCurve.OutCubic)
            anim.start(QPropertyAnimation.DeleteWhenStopped)

    def _animate_enter(
        self, toast: ToastWidget, start_pos: QPoint, target_pos: QPoint
    ) -> None:
        from PySide6.QtWidgets import QGraphicsOpacityEffect

        opacity_effect = QGraphicsOpacityEffect(toast)
        toast.setGraphicsEffect(opacity_effect)

        pos_anim = QPropertyAnimation(toast, b"pos", toast)
        pos_anim.setStartValue(start_pos)
        pos_anim.setEndValue(target_pos)
        pos_anim.setDuration(_ENTER_DURATION)
        pos_anim.setEasingCurve(QEasingCurve.OutCubic)

        opacity_anim = QPropertyAnimation(opacity_effect, b"opacity", toast)
        opacity_anim.setStartValue(0.0)
        opacity_anim.setEndValue(1.0)
        opacity_anim.setDuration(_ENTER_DURATION)
        opacity_anim.setEasingCurve(QEasingCurve.OutCubic)

        group = QParallelAnimationGroup(toast)
        group.addAnimation(pos_anim)
        group.addAnimation(opacity_anim)
        group.start(QParallelAnimationGroup.DeleteWhenStopped)

    def _animate_exit(self, toast: ToastWidget, on_finished: object) -> None:
        from PySide6.QtWidgets import QGraphicsOpacityEffect

        opacity_effect = QGraphicsOpacityEffect(toast)
        toast.setGraphicsEffect(opacity_effect)

        current_pos = toast.pos()
        end_pos = QPoint(current_pos.x() + _TOAST_WIDTH + 40, current_pos.y())

        pos_anim = QPropertyAnimation(toast, b"pos", toast)
        pos_anim.setStartValue(current_pos)
        pos_anim.setEndValue(end_pos)
        pos_anim.setDuration(_EXIT_DURATION)
        pos_anim.setEasingCurve(QEasingCurve.InCubic)

        opacity_anim = QPropertyAnimation(opacity_effect, b"opacity", toast)
        opacity_anim.setStartValue(1.0)
        opacity_anim.setEndValue(0.0)
        opacity_anim.setDuration(_EXIT_DURATION)
        opacity_anim.setEasingCurve(QEasingCurve.InCubic)

        group = QParallelAnimationGroup(toast)
        group.addAnimation(pos_anim)
        group.addAnimation(opacity_anim)
        group.finished.connect(on_finished)
        group.start(QParallelAnimationGroup.DeleteWhenStopped)

    def _dismiss(self, toast: ToastWidget, animated: bool = True) -> None:
        if toast._dismissed:
            return
        toast._dismissed = True
        if toast in self._stack:
            self._stack.remove(toast)

        def _cleanup():
            toast.hide()
            toast.deleteLater()
            self._restack_animated()

        if animated:
            self._animate_exit(toast, _cleanup)
        else:
            _cleanup()

    def reposition(self) -> None:
        """Reposition all toasts — call this when the parent window resizes."""
        self._restack()
