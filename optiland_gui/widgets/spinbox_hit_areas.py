"""Keep embedded spinbox editors outside native arrow hit areas."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QLineEdit,
    QStyle,
    QStyleOptionSpinBox,
    QWidget,
)


class SpinBoxHitAreaGuard(QObject):
    """Preserve native arrow targets for spinboxes owned by a window.

    Qt's stylesheet geometry can include an arrow's first pixel column in the
    editor rectangle. Correct overlapping editor geometry after Qt lays it out,
    without replacing native arrow painting, stepping or keyboard behavior.
    """

    def __init__(self, window: QWidget) -> None:
        self._adjusting = False
        self._window = window
        super().__init__(window)
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Trim only editor geometry that intersects a native arrow target."""
        if (
            self._adjusting
            or event.type()
            not in (QEvent.Type.Move, QEvent.Type.Resize, QEvent.Type.Show)
            or not isinstance(watched, QLineEdit)
        ):
            return False
        spin = watched.parentWidget()
        if (
            not isinstance(spin, QAbstractSpinBox)
            or watched is not spin.lineEdit()
            or not self._window.isAncestorOf(spin)
            or spin.buttonSymbols() == QAbstractSpinBox.ButtonSymbols.NoButtons
        ):
            return False
        option = QStyleOptionSpinBox()
        spin.initStyleOption(option)
        rect = watched.geometry()
        for control in (
            QStyle.SubControl.SC_SpinBoxUp,
            QStyle.SubControl.SC_SpinBoxDown,
        ):
            button = spin.style().subControlRect(
                QStyle.ComplexControl.CC_SpinBox, option, control, spin
            )
            if rect.intersects(button):
                if button.center().x() >= spin.rect().center().x():
                    rect.setRight(button.left() - 1)
                else:
                    rect.setLeft(button.right() + 1)
        if rect.isValid() and rect != watched.geometry():
            self._adjusting = True
            try:
                watched.setGeometry(rect)
            finally:
                self._adjusting = False
        return False
