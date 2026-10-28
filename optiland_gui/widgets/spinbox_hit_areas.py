"""Keep embedded spinbox editors outside native arrow hit areas."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QDoubleSpinBox,
    QLineEdit,
    QSpinBox,
    QStyle,
    QStyleOptionSpinBox,
)


class SpinBoxHitAreaGuard(QObject):
    """Preserve native arrow targets by observing only one spinbox editor.

    Qt's stylesheet geometry can include an arrow's first pixel column in the
    editor rectangle. Correct overlapping editor geometry after Qt lays it out,
    without replacing native arrow painting, stepping or keyboard behavior.
    """

    def __init__(self, editor: QLineEdit) -> None:
        self._adjusting = False
        self._editor = editor
        super().__init__(editor)
        editor.installEventFilter(self)
        self._trim_editor()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Repair editor geometry without observing unrelated application events."""
        if not self._adjusting and event.type() in (
            QEvent.Type.Move,
            QEvent.Type.Resize,
            QEvent.Type.Show,
        ):
            self._trim_editor()
        return False

    def _trim_editor(self) -> None:
        editor = self._editor
        spin = editor.parentWidget()
        if (
            not isinstance(spin, QAbstractSpinBox)
            or editor is not spin.lineEdit()
            or spin.buttonSymbols() == QAbstractSpinBox.ButtonSymbols.NoButtons
        ):
            return
        option = QStyleOptionSpinBox()
        spin.initStyleOption(option)
        rect = editor.geometry()
        for control in (
            QStyle.SubControl.SC_SpinBoxUp,
            QStyle.SubControl.SC_SpinBoxDown,
        ):
            button = spin.style().subControlRect(
                QStyle.ComplexControl.CC_SpinBox, option, control, spin
            )
            if rect.intersects(button):
                # Wide arrows can extend past the widget's midpoint at large
                # fonts. Trim the editor's nearest edge, not the widget's.
                if button.center().x() >= rect.center().x():
                    rect.setRight(button.left() - 1)
                else:
                    rect.setLeft(button.right() + 1)
        if rect.isValid() and rect != editor.geometry():
            self._adjusting = True
            try:
                editor.setGeometry(rect)
            finally:
                self._adjusting = False


def protect_spinbox_editor(spin: QAbstractSpinBox) -> None:
    """Attach an editor-owned guard, including for native item-delegate editors."""
    editor = spin.lineEdit()
    if not hasattr(editor, "_spinbox_hit_area_guard"):
        editor._spinbox_hit_area_guard = SpinBoxHitAreaGuard(editor)


class _SpinBoxEditorMixin:
    """Register each editor at construction and replacement without tree scans."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        protect_spinbox_editor(self)

    def setLineEdit(self, editor: QLineEdit) -> None:  # noqa: N802
        """Keep replacement editors protected; Qt deletes the old editor/guard."""
        super().setLineEdit(editor)
        protect_spinbox_editor(self)


class SpinBox(_SpinBoxEditorMixin, QSpinBox):
    """Integer spinbox with editor-local protection of native arrow hit areas."""


class DoubleSpinBox(_SpinBoxEditorMixin, QDoubleSpinBox):
    """Decimal spinbox with editor-local protection of native arrow hit areas."""
