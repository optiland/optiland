"""Paint editor hover without changing item data, selection or optical state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QModelIndex, QObject, QPersistentModelIndex, QRect
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QLineEdit, QToolButton, QWidget

if TYPE_CHECKING:
    from .lens_editor import LensEditor


class EditorHoverPresentation(QObject):
    """Consume the shared surface state without duplicating pointer tracking."""

    def __init__(self, editor: LensEditor) -> None:
        super().__init__(editor)
        self.editor = editor
        self.table = editor.tableWidget
        self.state = editor.interaction_state
        self._previous_row = -1
        self._editors: dict[int, tuple[QWidget, QPersistentModelIndex]] = {}
        self.state.changed.connect(self.refresh)

    def hovered_row(self) -> int:
        index = self.state.index_of(self.state.hovered_surface)
        return self.editor.map_surface_index_to_ui_row(index) if index >= 0 else -1

    def tint(self, row: int, column: int) -> QColor | None:
        if row != self.hovered_row():
            return None
        pointed = column == self.state.hovered_column
        dark = self.table.palette().color(QPalette.Base).lightnessF() < 0.5
        selected = self.table.selectionModel().isRowSelected(row)
        # Preserve the old theme's 4% cell-hover tint across the row. The pointed
        # cell is lighter; selected rows keep their selection background/cue.
        if dark or (selected and pointed):
            return QColor(255, 255, 255, 25 if pointed else 10)
        return QColor(0, 0, 0, 5 if pointed else 10)

    @staticmethod
    def _transparent(widget: QWidget, enabled: bool) -> None:
        if widget.property("ldeHoverBackground") == enabled:
            return
        widget.setProperty("ldeHoverBackground", enabled)
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    def register_editor(self, widget: QWidget, index: QModelIndex) -> None:
        """Keep an active cell editor intact as the pointer crosses the table."""
        key = id(widget)
        self._editors[key] = (widget, QPersistentModelIndex(index))
        widget.destroyed.connect(lambda: self._editors.pop(key, None))
        self._transparent(widget, index.row() == self.hovered_row())

    def is_editing(self, index: QModelIndex) -> bool:
        """Identify cells whose text is being drawn by an editor widget."""
        return any(candidate == index for _, candidate in self._editors.values())

    def unregister_editor(self, widget: QWidget) -> None:
        """Restore display text when editing ends, before deferred destruction."""
        entry = self._editors.pop(id(widget), None)
        if entry is not None:
            self.table.viewport().update(self.table.visualRect(entry[1]))

    def refresh(self) -> None:
        row = self.hovered_row()
        for affected in {self._previous_row, row}:
            if not 0 <= affected < self.table.rowCount():
                continue
            rectangle = QRect(
                0,
                self.table.rowViewportPosition(affected),
                self.table.viewport().width(),
                self.table.rowHeight(affected),
            )
            self.table.viewport().update(rectangle)
            # Only the main Type cell is an embedded row control. Expanded
            # property panels retain their own backgrounds and editing behavior.
            control = self.table.cellWidget(affected, self.editor.connector.COL_TYPE)
            if control is not None and hasattr(control, "type_edit"):
                self._transparent(control, affected == row)
                for child in control.findChildren(QLineEdit) + control.findChildren(
                    QToolButton
                ):
                    self._transparent(child, affected == row)
        for widget, index in self._editors.values():
            self._transparent(widget, index.isValid() and index.row() == row)
        self._previous_row = row
