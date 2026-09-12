"""GUI-only surface identity and editor pointer state.

These references belong to the live document. A computed scene maps its surface
indices through the tuple captured when that scene was requested; workers must
never receive this QObject or the live surface references.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QTimer, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QApplication, QWidget


class SurfaceInteractionState(QObject):
    """Selection and hover, independent of optical edits and renderer objects."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.document = None
        self.surfaces = ()
        self.selected_surfaces = ()
        self.hovered_surface = None
        self.hovered_column = None

    def sync_document(self, document):
        """Prune deleted identities, or clear state for a different document."""
        surfaces = tuple(document.surfaces) if document is not None else ()
        old = self._signature()
        if document is not self.document:
            self.selected_surfaces = ()
            self.hovered_surface = None
            self.hovered_column = None
        self.document, self.surfaces = document, surfaces
        valid = {id(surface) for surface in surfaces}
        self.selected_surfaces = tuple(
            surface for surface in self.selected_surfaces if id(surface) in valid
        )
        if id(self.hovered_surface) not in valid:
            self.hovered_surface = None
            self.hovered_column = None
        if old != self._signature():
            self.changed.emit()

    def surface_at(self, index):
        return self.surfaces[index] if 0 <= index < len(self.surfaces) else None

    def index_of(self, surface):
        return next(
            (i for i, value in enumerate(self.surfaces) if value is surface), -1
        )

    def set_selected_indices(self, indices):
        selected = tuple(
            self.surfaces[i]
            for i in sorted(set(indices))
            if 0 <= i < len(self.surfaces)
        )
        if tuple(map(id, selected)) != tuple(map(id, self.selected_surfaces)):
            self.selected_surfaces = selected
            self.changed.emit()

    def set_hover(self, index=-1, column=None):
        surface = self.surface_at(index)
        column = column if surface is not None else None
        if surface is not self.hovered_surface or column != self.hovered_column:
            self.hovered_surface, self.hovered_column = surface, column
            self.changed.emit()

    def state_for(self, surfaces):
        """Resolve selected > hovered > normal for a surface or shared body."""
        identities = {id(surface) for surface in surfaces}
        if any(id(surface) in identities for surface in self.selected_surfaces):
            return "selected"
        if self.hovered_surface is not None and id(self.hovered_surface) in identities:
            return "hovered"
        return "normal"

    def _signature(self):
        return (
            id(self.document),
            tuple(map(id, self.surfaces)),
            tuple(map(id, self.selected_surfaces)),
            id(self.hovered_surface),
            self.hovered_column,
        )


class EditorHoverTracker(QObject):
    """Observe pointer events without consuming input from embedded cell widgets."""

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.table = editor.tableWidget
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self.refresh)
        self._tracking_timer = QTimer(self)
        self._tracking_timer.setSingleShot(True)
        self._tracking_timer.timeout.connect(self._enable_tracking)
        self.table.setMouseTracking(True)
        self.table.viewport().setMouseTracking(True)
        self.table.verticalHeader().setMouseTracking(True)
        self._enable_tracking()
        # Application filtering also sees active cell editors and newly inserted
        # child controls; a viewport-only filter misses their mouse events.
        QApplication.instance().installEventFilter(self)
        self.table.verticalScrollBar().valueChanged.connect(self.refresh)
        self.table.horizontalScrollBar().valueChanged.connect(self.refresh)

    def eventFilter(self, source, event):
        kind = event.type()
        if kind in (QEvent.MouseMove, QEvent.Enter):
            if hasattr(event, "globalPosition"):
                self.update_at(event.globalPosition().toPoint())
        elif kind in (QEvent.Leave, QEvent.Hide, QEvent.Resize, QEvent.LayoutRequest):
            if source is self.table or source is self.table.viewport():
                self._refresh_timer.start(0)
        elif (
            kind == QEvent.ChildAdded
            and isinstance(source, QWidget)
            and self.table.isAncestorOf(source)
        ):
            self._tracking_timer.start(0)
        return False

    def _enable_tracking(self):
        for widget in self.table.findChildren(QWidget):
            widget.setMouseTracking(True)

    def refresh(self, *args):
        self.update_at(QCursor.pos())

    def update_at(self, global_position):
        state = self.editor.interaction_state
        if not self.table.isVisible():
            state.set_hover()
            return
        target = QApplication.widgetAt(global_position)
        if target is None or (
            target is not self.table and not self.table.isAncestorOf(target)
        ):
            state.set_hover()
            return
        viewport = self.table.viewport()
        position = viewport.mapFromGlobal(global_position)
        header = self.table.verticalHeader()
        header_position = header.viewport().mapFromGlobal(global_position)
        column = None
        if viewport.rect().contains(position):
            row = self.table.rowAt(position.y())
            column = self.table.columnAt(position.x())
            if column < 0:
                row = -1
        elif header.viewport().rect().contains(header_position):
            row = header.logicalIndexAt(header_position)
        else:
            row = -1
        if row < 0:
            state.set_hover()
            return
        surface_index = self.editor.map_ui_row_to_surface_index(row)
        if (
            self.editor.open_prop_source_row >= 0
            and row == self.editor.open_prop_source_row + 1
        ):
            column = None
        state.set_hover(surface_index, column)
