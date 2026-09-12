"""Surface identity and real editor pointer/selection behavior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from PySide6.QtCore import QItemSelectionModel, QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QWidget

from optiland_gui.lens_editor import LensEditor
from optiland_gui.surface_interaction import SurfaceInteractionState


def test_identity_survives_insertion_and_prunes_deletion(qapp):
    a, b, c = object(), object(), object()
    optic = SimpleNamespace(surfaces=[a, b, c])
    state = SurfaceInteractionState()
    state.sync_document(optic)
    state.set_selected_indices([1])
    state.set_hover(2, 3)
    optic.surfaces.insert(0, object())
    state.sync_document(optic)
    assert state.selected_surfaces == (b,)
    assert state.index_of(b) == 2
    assert state.hovered_surface is c
    optic.surfaces.remove(b)
    state.sync_document(optic)
    assert state.selected_surfaces == ()
    state.sync_document(SimpleNamespace(surfaces=[a, c]))
    assert state.hovered_surface is None


def test_surface_and_body_precedence_are_independent(qapp):
    a, b, c = object(), object(), object()
    state = SurfaceInteractionState()
    state.sync_document(SimpleNamespace(surfaces=[a, b, c]))
    state.set_selected_indices([0])
    state.set_hover(1, 2)
    assert state.state_for([a, b]) == "selected"
    assert state.state_for([a]) == "selected"
    assert state.state_for([b]) == "hovered"
    assert state.state_for([c]) == "normal"
    state.set_hover(0, 4)
    assert state.state_for([a]) == "selected"
    state.set_selected_indices([])
    assert state.state_for([a]) == "hovered"
    state.set_hover()
    assert state.state_for([a]) == "normal"


def make_editor(minimal_optic):
    conn = MagicMock()
    conn.get_optic.return_value = minimal_optic
    conn.COL_TYPE, conn.COL_COMMENT, conn.COL_RADIUS = 0, 1, 2
    conn.COL_THICKNESS, conn.COL_MATERIAL, conn.COL_CONIC = 3, 4, 5
    conn.COL_SEMI_DIAMETER = 6
    conn.get_column_headers.return_value = [
        "Type",
        "Comment",
        "Radius",
        "Thickness",
        "Material",
        "Conic",
        "Semi-Diameter",
    ]
    conn.get_surface_count.side_effect = lambda: len(minimal_optic.surfaces)
    conn.get_optimization_variables.return_value = []
    conn.get_surface_type_info.return_value = {
        "display_text": "Standard",
        "is_changeable": True,
        "has_extra_params": False,
    }
    conn.get_surface_geometry_params.return_value = {}
    conn.get_available_surface_types.return_value = ["standard"]
    conn.get_surface_data.return_value = "0"
    editor = LensEditor(conn)
    editor.resize(850, 400)
    # Native desktop tools can retain foreground focus over a test window.
    # Keep this fixture unobscured so widgetAt tests the intended target.
    editor.setWindowFlag(Qt.WindowStaysOnTopHint)
    editor.show()
    editor.raise_()
    editor.activateWindow()
    QTest.qWaitForWindowExposed(editor)
    QTest.qWaitForWindowActive(editor)
    QTest.qWait(100)
    return editor, conn


def move_to_cell(editor, row, column):
    rect = editor.tableWidget.visualRect(editor.tableWidget.model().index(row, column))
    move_pointer(editor.tableWidget.viewport(), rect.center())


def move_pointer(widget, position):
    QTest.mouseMove(widget, position)
    # Native Windows delivers synthesized cursor motion through its event
    # queue; offscreen dispatches synchronously. Wait for the former too.
    QTest.qWait(100)


def test_editor_hover_selection_embedded_header_and_leave(qapp, minimal_optic):
    editor, conn = make_editor(minimal_optic)
    state = editor.interaction_state
    table = editor.tableWidget
    try:
        table.selectRow(1)
        selected = minimal_optic.surfaces[1]
        assert state.selected_surfaces == (selected,)
        move_to_cell(editor, 2, 3)
        assert state.hovered_surface is minimal_optic.surfaces[2]
        assert state.hovered_column == 3
        assert state.selected_surfaces == (selected,)
        embedded = table.cellWidget(2, 0).type_edit
        move_pointer(embedded, embedded.rect().center())
        assert state.hovered_surface is minimal_optic.surfaces[2]
        assert state.hovered_column == 0
        header = table.verticalHeader()
        move_pointer(
            header.viewport(), QPoint(5, header.sectionViewportPosition(1) + 10)
        )
        assert state.hovered_surface is selected
        assert state.hovered_column is None
        horizontal_header = table.horizontalHeader().viewport()
        move_pointer(horizontal_header, horizontal_header.rect().center())
        assert state.hovered_surface is None
        for column in range(table.columnCount()):
            table.setColumnWidth(column, 60)
        QTest.qWait(5)
        blank_position = QPoint(
            table.horizontalHeader().length() + 10, table.rowViewportPosition(1) + 10
        )
        assert table.viewport().rect().contains(blank_position)
        assert table.columnAt(blank_position.x()) == -1
        move_pointer(table.viewport(), blank_position)
        assert state.hovered_surface is None  # Empty space to the right of columns.
        move_pointer(table.viewport(), QPoint(20, table.viewport().height() - 5))
        assert state.hovered_surface is None
        assert state.selected_surfaces == (selected,)
        table.clearSelection()
        assert state.selected_surfaces == ()
        conn.set_surface_data.assert_not_called()
        conn.opticChanged.emit.assert_not_called()
    finally:
        editor.close()
        editor.deleteLater()


def test_selection_and_hover_survive_expanded_property_rows(qapp, minimal_optic):
    editor, conn = make_editor(minimal_optic)
    table = editor.tableWidget
    state = editor.interaction_state
    try:
        table.selectRow(2)
        editor.toggle_properties_widget(1)
        assert state.selected_surfaces == (minimal_optic.surfaces[2],)
        assert table.selectionModel().selectedRows()[0].row() == 3
        move_to_cell(editor, 3, 2)
        assert state.hovered_surface is minimal_optic.surfaces[2]
        props = table.cellWidget(2, 0)
        move_pointer(props, QPoint(15, 15))
        assert state.hovered_surface is minimal_optic.surfaces[1]
        assert state.hovered_column is None
        editor.load_data()
        assert state.selected_surfaces == (minimal_optic.surfaces[2],)
        table.selectionModel().select(
            table.model().index(1, 0),
            QItemSelectionModel.Select | QItemSelectionModel.Rows,
        )
        assert state.selected_surfaces == (
            minimal_optic.surfaces[1],
            minimal_optic.surfaces[2],
        )
        conn.set_surface_data.assert_not_called()
    finally:
        editor.close()
        editor.deleteLater()


def test_overlapping_window_and_pending_pointer_callbacks(qapp, minimal_optic):
    editor, _ = make_editor(minimal_optic)
    move_to_cell(editor, 1, 2)
    assert editor.interaction_state.hovered_surface is minimal_optic.surfaces[1]
    overlay = QWidget()
    overlay.setWindowFlag(Qt.WindowStaysOnTopHint)
    overlay.resize(editor.size())
    overlay.move(editor.mapToGlobal(QPoint(0, 0)))
    overlay.show()
    overlay.raise_()
    QTest.qWait(5)
    move_pointer(overlay, QPoint(150, 60))
    assert editor.interaction_state.hovered_surface is None
    overlay.close()
    editor.hover_tracker._refresh_timer.start(0)
    editor.hover_tracker._tracking_timer.start(0)
    editor.close()
    editor.deleteLater()
    QTest.qWait(5)  # Owned timers must not access widgets after destruction.


def test_keyboard_selection_scroll_and_pointer_leave(qapp, minimal_optic):
    editor, conn = make_editor(minimal_optic)
    table = editor.tableWidget
    state = editor.interaction_state
    try:
        table.setCurrentCell(1, 2)
        table.setFocus()
        QTest.keyClick(table, Qt.Key_Down)
        assert state.selected_surfaces == (minimal_optic.surfaces[2],)
        table.setFixedHeight(110)
        QTest.qWait(5)
        table.scrollToTop()
        move_to_cell(editor, 0, 2)
        assert state.hovered_surface is minimal_optic.surfaces[0]
        table.verticalScrollBar().setValue(table.rowHeight(0))
        assert state.hovered_surface is minimal_optic.surfaces[1]
        move_pointer(editor.btnAddSurface, editor.btnAddSurface.rect().center())
        assert state.hovered_surface is None
        assert state.selected_surfaces == (minimal_optic.surfaces[2],)
        conn.set_surface_data.assert_not_called()
    finally:
        editor.close()
        editor.deleteLater()
