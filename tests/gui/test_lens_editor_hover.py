"""Row/cell hover is visual feedback, never an edit or selection operation."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLineEdit

from optiland_gui.lens_editor import LensEditor
from tests.gui.test_surface_interaction import make_editor, move_pointer, move_to_cell


@pytest.fixture(params=["dark", "light"])
def themed_editor(qapp, minimal_optic, request):
    previous = qapp.styleSheet()
    path = Path(__file__).parents[2] / "optiland_gui/resources/styles"
    qapp.setStyleSheet(
        (path / f"{request.param}_theme.qss").read_text(encoding="utf-8")
    )
    editor, connector = make_editor(minimal_optic)
    yield editor, connector, request.param
    for widget, _ in tuple(editor.hover_presentation._editors.values()):
        QTest.keyClick(widget, Qt.Key_Escape)
    editor.close()
    editor.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.setStyleSheet(previous)


def sample(table, row, column):
    rect = table.visualRect(table.model().index(row, column))
    point = rect.topLeft() + QPoint(10, 5)
    image = table.viewport().grab().toImage()
    ratio = image.devicePixelRatio()
    return image.pixelColor(round(point.x() * ratio), round(point.y() * ratio))


def test_whole_row_changes_and_pointed_cell_is_lighter(themed_editor, qapp):
    editor, connector, theme = themed_editor
    table = editor.tableWidget
    move_pointer(editor.btnAddSurface, QPoint(5, 5))
    baseline = [sample(table, 1, column) for column in range(1, 7)]
    move_to_cell(editor, 1, 3)
    hovered = [sample(table, 1, column) for column in range(1, 7)]
    assert all(before != after for before, after in zip(baseline, hovered, strict=True))
    assert hovered[2].lightnessF() > hovered[1].lightnessF()
    move_to_cell(editor, 1, 4)
    assert sample(table, 1, 4).lightnessF() > sample(table, 1, 3).lightnessF()
    move_to_cell(editor, 2, 4)
    assert [sample(table, 1, column) for column in range(1, 7)] == baseline
    connector.set_surface_data.assert_not_called()
    connector.opticChanged.emit.assert_not_called()


def test_embedded_type_and_header_use_same_row_state(themed_editor):
    editor, _, _ = themed_editor
    table = editor.tableWidget
    embedded = table.cellWidget(1, 0)
    move_pointer(embedded.type_edit, embedded.type_edit.rect().center())
    assert embedded.type_edit.property("ldeHoverBackground") is True
    assert editor.interaction_state.hovered_column == 0
    assert editor.hover_presentation.tint(1, 0) != editor.hover_presentation.tint(1, 1)
    header = table.verticalHeader()
    move_pointer(header.viewport(), QPoint(5, header.sectionViewportPosition(1) + 10))
    assert editor.interaction_state.hovered_column is None
    assert editor.hover_presentation.tint(1, 0) == editor.hover_presentation.tint(1, 3)
    move_pointer(editor.btnAddSurface, QPoint(5, 5))
    assert embedded.type_edit.property("ldeHoverBackground") is False
    assert editor.hover_presentation.hovered_row() == -1


def test_selection_variable_and_active_editor_survive_hover(themed_editor, qapp):
    editor, connector, _ = themed_editor
    table = editor.tableWidget
    table.selectRow(1)
    item = table.item(1, 2)
    variable = QBrush(QColor(100, 150, 255, 80))
    table.blockSignals(True)
    item.setBackground(variable)
    table.blockSignals(False)
    selected = editor.interaction_state.selected_surfaces
    move_to_cell(editor, 1, 3)
    assert sample(table, 1, 3).lightnessF() > sample(table, 1, 4).lightnessF()
    assert editor.interaction_state.selected_surfaces == selected
    assert item.background() == variable
    rect = table.visualRect(table.model().index(1, 2))
    image = table.viewport().grab().toImage()
    ratio = image.devicePixelRatio()
    marker = image.pixelColor(
        round((rect.x() + 3) * ratio), round((rect.y() + 8) * ratio)
    )
    assert marker.blue() > marker.red()
    index = table.model().index(1, 1)
    table.edit(index)
    qapp.processEvents()
    active = next(
        widget
        for widget in table.viewport().findChildren(QLineEdit)
        if widget.objectName() != "SurfaceTypeLineEdit"
    )
    active.setText("editing remains intact")
    focus = qapp.focusWidget()
    for row, column in ((1, 1), (2, 3), (1, 4)):
        move_to_cell(editor, row, column)
        assert active.text() == "editing remains intact"
        assert qapp.focusWidget() is focus
        assert editor.interaction_state.selected_surfaces == selected
        assert item.background() == variable
    connector.set_surface_data.assert_not_called()


def test_expanded_panel_hover_paints_source_row_only(themed_editor, qapp):
    editor, _, _ = themed_editor
    editor.toggle_properties_widget(1)
    qapp.processEvents()
    panel = editor.tableWidget.cellWidget(2, 0)
    move_pointer(panel, QPoint(15, 15))
    assert editor.hover_presentation.hovered_row() == 1
    assert editor.hover_presentation.tint(1, 3) is not None
    assert editor.hover_presentation.tint(2, 3) is None
    assert panel.property("ldeHoverBackground") is None


def test_empty_active_editor_does_not_reveal_old_display_text(themed_editor, qapp):
    editor, connector, _ = themed_editor
    table = editor.tableWidget
    item = table.item(1, 1)
    table.blockSignals(True)
    item.setText("Old visible comment")
    table.blockSignals(False)
    table.setCurrentCell(1, 1)
    table.edit(table.model().index(1, 1))
    qapp.processEvents()
    active = next(iter(editor.hover_presentation._editors.values()))[0]
    active.clear()
    move_pointer(active, QPoint(20, 8))
    assert active.text() == ""
    assert item.text() == "Old visible comment"
    # Exclude the caret and native focus border. A blank draft must show only
    # the hovered cell background, not the old elided display text behind it.
    region = active.geometry().adjusted(12, 6, -12, -6)
    image = table.viewport().grab().toImage()
    ratio = image.devicePixelRatio()
    colors = {
        image.pixelColor(round(x * ratio), round(y * ratio)).rgba()
        for x in range(region.left(), region.right())
        for y in range(region.top(), region.bottom())
    }
    assert len(colors) == 1
    QTest.keyClick(active, Qt.Key_Escape)
    qapp.processEvents()
    assert not editor.hover_presentation.is_editing(table.model().index(1, 1))
    assert item.text() == "Old visible comment"
    image = table.viewport().grab().toImage()
    restored_colors = {
        image.pixelColor(round(x * ratio), round(y * ratio)).rgba()
        for x in range(region.left(), region.right())
        for y in range(region.top(), region.bottom())
    }
    assert len(restored_colors) > 1  # Stored text is visible again after Cancel.
    connector.set_surface_data.assert_not_called()


def test_hover_preserves_real_document_revision_selection_and_job_count(
    qapp, highlighting_connector
):
    connector = highlighting_connector
    editor = LensEditor(connector)
    editor.resize(850, 400)
    editor.show()
    QTest.qWaitForWindowExposed(editor)
    table = editor.tableWidget
    try:
        table.selectRow(1)
        selected = editor.interaction_state.selected_surfaces
        token = connector.document_state.token
        serial = connector.calculation_jobs._serial
        modified = connector.is_modified()
        for row, column in ((1, 2), (2, 3), (2, 0), (1, 1)):
            move_to_cell(editor, row, column)
            assert editor.interaction_state.selected_surfaces == selected
            assert connector.document_state.token == token
            assert connector.calculation_jobs._serial == serial
            assert connector.is_modified() == modified
        move_pointer(editor.btnAddSurface, QPoint(5, 5))
        assert editor.hover_presentation.hovered_row() == -1
    finally:
        editor.close()
        editor.deleteLater()


def test_theme_change_with_open_editor_preserves_the_draft(themed_editor, qapp):
    editor, connector, theme = themed_editor
    table = editor.tableWidget
    table.setCurrentCell(1, 1)
    table.edit(table.model().index(1, 1))
    qapp.processEvents()
    active = next(iter(editor.hover_presentation._editors.values()))[0]
    active.setText("Uncommitted draft")
    move_pointer(active, QPoint(20, 8))
    tint = editor.hover_presentation.tint(1, 2)
    path = Path(__file__).parents[2] / "optiland_gui/resources/styles"
    other = "light" if theme == "dark" else "dark"
    qapp.setStyleSheet((path / f"{other}_theme.qss").read_text(encoding="utf-8"))
    qapp.processEvents()
    assert active.text() == "Uncommitted draft"
    assert active.hasFocus()
    assert editor.hover_presentation.tint(1, 2) != tint
    assert active.property("ldeHoverBackground") is True
    connector.set_surface_data.assert_not_called()
