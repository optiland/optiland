"""Independent properties panels retain ownership, drafts and menu state."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMenu

from optiland_gui.lens_editor import LensEditor, SurfacePropertiesWidget
from optiland_gui.optiland_connector import OptilandConnector


@pytest.fixture
def editor(qapp, minimal_optic):
    connector = OptilandConnector()
    connector.load_optic_from_object(minimal_optic)
    connector.set_surface_type(1, "biconic")
    connector.set_surface_type(2, "biconic")
    connector._undo_redo_manager.clear_stacks()
    widget = LensEditor(connector)
    widget.resize(1000, 950)
    widget.show()
    qapp.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    connector.calculation_jobs.shutdown()
    connector.deleteLater()
    qapp.processEvents()


def panel(editor, owner):
    widget = editor.tableWidget.cellWidget(
        editor.map_surface_index_to_ui_row(owner) + 1, 0
    )
    assert isinstance(widget, SurfacePropertiesWidget)
    return widget


def menu_action(editor, owner, monkeypatch):
    actions = []

    class InspectMenu(QMenu):
        def exec(self, position):
            actions.extend(
                action
                for action in self.actions()
                if action.text() == "Toggle Surface Properties"
            )

    monkeypatch.setattr("optiland_gui.lens_editor.QMenu", InspectMenu)
    table = editor.tableWidget
    row = editor.map_surface_index_to_ui_row(owner)
    table.scrollToItem(table.item(row, 1))
    position = table.visualItemRect(table.item(row, 1)).center()
    editor.show_context_menu(position)
    assert len(actions) == 1
    return actions[0]


def test_close_keeps_other_panels_drafts_selection_and_optical_state(
    editor, monkeypatch
):
    connector = editor.connector
    token = connector.document_state.edit_token
    changed = []
    connector.document_state.committed.connect(changed.append)
    monkeypatch.setattr(editor, "load_data", MagicMock(side_effect=AssertionError))
    table = editor.tableWidget
    table.selectRow(2)
    selected = editor.interaction_state.selected_surfaces
    type_widget = table.cellWidget(2, 0)

    editor.toggle_properties_widget(2)
    second = panel(editor, 2)
    next(iter(second.input_widgets.values())).setText("1.234")
    editor.toggle_properties_widget(0)
    editor.toggle_properties_widget(1)
    first = panel(editor, 1)
    first.close_button.click()

    assert editor.open_prop_source_rows == {0, 2}
    assert panel(editor, 2) is second
    assert next(iter(second.input_widgets.values())).text() == "1.234"
    assert table.cellWidget(editor.map_surface_index_to_ui_row(2), 0) is type_widget
    assert editor.interaction_state.selected_surfaces == selected
    assert connector.document_state.edit_token == token
    assert changed == []
    assert not connector._undo_redo_manager.can_undo()
    # Closing an earlier panel has shifted this close button's visual row.
    second.close_button.click()
    assert editor.open_prop_source_rows == {0}
    assert table.rowCount() == 5
    editor.toggle_properties_widget(2)
    assert next(iter(panel(editor, 2).input_widgets.values())).text() != "1.234"


@pytest.mark.parametrize("owner", range(4))
def test_context_toggle_checked_for_its_owner_including_object_and_image(
    editor, monkeypatch, owner
):
    other = (owner + 1) % 4
    editor.toggle_properties_widget(other)
    action = menu_action(editor, owner, monkeypatch)
    assert action.isCheckable() and action.isEnabled() and not action.isChecked()
    action.trigger()
    assert editor.open_prop_source_rows == {other, owner}
    action = menu_action(editor, owner, monkeypatch)
    assert action.isChecked()
    action.trigger()
    assert editor.open_prop_source_rows == {other}
    editor.toggle_properties_widget(owner)
    panel(editor, owner).close_button.click()
    assert not menu_action(editor, owner, monkeypatch).isChecked()


def test_adjacent_panel_mappings_and_edits_after_non_lifo_closing(editor):
    for owner in range(4):
        editor.toggle_properties_widget(owner)
    assert [editor.map_ui_row_to_surface_index(row) for row in range(8)] == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]
    assert [editor.is_properties_row(row) for row in range(8)] == [False, True] * 4
    panel(editor, 1).close_button.click()
    assert [editor.map_ui_row_to_surface_index(row) for row in range(7)] == [
        0,
        0,
        1,
        2,
        2,
        3,
        3,
    ]
    table = editor.tableWidget
    table.item(editor.map_surface_index_to_ui_row(2), 1).setText("Right owner")
    assert editor.connector.get_optic().surfaces[2].comment == "Right owner"
    assert editor.connector.get_optic().surfaces[1].comment != "Right owner"
    for owner in (0, 2, 3):
        row = editor.map_surface_index_to_ui_row(owner) + 1
        assert table.columnSpan(row, 0) == table.columnCount()


def test_structural_changes_retain_each_owner(editor):
    editor.toggle_properties_widget(1)
    editor.toggle_properties_widget(2)
    editor.connector.add_surface(index=1)
    assert editor.open_prop_source_rows == {2, 3}
    assert panel(editor, 2).input_widgets
    assert panel(editor, 3).input_widgets
    editor.connector.remove_surface(2)
    assert editor.open_prop_source_rows == {2}
    assert panel(editor, 2).input_widgets
    panel(editor, 2).close_button.click()
    assert not editor.open_prop_source_rows


def test_type_tool_and_keyboard_close_share_state(editor, qapp):
    editor.tableWidget.cellWidget(1, 0).props_button.click()
    editor.toggle_properties_widget(2)
    close = panel(editor, 2).close_button
    assert "surface 2" in close.accessibleName()
    assert close.toolTip()
    close.setFocus()
    qapp.processEvents()
    QTest.keyClick(close, Qt.Key_Space)
    assert editor.open_prop_source_rows == {1}
    assert editor.tableWidget.hasFocus()
    assert editor.map_ui_row_to_surface_index(editor.tableWidget.currentRow()) == 2
    editor.tableWidget.cellWidget(1, 0).props_button.click()
    assert not editor.open_prop_source_rows


def test_new_document_drops_old_panels(editor):
    editor.toggle_properties_widget(1)
    editor.toggle_properties_widget(2)
    editor.connector.new_system()
    assert not editor.open_prop_source_rows
    assert editor.tableWidget.rowCount() == editor.connector.get_surface_count()
