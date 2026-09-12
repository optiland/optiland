"""Canonical notifications preserve editor state and avoid duplicate rebuilds."""

from __future__ import annotations

from unittest.mock import MagicMock

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLineEdit

from optiland_gui.lens_editor import LensEditor
from optiland_gui.optiland_connector import OptilandConnector
from optiland_gui.system_properties_panel import SystemPropertiesPanel


def make_editor(minimal_optic):
    connector = OptilandConnector()
    connector.load_optic_from_object(minimal_optic)
    editor = LensEditor(connector)
    editor.resize(900, 500)
    editor.show()
    QTest.qWait(10)
    return connector, editor


def test_polarization_invalidates_rays_without_reading_any_lens_cells(
    qapp, minimal_optic, monkeypatch
):
    connector, editor = make_editor(minimal_optic)
    props = SystemPropertiesPanel(connector)
    changes = []
    connector.document_state.committed.connect(changes.append)
    original = connector.document_state.token
    try:
        with monkeypatch.context() as scope:
            scope.setattr(
                connector,
                "get_surface_data",
                MagicMock(
                    side_effect=AssertionError("Polarization queried a lens cell")
                ),
            )
            connector.set_polarization_state("unpolarized")
        assert len(changes) == 1
        assert changes[0].categories == {"polarization"}
        assert changes[0].affects_optics
        assert connector.document_state.token.revision == original.revision + 1
        assert props.polarizationEditor.cmbMode.currentText() == "Unpolarized"
        queried = MagicMock(wraps=connector.get_surface_data)
        monkeypatch.setattr(connector, "get_surface_data", queried)
        with connector.change_transaction():
            connector.set_polarization_state("ignore")
            connector.set_surface_data(1, connector.COL_COMMENT, "Changed comment")
        assert (
            editor.tableWidget.item(1, connector.COL_COMMENT).text()
            == "Changed comment"
        )
        assert props.polarizationEditor.cmbMode.currentText() == "Ignore"
        assert len(queried.call_args_list) <= 2
        assert all(
            call.args == (1, connector.COL_COMMENT) for call in queried.call_args_list
        )
    finally:
        props.close()
        props.deleteLater()
        editor.close()
        editor.deleteLater()


def test_local_update_keeps_type_widget_selection_and_scroll(
    qapp, minimal_optic, monkeypatch
):
    connector, editor = make_editor(minimal_optic)
    try:
        table = editor.tableWidget
        table.setCurrentCell(1, connector.COL_RADIUS)
        original_widget = table.cellWidget(1, connector.COL_TYPE)
        rebuild = MagicMock(
            side_effect=AssertionError("Local cell edit rebuilt the table")
        )
        monkeypatch.setattr(editor, "full_refresh_from_optic", rebuild)
        connector.set_surface_data(1, connector.COL_RADIUS, "75")
        assert table.cellWidget(1, connector.COL_TYPE) is original_widget
        assert table.currentRow() == 1
        assert table.selectionModel().selectedRows()[0].row() == 1
        assert float(table.item(1, connector.COL_RADIUS).text()) == 75
        rebuild.assert_not_called()
    finally:
        editor.close()
        editor.deleteLater()


def test_structural_update_preserves_selected_surface_and_expanded_owner(
    qapp, minimal_optic
):
    connector, editor = make_editor(minimal_optic)
    try:
        surfaces = tuple(connector.get_optic().surfaces)
        editor.toggle_properties_widget(1)
        editor.tableWidget.setCurrentCell(3, connector.COL_RADIUS)
        connector.add_surface(index=1)
        assert connector.get_optic().surfaces[2] is surfaces[1]
        assert connector.get_optic().surfaces[3] is surfaces[2]
        assert editor.open_prop_source_row == 2
        assert editor.tableWidget.currentRow() == 4
        assert editor.tableWidget.selectionModel().selectedRows()[0].row() == 4
    finally:
        editor.close()
        editor.deleteLater()


def test_replacement_rebuilds_once_despite_public_loaded_and_changed(
    qapp, minimal_optic, monkeypatch
):
    connector, editor = make_editor(minimal_optic)
    rebuild = MagicMock(wraps=editor.full_refresh_from_optic)
    monkeypatch.setattr(editor, "full_refresh_from_optic", rebuild)
    invalidations, committed = [], []
    connector.document_state.changed.connect(invalidations.append)
    connector.document_state.committed.connect(committed.append)
    try:
        connector.new_system()
        assert rebuild.call_count == 1
        assert len(invalidations) == len(committed) == 1
        assert committed[0].categories == {"replacement"}
    finally:
        editor.close()
        editor.deleteLater()


def test_metadata_refresh_preserves_active_cell_draft(qapp, minimal_optic):
    connector, editor = make_editor(minimal_optic)
    try:
        table = editor.tableWidget
        table.setCurrentCell(1, connector.COL_COMMENT)
        table.editItem(table.item(1, connector.COL_COMMENT))
        QTest.qWait(10)
        cell_editor = QApplication.focusWidget()
        assert isinstance(cell_editor, QLineEdit)
        cell_editor.setText("Uncommitted user draft")
        connector.get_optic().surfaces[1].comment = "External label"
        connector.notify_change(
            "metadata", surface_indices=(1,), columns=(connector.COL_COMMENT,)
        )
        assert QApplication.focusWidget() is cell_editor
        assert cell_editor.text() == "Uncommitted user draft"
        QTest.keyClick(cell_editor, Qt.Key_Return)
        QTest.qWait(10)
        assert connector.get_optic().surfaces[1].comment == "Uncommitted user draft"
    finally:
        editor.close()
        editor.deleteLater()


def test_properties_have_one_refresh_owner_and_ignore_surface_labels(qapp, monkeypatch):
    connector = OptilandConnector()
    panel = SystemPropertiesPanel(connector)
    refresh = MagicMock(wraps=panel.load_properties)
    monkeypatch.setattr(panel, "load_properties", refresh)
    connector.notify_change("metadata", surface_indices=(1,), columns=(1,))
    assert not refresh.called
    connector.notify_change("replacement")
    assert refresh.call_count == 1
    panel.close()


def test_pickup_changes_refresh_derived_rows_without_rebuilding(
    qapp, minimal_optic, monkeypatch
):
    connector, editor = make_editor(minimal_optic)
    try:
        optic = connector.get_optic()
        optic.pickups.add(1, "radius", 2, scale=-1)
        rebuild = MagicMock(
            side_effect=AssertionError("Pickup data update rebuilt widgets")
        )
        monkeypatch.setattr(editor, "full_refresh_from_optic", rebuild)
        connector.set_surface_data(1, connector.COL_RADIUS, "65")
        assert float(editor.tableWidget.item(2, connector.COL_RADIUS).text()) == -65
        rebuild.assert_not_called()
    finally:
        editor.close()
        editor.deleteLater()


def test_removing_expanded_surface_retains_the_selected_surviving_surface(
    qapp, minimal_optic
):
    connector, editor = make_editor(minimal_optic)
    try:
        surviving = connector.get_optic().surfaces[2]
        editor.toggle_properties_widget(1)
        editor.tableWidget.setCurrentCell(3, connector.COL_RADIUS)
        editor.remove_surface_handler(1)
        assert editor.open_prop_source_row == -1
        assert connector.get_optic().surfaces[1] is surviving
        assert editor.tableWidget.currentRow() == 1
        assert editor.tableWidget.selectionModel().selectedRows()[0].row() == 1
    finally:
        editor.close()
        editor.deleteLater()


def test_stop_change_updates_both_type_labels_and_same_stop_is_noop(
    qapp, minimal_optic
):
    connector, editor = make_editor(minimal_optic)
    try:
        table = editor.tableWidget
        original = [table.cellWidget(row, 0) for row in (1, 2)]
        connector.set_stop_surface(1)
        for row, widget in zip((1, 2), original, strict=True):
            assert table.cellWidget(row, 0) is widget
            assert (
                widget.type_edit.text()
                == connector.get_surface_type_info(row)["display_text"]
            )
        token = connector.document_state.token
        connector.set_stop_surface(1)
        assert connector.document_state.token == token
    finally:
        editor.close()
        editor.deleteLater()
