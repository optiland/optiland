"""A whole property-table draft is validated before changing the live model."""

from __future__ import annotations

import pytest

from optiland.fields import ObjectHeightField
from optiland_gui.optiland_connector import OptilandConnector
from optiland_gui.system_properties_panel import (
    ApertureEditor,
    FieldsEditor,
    WavelengthsEditor,
)


@pytest.mark.parametrize(
    ("kind", "invalid_text"),
    [
        ("fields", "invalid"),
        ("fields", "nan"),
        ("fields", "inf"),
        ("wavelengths", "invalid"),
        ("wavelengths", "nan"),
        ("wavelengths", "inf"),
        ("wavelengths", "0"),
        ("wavelengths", "-1"),
    ],
)
def test_later_invalid_row_cannot_silently_mutate_earlier_rows(
    qapp, kind, invalid_text
):
    connector = OptilandConnector()
    optic = connector.get_optic()
    if kind == "fields":
        optic.fields.add(y=1)
        editor = FieldsEditor(connector)
        table = editor.tableFields
        apply = editor.apply_table_field_changes

        def read():
            return [(f.x, f.y, f.vx, f.vy) for f in optic.fields]

        valid = "2"
    else:
        optic.wavelengths.add(0.6328, is_primary=False)
        editor = WavelengthsEditor(connector)
        table = editor.tableWavelengths
        apply = editor.apply_table_wavelength_changes

        def read():
            return [wave.value for wave in optic.wavelengths]

        valid = "0.6000"
    try:
        editor.load_data()
        before, token = read(), connector.document_state.edit_token
        table.item(0, 0).setText(valid)
        table.item(1, 0).setText(invalid_text)
        apply()
        assert read() == before
        assert connector.document_state.edit_token == token
        assert not connector.is_modified()
        table.item(0, 0).setText(valid)
        commits = []
        connector.document_state.committed.connect(commits.append)
        apply()
        assert read() != before
        assert connector.document_state.edit_token.revision == token.revision + 1
        assert len(commits) == 1 and commits[0].affects_optics
        assert connector.is_modified()
        apply()
        assert len(commits) == 1
    finally:
        editor.deleteLater()


def test_aperture_and_primary_noop_do_not_create_document_edits(qapp):
    connector = OptilandConnector()
    aperture = ApertureEditor(connector)
    wavelengths = WavelengthsEditor(connector)
    aperture.load_data()
    wavelengths.load_data()
    wavelengths.tableWavelengths.setCurrentCell(0, 0)
    token = connector.document_state.edit_token
    aperture.apply_aperture_changes()
    wavelengths.set_primary_wavelength()
    assert connector.document_state.edit_token == token
    assert not connector.is_modified()
    aperture.deleteLater()
    wavelengths.deleteLater()


@pytest.mark.parametrize("kind", ["fields", "wavelengths"])
def test_stale_property_table_reloads_without_committing_a_draft(qapp, kind):
    connector = OptilandConnector()
    optic = connector.get_optic()
    if kind == "fields":
        editor = FieldsEditor(connector)
        table = editor.tableFields
        apply = editor.apply_table_field_changes

        def values():
            return [(f.x, f.y, f.vx, f.vy) for f in optic.fields]
    else:
        editor = WavelengthsEditor(connector)
        table = editor.tableWavelengths
        apply = editor.apply_table_wavelength_changes

        def values():
            return [wave.value for wave in optic.wavelengths]

    try:
        editor.load_data()
        original = values()
        token = connector.document_state.edit_token
        table.item(0, 0).setText("2")
        table.insertRow(table.rowCount())
        apply()
        assert values() == original
        assert table.rowCount() == len(original)
        assert connector.document_state.edit_token == token
        assert not connector.is_modified()
    finally:
        editor.deleteLater()


def test_aperture_and_field_type_commit_once_and_repeated_apply_is_noop(qapp):
    connector = OptilandConnector()
    aperture = ApertureEditor(connector)
    fields = FieldsEditor(connector)
    commits = []
    connector.document_state.committed.connect(commits.append)
    try:
        aperture.load_data()
        fields.load_data()
        token = connector.document_state.token
        aperture.spnApertureValue.setValue(12)
        assert connector.get_optic().aperture.value == 12
        assert len(commits) == 1 and commits[0].affects_optics
        aperture.apply_aperture_changes()
        assert len(commits) == 1
        fields.cmbFieldType.setCurrentIndex(
            fields.cmbFieldType.findData("object_height")
        )
        assert len(commits) == 2 and commits[1].affects_optics
        assert isinstance(
            connector.get_optic().fields.field_definition, ObjectHeightField
        )
        fields.apply_field_type_change()
        assert len(commits) == 2
        assert connector.document_state.token.revision == token.revision + 2
        assert connector.is_modified()
    finally:
        aperture.deleteLater()
        fields.deleteLater()
