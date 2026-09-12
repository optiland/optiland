"""A whole property-table draft is validated before changing the live model."""

from __future__ import annotations

import pytest

from optiland_gui.optiland_connector import OptilandConnector
from optiland_gui.system_properties_panel import (
    ApertureEditor,
    FieldsEditor,
    WavelengthsEditor,
)


@pytest.mark.parametrize("kind", ["fields", "wavelengths"])
def test_later_invalid_row_cannot_silently_mutate_earlier_rows(qapp, kind):
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
        table.item(1, 0).setText("invalid")
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
