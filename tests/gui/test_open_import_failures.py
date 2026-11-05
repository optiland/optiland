"""Pure captures and failure preservation at the public file-service boundary."""

from __future__ import annotations

import pytest

from optiland_gui.optiland_connector import OptilandConnector


@pytest.mark.parametrize(
    "method", ["load", "load_from_object", "import_zemax", "import_codev"]
)
def test_invalid_candidate_keeps_document_history_path_and_modified(
    qapp, minimal_optic, monkeypatch, tmp_path, method
):
    from optiland_gui.services import file_service as module

    connector = OptilandConnector()
    service = connector._file_service
    original = connector.get_optic()
    service._current_filepath = "existing.json"
    connector.set_modified(True)
    connector._undo_redo_manager.add_state({"marker": "existing undo"})
    token = connector.document_state.edit_token
    monkeypatch.setattr(module, "load_zemax_file", lambda path: minimal_optic)
    monkeypatch.setattr(module, "load_codev_file", lambda path: minimal_optic)

    def invalid(*args, **kwargs):
        raise ValueError("candidate validation failed")

    monkeypatch.setattr(connector, "_initialize_optic_structure", invalid)
    argument = (
        minimal_optic if method == "load_from_object" else str(tmp_path / "input.zmx")
    )
    getattr(service, method)(argument)
    assert connector.get_optic() is original
    assert service.get_current_filepath() == "existing.json"
    assert connector.is_modified()
    assert connector._undo_redo_manager.can_undo()
    assert connector.document_state.edit_token == token
