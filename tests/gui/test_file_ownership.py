"""Pure captures and failure preservation at the public file-service boundary."""

from __future__ import annotations

import pickle
import threading

import pytest

from optiland_gui.optiland_connector import OptilandConnector


def test_staging_collision_preserves_existing_file(minimal_optic, tmp_path):
    from optiland_gui.services.file_tasks import prepare_output
    from optiland_gui.services.job_records import OpticSnapshot

    stage = tmp_path / ".optiland-collision.tmp"
    stage.write_bytes(b"another operation owns this file")
    with pytest.raises(FileExistsError):
        prepare_output(
            OpticSnapshot.capture(minimal_optic),
            {
                "staged_path": str(stage),
                "path": str(tmp_path / "lens.json"),
                "format": "optiland",
            },
            lambda *args: None,
            threading.Event(),
        )
    assert stage.read_bytes() == b"another operation owns this file"


def test_undo_capture_is_owned_and_does_not_run_updater(
    qapp, minimal_optic, monkeypatch
):
    connector = OptilandConnector()
    connector._optic = minimal_optic
    before = pickle.dumps(minimal_optic.to_dict())
    monkeypatch.setattr(
        minimal_optic.updater, "update", lambda: pytest.fail("capture ran optics")
    )
    data = connector._capture_optic_state()
    assert pickle.dumps(minimal_optic.to_dict()) == before
    minimal_optic.surfaces[1].comment = "Later edit"
    assert pickle.dumps(data) == before


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


def test_failed_save_preserves_existing_file(qapp, tmp_path, monkeypatch):
    import optiland_gui.services.file_service as module

    connector = OptilandConnector()
    destination = tmp_path / "lens.json"
    destination.write_bytes(b"existing good file")

    def interrupted(data, stream, **kwargs):
        stream.write("partial output")
        raise OSError("simulated full disk")

    monkeypatch.setattr(module.json, "dump", interrupted)
    connector._file_service.save(str(destination))
    assert destination.read_bytes() == b"existing good file"
    assert list(tmp_path.iterdir()) == [destination]
