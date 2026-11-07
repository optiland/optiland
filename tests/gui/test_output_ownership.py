"""Pure captures and failure preservation at the public file-service boundary."""

from __future__ import annotations

import threading

import pytest

from optiland_gui.optiland_connector import OptilandConnector


def test_staging_collision_preserves_existing_file(minimal_optic, tmp_path):
    from optiland_gui.services.job_records import OpticSnapshot
    from optiland_gui.services.output_tasks import prepare_output

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


def test_sync_save_publishes_a_complete_native_document_and_marks_it_clean(
    qapp, tmp_path, minimal_optic
):
    from .test_file_save_export import read_optic

    connector = OptilandConnector()
    connector._optic = minimal_optic
    connector.set_modified(True)
    path = tmp_path / "saved.json"
    connector._file_service.save(str(path))
    assert read_optic(path).surfaces.num_surfaces == 4
    assert connector.get_current_filepath() == str(path)
    assert not connector.is_modified()
    assert list(tmp_path.iterdir()) == [path]
