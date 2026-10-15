"""File ownership boundaries also apply to public synchronous entry points."""

from __future__ import annotations

import json
import threading

import pytest

from optiland.fileio import load_codev_file, load_zemax_file, save_zemax_file
from optiland.optic import Optic
from optiland_gui.services.file_service import SpecialFloatEncoder
from optiland_gui.services.file_tasks import load_file
from optiland_gui.services.job_records import BackendConfig, JobResult, OpticSnapshot
from optiland_gui.services.model_initialization import ensure_valid_structure
from optiland_gui.services.prepared_optic import PreparedOptic

from .test_calculation_jobs import wait_for
from .test_file_operations import edit_comment
from .test_file_operations import file_operations as file_operations


def test_sync_native_load_publishes_path_clean_state_and_clears_history(
    file_operations, tmp_path
):
    connector, operations, _, _, _ = file_operations
    original = connector.get_optic()
    path = tmp_path / "native.json"
    path.write_text(json.dumps(original.to_dict(), cls=SpecialFloatEncoder))
    connector.set_surface_data(1, connector.COL_RADIUS, "60")
    assert connector.is_modified() and connector._undo_redo_manager.can_undo()
    connector._file_service.load(str(path))
    assert connector.get_optic() is not original
    assert float(connector.get_optic().surfaces[1].geometry.radius) == 50
    assert connector.get_current_filepath() == str(path)
    assert not connector.is_modified()
    assert not connector._undo_redo_manager.can_undo()
    assert not connector._undo_redo_manager.can_redo()
    assert not operations.busy


def test_sync_load_normalizes_candidate_and_new_document_cancels_pending_load(
    file_operations, monkeypatch, tmp_path
):
    connector, operations, _, _, _ = file_operations
    original = connector.get_optic()
    path = tmp_path / "lens.zmx"
    save_zemax_file(original, path)
    connector._file_service.load(str(path))
    assert connector.get_optic() is not original
    assert connector.get_optic().surfaces.num_surfaces == 4
    assert connector.get_current_filepath() is None
    assert not connector.is_modified()

    monkeypatch.setattr(operations.jobs, "_dispatch", lambda: None)
    operations.request_load(tmp_path / "obsolete.json")
    connector.new_system()
    assert not operations.busy
    assert operations._latest_open is None
    assert connector.get_optic().name == "New Untitled System"


def test_sync_save_refuses_to_overtake_an_owned_async_write(
    file_operations, monkeypatch, tmp_path
):
    connector, operations, _, _, notifications = file_operations
    monkeypatch.setattr(
        connector._file_service,
        "_toast",
        lambda message, severity, **kwargs: notifications.append((message, severity)),
    )
    path = tmp_path / "saved.json"
    path.write_bytes(b"existing contents")
    dispatch = operations.jobs._dispatch
    monkeypatch.setattr(operations.jobs, "_dispatch", lambda: None)
    operations.request_output(path)
    connector._file_service.save(str(path))
    assert path.read_bytes() == b"existing contents"
    assert connector.is_modified() and connector.get_current_filepath() is None
    assert any("asynchronous save" in message for message, _ in notifications)
    monkeypatch.setattr(operations.jobs, "_dispatch", dispatch)
    operations.cancel_pending()


@pytest.mark.parametrize(
    "file_format,extension,loader",
    [("zemax", "zmx", load_zemax_file), ("codev", "seq", load_codev_file)],
)
def test_sync_export_respects_owned_path_and_works_after_release(
    qapp, file_operations, monkeypatch, tmp_path, file_format, extension, loader
):
    connector, operations, _, _, notifications = file_operations
    path = tmp_path / f"export.{extension}"
    path.write_bytes(b"existing contents")
    token = connector.document_state.edit_token
    dispatch = operations.jobs._dispatch
    monkeypatch.setattr(operations.jobs, "_dispatch", lambda: None)
    operations.request_output(path)
    export = getattr(connector, f"export_{file_format}")
    export(str(path))
    assert path.read_bytes() == b"existing contents"
    assert connector.document_state.edit_token == token
    assert connector.is_modified() and connector.get_current_filepath() is None
    assert any(
        "asynchronous save" in message and severity == "error"
        for message, severity in notifications
    )
    monkeypatch.setattr(operations.jobs, "_dispatch", dispatch)
    operations.cancel_pending()
    wait_for(qapp, lambda: not operations.busy)
    export(str(path))
    restored = loader(str(path))
    assert restored.surfaces.num_surfaces == connector.get_optic().surfaces.num_surfaces
    assert float(restored.surfaces[1].geometry.radius) == 50
    assert connector.document_state.edit_token == token
    assert connector.is_modified() and connector.get_current_filepath() is None


def test_object_load_owns_its_snapshot(minimal_optic):
    snapshot = OpticSnapshot.capture(minimal_optic)
    prepared = load_file(
        snapshot,
        {"format": "object", "backend": BackendConfig.capture()},
        lambda *a: None,
        threading.Event(),
    )
    restored = prepared.restore()
    assert type(restored) is Optic
    assert restored is not minimal_optic
    assert restored.wavelengths.primary_index == 0
    restored.surfaces[1].comment = "independent edit"
    assert minimal_optic.surfaces[1].comment != "independent edit"


def test_minimal_structure_repairs_an_existing_wavelength_without_a_primary(
    minimal_optic,
):
    minimal_optic.wavelengths.wavelengths[0].is_primary = False
    with pytest.raises(ValueError, match="No primary wavelength"):
        _ = minimal_optic.wavelengths.primary_index
    ensure_valid_structure(minimal_optic)
    assert minimal_optic.wavelengths.primary_index == 0


@pytest.mark.parametrize("invalidated", ["cancel", "newer", "closing"])
def test_late_successful_load_cannot_replace_a_superseded_document(
    file_operations, monkeypatch, tmp_path, invalidated
):
    connector, operations, _, _, _ = file_operations
    original = connector.get_optic()
    monkeypatch.setattr(operations.jobs, "_dispatch", lambda: None)
    first = operations.request_load(tmp_path / "older.json")
    prepared = PreparedOptic.capture(original)
    # Completion was already en route when the user changed their intent.
    monkeypatch.setattr(operations.jobs, "cancel_target", lambda *a: None)
    if invalidated == "cancel":
        operations.cancel_load()
    elif invalidated == "newer":
        operations.request_load(tmp_path / "newer.json")
    else:
        operations._closing = True
        operations._close_edit_token = connector.document_state.edit_token
    operations._finished(JobResult(first, "succeeded", prepared, current=False))
    assert connector.get_optic() is original
    assert connector.is_modified() and connector.get_current_filepath() is None
    for operation in tuple(operations._operations.values()):
        operations._complete(operation)


def test_discarded_candidate_cannot_be_accepted_later(
    file_operations, monkeypatch, tmp_path
):
    connector, operations, _, _, _ = file_operations
    original = connector.get_optic()
    monkeypatch.setattr(operations.jobs, "_dispatch", lambda: None)
    request = operations.request_load(tmp_path / "candidate.json")
    prepared = PreparedOptic.capture(original)
    edit_comment(connector, "keep this edit")
    operations._finished(JobResult(request, "succeeded", prepared, current=False))
    identifier = request.context.identifier
    token = connector.document_state.edit_token
    assert not operations.resolve_candidate("wrong identifier", True, token)
    assert not operations.resolve_candidate(identifier, False, token)
    assert not operations.resolve_candidate(identifier, True, token)
    assert connector.get_optic() is original
    assert connector.get_optic().surfaces[1].comment == "keep this edit"


def test_failed_open_request_releases_ownership_and_late_retry_does_nothing(
    file_operations, monkeypatch, tmp_path
):
    connector, operations, _, _, notifications = file_operations
    attempted = []

    def reject(*args, **kwargs):
        attempted.append(kwargs["context"])
        raise RuntimeError("Worker could not start")

    monkeypatch.setattr(operations.jobs, "submit", reject)
    assert operations.request_load(tmp_path / "input.json") is None
    assert not operations.busy
    assert operations._queue(attempted[0], "load_file") is None
    assert len(attempted) == 1
    assert any("could not start" in message for message, _ in notifications)


def test_json_encoder_supports_scalar_protocol_without_string_fallback():
    class Scalar:
        def item(self):
            return 2.5

    assert json.loads(json.dumps({"value": Scalar()}, cls=SpecialFloatEncoder)) == {
        "value": 2.5
    }
