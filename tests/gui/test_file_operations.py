"""Real-process save/open ownership, cancellation and publication boundaries."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
from PySide6.QtCore import QTimer

from optiland.optic import Optic
from optiland_gui.optiland_connector import OptilandConnector
from optiland_gui.services.file_service import SpecialFloatEncoder, json_inf_nan_hook

from .test_calculation_jobs import wait_for


@pytest.fixture
def file_operations(qapp, minimal_optic):
    connector = OptilandConnector()
    connector._optic = minimal_optic
    connector.set_modified(True)
    operations = connector.file_operations
    jobs = operations.jobs
    jobs._cancel_grace_ms = 40
    progress, results, notifications = [], [], []
    jobs.progress.connect(lambda request, message: progress.append(message["stage"]))
    jobs.finished.connect(results.append)
    operations.service._toast = lambda text, kind: notifications.append((text, kind))
    yield connector, operations, progress, results, notifications
    operations.begin_close()
    wait_for(qapp, lambda: not operations.busy, timeout=30)
    jobs.shutdown()
    wait_for(qapp, lambda: jobs._process is None)


def stall(operations, handler, when="before", delay=0.2):
    operations.jobs._command = [
        sys.executable,
        "-u",
        str(Path(__file__).with_name("file_worker_fixture.py")),
        handler,
        when,
        str(delay),
    ]


def read_optic(path):
    return Optic.from_dict(json.loads(path.read_text(), object_hook=json_inf_nan_hook))


def edit_comment(connector, text):
    connector.get_optic().surfaces[1].comment = text
    connector.set_modified(True)
    connector.notify_change("metadata", surface_indices=(1,), columns=(1,))


def test_close_revision_change_aborts_and_explicit_cancel_resumes_requests(
    file_operations, monkeypatch, tmp_path
):
    connector, operations, _, _, notifications = file_operations
    dispatch = operations.jobs._dispatch
    monkeypatch.setattr(operations.jobs, "_dispatch", lambda: None)
    operations.request_output(tmp_path / "saved.json")
    assert operations.has_pending_writes
    operations.begin_close()
    assert operations.request_load(tmp_path / "other.json") is None
    assert operations.request_output(tmp_path / "other.json") is None
    edit_comment(connector, "edit while draining")
    assert not operations.confirm_close_revision()
    assert any("Close cancelled" in text for text, _ in notifications)
    operations.cancel_close()
    assert operations.request_load(tmp_path / "other.json") is not None
    monkeypatch.setattr(operations.jobs, "_dispatch", dispatch)
    operations.cancel_pending()


@pytest.mark.parametrize("failure", ["capture", "queue"])
def test_output_cannot_start_preserves_document_and_destination(
    file_operations, monkeypatch, tmp_path, failure
):
    from optiland_gui.services.job_records import OpticSnapshot

    connector, operations, _, _, notifications = file_operations
    original = connector.get_optic()
    path = tmp_path / "saved.json"
    path.write_bytes(b"previous valid contents")

    def reject(*args, **kwargs):
        raise RuntimeError("simulated request failure")

    if failure == "capture":
        monkeypatch.setattr(OpticSnapshot, "capture", reject)
    else:
        monkeypatch.setattr(operations.jobs, "submit", reject)
    assert operations.request_output(path) is None
    assert not operations.busy
    assert connector.get_optic() is original and connector.is_modified()
    assert path.read_bytes() == b"previous valid contents"
    assert any("simulated request failure" in text for text, _ in notifications)


def test_loaded_candidate_install_failure_preserves_current_document(
    file_operations, monkeypatch, tmp_path
):
    from optiland_gui.services.job_records import JobResult, OpticSnapshot

    connector, operations, _, _, notifications = file_operations
    original = connector.get_optic()
    token = connector.document_state.edit_token
    snapshot = OpticSnapshot.capture(original)
    monkeypatch.setattr(operations.jobs, "_dispatch", lambda: None)
    request = operations.request_load(tmp_path / "candidate.json")

    def reject(*args, **kwargs):
        raise ValueError("candidate installation failed")

    monkeypatch.setattr(operations.service, "_publish_candidate", reject)
    operations._finished(JobResult(request, "succeeded", snapshot, current=True))
    assert not operations.busy
    assert connector.get_optic() is original and connector.is_modified()
    assert connector.document_state.edit_token == token
    assert operations.service.get_current_filepath() is None
    assert any("candidate installation failed" in text for text, _ in notifications)


def test_worker_staging_collision_keeps_other_writers_file(
    qapp, tmp_path, file_operations
):
    connector, operations, progress, results, notifications = file_operations
    request = operations.request_output(tmp_path / "saved.json")
    stage = Path(request.context.staged_path)
    stage.write_bytes(b"another writer")
    wait_for(qapp, lambda: not operations.busy)
    assert stage.read_bytes() == b"another writer"
    assert not (tmp_path / "saved.json").exists()
    assert results[0].status == "failed"
    assert connector.is_modified()


def test_cancellation_before_creation_keeps_an_unowned_staging_file(
    qapp, tmp_path, file_operations
):
    connector, operations, progress, results, notifications = file_operations
    request = operations.request_output(tmp_path / "saved.json")
    stage = Path(request.context.staged_path)
    stage.write_bytes(b"another writer")
    operations.cancel_pending()
    wait_for(qapp, lambda: not operations.busy)
    assert stage.read_bytes() == b"another writer"
    assert connector.is_modified()


def test_staging_receipt_survives_document_change_and_replaced_file_is_preserved(
    qapp, tmp_path, file_operations
):
    connector, operations, progress, results, notifications = file_operations
    stall(operations, "prepare_output", "after", 10)
    transaction_stages = []
    operations.jobs.transaction_progress.connect(
        lambda request, message: transaction_stages.append(message["stage"])
    )
    request = operations.request_output(tmp_path / "saved.json")
    connector.set_surface_data(1, connector.COL_RADIUS, "55")
    wait_for(qapp, lambda: "Fixture completed prepare_output" in transaction_stages)
    assert request.context.staged_identity is not None
    stage = Path(request.context.staged_path)
    replacement = tmp_path / "replacement.tmp"
    replacement.write_bytes(b"replacement belonging to another writer")
    replacement.replace(stage)
    operations.cancel_pending()
    wait_for(qapp, lambda: not operations.busy)
    assert stage.read_bytes() == b"replacement belonging to another writer"
    assert connector.is_modified()
    assert any("cleanup failed" in text for text, severity in notifications)


@pytest.mark.parametrize("file_format,extension", [("zemax", "zmx"), ("codev", "seq")])
def test_real_worker_exports_and_imports_supported_foreign_formats(
    qapp, tmp_path, file_operations, file_format, extension
):
    from optiland.fileio import load_codev_file, load_zemax_file

    connector, operations, progress, results, notifications = file_operations
    path = tmp_path / f"exported.{extension}"
    original = connector.get_optic()
    operations.request_output(path, file_format)
    wait_for(qapp, lambda: not operations.busy)
    assert path.is_file(), notifications
    restored = (load_zemax_file if file_format == "zemax" else load_codev_file)(
        str(path)
    )
    assert float(restored.surfaces[1].geometry.radius) == 50
    assert connector.get_optic() is original
    assert connector.is_modified()
    assert operations.service.get_current_filepath() is None
    assert not list(tmp_path.glob(".optiland-*.tmp"))
    operations.request_load(path, file_format)
    wait_for(qapp, lambda: not operations.busy)
    assert connector.get_optic() is not original, notifications
    assert float(connector.get_optic().surfaces[1].geometry.radius) == 50
    assert connector.is_modified()
    assert operations.service.get_current_filepath() is None


def test_publication_retries_a_full_queue_without_losing_the_staged_save(
    qapp, tmp_path, file_operations
):
    connector, operations, progress, results, notifications = file_operations
    jobs = operations.jobs

    def fill_queue(request, state):
        if state != "succeeded" or not request.handler.endswith(":prepare_output"):
            return
        for index in range(jobs.max_pending):
            jobs.submit(
                f"queue-filler-{index}",
                "optiland_gui.services.file_tasks:cleanup_output",
                None,
                {
                    "staged_path": str(tmp_path / f".optiland-unused-{index}.tmp"),
                    "path": str(tmp_path / "unused.json"),
                    "staged_identity": None,
                },
                replace=False,
            )

    jobs.state_changed.connect(fill_queue)
    path = tmp_path / "saved.json"
    operations.request_output(path)
    wait_for(qapp, lambda: not operations.busy, timeout=30)
    assert read_optic(path).surfaces[1].geometry.radius == 50
    assert not connector.is_modified()
    assert not list(tmp_path.glob(".optiland-*.tmp"))


def test_save_uses_captured_document_and_keeps_later_edits_dirty(
    qapp, tmp_path, file_operations
):
    connector, operations, progress, results, notifications = file_operations
    edit_comment(connector, "saved version")
    stall(operations, "prepare_output")
    path = tmp_path / "saved.json"
    operations.request_output(path)
    edit_comment(connector, "new unsaved edit")
    wait_for(qapp, lambda: not operations.busy)
    assert read_optic(path).surfaces[1].comment == "saved version"
    assert connector.get_optic().surfaces[1].comment == "new unsaved edit"
    assert connector.is_modified()
    assert operations.service.get_current_filepath() == str(path)
    assert not list(tmp_path.glob(".optiland-*.tmp"))


def test_cancel_after_staging_never_replaces_destination(
    qapp, tmp_path, file_operations
):
    connector, operations, progress, results, notifications = file_operations
    stall(operations, "prepare_output", "after", 10)
    path = tmp_path / "saved.json"
    path.write_bytes(b"previous contents")
    operations.request_output(path)
    wait_for(qapp, lambda: "Fixture completed prepare_output" in progress)
    operations.cancel_pending()
    wait_for(qapp, lambda: not operations.busy)
    assert path.read_bytes() == b"previous contents"
    assert any(result.status == "cancelled" for result in results)
    assert not list(tmp_path.glob(".optiland-*.tmp"))
    assert connector.is_modified()


def test_new_save_supersedes_pending_save_for_same_destination(
    qapp, tmp_path, file_operations
):
    connector, operations, progress, results, notifications = file_operations
    stall(operations, "prepare_output", delay=0.1)
    path = tmp_path / "saved.json"
    edit_comment(connector, "old")
    operations.request_output(path)
    edit_comment(connector, "latest")
    operations.request_output(path)
    wait_for(qapp, lambda: not operations.busy)
    assert read_optic(path).surfaces[1].comment == "latest"
    assert not connector.is_modified()
    assert not list(tmp_path.glob(".optiland-*.tmp"))


def test_load_conflict_requires_current_explicit_acceptance(
    qapp, tmp_path, file_operations, minimal_optic
):
    connector, operations, progress, results, notifications = file_operations
    candidate = Optic.from_dict(minimal_optic.to_dict())
    candidate.surfaces[1].comment = "loaded candidate"
    path = tmp_path / "input.json"
    path.write_text(json.dumps(candidate.to_dict(), cls=SpecialFloatEncoder))
    stall(operations, "load_file")
    original = connector.get_optic()
    conflicts = []
    operations.candidate_conflict.connect(lambda *args: conflicts.append(args))
    operations.request_load(path)
    edit_comment(connector, "edit while loading")
    wait_for(qapp, lambda: not operations.busy)
    assert connector.get_optic() is original
    assert len(conflicts) == 1
    accepted = operations.resolve_candidate(
        conflicts[0][0], True, connector.document_state.edit_token
    )
    assert accepted
    assert connector.get_optic().surfaces[1].comment == "loaded candidate"
    assert not connector.is_modified()


def test_invalid_file_leaves_existing_document_and_history(
    qapp, tmp_path, file_operations
):
    connector, operations, progress, results, notifications = file_operations
    original = connector.get_optic()
    token = connector.document_state.edit_token
    connector._undo_redo_manager.add_state({"existing": "undo entry"})
    path = tmp_path / "invalid.json"
    path.write_text("{broken input")
    operations.request_load(path)
    wait_for(qapp, lambda: not operations.busy)
    assert connector.get_optic() is original
    assert connector.document_state.edit_token == token
    assert connector._undo_redo_manager.can_undo()
    assert connector.is_modified()
    assert any(kind == "error" for _, kind in notifications)


def test_unknown_publication_reconciles_and_close_drains_without_blocking(
    qapp, tmp_path, file_operations
):
    connector, operations, progress, results, notifications = file_operations
    stall(operations, "publish_output", "after", 10)
    operations.jobs._publication_deadline_ms = 500
    path = tmp_path / "saved.json"
    states, ticks, settled = [], [], []
    operations.state_changed.connect(lambda *args: states.append(args))
    operations.settled.connect(lambda: settled.append(True))
    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: ticks.append(time.monotonic()))
    timer.start()
    operations.request_output(path)
    wait_for(qapp, lambda: "Fixture completed publish_output" in progress)
    started = time.monotonic()
    operations.begin_close()
    assert time.monotonic() - started < 0.1
    assert operations.busy and not settled
    assert not states[-1][2]
    wait_for(qapp, lambda: not operations.busy, timeout=30)
    timer.stop()
    assert settled and len(ticks) >= 10
    assert (
        read_optic(path).surfaces.num_surfaces
        == connector.get_optic().surfaces.num_surfaces
    )
    assert any(result.outcome_unknown for result in results)
    assert any(kind == "success" for _, kind in notifications)
    assert not any("cancelled" in text.lower() for text, _ in notifications)
    assert not connector.is_modified()
    assert not list(tmp_path.glob(".optiland-*.tmp"))


def test_unconfirmed_publication_preserves_destination_and_aborts_close(
    qapp, tmp_path, file_operations
):
    connector, operations, progress, results, notifications = file_operations
    path = tmp_path / "saved.json"
    path.write_bytes(b"previous valid contents")
    stall(operations, "publish_output", "before", 10)
    operations.jobs._publication_deadline_ms = 500
    aborted = []
    operations.close_aborted.connect(lambda: aborted.append(True))
    operations.request_output(path)
    wait_for(qapp, lambda: "Fixture entered publish_output" in progress)
    operations.begin_close()
    wait_for(qapp, lambda: not operations.busy, timeout=30)
    assert path.read_bytes() == b"previous valid contents"
    assert connector.is_modified()
    assert aborted == [True]
    assert any("could not be confirmed" in text for text, severity in notifications)
    assert not list(tmp_path.glob(".optiland-*.tmp"))


def test_loaded_candidate_rechecks_edit_token_at_explicit_commit(
    qapp, tmp_path, file_operations, minimal_optic
):
    connector, operations, progress, results, notifications = file_operations
    path = tmp_path / "input.json"
    path.write_text(json.dumps(minimal_optic.to_dict(), cls=SpecialFloatEncoder))
    stall(operations, "load_file")
    original = connector.get_optic()
    operations.request_load(path)
    edit_comment(connector, "first edit")
    wait_for(qapp, lambda: not operations.busy)
    operation, snapshot = operations._candidate
    expected = connector.document_state.edit_token
    edit_comment(connector, "edit while confirmation was displayed")
    assert not operations.resolve_candidate(operation.identifier, True, expected)
    assert connector.get_optic() is original
    assert connector.get_optic().surfaces[1].comment.endswith("displayed")
