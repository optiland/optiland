"""Real-process save/open ownership, cancellation and publication boundaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from optiland.optic import Optic
from optiland_gui.optiland_connector import OptilandConnector
from optiland_gui.services.file_service import json_inf_nan_hook

from .test_calculation_jobs import wait_for


@pytest.fixture
def file_operations(qapp, minimal_optic):
    connector = OptilandConnector()
    connector._optic = minimal_optic
    connector.set_modified(True)
    operations = connector.output_operations
    jobs = operations.jobs
    jobs._cancel_grace_ms = 40
    progress, results, notifications = [], [], []
    jobs.progress.connect(lambda request, message: progress.append(message["stage"]))
    jobs.finished.connect(results.append)
    operations.service._toast = lambda text, kind: notifications.append((text, kind))
    yield connector, operations, progress, results, notifications
    operations.cancel_pending()
    wait_for(qapp, lambda: not operations.busy, timeout=30)
    jobs.shutdown()
    wait_for(qapp, lambda: jobs._process is None)


def stall(operations, handler, when="before", delay=0.2):
    operations.jobs._command = [
        sys.executable,
        "-u",
        str(Path(__file__).with_name("output_worker_fixture.py")),
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
