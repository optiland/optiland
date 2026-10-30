"""Real-process save/open ownership, cancellation and publication boundaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

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
    operations.cancel_pending()
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


def test_loaded_candidate_install_failure_preserves_current_document(
    file_operations, monkeypatch, tmp_path
):
    from optiland_gui.services.job_records import JobResult
    from optiland_gui.services.prepared_optic import PreparedOptic

    connector, operations, _, _, notifications = file_operations
    original = connector.get_optic()
    token = connector.document_state.edit_token
    snapshot = PreparedOptic.capture(original)
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
