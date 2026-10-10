"""Production-worker history preparation and atomic acceptance gates."""

from __future__ import annotations

from dataclasses import replace

import pytest
from PySide6.QtWidgets import QMainWindow

from optiland_gui.action_manager import ActionManager
from optiland_gui.optiland_connector import OptilandConnector
from optiland_gui.services.job_records import BackendConfig
from optiland_gui.services.prepared_optic import PreparedOptic
from tests.gui.test_calculation_jobs import wait_for


def test_history_queue_failure_preserves_model_and_stacks(
    history_connector, monkeypatch
):
    connector = history_connector
    original = connector.get_optic()
    manager = connector._undo_redo_manager
    revision, target = manager.revision, manager.peek("undo")
    outcomes = []
    connector.history.finished.connect(lambda *args: outcomes.append(args))

    def reject(*args, **kwargs):
        raise RuntimeError("Calculation queue is full")

    monkeypatch.setattr(connector.calculation_jobs, "submit", reject)
    assert connector.request_undo() is None
    assert not connector.history.busy
    assert connector.get_optic() is original
    assert manager.revision == revision and manager.peek("undo") is target
    assert outcomes == [("failed", "Calculation queue is full")]


def test_invalid_prepared_history_result_cannot_advance_stacks(
    history_connector, monkeypatch
):
    from optiland_gui.services.job_records import JobResult

    connector = history_connector
    original = connector.get_optic()
    manager = connector._undo_redo_manager
    revision, target = manager.revision, manager.peek("undo")
    monkeypatch.setattr(connector.calculation_jobs, "_dispatch", lambda: None)
    outcomes = []
    connector.history.finished.connect(lambda *args: outcomes.append(args))
    request = connector.request_undo()
    connector.history._finished(JobResult(request, "succeeded", None, current=True))
    assert not connector.history.busy
    assert connector.get_optic() is original
    assert manager.revision == revision and manager.peek("undo") is target
    assert outcomes == [("failed", "Invalid prepared history result.")]


@pytest.fixture
def history_connector(qapp, minimal_optic):
    connector = OptilandConnector()
    connector.load_optic_from_object(minimal_optic)
    connector._undo_redo_manager.clear_stacks()
    connector.set_surface_data(1, connector.COL_COMMENT, "New comment")
    yield connector
    connector.calculation_jobs.shutdown()
    wait_for(qapp, lambda: connector.calculation_jobs._process is None)
    connector.deleteLater()


def test_real_worker_undo_redo_keeps_public_dictionary_history(qapp, history_connector):
    c = history_connector
    old = c.get_optic()
    statuses = []
    c.history.finished.connect(lambda status, message: statuses.append(status))
    request = c.request_undo()
    assert request and c.history.busy
    assert c.get_optic() is old
    assert c._undo_redo_manager.can_undo()
    assert not c._undo_redo_manager.can_redo()
    wait_for(qapp, lambda: not c.history.busy)
    assert statuses == ["succeeded"]
    assert c.get_optic() is not old
    assert c.get_optic().surfaces[1].comment != "New comment"
    assert isinstance(c._undo_redo_manager.peek("redo"), dict)
    assert c.is_modified()
    c.request_redo()
    wait_for(qapp, lambda: not c.history.busy)
    assert statuses == ["succeeded", "succeeded"]
    assert c.get_optic().surfaces[1].comment == "New comment"


@pytest.mark.parametrize("intervention", ["metadata", "history", "replacement"])
def test_intervening_edit_or_history_change_revokes_acceptance(
    qapp, history_connector, intervention
):
    c = history_connector
    manager = c._undo_redo_manager
    old = c.get_optic()
    target = manager.peek("undo")
    optical_token = c.document_state.token
    c.request_undo()
    if intervention == "metadata":
        old.surfaces[1].comment = "Intervening comment"
        c.notify_change("metadata", surface_indices=(1,), columns=(c.COL_COMMENT,))
        assert c.document_state.token == optical_token
    elif intervention == "history":
        manager.add_state(c._capture_optic_state())
    else:
        c.notify_change("replacement")
    revision = manager.revision
    wait_for(qapp, lambda: not c.history.busy)
    assert c.get_optic() is old
    assert manager.revision == revision
    assert not manager.can_redo()
    if intervention != "history":
        assert manager.peek("undo") is target


def test_cancel_and_supersede_have_one_accepted_stack_move(qapp, history_connector):
    c = history_connector
    manager = c._undo_redo_manager
    revision = manager.revision
    old = c.get_optic()
    c.request_undo()
    c.history.cancel()
    wait_for(qapp, lambda: not c.history.busy)
    assert c.get_optic() is old and manager.revision == revision
    first = c.request_undo()
    second = c.request_undo()
    assert first.job_id != second.job_id
    wait_for(qapp, lambda: not c.history.busy)
    assert manager.revision == revision + 1
    assert manager.can_redo()


def test_shutdown_discards_active_history_without_advancing_stacks(
    qapp, history_connector
):
    c = history_connector
    old = c.get_optic()
    revision = c._undo_redo_manager.revision
    c.request_undo()
    wait_for(qapp, lambda: c.calculation_jobs.active_request is not None)
    c.calculation_jobs.shutdown()
    wait_for(qapp, lambda: c.calculation_jobs._process is None)
    assert not c.history.busy
    assert c.get_optic() is old
    assert c._undo_redo_manager.revision == revision


def test_bad_candidate_leaves_model_and_stacks_unchanged_then_recovers(
    qapp, history_connector
):
    c = history_connector
    manager = c._undo_redo_manager
    good = manager.peek("undo")
    manager.add_state({"invalid": "prescription"})
    revision = manager.revision
    bad = manager.peek("undo")
    old = c.get_optic()
    outcomes = []
    c.history.finished.connect(lambda status, text: outcomes.append(status))
    c.request_undo()
    wait_for(qapp, lambda: not c.history.busy)
    assert outcomes == ["failed"]
    assert manager.peek("undo") is bad
    assert c.get_optic() is old and manager.revision == revision
    with pytest.raises(KeyError):
        c.undo()
    assert manager.peek("undo") is bad
    assert c.get_optic() is old and manager.revision == revision
    manager.clear_stacks()
    manager.add_state(good)
    c.request_undo()
    wait_for(qapp, lambda: not c.history.busy)
    assert outcomes == ["failed", "succeeded"]


def test_prepared_model_preserves_types_and_rejects_backend_mismatch(minimal_optic):
    prepared = PreparedOptic.capture(minimal_optic)
    restored = prepared.restore()
    assert restored is not minimal_optic
    assert restored.surfaces[1] is not minimal_optic.surfaces[1]
    assert restored.to_dict() == minimal_optic.to_dict()
    with pytest.raises(ValueError, match="backend changed"):
        replace(prepared, backend=BackendConfig("numpy", precision=32)).restore()
    with pytest.raises(ValueError, match="component types"):
        replace(prepared, component_types=()).restore()


def test_actions_offer_progress_cancel_and_restore_availability(
    qapp, history_connector
):
    c = history_connector
    window = QMainWindow()
    actions = ActionManager(window, c)
    actions._create_edit_actions()
    actions.actions["undo"].trigger()
    assert c.history.busy
    assert not actions.actions["undo"].isEnabled()
    assert not actions.history_progress.isHidden()
    assert not actions.history_cancel.isHidden()
    assert "Preparing undo" in window.statusBar().currentMessage()
    actions.history_cancel.click()
    wait_for(qapp, lambda: not c.history.busy)
    assert actions.actions["undo"].isEnabled()
    assert actions.history_progress.isHidden()
    window.deleteLater()
