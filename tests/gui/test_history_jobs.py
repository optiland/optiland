"""Production-worker history preparation and atomic acceptance gates."""

from __future__ import annotations

import pickle
import threading
from dataclasses import replace

import pytest
from PySide6.QtWidgets import QMainWindow

import optiland.backend as be
from optiland.optic import Optic
from optiland.rays import create_polarization
from optiland_gui.action_manager import ActionManager
from optiland_gui.optiland_connector import OptilandConnector
from optiland_gui.services.job_records import BackendConfig, JobResult
from optiland_gui.services.prepared_optic import PreparedOptic
from optiland_gui.undo_redo_manager import UndoRedoManager
from tests.gui.test_calculation_jobs import wait_for


def test_undo_capture_is_owned_and_does_not_run_updater(history_connector, monkeypatch):
    connector = history_connector
    optic = connector.get_optic()
    before = pickle.dumps(optic.to_dict(), protocol=5)
    monkeypatch.setattr(
        optic.updater, "update", lambda: pytest.fail("capture ran optics")
    )
    data = connector._capture_optic_state()
    assert pickle.dumps(optic.to_dict(), protocol=5) == before
    optic.ray_tracer.ray_aiming_config["capture_test"] = {"nested": [1, 2]}
    assert pickle.dumps(data, protocol=5) == before
    optic.wavelengths.wavelengths.clear()
    connector._capture_optic_state()
    assert not optic.wavelengths.wavelengths


def test_public_synchronous_undo_redo_and_noop(history_connector):
    connector = history_connector
    manager = connector._undo_redo_manager
    connector.undo()
    assert connector.get_optic().surfaces[1].comment != "New comment"
    assert manager.can_redo() and not manager.can_undo()
    assert connector.is_modified()
    connector.redo()
    assert connector.get_optic().surfaces[1].comment == "New comment"
    assert manager.can_undo() and not manager.can_redo()
    revision = manager.revision
    connector.redo()
    assert manager.revision == revision
    manager.clear_stacks()
    assert connector.request_undo() is None
    assert connector.request_redo() is None


def test_public_restore_normalizes_candidate_before_adoption(history_connector):
    connector = history_connector
    state = connector._capture_optic_state()
    state["wavelengths"]["wavelengths"] = []
    original = connector.get_optic()
    connector._restore_optic_state(state)
    assert connector.get_optic() is not original
    assert connector.get_optic().wavelengths.num_wavelengths == 1


@pytest.mark.parametrize("method", ["undo", "_restore_optic_state"])
def test_public_restoration_failure_is_transactional(
    history_connector, monkeypatch, method
):
    connector = history_connector
    original = connector.get_optic()
    manager = connector._undo_redo_manager
    target, revision = manager.peek("undo"), manager.revision
    token = connector.document_state.edit_token

    def reject(candidate, **kwargs):
        assert candidate is not original
        raise ValueError("candidate update failed")

    monkeypatch.setattr(connector, "_initialize_optic_structure", reject)
    with pytest.raises(ValueError, match="candidate update failed"):
        if method == "undo":
            connector.undo()
        else:
            connector._restore_optic_state(target)
    assert connector.get_optic() is original
    assert manager.peek("undo") is target and manager.revision == revision
    assert connector.document_state.edit_token == token


@pytest.mark.parametrize("checkpoint", ["restored", "updated"])
def test_history_task_cancel_checkpoints(minimal_optic, monkeypatch, checkpoint):
    from optiland_gui.services.history_tasks import prepare_history
    from optiland_gui.services.job_records import CalculationCancelled, OpticSnapshot

    snapshot = OpticSnapshot.capture(minimal_optic)
    cancelled = threading.Event()
    updated = []

    def restore(self):
        if checkpoint == "restored":
            cancelled.set()
        return minimal_optic

    def update():
        updated.append(True)
        cancelled.set()

    monkeypatch.setattr(OpticSnapshot, "restore", restore)
    monkeypatch.setattr(minimal_optic.updater, "update", update)
    with pytest.raises(CalculationCancelled):
        prepare_history(snapshot, {}, lambda *args: None, cancelled)
    assert updated == ([] if checkpoint == "restored" else [True])


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


def test_real_worker_undo_redo_keeps_public_dictionary_history(
    qapp, history_connector, monkeypatch
):
    c = history_connector
    old = c.get_optic()
    statuses = []
    c.history.finished.connect(lambda status, message: statuses.append(status))
    request = c.request_undo()
    assert request and c.history.busy
    assert c.get_optic() is old
    assert c._undo_redo_manager.can_undo()
    assert not c._undo_redo_manager.can_redo()

    def reject_reconstruction(*args, **kwargs):
        pytest.fail("History adoption reconstructed optics on the GUI thread")

    monkeypatch.setattr(Optic, "from_dict", reject_reconstruction)
    monkeypatch.setattr(c, "_initialize_optic_structure", reject_reconstruction)
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


@pytest.mark.parametrize("override", ["trace", "trace_generic", "subclass"])
def test_prepared_model_rejects_runtime_tracing_extensions(minimal_optic, override):
    if override == "subclass":

        class CustomOptic(Optic):
            pass

        optic = CustomOptic()
    else:
        optic = minimal_optic
        setattr(optic, override, lambda *args: None)
    with pytest.raises(ValueError, match="owned preparation adapter"):
        PreparedOptic.capture(optic)


@pytest.mark.parametrize("payload", [None, "invalid pickle"])
def test_malformed_prepared_history_preserves_document_and_stacks(
    history_connector, monkeypatch, payload
):
    connector = history_connector
    jobs = connector.calculation_jobs
    monkeypatch.setattr(jobs, "_dispatch", lambda: None)
    original = connector.get_optic()
    manager = connector._undo_redo_manager
    target, revision = manager.peek("undo"), manager.revision
    request = connector.request_undo()
    prepared = PreparedOptic.capture(original)
    data = pickle.dumps(None) if payload is None else payload.encode()
    outcomes = []
    connector.history.finished.connect(lambda *args: outcomes.append(args))
    connector.history._finished(
        JobResult(request, "succeeded", replace(prepared, data=data), current=True)
    )
    assert outcomes[0][0] == "failed"
    assert not connector.history.busy
    assert connector.get_optic() is original
    assert manager.peek("undo") is target and manager.revision == revision
    assert not manager.can_redo()


@pytest.mark.parametrize("intervention", ["hint", "backend", "identity"])
def test_acceptance_rechecks_job_backend_and_stack_identity(
    history_connector, monkeypatch, intervention
):
    connector = history_connector
    jobs = connector.calculation_jobs
    monkeypatch.setattr(jobs, "_dispatch", lambda: None)
    original = connector.get_optic()
    manager = connector._undo_redo_manager
    request = connector.request_undo()
    prepared = PreparedOptic.capture(request.snapshot.restore())
    if intervention == "backend":
        monkeypatch.setattr(
            BackendConfig,
            "capture",
            classmethod(lambda cls: BackendConfig("numpy", precision=32)),
        )
    elif intervention == "identity":
        manager._undo_stack[-1] = pickle.loads(pickle.dumps(manager.peek("undo")))
    target, revision = manager.peek("undo"), manager.revision
    outcomes = []
    connector.history.finished.connect(lambda *args: outcomes.append(args))
    connector.history._finished(
        JobResult(request, "succeeded", prepared, current=intervention != "hint")
    )
    assert outcomes[0][0] == "stale"
    assert connector.get_optic() is original
    assert manager.peek("undo") is target and manager.revision == revision
    assert not manager.can_redo()


def test_other_job_result_does_not_clear_history_request(
    history_connector, monkeypatch
):
    connector = history_connector
    monkeypatch.setattr(connector.calculation_jobs, "_dispatch", lambda: None)
    request = connector.request_undo()
    connector.history._finished(
        JobResult(replace(request, job_id=request.job_id + 1), "failed", error="other")
    )
    assert connector.history.request is request and connector.history.busy


def test_manager_public_transfers_and_empty_moves(qapp):
    manager = UndoRedoManager()
    availability = []
    manager.undoStackAvailabilityChanged.connect(
        lambda value: availability.append(("undo", value))
    )
    manager.redoStackAvailabilityChanged.connect(
        lambda value: availability.append(("redo", value))
    )
    original, edited = {"name": "original"}, {"name": "edited"}
    assert manager.move("undo", edited) is None
    manager.add_state(original)
    revision = manager.revision
    availability.clear()
    assert manager.undo(edited) is original
    assert availability == [("undo", False), ("redo", True)]
    assert manager.revision == revision + 1
    assert manager.redo(original) is edited
    assert manager.revision == revision + 2
    assert manager.peek("undo") is original
    assert manager.redo(original) is None


def test_polarized_history_worker_preserves_backend_without_core_json_fix(
    qapp, set_test_backend, minimal_optic
):
    connector = OptilandConnector()
    try:
        minimal_optic.polarization = create_polarization("RCP")
        connector.load_optic_from_object(minimal_optic)
        connector._undo_redo_manager.clear_stacks()
        connector.set_surface_data(1, connector.COL_COMMENT, "Edited")
        backend = BackendConfig.capture()
        outcomes = []
        connector.history.finished.connect(
            lambda status, message: outcomes.append(status)
        )
        connector.request_undo()
        wait_for(qapp, lambda: not connector.history.busy)
        assert outcomes == ["succeeded"]
        assert BackendConfig.capture() == backend
        restored = connector.get_optic()
        assert restored.surfaces[1].comment != "Edited"
        assert restored.polarization.is_polarized
        assert float(be.to_numpy(restored.polarization.phase_y)) == pytest.approx(
            -be.pi / 2
        )
        assert isinstance(restored.polarization.phase_y, type(be.cast(0.0)))
    finally:
        connector.calculation_jobs.shutdown()
        wait_for(qapp, lambda: connector.calculation_jobs._process is None)
        connector.deleteLater()
