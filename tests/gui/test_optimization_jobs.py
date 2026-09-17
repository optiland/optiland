"""Real optimizer cancellation and Qt-owned candidate acceptance."""

from __future__ import annotations

import pickle
import threading

import pytest
from PySide6.QtCore import QThread

from optiland_gui.optiland_connector import OptilandConnector
from optiland_gui.services.job_records import CalculationCancelled, OpticSnapshot
from optiland_gui.services.optimization_jobs import optimize
from optiland_gui.services.optimization_service import OptimizationService
from tests.gui.optimization_fixture import DelayedOptimizer, FailedOptimizer
from tests.gui.test_calculation_jobs import wait_for


def definitions():
    return (
        [{"surface_number": 1, "type": "thickness", "min_val": 1.0, "max_val": 15.0}],
        [
            {
                "type": "total_track",
                "target": 53.0,
                "weight": 1.0,
                "input_data_str": "{}",
            }
        ],
    )


def parameters(algorithm, kwargs):
    variables, operands = definitions()
    return {
        "optimizer": f"{algorithm.__module__}:{algorithm.__qualname__}",
        "optimizer_kwargs": kwargs,
        "variables": variables,
        "operands": operands,
        "operand_metadata": {},
    }


@pytest.fixture
def owned_service(qapp, minimal_optic):
    connector = OptilandConnector()
    connector._optic = minimal_optic
    connector.opticLoaded.emit()
    service = connector._optimization_service
    service._variables, service._operands = definitions()
    yield connector, service
    connector.calculation_jobs.shutdown()
    wait_for(qapp, lambda: connector.calculation_jobs._process is None)


def test_least_squares_candidate_matches_reference_without_mutating_input(
    set_test_backend,
    minimal_optic,
):
    from optiland.optimization.optimizer.scipy import LeastSquares

    snapshot = OpticSnapshot.capture(minimal_optic)
    before = pickle.dumps(minimal_optic.to_dict())
    messages = []
    result = optimize(
        snapshot,
        parameters(LeastSquares, {"method_choice": "trf", "maxiter": 30}),
        lambda *args, **kwargs: messages.append(kwargs),
        threading.Event(),
    )
    candidate = result["candidate"].restore()
    assert candidate.surfaces[1].thickness == pytest.approx(8.0, abs=1e-5)
    assert pickle.dumps(minimal_optic.to_dict()) == before
    assert result["converged"]
    assert result["final_merit"] < 1e-5
    assert any("details" in message for message in messages)


def test_solver_success_does_not_accept_ignored_variable_bounds(minimal_optic):
    from optiland.optimization.optimizer.scipy import OptimizerGeneric

    options = parameters(OptimizerGeneric, {"method": "BFGS", "disp": False})
    options["variables"][0]["max_val"] = 6.0
    result = optimize(
        OpticSnapshot.capture(minimal_optic),
        options,
        lambda *args, **kwargs: None,
        threading.Event(),
    )
    assert result["candidate"].restore().surfaces[1].thickness > 6.0
    assert not result["converged"]
    assert "violates configured bounds" in result["message"]


@pytest.mark.parametrize(
    "group,index",
    [
        ("Local", 0),
        ("Local", 1),
        ("Local", 2),
        ("Global", 0),
        ("Global", 1),
        ("Global", 2),
        ("Global", 3),
    ],
)
def test_every_exposed_family_stops_at_evaluation_boundary(
    minimal_optic, group, index, monkeypatch
):
    import itertools

    clock = itertools.count()
    monkeypatch.setattr(
        "optiland_gui.services.optimization_jobs.time.monotonic",
        lambda: next(clock) * 0.3,
    )
    algorithm = OptimizationService.get_optimizer_groups()[group][index][1]
    kwargs = (
        {"max_iter": 3} if algorithm.__name__ == "OrthogonalDescent" else {"maxiter": 3}
    )
    if algorithm.__name__ == "LeastSquares":
        kwargs["method_choice"] = "trf"
    options = parameters(algorithm, kwargs)
    if algorithm.__name__ == "BasinHopping":
        options["variables"][0].update(min_val=None, max_val=None)
    cancelled = threading.Event()

    def progress(stage, **payload):
        if payload.get("details", {}).get("evaluations", 0) >= 1:
            cancelled.set()

    with pytest.raises(CalculationCancelled):
        optimize(OpticSnapshot.capture(minimal_optic), options, progress, cancelled)


def test_actual_process_commit_and_callbacks_use_gui_thread(qapp, owned_service):
    connector, service = owned_service
    original = connector.get_optic()
    callbacks, errors, threads = [], [], []
    service.run(
        DelayedOptimizer,
        {},
        lambda n: threads.append(QThread.currentThread()),
        lambda summary: (
            callbacks.append(summary),
            threads.append(QThread.currentThread()),
        ),
        errors.append,
    )
    assert connector.get_optic() is original
    assert not connector._undo_redo_manager.can_undo()
    wait_for(qapp, lambda: callbacks or errors, timeout=25)
    assert not errors
    assert connector.get_optic() is not original
    assert original.surfaces[1].thickness == 5.0
    assert connector.get_optic().surfaces[1].thickness == 8.0
    assert len(connector._undo_redo_manager._undo_stack) == 1
    assert all(thread is qapp.thread() for thread in threads)
    connector.undo()
    assert connector.get_optic().surfaces[1].thickness == 5.0


@pytest.mark.parametrize("kind", ["metadata", "optical", "replacement"])
def test_intervening_edit_keeps_candidate_separate(qapp, owned_service, kind):
    connector, service = owned_service
    original = connector.get_optic()
    completed, errors = [], []
    service.run(
        DelayedOptimizer, {}, on_finished=completed.append, on_error=errors.append
    )
    original.surfaces[1].comment = "User edit"
    connector.notify_change(kind)
    wait_for(qapp, lambda: completed or errors, timeout=25)
    assert not errors
    assert connector.get_optic() is original
    assert original.surfaces[1].comment == "User edit"
    assert not connector._undo_redo_manager.can_undo()
    assert service.take_candidate() is not None


def test_cancel_and_failure_leave_model_and_undo_unchanged(qapp, owned_service):
    connector, service = owned_service
    original = connector.get_optic()
    completed, errors = [], []
    service.run(
        DelayedOptimizer,
        {"delay": 10},
        on_finished=completed.append,
        on_error=errors.append,
    )
    wait_for(qapp, lambda: connector.calculation_jobs.active_request is not None)
    service.stop()
    wait_for(qapp, lambda: completed or errors)
    assert completed and "cancelled" in completed[0].lower()
    assert not errors
    service.run(
        FailedOptimizer, {}, on_finished=completed.append, on_error=errors.append
    )
    wait_for(qapp, lambda: errors, timeout=25)
    assert "intentional optimization failure" in errors[0]
    assert connector.get_optic() is original
    assert not connector._undo_redo_manager.can_undo()


def test_definition_edit_prevents_stale_candidate_commit(qapp, owned_service):
    connector, service = owned_service
    original = connector.get_optic()
    completed, errors = [], []
    service.run(
        DelayedOptimizer, {}, on_finished=completed.append, on_error=errors.append
    )
    service.set_operand(0, {**service.get_operands()[0], "target": 60.0})
    wait_for(qapp, lambda: completed or errors, timeout=25)
    assert not errors
    assert connector.get_optic() is original
    assert not connector._undo_redo_manager.can_undo()
    assert "definitions changed" in completed[0]
    assert service.take_candidate() is not None


def test_bad_operand_fails_setup_without_silently_dropping_it(qapp, owned_service):
    connector, service = owned_service
    original = connector.get_optic()
    service.add_operand({"type": "invalid-operand", "target": 0.0})
    errors = []
    service.run(DelayedOptimizer, {}, on_error=errors.append)
    wait_for(qapp, lambda: errors, timeout=25)
    assert connector.get_optic() is original
    assert not connector._undo_redo_manager.can_undo()
    assert "invalid-operand" in errors[0]


def test_candidate_preview_and_panel_slots_do_not_modify_live_document(
    qapp, owned_service
):
    from optiland_gui.optimization_panel import OptimizationPanel

    connector, service = owned_service
    original = connector.get_optic()
    panel = OptimizationPanel(connector)
    panel._refresh_variables_table()
    panel.chkLiveVars.setChecked(True)
    panel._preview_context = {
        "document_id": connector.document_state.token.document_id,
        "surface_identities": tuple(original.surfaces),
    }
    service.preview_options = {"enabled": True, "frequency": 1}
    completed, errors, progress = [], [], []
    service.progressChanged.connect(progress.append)
    service.run(
        DelayedOptimizer, {}, on_finished=completed.append, on_error=errors.append
    )
    connector.notify_change("metadata")
    wait_for(qapp, lambda: completed or errors, timeout=25)
    try:
        assert not errors
        assert any("preview" in item for item in progress)
        assert panel._preview is not None
        assert panel._preview.ax.lines
        assert connector.get_optic() is original
        panel.update_theme("light")
        assert panel._preview.current_theme == "light"
        assert panel.btnRun.isEnabled()
        assert not panel.btnStop.isEnabled()
        assert panel.btnCandidate.isEnabled()
    finally:
        panel.close()


def test_separate_candidate_window_is_released_after_close(
    qapp, owned_service, monkeypatch
):
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QWidget

    from optiland_gui.optimization_panel import OptimizationPanel

    class CandidateWindow(QWidget):
        def __init__(self):
            super().__init__()
            self.connector = OptilandConnector()

    monkeypatch.setattr("optiland_gui.main_window.MainWindow", CandidateWindow)
    connector, service = owned_service
    panel = OptimizationPanel(connector)
    service._candidate = {"candidate": OpticSnapshot.capture(connector.get_optic())}
    panel._open_candidate()
    assert len(panel._candidate_windows) == 1
    window = panel._candidate_windows[0]
    assert window.connector.get_optic() is not connector.get_optic()
    window.close()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not panel._candidate_windows
    panel.close()
