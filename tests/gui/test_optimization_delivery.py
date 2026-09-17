"""Delivery regressions for optimization lifecycle and candidate previews."""

from __future__ import annotations

import threading

import pytest

from optiland_gui.optimization_panel import OptimizationPanel
from optiland_gui.optimization_preview import OptimizationPreview
from optiland_gui.services.job_records import OpticSnapshot
from optiland_gui.services.layout_tasks import prepare_2d
from optiland_gui.services.optimization_jobs import build_problem, optimize
from optiland_gui.services.optimization_service import OptimizationService
from tests.gui import test_optimization_jobs
from tests.gui.optimization_fixture import DelayedOptimizer, NonFiniteOptimizer
from tests.gui.test_calculation_jobs import wait_for

owned_service = test_optimization_jobs.owned_service


def test_no_document_reports_error_to_signal_and_callback(qapp, owned_service):
    connector, service = owned_service
    connector._optic = None
    errors, signals = [], []
    service.failed.connect(signals.append)
    service.run(DelayedOptimizer, {}, on_error=errors.append)
    assert errors == signals == ["Open an optical system before running optimization."]
    assert not service.is_running
    panel = OptimizationPanel(connector)
    panel._on_run()
    assert "Open an optical system" in panel.txtLog.toPlainText()
    assert panel.btnRun.isEnabled() and not panel.btnStop.isEnabled()
    panel.close()


def test_repeated_preview_keeps_navigation_without_changing_document(
    qapp, minimal_optic
):
    snapshot = OpticSnapshot.capture(minimal_optic)
    scene = prepare_2d(
        snapshot,
        {"num_rays": 3, "distribution": "line_y"},
        lambda *args: None,
        threading.Event(),
    )
    preview = OptimizationPreview(None)
    preview.update_theme("light")
    context = {
        "document_id": "candidate",
        "surface_identities": tuple(minimal_optic.surfaces),
    }
    try:
        preview.install(scene, context)
        preview.ax.set_xlim(-12, 17)
        preview.ax.set_ylim(-7, 9)
        preview.install(scene, context)
        assert preview.ax.get_xlim() == (-12, 17)
        assert preview.ax.get_ylim() == (-7, 9)
        assert OpticSnapshot.capture(minimal_optic).data == snapshot.data
    finally:
        preview.close()


@pytest.mark.parametrize("converged", [True, False])
def test_completion_owns_request_until_document_or_candidate_notifications_finish(
    qapp, owned_service, converged
):
    connector, service = owned_service
    completed, errors, attempted, changes = [], [], [], []
    signal = connector.opticLoaded if converged else service.candidateAvailable

    def try_restart(*args):
        if args and not args[0]:
            return
        attempted.append(service.is_running)
        service.run(
            DelayedOptimizer,
            {},
            on_finished=lambda _: errors.append("unexpected second run"),
        )

    signal.connect(try_restart)
    connector.document_state.committed.connect(changes.append)
    service.run(
        DelayedOptimizer, {}, on_finished=completed.append, on_error=errors.append
    )
    if not converged:
        connector.notify_change("metadata")
    wait_for(qapp, lambda: completed or errors, timeout=25)
    assert not errors
    assert attempted == [True]
    assert len(completed) == 1
    assert not service.is_running
    if converged:
        assert "was applied" in completed[0]
        assert len(changes) == 1
        assert connector._document_optic is connector.get_optic()
    else:
        assert service.take_candidate() is not None


def test_completion_listener_can_start_next_run(qapp, owned_service):
    connector, service = owned_service
    completed, errors = [], []

    def restart(summary):
        completed.append(summary)
        if len(completed) == 1:
            service.run(
                DelayedOptimizer,
                {},
                on_finished=completed.append,
                on_error=errors.append,
            )

    service.completed.connect(restart)
    service.run(DelayedOptimizer, {}, on_error=errors.append)
    wait_for(qapp, lambda: len(completed) == 3 or errors, timeout=25)
    assert not errors
    assert not service.is_running
    assert len(connector._undo_redo_manager._undo_stack) == 2


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
def test_exposed_algorithms_return_finite_detached_candidates(
    minimal_optic, group, index
):
    algorithm = OptimizationService.get_optimizer_groups()[group][index][1]
    kwargs = (
        {"max_iter": 3}
        if algorithm.__name__ == "OrthogonalDescent"
        else {"maxiter": 3, "disp": False}
    )
    if algorithm.__name__ == "LeastSquares":
        kwargs["method_choice"] = "trf"
    options = test_optimization_jobs.parameters(algorithm, kwargs)
    if algorithm.__name__ == "BasinHopping":
        options["variables"][0].update(min_val=None, max_val=None)
    snapshot = OpticSnapshot.capture(minimal_optic)
    result = optimize(snapshot, options, lambda *args, **kw: None, threading.Event())
    assert result["final_merit"] <= result["initial_merit"]
    assert result["evaluations"] > 0
    assert OpticSnapshot.capture(minimal_optic).data == snapshot.data
    if algorithm.__name__ == "OrthogonalDescent":
        assert not result["converged"] and result["iterations"] is None


@pytest.mark.parametrize("fail_preview", [False, True])
def test_preview_is_bounded_and_drawing_failure_does_not_lose_candidate(
    minimal_optic, monkeypatch, fail_preview
):
    import itertools

    from optiland_gui.services import layout_tasks

    clock = itertools.count()
    monkeypatch.setattr(
        "optiland_gui.services.optimization_jobs.time.monotonic",
        lambda: next(clock) * 0.3,
    )
    if fail_preview:

        def fail(*args, **kwargs):
            raise ValueError("drawing unavailable")

        monkeypatch.setattr(layout_tasks, "prepare_2d", fail)
    options = test_optimization_jobs.parameters(DelayedOptimizer, {"delay": 0})
    options["preview"] = {"enabled": True, "frequency": 1}
    progress = []
    result = optimize(
        OpticSnapshot.capture(minimal_optic),
        options,
        lambda *args, **kw: progress.append(kw.get("details", {})),
        threading.Event(),
    )
    assert result["converged"]
    key = "preview_error" if fail_preview else "preview"
    assert sum(key in entry for entry in progress) == 1
    assert result["candidate"].restore().surfaces[1].thickness == 8


@pytest.mark.parametrize(
    "variable,message",
    [("thickness", "non-finite final value"), ("radius", "non-finite final value")],
)
def test_nonfinite_candidate_is_rejected_even_when_solver_reports_success(
    minimal_optic, variable, message
):
    options = test_optimization_jobs.parameters(NonFiniteOptimizer, {})
    options["variables"][0]["type"] = variable
    with pytest.raises(ValueError, match=message):
        optimize(
            OpticSnapshot.capture(minimal_optic),
            options,
            lambda *args, **kw: None,
            threading.Event(),
        )


def test_nonfinite_merit_cannot_be_applied(minimal_optic):
    options = test_optimization_jobs.parameters(DelayedOptimizer, {"delay": 0})
    options["operands"][0]["weight"] = float("inf")
    with pytest.raises(ValueError, match="non-finite final merit"):
        optimize(
            OpticSnapshot.capture(minimal_optic),
            options,
            lambda *args, **kw: None,
            threading.Event(),
        )


@pytest.mark.parametrize("missing", ["variables", "operands"])
def test_incomplete_problem_is_rejected(minimal_optic, missing):
    variables, operands = test_optimization_jobs.definitions()
    if missing == "variables":
        variables = []
    else:
        operands = []
    with pytest.raises(ValueError, match="at least one variable and one operand"):
        build_problem(minimal_optic, variables, operands, {})


def test_setup_error_and_late_notifications_do_not_poison_next_run(qapp, owned_service):
    from optiland_gui.services.job_records import JobResult

    connector, service = owned_service
    errors, completed, progress = [], [], []
    service.progressChanged.connect(progress.append)

    class LocalOptimizer:
        pass

    service.run(LocalOptimizer, {}, on_error=errors.append)
    assert len(errors) == 1 and "importable" in errors[0]
    service.run(
        DelayedOptimizer, {}, on_finished=completed.append, on_error=errors.append
    )
    old = service._request
    wait_for(qapp, lambda: completed or len(errors) > 1, timeout=25)
    assert len(errors) == 1
    count = len(progress)
    service._on_progress(old, {"stage": "late"})
    service._on_finished(JobResult(old, "failed", error="late failure"))
    service.stop()
    assert len(progress) == count and len(errors) == 1 and len(completed) == 1


def test_restore_failure_leaves_document_and_undo_unchanged(
    qapp, owned_service, monkeypatch
):
    connector, service = owned_service
    original = connector.get_optic()
    errors = []

    def fail(_):
        raise ValueError("restore unavailable")

    monkeypatch.setattr(OpticSnapshot, "restore", fail)
    service.run(DelayedOptimizer, {}, on_error=errors.append)
    wait_for(qapp, lambda: errors, timeout=25)
    assert "Could not restore" in errors[0]
    assert connector.get_optic() is original
    assert not connector._undo_redo_manager.can_undo()


def test_algorithm_without_convergence_status_is_retained_from_panel(
    qapp, owned_service
):
    from optiland.optimization.optimizer.scipy import OrthogonalDescent

    connector, service = owned_service
    panel = OptimizationPanel(connector)
    panel.cmbAlgorithm.setCurrentIndex(panel.cmbAlgorithm.findData(OrthogonalDescent))
    completed, errors = [], []
    service.completed.connect(completed.append)
    service.failed.connect(errors.append)
    original = connector.get_optic()
    panel._on_run()
    panel._on_optimization_finished("Earlier run notification")
    assert not panel.btnRun.isEnabled()
    wait_for(qapp, lambda: completed or errors, timeout=25)
    assert not errors
    assert "without convergence" in completed[0] and "not reported" in completed[0]
    assert panel.btnCandidate.isEnabled()
    assert connector.get_optic() is original
    panel.close()


def test_candidate_open_errors_retain_candidate_and_preview_error_is_visible(
    qapp, owned_service, monkeypatch
):
    from dataclasses import replace

    from optiland_gui.services.job_records import BackendConfig

    connector, service = owned_service
    panel = OptimizationPanel(connector)
    panel.update_theme("light")
    panel._open_candidate()  # There is no retained result yet.
    snapshot = OpticSnapshot.capture(connector.get_optic())
    service._candidate = {
        "candidate": replace(snapshot, backend=BackendConfig("unavailable"))
    }
    panel._open_candidate()
    assert "Switch back" in panel.txtLog.toPlainText()
    service._candidate = {"candidate": snapshot}

    def fail(_):
        raise ValueError("restore unavailable")

    monkeypatch.setattr(OpticSnapshot, "restore", fail)
    panel._open_candidate()
    assert "Could not open candidate" in panel.txtLog.toPlainText()
    assert service._candidate is not None
    panel._on_optimization_progress(
        {
            "stage": "Optimizing",
            "preview_error": "No layout",
            "definitions_current": False,
        }
    )
    assert "Candidate preview unavailable: No layout" in panel.txtLog.toPlainText()
    panel.close()


def test_panel_progress_reuses_preview_and_tolerates_pending_table_refresh(
    qapp, owned_service
):
    connector, service = owned_service
    optic = connector.get_optic()
    scene = prepare_2d(
        OpticSnapshot.capture(optic),
        {"num_rays": 3, "distribution": "line_y"},
        lambda *args: None,
        threading.Event(),
    )
    panel = OptimizationPanel(connector)
    panel._preview_context = {
        "document_id": "candidate",
        "surface_identities": tuple(optic.surfaces),
    }
    panel.chkLiveVars.setChecked(True)
    panel.tblVariables.setRowCount(0)
    payload = {"stage": "Optimizing", "variables": [8.0], "preview": scene}
    panel._on_optimization_progress(payload)
    preview = panel._preview
    preview.ax.set_xlim(-12, 17)
    panel._on_optimization_progress(payload)
    assert panel._preview is preview
    assert preview.ax.get_xlim() == (-12, 17)
    assert panel.tblVariables.rowCount() == 0
    assert connector.get_optic() is optic
    panel.close()


def test_shgo_defaults_without_iteration_override(minimal_optic):
    from optiland.optimization.optimizer.scipy import SHGO

    result = optimize(
        OpticSnapshot.capture(minimal_optic),
        test_optimization_jobs.parameters(SHGO, {"disp": False}),
        lambda *args, **kw: None,
        threading.Event(),
    )
    assert result["converged"]
    assert result["final_merit"] < 1e-5


def test_setup_error_without_callback_is_delivered_by_signal(qapp, owned_service):
    _, service = owned_service
    errors = []
    service.failed.connect(errors.append)

    class LocalOptimizer:
        pass

    service.run(LocalOptimizer, {})
    assert len(errors) == 1 and "importable" in errors[0]
    assert not service.is_running
