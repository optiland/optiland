"""Owned scalar previews, stale-result rejection and bounded Qt table updates."""

from __future__ import annotations

import os
import pickle
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QThread, QTimer

from optiland.optimization.operand.operand import Operand, operand_registry
from optiland_gui.optiland_connector import OptilandConnector
from optiland_gui.optimization_panel import OptimizationPanel
from optiland_gui.services.calculation_jobs import CalculationJobs, DocumentState
from optiland_gui.services.job_records import (
    CalculationCancelled,
    JobResult,
    OpticSnapshot,
)
from optiland_gui.services.operand_preview_worker import prepare_operand_values
from optiland_gui.services.operand_previews import OperandPreviews
from tests.gui.test_calculation_jobs import wait_for


def parameters(operands):
    return {
        "operands": operands,
        "operand_metadata": {"real_y_intercept": {"wavelength": {}}},
    }


def test_preview_failure_labels_rows_and_replacement_clears_old_values(
    preview_service, monkeypatch
):
    state, jobs, service, definitions = preview_service
    monkeypatch.setattr(jobs, "_dispatch", lambda: None)
    service.refresh(explicit=True)
    service._timer.stop()
    service._submit()
    request = service._request
    service._install([{"index": 0, "value": 50.0, "error": ""}])
    service._on_finished(
        JobResult(request, "failed", error="worker failed", current=True)
    )
    assert service.rows == [{"value": 50.0, "error": "worker failed", "state": "error"}]
    state.replace()
    assert service.rows == [{"value": None, "error": "", "state": "stale"}]


def test_preview_row_bound_reports_error_without_submitting(
    preview_service, monkeypatch
):
    import optiland_gui.services.operand_previews as module

    state, jobs, service, definitions = preview_service
    monkeypatch.setattr(module, "MAX_OPERAND_ROWS", 1)
    definitions.append({"type": "total_track"})
    service.refresh(explicit=True)
    service._timer.stop()
    service._submit()
    assert jobs._serial == 0
    assert all(row["state"] == "error" for row in service.rows)
    assert all("at most 1" in row["error"] for row in service.rows)


def test_worker_matches_owned_reference_once_per_operand(minimal_optic, monkeypatch):
    snapshot = OpticSnapshot.capture(minimal_optic)
    before = pickle.dumps(minimal_optic.to_dict())
    calls = []

    def counted(optic):
        calls.append(optic)
        optic.surfaces[1].comment = "owned calculation"
        return 12.5

    monkeypatch.setitem(operand_registry._registry, "counted_preview", counted)
    definitions = [
        {"type": "counted_preview"},
        {"type": "total_track"},
        {
            "type": "real_y_intercept",
            "input_data": {
                "surface_number": 3,
                "Hx": 0.0,
                "Hy": 0.0,
                "Px": 0.0,
                "Py": 0.0,
                "wavelength": "[0.55]",
            },
        },
    ]
    result = prepare_operand_values(
        snapshot, parameters(definitions), lambda *a, **kw: None, threading.Event()
    )
    # Once for the row value. The merit for the shares adds the optimizer's
    # own two: the default target (none was set) and the batched residual.
    assert len(calls) == 3
    assert calls[0] is not minimal_optic
    assert result["rows"][0]["value"] == 12.5
    reference = snapshot.restore()
    reference.updater.update()
    assert result["rows"][1]["value"] == pytest.approx(
        float(Operand("total_track", target=0.0, input_data={"optic": reference}).value)
    )
    assert result["rows"][2]["value"] == pytest.approx(0.0)
    assert pickle.dumps(minimal_optic.to_dict()) == before


def test_worker_reports_each_error_and_continues(minimal_optic):
    messages = []
    result = prepare_operand_values(
        OpticSnapshot.capture(minimal_optic),
        parameters(
            [
                {"type": "invalid"},
                {"type": "real_y_intercept", "input_data": {"wavelength": "all"}},
                {"type": "total_track"},
            ]
        ),
        lambda *a, **kw: messages.append(kw),
        threading.Event(),
    )
    assert "invalid" in result["rows"][0]["error"]
    assert "Select one positive wavelength" in result["rows"][1]["error"]
    assert result["rows"][2]["value"] == 50.0
    assert messages[-1]["details"]["rows"] == result["rows"]
    # The merit is undefined while any operand fails.
    assert all("share" not in row for row in result["rows"])


def test_worker_reports_each_operands_share_of_the_merit(minimal_optic):
    result = prepare_operand_values(
        OpticSnapshot.capture(minimal_optic),
        parameters(
            [
                {"type": "total_track", "target": 47.0, "weight": 1.0},
                {"type": "total_track", "target": 50.0, "weight": 1.0},
                {"type": "total_track", "target": 44.0, "weight": 0.5},
            ]
        ),
        lambda *a, **kw: None,
        threading.Event(),
    )
    # Residuals 3, 0 and 0.5 * 6 = 3: squared 9, 0 and 9.
    shares = [row["share"] for row in result["rows"]]
    assert shares == pytest.approx([50.0, 0.0, 50.0])


def test_worker_cancels_between_rows_without_extra_evaluation(
    minimal_optic, monkeypatch
):
    cancelled = threading.Event()
    calls = []

    def stop(optic):
        calls.append(1)
        cancelled.set()
        return 1.0

    monkeypatch.setitem(operand_registry._registry, "stop_preview", stop)
    with pytest.raises(CalculationCancelled):
        prepare_operand_values(
            OpticSnapshot.capture(minimal_optic),
            parameters([{"type": "stop_preview"}] * 3),
            lambda *a, **kw: None,
            cancelled,
        )
    assert calls == [1]


def test_ray_preview_exports_a_scalar_on_each_backend(set_test_backend, minimal_optic):
    result = prepare_operand_values(
        OpticSnapshot.capture(minimal_optic),
        parameters(
            [
                {
                    "type": "real_y_intercept",
                    "input_data": {
                        "surface_number": 3,
                        "Hx": 0.0,
                        "Hy": 0.0,
                        "Px": 0.0,
                        "Py": 0.0,
                        "wavelength": "primary",
                    },
                }
            ]
        ),
        lambda *args, **kwargs: None,
        threading.Event(),
    )
    row = result["rows"][0]
    assert not row["error"]
    assert type(row["value"]) is float
    assert row["value"] == pytest.approx(0.0)


@pytest.fixture
def preview_service(qapp, minimal_optic):
    state = DocumentState()
    jobs = CalculationJobs(state, cancel_grace_ms=50)
    definitions = [{"type": "total_track"}]
    connector = SimpleNamespace(
        document_state=state,
        calculation_jobs=jobs,
        get_optic=lambda: minimal_optic,
        get_optimization_operands=lambda: definitions,
        _optimization_service=SimpleNamespace(OPERAND_METADATA={}),
    )
    service = OperandPreviews(connector)
    yield state, jobs, service, definitions
    service.stop()
    jobs.shutdown()
    wait_for(qapp, lambda: jobs._process is None)


def test_hidden_refresh_never_submits_until_visible(preview_service, qapp):
    state, jobs, service, definitions = preview_service
    service.refresh()
    assert not service.busy and jobs._serial == 0
    state.change()
    assert not service.busy and jobs._serial == 0
    service.set_visible(True)
    wait_for(qapp, lambda: service.rows[0]["state"] == "current", timeout=25)
    assert service.rows[0]["value"] == 50.0
    service.set_visible(False)
    state.change()
    assert service.rows[0]["value"] == 50.0
    assert service.rows[0]["state"] == "stale"
    assert not service.busy


def test_latest_definitions_revoke_pending_results_and_preserve_previous_values(
    preview_service, qapp, monkeypatch
):
    state, jobs, service, definitions = preview_service
    monkeypatch.setattr(jobs, "_dispatch", lambda: None)
    service.refresh(explicit=True)
    service._timer.stop()
    service._submit()
    first = service._request
    service._install([{"index": 0, "value": 50.0, "error": ""}])
    definitions.append({"type": "invalid"})
    service.refresh(explicit=True)
    service._timer.stop()
    service._submit()
    latest = service._request
    service._on_finished(
        JobResult(
            first,
            "succeeded",
            {"rows": [{"index": 0, "value": -99, "error": ""}]},
            current=True,
        )
    )
    assert service.rows[0]["value"] == 50.0
    assert latest.job_id != first.job_id
    assert latest.parameters["operands"] == definitions
    definitions[0]["target"] = 20.0
    service._on_progress(
        latest, {"details": {"rows": [{"index": 0, "value": -80, "error": ""}]}}
    )
    assert service.rows[0]["value"] == 50.0


def test_document_change_and_backend_currency_are_checked_again(
    preview_service, monkeypatch
):
    state, jobs, service, definitions = preview_service
    monkeypatch.setattr(jobs, "_dispatch", lambda: None)
    service.refresh(explicit=True)
    service._timer.stop()
    service._submit()
    request = service._request
    state.change()
    service._on_finished(
        JobResult(
            request,
            "succeeded",
            {"rows": [{"index": 0, "value": -1, "error": ""}]},
            current=True,
        )
    )
    assert service.rows[0]["value"] is None
    service.refresh(explicit=True)
    service._timer.stop()
    service._submit()
    request = service._request
    monkeypatch.setattr(jobs, "is_current", lambda request: False)
    service._on_progress(
        request, {"details": {"rows": [{"index": 0, "value": -1, "error": ""}]}}
    )
    assert service.rows[0]["value"] is None


def test_metadata_edit_does_not_stale_completed_values(preview_service, qapp):
    state, jobs, service, definitions = preview_service
    service.refresh(explicit=True)
    wait_for(qapp, lambda: service.rows[0]["state"] == "current", timeout=25)
    state.record("metadata")
    assert service.rows[0]["state"] == "current"
    assert not service.busy


def test_failed_refresh_keeps_prior_value_with_error(preview_service, monkeypatch):
    state, jobs, service, definitions = preview_service
    monkeypatch.setattr(jobs, "_dispatch", lambda: None)
    service.refresh(explicit=True)
    service._timer.stop()
    service._submit()
    service._install([{"index": 0, "value": 50.0, "error": ""}])
    request = service._request
    service._on_finished(
        JobResult(
            request,
            "succeeded",
            {
                "rows": [
                    {
                        "index": 0,
                        "value": None,
                        "error": "Trace failed for this operand",
                    },
                ]
            },
            current=True,
        )
    )
    assert service.rows[0] == {
        "state": "error",
        "value": 50.0,
        "error": "Trace failed for this operand",
    }
    assert not service.busy


def test_real_slow_operand_cancel_recovery_and_heartbeat(
    qapp, minimal_optic, record_property
):
    state = DocumentState()
    jobs = CalculationJobs(
        state,
        cancel_grace_ms=50,
        worker_command=[
            sys.executable,
            "-u",
            str(Path(__file__).with_name("operand_preview_fixture.py")),
        ],
    )
    definitions = [{"type": "test_delayed_value", "input_data": {"delay": 3.0}}]
    connector = SimpleNamespace(
        document_state=state,
        calculation_jobs=jobs,
        get_optic=lambda: minimal_optic,
        get_optimization_operands=lambda: definitions,
        _optimization_service=SimpleNamespace(OPERAND_METADATA={}),
    )
    service = OperandPreviews(connector)
    heartbeat = QTimer()
    times = []
    heartbeat.timeout.connect(lambda: times.append(time.monotonic()))
    heartbeat.start(10)
    threads = []
    service.rowsChanged.connect(lambda _: threads.append(QThread.currentThread()))
    try:
        service.refresh(explicit=True)
        wait_for(qapp, lambda: jobs.active_request is not None, timeout=25)
        started = time.monotonic()
        wait_for(qapp, lambda: time.monotonic() - started > 0.15)
        service.stop()
        wait_for(qapp, lambda: jobs._active is None, timeout=3)
        record_property("cancel_seconds", time.monotonic() - started - 0.15)
        assert not service.busy
        definitions[:] = [{"type": "total_track"}]
        service.refresh(explicit=True)
        wait_for(qapp, lambda: service.rows[0]["state"] == "current", timeout=25)
        assert service.rows[0]["value"] == 50.0
        assert len(times) > 10
        gaps = [
            later - earlier for earlier, later in zip(times, times[1:], strict=False)
        ]
        record_property("max_heartbeat_gap_seconds", max(gaps))
        assert max(gaps) < 0.5
        assert all(thread == qapp.thread() for thread in threads)
    finally:
        heartbeat.stop()
        service.stop()
        jobs.shutdown()
        wait_for(qapp, lambda: jobs._process is None)


def test_panel_keeps_selection_and_only_changes_value_cells(
    qapp, minimal_optic, monkeypatch
):
    connector = OptilandConnector()
    connector._optic = minimal_optic
    connector.opticLoaded.emit()
    service = connector._optimization_service
    service.add_operand({"type": "total_track", "target": 51.0, "weight": 1.0})
    service.add_operand({"type": "invalid", "target": 0.0, "weight": 1.0})
    monkeypatch.setattr(
        connector,
        "get_operand_current_value",
        lambda _: pytest.fail("GUI evaluated a live operand"),
    )
    panel = OptimizationPanel(connector)
    panel.resize(1000, 650)
    try:
        panel.show()
        qapp.processEvents()
        assert connector.calculation_jobs._serial == 0
        panel._tabs.setCurrentIndex(1)
        panel.tblOperands.selectRow(1)
        target_item = panel.tblOperands.item(0, 3)
        wait_for(
            qapp, lambda: panel.tblOperands.item(0, 2).text() == "50.000000", timeout=25
        )
        wait_for(
            qapp, lambda: panel.tblOperands.item(1, 2).text() == "Error", timeout=25
        )
        assert "invalid" in panel.tblOperands.item(1, 2).toolTip()
        assert panel.tblOperands.item(0, 6).text() == "—"
        assert "1 failed" in panel.lblOperandStatus.text()
        assert panel.tblOperands.item(0, 3) is target_item
        assert panel.tblOperands.currentRow() == 1
        if capture_path := os.environ.get("OPTILAND_TEST_GUI_CAPTURE"):
            assert panel.grab().save(capture_path)
        panel._tabs.setCurrentIndex(0)
        connector.notify_change("optical")
        qapp.processEvents()
        assert not panel._operand_previews.busy
        assert panel.tblOperands.item(0, 2).text() == "50.000000 (stale)"
    finally:
        panel.close()
        connector.calculation_jobs.shutdown()
        wait_for(qapp, lambda: connector.calculation_jobs._process is None)


def test_worker_bounds_rows_and_rejects_nonfinite_and_malformed_values(
    minimal_optic, monkeypatch
):
    import optiland_gui.services.operand_preview_worker as worker

    monkeypatch.setattr(worker, "MAX_OPERAND_ROWS", 3)
    with pytest.raises(ValueError, match="at most 3"):
        prepare_operand_values(
            None,
            parameters([{"type": "total_track"}] * 4),
            lambda *a, **kw: None,
            threading.Event(),
        )
    monkeypatch.setitem(
        operand_registry._registry, "nonfinite_preview", lambda optic: float("nan")
    )
    result = prepare_operand_values(
        OpticSnapshot.capture(minimal_optic),
        parameters(
            [
                {"type": "nonfinite_preview"},
                {"type": "total_track", "input_data_str": "[]"},
                {"type": "total_track", "input_data_str": "{}"},
            ]
        ),
        lambda *a, **kw: None,
        threading.Event(),
    )
    assert "non-finite" in result["rows"][0]["error"]
    assert "JSON object" in result["rows"][1]["error"]
    assert result["rows"][2]["value"] == 50.0


def test_worker_propagates_operand_cancellation(minimal_optic, monkeypatch):
    def cancel(optic):
        raise CalculationCancelled()

    monkeypatch.setitem(operand_registry._registry, "cancel_preview", cancel)
    with pytest.raises(CalculationCancelled):
        prepare_operand_values(
            OpticSnapshot.capture(minimal_optic),
            parameters([{"type": "cancel_preview"}, {"type": "total_track"}]),
            lambda *a, **kw: None,
            threading.Event(),
        )


def test_reordered_duplicates_keep_their_values_without_aliasing_definitions(
    preview_service, monkeypatch
):
    state, jobs, service, definitions = preview_service
    monkeypatch.setattr(jobs, "_dispatch", lambda: None)
    definitions[:] = [
        {"type": "total_track"},
        {"type": "invalid"},
        {"type": "total_track"},
    ]
    service.refresh(explicit=True)
    service._timer.stop()
    service._submit()
    first = service._request
    service.refresh()
    assert service._request is first
    service._install(
        [
            {"index": 0, "value": 50.0, "error": ""},
            {"index": 1, "value": 3.0, "error": ""},
            {"index": 2, "value": 51.0, "error": ""},
            {"index": 100, "value": -1, "error": ""},
        ]
    )
    definitions[:] = [definitions[0], definitions[2], definitions[1]]
    service.refresh()
    assert [row["value"] for row in service.rows] == [50.0, 51.0, 3.0]
    assert all(row["state"] == "stale" for row in service.rows)
    definitions[0]["target"] = 10
    assert "target" not in service._definitions[0]
    assert "target" not in first.parameters["operands"][0]


def test_debounce_recaptures_latest_definitions_and_empty_input_does_not_submit(
    preview_service, monkeypatch
):
    state, jobs, service, definitions = preview_service
    monkeypatch.setattr(jobs, "_dispatch", lambda: None)
    service._submit()  # Hidden panels do not start work.
    service.refresh(explicit=True)
    definitions[:] = [{"type": "invalid"}]
    service._timer.stop()
    service._submit()
    assert service._definitions == definitions
    assert service._request is None
    service._timer.stop()
    service._submit()
    assert service._request.parameters["operands"] == definitions
    service.stop()
    definitions.clear()
    service.refresh(explicit=True)
    service._timer.stop()
    serial = jobs._serial
    service._submit()
    assert jobs._serial == serial
    assert not service.busy
    service._submit()  # A clean empty panel remains idle.
    assert jobs._serial == serial


def test_obsolete_completion_requeues_visible_preview_without_publishing(
    preview_service, monkeypatch
):
    state, jobs, service, definitions = preview_service
    monkeypatch.setattr(jobs, "_dispatch", lambda: None)
    service.set_visible(True)
    service._timer.stop()
    service._submit()
    request = service._request
    service._on_finished(
        JobResult(
            request,
            "succeeded",
            {"rows": [{"index": 0, "value": -99.0, "error": ""}]},
            current=False,
        )
    )
    assert service.rows[0]["value"] is None
    assert service.rows[0]["state"] == "stale"
    assert service._timer.isActive()
    service._timer.stop()
    service._submit()
    assert service._request.job_id != request.job_id


def test_submit_failure_is_visible_and_retry_succeeds(preview_service, monkeypatch):
    state, jobs, service, definitions = preview_service
    monkeypatch.setattr(jobs, "_dispatch", lambda: None)
    statuses = []
    service.statusChanged.connect(lambda *args: statuses.append(args[0]))
    jobs.max_pending = 0
    service.refresh(explicit=True)
    service._timer.stop()
    service._submit()
    assert "queue is full" in statuses[-1]
    assert service.rows[0]["state"] == "error"
    assert not service.busy
    jobs.max_pending = 1
    service.refresh(explicit=True)
    service._timer.stop()
    service._submit()
    assert service.rows[0]["state"] == "pending"
    assert service.busy


def test_backend_change_invalidates_completed_values_without_a_document_edit(
    preview_service, qapp, monkeypatch
):
    from optiland_gui.services.job_records import BackendConfig

    state, jobs, service, definitions = preview_service
    monkeypatch.setattr(jobs, "_dispatch", lambda: None)
    service.set_visible(True)
    service._timer.stop()
    service._submit()
    request = service._request
    service._on_finished(
        JobResult(
            request,
            "succeeded",
            {"rows": [{"index": 0, "value": 50.0, "error": ""}]},
            current=True,
        )
    )
    token = state.edit_token
    service._check_backend()
    assert not service.busy
    assert service.rows[0]["state"] == "current"

    backend = BackendConfig("torch", "cpu", 64)
    monkeypatch.setattr(BackendConfig, "capture", lambda: backend)
    wait_for(qapp, lambda: service.rows[0]["state"] == "stale")
    assert service.rows[0]["value"] == 50.0
    assert state.edit_token == token
    service._timer.stop()
    service._submit()
    assert service._request.snapshot.backend == backend
    assert service._request.job_id != request.job_id

    service.set_visible(False)
    serial = jobs._serial
    backend = BackendConfig("numpy")
    service._check_backend()
    assert jobs._serial == serial
    assert service.rows[0]["state"] == "stale"
    assert not service.busy
    jobs.shutdown()
    assert not service._backend_timer.isActive()


def test_large_operand_table_installs_in_batches_and_keeps_selection(
    qapp, minimal_optic, monkeypatch
):
    connector = OptilandConnector()
    connector._optic = minimal_optic
    connector._optimization_service._operands = [
        {"type": "total_track", "target": 51.0, "weight": 1.0} for _ in range(260)
    ]
    monkeypatch.setattr(connector.calculation_jobs, "_dispatch", lambda: None)
    panel = OptimizationPanel(connector)
    try:
        panel._operand_install_timer.stop()
        preview = panel._operand_previews
        preview._install(
            [
                {"index": index, "value": float(index), "error": ""}
                for index in range(260)
            ]
        )
        panel.tblOperands.selectRow(200)
        target = panel.tblOperands.item(200, 3)
        panel.tblOperands.takeItem(10, 2)
        panel._operand_rows_pending.add(1000)
        panel._install_operand_rows()
        assert panel.tblOperands.item(0, 2).text() == "0.000000"
        assert panel.tblOperands.item(10, 2).text() == "10.000000"
        assert panel.tblOperands.item(259, 2).text() == "—"
        assert panel._operand_install_timer.isActive()
        while panel._operand_rows_pending:
            panel._operand_install_timer.stop()
            panel._install_operand_rows()
        assert panel.tblOperands.item(259, 2).text() == "259.000000"
        assert panel.tblOperands.currentRow() == 200
        assert panel.tblOperands.item(200, 3) is target

        preview.rows[0].update(state="pending")
        preview.rows[1].update(state="error", error="calculation failed")
        preview.rows[2].update(value=None, state="pending")
        panel._queue_operand_rows([0, 1, 2])
        panel._operand_install_timer.stop()
        panel._install_operand_rows()
        assert panel.tblOperands.item(0, 2).text() == "0.000000 (updating)"
        assert panel.tblOperands.item(1, 2).text() == "1.000000 (stale; error)"
        assert panel.tblOperands.item(1, 2).toolTip() == "calculation failed"
        assert panel.tblOperands.item(2, 2).text() == "Pending…"

        connector._optimization_service._operands.pop()
        panel._queue_operand_rows([259])
        panel._operand_install_timer.stop()
        panel._install_operand_rows()
        assert panel.tblOperands.rowCount() == 259
        panel._refresh_operands_current_values()
        assert preview._explicit
    finally:
        panel.close()
        connector.calculation_jobs.shutdown()
        wait_for(qapp, lambda: connector.calculation_jobs._process is None)
        panel.deleteLater()
