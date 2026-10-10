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
    assert len(calls) == 1
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
