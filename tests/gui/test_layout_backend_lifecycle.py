"""Actual-worker regression for backend changes during a visible layout job."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QVBoxLayout, QWidget

from optiland_gui.services.calculation_jobs import CalculationJobs, DocumentState
from optiland_gui.services.job_records import BackendConfig
from optiland_gui.widgets.layout_job_view import LayoutJobView
from tests.gui.test_calculation_jobs import wait_for


@pytest.fixture
def layout_job(qapp, minimal_optic, monkeypatch):
    pytest.importorskip("torch")
    current_backend = [BackendConfig()]
    monkeypatch.setattr(
        BackendConfig, "capture", classmethod(lambda cls: current_backend[0])
    )
    state = DocumentState()
    jobs = CalculationJobs(state)
    viewer = QWidget()
    viewer.connector = SimpleNamespace(
        document_state=state,
        calculation_jobs=jobs,
        get_optic=lambda: minimal_optic,
    )
    parameters = {"num_rays": 3, "distribution": "line_y"}
    installed = []
    helper = LayoutJobView(
        viewer,
        "2d",
        QVBoxLayout(viewer),
        lambda: dict(parameters),
        lambda data, context, restyle: installed.append((data, context, restyle)),
    )
    yield viewer, helper, jobs, parameters, current_backend, installed
    viewer.hide()
    jobs.shutdown()
    wait_for(qapp, lambda: jobs._process is None)
    viewer.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_backend_changed_completion_keeps_last_display_and_replaces_once(
    qapp, layout_job
):
    viewer, helper, jobs, parameters, backend, installed = layout_job
    viewer.show()
    wait_for(qapp, lambda: helper.data is not None, timeout=30)
    previous = helper.data
    serial = jobs._serial
    parameters["num_rays"] = 7
    helper.request()
    obsolete_id = helper._job_id
    stale_completion = []

    def switch_backend(request, message):
        if request.job_id == obsolete_id:
            backend[0] = BackendConfig("torch")

    def completed(result):
        if result.request.job_id == obsolete_id:
            stale_completion.append(
                (
                    result,
                    helper.data,
                    helper.label.text(),
                    helper._refresh_timer.isActive(),
                )
            )

    jobs.progress.connect(switch_backend)
    jobs.finished.connect(completed)
    wait_for(qapp, lambda: len(installed) == 2 and not jobs.running, timeout=30)
    result, retained, status, refresh_pending = stale_completion[0]
    assert result.status == "succeeded" and not result.current
    assert retained is previous
    assert all(data is not result.data for data, _, _ in installed)
    assert "stale" in status.lower() and refresh_pending
    assert jobs._serial == serial + 2
    assert installed[-1][1]["key"][1] == BackendConfig("torch")
    assert helper.label.text() == "Layout is up to date."
    assert not helper._refresh_timer.isActive()
    assert not helper._active


def test_backend_change_does_not_automatically_retry_a_calculation_error(
    qapp, layout_job
):
    viewer, helper, jobs, parameters, backend, installed = layout_job
    parameters["distribution"] = "invalid-layout-distribution"
    results = []
    jobs.finished.connect(results.append)
    jobs.progress.connect(
        lambda request, message: backend.__setitem__(0, BackendConfig("torch"))
    )
    viewer.show()
    wait_for(qapp, lambda: bool(results), timeout=30)
    qapp.processEvents()
    assert results[0].status == "failed"
    assert not installed
    assert jobs._serial == 1
    assert not jobs.running
    assert not helper._refresh_timer.isActive()
    assert not helper._active
    assert "retry" in helper.label.text().lower()
