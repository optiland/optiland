"""Preview coalescing must preserve the newest request and its busy state."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QVBoxLayout, QWidget

from optiland_gui.services.calculation_jobs import CalculationJobs, DocumentState
from optiland_gui.widgets.layout_job_view import LayoutJobView
from tests.gui.test_calculation_jobs import wait_for


@pytest.fixture
def preview(qapp, minimal_optic):
    state = DocumentState()
    jobs = CalculationJobs(
        state,
        cancel_grace_ms=40,
        worker_command=[
            sys.executable,
            "-u",
            str(Path(__file__).with_name("calculation_worker_fixture.py")),
        ],
    )
    viewer = QWidget()
    viewer.connector = SimpleNamespace(
        document_state=state,
        calculation_jobs=jobs,
        get_optic=lambda: minimal_optic,
    )
    parameters = {"num_rays": 3, "distribution": "line_y", "delay": 0.15}
    helper = LayoutJobView(
        viewer,
        "2d",
        QVBoxLayout(viewer),
        lambda: dict(parameters),
        lambda *_: None,
    )
    viewer.show()
    yield helper, jobs, parameters
    viewer.hide()
    jobs.shutdown()
    wait_for(qapp, lambda: jobs._process is None)
    viewer.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_repeated_apply_reuses_the_latest_pending_preview(preview):
    helper, jobs, parameters = preview
    # All clicks arrive before the worker dispatch timer. Replacing a queued
    # preview emits its cancellation synchronously inside the second submit.
    helper.request()
    parameters["num_rays"] = 5
    helper.request()
    newest_id = helper._job_id
    helper.request()
    assert helper._job_id == newest_id
    assert jobs._serial == 2
    assert len(jobs._pending) == 1
    assert helper._active


def test_cancelled_older_preview_does_not_clear_latest_busy_state(qapp, preview):
    helper, jobs, parameters = preview
    progress = []
    jobs.progress.connect(lambda request, _message: progress.append(request.job_id))
    helper.request()
    previous_id = helper._job_id
    wait_for(qapp, lambda: previous_id in progress)
    parameters["num_rays"] = 5
    helper.request()
    latest_id = helper._job_id
    observations = []

    def finished(result):
        if result.request.job_id == previous_id:
            observations.append((helper._active, helper.cancel_button.isVisible()))

    jobs.finished.connect(finished)
    wait_for(qapp, lambda: bool(observations))
    assert observations == [(True, True)]
    assert helper._job_id == latest_id
    wait_for(qapp, lambda: not jobs.running)
    assert not helper._active
    assert helper.label.text() == "Layout is up to date."
