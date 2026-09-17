"""Controllable shared executor for analysis page lifecycle tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from PySide6.QtCore import QObject, Signal

from optiland_gui.services.analysis_runner import AnalysisRunner
from optiland_gui.services.calculation_jobs import DocumentState
from optiland_gui.services.job_records import JobRequest, JobResult


class ManualJobs(QObject):
    state_changed = Signal(object, str)
    finished = Signal(object)
    progress = Signal(object, dict)

    def __init__(self, document):
        super().__init__()
        self.document = document
        self.submissions = []
        self.requests = {}
        self.cancelled = []

    def submit(self, target, handler, snapshot, parameters, **options):
        self.cancel_target(target)
        request = JobRequest(
            len(self.submissions) + 1,
            options.get("document_token") or self.document.token,
            target,
            0,
            handler,
            snapshot,
            parameters,
            options.get("cancel_on_document_change", True),
        )
        self.submissions.append(request)
        self.requests[target] = request
        return request

    def cancel_target(self, target):
        self.cancelled.append(target)
        request = self.requests.pop(target, None)
        if request:
            self.finished.emit(JobResult(request, "cancelled"))

    def complete(
        self,
        target,
        data=None,
        status="succeeded",
        error="",
        infrastructure_error=False,
    ):
        request = self.requests.pop(target)
        self.finished.emit(
            JobResult(
                request,
                status,
                data,
                error,
                request.document == self.document.token,
                infrastructure_error,
            )
        )


def connector_for(optic):
    state = DocumentState()
    connector = SimpleNamespace(
        document_state=state,
        calculation_jobs=ManualJobs(state),
        get_optic=lambda: optic,
        toast_manager=Mock(),
        get_field_options=lambda: [("All", "all")],
        get_wavelength_options=lambda: [("Primary", "primary")],
    )
    connector._analysis_runner = AnalysisRunner(connector)
    return connector


def result_data():
    return {
        "plot": {"figsize": (7, 5), "axes": [], "texts": [], "legends": []},
        "size_bytes": 100,
        "summary": "Completed data",
    }
