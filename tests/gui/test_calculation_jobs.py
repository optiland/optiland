"""Behavioral coverage of owned snapshots, Qt delivery and process lifecycle."""

from __future__ import annotations

import pickle
import sys
import threading
import time
from operator import attrgetter
from pathlib import Path
from types import MethodType

import numpy as np
import pytest
from PySide6.QtCore import QObject, QThread, QTimer, Slot

from optiland_gui.services.calculation_jobs import CalculationJobs, DocumentState
from optiland_gui.services.job_records import OpticSnapshot


def wait_for(qapp, predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while not predicate():
        qapp.processEvents()
        if time.monotonic() > deadline:
            raise AssertionError("Calculation condition timed out")
        time.sleep(0.002)


def test_progress_preserves_bounded_plain_details_and_cancellation():
    from optiland_gui.services.calculation_worker import (
        _progress_reporter,
        encode_message,
    )
    from optiland_gui.services.job_records import CalculationCancelled

    encoded = []
    cancelled = threading.Event()
    report = _progress_reporter(
        7, cancelled, lambda message: encoded.append(encode_message(message))
    )
    details = {"merit": 0.123, "evaluations": 12, "variables": [1.5, 2.5]}
    report("Optimizing", details=details)
    message = pickle.loads(encoded[0][4:])
    assert message["job_id"] == 7
    assert message["details"] == details
    assert message["completed"] is None and message["total"] is None
    cancelled.set()
    with pytest.raises(CalculationCancelled):
        report("Optimizing", details=details)
    assert len(encoded) == 1


class Receiver(QObject):
    def __init__(self):
        super().__init__()
        self.results = []
        self.threads = []
        self.progress = []

    @Slot(object)
    def result(self, result):
        self.threads.append(QThread.currentThread())
        self.results.append(result)

    @Slot(object, dict)
    def update(self, request, message):
        self.threads.append(QThread.currentThread())
        self.progress.append(request.job_id)


@pytest.fixture
def jobs(qapp):
    state = DocumentState()
    service = CalculationJobs(
        state,
        cancel_grace_ms=40,
        worker_command=[
            sys.executable,
            "-u",
            str(Path(__file__).with_name("calculation_worker_fixture.py")),
        ],
    )
    receiver = Receiver()
    service.finished.connect(receiver.result)
    service.progress.connect(receiver.update)
    yield state, service, receiver
    service.shutdown()
    wait_for(qapp, lambda: service._process is None)


def test_snapshot_is_owned_pure_and_numerically_equivalent(minimal_optic, monkeypatch):
    before = pickle.dumps(minimal_optic.to_dict(), protocol=5)
    monkeypatch.setattr(
        minimal_optic.updater, "update", lambda: pytest.fail("capture ran solves")
    )
    snapshot = OpticSnapshot.capture(minimal_optic)
    assert pickle.dumps(minimal_optic.to_dict(), protocol=5) == before
    copy = snapshot.restore()
    assert copy is not minimal_optic
    assert copy.surfaces[1] is not minimal_optic.surfaces[1]
    minimal_optic.trace(0, 0, 0.55, 5, "line_y")
    copy.trace(0, 0, 0.55, 5, "line_y")
    np.testing.assert_allclose(copy.surfaces.y, minimal_optic.surfaces.y)
    copy.surfaces[1].comment = "Worker changed"
    assert minimal_optic.surfaces[1].comment != "Worker changed"


def test_polarized_snapshot_preserves_aperture_and_incident_state(minimal_optic):
    from optiland.physical_apertures import RectangularAperture
    from optiland.rays import PolarizationState

    minimal_optic.polarization = PolarizationState(is_polarized=False)
    minimal_optic.surfaces[1].aperture = RectangularAperture(
        x_min=-4.0, x_max=4.0, y_min=-3.0, y_max=3.0
    )
    copy = OpticSnapshot.capture(minimal_optic).restore()
    assert not copy.polarization.is_polarized
    assert (
        copy.surfaces[1].aperture.to_dict()
        == minimal_optic.surfaces[1].aperture.to_dict()
    )


@pytest.mark.parametrize("component_name", ["ray_tracer", "ray_generator"])
@pytest.mark.parametrize("override_kind", ["subclass", "instance_method"])
def test_snapshot_rejects_unserializable_tracing_extensions(
    minimal_optic, component_name, override_kind
):
    component = minimal_optic.ray_tracer
    method_name = "trace"
    if component_name == "ray_generator":
        component = component.ray_generator
        method_name = "generate_rays"
    if override_kind == "subclass":

        class CustomTracingComponent(type(component)):
            pass

        component.__class__ = CustomTracingComponent
    else:
        setattr(component, method_name, lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="snapshot adapter"):
        OpticSnapshot.capture(minimal_optic)


@pytest.mark.parametrize("method_name", ["trace", "trace_generic"])
@pytest.mark.parametrize("override_kind", ["subclass", "instance_method"])
def test_snapshot_rejects_custom_optic_trace_methods(
    minimal_optic, method_name, override_kind
):
    if override_kind == "subclass":
        minimal_optic.__class__ = type(
            "CustomOptic",
            (type(minimal_optic),),
            {method_name: lambda *args, **kwargs: None},
        )
    else:
        setattr(minimal_optic, method_name, lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="snapshot adapter"):
        OpticSnapshot.capture(minimal_optic)


@pytest.mark.parametrize(
    ("component_path", "method_name"),
    [
        ("", "trace"),
        ("", "trace_generic"),
        ("ray_tracer", "trace"),
        ("ray_tracer", "trace_generic"),
        ("ray_tracer.ray_generator", "generate_rays"),
    ],
)
@pytest.mark.parametrize("binding", ["same_owner", "foreign_owner", "wrapper"])
def test_snapshot_accepts_only_behaviorally_identical_rebound_trace_methods(
    minimal_optic, component_path, method_name, binding
):
    def component(optic):
        return attrgetter(component_path)(optic) if component_path else optic

    owner = component(minimal_optic)
    method = getattr(owner, method_name)
    if binding == "foreign_owner":
        other = OpticSnapshot.capture(minimal_optic).restore()
        method = getattr(component(other), method_name)
    elif binding == "wrapper":
        original = method

        def wrapper(self, *args, **kwargs):
            return original(*args, **kwargs)

        method = MethodType(wrapper, owner)
    setattr(owner, method_name, method)

    if binding != "same_owner":
        with pytest.raises(ValueError, match="snapshot adapter"):
            OpticSnapshot.capture(minimal_optic)
        return

    restored = OpticSnapshot.capture(minimal_optic).restore()
    assert method_name not in vars(component(restored))
    minimal_optic.trace(0, 0, 0.55, 5, "line_y")
    restored.trace(0, 0, 0.55, 5, "line_y")
    np.testing.assert_allclose(restored.surfaces.y, minimal_optic.surfaces.y)


def test_worker_keeps_heartbeat_and_delivers_slots_on_gui(qapp, jobs):
    state, service, receiver = jobs
    ticks = []
    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: ticks.append(time.monotonic()))
    timer.start()
    request = service.submit("2d", "unused", None, {"delay": 0.3})
    wait_for(qapp, lambda: len(receiver.results) == 1)
    timer.stop()
    assert len(ticks) >= 10
    assert receiver.results[0].request == request
    assert receiver.results[0].current
    assert receiver.results[0].data == 42
    assert all(thread == qapp.thread() for thread in receiver.threads)


def test_replacement_kills_stubborn_active_and_runs_latest(qapp, jobs):
    state, service, receiver = jobs
    first = service.submit("2d", "unused", None, {"delay": 10})
    wait_for(qapp, lambda: first.job_id in receiver.progress)
    second = service.submit("2d", "unused", None, {"value": "latest"})
    wait_for(qapp, lambda: len(receiver.results) == 2)
    assert [r.status for r in receiver.results] == ["cancelled", "succeeded"]
    assert receiver.results[1].request == second
    assert receiver.results[1].data == "latest"
    assert not receiver.results[0].current
    assert not receiver.results[0].infrastructure_error


def test_document_replacement_rejects_old_completion(qapp, jobs):
    state, service, receiver = jobs
    first = service.submit("2d", "unused", None, {"delay": 0.15})
    wait_for(qapp, lambda: first.job_id in receiver.progress)
    state.replace()
    wait_for(qapp, lambda: len(receiver.results) == 1)
    assert not receiver.results[0].current
    assert receiver.results[0].status == "cancelled"


def test_pending_replacement_has_exactly_one_terminal_each(qapp, jobs):
    state, service, receiver = jobs
    requests = [
        service.submit("2d", "unused", None, {"value": value}) for value in range(5)
    ]
    wait_for(qapp, lambda: len(receiver.results) == 5)
    assert sorted(r.request.job_id for r in receiver.results) == [
        r.job_id for r in requests
    ]
    assert sum(r.status == "succeeded" for r in receiver.results) == 1
    assert receiver.results[-1].data == 4


def test_hidden_target_never_dispatches(qapp, jobs):
    state, service, receiver = jobs
    service.set_target_visible("3d", False)
    service.submit("3d", "unused", None, {})
    wait_for(qapp, lambda: receiver.results)
    assert receiver.results[0].status == "cancelled"
    assert not receiver.progress


@pytest.mark.parametrize("stderr", ["", "Native library aborted"])
def test_crash_has_terminal_error_and_service_recovers(qapp, jobs, stderr):
    state, service, receiver = jobs
    service.submit("2d", "unused", None, {"crash": True, "stderr": stderr})
    wait_for(qapp, lambda: receiver.results)
    assert receiver.results[0].status == "failed"
    assert receiver.results[0].infrastructure_error
    if stderr:
        assert stderr in receiver.results[0].error
    service.submit("2d", "unused", None, {})
    wait_for(qapp, lambda: len(receiver.results) == 2)
    assert receiver.results[1].status == "succeeded"
    assert not receiver.results[1].infrastructure_error


def test_shutdown_revokes_and_reaps_without_blocking(qapp, jobs):
    state, service, receiver = jobs
    request = service.submit("2d", "unused", None, {"delay": 10})
    wait_for(qapp, lambda: request.job_id in receiver.progress)
    start = time.monotonic()
    service.shutdown()
    assert time.monotonic() - start < 0.1
    wait_for(qapp, lambda: service._process is None)
    assert len(receiver.results) == 1
    assert not receiver.results[0].current
    with pytest.raises(RuntimeError, match="closed"):
        service.submit("2d", "unused", None, {})


def test_failed_start_finishes_queued_request(qapp):
    service = CalculationJobs(DocumentState(), worker_command=["nonexistent-worker"])
    receiver = Receiver()
    service.finished.connect(receiver.result)
    service.submit("2d", "unused", None, {})
    wait_for(qapp, lambda: receiver.results)
    assert receiver.results[0].status == "failed"
    assert receiver.results[0].infrastructure_error
    assert not service.running
    service.shutdown()


def test_explicit_jobs_keep_fifo_and_enforce_queue_bound(qapp, jobs):
    state, service, receiver = jobs
    service.max_pending = 2
    service.submit("analysis-a", "unused", None, {"value": "a"}, replace=False)
    service.submit("analysis-b", "unused", None, {"value": "b"}, replace=False)
    with pytest.raises(RuntimeError, match="queue is full"):
        service.submit("analysis-c", "unused", None, {}, replace=False)
    wait_for(qapp, lambda: len(receiver.results) == 2)
    assert [r.data for r in receiver.results] == ["a", "b"]


def test_actual_worker_reports_handler_failure_and_closes(qapp):
    state = DocumentState()
    service = CalculationJobs(state)
    receiver = Receiver()
    service.finished.connect(receiver.result)
    service.submit("bad", "optiland_gui.services.job_records:does_not_exist", None, {})
    try:
        wait_for(qapp, lambda: receiver.results)
        assert receiver.results[0].status == "failed"
        assert "does_not_exist" in receiver.results[0].error
        assert not receiver.results[0].infrastructure_error
    finally:
        service.shutdown()
        wait_for(qapp, lambda: service._process is None)


def test_malformed_worker_transport_is_infrastructure_failure(qapp, jobs):
    _, service, receiver = jobs
    service.submit("bad-transport", "unused", None, {"malformed": True})
    wait_for(qapp, lambda: receiver.results)
    assert receiver.results[0].status == "failed"
    assert receiver.results[0].infrastructure_error
    assert "Invalid calculation-worker message size" in receiver.results[0].error


def test_local_request_encoding_failure_is_infrastructure_error(
    qapp, jobs, monkeypatch
):
    _, service, receiver = jobs
    service.submit("warmup", "unused", None, {})
    wait_for(qapp, lambda: receiver.results)
    original = service._send

    def failing_send(message):
        if message["command"] == "run":
            raise ValueError("Request exceeds transport limit")
        original(message)

    monkeypatch.setattr(service, "_send", failing_send)
    service.submit("oversized", "unused", None, {})
    wait_for(qapp, lambda: len(receiver.results) == 2)
    result = receiver.results[-1]
    assert result.status == "failed" and result.infrastructure_error
    assert "transport limit" in result.error


def test_detached_explicit_job_survives_document_edit_as_stale(qapp, jobs):
    state, service, receiver = jobs
    request = service.submit(
        "save",
        "unused",
        None,
        {"delay": 0.2},
        replace=False,
        cancel_on_document_change=False,
    )
    # Still pending: a save/analysis batch may start after edits to its document.
    state.change()
    wait_for(qapp, lambda: service.active_request is request)
    state.replace()
    wait_for(qapp, lambda: receiver.results)
    assert receiver.results[0].status == "succeeded"
    assert receiver.results[0].data == 42
    assert not receiver.results[0].current


def test_current_must_be_rechecked_at_commit(qapp, jobs):
    state, service, receiver = jobs
    request = service.submit("analysis", "unused", None, {})
    wait_for(qapp, lambda: receiver.results)
    assert receiver.results[0].current
    state.change()
    assert not service.is_current(request)


def test_sequential_batch_preserves_its_captured_document_token(qapp, jobs):
    state, service, receiver = jobs
    captured = state.token
    service.submit(
        "batch-first",
        "unused",
        None,
        {},
        document_token=captured,
        cancel_on_document_change=False,
    )
    wait_for(qapp, lambda: receiver.results)
    state.change()
    second = service.submit(
        "batch-second",
        "unused",
        None,
        {},
        document_token=captured,
        cancel_on_document_change=False,
    )
    wait_for(qapp, lambda: len(receiver.results) == 2)
    assert second.document == captured
    assert receiver.results[-1].status == "succeeded"
    assert not receiver.results[-1].current
