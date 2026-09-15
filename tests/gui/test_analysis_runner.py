"""Shared-job ownership and finite snapshot batch behavior."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tests.gui.analysis_job_fakes import connector_for


@pytest.fixture()
def connector(minimal_optic, qapp):
    return connector_for(minimal_optic)


def entries():
    return [
        {
            "target": target,
            "name": "Ray Fan",
            "constructor_args": {"num_points": 16},
            "view_args": {},
            "theme": "dark",
        }
        for target in ("a", "b", "c")
    ]


def test_run_only_captures_inputs_without_constructing_analysis(connector):
    runner = connector._analysis_runner
    with patch(
        "optiland.analysis.SpotDiagram.__init__",
        side_effect=AssertionError("GUI calculation"),
    ):
        request = runner.run("Spot Diagram", {}, connector.get_optic(), target="page")
    assert request.snapshot is not None
    assert request.handler.endswith(":prepare_analysis")
    assert not request.cancel_on_document_change
    assert runner.busy
    runner.stop()
    assert not runner.busy


def test_run_all_freezes_order_settings_and_document_once(connector, qapp):
    runner, jobs = connector._analysis_runner, connector.calculation_jobs
    configuration = entries()
    token = connector.document_state.token
    runner.run_batch(configuration, connector.get_optic())
    configuration[1]["constructor_args"]["num_points"] = 500
    qapp.processEvents()
    assert [r.target for r in jobs.submissions] == ["a"]
    snapshot = jobs.submissions[0].snapshot
    connector.document_state.change()
    jobs.complete("a")
    qapp.processEvents()
    assert [r.target for r in jobs.submissions] == ["a", "b"]
    assert jobs.submissions[1].document == token
    assert jobs.submissions[1].snapshot is snapshot
    assert jobs.submissions[1].parameters["constructor_args"]["num_points"] == 16
    jobs.complete("b", status="failed", error="Numerical page error")
    qapp.processEvents()
    assert jobs.submissions[-1].target == "c"
    jobs.complete("c")
    qapp.processEvents()
    assert not runner.busy
    assert runner._batch_snapshot is None


def test_stop_drops_entire_unsent_batch(connector, qapp):
    runner, jobs = connector._analysis_runner, connector.calculation_jobs
    runner.run_batch(entries(), connector.get_optic())
    qapp.processEvents()
    runner.stop()
    qapp.processEvents()
    assert [r.target for r in jobs.submissions] == ["a"]
    assert not runner.busy


def test_replacement_stops_batch_but_optical_edit_does_not(connector, qapp):
    runner = connector._analysis_runner
    runner.run_batch(entries(), connector.get_optic())
    qapp.processEvents()
    connector.document_state.change()
    assert runner.busy
    connector.document_state.replace()
    qapp.processEvents()
    assert not runner.busy


def test_removed_queued_page_cannot_run(connector, qapp):
    runner, jobs = connector._analysis_runner, connector.calculation_jobs
    runner.run_batch(entries(), connector.get_optic())
    runner.cancel("b")
    qapp.processEvents()
    jobs.complete("a")
    qapp.processEvents()
    assert [r.target for r in jobs.submissions] == ["a", "c"]
    runner.stop()


def test_registry_and_workload_validation_before_snapshot(connector):
    runner = connector._analysis_runner
    assert len(runner.get_analysis_registry()) == 22
    with patch(
        "optiland_gui.services.analysis_runner.OpticSnapshot.capture"
    ) as capture:
        with pytest.raises(ValueError, match="512 MiB"):
            runner.run("FFT PSF", {"num_rays": 10000}, connector.get_optic())
        capture.assert_not_called()


def test_explicit_rerun_supersedes_unsent_batch_entry(connector, qapp):
    runner, jobs = connector._analysis_runner, connector.calculation_jobs
    runner.run_batch(entries(), connector.get_optic())
    qapp.processEvents()
    runner.run("Ray Fan", {"num_points": 32}, connector.get_optic(), target="b")
    jobs.complete("a")
    qapp.processEvents()
    assert [r.target for r in jobs.submissions] == ["a", "b", "c"]
    assert jobs.submissions[1].parameters["constructor_args"]["num_points"] == 32
    runner.stop()


def test_unrecoverable_worker_exit_stops_remaining_batch(connector, qapp):
    runner, jobs = connector._analysis_runner, connector.calculation_jobs
    runner.run_batch(entries(), connector.get_optic())
    qapp.processEvents()
    jobs.complete(
        "a", status="failed", error="Native library aborted", infrastructure_error=True
    )
    qapp.processEvents()
    assert [r.target for r in jobs.submissions] == ["a"]
    assert not runner.busy


def test_numerical_failure_with_transport_words_does_not_stop_batch(connector, qapp):
    runner, jobs = connector._analysis_runner, connector.calculation_jobs
    runner.run_batch(entries(), connector.get_optic())
    qapp.processEvents()
    jobs.complete("a", status="failed", error="Calculation worker exited unexpectedly.")
    qapp.processEvents()
    assert [r.target for r in jobs.submissions] == ["a", "b"]
    runner.stop()


def test_batch_waiting_for_queue_can_be_stopped_without_later_submission(
    connector, qapp, monkeypatch
):
    runner, jobs = connector._analysis_runner, connector.calculation_jobs
    submit = jobs.submit
    monkeypatch.setattr(
        jobs,
        "submit",
        lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("Calculation queue is full")
        ),
    )
    runner.run_batch(entries(), connector.get_optic())
    qapp.processEvents()
    assert runner.busy and not jobs.submissions
    runner.stop()
    monkeypatch.setattr(jobs, "submit", submit)
    runner._dispatch_batch()
    assert not runner.busy and not jobs.submissions


def test_batch_closed_service_reports_first_failure_and_cancels_rest(
    connector, qapp, monkeypatch
):
    runner, jobs = connector._analysis_runner, connector.calculation_jobs
    finished, states = [], []
    runner.finished.connect(lambda target, result: finished.append((target, result)))
    runner.state_changed.connect(lambda target, state: states.append((target, state)))
    monkeypatch.setattr(
        jobs,
        "submit",
        lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("Calculation service is closed.")
        ),
    )
    runner.run_batch(entries(), connector.get_optic())
    qapp.processEvents()
    assert finished[0][0] == "a" and finished[0][1].status == "failed"
    assert ("b", "cancelled") in states and ("c", "cancelled") in states
    assert not runner.busy and runner._batch_snapshot is None
