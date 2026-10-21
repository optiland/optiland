"""Unknown publication outcomes remain owned until verification and cleanup."""

from __future__ import annotations

import time

from PySide6.QtCore import QTimer

from .test_calculation_jobs import wait_for
from .test_file_save_export import file_operations as file_operations
from .test_file_save_export import read_optic, stall


def test_unknown_publication_reconciles_and_close_drains_without_blocking(
    qapp, tmp_path, file_operations
):
    connector, operations, progress, results, notifications = file_operations
    stall(operations, "publish_output", "after", 10)
    operations.jobs._publication_deadline_ms = 500
    path = tmp_path / "saved.json"
    states, ticks, settled = [], [], []
    operations.state_changed.connect(lambda *args: states.append(args))
    operations.completed.connect(lambda: settled.append(True))
    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: ticks.append(time.monotonic()))
    timer.start()
    operations.request_output(path)
    wait_for(qapp, lambda: "Fixture completed publish_output" in progress)
    started = time.monotonic()
    operations._closing = True
    assert time.monotonic() - started < 0.1
    assert operations.busy and not settled
    assert not states[-1][2]
    wait_for(qapp, lambda: not operations.busy, timeout=30)
    timer.stop()
    assert settled and len(ticks) >= 10
    assert (
        read_optic(path).surfaces.num_surfaces
        == connector.get_optic().surfaces.num_surfaces
    )
    assert any(result.outcome_unknown for result in results)
    assert any(kind == "success" for _, kind in notifications)
    assert not any("cancelled" in text.lower() for text, _ in notifications)
    assert not connector.is_modified()
    assert not list(tmp_path.glob(".optiland-*.tmp"))


def test_unconfirmed_publication_preserves_destination_and_aborts_close(
    qapp, tmp_path, file_operations
):
    connector, operations, progress, results, notifications = file_operations
    path = tmp_path / "saved.json"
    path.write_bytes(b"previous valid contents")
    stall(operations, "publish_output", "before", 10)
    operations.jobs._publication_deadline_ms = 500
    aborted = []
    operations.failed.connect(lambda reason: aborted.append(True))
    operations.request_output(path)
    wait_for(qapp, lambda: "Fixture entered publish_output" in progress)
    operations._closing = True
    wait_for(qapp, lambda: not operations.busy, timeout=30)
    assert path.read_bytes() == b"previous valid contents"
    assert connector.is_modified()
    assert aborted == [True]
    assert any("could not be confirmed" in text for text, severity in notifications)
    assert not list(tmp_path.glob(".optiland-*.tmp"))


def test_rejected_publication_dispatch_cleans_owned_preparation(
    qapp, tmp_path, file_operations, monkeypatch
):
    connector, operations, _, _, notifications = file_operations
    destination = tmp_path / "saved.json"
    destination.write_bytes(b"previous valid file")
    submit = operations.jobs.submit

    def reject_publication(*args, **kwargs):
        if args[1].endswith(":publish_output"):
            raise RuntimeError("publication dispatch unavailable")
        return submit(*args, **kwargs)

    monkeypatch.setattr(operations.jobs, "submit", reject_publication)
    operations.request_output(destination)
    wait_for(qapp, lambda: not operations.busy)
    assert destination.read_bytes() == b"previous valid file"
    assert connector.is_modified()
    assert not list(tmp_path.glob(".optiland-*.tmp"))
    assert any("publication dispatch unavailable" in text for text, _ in notifications)
