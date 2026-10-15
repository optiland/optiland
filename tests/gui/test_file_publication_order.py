"""A retried commit must not overwrite a newer write to the same destination."""

from __future__ import annotations

import pytest

from optiland.fileio import load_zemax_file
from tests.gui.test_calculation_jobs import wait_for
from tests.gui.test_file_operations import (
    edit_comment,
    read_optic,
)
from tests.gui.test_file_operations import (
    file_operations as file_operations,
)


@pytest.mark.parametrize(
    "first_format,second_format,closing",
    [
        ("optiland", "optiland", False),
        ("zemax", "zemax", True),
        ("optiland", "zemax", False),
    ],
)
def test_same_destination_publication_waits_for_older_retry(
    qapp, tmp_path, file_operations, monkeypatch, first_format, second_format, closing
):
    connector, operations, _, _, _ = file_operations
    jobs = operations.jobs
    destination = tmp_path / "output.dat"
    original_submit = jobs.submit
    held = []
    release = False

    def backpressure(*args, **kwargs):
        operation = kwargs["context"]
        if args[1].endswith(":publish_output") and not held:
            held.append(operation)
        if (
            held
            and operation is held[0]
            and operation.phase == "publishing"
            and not release
        ):
            raise RuntimeError("Calculation queue is full; retry later.")
        return original_submit(*args, **kwargs)

    monkeypatch.setattr(jobs, "submit", backpressure)
    edit_comment(connector, "older contents")
    first = operations.request_output(destination, first_format).context
    wait_for(qapp, lambda: bool(held))
    connector.set_surface_data(1, connector.COL_RADIUS, "60")
    edit_comment(connector, "newer contents")
    second = operations.request_output(destination, second_format).context
    wait_for(qapp, lambda: second.phase != "preparing")

    assert first.phase == "publishing"
    assert second.phase == "waiting"
    assert operations.busy and operations.has_pending_writes
    assert not destination.exists()
    assert first.staged_identity is not None and second.staged_identity is not None
    assert not any(
        request.context is second and request.handler.endswith(":publish_output")
        for request in jobs._pending
    )
    # Both stages have reached the accepted commit phase: cancellation must not
    # abandon them, and closing must drain both in destination order.
    operations.cancel_pending()
    settled = []
    if closing:
        operations.settled.connect(lambda: settled.append(True))
        operations.begin_close()
        assert not settled

    published = []
    jobs.finished.connect(
        lambda result: (
            published.append(result.request.context)
            if result.request.handler.endswith(":publish_output")
            and result.status == "succeeded"
            else None
        )
    )
    release = True
    wait_for(qapp, lambda: not operations.busy, timeout=30)
    assert published == [first, second]
    restored = (
        read_optic(destination)
        if second_format == "optiland"
        else load_zemax_file(str(destination))
    )
    assert float(restored.surfaces[1].geometry.radius) == 60
    assert connector.is_modified() == (second_format != "optiland")
    assert not list(tmp_path.glob(".optiland-*.tmp"))
    if closing:
        assert settled


def test_unrelated_destination_does_not_wait_for_a_retried_publication(
    qapp, tmp_path, file_operations, monkeypatch
):
    connector, operations, _, _, _ = file_operations
    jobs = operations.jobs
    older_path, newer_path = tmp_path / "older.json", tmp_path / "newer.json"
    original_submit = jobs.submit
    held = []
    release = False

    def backpressure(*args, **kwargs):
        operation = kwargs["context"]
        if (
            args[1].endswith(":publish_output")
            and operation.path == str(older_path)
            and not release
        ):
            held.append(True)
            raise RuntimeError("Calculation queue is full; retry later.")
        return original_submit(*args, **kwargs)

    monkeypatch.setattr(jobs, "submit", backpressure)
    edit_comment(connector, "older contents")
    operations.request_output(older_path)
    wait_for(qapp, lambda: bool(held))
    edit_comment(connector, "newer contents")
    operations.request_output(newer_path)
    wait_for(qapp, lambda: not operations.path_busy(newer_path))
    assert not older_path.exists()
    assert read_optic(newer_path).surfaces[1].comment == "newer contents"
    assert connector.get_current_filepath() == str(newer_path)
    release = True
    wait_for(qapp, lambda: not operations.busy, timeout=30)
    assert read_optic(older_path).surfaces[1].comment == "older contents"
    assert connector.get_current_filepath() == str(newer_path)
    assert not list(tmp_path.glob(".optiland-*.tmp"))
