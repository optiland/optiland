"""Real-process Save As ordering and close-after-save data preservation."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLineEdit, QMessageBox, QWidget

from optiland_gui.main_window import MainWindow
from optiland_gui.optiland_connector import OptilandConnector

from .test_calculation_jobs import wait_for
from .test_file_operations import edit_comment, read_optic, stall


class ClosingWindow(QWidget):
    closeEvent = MainWindow.closeEvent
    _calculations_stopped = MainWindow._calculations_stopped
    _files_settled = MainWindow._files_settled
    _file_close_aborted = MainWindow._file_close_aborted
    _confirm_discard_changes = MainWindow._confirm_discard_changes
    _confirm_close_intent = MainWindow._confirm_close_intent


@pytest.fixture
def owner(qapp, minimal_optic, monkeypatch):
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args: pytest.fail("Unexpected discard prompt")
    )
    connector = OptilandConnector()
    connector._optic = minimal_optic
    connector.opticLoaded.emit()
    operations = connector.file_operations
    operations.jobs._cancel_grace_ms = 40
    messages, results, progress = [], [], []
    operations.service._toast = lambda text, severity: messages.append((text, severity))
    operations.jobs.finished.connect(results.append)
    operations.jobs.progress.connect(
        lambda request, message: progress.append(message["stage"])
    )
    window = ClosingWindow()
    window.connector = connector
    window.panel_manager = SimpleNamespace(
        python_terminal=SimpleNamespace(shutdown_kernel=lambda: None)
    )
    yield connector, operations, window, messages, results, progress
    window.hide()
    operations.cancel_pending()
    wait_for(qapp, lambda: not operations.busy, timeout=30)
    for jobs in connector.calculation_services:
        jobs.shutdown()
    wait_for(
        qapp,
        lambda: all(job._process is None for job in connector.calculation_services),
    )
    window.deleteLater()


def test_older_reconciliation_cannot_replace_newer_save_as_choice(
    qapp, tmp_path, owner
):
    connector, operations, window, messages, results, progress = owner
    jobs = operations.jobs
    jobs._publication_deadline_ms = 250
    jobs._command = [
        sys.executable,
        "-u",
        str(Path(__file__).with_name("save_order_worker_fixture.py")),
    ]
    older, newer = tmp_path / "older.json", tmp_path / "newer.json"
    edit_comment(connector, "older snapshot")
    operations.request_output(older)
    edit_comment(connector, "newer snapshot")
    operations.request_output(newer)
    wait_for(qapp, lambda: not operations.busy, timeout=30)
    assert any(result.outcome_unknown for result in results)
    handlers = [result.request.handler.rsplit(":", 1)[-1] for result in results]
    assert handlers.index("reconcile_output") > max(
        index for index, handler in enumerate(handlers) if handler == "publish_output"
    )
    assert connector.get_current_filepath() == str(newer)
    assert not connector.is_modified()
    assert read_optic(older).surfaces[1].comment == "older snapshot"
    assert read_optic(newer).surfaces[1].comment == "newer snapshot"


@pytest.mark.parametrize("outcome", ["cancelled", "failed"])
def test_later_unsuccessful_save_as_retains_last_successful_destination(
    qapp, tmp_path, owner, outcome
):
    connector, operations, window, messages, results, progress = owner
    older = tmp_path / "confirmed.json"
    edit_comment(connector, "confirmed contents")
    operations.request_output(older)
    later = (
        tmp_path / "missing" / "failed.json"
        if outcome == "failed"
        else tmp_path / "cancelled.json"
    )
    request = operations.request_output(later)
    if outcome == "cancelled":
        operations.jobs.cancel_target(request.target)
    wait_for(qapp, lambda: not operations.busy, timeout=30)
    assert connector.get_current_filepath() == str(older)
    assert read_optic(older).surfaces[1].comment == "confirmed contents"
    assert not later.exists()


def test_later_public_synchronous_save_keeps_its_active_destination(
    qapp, tmp_path, owner
):
    connector, operations, window, messages, results, progress = owner
    older, newer = tmp_path / "async.json", tmp_path / "script.json"
    edit_comment(connector, "async snapshot")
    operations.request_output(older)
    edit_comment(connector, "later script snapshot")
    connector.save_optic_to_file(str(newer))
    assert connector.get_current_filepath() == str(newer)
    wait_for(qapp, lambda: not operations.busy, timeout=30)
    assert connector.get_current_filepath() == str(newer)
    assert not connector.is_modified()
    assert read_optic(older).surfaces[1].comment == "async snapshot"
    assert read_optic(newer).surfaces[1].comment == "later script snapshot"


def test_close_drains_requested_save_preparation(qapp, tmp_path, owner):
    connector, operations, window, messages, results, progress = owner
    stall(operations, "prepare_output", "after", 0.25)
    path = tmp_path / "saved.json"
    path.write_text("old contents")
    edit_comment(connector, "requested save")
    operations.request_output(path)
    window.show()
    assert not window.close()
    wait_for(qapp, lambda: not window.isVisible(), timeout=30)
    assert read_optic(path).surfaces[1].comment == "requested save"
    assert not connector.is_modified()
    assert not any(result.status == "cancelled" for result in results)
    assert not list(tmp_path.glob(".optiland-*.tmp"))


def test_failed_requested_save_aborts_close_and_allows_retry(qapp, tmp_path, owner):
    connector, operations, window, messages, results, progress = owner
    edit_comment(connector, "preserve on failure")
    operations.request_output(tmp_path / "missing" / "failed.json")
    window.show()
    window.close()
    wait_for(qapp, lambda: not operations.busy, timeout=30)
    assert window.isVisible()
    assert connector.is_modified()
    assert not connector._calculation_shutdown_started
    assert not operations._closing
    assert any(
        "Close cancelled" in text and severity == "error" for text, severity in messages
    )
    retry = tmp_path / "retry.json"
    assert operations.request_output(retry) is not None
    window.close()
    wait_for(qapp, lambda: not window.isVisible(), timeout=30)
    assert read_optic(retry).surfaces[1].comment == "preserve on failure"


def test_edit_while_waiting_for_save_aborts_close(qapp, tmp_path, owner):
    connector, operations, window, messages, results, progress = owner
    stall(operations, "prepare_output", "after", 0.25)
    path = tmp_path / "captured.json"
    edit_comment(connector, "captured contents")
    operations.request_output(path)
    window.show()
    window.close()
    edit_comment(connector, "new unsaved edit while waiting")
    wait_for(qapp, lambda: not operations.busy, timeout=30)
    assert window.isVisible()
    assert connector.is_modified()
    assert not connector._calculation_shutdown_started
    assert read_optic(path).surfaces[1].comment == "captured contents"
    assert connector.get_optic().surfaces[1].comment == "new unsaved edit while waiting"
    assert any("document changed while waiting" in text.lower() for text, _ in messages)


def test_explicit_cancel_during_close_preserves_open_document(qapp, tmp_path, owner):
    connector, operations, window, messages, results, progress = owner
    stall(operations, "prepare_output", "after", 10)
    path = tmp_path / "cancelled.json"
    path.write_text("old contents")
    edit_comment(connector, "unsaved contents")
    operations.request_output(path)
    window.show()
    window.close()
    wait_for(qapp, lambda: "Fixture completed prepare_output" in progress, timeout=30)
    operations.cancel_pending()
    wait_for(qapp, lambda: not operations.busy, timeout=30)
    assert window.isVisible()
    assert connector.is_modified()
    assert path.read_text() == "old contents"
    assert not connector._calculation_shutdown_started
    assert any("Close cancelled" in text for text, _ in messages)


def test_edit_queued_at_settlement_is_checked_before_shutdown(qapp, tmp_path, owner):
    connector, operations, window, messages, results, progress = owner
    edit_comment(connector, "saved snapshot")
    operations.request_output(tmp_path / "saved.json")
    operations.settled.connect(
        lambda: QTimer.singleShot(
            0,
            lambda: edit_comment(connector, "edit after settlement"),
        )
    )
    window.show()
    window.close()
    wait_for(qapp, lambda: not operations.busy, timeout=30)
    wait_for(qapp, lambda: not connector._calculation_shutdown_started)
    assert window.isVisible()
    assert connector.is_modified()
    assert connector.get_optic().surfaces[1].comment == "edit after settlement"


def test_dirty_close_asks_once_per_intent_and_cancel_keeps_document(
    qapp, owner, monkeypatch
):
    connector, operations, window, messages, results, progress = owner
    prompts = []
    responses = iter(
        [QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Yes]
    )

    def answer(parent, title, text, buttons, default):
        prompts.append(text)
        assert default == QMessageBox.StandardButton.Cancel
        return next(responses)

    monkeypatch.setattr(QMessageBox, "question", answer)
    cancellations = []
    cancel = operations.jobs.cancel_cancellable

    def record_cancel(**kwargs):
        cancellations.append(kwargs)
        return cancel(**kwargs)

    monkeypatch.setattr(operations.jobs, "cancel_cancellable", record_cancel)
    edit_comment(connector, "unsaved document")
    window.show()
    window.close()
    wait_for(qapp, lambda: len(prompts) == 1)
    assert "Closing will discard" in prompts[0]
    assert window.isVisible()
    assert connector.is_modified()
    assert not getattr(connector, "_calculation_shutdown_started", False)
    assert not operations._closing
    assert not cancellations
    window.close()
    wait_for(qapp, lambda: not window.isVisible())
    assert len(prompts) == 2


def test_edit_during_discard_dialog_cannot_be_discarded_by_old_confirmation(
    qapp, owner, monkeypatch
):
    connector, operations, window, messages, results, progress = owner
    edit_comment(connector, "before confirmation")

    def answer(*args):
        edit_comment(connector, "edit while dialog was open")
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", answer)
    window.show()
    window.close()
    assert window.isVisible() and window.isEnabled()
    assert not getattr(connector, "_calculation_shutdown_started", False)
    assert connector.get_optic().surfaces[1].comment == "edit while dialog was open"
    assert any("changed during confirmation" in text for text, _ in messages)


def test_final_worker_shutdown_disables_editing_until_close(qapp, owner):
    connector, operations, window, messages, results, progress = owner
    jobs = operations.jobs
    jobs._cancel_grace_ms = 500
    jobs._command = [
        sys.executable,
        "-u",
        str(Path(__file__).with_name("calculation_worker_fixture.py")),
    ]
    editor = QLineEdit("preserved", window)
    editor.textChanged.connect(lambda text: edit_comment(connector, text))
    window.show()
    editor.setFocus()
    jobs.submit("stubborn-preview", "unused", None, {"delay": 10})
    wait_for(qapp, lambda: jobs.active_request is not None)
    original = connector.get_optic().surfaces[1].comment
    window.close()
    wait_for(qapp, lambda: getattr(window, "_calculation_shutdown_requested", False))
    assert window.isVisible() and jobs._process is not None
    assert not window.isEnabled() and not editor.isEnabled()
    QTest.keyClicks(editor, "must not edit")
    assert editor.text() == "preserved"
    assert connector.get_optic().surfaces[1].comment == original
    wait_for(qapp, lambda: not window.isVisible())
