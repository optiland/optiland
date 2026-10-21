"""Standalone output delivery never destroys an owner before a write finishes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QWidget

from optiland_gui.main_window import MainWindow

from .test_calculation_jobs import wait_for
from .test_file_save_export import edit_comment
from .test_file_save_export import file_operations as file_operations


class WriteWindow(QWidget):
    closeEvent = MainWindow.closeEvent
    _calculations_stopped = MainWindow._calculations_stopped
    _output_settled = MainWindow._output_settled
    _output_failure_during_close = MainWindow._output_failure_during_close


@pytest.mark.parametrize("outcome", ["saved", "edited", "failed"])
def test_standalone_output_waits_and_keeps_failed_or_newer_document_open(
    qapp, file_operations, tmp_path, monkeypatch, outcome
):
    connector, operations, _, _, _ = file_operations
    window = WriteWindow()
    window.connector = connector
    window.panel_manager = SimpleNamespace(python_terminal=None)
    window.show()
    dispatch = operations.jobs._dispatch
    monkeypatch.setattr(operations.jobs, "_dispatch", lambda: None)
    path = tmp_path / ("missing" if outcome == "failed" else "") / "output.json"
    operations.request_output(path)
    window.close()
    assert window.isVisible()
    if outcome == "edited":
        edit_comment(connector, "preserve newer unsaved edit")
    monkeypatch.setattr(operations.jobs, "_dispatch", dispatch)
    operations.jobs._dispatch()
    wait_for(qapp, lambda: not operations.busy)
    if outcome == "saved":
        wait_for(qapp, lambda: not window.isVisible())
        assert path.exists() and not connector.is_modified()
    else:
        qapp.processEvents()
        assert window.isVisible() and connector.is_modified()
    window.hide()
    window.deleteLater()
