"""File controls submit owned jobs and keep cancellation feedback visible."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QFileDialog, QWidget

from optiland_gui.main_window import MainWindow
from optiland_gui.widgets.file_operation_status import FileOperationStatus

from .test_calculation_jobs import wait_for
from .test_file_save_export import file_operations as file_operations


class FileWindow(QWidget):
    new_system_action = MainWindow.new_system_action
    open_system_action = MainWindow.open_system_action
    import_zemax_action = MainWindow.import_zemax_action
    import_codev_action = MainWindow.import_codev_action
    save_system_action = MainWindow.save_system_action
    save_system_as_action = MainWindow.save_system_as_action
    export_zemax_action = MainWindow.export_zemax_action
    export_codev_action = MainWindow.export_codev_action
    _confirm_discard_changes = MainWindow._confirm_discard_changes


@pytest.fixture
def file_window(qapp):
    window = FileWindow()
    window.connector = SimpleNamespace(
        output_operations=SimpleNamespace(
            request_load=Mock(), request_output=Mock(), resolve_candidate=Mock()
        ),
        is_modified=lambda: True,
        get_current_filepath=lambda: None,
        document_state=SimpleNamespace(edit_token="original edit"),
        new_system=Mock(),
    )
    window._update_project_name_in_title_bar = Mock()
    yield window
    window.deleteLater()


class OperationSignals(QObject):
    state_changed = Signal(str, bool, bool)

    def __init__(self):
        super().__init__()
        self.cancel_pending = Mock()


@pytest.mark.parametrize(
    "action,extension,format_args",
    [
        ("save_system_action", ".json", ()),
        ("save_system_as_action", ".json", ()),
        ("export_zemax_action", ".zmx", ("zemax",)),
        ("export_codev_action", ".seq", ("codev",)),
    ],
)
@pytest.mark.parametrize("filename", ["", "output", "with suffix"])
def test_output_controls_submit_selected_destination_without_writing_on_gui(
    file_window, monkeypatch, action, extension, format_args, filename
):
    path = "output" + extension if filename == "with suffix" else filename
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *a: (path, "Optiland JSON Files (*.json)"),
    )
    getattr(file_window, action)()
    request = file_window.connector.output_operations.request_output
    if path:
        expected = path if path.endswith(extension) else path + extension
        request.assert_called_once_with(expected, *format_args)
    else:
        request.assert_not_called()


def test_save_uses_current_destination_without_dialog(file_window, monkeypatch):
    file_window.connector.get_current_filepath = lambda: "existing.json"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *a: pytest.fail("Unexpected Save As")
    )
    file_window.save_system_action()
    file_window.connector.output_operations.request_output.assert_called_once_with(
        "existing.json"
    )


def test_file_status_delays_spinner_and_cancels_only_preparation(qapp):
    operations = OperationSignals()
    status = FileOperationStatus(operations)
    assert status.isHidden()
    operations.state_changed.emit("Preparing output", True, True)
    assert status.isVisible() and status.label.text() == "Preparing output"
    assert status.activity.isHidden() and status.timer.isActive()
    status.cancel.click()
    operations.cancel_pending.assert_called_once()
    wait_for(qapp, status.activity.isVisible)
    operations.state_changed.emit("Publishing output", True, False)
    assert status.cancel.isHidden() and status.activity.isVisible()
    operations.state_changed.emit("Saved", False, False)
    assert status.activity.isHidden() and not status.timer.isActive()
    status._show_activity()  # A queued timeout after completion stays harmless.
    assert status.activity.isHidden()
    status.deleteLater()


def test_fast_file_completion_never_flashes_a_spinner(qapp):
    operations = OperationSignals()
    status = FileOperationStatus(operations)
    operations.state_changed.emit("Reading", True, True)
    operations.state_changed.emit("Opened", False, False)
    assert status.activity.isHidden() and not status.timer.isActive()
    assert status.cancel.isHidden() and status.label.text() == "Opened"
    status.deleteLater()
