"""File controls submit owned jobs and keep cancellation feedback visible."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtCore import QObject, QSettings, Signal
from PySide6.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QWidget

from optiland_gui.main_window import MainWindow
from optiland_gui.widgets.file_operation_status import FileOperationStatus

from .test_calculation_jobs import wait_for
from .test_file_open_import import file_operations as file_operations


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
    _resolve_file_candidate = MainWindow._resolve_file_candidate


@pytest.fixture
def file_window(qapp):
    window = FileWindow()
    window.connector = SimpleNamespace(
        file_operations=SimpleNamespace(
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
    "action,format_args",
    [
        ("open_system_action", ()),
        ("import_zemax_action", ("zemax",)),
        ("import_codev_action", ("codev",)),
    ],
)
@pytest.mark.parametrize("choice", ["discard", "keep", "no file"])
def test_open_controls_preserve_discard_choice_and_submit_worker(
    file_window, monkeypatch, action, format_args, choice
):
    reply = QMessageBox.StandardButton
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *a: reply.Cancel if choice == "keep" else reply.Yes,
    )
    picker = Mock(return_value=("" if choice == "no file" else "input.dat", ""))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", picker)
    getattr(file_window, action)()
    request = file_window.connector.file_operations.request_load
    if choice == "discard":
        request.assert_called_once_with("input.dat", *format_args)
    else:
        request.assert_not_called()
    if choice == "keep":
        picker.assert_not_called()


@pytest.mark.parametrize("accepted", [False, True])
def test_candidate_confirmation_retains_pre_dialog_edit_token(
    file_window, monkeypatch, accepted
):
    def answer(*args):
        file_window.connector.document_state.edit_token = "edit while modal"
        return (
            QMessageBox.StandardButton.Yes
            if accepted
            else QMessageBox.StandardButton.Cancel
        )

    monkeypatch.setattr(QMessageBox, "question", answer)
    file_window._resolve_file_candidate("candidate", "input.json")
    file_window.connector.file_operations.resolve_candidate.assert_called_once_with(
        "candidate", accepted, "original edit"
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


def test_main_window_finalization_connects_file_status_and_confirmation(
    qapp, file_operations, tmp_path, monkeypatch
):
    from optiland_gui import main_window
    from optiland_gui.widgets.command_palette import CommandRegistry

    connector, operations, _, _, _ = file_operations
    window = QMainWindow()
    window.connector = connector
    window.settings = QSettings(
        str(tmp_path / "settings.ini"), QSettings.Format.IniFormat
    )
    window.command_registry = CommandRegistry()
    window.action_manager = SimpleNamespace(get_action=lambda name: None)
    window.panel_manager = SimpleNamespace(connect_signals=Mock())
    window.load_stylesheets = Mock()
    window._connect_dock_animations = Mock()
    window._register_palette_commands = Mock()
    window._update_project_name_in_title_bar = Mock()
    window._resolve_file_candidate = lambda identifier, path: (
        MainWindow._resolve_file_candidate(window, identifier, path)
    )
    monkeypatch.setattr(main_window._log_handler, "install", lambda *args: None)
    MainWindow._finalize_setup(window)
    try:
        assert window.file_operation_status.parent() is window.statusBar()
        operations.state_changed.emit("Reading input.json", True, True)
        assert window.file_operation_status.label.text() == "Reading input.json"
        assert not window.file_operation_status.cancel.isHidden()
        window._update_project_name_in_title_bar.reset_mock()
        operations.completed.emit()
        window._update_project_name_in_title_bar.assert_called_once()
        monkeypatch.setattr(
            QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Cancel
        )
        resolve = Mock()
        monkeypatch.setattr(operations, "resolve_candidate", resolve)
        token = connector.document_state.edit_token
        operations.candidate_conflict.emit("candidate", "input.json")
        resolve.assert_called_once_with("candidate", False, token)
    finally:
        window.deleteLater()
