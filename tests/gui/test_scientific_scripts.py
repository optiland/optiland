"""Scientific ownership, bounded output and preserved GUI-extension contracts."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QWidget

from optiland_gui.optiland_connector import OptilandConnector
from optiland_gui.services.scientific_scripts import ScientificScripts
from optiland_gui.services.scientific_tasks import (
    LINE_CHARACTERS,
    OUTPUT_BYTES,
    OUTPUT_LINES,
    BoundedOutput,
    GuiCommands,
)

from .test_calculation_jobs import wait_for


def test_script_worker_crash_reports_failure_without_an_applicable_candidate(
    qapp, scripts
):
    connector, service, output, statuses, results = scripts
    original = connector.get_optic()
    service.run("import os\nos._exit(3)")
    wait_for(qapp, lambda: results, timeout=30)
    assert results[-1].status == "failed"
    assert "Calculation worker exited unexpectedly" in output[-1]
    assert not service.can_apply()
    assert connector.get_optic() is original


def test_script_candidate_restore_failure_preserves_model_and_history(
    scripts, monkeypatch
):
    from optiland_gui.services.job_records import OpticSnapshot

    connector, service, output, statuses, results = scripts
    original = connector.get_optic()
    monkeypatch.setattr(service.jobs, "_dispatch", lambda: None)
    service.run("print('result')")
    service.candidate = OpticSnapshot.capture(original)
    revision = connector._undo_redo_manager.revision

    def reject(*args, **kwargs):
        raise ValueError("candidate restore failed")

    monkeypatch.setattr(OpticSnapshot, "restore", reject)
    assert not service.apply_candidate()
    assert connector.get_optic() is original
    assert connector._undo_redo_manager.revision == revision
    assert "candidate restore failed" in statuses[-1][0]


@pytest.fixture
def scripts(qapp, minimal_optic):
    connector = OptilandConnector()
    connector._optic = minimal_optic
    service = ScientificScripts(connector)
    service.jobs._cancel_grace_ms = 40
    output, statuses, results = [], [], []
    service.output_changed.connect(output.append)
    service.status_changed.connect(lambda *args: statuses.append(args))
    service.jobs.finished.connect(results.append)
    yield connector, service, output, statuses, results
    connector._calculation_shutdown_started = True
    for jobs in tuple(connector.calculation_services):
        jobs.shutdown()
    wait_for(
        qapp,
        lambda: all(jobs._process is None for jobs in connector.calculation_services),
    )


@pytest.mark.parametrize(
    "text",
    ["😀" * 100000, "x\n" * 100000, "z" * 1000000],
    ids=["unicode", "many-lines", "long-line"],
)
def test_output_limits_bytes_lines_and_single_line(text):
    output = BoundedOutput()
    assert output.write(text) == len(text)
    assert output.truncated
    assert len(output.text.encode("utf-8")) <= OUTPUT_BYTES
    assert output.text.count("\n") <= OUTPUT_LINES
    assert max(map(len, output.text.splitlines())) <= LINE_CHARACTERS
    assert "truncated" in output.value


def test_gui_command_vocabulary_and_count_are_bounded():
    commands = GuiCommands()
    with pytest.raises(ValueError):
        commands.show_panel("arbitrary widget")
    with pytest.raises(AttributeError):
        commands.get_main_window()
    for _ in range(32):
        commands.refresh_views()
    with pytest.raises(ValueError, match="32"):
        commands.show_panel("viewer")


def test_rejected_new_run_revokes_older_intent(qapp, scripts):
    connector, service, output, statuses, results = scripts
    first = service.run("print('older request')")
    assert service.run("x" * (1024 * 1024 + 1)) is None
    wait_for(qapp, lambda: results)
    assert results[-1].request is first
    assert results[-1].status == "cancelled"
    assert service.request is None and not service.can_apply()


def test_scientific_run_keeps_gui_and_preview_queue_free_and_apply_is_explicit(
    qapp, scripts
):
    connector, service, output, statuses, results = scripts
    original = connector.get_optic()
    connector.calculation_jobs._command = [
        sys.executable,
        "-u",
        str(Path(__file__).with_name("calculation_worker_fixture.py")),
    ]
    previews, ticks, commands = [], [], []
    connector.calculation_jobs.finished.connect(previews.append)
    service.commands_ready.connect(commands.append)
    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: ticks.append(time.monotonic()))
    timer.start()
    request = service.run(
        "import time\nprint('started')\n"
        "optic.surfaces[1].comment = 'script result'\n"
        "time.sleep(0.5)\ngui.show_panel('viewer')\nprint('done')"
    )
    connector.calculation_jobs.submit("preview", "unused", None, {"value": 123})
    wait_for(qapp, lambda: results and previews)
    timer.stop()
    assert results[-1].request is request
    assert previews[-1].data == 123
    assert len(ticks) > 15
    assert connector.get_optic() is original
    assert original.surfaces[1].comment != "script result"
    assert "done" in output[-1]
    assert commands == [(("show_panel", "viewer"),)]
    assert service.can_apply()
    assert service.apply_candidate()
    assert connector.get_optic().surfaces[1].comment == "script result"
    assert connector._undo_redo_manager.can_undo()
    connector.undo()
    assert connector.get_optic().surfaces[1].comment != "script result"


def test_comment_edit_revokes_candidate_and_commands(qapp, scripts):
    connector, service, output, statuses, results = scripts
    commands = []
    service.commands_ready.connect(commands.append)
    service.run("import time\ntime.sleep(.2)\ngui.show_panel('viewer')")
    connector.get_optic().surfaces[1].comment = "intervening edit"
    connector.notify_change("metadata", surface_indices=(1,), columns=(1,))
    wait_for(qapp, lambda: results)
    assert service.candidate is not None
    assert not service.can_apply()
    assert not service.apply_candidate()
    assert connector.get_optic().surfaces[1].comment == "intervening edit"
    assert not commands


def test_script_error_preserves_document_and_bounds_output(qapp, scripts):
    connector, service, output, statuses, results = scripts
    original = connector.get_optic()
    service.run("print('x' * 1000000)\nraise ValueError('failed calculation')")
    wait_for(qapp, lambda: results)
    assert connector.get_optic() is original
    assert service.candidate is None
    assert "Output truncated" in output[-1]
    assert "failed calculation" in output[-1]
    assert len(output[-1]) < 2 * OUTPUT_BYTES + 100


def test_stop_and_restart_reap_isolated_process_and_preserve_document(qapp, scripts):
    connector, service, output, statuses, results = scripts
    original = connector.get_optic()
    service.run("print('loop entered')\nwhile True: pass")
    wait_for(qapp, lambda: any("loop entered" in text for text in output))
    started = time.monotonic()
    service.cancel()
    assert time.monotonic() - started < 0.1
    wait_for(qapp, lambda: results)
    assert results[-1].status == "cancelled"
    assert connector.get_optic() is original
    old = service.jobs
    service.restart()
    wait_for(qapp, lambda: service.jobs is not old)
    assert old not in connector.calculation_services
    assert service.jobs in connector.calculation_services
    service.run("print('new process')")
    wait_for(qapp, lambda: service.candidate is not None)
    assert "new process" in output[-1]


def test_backend_change_revokes_every_registered_service_without_persisted_edit(
    qapp, scripts, monkeypatch
):
    from optiland_gui.services.job_records import BackendConfig

    connector, service, output, statuses, results = scripts
    called = []
    for index, jobs in enumerate(connector.calculation_services):
        monkeypatch.setattr(
            jobs, "cancel_cancellable", lambda i=index: called.append(i)
        )
    token = connector.document_state.edit_token
    monkeypatch.setattr(BackendConfig, "capture", lambda: BackendConfig("torch"))
    connector._check_calculation_backend()
    assert called == [0, 1]
    assert connector.document_state.edit_token == token


def test_main_window_close_waits_for_both_services(qapp, scripts):
    from optiland_gui.main_window import MainWindow

    connector, service, output, statuses, results = scripts
    closed = []

    class Window(QWidget):
        closeEvent = MainWindow.closeEvent
        _calculations_stopped = MainWindow._calculations_stopped
        _files_settled = MainWindow._files_settled
        _file_close_aborted = MainWindow._file_close_aborted
        _confirm_discard_changes = MainWindow._confirm_discard_changes
        _confirm_close_intent = MainWindow._confirm_close_intent

    window = Window()
    window.connector = connector
    window.panel_manager = SimpleNamespace(
        python_terminal=SimpleNamespace(shutdown_kernel=lambda: closed.append(True))
    )
    window.show()
    service.run("print('loop entered')\nwhile True: pass")
    wait_for(qapp, lambda: any("loop entered" in text for text in output))
    assert not window.close()
    assert window.isVisible()
    wait_for(qapp, lambda: not window.isVisible())
    assert closed == [True]
    assert all(jobs._process is None for jobs in connector.calculation_services)
    window.deleteLater()


def test_original_console_keeps_real_connector_iface_and_editor_execution(
    qapp, monkeypatch
):
    from optiland_gui.widgets.python_terminal import PythonTerminalWidget

    connector = OptilandConnector()
    iface = object()
    widget = PythonTerminalWidget(
        custom_variables={"connector": connector, "iface": iface}
    )
    try:
        namespace = widget.kernel_manager.kernel.shell.user_ns
        assert namespace["connector"] is connector
        assert namespace["iface"] is iface
        submitted = []
        monkeypatch.setattr(
            widget.kernel_client,
            "execute",
            lambda code, **kwargs: submitted.append(code),
        )
        widget._get_current_editor().setPlainText("connector.get_optic()")
        widget._run_script_from_editor()
        assert submitted == ["connector.get_optic()"]
        assert widget.scientific_panel.service.jobs is not connector.calculation_jobs
        widget.scientific_panel.service.restart()
        assert widget._get_current_editor().toPlainText() == "connector.get_optic()"
    finally:
        connector._calculation_shutdown_started = True
        for jobs in tuple(connector.calculation_services):
            jobs.shutdown()
        wait_for(
            qapp,
            lambda: all(j._process is None for j in connector.calculation_services),
        )
        widget.shutdown_kernel()
        widget.deleteLater()
