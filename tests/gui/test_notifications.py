"""Tests verifying that QMessageBox popups have been replaced with toasts."""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class TestFileServiceNoQMessageBox:
    """Verify file_service.py no longer calls QMessageBox directly."""

    def test_no_qmessagebox_critical_in_source(self):
        from optiland_gui.services import file_service

        src = inspect.getsource(file_service)
        assert "QMessageBox.critical" not in src
        assert "QMessageBox.warning" not in src

    def test_load_failure_calls_toast(self, tmp_path, qapp):
        conn = MagicMock()
        conn.toast_manager = MagicMock()
        conn._undo_redo_manager = MagicMock()
        conn._optic = None

        from optiland_gui.services.file_service import FileService

        svc = FileService(conn)
        bad_file = str(tmp_path / "bad.json")
        Path(bad_file).write_text("{ not valid json", encoding="utf-8")

        svc.load(bad_file)

        conn.toast_manager.notify.assert_called()
        severity = conn.toast_manager.notify.call_args[0][1]
        assert severity == "error"

    def test_save_failure_calls_toast(self, tmp_path, qapp):
        conn = MagicMock()
        conn.toast_manager = MagicMock()
        conn._undo_redo_manager = MagicMock()
        conn._capture_optic_state.side_effect = RuntimeError("capture failed")

        from optiland_gui.services.file_service import FileService

        svc = FileService(conn)
        svc.save(str(tmp_path / "out.json"))

        conn.toast_manager.notify.assert_called()
        severity = conn.toast_manager.notify.call_args[0][1]
        assert severity == "error"

    def test_load_from_object_failure_calls_toast(self, qapp):
        conn = MagicMock()
        conn.toast_manager = MagicMock()
        conn._undo_redo_manager = MagicMock()

        bad_obj = MagicMock()
        bad_obj.to_dict.side_effect = RuntimeError("serialisation error")

        from optiland_gui.services.file_service import FileService

        svc = FileService(conn)
        svc.load_from_object(bad_obj)

        conn.toast_manager.notify.assert_called()
        severity = conn.toast_manager.notify.call_args[0][1]
        assert severity == "error"



class TestSidebarNoQMessageBox:
    def test_no_qmessagebox_in_sidebar_source(self):
        from optiland_gui.widgets import sidebar

        src = inspect.getsource(sidebar)
        assert "QMessageBox" not in src

    def test_wip_button_emits_signal_not_dialog(self, qapp):
        from optiland_gui.widgets.sidebar import SidebarWidget

        widget = SidebarWidget()
        messages = []
        widget.showWipMessage.connect(messages.append)

        # Find a WIP button and click it
        wip_name = widget._wip_buttons[0]
        btn_item = next(
            (b for b in widget._buttons_list if b["name"] == wip_name), None
        )
        assert btn_item is not None, "WIP button not found"

        btn_item["widget"].click()

        assert len(messages) == 1
        assert "development" in messages[0].lower()
