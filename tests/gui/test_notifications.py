"""Tests verifying that QMessageBox popups have been replaced with toasts."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock


class TestFileServiceNoQMessageBox:
    """Verify file_service.py no longer calls QMessageBox directly."""

    def test_no_qmessagebox_critical_in_source(self):
        from optiland_gui.services import file_service

        src = inspect.getsource(file_service)
        assert "QMessageBox.critical" not in src
        assert "QMessageBox.warning" not in src

    def test_load_failure_calls_toast(self, tmp_path, highlighting_connector):
        conn = highlighting_connector
        conn.toast_manager = MagicMock()
        original = conn.get_optic()
        token = conn.document_state.edit_token
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{ not valid json", encoding="utf-8")

        conn._file_service.load(str(bad_file))

        conn.toast_manager.notify.assert_called_once()
        message, severity = conn.toast_manager.notify.call_args.args[:2]
        assert "Load failed" in message and severity == "error"
        assert conn.get_optic() is original
        assert conn.document_state.edit_token == token
        assert conn.is_modified() and conn.get_current_filepath() is None

    def test_save_failure_calls_toast(
        self, tmp_path, highlighting_connector, monkeypatch
    ):
        conn = highlighting_connector
        conn.toast_manager = MagicMock()
        original = conn.get_optic()
        token = conn.document_state.edit_token
        capture = MagicMock(side_effect=RuntimeError("capture failed"))
        monkeypatch.setattr(conn, "_capture_optic_state", capture)
        destination = tmp_path / "out.json"
        destination.write_bytes(b"previous valid contents")

        conn._file_service.save(str(destination))

        capture.assert_called_once_with()
        conn.toast_manager.notify.assert_called_once()
        message, severity = conn.toast_manager.notify.call_args.args[:2]
        assert "capture failed" in message and severity == "error"
        assert destination.read_bytes() == b"previous valid contents"
        assert conn.get_optic() is original
        assert conn.document_state.edit_token == token
        assert conn.is_modified() and conn.get_current_filepath() is None

    def test_load_from_object_failure_calls_toast(self, highlighting_connector):
        conn = highlighting_connector
        conn.toast_manager = MagicMock()
        original = conn.get_optic()
        token = conn.document_state.edit_token
        bad_obj = MagicMock()
        bad_obj.to_dict.side_effect = RuntimeError("serialisation error")

        conn._file_service.load_from_object(bad_obj)

        bad_obj.to_dict.assert_called_once_with()
        conn.toast_manager.notify.assert_called_once()
        message, severity = conn.toast_manager.notify.call_args.args[:2]
        assert "serialisation error" in message and severity == "error"
        assert conn.get_optic() is original
        assert conn.document_state.edit_token == token
        assert conn.is_modified() and conn.get_current_filepath() is None


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
