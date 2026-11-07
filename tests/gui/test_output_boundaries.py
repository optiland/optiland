"""File ownership boundaries also apply to public synchronous entry points."""

from __future__ import annotations

import json

import pytest

from optiland.fileio import load_codev_file, load_zemax_file
from optiland_gui.services.file_service import SpecialFloatEncoder

from .test_calculation_jobs import wait_for
from .test_file_save_export import file_operations as file_operations


def test_sync_save_refuses_to_overtake_an_owned_async_write(
    file_operations, monkeypatch, tmp_path
):
    connector, operations, _, _, notifications = file_operations
    monkeypatch.setattr(
        connector._file_service,
        "_toast",
        lambda message, severity, **kwargs: notifications.append((message, severity)),
    )
    path = tmp_path / "saved.json"
    path.write_bytes(b"existing contents")
    dispatch = operations.jobs._dispatch
    monkeypatch.setattr(operations.jobs, "_dispatch", lambda: None)
    operations.request_output(path)
    connector._file_service.save(str(path))
    assert path.read_bytes() == b"existing contents"
    assert connector.is_modified() and connector.get_current_filepath() is None
    assert any("asynchronous save" in message for message, _ in notifications)
    monkeypatch.setattr(operations.jobs, "_dispatch", dispatch)
    operations.cancel_pending()


@pytest.mark.parametrize(
    "file_format,extension,loader",
    [("zemax", "zmx", load_zemax_file), ("codev", "seq", load_codev_file)],
)
def test_sync_export_respects_owned_path_and_works_after_release(
    qapp, file_operations, monkeypatch, tmp_path, file_format, extension, loader
):
    connector, operations, _, _, notifications = file_operations
    path = tmp_path / f"export.{extension}"
    path.write_bytes(b"existing contents")
    token = connector.document_state.edit_token
    dispatch = operations.jobs._dispatch
    monkeypatch.setattr(operations.jobs, "_dispatch", lambda: None)
    operations.request_output(path)
    export = getattr(connector, f"export_{file_format}")
    export(str(path))
    assert path.read_bytes() == b"existing contents"
    assert connector.document_state.edit_token == token
    assert connector.is_modified() and connector.get_current_filepath() is None
    assert any(
        "asynchronous save" in message and severity == "error"
        for message, severity in notifications
    )
    monkeypatch.setattr(operations.jobs, "_dispatch", dispatch)
    operations.cancel_pending()
    wait_for(qapp, lambda: not operations.busy)
    export(str(path))
    restored = loader(str(path))
    assert restored.surfaces.num_surfaces == connector.get_optic().surfaces.num_surfaces
    assert float(restored.surfaces[1].geometry.radius) == 50
    assert connector.document_state.edit_token == token
    assert connector.is_modified() and connector.get_current_filepath() is None


def test_json_encoder_supports_scalar_protocol_without_string_fallback():
    class Scalar:
        def item(self):
            return 2.5

    assert json.loads(json.dumps({"value": Scalar()}, cls=SpecialFloatEncoder)) == {
        "value": 2.5
    }
