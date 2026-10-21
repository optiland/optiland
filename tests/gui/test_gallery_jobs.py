"""Gallery constructors execute with owned models outside the Qt event loop."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from optiland.samples.objectives import CookeTriplet
from optiland_gui.main_window import MainWindow
from optiland_gui.optiland_connector import OptilandConnector

from .test_calculation_jobs import wait_for
from .test_file_open_import import edit_comment
from .test_file_open_import import file_operations as file_operations


def test_gallery_constructor_runs_only_in_owned_worker(qapp, monkeypatch):
    connector = OptilandConnector()
    original = connector.get_optic()
    changes = []
    connector.document_state.committed.connect(changes.append)
    host = SimpleNamespace(connector=connector, _confirm_discard_changes=lambda: True)
    monkeypatch.setattr(
        CookeTriplet,
        "__init__",
        lambda self: pytest.fail("Sample constructor ran in the GUI process"),
    )
    MainWindow._load_sample_action(host, CookeTriplet)
    assert connector.get_optic() is original
    try:
        wait_for(qapp, lambda: not connector.file_operations.busy)
        assert connector.get_optic() is not original
        assert connector.get_optic().surfaces.num_surfaces == 8
        assert connector.is_modified()
        assert connector.get_current_filepath() is None
        assert len(changes) == 1
    finally:
        connector.file_operations.cancel_pending()
        connector.calculation_jobs.shutdown()
        wait_for(qapp, lambda: connector.calculation_jobs._process is None)


def test_gallery_discard_cancellation_does_not_queue(qapp):
    connector = OptilandConnector()
    host = SimpleNamespace(connector=connector, _confirm_discard_changes=lambda: False)
    MainWindow._load_sample_action(host, CookeTriplet)
    assert not connector.file_operations.busy
    assert not connector.calculation_jobs.running


@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("builtins", "object"),
        ("optiland.samples.objectives", "__name__"),
    ],
)
def test_invalid_gallery_class_cannot_replace_document(
    qapp, file_operations, module_name, class_name
):
    connector, operations, _, results, _ = file_operations
    original = connector.get_optic()
    operations.request_load(
        "invalid", "sample", load_options={"sample_class": (module_name, class_name)}
    )
    wait_for(qapp, lambda: not operations.busy)
    assert results[-1].status == "failed"
    assert connector.get_optic() is original


def test_gallery_candidate_does_not_discard_edits_made_during_construction(
    qapp, file_operations, monkeypatch
):
    connector, operations, _, _, _ = file_operations
    dispatch = operations.jobs._dispatch
    monkeypatch.setattr(operations.jobs, "_dispatch", lambda: None)
    request = operations.request_load(
        "CookeTriplet",
        "sample",
        load_options={"sample_class": (CookeTriplet.__module__, CookeTriplet.__name__)},
    )
    original = connector.get_optic()
    edit_comment(connector, "newer edits")
    token = connector.document_state.edit_token
    monkeypatch.setattr(operations.jobs, "_dispatch", dispatch)
    operations.jobs._dispatch()
    wait_for(qapp, lambda: not operations.busy)
    assert connector.get_optic() is original
    assert (
        operations.resolve_candidate(request.context.identifier, False, token) is False
    )
    assert connector.get_optic().surfaces[1].comment == "newer edits"
