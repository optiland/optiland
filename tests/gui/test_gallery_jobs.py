"""Gallery constructors execute with owned models outside the Qt event loop."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from optiland.samples.objectives import CookeTriplet
from optiland_gui.main_window import MainWindow
from optiland_gui.optiland_connector import OptilandConnector

from .test_calculation_jobs import wait_for


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
        connector.file_operations.begin_close()
        connector.calculation_jobs.shutdown()
        wait_for(qapp, lambda: connector.calculation_jobs._process is None)


def test_gallery_discard_cancellation_does_not_queue(qapp):
    connector = OptilandConnector()
    host = SimpleNamespace(connector=connector, _confirm_discard_changes=lambda: False)
    MainWindow._load_sample_action(host, CookeTriplet)
    assert not connector.file_operations.busy
    assert not connector.calculation_jobs.running
