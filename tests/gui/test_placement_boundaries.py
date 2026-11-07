"""Placement validation and editor replacement while a rebuild is in flight."""

from __future__ import annotations

from threading import Event

import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland_gui.lens_editor import LensEditor
from optiland_gui.services.job_records import JobResult
from optiland_gui.services.placement_tasks import (
    apply_placement,
    placement_values,
    prepare_placement,
    validate_changes,
)
from optiland_gui.services.prepared_optic import PreparedOptic

from .test_calculation_jobs import wait_for
from .test_editor_batches import make_large_connector
from .test_surface_placement import connector as connector


def test_invalid_frames_components_and_nonfinite_object_are_rejected(minimal_optic):
    with pytest.raises(ValueError, match="Global or Reference"):
        placement_values(minimal_optic.surfaces[1], "bad")
    with pytest.raises(ValueError, match="Unknown placement"):
        validate_changes({"other": 1})
    with pytest.raises(ValueError, match="no longer available"):
        apply_placement(minimal_optic, 99, "global", {"x": 1})
    with pytest.raises(ValueError, match="including Z"):
        apply_placement(minimal_optic, 0, "global", {"x": 1})
    old = minimal_optic.surfaces[1].geometry.cs
    old.reference_cs = CoordinateSystem(x=-1e308)
    with pytest.raises(ValueError, match="finite coordinate limits"):
        apply_placement(minimal_optic, 1, "global", {"x": 1e308})
    assert minimal_optic.surfaces[1].geometry.cs is old


def test_submission_guards_leave_history_and_document_untouched(connector):
    c = connector
    surface = c.get_optic().surfaces[1]
    token = c.document_state.edit_token
    c._calculation_shutdown_started = True
    assert not c.placement.submit(surface, "global", {"x": 1}, token)
    c._calculation_shutdown_started = False
    foreign = PreparedOptic.capture(c.get_optic()).restore().surfaces[1]
    assert not c.placement.submit(foreign, "global", {"x": 1}, token)
    assert "no longer available" in c.placement.message
    assert not c.placement.submit(c.get_optic().surfaces[0], "global", {"x": 1}, token)
    assert "including Z" in c.placement.message
    assert not c.placement.submit(surface, "global", {}, token)
    assert c.placement.message == "Placement is unchanged."
    assert not c._undo_redo_manager.can_undo()


@pytest.mark.parametrize("kind", ["invalid", "structure"])
def test_malformed_placement_result_cannot_create_history(connector, monkeypatch, kind):
    c = connector
    monkeypatch.setattr(c.calculation_jobs, "_dispatch", lambda: None)
    optic = c.get_optic()
    assert c.placement.submit(
        optic.surfaces[1], "global", {"x": 1}, c.document_state.edit_token
    )
    request = c.placement.request
    if kind == "structure":
        other = PreparedOptic.capture(optic).restore()
        other.surfaces.remove(3)
        data = PreparedOptic.capture(other)
    else:
        data = object()
    c.placement._finished(JobResult(request, "succeeded", data, current=True))
    assert "Placement failed" in c.placement.message
    assert c.get_optic() is optic and not c._undo_redo_manager.can_undo()


def test_placement_rebind_while_editor_batch_is_pending(qapp, monkeypatch):
    c = make_large_connector(100)
    editor = LensEditor(c)
    try:
        wait_for(qapp, lambda: not editor._table_loading)
        editor.toggle_properties_widget(5)
        editor.toggle_properties_widget(72)
        panel = editor.tableWidget.cellWidget(
            editor.map_surface_index_to_ui_row(72) + 1, 0
        )
        panel.placement.frame.setCurrentIndex(1)
        monkeypatch.setattr(c.calculation_jobs, "_dispatch", lambda: None)
        c.placement.submit(
            c.get_optic().surfaces[72],
            "reference",
            {"x": 1},
            c.document_state.edit_token,
        )
        request = c.placement.request
        candidate = prepare_placement(
            request.snapshot, request.parameters, lambda *a: None, Event()
        )
        editor.load_data()
        assert editor._table_loading
        c.placement._finished(JobResult(request, "succeeded", candidate, current=True))
        wait_for(qapp, lambda: not editor._table_loading)
        assert editor.open_prop_source_rows == {5, 72}
        panel = editor.tableWidget.cellWidget(
            editor.map_surface_index_to_ui_row(72) + 1, 0
        )
        assert panel.placement.frame.currentData() == "reference"
        assert float(c.get_optic().surfaces[72].geometry.cs.x) == 1
    finally:
        c.calculation_jobs.shutdown()
        wait_for(qapp, lambda: c.calculation_jobs._process is None)
        editor.deleteLater()
        c.deleteLater()


def test_draft_refresh_and_failed_cell_edit_remain_visible(qapp, connector):
    c = connector
    editor = LensEditor(c)
    editor.toggle_properties_widget(1)
    widget = editor.tableWidget.cellWidget(2, 0).placement
    widget.inputs["x"].setText("1")
    widget.refresh_values()
    assert widget.inputs["x"].text() == "1"
    c.set_surface_data(2, c.COL_COMMENT, "a newer edit")
    widget.refresh_values()
    assert "design changed" in widget.status.text()
    c.placement.busy = True
    widget.apply()
    c.placement.busy = False
    widget.frame.setCurrentIndex(-1)
    widget.refresh_values(force=True)
    assert not widget._editable and "Unable to read placement" in widget.status.text()
    row = editor.map_surface_index_to_ui_row(2)
    apply_placement(c.get_optic(), 1, "global", {"z": 3})
    item = editor.tableWidget.item(row, c.COL_THICKNESS)
    item.setText("12")
    assert not editor._edit_error.isHidden()
    assert item.text() != "12"
    assert "Placement" in editor._edit_error.text()
    editor.deleteLater()


def test_positive_thickness_and_global_rotation_only_preserve_their_contracts(
    connector,
):
    import numpy as np

    c = connector
    c.set_surface_data(1, c.COL_THICKNESS, "6")
    assert float(c.get_optic().surfaces[1].thickness) == 6
    surface = c.get_optic().surfaces[2]
    surface.geometry.cs.reference_cs = CoordinateSystem(x=2, y=3, rx=0.2)
    before = surface.geometry.cs.get_effective_transform()[0]
    before = np.array([float(v) for v in before])
    apply_placement(c.get_optic(), 2, "global", {"rx": 20})
    after = surface.geometry.cs.get_effective_transform()[0]
    np.testing.assert_allclose([float(v) for v in after], before)


def test_unchanged_owned_preparation_skips_solves(minimal_optic, monkeypatch):
    def reject(updater):
        pytest.fail("unchanged placement reran solves")

    monkeypatch.setattr(type(minimal_optic.updater), "update", reject)
    prepared = prepare_placement(
        PreparedOptic.capture(minimal_optic),
        {"row": 1, "frame": "global", "changes": {}},
        lambda *a: None,
        Event(),
    )
    assert float(prepared.restore().surfaces[1].geometry.cs.z) == 0
