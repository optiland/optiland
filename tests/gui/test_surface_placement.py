"""Physical placement, owned transactions and editor identity regressions."""

from __future__ import annotations

import math
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from scipy.spatial.transform import Rotation

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.solves import MarginalRayHeightThicknessSolve
from optiland_gui.lens_editor import LensEditor, SurfacePropertiesWidget
from optiland_gui.optiland_connector import OptilandConnector
from optiland_gui.services.job_records import JobResult
from optiland_gui.services.placement_tasks import (
    apply_placement,
    placement_values,
    prepare_placement,
)
from optiland_gui.services.prepared_optic import PreparedOptic
from tests.gui.test_calculation_jobs import wait_for
from tests.gui.test_editor_batches import make_large_connector
from tests.test_folded_paraxial import folded


@pytest.fixture(params=["numpy", "torch"])
def backend(request):
    if request.param == "torch":
        pytest.importorskip("torch")
    be.set_backend(request.param)
    if request.param == "torch":
        be.set_device("cpu")
        be.set_precision("float64")
    yield
    be.set_backend("numpy")


def transform(surface):
    position, rotation = surface.geometry.cs.get_effective_transform()
    return be.to_numpy(position).copy(), be.to_numpy(rotation).copy()


def test_reference_global_rotation_and_single_surface_scope(backend, minimal_optic):
    optic = minimal_optic
    parent = CoordinateSystem(x=3, y=-7, z=14, rx=0.4, ry=-0.2, rz=0.3)
    original = CoordinateSystem(
        x=1, y=2, z=3, rx=0.1, ry=0.2, rz=-0.3, reference_cs=parent
    )
    optic.surfaces[1].geometry.cs = original
    # Another surface references the edited surface's frame.
    optic.surfaces[2].geometry.cs = CoordinateSystem(z=8, reference_cs=original)
    untouched = transform(optic.surfaces[2])
    assert apply_placement(
        optic, 1, "global", {"x": 12, "y": -6, "z": 25, "rx": 25, "ry": -30, "rz": 40}
    )
    position, rotation = transform(optic.surfaces[1])
    np.testing.assert_allclose(position, [12, -6, 25], atol=1e-12)
    np.testing.assert_allclose(
        rotation,
        Rotation.from_euler("xyz", [25, -30, 40], degrees=True).as_matrix(),
        atol=1e-12,
    )
    assert optic.surfaces[1].geometry.cs.reference_cs is parent
    assert optic.surfaces[2].geometry.cs.reference_cs is original
    np.testing.assert_allclose(transform(optic.surfaces[2])[0], untouched[0])
    np.testing.assert_allclose(transform(optic.surfaces[2])[1], untouched[1])


def test_decenter_keeps_rotation_exact_and_syncs_later_thickness(
    backend, minimal_optic
):
    optic = minimal_optic
    original = optic.surfaces[2].geometry.cs
    original.rx = 0.123456789123456
    assert apply_placement(optic, 2, "reference", {"x": 0.25, "y": -0.5})
    cs = optic.surfaces[2].geometry.cs
    assert float(cs.rx) == float(original.rx)
    assert float(cs.z) == float(original.z)
    assert float(cs.x) == 0.25 and float(cs.y) == -0.5
    cs.rx = 0
    assert apply_placement(optic, 2, "global", {"z": 7})
    # Updating another spacing must start from the newly stored thicknesses.
    optic.updater.set_thickness(9, 2)
    assert float(optic.surfaces[2].geometry.cs.z) == 7
    assert float(optic.surfaces[3].geometry.cs.z) == 16


@pytest.mark.parametrize("frame", ["reference", "global"])
def test_untouched_and_position_only_gimbal_lock(backend, minimal_optic, frame):
    surface = minimal_optic.surfaces[2]
    parent = CoordinateSystem(rx=0.2, ry=math.pi / 2, rz=0.3)
    surface.geometry.cs.reference_cs = parent
    old = surface.geometry.cs
    before = transform(surface)
    assert not apply_placement(
        minimal_optic, 2, frame, placement_values(surface, frame)
    )
    assert surface.geometry.cs is old
    values = placement_values(surface, frame)
    assert apply_placement(minimal_optic, 2, frame, {"x": values["x"] + 0.5})
    np.testing.assert_allclose(transform(surface)[1], before[1], atol=1e-12)
    assert float(surface.geometry.cs.rx) == float(old.rx)
    assert float(surface.geometry.cs.ry) == float(old.ry)
    assert float(surface.geometry.cs.rz) == float(old.rz)


@pytest.mark.parametrize("value", ["", "bad", "nan", "inf", "-inf"])
def test_invalid_input_leaves_every_coordinate_unchanged(minimal_optic, value):
    cs = minimal_optic.surfaces[1].geometry.cs
    before = cs.to_dict()
    with pytest.raises(ValueError, match="finite"):
        apply_placement(minimal_optic, 1, "global", {"x": 2, "ry": value})
    assert cs.to_dict() == before
    assert minimal_optic.surfaces[1].geometry.cs is cs


def test_infinite_object_can_be_placed_finitely_and_image_can_move(minimal_optic):
    assert apply_placement(minimal_optic, 0, "global", {"z": -20})
    assert float(minimal_optic.surfaces[0].geometry.cs.z) == -20
    assert float(minimal_optic.surfaces[0].thickness) == 20
    assert apply_placement(minimal_optic, 3, "reference", {"y": 1, "rx": 10})
    assert float(minimal_optic.surfaces[3].geometry.cs.rx) == math.radians(10)


@pytest.fixture
def connector(qapp, minimal_optic):
    c = OptilandConnector()
    c.load_optic_from_object(minimal_optic)
    c._undo_redo_manager.clear_stacks()
    yield c
    c.calculation_jobs.shutdown()
    wait_for(qapp, lambda: c.calculation_jobs._process is None)
    c.deleteLater()


def test_real_worker_commit_editor_rebind_and_history(qapp, connector):
    c = connector
    editor = LensEditor(c)
    editor.resize(950, 650)
    editor.show()
    editor.tableWidget.selectRow(2)
    editor.toggle_properties_widget(2)
    qapp.processEvents()
    editor.interaction_state.set_hover(1, 3)
    widget = editor.tableWidget.cellWidget(3, 0).placement
    before = c.get_optic()
    widget.inputs["x"].setText("0.5")
    widget.inputs["rx"].setText("12.5")
    QTest.mouseClick(widget.apply_button, Qt.LeftButton)
    assert c.placement.busy and c.get_optic() is before, (
        c.placement.message,
        widget.status.text(),
        widget.apply_button.isEnabled(),
        widget.apply_button.isVisible(),
        widget.apply_button.geometry(),
    )
    wait_for(qapp, lambda: not c.placement.busy)
    assert "applied" in c.placement.message, c.placement.message
    assert len(c._undo_redo_manager._undo_stack) == 1
    assert c.get_optic() is not before
    assert float(c.get_optic().surfaces[2].geometry.cs.x) == 0.5
    assert float(c.get_optic().surfaces[2].geometry.cs.rx) == math.radians(12.5)
    assert editor.open_prop_source_rows == {2}
    assert isinstance(editor.tableWidget.cellWidget(3, 0), SurfacePropertiesWidget)
    assert editor.interaction_state.selected_surfaces == (c.get_optic().surfaces[2],)
    # Verify actual history reconstruction round-trips the placement.
    assert c.request_undo()
    wait_for(qapp, lambda: not c.history.busy)
    assert float(c.get_optic().surfaces[2].geometry.cs.x) == 0
    assert c.request_redo()
    wait_for(qapp, lambda: not c.history.busy)
    assert float(c.get_optic().surfaces[2].geometry.cs.x) == 0.5
    editor.close()
    editor.deleteLater()


def test_expansion_and_drafts_for_standard_mirror_object_image(qapp, connector):
    c = connector
    c.get_optic().surfaces[1].interaction_model.is_reflective = True
    editor = LensEditor(c)
    for row in range(4):
        ui_row = editor.map_surface_index_to_ui_row(row)
        assert not editor.tableWidget.cellWidget(ui_row, 0).props_button.isHidden()
        editor.toggle_properties_widget(row)
        panel = editor.tableWidget.cellWidget(
            editor.map_surface_index_to_ui_row(row) + 1, 0
        )
        assert panel.placement.inputs["z"].isEnabled()
    widget = editor.tableWidget.cellWidget(
        editor.map_surface_index_to_ui_row(3) + 1, 0
    ).placement
    original = widget.inputs["x"].text()
    assert not widget.apply_button.isEnabled()
    widget.frame.setCurrentIndex(1)
    assert not c._undo_redo_manager.can_undo()
    widget.inputs["x"].setText("bad")
    assert not widget.frame.isEnabled()
    widget.apply()
    assert "finite number" in widget.status.text()
    assert not c.placement.busy
    widget.reload_button.click()
    assert widget.inputs["x"].text() == original
    widget.inputs["x"].setText("2")
    c.set_surface_data(1, c.COL_COMMENT, "Newer edit")
    widget.apply()
    assert "design changed" in widget.status.text()
    assert widget.inputs["x"].text() == "2"
    assert not c.placement.busy
    editor.deleteLater()


@pytest.mark.parametrize("intervention", ["metadata", "replacement", "cancel"])
def test_real_worker_rejects_intervening_change_or_cancel(
    qapp, connector, intervention
):
    c = connector
    c.placement.submit(
        c.get_optic().surfaces[2], "global", {"x": 5}, c.document_state.edit_token
    )
    assert c.placement.busy
    if intervention == "metadata":
        c.set_surface_data(1, c.COL_COMMENT, "Keep this")
    elif intervention == "replacement":
        c.notify_change("replacement")
    else:
        c.placement.cancel()
    current = c.get_optic()
    history_size = len(c._undo_redo_manager._undo_stack)
    wait_for(qapp, lambda: not c.placement.busy)
    assert c.get_optic() is current
    assert float(current.surfaces[2].geometry.cs.x) == 0
    assert len(c._undo_redo_manager._undo_stack) == history_size


def test_backend_changed_before_acceptance_preserves_document(
    qapp, connector, monkeypatch
):
    c = connector
    # Hold dispatch to exercise the GUI acceptance gate with a completed candidate.
    monkeypatch.setattr(c.calculation_jobs, "_dispatch", lambda: None)
    c.placement.submit(
        c.get_optic().surfaces[2], "global", {"x": 5}, c.document_state.edit_token
    )
    request = c.placement.request
    candidate = prepare_placement(
        request.snapshot, request.parameters, lambda *a: None, Event()
    )
    invalid = replace(candidate, backend=replace(candidate.backend, name="torch"))
    current = c.get_optic()
    c.placement._finished(JobResult(request, "succeeded", invalid, current=True))
    assert c.get_optic() is current
    assert not c._undo_redo_manager.can_undo()
    assert "backend changed" in c.placement.message


def test_shifted_first_surface_blocks_destructive_thickness_edit(connector):
    c = connector
    apply_placement(c.get_optic(), 1, "global", {"z": 3})
    original = c.get_optic()
    with pytest.raises(ValueError, match="Placement"):
        c.set_surface_data(2, c.COL_THICKNESS, "12")
    assert c.get_optic() is original
    assert float(c.get_optic().surfaces[1].geometry.cs.z) == 3
    assert not c._undo_redo_manager.can_undo()


def test_display_precision_does_not_modify_untouched_components(qapp, connector):
    c = connector
    surface = c.get_optic().surfaces[2]
    surface.geometry.cs.z = 18.774066999999999
    surface.geometry.cs.rx = math.radians(-44.95000000000001)
    panel = SurfacePropertiesWidget(2, c)
    widget = panel.placement
    assert widget.inputs["z"].text() == "18.774067"
    assert widget.inputs["rx"].text() == "-44.95"
    widget.inputs["x"].setText(".25")
    assert set(widget.changes()) == {"x"}
    before_z, before_rx = float(surface.geometry.cs.z), float(surface.geometry.cs.rx)
    widget.apply()
    wait_for(qapp, lambda: not c.placement.busy)
    assert float(c.get_optic().surfaces[2].geometry.cs.z) == before_z
    assert float(c.get_optic().surfaces[2].geometry.cs.rx) == before_rx
    panel.deleteLater()


def test_nonfinite_solve_output_fails_without_mutating_source(
    minimal_optic, monkeypatch
):
    snapshot = PreparedOptic.capture(minimal_optic)

    def invalid_update(updater):
        updater.optic.surfaces[2].geometry.cs.x = float("nan")

    monkeypatch.setattr(type(minimal_optic.updater), "update", invalid_update)
    with pytest.raises(ValueError, match="nonfinite X"):
        prepare_placement(
            snapshot,
            {"row": 1, "frame": "global", "changes": {"x": 1}},
            lambda *args: None,
            Event(),
        )
    assert float(minimal_optic.surfaces[1].geometry.cs.x) == 0
    assert float(minimal_optic.surfaces[2].geometry.cs.x) == 0


def test_folded_mirror_edit_keeps_other_frames_and_spacing_storage(backend):
    optic = folded()
    row = next(
        i
        for i, surface in enumerate(optic.surfaces)
        if getattr(surface.interaction_model, "is_reflective", False)
    )
    before = [transform(s) for s in optic.surfaces]
    thicknesses = [float(s.thickness) for s in optic.surfaces]
    values = placement_values(optic.surfaces[row], "global")
    apply_placement(optic, row, "global", {"rx": values["rx"] + 0.5})
    assert [float(s.thickness) for s in optic.surfaces] == thicknesses
    assert not np.allclose(transform(optic.surfaces[row])[1], before[row][1])
    for index, surface in enumerate(optic.surfaces):
        if index != row:
            for actual, expected in zip(transform(surface), before[index], strict=True):
                np.testing.assert_allclose(actual, expected)


def test_real_worker_failed_solve_does_not_change_live_optic(qapp, connector):
    c = connector
    optic = c.get_optic()
    # A stale solve referring to a removed surface must fail only on the candidate.
    optic.solves.solves.append(MarginalRayHeightThicknessSolve(optic, 99, 0))
    c.placement.submit(
        optic.surfaces[1], "global", {"rx": 45}, c.document_state.edit_token
    )
    wait_for(qapp, lambda: not c.placement.busy)
    assert "failed" in c.placement.message.lower(), c.placement.message
    assert c.get_optic() is optic
    assert float(optic.surfaces[1].geometry.cs.rx) == 0
    assert not c._undo_redo_manager.can_undo()


def test_placement_rebind_survives_batched_editor_and_keeps_reference_mode(qapp):
    c = make_large_connector(100)
    editor = LensEditor(c)
    try:
        wait_for(qapp, lambda: not editor._table_loading)
        editor.tableWidget.selectRow(72)
        editor.toggle_properties_widget(5)
        editor.toggle_properties_widget(72)
        wait_for(qapp, lambda: not editor._table_loading)
        widget = editor.tableWidget.cellWidget(
            editor.map_surface_index_to_ui_row(72) + 1, 0
        ).placement
        widget.frame.setCurrentIndex(1)
        widget.inputs["y"].setText("0.25")
        widget.apply()
        wait_for(qapp, lambda: not c.placement.busy and not editor._table_loading)
        assert "applied" in c.placement.message, c.placement.message
        assert editor.open_prop_source_rows == {5, 72}
        assert editor.interaction_state.selected_surfaces == (
            c.get_optic().surfaces[72],
        )
        assert (
            editor.tableWidget.cellWidget(
                editor.map_surface_index_to_ui_row(72) + 1, 0
            ).placement.frame.currentData()
            == "reference"
        )
        assert float(c.get_optic().surfaces[72].geometry.cs.y) == 0.25
        other = editor.tableWidget.cellWidget(
            editor.map_surface_index_to_ui_row(5) + 1, 0
        )
        assert other.placement.frame.currentData() == "global"
    finally:
        c.calculation_jobs.shutdown()
        wait_for(qapp, lambda: c.calculation_jobs._process is None)
        editor.deleteLater()
        c.deleteLater()
