"""Large LDE installation yields, preserves identities and abandons obsolete rows."""

from __future__ import annotations

from unittest.mock import PropertyMock

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QTimer

import optiland.backend as be
from optiland.optic import Optic
from optiland.solves import MarginalRayHeightThicknessSolve
from optiland_gui.lens_editor import LensEditor, SurfacePropertiesWidget
from optiland_gui.optiland_connector import OptilandConnector
from tests.gui.test_calculation_jobs import wait_for
from tests.test_folded_paraxial import folded, straight


def make_large_connector(count=100):
    optic = Optic()
    optic.set_aperture("EPD", 2)
    optic.fields.set_type("angle")
    optic.fields.add(y=0)
    optic.wavelengths.add(0.55, is_primary=True)
    for index in range(count):
        optic.surfaces.add(
            index=index,
            radius=float("inf"),
            thickness=5,
            material="air",
            is_stop=index == 1,
            comment=f"Surface {index}",
        )
    connector = OptilandConnector()
    connector._optic = optic
    connector.notify_change("replacement")
    return connector


def test_display_capture_computes_positions_once_and_matches_public_cells(monkeypatch):
    c = make_large_connector()
    group = type(c.get_optic().surfaces)
    getter = group.positions.fget
    calls = PropertyMock(side_effect=lambda: getter(c.get_optic().surfaces))
    monkeypatch.setattr(group, "positions", calls)
    rows = c.get_surface_display_rows()
    assert calls.call_count == 1
    for row in (0, 1, 50, 99):
        for column in range(7):
            assert rows[row][column] == c.get_surface_data(row, column)
    c.deleteLater()


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_folded_spacing_and_solve_changes_match_public_cells(qapp, backend):
    if backend == "torch":
        pytest.importorskip("torch")
    be.set_backend(backend)
    if backend == "torch":
        be.set_device("cpu")
        be.set_precision("float64")
    c = OptilandConnector()
    try:
        c._optic = folded()
        c.notify_change("replacement")
        rows = c.get_surface_display_rows()
        assert rows[3][c.COL_THICKNESS] == "-26.0000"
        for row, cells in rows.items():
            for column in range(7):
                assert cells[column] == c.get_surface_data(row, column)
        c._optic = straight()
        c.notify_change("replacement")
        c._optic.solves.solves.append(MarginalRayHeightThicknessSolve(c._optic, 3, 0))
        c._optic.updater.update()
        before = c.get_surface_display_rows()[2][c.COL_THICKNESS]
        c.set_surface_data(1, c.COL_RADIUS, "35.0")
        after = c.get_surface_display_rows()[2][c.COL_THICKNESS]
        assert before != after
        assert after == c.get_surface_data(2, c.COL_THICKNESS)
        c.undo()
        assert c.get_surface_display_rows()[2][c.COL_THICKNESS] == before
    finally:
        c.deleteLater()
        be.set_backend("numpy")


def test_large_table_yields_and_supersedes_partial_refresh(qapp, monkeypatch):
    c = make_large_connector()
    monkeypatch.setattr(LensEditor, "_BATCH_SECONDS", 0)
    editor = LensEditor(c)
    completed = []
    editor.loadingFinished.connect(lambda: completed.append(True))
    assert editor._table_loading and not editor.tableWidget.isEnabled()
    turns = []
    QTimer.singleShot(0, lambda: turns.append(editor._next_load_row))
    wait_for(qapp, lambda: bool(turns))
    assert turns[0] < 100
    c.get_optic().surfaces[20].comment = "Changed while loading"
    c.notify_change("metadata", surface_indices=(20,), columns=(c.COL_COMMENT,))
    wait_for(qapp, lambda: not editor._table_loading)
    assert completed == [True]
    assert editor.tableWidget.item(20, c.COL_COMMENT).text() == "Changed while loading"
    assert editor.tableWidget.isEnabled()
    assert not editor.hover_tracker._suspended
    assert editor.tableWidget.cellWidget(1, 0).surface_menu.actions() == []
    editor.tableWidget.cellWidget(1, 0)._populate_surface_menu()
    assert len(editor.tableWidget.cellWidget(1, 0).surface_menu.actions()) > 0
    editor.deleteLater()
    c.deleteLater()


def test_batched_structural_refresh_restores_surviving_surface_selection(qapp):
    c = make_large_connector()
    editor = LensEditor(c)
    wait_for(qapp, lambda: not editor._table_loading)
    selected = c.get_optic().surfaces[50]
    editor.tableWidget.setCurrentCell(50, c.COL_RADIUS)
    c.add_surface(index=20)
    assert editor._table_loading
    wait_for(qapp, lambda: not editor._table_loading)
    assert editor.interaction_state.selected_surfaces == (selected,)
    assert editor.tableWidget.currentRow() == 51
    assert editor.tableWidget.currentColumn() == c.COL_RADIUS
    editor.deleteLater()
    c.deleteLater()


def test_batched_refresh_preserves_multiple_panels_and_closes_shifted_owner(
    qapp, monkeypatch
):
    from PySide6.QtWidgets import QWidget

    original = SurfacePropertiesWidget._populate_properties_form

    def two_tabs(panel):
        original(panel)
        panel.tabs.addTab(QWidget(), "Extra properties")

    monkeypatch.setattr(SurfacePropertiesWidget, "_populate_properties_form", two_tabs)
    monkeypatch.setattr(LensEditor, "_BATCH_SECONDS", 0)
    c = make_large_connector()
    editor = LensEditor(c)
    wait_for(qapp, lambda: not editor._table_loading)
    for owner in (20, 50):
        editor.toggle_properties_widget(owner)
    editor.tableWidget.cellWidget(
        editor.map_surface_index_to_ui_row(50) + 1, 0
    ).tabs.setCurrentIndex(1)
    selected = c.get_optic().surfaces[50]
    editor.tableWidget.setCurrentCell(
        editor.map_surface_index_to_ui_row(50), c.COL_RADIUS
    )
    c.add_surface(index=10)
    wait_for(qapp, lambda: not editor._table_loading)
    assert editor.open_prop_source_rows == {21, 51}
    assert editor.interaction_state.selected_surfaces == (selected,)
    assert editor.tableWidget.currentRow() == editor.map_surface_index_to_ui_row(51)
    first = editor.tableWidget.cellWidget(editor.map_surface_index_to_ui_row(21) + 1, 0)
    second = editor.tableWidget.cellWidget(
        editor.map_surface_index_to_ui_row(51) + 1, 0
    )
    assert first.tabs.currentIndex() == 0
    assert second.tabs.currentIndex() == 1
    second.close_button.click()
    assert editor.open_prop_source_rows == {21}
    assert (
        editor.tableWidget.cellWidget(editor.map_surface_index_to_ui_row(21) + 1, 0)
        is first
    )
    editor.deleteLater()
    c.deleteLater()


def test_failed_batch_is_recoverable_and_destruction_cancels_pending_timers(
    qapp, monkeypatch
):
    c = make_large_connector()
    editor = LensEditor(c)
    wait_for(qapp, lambda: not editor._table_loading)
    original = editor._process_table_row
    errors = []
    editor.loadingFailed.connect(errors.append)
    with monkeypatch.context() as scope:

        def fail(*args):
            raise ValueError("Invalid display data")

        scope.setattr(editor, "_process_table_row", fail)
        editor.load_data()
        wait_for(qapp, lambda: not editor._table_loading)
    assert errors == ["Invalid display data"]
    assert not editor.tableWidget.signalsBlocked()
    assert "Invalid display data" in editor._load_progress.format()
    assert editor._process_table_row == original
    editor.load_data()
    wait_for(qapp, lambda: not editor._table_loading)
    editor._flash_cell(2, c.COL_COMMENT, True, duration_ms=1)
    editor.load_data()
    assert editor._load_timer.isActive()
    editor.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    qapp.processEvents()
    c.deleteLater()
