"""Expansion keeps upper rows anchored and scrolls only for clipped panels."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import QSize
from PySide6.QtTest import QTest

from optiland.optic import Optic
from optiland_gui.lens_editor import LensEditor, SurfacePropertiesWidget
from optiland_gui.optiland_connector import OptilandConnector


def make_large_connector(count):
    optic = Optic()
    for index in range(count):
        optic.surfaces.add(
            index=index, radius=float("inf"), thickness=1, is_stop=index == 1
        )
    optic.set_aperture(aperture_type="EPD", value=1)
    optic.fields.set_type("angle")
    optic.fields.add(y=0)
    optic.wavelengths.add(value=0.55, is_primary=True)
    connector = OptilandConnector()
    connector.load_optic_from_object(optic)
    connector.set_surface_type(1, "biconic")
    return connector


@pytest.fixture
def editor(qapp):
    connector = make_large_connector(35)
    widget = LensEditor(connector)
    widget.resize(900, 950)
    widget.show()
    QTest.qWait(10)
    yield widget
    widget.close()
    widget.deleteLater()
    connector.calculation_jobs.shutdown()
    connector.deleteLater()
    qapp.processEvents()


def bounds(editor, owner):
    table = editor.tableWidget
    row = editor.map_surface_index_to_ui_row(owner) + 1
    top = table.rowViewportPosition(row)
    return top, top + table.rowHeight(row)


@pytest.mark.parametrize("scrolled", [False, True])
def test_panel_that_fits_keeps_upper_rows_and_both_scroll_offsets(
    editor, qapp, monkeypatch, scrolled
):
    table = editor.tableWidget
    for column in range(table.columnCount()):
        table.setColumnWidth(column, 180)
    qapp.processEvents()
    table.horizontalScrollBar().setValue(40)
    assert table.horizontalScrollBar().value() == 40
    if scrolled:
        table.verticalScrollBar().setValue(150)
    owner = table.rowAt(150)
    before = table.rowViewportPosition(owner)
    vertical = table.verticalScrollBar().value()
    horizontal = table.horizontalScrollBar().value()
    token = editor.connector.document_state.edit_token
    serial = editor.connector.calculation_jobs._serial
    monkeypatch.setattr(editor, "load_data", MagicMock(side_effect=AssertionError))
    editor.toggle_properties_widget(owner)
    qapp.processEvents()
    assert bounds(editor, owner)[1] <= table.viewport().height()
    assert table.rowViewportPosition(owner) == before
    assert table.verticalScrollBar().value() == vertical
    assert table.horizontalScrollBar().value() == horizontal
    assert editor.connector.document_state.edit_token == token
    assert editor.connector.calculation_jobs._serial == serial


@pytest.mark.parametrize("distance_from_bottom", [0, 40])
def test_overflow_scrolls_minimum_and_aligns_panel_bottom(
    editor, qapp, distance_from_bottom
):
    table = editor.tableWidget
    table.verticalScrollBar().setValue(75)
    owner = table.rowAt(table.viewport().height() - 1 - distance_from_bottom)
    assert owner >= 0 and owner < editor.connector.get_surface_count() - 1
    before_y = table.rowViewportPosition(owner)
    before_scroll = table.verticalScrollBar().value()
    header_height = table.rowHeight(owner)
    editor.toggle_properties_widget(owner)
    qapp.processEvents()
    top, bottom = bounds(editor, owner)
    height = bottom - top
    overflow = before_y + header_height + height - table.viewport().height()
    assert overflow > 0
    assert top >= 0
    assert bottom == table.viewport().height()
    assert table.verticalScrollBar().value() == before_scroll + overflow
    assert table.rowViewportPosition(owner) == before_y - overflow
    # There are still ordinary surfaces below the visible panel.
    following = editor.map_surface_index_to_ui_row(owner + 1)
    assert table.rowViewportPosition(following) == table.viewport().height()


def test_actual_last_surface_expands_at_bottom_without_blank_space(editor, qapp):
    table = editor.tableWidget
    table.verticalScrollBar().setValue(table.verticalScrollBar().maximum())
    owner = editor.connector.get_surface_count() - 1
    editor.toggle_properties_widget(owner)
    qapp.processEvents()
    top, bottom = bounds(editor, owner)
    assert top >= 0
    assert bottom == table.viewport().height()
    assert table.verticalScrollBar().value() == table.verticalScrollBar().maximum()


def test_expansion_that_introduces_vertical_scrollbar_uses_final_viewport(
    qapp, minimal_optic
):
    connector = OptilandConnector()
    connector.load_optic_from_object(minimal_optic)
    editor = LensEditor(connector)
    editor.resize(900, 220)
    editor.show()
    try:
        qapp.processEvents()
        table = editor.tableWidget
        assert table.verticalScrollBar().maximum() == 0
        editor.toggle_properties_widget(3)
        qapp.processEvents()
        top, bottom = bounds(editor, 3)
        assert table.verticalScrollBar().maximum() > 0
        assert top >= 0
        assert bottom == table.viewport().height()
    finally:
        editor.close()
        editor.deleteLater()
        connector.calculation_jobs.shutdown()
        connector.deleteLater()
        qapp.processEvents()


def test_other_panel_and_draft_stay_in_place_when_new_panel_fits(editor, qapp):
    table = editor.tableWidget
    editor.toggle_properties_widget(1)
    first = table.cellWidget(2, 0)
    first.input_widgets["Radius X"].setText("1.2345")
    first_position = bounds(editor, 1)
    source_row = editor.map_surface_index_to_ui_row(2)
    owner_position = table.rowViewportPosition(source_row)
    editor.toggle_properties_widget(2)
    qapp.processEvents()
    assert bounds(editor, 2)[1] <= table.viewport().height()
    assert bounds(editor, 1) == first_position
    assert table.rowViewportPosition(source_row) == owner_position
    assert table.cellWidget(2, 0) is first
    assert first.input_widgets["Radius X"].text() == "1.2345"


def test_oversized_panel_keeps_top_accessible_and_bottom_scrollable(
    editor, qapp, monkeypatch
):
    monkeypatch.setattr(
        SurfacePropertiesWidget, "sizeHint", lambda self: QSize(420, 900)
    )
    editor.resize(900, 250)
    qapp.processEvents()
    table = editor.tableWidget
    editor.toggle_properties_widget(2)
    qapp.processEvents()
    top, bottom = bounds(editor, 2)
    assert bottom > table.viewport().height()
    assert top == 0
    assert table.verticalScrollBar().value() < table.verticalScrollBar().maximum()


def test_repeated_reveal_does_not_move_a_panel_already_visible(editor, qapp):
    editor.toggle_properties_widget(2)
    before = bounds(editor, 2)
    editor._scroll_to_properties(2)
    qapp.processEvents()
    assert bounds(editor, 2) == before
