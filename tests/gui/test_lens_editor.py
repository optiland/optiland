"""Tests for LensEditor variable highlighting and SurfaceTypeWidget badge."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def mock_connector(minimal_optic, qapp):
    conn = MagicMock()
    conn._optic = minimal_optic
    conn.toast_manager = MagicMock()
    conn.COL_TYPE = 0
    conn.COL_COMMENT = 1
    conn.COL_RADIUS = 2
    conn.COL_THICKNESS = 3
    conn.COL_MATERIAL = 4
    conn.COL_CONIC = 5
    conn.COL_SEMI_DIAMETER = 6
    conn.get_column_headers.return_value = [
        "Type", "Comment", "Radius", "Thickness", "Material", "Conic", "Semi-Diameter"
    ]
    conn.get_surface_count.return_value = 4
    conn.get_optimization_variables.return_value = []
    conn.get_surface_type_info.return_value = {
        "display_text": "Standard",
        "is_changeable": True,
        "has_extra_params": False,
    }
    conn.get_surface_geometry_params.return_value = {}
    conn.get_surface_data.return_value = ""
    conn.get_available_surface_types.return_value = ["standard", "aspheric"]
    return conn


class TestSurfaceTypeWidgetBadge:
    def _make_widget(self, mock_connector):
        from optiland_gui.lens_editor import SurfaceTypeWidget

        type_info = {
            "display_text": "Standard",
            "is_changeable": True,
            "has_extra_params": False,
        }
        return SurfaceTypeWidget(1, type_info, mock_connector)

    def test_badge_hidden_by_default(self, qapp, mock_connector):
        w = self._make_widget(mock_connector)
        # isHidden() checks only the widget's own flag (parent need not be shown)
        assert w._var_badge.isHidden()

    def test_badge_shown_when_variables_set(self, qapp, mock_connector):
        w = self._make_widget(mock_connector)
        w.setHasVariables(["asphere_coeff"])
        assert not w._var_badge.isHidden()
        assert "asphere_coeff" in w._var_badge.toolTip()

    def test_badge_hidden_again_when_cleared(self, qapp, mock_connector):
        w = self._make_widget(mock_connector)
        w.setHasVariables(["asphere_coeff"])
        w.setHasVariables([])
        assert w._var_badge.isHidden()


class TestLensEditorVariableHighlighting:
    def test_no_highlight_when_no_variables(self, qapp, mock_connector):
        from PySide6.QtGui import QColor

        from optiland_gui.lens_editor import LensEditor

        mock_connector.get_optimization_variables.return_value = []
        editor = LensEditor(mock_connector)
        editor.load_data()

        # The highlight color used for variables is (100, 150, 255, 80).
        highlight = QColor(100, 150, 255, 80)
        item = editor.tableWidget.item(1, mock_connector.COL_RADIUS)
        if item is not None:
            bg = item.background().color()
            assert bg != highlight

    def test_radius_variable_highlights_radius_cell(self, qapp, mock_connector):
        from PySide6.QtGui import QColor

        from optiland_gui.lens_editor import LensEditor

        mock_connector.get_optimization_variables.return_value = [
            {"surface_number": 1, "type": "radius", "min_val": None, "max_val": None}
        ]
        editor = LensEditor(mock_connector)
        editor.load_data()

        item = editor.tableWidget.item(1, mock_connector.COL_RADIUS)
        assert item is not None
        # Blue highlight (100, 150, 255, 80) must be set
        expected = QColor(100, 150, 255, 80)
        assert item.background().color() == expected

    def test_asphere_variable_shows_badge_on_type_column(self, qapp, mock_connector):
        from optiland_gui.lens_editor import LensEditor, SurfaceTypeWidget

        mock_connector.get_optimization_variables.return_value = [
            {
                "surface_number": 1,
                "type": "asphere_coeff",
                "min_val": None,
                "max_val": None,
                "coeff_number": 0,
            }
        ]
        editor = LensEditor(mock_connector)
        editor.load_data()

        widget = editor.tableWidget.cellWidget(1, mock_connector.COL_TYPE)
        assert isinstance(widget, SurfaceTypeWidget)
        # isHidden() checks the widget's own flag — no need for parent to be shown
        assert not widget._var_badge.isHidden()
