"""Presentation-time zoom preferences coexist with explicit navigation."""

from __future__ import annotations

import numpy as np
import pytest

from optiland_gui.optiland_connector import OptilandConnector


@pytest.fixture()
def scene(qapp, monkeypatch):
    import optiland_gui.viewer_panel as module

    monkeypatch.setattr(module, "VTK_AVAILABLE", False)
    connector = OptilandConnector()
    panel = module.ViewerPanel(connector)
    viewer = panel.viewer2D
    data = {
        "name": "Synthetic layout",
        "primitives": [
            {
                "kind": "line",
                "role": "surface",
                "surfaces": (),
                "xy": np.array([[0.0, -1.0], [10.0, 1.0]]),
                "linewidth": 1.0,
                "linestyle": "-",
                "label": "surface",
            }
        ],
        "annotations": [],
        "boundaries": {},
        "references": {},
    }
    context = {
        "document_id": connector.document_state.token.document_id,
        "surface_identities": tuple(connector.get_optic().surfaces),
    }
    viewer._present_layout(data, context)
    viewer.layout_job.data, viewer.layout_job.context = data, context
    yield panel, viewer, data, context
    panel.close()
    panel.deleteLater()
    connector.calculation_jobs.shutdown()


def set_automatic_limits(viewer):
    viewer._is_plotting = True
    viewer.ax.set_xlim(20, 30)
    viewer.ax.set_ylim(-4, 4)
    viewer._is_plotting = False
    viewer._user_initiated_view_change = False


def test_checkbox_is_read_when_pending_result_is_presented(scene):
    panel, viewer, data, context = scene
    set_automatic_limits(viewer)
    panel.preserve_zoom_checkbox.setChecked(True)
    viewer.plot_optic()  # Explicit default must not clear the persistent preference.
    viewer._present_layout(data, context)
    np.testing.assert_allclose(viewer.ax.get_xlim(), (20, 30))
    panel.preserve_zoom_checkbox.setChecked(False)
    viewer._present_layout(data, context)
    assert viewer.ax.get_xlim()[1] < 20
    assert panel.connector.calculation_jobs._serial == 0


def test_home_fits_retained_data_even_when_preserve_is_checked(scene):
    panel, viewer, data, context = scene
    panel.preserve_zoom_checkbox.setChecked(True)
    set_automatic_limits(viewer)
    viewer.reset_view()
    assert viewer.ax.get_xlim()[1] < 20
    assert panel.preserve_zoom_checkbox.isChecked()
    assert viewer.preserve_zoom
    assert panel.connector.calculation_jobs._serial == 0


def test_explicit_preservation_and_new_document_fit_remain_distinct(scene):
    panel, viewer, data, context = scene
    set_automatic_limits(viewer)
    viewer.plot_optic(preserve_zoom=True)
    viewer._present_layout(data, context)
    np.testing.assert_allclose(viewer.ax.get_xlim(), (20, 30))
    panel.preserve_zoom_checkbox.setChecked(True)
    viewer._present_layout(data, {**context, "document_id": "replacement"})
    assert viewer.ax.get_xlim()[1] < 20
    assert panel.connector.calculation_jobs._serial == 0
