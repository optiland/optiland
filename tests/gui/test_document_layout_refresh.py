"""Canonical document edits reach only the latest visible layout request."""

from __future__ import annotations

import numpy as np

from optiland_gui.optiland_connector import OptilandConnector
from tests.gui.test_calculation_jobs import wait_for


def test_metadata_noop_transactions_hidden_views_and_theme_have_bounded_work(
    qapp, minimal_optic, monkeypatch
):
    import optiland_gui.viewer_panel as module

    monkeypatch.setattr(module, "VTK_AVAILABLE", False)
    connector = OptilandConnector()
    connector.load_optic_from_object(minimal_optic)
    connector.set_polarization_state("unpolarized")
    panel = module.ViewerPanel(connector)
    jobs = connector.calculation_jobs
    two_d = panel.viewer2D.layout_job
    sag = panel.sagViewer.layout_job
    panel.resize(700, 600)
    panel.show()
    try:
        wait_for(qapp, lambda: two_d._completed_key is not None, timeout=30)
        serial, token = jobs._serial, connector.document_state.token
        connector.set_surface_data(1, connector.COL_COMMENT, "Useful new label")
        connector.set_polarization_state("unpolarized")
        panel.update_theme("light")
        qapp.processEvents()
        assert connector.document_state.token == token
        assert jobs._serial == serial
        assert sag.data is None

        panel.preserve_zoom_checkbox.setChecked(True)
        panel.viewer2D.ax.set_xlim(0, 40)
        panel.viewer2D.ax.set_ylim(-3, 3)
        with connector.change_transaction():
            connector.notify_change("optical")
            connector.notify_change("optical")
            connector.notify_change("metadata")
            assert connector.document_state.token.revision == token.revision + 1
            assert jobs._serial == serial  # Request waits for the event-loop turn.
            assert two_d._completed_key is None
        wait_for(qapp, lambda: two_d._completed_key is not None, timeout=30)
        assert jobs._serial == serial + 1
        np.testing.assert_allclose(panel.viewer2D.ax.get_xlim(), (0, 40))
        np.testing.assert_allclose(panel.viewer2D.ax.get_ylim(), (-3, 3))
        assert sag.data is None

        panel.hide()
        serial = jobs._serial
        connector.notify_change("optical")
        connector.notify_change("optical")
        qapp.processEvents()
        assert jobs._serial == serial
        panel.show()
        wait_for(qapp, lambda: two_d._completed_key is not None, timeout=30)
        assert jobs._serial == serial + 1
        panel.tabWidget.setCurrentWidget(panel.sagViewer)
        wait_for(qapp, lambda: sag._completed_key is not None, timeout=30)
        serial = jobs._serial
        panel.tabWidget.setCurrentIndex(0)
        qapp.processEvents()
        assert jobs._serial == serial
    finally:
        panel.hide()
        jobs.shutdown()
        wait_for(qapp, lambda: jobs._process is None)
        panel.deleteLater()
