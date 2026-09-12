"""Scene replacement and panel wiring preserve presentation-only interaction."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from matplotlib.patches import Polygon
from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget

from optiland_gui.surface_interaction import SurfaceInteractionState
from optiland_gui.viewer_panel import MatplotlibViewer


def test_prepared_scene_before_state_and_state_replacement(qapp, minimal_optic):
    connector = MagicMock()
    connector.get_optic.return_value = minimal_optic
    viewer = MatplotlibViewer(connector)
    viewer.clear_2d_highlights()
    viewer.ax.clear()
    body = Polygon([(0, 0), (1, 0), (1, 1)], facecolor="grey")
    viewer.ax.add_patch(body)
    normal_width = body.get_linewidth()
    surfaces = tuple(minimal_optic.surfaces)
    coordinates = (np.array([0, 0]), np.array([0, 1]))
    viewer.install_2d_highlight_bindings(
        {body: surfaces[1:3]},
        {},
        {surfaces[1]: coordinates},
        {surfaces[3]: (np.array([2, 2]), np.array([0, 0]))},
    )
    first = SurfaceInteractionState(viewer)
    first.sync_document(minimal_optic)
    first.set_selected_indices([1])
    viewer.set_interaction_state(first)
    assert body.get_linewidth() > normal_width
    assert any(
        binding.artist.get_marker() == "+"
        for binding in viewer.highlight_controller.bindings
        if binding.overlay
    )

    second = SurfaceInteractionState(viewer)
    second.sync_document(minimal_optic)
    viewer.set_interaction_state(second)
    assert body.get_linewidth() == normal_width
    first.set_hover(2)
    assert body.get_linewidth() == normal_width  # Old document signals are detached.
    second.set_hover(1)
    assert body.get_linewidth() > normal_width
    viewer.clear_2d_highlights()
    assert body.get_linewidth() == normal_width
    assert viewer.highlight_controller.bindings == []
    assert len(viewer.ax.lines) == 0
    viewer.close()


def test_reject_mismatched_scene_identity_and_nonfinite_reference(
    qapp, minimal_optic, monkeypatch
):
    from optiland_gui.layout_highlighting import Surface2D

    connector = MagicMock()
    connector.get_optic.return_value = minimal_optic
    viewer = MatplotlibViewer(connector)
    state = SurfaceInteractionState(viewer)
    state.sync_document(minimal_optic)
    viewer.set_interaction_state(state)
    state.set_selected_indices([1])
    viewer.highlight_controller.install({}, minimal_optic, (object(),))
    assert viewer.highlight_controller.bindings == []
    monkeypatch.setattr(
        Surface2D,
        "_compute_sag",
        lambda *_: tuple(np.full(3, np.nan) for _ in range(3)),
    )
    viewer.highlight_controller.install({}, minimal_optic)
    assert viewer.highlight_controller.bindings == []
    viewer.close()


def test_panel_manager_connects_actual_editor_and_2d_viewer_to_one_state(
    qapp, highlighting_connector, monkeypatch
):
    from optiland_gui import panel_manager
    from tests.gui.test_calculation_jobs import wait_for

    # Unrelated panels must not start VTK windows or an IPython kernel here.
    class PassivePanel(QWidget):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class TwoDimensionalPanel(QWidget):
        def __init__(self, connector):
            super().__init__()
            self.viewer2D = MatplotlibViewer(connector)
            QVBoxLayout(self).addWidget(self.viewer2D)

    for name in (
        "SidebarWidget",
        "SystemPropertiesPanel",
        "AnalysisPanel",
        "OptimizationPanel",
        "PythonTerminalWidget",
    ):
        monkeypatch.setattr(panel_manager, name, PassivePanel)
    monkeypatch.setattr(panel_manager, "ViewerPanel", TwoDimensionalPanel)
    connector = highlighting_connector
    window = QMainWindow()
    window.iface = SimpleNamespace()
    manager = panel_manager.PanelManager(window, connector)
    manager.create_all_panels(window)
    viewer = manager.viewer_panel.viewer2D
    window.show()
    manager.viewer_panel.show()
    wait_for(qapp, lambda: viewer.layout_job.data is not None)
    state = manager.surface_interaction
    assert manager.lens_editor.interaction_state is state
    assert manager.viewer_panel.viewer2D.interaction_state is state
    changed = MagicMock()
    connector.opticChanged.connect(changed)
    manager.lens_editor.tableWidget.selectRow(1)
    assert state.selected_surfaces == (connector.get_optic().surfaces[1],)
    bodies = [
        binding
        for binding in manager.viewer_panel.viewer2D.highlight_controller.bindings
        if binding.body
    ]
    assert bodies and all(binding.artist.get_linewidth() == 1.5 for binding in bodies)
    changed.assert_not_called()
    manager.viewer_panel.close()
    manager.viewer_panel.deleteLater()
    window.close()
    window.deleteLater()
