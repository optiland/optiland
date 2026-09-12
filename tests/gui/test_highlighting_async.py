"""Worker acceptance binds retained artists to the current editor identities."""

from __future__ import annotations

from unittest.mock import MagicMock

from PySide6.QtCore import QCoreApplication, QEvent

from optiland.optic import Optic
from optiland_gui.surface_interaction import SurfaceInteractionState
from optiland_gui.viewer_panel import MatplotlibViewer
from tests.gui.test_calculation_jobs import wait_for


def test_worker_uses_latest_selection_and_rejects_previous_document(
    qapp, highlighting_connector, monkeypatch
):
    connector = highlighting_connector
    optic = connector.get_optic()
    viewer = MatplotlibViewer(connector)
    state = SurfaceInteractionState()
    state.sync_document(optic)
    viewer.set_interaction_state(state)
    connector.opticLoaded.connect(lambda: state.sync_document(connector.get_optic()))
    completed = []
    connector.calculation_jobs.finished.connect(completed.append)
    trace = MagicMock(side_effect=AssertionError("GUI traced the live document"))
    monkeypatch.setattr(type(optic), "trace", trace)
    viewer.show()
    viewer.plot_optic()
    # A real QProcess request has captured its snapshot; change only UI state
    # before servicing worker output. The presenter must use this latest state.
    assert viewer.layout_job._job_id is not None
    state.set_selected_indices([2])
    state.set_hover(1)
    wait_for(qapp, lambda: viewer.layout_job.data is not None)
    controller = viewer.highlight_controller
    faces = {b.surfaces[0]: b.artist for b in controller.bindings if b.overlay}
    assert faces[optic.surfaces[2]].get_linewidth() == 2.4
    assert faces[optic.surfaces[1]].get_linewidth() == 2.1
    old_artists = [b.artist for b in controller.bindings]
    old_result = completed[-1]
    old_data = viewer.layout_job.data

    replacement = Optic.from_dict(optic.to_dict())
    replacement.name = "Replacement document"
    connector.load_optic_from_object(replacement)
    replacement = connector.get_optic()
    state.set_selected_indices([1])
    # Stale data may stay visible during calculation, but must never pick up
    # another document's selection, including a theme redraw of retained data.
    viewer.update_theme("light")
    assert all(not b.artist.get_visible() for b in controller.bindings if b.overlay)
    viewer.layout_job._finished(old_result)
    assert viewer.layout_job.data is old_data
    wait_for(qapp, lambda: viewer.layout_job.data is not old_data)
    assert all(artist.axes is None for artist in old_artists)
    assert controller.document is replacement
    assert all(
        state.index_of(surface) >= 0
        for binding in controller.bindings
        for surface in binding.surfaces
    )
    face = next(
        b.artist
        for b in controller.bindings
        if b.overlay and b.surfaces == (replacement.surfaces[1],)
    )
    assert face.get_visible() and face.get_linewidth() == 2.4
    trace.assert_not_called()
    viewer.close()
    viewer.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    # QObject destruction must disconnect the controller from the shared state.
    state.set_selected_indices([2])
