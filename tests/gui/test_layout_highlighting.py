"""Retained 2D artists identify exact faces without optical recomputation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from optiland.optic import Optic
from optiland.physical_apertures import RadialAperture
from optiland.visualization.system.system import OpticalSystem
from optiland_gui.layout_highlighting import LayoutHighlightController
from optiland_gui.surface_interaction import SurfaceInteractionState


def scene(optic, qapp):
    figure = Figure()
    canvas = FigureCanvasQTAgg(figure)
    ax = figure.add_subplot()
    system = OpticalSystem(
        optic, SimpleNamespace(r_extent=np.full(len(optic.surfaces), 5.0))
    )
    artists = system.plot(ax)
    state = SurfaceInteractionState(canvas)
    state.sync_document(optic)
    controller = LayoutHighlightController(ax, canvas, state, lambda: "dark")
    limits = ax.get_xlim(), ax.get_ylim()
    controller.install(artists, optic)
    assert (ax.get_xlim(), ax.get_ylim()) == limits
    return canvas, ax, state, controller


def test_exact_faces_and_body_have_independent_styles(qapp, minimal_optic, monkeypatch):
    canvas, ax, state, controller = scene(minimal_optic, qapp)
    body = next(binding for binding in controller.bindings if binding.body)
    faces = {
        state.index_of(binding.surfaces[0]): binding
        for binding in controller.bindings
        if binding.overlay
    }
    original = dict(body.normal)
    indicator = next(line for line in ax.lines if line.get_linewidth() == 0.3)
    limits = ax.get_xlim(), ax.get_ylim()
    trace = MagicMock(side_effect=AssertionError("Hover retraced rays"))
    monkeypatch.setattr(minimal_optic, "trace", trace)
    state.set_selected_indices([1])
    selected_fill = body.artist.get_facecolor()
    selected_line = faces[1].artist.get_color()
    assert selected_fill[3] < original["facecolor"][3]
    assert faces[1].artist.get_linewidth() > body.artist.get_linewidth()
    assert indicator.get_linewidth() == 0.3
    state.set_hover(2, 3)
    assert body.artist.get_facecolor() == selected_fill
    assert faces[1].artist.get_color() == selected_line
    assert faces[2].artist.get_color() != selected_line
    assert faces[2].artist.get_linewidth() > body.artist.get_linewidth()
    state.set_hover(1, 3)
    assert faces[1].artist.get_color() == selected_line
    assert not faces[2].artist.get_visible()
    state.set_selected_indices([2])
    assert body.artist.get_facecolor() == selected_fill
    assert faces[2].artist.get_color() == selected_line
    state.set_hover()
    assert not faces[1].artist.get_visible()
    state.set_selected_indices([])
    assert body.artist.get_facecolor() == original["facecolor"]
    assert body.artist.get_edgecolor() == original["edgecolor"]
    assert (ax.get_xlim(), ax.get_ylim()) == limits
    trace.assert_not_called()
    canvas.close()


def test_standalone_surface_and_reference_marker(qapp):
    optic = Optic()
    optic.wavelengths.add(value=0.55, is_primary=True)
    optic.surfaces.add(index=0, radius=np.inf, thickness=10)
    optic.surfaces.add(index=1, radius=np.inf, thickness=5)  # neutral reference
    optic.surfaces.add(index=2, radius=np.inf, thickness=-5, material="mirror")
    optic.surfaces.add(index=3, radius=np.inf)
    canvas, ax, state, controller = scene(optic, qapp)
    assert not any(binding.body for binding in controller.bindings)
    mirror = next(
        binding
        for binding in controller.bindings
        if binding.surfaces == (optic.surfaces[2],)
    )
    marker = next(
        binding
        for binding in controller.bindings
        if binding.surfaces == (optic.surfaces[1],)
    )
    assert marker.overlay and not marker.artist.get_visible()
    limits = ax.get_xlim(), ax.get_ylim()
    state.set_selected_indices([2])
    assert mirror.artist.get_linewidth() > mirror.normal["linewidth"]
    state.set_hover(1)
    assert marker.artist.get_visible()
    assert marker.artist.get_linestyle() == "--"
    assert (ax.get_xlim(), ax.get_ylim()) == limits
    canvas.close()


def test_cemented_interface_highlights_adjoining_bodies_and_annular_patches(qapp):
    optic = Optic()
    optic.wavelengths.add(value=0.55, is_primary=True)
    optic.surfaces.add(index=0, radius=np.inf, thickness=10)
    optic.surfaces.add(
        index=1,
        radius=np.inf,
        thickness=3,
        material="N-BK7",
        aperture=RadialAperture(r_max=5, r_min=1),
    )
    optic.surfaces.add(
        index=2,
        radius=np.inf,
        thickness=3,
        material="F2",
        aperture=RadialAperture(r_max=5, r_min=1),
    )
    optic.surfaces.add(
        index=3, radius=np.inf, thickness=10, aperture=RadialAperture(r_max=5, r_min=1)
    )
    optic.surfaces.add(index=4, radius=np.inf)
    canvas, _, state, controller = scene(optic, qapp)
    bodies = [binding for binding in controller.bindings if binding.body]
    assert len(bodies) >= 4  # split patches, including both adjoining glass regions
    assert {tuple(state.index_of(s) for s in b.surfaces) for b in bodies} == {
        (1, 2),
        (2, 3),
    }
    state.set_selected_indices([2])
    assert all(binding.artist.get_linewidth() == 1.5 for binding in bodies)
    interface = [
        binding
        for binding in controller.bindings
        if binding.overlay and binding.surfaces == (optic.surfaces[2],)
    ]
    assert len(interface) == 1
    assert interface[0].artist.get_linewidth() > bodies[0].artist.get_linewidth()
    canvas.close()


def test_snapshot_maps_to_live_identity_and_uses_latest_state(qapp, minimal_optic):
    canvas, ax, state, controller = scene(minimal_optic, qapp)
    captured = tuple(minimal_optic.surfaces)
    snapshot = Optic.from_dict(minimal_optic.to_dict())
    controller.clear()
    ax.clear()
    system = OpticalSystem(snapshot, SimpleNamespace(r_extent=np.full(4, 5.0)))
    artists = system.plot(ax)
    state.set_selected_indices([2])
    controller.install(artists, snapshot, captured)
    body = next(binding for binding in controller.bindings if binding.body)
    assert body.surfaces == captured[1:3]
    assert body.artist.get_linewidth() == 1.5
    state.sync_document(Optic())
    assert body.artist.get_linewidth() == body.normal["linewidth"]
    canvas.close()


def test_prepared_bindings_install_and_clear_without_reconstructing_optic(
    qapp, minimal_optic
):
    canvas, ax, state, controller = scene(minimal_optic, qapp)
    body = next(binding for binding in controller.bindings if binding.body)
    front = next(
        binding
        for binding in controller.bindings
        if binding.overlay and binding.surfaces == (minimal_optic.surfaces[1],)
    )
    coordinates = tuple(np.copy(value) for value in front.artist.get_data())
    body_artist, owned = body.artist, body.surfaces
    controller.clear()
    baseline_lines = len(ax.lines)
    state.set_selected_indices([1])
    controller.install_bindings({body_artist: owned}, {}, {owned[0]: coordinates}, {})
    assert body_artist.get_linewidth() == 1.5
    assert len(ax.lines) == baseline_lines + 1
    controller.install_bindings({body_artist: owned}, {}, {owned[0]: coordinates}, {})
    assert len(ax.lines) == baseline_lines + 1  # Reinstallation removes old overlays.
    controller.clear()
    assert len(ax.lines) == baseline_lines
    assert body_artist.get_linewidth() == body.normal["linewidth"]
    canvas.close()


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_viewer_reuses_current_artists_on_hover_and_restores_after_rebuild(
    qapp, highlighting_connector, monkeypatch, theme
):
    from optiland_gui.viewer_panel import MatplotlibViewer
    from tests.gui.test_calculation_jobs import wait_for

    connector = highlighting_connector
    optic = connector.get_optic()
    viewer = MatplotlibViewer(connector)
    state = SurfaceInteractionState(viewer)
    state.sync_document(optic)
    viewer.set_interaction_state(state)
    viewer.show()
    wait_for(qapp, lambda: viewer.layout_job.data is not None)
    viewer.update_theme(theme)
    changed = MagicMock()
    connector.opticChanged.connect(changed)
    limits = viewer.ax.get_xlim(), viewer.ax.get_ylim()
    original_artist = viewer.highlight_controller.bindings[0].artist
    original_plot = viewer.plot_optic
    spy = MagicMock(side_effect=AssertionError("Selection rebuilt the plot"))
    monkeypatch.setattr(viewer, "plot_optic", spy)
    state.set_selected_indices([1])
    state.set_hover(2, 4)
    assert viewer.highlight_controller.bindings[0].artist is original_artist
    assert (viewer.ax.get_xlim(), viewer.ax.get_ylim()) == limits
    old_data = viewer.layout_job.data
    viewer.num_rays_spinbox.setValue(7)
    original_plot(preserve_zoom=True)
    wait_for(qapp, lambda: viewer.layout_job.data is not old_data)
    assert state.selected_surfaces == (optic.surfaces[1],)
    assert viewer.highlight_controller.bindings[0].artist is not original_artist
    assert original_artist.axes is None
    np.testing.assert_allclose((viewer.ax.get_xlim(), viewer.ax.get_ylim()), limits)
    body = next(b for b in viewer.highlight_controller.bindings if b.body)
    assert body.artist.get_linewidth() == 1.5
    spy.assert_not_called()
    changed.assert_not_called()
    viewer.close()
    viewer.deleteLater()
