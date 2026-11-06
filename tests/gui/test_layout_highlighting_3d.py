"""Precise 3D actor ownership and redraw-only editor linkage."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import vtk
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QWidget

from optiland.optic import Optic
from optiland.physical_apertures import RadialAperture
from optiland.visualization.system.lens import Lens3D
from optiland.visualization.system.surface import Surface3D
from optiland_gui.layout_highlighting_3d import LayoutHighlightController3D
from optiland_gui.services.job_records import OpticSnapshot
from optiland_gui.services.layout_tasks import (
    _ActorCollector,
    _highlight_edges,
    prepare_3d,
)
from optiland_gui.surface_interaction import SurfaceInteractionState


def prepared(optic):
    return prepare_3d(
        OpticSnapshot.capture(optic), {}, lambda *args: None, threading.Event()
    )


def doublet(*, tilted=False, annular=False):
    optic = Optic()
    optic.set_aperture("EPD", 8)
    optic.fields.set_type("angle")
    optic.fields.add(y=0)
    optic.wavelengths.add(0.55, is_primary=True)
    optic.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
    for radius, material in ((50, "N-BK7"), (-80, "N-BK7"), (-50, "Air")):
        optic.surfaces.add(
            index=len(optic.surfaces),
            radius=radius,
            thickness=3,
            material=material,
            aperture=RadialAperture(r_max=5, r_min=1 if annular else 0),
        )
    optic.surfaces.add(index=4, radius=np.inf)
    optic.surfaces.stop_index = 1
    if tilted:
        optic.surfaces[2].geometry.cs.rx = 0.1
    return optic


@pytest.mark.parametrize("tilted", [False, True])
def test_cemented_body_meshes_and_exact_faces_have_explicit_ownership(
    set_test_backend, tilted
):
    data = prepared(doublet(tilted=tilted))
    owned = {mesh["surfaces"] for mesh in data["meshes"] if mesh["role"] == "lens"}
    assert (1, 2) in owned and (2, 3) in owned
    if not tilted:
        assert owned == {(1, 2), (2, 3)}
    faces = [mesh for mesh in data["meshes"] if mesh["role"] == "face_highlight"]
    assert {mesh["surfaces"] for mesh in faces} == {(1,), (2,), (3,), (4,)}
    assert all(np.isfinite(mesh["points"]).all() for mesh in faces)
    assert all(len(mesh["surfaces"]) == 1 for mesh in faces)


def actor(role, surfaces):
    result = vtk.vtkActor()
    result.GetProperty().SetColor(0.8, 0.7, 0.6)
    result.GetProperty().SetOpacity(0.7)
    result.GetProperty().SetSpecular(0.95)
    result.GetProperty().SetSpecularPower(77)
    result.SetVisibility(role not in {"body_edge", "surface_edge", "face_highlight"})
    return result, {"role": role, "surfaces": surfaces}


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_selection_hover_precedence_exact_face_and_style_restoration(qapp, theme):
    optic = doublet()
    viewer = QWidget()
    viewer.current_theme = theme
    window = MagicMock()
    viewer.vtkWidget = SimpleNamespace(GetRenderWindow=lambda: window)
    state = SurfaceInteractionState(viewer)
    state.sync_document(optic)
    controller = LayoutHighlightController3D(viewer, state)
    viewer.show()
    qapp.processEvents()
    specs = [
        actor("lens", (1, 2)),
        actor("lens", (2, 3)),
        actor("body_edge", (1, 2)),
        actor("surface_edge", (1,)),
        actor("face_highlight", (1,)),
        actor("face_highlight", (2,)),
    ]
    controller.install(specs, tuple(optic.surfaces))
    state.set_selected_indices([1])
    state.set_hover(3)
    qapp.processEvents()
    selected = QColor("#007ACC" if theme == "dark" else "#6C757D")
    selected_rgb = selected.redF(), selected.greenF(), selected.blueF()
    np.testing.assert_allclose(specs[0][0].GetProperty().GetColor(), selected_rgb)
    assert specs[1][0].GetProperty().GetColor() != selected_rgb
    assert (
        specs[3][0].GetProperty().GetLineWidth()
        > specs[2][0].GetProperty().GetLineWidth()
    )
    assert specs[4][0].GetVisibility() and not specs[5][0].GetVisibility()
    renders = window.Render.call_count
    state.set_hover(1)
    state.set_hover(2)
    qapp.processEvents()
    assert window.Render.call_count == renders + 1
    np.testing.assert_allclose(specs[0][0].GetProperty().GetColor(), selected_rgb)
    assert specs[5][0].GetVisibility()
    controller.clear()
    np.testing.assert_allclose(specs[0][0].GetProperty().GetColor(), (0.8, 0.7, 0.6))
    assert specs[0][0].GetProperty().GetOpacity() == 0.7
    assert specs[0][0].GetProperty().GetSpecularPower() == 77
    assert not specs[4][0].GetVisibility()
    viewer.close()
    viewer.deleteLater()


def test_hidden_changes_no_render_and_deleted_owner_cancels_pending_callback(qapp):
    optic = doublet()
    viewer = QWidget()
    viewer.current_theme = "dark"
    window = MagicMock()
    viewer.vtkWidget = SimpleNamespace(GetRenderWindow=lambda: window)
    state = SurfaceInteractionState()
    state.sync_document(optic)
    controller = LayoutHighlightController3D(viewer, state)
    specs = [actor("lens", (1, 2))]
    controller.install(specs, tuple(optic.surfaces))
    state.set_selected_indices([1])
    qapp.processEvents()
    window.Render.assert_not_called()
    viewer.show()
    qapp.processEvents()
    assert window.Render.call_count == 1
    viewer.hide()
    state.set_hover(3)
    qapp.processEvents()
    assert window.Render.call_count == 1
    viewer.show()
    state.set_selected_indices([2])
    viewer.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    qapp.processEvents()
    assert window.Render.call_count == 1


def test_document_replacement_restores_old_scene_materials(qapp):
    optic = doublet()
    viewer = QWidget()
    viewer.current_theme = "dark"
    viewer.vtkWidget = SimpleNamespace(GetRenderWindow=lambda: MagicMock())
    state = SurfaceInteractionState(viewer)
    state.sync_document(optic)
    controller = LayoutHighlightController3D(viewer, state)
    viewer.show()
    specs = [actor("lens", (1, 2))]
    controller.install(specs, tuple(optic.surfaces))
    state.set_selected_indices([1])
    qapp.processEvents()
    state.sync_document(doublet())
    qapp.processEvents()
    np.testing.assert_allclose(specs[0][0].GetProperty().GetColor(), (0.8, 0.7, 0.6))
    viewer.close()
    viewer.deleteLater()


def test_annular_surface_uses_finite_clipped_cells_and_preserves_body_identity():
    from vtk.util.numpy_support import vtk_to_numpy

    optic = doublet(annular=True)
    surfaces = [Surface3D(surface, 5) for surface in tuple(optic.surfaces)[1:4]]
    lens = Lens3D(surfaces)
    assert not lens.is_symmetric
    collector = _ActorCollector()
    lens.plot(collector)
    assert set(lens.artist_surfaces) == set(collector.actors)
    for surface in surfaces:
        assert not surface.supports_revolution
        mesh_actor = surface.get_surface()
        mesh_actor.GetMapper().Update()
        mesh = mesh_actor.GetMapper().GetInput()
        points = vtk_to_numpy(mesh.GetPoints().GetData())
        cells = vtk_to_numpy(mesh.GetPolys().GetConnectivityArray())
        assert np.isfinite(points).all()
        radius = np.linalg.norm(points[cells, :2], axis=1)
        assert radius.min() >= 1 and radius.max() <= 5


def test_standalone_mirror_and_omitted_reference_have_exact_face_ids():
    optic = Optic()
    optic.set_aperture("EPD", 2)
    optic.fields.set_type("angle")
    optic.fields.add(y=0)
    optic.wavelengths.add(0.55, is_primary=True)
    optic.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
    optic.surfaces.add(index=1, radius=np.inf, thickness=5)
    optic.surfaces.add(
        index=2, radius=np.inf, thickness=-5, material="mirror", is_stop=True
    )
    optic.surfaces.add(index=3, radius=np.inf)
    meshes = prepared(optic)["meshes"]
    assert not any(mesh["role"] == "lens" for mesh in meshes)
    mirror = next(
        mesh
        for mesh in meshes
        if mesh["role"] == "surface" and mesh["surfaces"] == (2,)
    )
    assert mirror["opacity"] == 1
    assert {(1,), (2,)} <= {
        mesh["surfaces"] for mesh in meshes if mesh["role"] == "face_highlight"
    }


@pytest.mark.skipif(
    os.environ.get("QT_QPA_PLATFORM") != "windows",
    reason="Native VTK appearance/retained-scene validation uses the desktop lane",
)
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_native_retained_scene_camera_and_zero_calculation_on_interaction(
    qapp, monkeypatch, theme
):
    from PySide6.QtTest import QTest

    from optiland_gui.optiland_connector import OptilandConnector
    from optiland_gui.viewer_panel import VTKViewer

    optic = doublet()
    data = prepared(optic)
    connector = OptilandConnector()
    connector._optic = optic
    connector.notify_change("replacement")
    viewer = VTKViewer(connector)
    viewer.layout_job._refresh_timer.timeout.disconnect()
    viewer.current_theme = theme
    state = SurfaceInteractionState(viewer)
    state.sync_document(optic)
    viewer.clear_3d_highlights()
    viewer.resize(920, 700)
    viewer.show()
    QTest.qWaitForWindowExposed(viewer)
    QTest.qWait(100)
    try:
        context = {
            "surface_identities": tuple(optic.surfaces),
            "document_id": connector.document_state.token.document_id,
        }
        viewer._present_layout(data, context)
        previous = SurfaceInteractionState(viewer)
        previous.sync_document(optic)
        viewer.set_interaction_state(previous)
        viewer.set_interaction_state(state)
        previous.set_selected_indices([4])
        viewer.layout_job.data, viewer.layout_job.context = data, context
        camera = viewer.renderer.GetActiveCamera()
        camera.Azimuth(25)
        camera.Elevation(15)
        camera.Zoom(1.6)
        camera_state = camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp()
        specs = list(viewer._scene_actor_specs)
        meshes = [actor.GetMapper().GetInput() for actor, _ in specs]
        guard = MagicMock(side_effect=AssertionError("Interaction captured optics"))
        monkeypatch.setattr(OpticSnapshot, "capture", guard)
        state.set_selected_indices([1])
        state.set_hover(3)
        QTest.qWait(50)
        assert viewer._scene_actor_specs == specs
        assert all(
            actor.GetMapper().GetInput() is mesh
            for (actor, _), mesh in zip(specs, meshes, strict=True)
        )
        assert (
            camera.GetPosition(),
            camera.GetFocalPoint(),
            camera.GetViewUp(),
        ) == camera_state
        assert viewer.renderer.GetActors().GetNumberOfItems() == len(specs)
        token = connector.document_state.edit_token
        viewer.update_theme("light" if theme == "dark" else "dark")
        viewer.update_theme(theme)
        QTest.qWait(50)
        assert (
            camera.GetPosition(),
            camera.GetFocalPoint(),
            camera.GetViewUp(),
        ) == camera_state
        assert connector.document_state.edit_token == token
        guard.assert_not_called()
        capture_dir = os.environ.get("OPTILAND_TEST_3D_CAPTURE_DIR")
        if capture_dir:
            capture = vtk.vtkWindowToImageFilter()
            capture.SetInput(viewer.vtkWidget.GetRenderWindow())
            capture.ReadFrontBufferOff()
            capture.Update()
            writer = vtk.vtkPNGWriter()
            writer.SetFileName(str(Path(capture_dir) / f"gui-013-{theme}.png"))
            writer.SetInputConnection(capture.GetOutputPort())
            writer.Write()
    finally:
        viewer.close()
        viewer.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        qapp.processEvents()


@pytest.mark.skipif(
    os.environ.get("QT_QPA_PLATFORM") != "windows",
    reason="Native VTK appearance validation uses the desktop lane",
)
def test_native_annular_and_opaque_surface_visibility(qapp):
    from PySide6.QtTest import QTest

    from optiland.visualization.system.mirror import Mirror3D
    from optiland_gui.optiland_connector import OptilandConnector
    from optiland_gui.viewer_panel import VTKViewer

    optic = doublet(annular=True)
    connector = OptilandConnector()
    connector._optic = optic
    connector.notify_change("replacement")
    viewer = VTKViewer(connector)
    viewer.layout_job._refresh_timer.timeout.disconnect()
    state = SurfaceInteractionState(viewer)
    state.sync_document(optic)
    viewer.set_interaction_state(state)
    viewer.resize(920, 700)
    viewer.show()
    QTest.qWaitForWindowExposed(viewer)
    QTest.qWait(100)
    try:
        identities = tuple(optic.surfaces)
        surfaces = [Surface3D(surface, 5) for surface in identities[1:4]]
        component = Lens3D(surfaces)
        collector = _ActorCollector()
        component.plot(collector)
        specs = []
        for item in collector.actors:
            owned = tuple(identities.index(s) for s in component.artist_surfaces[item])
            specs.append((item, {"role": "lens", "surfaces": owned}))
            edge = _highlight_edges(item)
            edge.VisibilityOff()
            edge.UseBoundsOff()
            specs.append((edge, {"role": "body_edge", "surfaces": owned}))
        for index, surface in enumerate(surfaces, 1):
            face = surface.get_surface()
            edge = _highlight_edges(face, boundary_only=True)
            for item, role in ((face, "face_highlight"), (edge, "surface_edge")):
                item.VisibilityOff()
                item.UseBoundsOff()
                specs.append((item, {"role": role, "surfaces": (index,)}))
        mirror = Mirror3D(identities[4], 3).get_surface()
        specs.append((mirror, {"role": "surface", "surfaces": (4,)}))
        for item, _ in specs:
            viewer.renderer.AddActor(item)
        viewer.renderer.SetBackground(0.1, 0.1, 0.1)
        viewer.renderer.ResetCamera()
        camera = viewer.renderer.GetActiveCamera()
        camera.SetPosition(18, 5, 30)
        camera.SetFocalPoint(0, 0, 7)
        camera.SetViewUp(0, 1, 0)
        viewer.renderer.ResetCameraClippingRange()
        viewer.install_3d_highlights(specs, {"surface_identities": identities})
        state.set_selected_indices([1])
        state.set_hover(3)
        QTest.qWait(50)
        assert mirror.GetProperty().GetOpacity() == 1
        capture_dir = os.environ.get("OPTILAND_TEST_3D_CAPTURE_DIR")
        if capture_dir:
            capture = vtk.vtkWindowToImageFilter()
            capture.SetInput(viewer.vtkWidget.GetRenderWindow())
            capture.ReadFrontBufferOff()
            capture.Update()
            writer = vtk.vtkPNGWriter()
            writer.SetFileName(str(Path(capture_dir) / "gui-013-annular-opaque.png"))
            writer.SetInputConnection(capture.GetOutputPort())
            writer.Write()
        state.set_selected_indices([4])
        state.set_hover(-1)
        QTest.qWait(50)
        assert mirror.GetProperty().GetOpacity() == 1
        state.set_selected_indices([])
        QTest.qWait(50)
        assert mirror.GetProperty().GetOpacity() == 1
    finally:
        viewer.close()
        viewer.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        qapp.processEvents()
