"""Compact planar geometry, aperture holes and shared scene transport."""

from __future__ import annotations

import pickle
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from optiland.geometries.standard import StandardGeometry
from optiland.optic import Optic
from optiland.physical_apertures import (
    EllipticalAperture,
    OffsetRadialAperture,
    RadialAperture,
    RectangularAperture,
)
from optiland.visualization.system.surface import Surface3D
from optiland_gui.services.calculation_worker import encode_message
from optiland_gui.services.job_records import OpticSnapshot
from optiland_gui.services.layout_tasks import prepare_3d
from optiland_gui.services.planar_layout_mesh import compact_planar_face


@pytest.mark.parametrize(
    "aperture,area,hole",
    [
        (None, np.pi * 25, None),
        (RadialAperture(5), np.pi * 25, None),
        (RadialAperture(5, 2), np.pi * 21, (0, 0, 0)),
        (OffsetRadialAperture(5, 2, 3, -4), np.pi * 21, (3, -4, 0)),
        (RectangularAperture(-4, 6, -3, 5), 80, None),
        (EllipticalAperture(5, 3), np.pi * 15, None),
    ],
)
@pytest.mark.parametrize("geometry_kind", ["plane", "standard"])
def test_compact_plane_area_pose_and_holes(
    minimal_optic, aperture, area, hole, geometry_kind
):
    surface = minimal_optic.surfaces[-1]
    if geometry_kind == "standard":
        surface.geometry = StandardGeometry(surface.geometry.cs, np.inf)
    surface.geometry.cs.rx = 0.37
    surface.geometry.cs.y = -13
    surface.aperture = aperture
    view = Surface3D(surface, 5)
    actor = compact_planar_face(view)
    assert actor is not None
    mesh = actor.GetMapper().GetInput()
    assert mesh.GetNumberOfPoints() <= 512
    assert np.isfinite(vtk_to_numpy(mesh.GetPoints().GetData())).all()
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputData(mesh)
    mass = vtk.vtkMassProperties()
    mass.SetInputConnection(triangles.GetOutputPort())
    assert mass.GetSurfaceArea() == pytest.approx(area, rel=0.0002)
    original_matrix = view.get_surface().GetMatrix()
    np.testing.assert_allclose(
        [[actor.GetMatrix().GetElement(i, j) for j in range(4)] for i in range(4)],
        [[original_matrix.GetElement(i, j) for j in range(4)] for i in range(4)],
    )
    if hole:
        locator = vtk.vtkStaticCellLocator()
        locator.SetDataSet(mesh)
        locator.BuildLocator()
        assert locator.FindCell(hole) == -1


def test_curved_custom_and_composite_surfaces_keep_existing_mesh(minimal_optic):
    curved = minimal_optic.surfaces[1]
    assert compact_planar_face(Surface3D(curved, 5)) is None
    planar = minimal_optic.surfaces[-1]
    planar.aperture = RectangularAperture(-5, 5, -5, 5) - RadialAperture(1)
    assert compact_planar_face(Surface3D(planar, 5)) is None
    planar.aperture = None

    class CustomPlane(type(planar.geometry)):
        pass

    planar.geometry.__class__ = CustomPlane
    assert compact_planar_face(Surface3D(planar, 5)) is None


@pytest.mark.parametrize(
    "aperture",
    [
        EllipticalAperture(5, 3, offset_x=1),
        RadialAperture(float("nan")),
        RadialAperture(0),
        RadialAperture(3, 4),
        RectangularAperture(float("nan"), 5, -4, 4),
    ],
)
def test_invalid_or_offset_aperture_does_not_create_a_misleading_compact_face(
    minimal_optic, aperture
):
    surface = minimal_optic.surfaces[-1]
    surface.aperture = aperture
    assert compact_planar_face(Surface3D(surface, 5)) is None


def plane_sequence(count, *, mirror=False):
    optic = Optic()
    optic.set_aperture("EPD", 2)
    optic.fields.set_type("angle")
    optic.fields.add(y=0)
    optic.wavelengths.add(0.55, is_primary=True)
    for index in range(count):
        optic.surfaces.add(
            index=index,
            radius=np.inf,
            thickness=np.inf if index == 0 else 1,
            aperture=RectangularAperture(-5, 5, -4, 4),
            is_stop=index == 1,
            material="mirror" if mirror and index == count - 2 else "air",
        )
    return optic


def prepare(optic):
    return prepare_3d(
        OpticSnapshot.capture(optic), {}, lambda *args: None, threading.Event()
    )


def test_94_planar_surfaces_fit_transport_without_dense_overlay_grids():
    data = prepare(plane_sequence(94))
    faces = [mesh for mesh in data["meshes"] if mesh["role"] == "face_highlight"]
    assert len(faces) == 93
    assert all(len(mesh["points"]) == 4 for mesh in faces)
    assert len(encode_message({"event": "result", "data": data})) < 1_000_000


def test_zero_width_bundle_keeps_reference_highlights_finite_and_visible():
    optic = plane_sequence(4)
    optic.set_aperture("EPD", 0)
    for surface in optic.surfaces:
        surface.aperture = None
    before = OpticSnapshot.capture(optic).data
    faces = {
        mesh["surfaces"]: mesh
        for mesh in prepare(optic)["meshes"]
        if mesh["role"] == "face_highlight"
    }
    for identity in ((1,), (2,)):
        points = faces[identity]["points"]
        assert np.isfinite(points).all()
        np.testing.assert_allclose(np.ptp(points, axis=0), [0.2, 0.2, 0])
    assert OpticSnapshot.capture(optic).data == before


@pytest.mark.parametrize("with_normals", [False, True])
def test_standalone_surface_reuses_payload_and_retained_polydata(qapp, with_normals):
    from optiland_gui.layout_presenter import present_3d

    optic = plane_sequence(4, mirror=True)
    # The actual wire transport must retain shared arrays as well as direct calls.
    data = pickle.loads(pickle.dumps(prepare(optic), protocol=5))
    base = next(
        mesh
        for mesh in data["meshes"]
        if mesh["role"] == "surface" and mesh["surfaces"] == (2,)
    )
    overlay = next(
        mesh
        for mesh in data["meshes"]
        if mesh["role"] == "face_highlight" and mesh["surfaces"] == (2,)
    )
    assert base["points"] is overlay["points"]
    assert base["cells"] is overlay["cells"]
    if with_normals:
        normals = np.tile([0.0, 0.0, 1.0], (len(base["points"]), 1))
        base["normals"] = overlay["normals"] = normals
    viewer = SimpleNamespace(
        current_theme="dark",
        _initialized=True,
        _has_scene=False,
        clear_3d_highlights=lambda: None,
        install_3d_highlights=lambda *args: None,
        renderer=vtk.vtkRenderer(),
        vtkWidget=MagicMock(),
    )
    present_3d(viewer, data, {"document_id": "test"})
    matching = [
        actor
        for actor, mesh in viewer._scene_actor_specs
        if mesh is base or mesh is overlay
    ]
    assert len(matching) == 2
    assert matching[0] is not matching[1]
    assert matching[0].GetProperty() is not matching[1].GetProperty()
    assert matching[0].GetMapper().GetInput() is matching[1].GetMapper().GetInput()
    if with_normals:
        np.testing.assert_allclose(
            vtk_to_numpy(
                matching[0].GetMapper().GetInput().GetPointData().GetNormals()
            ),
            normals,
        )
