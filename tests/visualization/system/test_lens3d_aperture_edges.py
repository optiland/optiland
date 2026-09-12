"""3D body walls must follow their faces, including rectangular folded optics."""

from __future__ import annotations

import json

import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.optic import Optic
from optiland.physical_apertures import (
    EllipticalAperture,
    OffsetRadialAperture,
    RadialAperture,
    RectangularAperture,
)
from optiland.visualization.system.lens import Lens3D
from optiland.visualization.system.surface import Surface3D


def _lens(apertures, **last_position):
    optic = Optic()
    optic.surfaces.add(index=0, z=-10)
    optic.surfaces.add(index=1, z=0, aperture=apertures[0])
    optic.surfaces.add(index=2, **{"z": 10, **last_position}, aperture=apertures[1])
    return optic, Lens3D([Surface3D(s, 1) for s in list(optic.surfaces)[1:]])


def _world_points(actor):
    actor.GetMapper().Update()
    points = vtk_to_numpy(actor.GetMapper().GetInput().GetPoints().GetData())
    matrix = actor.GetMatrix()
    matrix = np.array([[matrix.GetElement(i, j) for j in range(4)] for i in range(4)])
    return np.column_stack((points, np.ones(len(points)))) @ matrix.T


@pytest.mark.parametrize("folded", [False, True])
def test_rectangular_complete_body_has_no_circular_shell(set_test_backend, folded):
    optic, lens = _lens(
        [
            RectangularAperture(-5, 5, -5, 5),
            RectangularAperture(-5, 5, -5 * np.sqrt(2), 5 * np.sqrt(2))
            if folded
            else RectangularAperture(-5, 5, -5, 5),
        ],
        z=5 if folded else 10,
        rx=np.pi / 4 if folded else 0,
    )
    before = json.dumps(optic.to_dict(), sort_keys=True)
    renderer = vtk.vtkRenderer()
    lens.plot(renderer)
    points = np.vstack([_world_points(a)[:, :3] for a in renderer.GetActors()])
    assert np.isfinite(points).all()
    np.testing.assert_allclose(points.min(axis=0), [-5, -5, 0], atol=1e-6)
    np.testing.assert_allclose(points.max(axis=0), [5, 5, 10], atol=1e-6)
    # The wall contains the rectangular corners, not just an inscribed circle.
    for x in (-5, 5):
        for y in (-5, 5):
            assert np.any(np.all(np.isclose(points, [x, y, 0], atol=1e-6), axis=1))
    assert json.dumps(optic.to_dict(), sort_keys=True) == before


def test_rectangular_walls_follow_offsets_and_reversed_frames(set_test_backend):
    optic, lens = _lens(
        [RectangularAperture(1, 5, -3, 3), RectangularAperture(1, 5, -3, 3)],
        rx=np.pi,
    )
    parent = CoordinateSystem(x=7, y=-2, z=13, rx=0.3, ry=0.4)
    for s in list(optic.surfaces)[1:]:
        s.geometry.cs.reference_cs = parent
    renderer = vtk.vtkRenderer()
    lens._plot_surface_edges(renderer)
    assert renderer.GetActors().GetNumberOfItems() == 1
    points = _world_points(next(iter(renderer.GetActors())))[:, :3]
    origin, rotation = parent.get_effective_transform()
    local = (points - be.to_numpy(origin)) @ be.to_numpy(rotation)
    first, second = np.split(local, 2)
    # Every connecting segment is axial; reversed local normals must not twist it.
    np.testing.assert_allclose(first[:, :2], second[:, :2], atol=2e-6)
    np.testing.assert_allclose(second[:, 2] - first[:, 2], 10, atol=2e-6)


@pytest.mark.parametrize("decenter", [0, 2])
def test_circular_rim_is_shared_only_on_common_axis(set_test_backend, decenter):
    _, lens = _lens([RadialAperture(3), RadialAperture(5)], x=decenter)
    x, y, _ = lens._get_edge_points(lens.surfaces[0])
    np.testing.assert_allclose(
        np.hypot(be.to_numpy(x), be.to_numpy(y)), 3 if decenter else 5
    )


@pytest.mark.parametrize(
    "aperture",
    [
        RectangularAperture(1, 5, -3, 3),
        EllipticalAperture(4, 2, offset_x=1, offset_y=-1),
        OffsetRadialAperture(3, offset_x=2, offset_y=-1),
    ],
)
def test_local_side_wall_boundary_matches_aperture(set_test_backend, aperture):
    _, lens = _lens([aperture, aperture])
    x, y, z = (be.to_numpy(a) for a in lens._get_edge_points(lens.surfaces[0]))
    assert len(x) == 256
    assert np.isfinite(np.column_stack((x, y, z))).all()
    assert not np.allclose([x[0], y[0]], [x[-1], y[-1]])
    if isinstance(aperture, RectangularAperture):
        assert np.all(
            np.isclose(x, 1) | np.isclose(x, 5) | np.isclose(y, -3) | np.isclose(y, 3)
        )
        np.testing.assert_allclose(
            [x.min(), x.max(), y.min(), y.max()], aperture.extent
        )
    elif isinstance(aperture, EllipticalAperture):
        np.testing.assert_allclose(((x - 1) / 4) ** 2 + ((y + 1) / 2) ** 2, 1)
    else:
        np.testing.assert_allclose((x - 2) ** 2 + (y + 1) ** 2, 9)


def test_rectangular_faces_do_not_acquire_an_annulus(set_test_backend):
    _, lens = _lens(
        [RectangularAperture(-3, 3, -3, 3), RectangularAperture(-5, 5, -5, 5)]
    )
    renderer = vtk.vtkRenderer()
    lens._plot_surfaces(renderer)
    assert renderer.GetActors().GetNumberOfItems() == 2


@pytest.mark.parametrize("offset", [(5, -3), (-5, 3)])
def test_offset_elliptical_faces_meet_body_walls(set_test_backend, offset):
    aperture = EllipticalAperture(4, 2, offset_x=offset[0], offset_y=offset[1])
    optic, lens = _lens([aperture, aperture])
    before = json.dumps(optic.to_dict(), sort_keys=True)
    renderer = vtk.vtkRenderer()
    lens.plot(renderer)
    actors = list(renderer.GetActors())
    assert len(actors) == 3
    expected = np.array([offset[0] - 4, offset[1] - 2, offset[0] + 4, offset[1] + 2])
    for actor in actors[:2]:
        mesh = actor.GetMapper().GetInput()
        # Bounds of unreferenced grid points can hide missing face geometry.
        used = np.unique(vtk_to_numpy(mesh.GetPolys().GetConnectivityArray()))
        assert used.size > 0
        points = _world_points(actor)[used, :2]
        actual = np.concatenate((points.min(axis=0), points.max(axis=0)))
        np.testing.assert_allclose(actual, expected, atol=0.07)
    wall_points = _world_points(actors[2])[:, :2]
    np.testing.assert_allclose(
        np.concatenate((wall_points.min(axis=0), wall_points.max(axis=0))),
        expected,
        atol=1e-6,
    )
    assert json.dumps(optic.to_dict(), sort_keys=True) == before
