"""Core 3D rendering preserves annular openings without GUI dependencies."""

from __future__ import annotations

import json

import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import optiland.backend as be
from optiland.optic import Optic
from optiland.physical_apertures import (
    OffsetRadialAperture,
    RadialAperture,
    RectangularAperture,
)
from optiland.visualization.system.lens import Lens3D
from optiland.visualization.system.surface import Surface3D


def _aperture(kind):
    # Construct backend-owned fixtures after set_test_backend has run.
    if kind == "none":
        return None
    if kind == "circular":
        return RadialAperture(r_max=4)
    if kind == "annular":
        return RadialAperture(r_max=4, r_min=1.5)
    if kind == "rectangular":
        return RectangularAperture(-4, 4, -3, 3)
    if kind == "offset":
        return OffsetRadialAperture(r_max=4, offset_x=1, offset_y=-1)
    raise ValueError(kind)


def _surface(aperture_kind, radius=np.inf, *, asymmetric=False):
    optic = Optic()
    optic.surfaces.add(index=0, z=-10)
    geometry = (
        {"surface_type": "biconic", "radius_x": 20, "radius_y": 30}
        if asymmetric
        else {"radius": radius}
    )
    optic.surfaces.add(index=1, z=0, aperture=_aperture(aperture_kind), **geometry)
    return optic, Surface3D(optic.surfaces[1], 4)


def _mesh(actor):
    actor.GetMapper().Update()
    mesh = actor.GetMapper().GetInput()
    points = vtk_to_numpy(mesh.GetPoints().GetData())
    assert len(points) > 0 and np.isfinite(points).all()
    referenced = []
    # VTK revolution emits triangle strips; clipped grids emit polygons.
    for cells in (mesh.GetPolys(), mesh.GetStrips()):
        if cells.GetNumberOfCells() == 0:
            continue
        connectivity = vtk_to_numpy(cells.GetConnectivityArray())
        offsets = vtk_to_numpy(cells.GetOffsetsArray())
        assert offsets[0] == 0 and offsets[-1] == len(connectivity)
        assert np.all(np.diff(offsets) >= 3)
        assert connectivity.min() >= 0 and connectivity.max() < len(points)
        referenced.append(connectivity)
    assert referenced, "The actor must contain optical-face or body cells."
    return mesh, points[np.unique(np.concatenate(referenced))]


def _assert_annular_mesh(actor, surface):
    mesh, used_points = _mesh(actor)
    aperture = surface.surf.aperture
    radius = np.hypot(used_points[:, 0], used_points[:, 1])
    assert radius.min() >= aperture.r_min - 1e-6
    assert radius.max() <= aperture.r_max + 1e-6
    # Boundary cells stay inside the aperture; allow one grid diagonal for the
    # clipped mesh's sampling error rather than requiring an analytic circle.
    tolerance = 2 * aperture.r_max * np.sqrt(2) / 255
    assert radius.min() <= aperture.r_min + tolerance
    assert radius.max() >= aperture.r_max - tolerance
    expected_sag = surface.surf.geometry.sag(
        be.array(used_points[:, 0]), be.array(used_points[:, 1])
    )
    np.testing.assert_allclose(used_points[:, 2], be.to_numpy(expected_sag), atol=1e-6)

    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()

    def intersects(x):
        t = vtk.mutable(0.0)
        cell_id = vtk.mutable(0)
        sub_id = vtk.mutable(0)
        return locator.IntersectWithLine(
            (x, 0, -1), (x, 0, 1), 1e-8, t, [0.0] * 3, [0.0] * 3, sub_id, cell_id
        )

    assert not intersects(0), "An axial probe must pass through the opening."
    assert intersects(2.5), "An annular probe must hit the optical face."


@pytest.mark.parametrize(
    "aperture_kind,asymmetric,expected",
    [
        ("none", False, True),
        ("circular", False, True),
        ("annular", False, False),
        ("rectangular", False, False),
        ("offset", False, False),
        ("circular", True, False),
        ("none", True, False),
    ],
)
def test_revolution_requires_symmetric_geometry_and_an_uninterrupted_circle(
    set_test_backend, aperture_kind, asymmetric, expected
):
    _, surface = _surface(aperture_kind, asymmetric=asymmetric)
    assert bool(surface.supports_revolution) is expected
    # Verify dispatch using the real mapper output, not mocked path calls.
    mesh, _ = _mesh(surface.get_surface())
    assert mesh.GetNumberOfCells() > 0


@pytest.mark.parametrize("radius", [np.inf, 20], ids=["flat", "curved"])
def test_annular_surface_mesh_is_finite_and_keeps_the_opening(set_test_backend, radius):
    optic, surface = _surface("annular", radius)
    before = json.dumps(optic.to_dict(), sort_keys=True)
    _assert_annular_mesh(surface.get_surface(), surface)
    assert json.dumps(optic.to_dict(), sort_keys=True) == before


@pytest.mark.parametrize("radius", [np.inf, 20], ids=["flat", "curved"])
@pytest.mark.parametrize("annular_end", [0, 1], ids=["entrance", "exit"])
def test_lens_plot_preserves_an_annular_face_at_either_end(
    set_test_backend, radius, annular_end
):
    optic, entrance = _surface("annular" if annular_end == 0 else "circular", radius)
    optic.surfaces.add(
        index=2,
        z=10,
        radius=-radius,
        aperture=_aperture("annular" if annular_end == 1 else "circular"),
    )
    faces = [entrance, Surface3D(optic.surfaces[2], 4)]
    lens = Lens3D(faces)
    before = json.dumps(optic.to_dict(), sort_keys=True)
    assert not lens.is_symmetric
    renderer = vtk.vtkRenderer()
    lens.plot(renderer)
    actors = list(renderer.GetActors())
    assert len(actors) == 3  # Two optical faces and their outer body wall.
    for actor in actors:
        _mesh(actor)
    _assert_annular_mesh(actors[annular_end], faces[annular_end])
    assert json.dumps(optic.to_dict(), sort_keys=True) == before


@pytest.mark.parametrize("radius", [np.inf, 20], ids=["flat", "curved"])
def test_full_circular_lens_still_renders_as_a_revolved_body(set_test_backend, radius):
    optic, entrance = _surface("circular", radius)
    optic.surfaces.add(index=2, z=10, radius=-radius, aperture=_aperture("circular"))
    lens = Lens3D([entrance, Surface3D(optic.surfaces[2], 4)])
    assert lens.is_symmetric
    renderer = vtk.vtkRenderer()
    lens.plot(renderer)
    actors = list(renderer.GetActors())
    assert len(actors) == 1
    mesh, _ = _mesh(actors[0])
    assert mesh.GetNumberOfCells() > 0
