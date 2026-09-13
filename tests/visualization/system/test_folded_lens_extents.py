"""Local diagonal radii must not enlarge other prism/lens faces."""

from __future__ import annotations

import json
from unittest.mock import Mock

import numpy as np
import pytest
import vtk
from matplotlib.figure import Figure

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.optic import Optic
from optiland.physical_apertures import RadialAperture, RectangularAperture
from optiland.visualization.system.lens import Lens2D, Lens3D
from optiland.visualization.system.surface import Surface2D, Surface3D


def _folded_faces(view_type=Surface2D):
    optic = Optic()
    optic.surfaces.add(index=0, z=-10)
    optic.surfaces.add(index=1, z=0, aperture=RectangularAperture(-5, 5, -5, 5))
    optic.surfaces.add(
        index=2,
        z=5,
        rx=np.pi / 4,
        material="mirror",
        aperture=RectangularAperture(-5, 5, -5 * np.sqrt(2), 5 * np.sqrt(2)),
    )
    return optic, [view_type(surface, 1) for surface in list(optic.surfaces)[1:]]


def test_folded_outline_stays_inside_analytic_prism(set_test_backend):
    optic, faces = _folded_faces()
    before = json.dumps(optic.to_dict(), sort_keys=True)
    component = Lens2D(faces)
    figure = Figure()
    axes = figure.add_subplot()
    component.plot(axes)
    assert axes.patches
    for patch in axes.patches:
        vertices = patch.get_xy()
        assert np.isfinite(vertices).all()
        np.testing.assert_allclose(vertices.min(axis=0), [0, -5], atol=1e-10)
        np.testing.assert_allclose(vertices.max(axis=0), [10, 5], atol=1e-10)
    local = component._compute_sag(apply_transform=False)
    np.testing.assert_allclose(be.to_numpy(local[0][1])[[0, -1]], [-5, 5])
    np.testing.assert_allclose(
        be.to_numpy(local[1][1])[[0, -1]], [-5 * np.sqrt(2), 5 * np.sqrt(2)]
    )
    assert json.dumps(optic.to_dict(), sort_keys=True) == before


@pytest.mark.parametrize("decenter,reverse", [(0, False), (0, True), (2, False)])
def test_shared_rim_requires_coaxial_faces(set_test_backend, decenter, reverse):
    optic = Optic()
    optic.surfaces.add(index=0, z=-10)
    optic.surfaces.add(index=1, z=0, aperture=RadialAperture(3))
    optic.surfaces.add(
        index=2,
        z=5,
        y=decenter,
        rx=np.pi if reverse else 0,
        aperture=RadialAperture(5),
    )
    # A common rigid parent transform must not change the coaxial decision.
    parent = CoordinateSystem(x=7, y=11, z=-13, rx=0.4, ry=0.2)
    for surface in optic.surfaces:
        surface.geometry.cs.reference_cs = parent
    component = Lens2D([Surface2D(s, 1) for s in list(optic.surfaces)[1:]])
    local = component._compute_sag(apply_transform=False)
    expected = 3 if decenter else 5
    np.testing.assert_allclose(be.to_numpy(local[0][1])[[0, -1]], [-expected, expected])


def test_3d_fold_does_not_add_a_circular_annulus(set_test_backend):
    _, faces = _folded_faces(Surface3D)
    component = Lens3D(faces)
    component._plot_annulus = Mock(side_effect=AssertionError("False prism rim"))
    renderer = vtk.vtkRenderer()
    component._plot_surfaces(renderer)
    assert renderer.GetActors().GetNumberOfItems() == 2
    component._plot_annulus.assert_not_called()


def test_3d_coaxial_rim_still_extends(set_test_backend):
    optic = Optic()
    optic.surfaces.add(index=0, z=-10)
    optic.surfaces.add(index=1, z=0, aperture=RadialAperture(3))
    optic.surfaces.add(index=2, z=5, aperture=RadialAperture(5))
    component = Lens3D([Surface3D(s, 1) for s in list(optic.surfaces)[1:]])
    component._plot_annulus = Mock()
    component._plot_surfaces(vtk.vtkRenderer())
    component._plot_annulus.assert_called_once()


def test_single_face_keeps_its_own_extent(set_test_backend):
    optic = Optic()
    optic.surfaces.add(index=0, z=-10)
    optic.surfaces.add(index=1, z=5, aperture=RadialAperture(3))
    face = Surface2D(optic.surfaces[1], 1)
    component = Lens2D([face])
    x, y, z = component._compute_sag()[0]
    np.testing.assert_allclose(be.to_numpy(x), 0)
    np.testing.assert_allclose(be.to_numpy(y)[[0, -1]], [-3, 3])
    np.testing.assert_allclose(be.to_numpy(z), 5)
