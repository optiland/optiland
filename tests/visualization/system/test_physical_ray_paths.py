"""Physical ray-polyline regressions independent of importers and Qt."""

from __future__ import annotations

import math
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
import vtk

import optiland.backend as be
from optiland.geometries import Plane
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.physical_apertures import RadialAperture
from optiland.propagation.homogeneous import HomogeneousPropagation
from optiland.rays import RealRays
from optiland.visualization.system.ray_path import (
    neutral_reference_mask,
    physical_ray_path,
)
from optiland.visualization.system.rays import Rays2D, Rays3D


def _folded_optic(before=True, after=True, second=False):
    optic = Optic()
    optic.surfaces.add(index=0, z=-5)
    if before:
        optic.surfaces.add(index=len(optic.surfaces), z=0)
    optic.surfaces.add(
        index=len(optic.surfaces), z=0, rx=-math.pi / 4, material="mirror"
    )
    if after:
        optic.surfaces.add(index=len(optic.surfaces), z=0, rx=-math.pi / 2)
    if second:
        if before:
            optic.surfaces.add(index=len(optic.surfaces), z=0, y=-5, rx=-math.pi / 2)
        optic.surfaces.add(
            index=len(optic.surfaces), z=0, y=-5, rx=math.pi / 4, material="mirror"
        )
        if after:
            optic.surfaces.add(index=len(optic.surfaces), z=0, y=-5)
        optic.surfaces.add(index=len(optic.surfaces), z=-5, y=-5)
    else:
        optic.surfaces.add(index=len(optic.surfaces), z=0, y=-5, rx=-math.pi / 2)
    return optic


def _trace(optic, ray_type=Rays2D):
    rays = RealRays(
        be.zeros(3),
        be.array([-1.0, 0.0, 1.0]),
        be.full(3, -5.0),
        be.zeros(3),
        be.zeros(3),
        be.ones(3),
        be.ones(3),
        be.full(3, 0.55),
    )
    rays.record_on_surface(optic.surfaces[0])
    for surface in optic.surfaces[1:]:
        rays = surface.trace(rays)
    plotter = ray_type(optic)
    plotter._process_traced_rays()
    return plotter


@pytest.mark.parametrize(
    "before,after", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("second", [False, True])
def test_physical_vertices_match_analytic_fold(set_test_backend, before, after, second):
    plotter = _trace(_folded_optic(before, after, second))
    paths = list(plotter._iter_physical_paths())
    for height, path in zip([-1, 0, 1], paths, strict=True):
        expected = [[0, height, -5], [0, height, -height]]
        if second:
            expected.extend([[0, -5 - height, -height], [0, -5 - height, -5]])
        else:
            expected.append([0, -5, -height])
        np.testing.assert_allclose(path, expected, rtol=0, atol=2e-12)


@pytest.mark.parametrize(
    "projection,axes", [("YZ", (2, 1)), ("XZ", (2, 0)), ("XY", (0, 1))]
)
def test_2d_renderer_uses_physical_vertices(set_test_backend, projection, axes):
    plotter = _trace(_folded_optic())
    paths = list(plotter._iter_physical_paths())
    fig, ax = plt.subplots()
    try:
        artists = plotter._plot_lines(ax, 0, (0, 0), projection=projection)
        for artist, path in zip(artists, paths, strict=True):
            np.testing.assert_allclose(artist.get_xdata(), path[:, axes[0]])
            np.testing.assert_allclose(artist.get_ydata(), path[:, axes[1]])
            assert len(path) == 3
    finally:
        plt.close(fig)


def test_3d_renderer_has_the_same_physical_segments(set_test_backend):
    plotter = _trace(_folded_optic(), Rays3D)
    paths = list(plotter._iter_physical_paths())
    renderer = vtk.vtkRenderer()
    plotter._plot_lines(renderer, 0, (0, 0))
    actors = renderer.GetActors()
    assert actors.GetNumberOfItems() == 6
    actors.InitTraversal()
    for path in paths:
        for first, last in zip(path[:-1], path[1:], strict=True):
            source = actors.GetNextActor().GetMapper().GetInputAlgorithm()
            np.testing.assert_allclose(source.GetPoint1(), first, rtol=0, atol=2e-12)
            np.testing.assert_allclose(source.GetPoint2(), last, rtol=0, atol=2e-12)


def test_display_preserves_trace_coordinates_intensity_and_optical_path(
    set_test_backend,
):
    plotter = _trace(_folded_optic())
    attributes = ("x", "y", "z", "i")
    original = [be.to_numpy(getattr(plotter, key)).copy() for key in attributes]
    paths = be.to_numpy(plotter.optic.surfaces.opd).copy()
    list(plotter._iter_physical_paths())
    for key, value in zip(attributes, original, strict=True):
        np.testing.assert_array_equal(be.to_numpy(getattr(plotter, key)), value)
    np.testing.assert_array_equal(be.to_numpy(plotter.optic.surfaces.opd), paths)
    # Preparing display paths must never rewrite phase-analysis input.


def test_matching_absorbing_ideal_medium_has_no_boundary_event(set_test_backend):
    optic = _folded_optic()
    for surface in optic.surfaces:
        surface.material_post = IdealMaterial(1.5, 1e-6)
    assert neutral_reference_mask(list(optic.surfaces)).tolist() == [
        False,
        True,
        False,
        True,
        False,
    ]
    for path in _trace(optic)._iter_physical_paths():
        assert len(path) == 3


def test_real_aperture_hit_is_preserved_and_hide_vignetted_still_works(
    set_test_backend,
):
    optic = _folded_optic()
    optic.surfaces[1].aperture = RadialAperture(0.5)
    plotter = _trace(optic)
    paths = list(plotter._iter_physical_paths())
    np.testing.assert_allclose(paths[0], [[0, -1, -5], [0, -1, 0]])
    np.testing.assert_allclose(paths[2], [[0, 1, -5], [0, 1, 0]])
    survivors = list(plotter._iter_physical_paths(hide_vignetted=True))
    assert len(survivors) == 1
    np.testing.assert_allclose(survivors[0][-1], [0, -5, 0], atol=2e-12)


@pytest.mark.parametrize(
    "event",
    [
        "stop",
        "aperture",
        "coating",
        "scatter",
        "phase",
        "curved",
        "custom_geometry",
        "custom_surface",
        "custom_material",
        "custom_propagation",
        "refraction",
        "absorption",
    ],
)
def test_nontrivial_events_cannot_be_classified_as_reference_planes(
    set_test_backend, event
):
    optic = _folded_optic()
    surface = optic.surfaces[1]
    assert neutral_reference_mask(list(optic.surfaces))[1]
    if event == "stop":
        surface.is_stop = True
    elif event == "aperture":
        surface.aperture = RadialAperture(be.inf, r_min=0.5)
    elif event == "coating":
        surface.interaction_model.coating = object()
    elif event == "scatter":
        surface.interaction_model.bsdf = object()
    elif event == "phase":
        surface.interaction_model = SimpleNamespace(interaction_type="phase")
    elif event == "curved":
        surface.geometry = SimpleNamespace(radius=50)
    elif event == "custom_geometry":

        class CustomPlane(Plane):
            pass

        surface.geometry = CustomPlane(surface.geometry.cs)
    elif event == "custom_surface":
        surfaces = list(optic.surfaces)
        surfaces[1] = SimpleNamespace(interaction_model=surface.interaction_model)
        assert not neutral_reference_mask(surfaces)[1]
        return
    elif event == "custom_material":

        class CustomMaterial(IdealMaterial):
            pass

        surface.material_post = CustomMaterial(1)
    elif event == "custom_propagation":

        class CustomPropagation(HomogeneousPropagation):
            pass

        surface.material_post.propagation_model = CustomPropagation(
            surface.material_post
        )
    elif event == "refraction":
        surface.material_post = IdealMaterial(1.5)
    else:
        surface.material_post = IdealMaterial(1.0, 1e-6)
    assert not neutral_reference_mask(list(optic.surfaces))[1]


def test_blocked_ray_stops_at_its_first_hit_and_is_never_revived():
    points = np.array([[0, 0, 0], [0, 0, 2], [0, 0, 1], [0, 0, 1e9]], dtype=float)
    intensity = np.array([1, 1, 0, 1])
    neutral = np.array([False, True, True, False])
    result = physical_ray_path(points, intensity, neutral)
    np.testing.assert_array_equal(result, points[[0, 2]])
    assert physical_ray_path(points, intensity, neutral, hide_vignetted=True).shape == (
        0,
        3,
    )


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -1])
def test_invalid_intensity_terminates_before_unknown_sample(invalid):
    points = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]], dtype=float)
    result = physical_ray_path(
        points, np.array([1, invalid, 1]), np.zeros(3, dtype=bool)
    )
    np.testing.assert_array_equal(result, points[:1])


def test_nonfinite_coordinate_gap_is_not_bridged():
    points = np.array([[0, 0, 0], [0, np.inf, 1], [0, 0, 2]], dtype=float)
    result = physical_ray_path(points, np.ones(3), np.array([False, True, False]))
    assert np.isnan(result[1]).all()
    np.testing.assert_array_equal(result[[0, 2]], points[[0, 2]])
    assert np.isinf(points[1, 1])


def test_collinearity_is_checked_in_3d_before_projection():
    points = np.array([[0, 0, 0], [0.1, 0, 1], [0, 0, 2]], dtype=float)
    result = physical_ray_path(points, np.ones(3), np.array([False, True, False]))
    np.testing.assert_array_equal(result, points)


def test_read_only_input_and_coincident_reference_points():
    points = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=float)
    points.setflags(write=False)
    result = physical_ray_path(points, np.ones(3), np.array([False, True, False]))
    np.testing.assert_array_equal(result, points[[0, 2]])


@pytest.mark.parametrize("mismatch", ["coordinates", "intensity", "surfaces"])
def test_inconsistent_recorded_path_dimensions_are_rejected(mismatch):
    points = np.zeros((3, 3))
    intensity = np.ones(3)
    neutral = np.zeros(3, dtype=bool)
    if mismatch == "coordinates":
        points = np.zeros((3, 2))
    elif mismatch == "intensity":
        intensity = np.ones(2)
    else:
        neutral = np.zeros(2, dtype=bool)
    with pytest.raises(ValueError, match="must have equal length"):
        physical_ray_path(points, intensity, neutral)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_collinearity_keeps_real_bends_at_extreme_coordinate_scales(scale):
    points = np.array([[0, 0, 0], [scale, 0, 0], [scale, scale, 0]])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = physical_ray_path(points, np.ones(3), np.array([False, True, False]))
    np.testing.assert_array_equal(result, points)
