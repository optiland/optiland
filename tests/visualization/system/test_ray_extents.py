"""Incident-ray extent and schematic image-plane display regressions."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

import optiland.backend as be
from optiland.optic import Optic
from optiland.physical_apertures import (
    OffsetRadialAperture,
    RadialAperture,
    RectangularAperture,
)
from optiland.visualization.info.providers import SurfaceInfoProvider
from optiland.visualization.system.optic_viewer import OpticViewer
from optiland.visualization.system.rays import Rays2D, Rays3D
from optiland.visualization.system.surface import Surface2D, Surface3D
from optiland.visualization.system.system import OpticalSystem


def _optic():
    optic = Optic()
    for index in range(4):
        optic.surfaces.add(index=index, z=float(index), is_stop=index == 1)
    optic.set_aperture("EPD", 1)
    optic.fields.set_type("angle")
    optic.fields.add(0, 0)
    optic.wavelengths.add(0.55)
    return optic


def _recorded_rays(ray_type=Rays2D):
    rays = ray_type(_optic())
    rays.x = be.zeros((4, 3))
    rays.y = be.array([[0, 1, 2], [0, 2, 4], [0, 1e6, 6], [0.1, 1e9, 1e8]])
    rays.z = be.array([[0] * 3, [1] * 3, [2] * 3, [3] * 3])
    rays.i = be.array([[1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 0, 0]])
    return rays


def test_extents_include_first_blocking_hit_but_exclude_later_coordinates(
    set_test_backend,
):
    rays = _recorded_rays()
    rays._update_surface_extents()
    np.testing.assert_allclose(be.to_numpy(rays.r_extent), [2, 4, 6, 0.1])


def test_invalid_or_extinguished_samples_cannot_regain_liveness(set_test_backend):
    rays = _recorded_rays()
    rays.i = be.array([[1, 1, 1], [1, be.nan, 1], [1, 0, 0], [1, 1, 1]])
    rays._update_surface_extents()
    np.testing.assert_allclose(be.to_numpy(rays.r_extent), [2, 4, 6, 0.1])


@pytest.mark.parametrize("case", ["empty", "nan", "infinity"])
def test_empty_and_nonfinite_bundles_have_zero_sample_extent(set_test_backend, case):
    rays = _recorded_rays()
    if case == "empty":
        rays.x = rays.y = rays.z = rays.i = be.zeros((4, 0))
    else:
        rays.y = be.full((4, 3), be.nan if case == "nan" else be.inf)
    with np.errstate(all="raise"):
        rays._update_surface_extents()
    np.testing.assert_array_equal(be.to_numpy(rays.r_extent), np.zeros(4))


@pytest.mark.parametrize("ray_type", [Rays2D, Rays3D])
def test_display_mask_does_not_modify_trace_arrays(
    set_test_backend, ray_type, monkeypatch
):
    rays = _recorded_rays(ray_type)
    originals = [be.to_numpy(value).copy() for value in (rays.x, rays.y, rays.z)]
    plotted = []

    def capture_line(ax, x, y, z, *args, **kwargs):
        plotted.append((x.copy(), y.copy(), z.copy()))
        return object(), SimpleNamespace()

    monkeypatch.setattr(rays, "_plot_single_line", capture_line)
    rays._plot_lines(None, 0, (0, 0))
    # The shared physical-path renderer truncates at the first blocking hit.
    # Assert the exact retained segment rather than an empty downstream slice.
    np.testing.assert_array_equal(np.column_stack(plotted[1]), [[0, 1, 0], [0, 2, 1]])
    for original, current in zip(originals, (rays.x, rays.y, rays.z), strict=True):
        np.testing.assert_array_equal(be.to_numpy(current), original)


@pytest.mark.parametrize("ray_type", [Rays2D, Rays3D])
def test_new_plot_resets_old_extents(set_test_backend, ray_type, monkeypatch):
    rays = _recorded_rays(ray_type)
    monkeypatch.setattr(rays, "_trace", lambda *args: rays._update_surface_extents())
    rays.plot(None, distribution=None)
    assert float(be.to_numpy(rays.r_extent[2])) == 6
    rays.y = be.full((4, 3), 0.25)
    rays.plot(None, distribution=None)
    np.testing.assert_allclose(be.to_numpy(rays.r_extent), [0.25] * 4)


@pytest.mark.parametrize("component_type", [Surface2D, Surface3D])
def test_unbounded_image_uses_nearest_finite_aperture_only_as_display_scale(
    set_test_backend, component_type
):
    optic = _optic()
    optic.surfaces[1].aperture = RadialAperture(9)
    optic.surfaces[2].aperture = RadialAperture(2.5)
    for sampled_extent in (0, 0.1, 1e6, be.nan):
        component = component_type(
            optic.surfaces[-1], sampled_extent, is_image_plane=True
        )
        assert component.extent == 2.5
        assert component.extent_source == "schematic"
        assert component.surf is optic.surfaces[-1]
    assert optic.surfaces[-1].aperture is None
    assert optic.surfaces[-1].semi_aperture is None


def test_image_without_finite_apertures_has_documented_fallback(set_test_backend):
    optic = _optic()
    optic.surfaces[2].aperture = RadialAperture(be.inf, r_min=0.25)
    component = Surface2D(optic.surfaces[-1], 1e9, is_image_plane=True)
    assert component.extent == 1.0
    assert component.extent_source == "schematic"
    info = SurfaceInfoProvider(optic.surfaces).get_info(component)
    assert "schematic; physical size unspecified" in info


@pytest.mark.parametrize("aperture_kind", ["radial", "offset", "rectangular"])
def test_explicit_image_bounds_override_samples_and_schematic_policy(
    set_test_backend, aperture_kind
):
    optic = _optic()
    image = optic.surfaces[-1]
    optic.surfaces[2].aperture = RadialAperture(20)
    if aperture_kind == "radial":
        image.aperture = RadialAperture(3)
        expected = 3
    elif aperture_kind == "offset":
        image.aperture = OffsetRadialAperture(2, offset_y=-5)
        expected = 7
    else:
        image.aperture = RectangularAperture(-2, 2, -4, 4)
        expected = 4
    component = Surface2D(image, 1e9, is_image_plane=True)
    assert component.extent == expected
    assert component.extent_source == "physical_aperture"


def test_computed_semi_aperture_does_not_define_physical_image_size(set_test_backend):
    optic = _optic()
    # update_paraxial() stores sampled marginal/chief heights in this field.
    optic.surfaces[-1].semi_aperture = 1e9
    optic.surfaces[-2].semi_aperture = 1e8
    component = Surface2D(optic.surfaces[-1], 1e6, is_image_plane=True)
    assert component.extent == 1.0
    assert component.extent_source == "schematic"


@pytest.mark.parametrize("num_rays", [3, 5, 8, 14])
def test_image_marker_is_stable_and_included_in_default_limits(
    set_test_backend, num_rays
):
    optic = _optic()
    optic.surfaces[2].aperture = RadialAperture(3)
    viewer = OpticViewer(optic)
    fig, ax, _ = viewer.view(num_rays=num_rays, show=False)
    try:
        image = next(
            component
            for component in viewer.system.components
            if isinstance(component, Surface2D) and component.surf is optic.surfaces[-1]
        )
        assert image.extent == 3
        line = next(
            line for line in ax.lines if "schematic image plane" in line.get_label()
        )
        np.testing.assert_allclose(
            [line.get_ydata().min(), line.get_ydata().max()], [-3, 3]
        )
        assert ax.get_ylim()[0] < -3 and ax.get_ylim()[1] > 3
        np.testing.assert_allclose(line.get_xdata(), 3)
        assert optic.surfaces[-1].aperture is None
    finally:
        plt.close(fig)


def test_dead_image_coordinates_do_not_expand_default_limits(set_test_backend):
    rays = _recorded_rays()
    rays._update_surface_extents()
    viewer = OpticViewer(rays.optic)
    viewer.rays = rays
    viewer.system = OpticalSystem(rays.optic, rays)
    viewer.system._identify_components()
    _, limits = viewer._default_axis_limits("YZ")
    assert -10 < limits[0] < 0 < limits[1] < 10


def test_three_dimensional_image_actor_uses_schematic_extent(set_test_backend):
    optic = _optic()
    optic.surfaces[2].aperture = RadialAperture(3)
    rays = Rays3D(optic)
    rays.r_extent = be.full(4, 1e6)
    system = OpticalSystem(optic, rays, projection="3d")
    system._identify_components()
    image = next(
        component
        for component in system.components
        if component.surf is optic.surfaces[-1]
    )
    assert isinstance(image, Surface3D)
    assert image.extent_source == "schematic"
    bounds = image.get_surface().GetBounds()
    np.testing.assert_allclose(bounds[4:], [3, 3])
    assert max(abs(value) for value in bounds[:4]) == pytest.approx(3)


def test_custom_surface_renderer_keeps_existing_positional_data(
    set_test_backend, monkeypatch
):
    from optiland.visualization.system import system as system_module

    optic = _optic()
    rays = Rays2D(optic)
    calls = []
    custom = SimpleNamespace(render_2d=lambda ax, data: calls.append(data))
    monkeypatch.setitem(system_module._CUSTOM_RENDERER_REGISTRY, "surface", custom)
    fig, ax = plt.subplots()
    try:
        OpticalSystem(optic, rays).plot(ax, show_apertures=False)
        assert calls[-1]["args"][0] is optic.surfaces[-1]
        assert calls[-1]["args"][1] == 0
        assert calls[-1]["kwargs"] == {"is_image_plane": True}
    finally:
        plt.close(fig)


@pytest.mark.parametrize("projection", ["YZ", "XZ"])
def test_first_blocking_hit_cannot_override_physical_aperture_fit(
    set_test_backend, projection
):
    rays = _recorded_rays()
    for surface in rays.optic.surfaces:
        surface.aperture = RadialAperture(1)
    rays.y = be.array([[0, 1, 1], [0, 1e6, 1], [0, 1e9, 1], [0, 1e9, 1]])
    rays._update_surface_extents()
    assert float(be.to_numpy(rays.r_extent[1])) == 1e6
    viewer = OpticViewer(rays.optic)
    viewer.rays = rays
    viewer.system = OpticalSystem(rays.optic, rays)
    viewer.system._identify_components()
    _, limits = viewer._default_axis_limits(projection)
    np.testing.assert_allclose(limits, [-1.15, 1.15])


@pytest.mark.parametrize("projection,tilt", [("YZ", "rx"), ("XZ", "ry")])
def test_tilted_image_marker_is_not_cropped_at_vertex_z_limits(
    set_test_backend, projection, tilt
):
    optic = _optic()
    optic.surfaces[-2].aperture = RadialAperture(8)
    setattr(optic.surfaces[-1].geometry.cs, tilt, np.pi / 3)
    viewer = OpticViewer(optic)
    fig, ax, _ = viewer.view(num_rays=3, projection=projection, show=False)
    try:
        line = next(
            line for line in ax.lines if "schematic image plane" in line.get_label()
        )
        assert np.ptp(line.get_xdata()) > 10
        assert ax.get_xlim()[0] < np.min(line.get_xdata())
        assert ax.get_xlim()[1] > np.max(line.get_xdata())
    finally:
        plt.close(fig)


def test_nonimage_invalid_sample_extent_is_zero(set_test_backend):
    component = Surface2D(_optic().surfaces[1], be.nan)
    assert component.extent == 0
    assert component.extent_source == "ray_samples"
