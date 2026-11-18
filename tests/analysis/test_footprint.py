from __future__ import annotations

from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.contour import ContourSet
from matplotlib.figure import Figure

import optiland.backend as be
from optiland.analysis import FootprintDiagram, SpotDiagram
from optiland.materials import Material
from optiland.optic import Optic
from optiland.physical_apertures import RadialAperture
from optiland.samples.objectives import CookeTriplet, DoubleGauss
from tests.utils import assert_allclose

matplotlib.use("Agg")

EPD = 10.0


def singlet(aperture=None, dy=0.0):
    """N-BK7 singlet at infinite conjugate with the stop on its first surface.

    ``aperture`` and ``dy`` are applied to the second lens surface.
    """
    lens = Optic()
    lens.surfaces.add(index=0, thickness=be.inf)
    lens.surfaces.add(
        index=1,
        radius=50.0,
        thickness=5.0,
        material=Material("N-BK7", catalog="schott"),
        is_stop=True,
    )
    lens.surfaces.add(index=2, radius=-50.0, thickness=45.0, aperture=aperture, dy=dy)
    lens.surfaces.add(index=3)
    lens.set_aperture(aperture_type="EPD", value=EPD)
    lens.fields.set_type("angle")
    lens.fields.add(y=0.0)
    lens.fields.add(y=5.0)
    lens.wavelengths.add(value=0.55, is_primary=True)
    return lens


def numpy_mask(points, name):
    return be.to_numpy(getattr(points, name)).astype(bool)


def test_image_surface_matches_spot_diagram(set_test_backend):
    lens = CookeTriplet()
    footprint = FootprintDiagram(lens, -1, wavelengths="all", num_rays=6)
    spot = SpotDiagram(lens, wavelengths="all", num_rings=6, coordinates="local")

    assert footprint.surface_index == len(lens.surfaces) - 1
    for field_fp, field_spot in zip(footprint.data, spot.data, strict=True):
        for points, spot_points in zip(field_fp, field_spot, strict=True):
            passed = numpy_mask(points, "reached") & ~numpy_mask(points, "clipped")
            np.testing.assert_array_equal(
                be.to_numpy(points.x)[passed], be.to_numpy(spot_points.x)
            )
            np.testing.assert_array_equal(
                be.to_numpy(points.y)[passed], be.to_numpy(spot_points.y)
            )


def test_stop_max_radius_matches_paraxial_marginal_ray(set_test_backend):
    lens = DoubleGauss()
    stop = lens.surfaces.stop_index
    footprint = FootprintDiagram(lens, stop, fields=[(0.0, 0.0)])

    y_marginal, _ = lens.paraxial.marginal_ray()
    expected = abs(float(be.to_numpy(y_marginal[stop]).item()))
    assert footprint.max_radius() == pytest.approx(expected, rel=0.01)


def test_data_layout(set_test_backend):
    lens = CookeTriplet()
    footprint = FootprintDiagram(
        lens, 2, fields=[(0.0, 0.0), (0.0, 1.0)], wavelengths="all", num_rays=4
    )
    assert len(footprint.data) == 2
    assert all(len(field_data) == 3 for field_data in footprint.data)
    for field_data in footprint.data:
        for points in field_data:
            n = be.size(points.x)
            assert be.size(points.y) == be.size(points.reached) == n
            assert be.size(points.clipped) == n


def test_default_wavelength_is_primary(set_test_backend):
    footprint = FootprintDiagram(CookeTriplet(), 1)
    assert len(footprint.wavelengths) == 1
    assert footprint.wavelengths[0].value == CookeTriplet().primary_wavelength


def test_on_axis_extent_is_symmetric(set_test_backend):
    lens = singlet()
    footprint = FootprintDiagram(lens, 1, fields=[(0.0, 0.0)])
    x_min, x_max, y_min, y_max = footprint.extent()

    assert_allclose(x_max, EPD / 2)
    assert_allclose(y_max, EPD / 2)
    assert_allclose(x_min, -x_max)
    assert_allclose(y_min, -y_max)
    assert_allclose(footprint.max_radius(), EPD / 2)


def test_local_coordinates_on_decentered_surface(set_test_backend):
    dy = 0.3
    lens = singlet(dy=dy)
    footprint = FootprintDiagram(lens, 2, fields=[(0.0, 1.0)])

    lens.trace(0.0, 1.0, lens.primary_wavelength, 12, "hexapolar")
    surface = lens.surfaces[2]
    points = footprint.data[0][0]
    assert_allclose(points.x, surface.x)
    assert_allclose(points.y, surface.y - dy)


def test_physical_aperture_clips_rays(set_test_backend):
    r_max = 3.0
    lens = singlet(aperture=RadialAperture(r_max=r_max))
    footprint = FootprintDiagram(lens, 2)

    any_clipped = False
    for field_data in footprint.data:
        for points in field_data:
            r = np.hypot(be.to_numpy(points.x), be.to_numpy(points.y))
            reached = numpy_mask(points, "reached")
            clipped = numpy_mask(points, "clipped")
            assert reached.all()
            assert (r[clipped] > r_max).all()
            assert (r[reached & ~clipped] <= r_max).all()
            any_clipped = any_clipped or clipped.any()
    assert any_clipped
    assert footprint.max_radius() > r_max

    fig, ax = footprint.view(show=False)
    assert any(isinstance(artist, ContourSet) for artist in ax.collections)
    assert "Clipped" in ax.get_legend_handles_labels()[1]
    plt.close(fig)


def test_rays_clipped_upstream_do_not_reach(set_test_backend):
    lens = singlet(aperture=RadialAperture(r_max=3.0))
    upstream = FootprintDiagram(lens, 2)
    downstream = FootprintDiagram(lens, 3)

    for field_up, field_down in zip(upstream.data, downstream.data, strict=True):
        for up, down in zip(field_up, field_down, strict=True):
            passed_up = numpy_mask(up, "reached") & ~numpy_mask(up, "clipped")
            np.testing.assert_array_equal(numpy_mask(down, "reached"), passed_up)
            assert not numpy_mask(down, "clipped").any()


def test_no_ray_reaches_surface(set_test_backend):
    lens = singlet(aperture=RadialAperture(r_max=1e-3, r_min=1e-4))
    footprint = FootprintDiagram(lens, 3)

    with pytest.raises(ValueError, match="No ray reaches surface 3"):
        footprint.extent()
    with pytest.raises(ValueError, match="No ray reaches surface 3"):
        footprint.max_radius()
    fig, ax = footprint.view(show=False)
    assert ax.get_legend() is None
    plt.close(fig)


def test_rays_missing_the_surface_are_left_out(set_test_backend):
    """A 6 mm sphere in a 10 mm beam: the outer rays never intersect it."""
    lens = Optic()
    lens.surfaces.add(index=0, thickness=be.inf)
    lens.surfaces.add(
        index=1,
        radius=100.0,
        thickness=5.0,
        material=Material("N-BK7", catalog="schott"),
        is_stop=True,
    )
    lens.surfaces.add(index=2, radius=-6.0, thickness=20.0)
    lens.surfaces.add(index=3)
    lens.set_aperture(aperture_type="EPD", value=20.0)
    lens.fields.set_type("angle")
    lens.fields.add(y=0.0)
    lens.wavelengths.add(value=0.55, is_primary=True)
    footprint = FootprintDiagram(lens, 2)

    points = footprint.data[0][0]
    missed = np.isnan(be.to_numpy(points.x))
    assert missed.any()
    assert numpy_mask(points, "clipped")[missed].all()
    assert np.isfinite(footprint.extent()).all()
    assert footprint.max_radius() <= 6.0


def test_semi_aperture_outline(set_test_backend):
    lens = singlet()
    lens.surfaces[2].set_semi_aperture(4.0)
    fig, ax = FootprintDiagram(lens, 2).view(show=False)

    lines = ax.get_lines()
    assert len(lines) == 1
    radius = np.hypot(lines[0].get_xdata(), lines[0].get_ydata())
    assert_allclose(radius, 4.0)
    plt.close(fig)


def test_no_outline_without_aperture(set_test_backend):
    fig, ax = FootprintDiagram(singlet(), 2).view(show=False)
    assert ax.get_lines() == []
    assert not any(isinstance(artist, ContourSet) for artist in ax.collections)
    plt.close(fig)


@pytest.mark.parametrize("index", [0, -4, 4, 10])
def test_invalid_surface_index(set_test_backend, index):
    with pytest.raises(ValueError):
        FootprintDiagram(singlet(), index)


def test_negative_index_counts_from_end(set_test_backend):
    lens = singlet()
    assert FootprintDiagram(lens, -2).surface_index == 2


def test_view(set_test_backend):
    footprint = FootprintDiagram(singlet(), 2)
    fig, ax = footprint.view(show=False)

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert ax.get_title() == "Footprint, surface 2"
    assert ax.get_xlabel() == "X (mm)"
    assert ax.get_ylabel() == "Y (mm)"
    assert ax.get_aspect() == 1.0
    labels = ax.get_legend_handles_labels()[1]
    assert labels == ["Hx: 0.000, Hy: 0.000", "Hx: 0.000, Hy: 1.000"]
    plt.close(fig)


def test_view_on_existing_figure(set_test_backend):
    footprint = FootprintDiagram(singlet(), 2)
    existing = plt.figure()
    fig, ax = footprint.view(fig_to_plot_on=existing)

    assert fig is existing
    assert ax in existing.axes
    plt.close(fig)


def test_view_calls_show(set_test_backend):
    footprint = FootprintDiagram(singlet(), 2)
    with patch("matplotlib.pyplot.show") as mock_show:
        fig, _ = footprint.view()
    mock_show.assert_called_once()
    plt.close(fig)
