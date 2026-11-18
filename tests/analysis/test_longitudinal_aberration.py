from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import optiland.backend as be
from optiland.analysis import LongitudinalAberration
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.physical_apertures import RadialAperture
from optiland.samples.objectives import CookeTriplet, DoubleGauss
from tests.utils import assert_allclose

matplotlib.use("Agg")

EPD = 40.0


def parabolic_mirror(defocus=0.0, aperture=None, image_radius=be.inf):
    """On-axis parabolic mirror (f = 100 mm) with its image surface moved by
    ``defocus`` along the axis from the paraxial focus."""
    lens = Optic()
    lens.surfaces.add(index=0, thickness=be.inf)
    lens.surfaces.add(
        index=1,
        radius=-200.0,
        thickness=-100.0 + defocus,
        conic=-1.0,
        material="mirror",
        is_stop=True,
        aperture=aperture,
    )
    lens.surfaces.add(index=2, radius=image_radius)
    lens.set_aperture(aperture_type="EPD", value=EPD)
    lens.fields.set_type("angle")
    lens.fields.add(y=0.0)
    lens.wavelengths.add(value=0.55, is_primary=True)
    return lens


def flat_window():
    """A plane-parallel plate: on-axis rays leave parallel to the axis."""
    lens = Optic()
    lens.surfaces.add(index=0, thickness=be.inf)
    lens.surfaces.add(index=1, thickness=5.0, material=IdealMaterial(1.5), is_stop=True)
    lens.surfaces.add(index=2, thickness=10.0)
    lens.surfaces.add(index=3)
    lens.set_aperture(aperture_type="EPD", value=10.0)
    lens.fields.set_type("angle")
    lens.fields.add(y=0.0)
    lens.wavelengths.add(value=0.55, is_primary=True)
    return lens


def paraxial_focus_offset(optic, wavelength):
    """Paraxial axis crossing measured from the image surface, -y/u."""
    optic.paraxial.trace(0, 1, wavelength)
    return -optic.surfaces.y[-1] / optic.surfaces.u[-1]


class TestLongitudinalAberrationData:
    def test_data_layout(self, set_test_backend):
        la = LongitudinalAberration(CookeTriplet(), num_points=16)
        assert len(la.data["delta_z"]) == len(la.wavelengths) == 3
        for values in la.data["delta_z"]:
            assert be.size(values) == 16
        assert_allclose(la.data["Py"], be.linspace(1 / 16, 1, 16))

    @pytest.mark.parametrize("sample", [CookeTriplet, DoubleGauss])
    def test_small_zone_matches_paraxial_focus(self, set_test_backend, sample):
        optic = sample()
        la = LongitudinalAberration(optic)  # first zone Py = 1/64
        for wp, values in zip(la.wavelengths, la.data["delta_z"], strict=True):
            expected = paraxial_focus_offset(optic, wp.value)
            assert_allclose(values[0], expected, atol=1e-3)

    @pytest.mark.parametrize("defocus", [0.0, 0.5])
    def test_parabola_has_no_spherical_aberration(self, set_test_backend, defocus):
        # Every zone of a parabola focuses at the same point. Moving the image
        # surface by +defocus (toward the mirror) leaves that focus at -defocus
        # along local z, the propagation direction after reflection.
        la = LongitudinalAberration(parabolic_mirror(defocus), num_points=8)
        assert_allclose(la.data["delta_z"][0], be.full((8,), -defocus), atol=1e-9)

    @pytest.mark.parametrize("image_radius", [30.0, -30.0])
    def test_curved_image_surface_measures_from_vertex(
        self, set_test_backend, image_radius
    ):
        # Rays land at different sags on a curved image surface; measuring from
        # the vertex keeps the parabola free of (fake) spherical aberration.
        optic = parabolic_mirror(defocus=5.0, image_radius=image_radius)
        la = LongitudinalAberration(optic, num_points=8)
        assert_allclose(la.data["delta_z"][0], be.full((8,), -5.0), atol=1e-9)

    def test_rays_parallel_to_axis_give_nan(self, set_test_backend):
        la = LongitudinalAberration(flat_window(), num_points=8)
        assert np.all(np.isnan(be.to_numpy(la.data["delta_z"][0])))

    def test_num_points_must_be_positive(self, set_test_backend):
        with pytest.raises(ValueError, match="num_points"):
            LongitudinalAberration(CookeTriplet(), num_points=0)

    def test_vignetted_zones_give_nan(self, set_test_backend):
        # An 11 mm aperture on a 40 mm EPD passes zones with Py < 0.55, so the
        # zones up to 0.5 pass and 0.625 onwards are blocked.
        optic = parabolic_mirror(aperture=RadialAperture(r_max=11.0))
        la = LongitudinalAberration(optic, num_points=8)
        values = be.to_numpy(la.data["delta_z"][0])
        passed = be.to_numpy(la.data["Py"]) <= 0.5
        assert np.all(np.isfinite(values[passed]))
        assert np.all(np.isnan(values[~passed]))


class TestLongitudinalAberrationView:
    def test_view_plots_each_wavelength(self, set_test_backend):
        la = LongitudinalAberration(CookeTriplet(), num_points=8)
        fig, ax = la.view(show=False)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert ax.get_xlabel() == "Longitudinal Aberration (mm)"
        assert ax.get_ylim() == (0, 1)
        assert len(ax.get_legend().get_texts()) == 3
        for k, values in enumerate(la.data["delta_z"]):
            assert_allclose(ax.lines[k].get_xdata(), values)
            assert_allclose(ax.lines[k].get_ydata(), la.data["Py"])
        plt.close(fig)

    def test_view_on_existing_figure(self, set_test_backend):
        la = LongitudinalAberration(CookeTriplet(), num_points=8)
        existing = plt.figure()
        fig, ax = la.view(fig_to_plot_on=existing, show=False)
        assert fig is existing
        assert ax in existing.axes
        plt.close(fig)
