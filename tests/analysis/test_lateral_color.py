from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import optiland.backend as be
from optiland.analysis import LateralColor
from optiland.samples.objectives import CookeTriplet
from optiland.samples.simple import CementedAchromat, Edmund_49_847
from optiland.samples.telescopes import HubbleTelescope
from tests.utils import assert_allclose

matplotlib.use("Agg")


def unobscured_hubble():
    """HubbleTelescope without apertures, so the chief ray reaches the image.

    The sample's primary mirror has a central obscuration that blocks every
    chief ray (they pass through the pupil centre).
    """
    optic = HubbleTelescope()
    for surface in optic.surfaces:
        surface.aperture = None
    return optic


class TestLateralColorData:
    def test_data_layout(self, set_test_backend):
        lc = LateralColor(CookeTriplet(), num_points=16)
        assert len(lc.data["lateral_color"]) == len(lc.wavelengths) == 3
        for values in lc.data["lateral_color"]:
            assert be.size(values) == 16
        assert_allclose(lc.data["Hy"], be.linspace(0, 1, 16))

    @pytest.mark.parametrize("sample", [CookeTriplet, CementedAchromat])
    def test_on_axis_is_zero_for_every_wavelength(self, set_test_backend, sample):
        lc = LateralColor(sample(), num_points=8)
        for values in lc.data["lateral_color"]:
            assert_allclose(values[0], 0.0, atol=1e-12)

    def test_primary_wavelength_is_zero_everywhere(self, set_test_backend):
        lc = LateralColor(CookeTriplet(), wavelengths="primary", num_points=8)
        assert len(lc.data["lateral_color"]) == 1
        assert_allclose(lc.data["lateral_color"][0], be.zeros(8), atol=1e-12)

    def test_off_axis_is_nonzero_for_refractive_lens(self, set_test_backend):
        lc = LateralColor(Edmund_49_847(), num_points=8)
        short = be.to_numpy(lc.data["lateral_color"][0])
        assert np.all(np.abs(short[1:]) > 1e-6)

    def test_all_reflective_system_has_no_lateral_color(self, set_test_backend):
        lc = LateralColor(
            unobscured_hubble(), wavelengths=[0.48, 0.55, 0.65], num_points=8
        )
        for values in lc.data["lateral_color"]:
            assert_allclose(values, be.zeros(8), atol=1e-9)

    def test_blocked_chief_rays_give_nan(self, set_test_backend):
        lc = LateralColor(HubbleTelescope(), wavelengths=[0.48, 0.55], num_points=4)
        for values in lc.data["lateral_color"]:
            assert np.all(np.isnan(be.to_numpy(values)))

    @pytest.mark.parametrize("sample", [CementedAchromat, Edmund_49_847])
    def test_small_field_slope_matches_tchc(self, set_test_backend, sample):
        # Sign convention: TchC differences the system's shortest minus longest
        # wavelength, so its sum predicts d[y(short) - y(long)] / dHy near axis.
        optic = sample()
        values = optic.wavelengths.get_wavelengths()
        lc = LateralColor(optic, wavelengths=[min(values), max(values)], num_points=64)
        y_short, y_long = lc.data["lateral_color"]
        slope = (y_short[1] - y_long[1]) / lc.data["Hy"][1]
        assert_allclose(slope, be.sum(optic.aberrations.TchC()), rtol=0.05)


class TestLateralColorView:
    def test_view_returns_figure_and_axes(self, set_test_backend):
        lc = LateralColor(CookeTriplet(), num_points=8)
        fig, ax = lc.view(show=False)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert ax.get_xlabel() == "Lateral Color (µm)"
        assert len(ax.get_legend().get_texts()) == 3
        plt.close(fig)

    def test_view_plots_micrometers(self, set_test_backend):
        lc = LateralColor(Edmund_49_847(), wavelengths=[0.4861327], num_points=8)
        fig, ax = lc.view(show=False)
        x_plotted = ax.lines[0].get_xdata()
        expected_um = be.to_numpy(lc.data["lateral_color"][0]) * 1e3
        assert_allclose(x_plotted, expected_um)
        plt.close(fig)

    def test_view_on_existing_figure(self, set_test_backend):
        lc = LateralColor(CookeTriplet(), num_points=8)
        existing = plt.figure()
        fig, ax = lc.view(fig_to_plot_on=existing, show=False)
        assert fig is existing
        assert ax in existing.axes
        plt.close(fig)
