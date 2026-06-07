from __future__ import annotations

from unittest.mock import MagicMock, patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import optiland.backend as be
from optiland import analysis
from optiland.analysis import ThroughFocusSpotDiagram
from optiland.optic import Optic
from optiland.physical_apertures import RectangularAperture
from optiland.rays import RealRays
from optiland.samples.objectives import CookeTriplet, TripletTelescopeObjective
from optiland.sources import SMFSource
from optiland.utils import FieldPoint, WavelengthPoint

from .utils import assert_allclose

matplotlib.use("Agg")  # use non-interactive backend for testing


@pytest.fixture
def cooke_triplet():
    return CookeTriplet()


@pytest.fixture
def telescope_objective():
    return TripletTelescopeObjective()


@pytest.fixture
def triplet_four_fields():
    lens = Optic()

    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    lens.surfaces.add(index=1, radius=22.01359, thickness=3.25896, material="SK16")
    lens.surfaces.add(index=2, radius=-435.76044, thickness=6.00755)
    lens.surfaces.add(
        index=3,
        radius=-22.21328,
        thickness=0.99997,
        material=("F2", "schott"),
    )
    lens.surfaces.add(index=4, radius=20.29192, thickness=4.75041, is_stop=True)
    lens.surfaces.add(index=5, radius=79.68360, thickness=2.95208, material="SK16")
    lens.surfaces.add(index=6, radius=-18.39533, thickness=42.20778)
    lens.surfaces.add(index=7)

    lens.set_aperture(aperture_type="EPD", value=10)

    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0)
    lens.fields.add(y=10)
    lens.fields.add(y=15)
    lens.fields.add(y=20)

    lens.wavelengths.add(value=0.48)
    lens.wavelengths.add(value=0.55, is_primary=True)
    lens.wavelengths.add(value=0.65)

    lens.updater.update_paraxial()
    return lens


class TestCookeTripetSpotDiagram:
    def test_spot_geometric_radius(self, set_test_backend, cooke_triplet):
        spot = analysis.SpotDiagram(cooke_triplet)
        geo_radius = spot.geometric_spot_radius()

        assert_allclose(geo_radius[0][0], 0.00597244087781)
        assert_allclose(geo_radius[0][1], 0.00628645771124)
        assert_allclose(geo_radius[0][2], 0.00931911440064)

        assert_allclose(geo_radius[1][0], 0.03928464835617618)
        assert_allclose(geo_radius[1][1], 0.04075295155639047)
        assert_allclose(geo_radius[1][2], 0.04772194200606705)

        assert_allclose(geo_radius[2][0], 0.018909146395329878)
        assert_allclose(geo_radius[2][1], 0.022501847359635008)
        assert_allclose(geo_radius[2][2], 0.036545592330568866)

    def test_spot_rms_radius(self, set_test_backend, cooke_triplet):
        spot = analysis.SpotDiagram(cooke_triplet)
        rms_radius = spot.rms_spot_radius()

        assert_allclose(rms_radius[0][0], 0.003791335461448)
        assert_allclose(rms_radius[0][1], 0.004293689564257)
        assert_allclose(rms_radius[0][2], 0.006195618755672)

        assert_allclose(rms_radius[1][0], 0.01582480029344623)
        assert_allclose(rms_radius[1][1], 0.016918412809703662)
        assert_allclose(rms_radius[1][2], 0.019221165873836682)

        assert_allclose(rms_radius[2][0], 0.013236232767092956)
        assert_allclose(rms_radius[2][1], 0.012116688566406967)
        assert_allclose(rms_radius[2][2], 0.013648684944411313)

    def test_airy_disc(self, set_test_backend, cooke_triplet):
        spot = analysis.SpotDiagram(cooke_triplet)
        airy_radius = spot.airy_disc_x_y(wavelength=cooke_triplet.primary_wavelength)
        airy_radius_x, airy_radius_y = airy_radius

        assert_allclose(airy_radius_x[0], 0.0033403700287742426)
        assert_allclose(airy_radius_x[1], 0.003579789351003376)
        assert_allclose(airy_radius_x[2], 0.003825501589577882)

        assert_allclose(airy_radius_y[0], 0.0033403700287742426)
        assert_allclose(airy_radius_y[1], 0.003430811760325915)
        assert_allclose(airy_radius_y[2], 0.0035453238661865244)

    def test_airy_disc_in_view_spot_diagram(self, set_test_backend, cooke_triplet):
        spot = analysis.SpotDiagram(cooke_triplet)
        fig, axes = spot.view(add_airy_disk=True)

        assert fig is not None
        assert len(axes) > 0
        assert isinstance(fig, Figure)
        assert all(isinstance(ax, Axes) for ax in axes)
        plt.close(fig)

    def test_view_spot_diagram(self, set_test_backend, cooke_triplet):
        spot = analysis.SpotDiagram(cooke_triplet)
        fig, axes = spot.view()
        assert fig is not None
        assert len(axes) > 0
        assert isinstance(fig, Figure)
        assert all(isinstance(ax, Axes) for ax in axes)
        plt.close(fig)

    def test_view_spot_diagram_larger_fig(self, set_test_backend, cooke_triplet):
        spot = analysis.SpotDiagram(cooke_triplet)
        fig, axes = spot.view(figsize=(20, 10))
        assert fig is not None
        assert len(axes) > 0
        assert isinstance(fig, Figure)
        assert all(isinstance(ax, Axes) for ax in axes)
        plt.close(fig)


class TestTripletSpotDiagram:
    def test_view_spot_diagram(self, set_test_backend, triplet_four_fields):
        spot = analysis.SpotDiagram(triplet_four_fields)
        fig, axes = spot.view()
        assert fig is not None
        assert len(axes) > 0
        assert isinstance(fig, Figure)
        assert all(isinstance(ax, Axes) for ax in axes)
        plt.close(fig)

    def test_view_spot_diagram_larger_fig(self, set_test_backend, triplet_four_fields):
        spot = analysis.SpotDiagram(triplet_four_fields)
        fig, axes = spot.view(figsize=(20, 10))
        assert fig is not None
        assert len(axes) > 0
        assert isinstance(fig, Figure)
        assert all(isinstance(ax, Axes) for ax in axes)
        plt.close(fig)


class TestCookeTripletEncircledEnergy:
    def test_encircled_energy_centroid(self, set_test_backend, cooke_triplet):
        encircled_energy = analysis.EncircledEnergy(cooke_triplet)
        centroid = encircled_energy.centroid()

        # encircled energy calculation includes randomness, so abs. tolerance
        # is set to 1e-3

        assert_allclose(centroid[0][0], -8.207497747771947e-06, atol=1e-3, rtol=1e-3)
        assert_allclose(centroid[0][1], 1.989147771098717e-06, atol=1e-3, rtol=1e-3)

        assert_allclose(centroid[1][0], 3.069405792964239e-05, atol=1e-3, rtol=1e-3)
        assert_allclose(centroid[1][1], 12.421326489507168, atol=1e-3, rtol=1e-3)

        assert_allclose(centroid[2][0], 3.1631726815066986e-07, atol=1e-3, rtol=1e-3)
        assert_allclose(centroid[2][1], 18.13502264954927, atol=1e-3, rtol=1e-3)

    def test_view_encircled_energy(self, set_test_backend, cooke_triplet):
        encircled_energy = analysis.EncircledEnergy(cooke_triplet)
        fig, ax = encircled_energy.view()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_view_encircled_energy_larger_fig(self, set_test_backend, cooke_triplet):
        encircled_energy = analysis.EncircledEnergy(cooke_triplet)
        fig, ax = encircled_energy.view(figsize=(20, 10))
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_view_encircled_energy_larger_fig(self, set_test_backend, cooke_triplet):
        encircled_energy = analysis.EncircledEnergy(cooke_triplet)
        fig, ax = encircled_energy.view(figsize=(20, 10))
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)


class TestCookeTripletRayFan:
    def test_ray_fan(self, set_test_backend, cooke_triplet):
        fan = analysis.RayFan(cooke_triplet)

        assert_allclose(fan.data["Px"][0], -1)
        assert_allclose(fan.data["Px"][-1], 1)

        assert_allclose(fan.data["Py"][0], -1)
        assert_allclose(fan.data["Py"][-1], 1)

        assert_allclose(
            fan.data["(0.0, 0.0)"]["0.48"]["x"][0],
            0.00238814980958324,
            atol=1e-9,
        )

        assert_allclose(
            fan.data["(0.0, 0.0)"]["0.48"]["x"][0],
            0.00238814980958324,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.0)"]["0.48"]["x"][-1],
            -0.00238814980958324,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.0)"]["0.48"]["y"][0],
            0.00238814980958324,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.0)"]["0.48"]["y"][-1],
            -0.00238814980958324,
            atol=1e-9,
        )

        assert_allclose(
            fan.data["(0.0, 0.0)"]["0.55"]["x"][0],
            0.004195677081323623,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.0)"]["0.55"]["x"][-1],
            -0.004195677081323623,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.0)"]["0.55"]["y"][0],
            0.004195677081323623,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.0)"]["0.55"]["y"][-1],
            -0.004195677081323623,
            atol=1e-9,
        )

        assert_allclose(
            fan.data["(0.0, 0.0)"]["0.65"]["x"][0],
            -8.284696919602652e-06,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.0)"]["0.65"]["x"][-1],
            8.284696919602652e-06,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.0)"]["0.65"]["y"][0],
            -8.284696919602652e-06,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.0)"]["0.65"]["y"][-1],
            8.284696919602652e-06,
            atol=1e-9,
        )

        assert_allclose(
            fan.data["(0.0, 0.7)"]["0.48"]["x"][0],
            0.01973142095198721,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.7)"]["0.48"]["x"][-1],
            -0.01973142095198721,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.7)"]["0.48"]["y"][0],
            -0.023207115035676296,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.7)"]["0.48"]["y"][-1],
            0.03928464835618861,
            atol=1e-9,
        )

        assert_allclose(
            fan.data["(0.0, 0.7)"]["0.55"]["x"][0],
            0.021420191179537973,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.7)"]["0.55"]["x"][-1],
            -0.021420191179537973,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.7)"]["0.55"]["y"][0],
            -0.024812371459915994,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.7)"]["0.55"]["y"][-1],
            0.04075295155640113,
            atol=1e-9,
        )

        assert_allclose(
            fan.data["(0.0, 0.7)"]["0.65"]["x"][0],
            0.017025487217305013,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.7)"]["0.65"]["x"][-1],
            -0.017025487217305013,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.7)"]["0.65"]["y"][0],
            -0.03229666187094615,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.7)"]["0.65"]["y"][-1],
            0.047721942006075935,
            atol=1e-9,
        )

        assert_allclose(
            fan.data["(0.0, 1.0)"]["0.48"]["x"][0],
            0.01563881685548374,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 1.0)"]["0.48"]["x"][-1],
            -0.01563881685548374,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 1.0)"]["0.48"]["y"][0],
            -0.0044989771745065354,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 1.0)"]["0.48"]["y"][-1],
            0.013000385049824814,
            atol=1e-9,
        )

        assert_allclose(
            fan.data["(0.0, 1.0)"]["0.55"]["x"][0],
            0.016936433773790505,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 1.0)"]["0.55"]["x"][-1],
            -0.016936433773790505,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 1.0)"]["0.55"]["y"][0],
            -0.01705141007843025,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 1.0)"]["0.55"]["y"][-1],
            0.022501847359645666,
            atol=1e-9,
        )

        assert_allclose(
            fan.data["(0.0, 1.0)"]["0.65"]["x"][0],
            0.01214534602206907,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 1.0)"]["0.65"]["x"][-1],
            -0.01214534602206907,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 1.0)"]["0.65"]["y"][0],
            -0.033957537601747134,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 1.0)"]["0.65"]["y"][-1],
            0.036545592330593735,
            atol=1e-9,
        )

    def test_view_ray_fan(self, set_test_backend, cooke_triplet):
        ray_fan = analysis.RayFan(cooke_triplet)
        fig, axes = ray_fan.view()
        assert fig is not None
        assert len(axes) > 0
        assert isinstance(fig, Figure)
        assert all(isinstance(ax, Axes) for ax in axes)
        plt.close(fig)

    def test_view_ray_fan_larger_fig(self, set_test_backend, cooke_triplet):
        ray_fan = analysis.RayFan(cooke_triplet)
        fig, axes = ray_fan.view(figsize=(20, 10))
        assert fig is not None
        assert len(axes) > 0
        assert isinstance(fig, Figure)
        assert all(isinstance(ax, Axes) for ax in axes)
        plt.close(fig)


class TestTelescopeTripletYYbar:
    def test_view_yybar(self, set_test_backend, telescope_objective):
        yybar = analysis.YYbar(telescope_objective)
        fig, ax = yybar.view()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_view_yybar_larger_fig(self, set_test_backend, telescope_objective):
        yybar = analysis.YYbar(telescope_objective)
        fig, ax = yybar.view(figsize=(12.4, 10))
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)


class TestTelescopeTripletDistortion:
    def test_distortion_values(self, set_test_backend, telescope_objective):
        dist = analysis.Distortion(telescope_objective)

        assert_allclose(dist.data[0][0], 0.0, atol=1e-9)
        assert_allclose(dist.data[0][-1], 0.005950509480884957, atol=1e-9)

        assert_allclose(dist.data[1][0], 0.0, atol=1e-9)
        assert_allclose(dist.data[1][-1], 0.005786305783771451, atol=1e-9)

        assert_allclose(dist.data[0][0], 0.0, atol=1e-9)
        assert_allclose(dist.data[2][-1], 0.005720392850412076, atol=1e-9)

    def test_f_theta_distortion(self, set_test_backend, telescope_objective):
        dist = analysis.Distortion(telescope_objective, distortion_type="f-theta")

        assert_allclose(dist.data[0][0], 0.0, atol=1e-9)
        assert_allclose(dist.data[0][-1], 0.016106265133212852, atol=1e-9)

        assert_allclose(dist.data[1][0], 0.0, atol=1e-9)
        assert_allclose(dist.data[1][-1], 0.015942044760968603, atol=1e-9)

        assert_allclose(dist.data[2][0], 0.0, atol=1e-9)
        assert_allclose(dist.data[2][-1], 0.015876125134060767, atol=1e-9)

    def test_invalid_distortion_type(self, set_test_backend, telescope_objective):
        with pytest.raises(ValueError):
            analysis.Distortion(telescope_objective, distortion_type="invalid")

    def test_view_distortion(self, set_test_backend, telescope_objective):
        dist = analysis.Distortion(telescope_objective)
        fig, ax = dist.view()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_view_distortion_larger_fig(self, set_test_backend, telescope_objective):
        dist = analysis.Distortion(telescope_objective)
        fig, ax = dist.view(figsize=(12.4, 10))
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_view_distortion_larger_fig(self, set_test_backend, telescope_objective):
        dist = analysis.Distortion(telescope_objective)
        fig, ax = dist.view(figsize=(12.4, 10))
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_view_distortion_larger_fig(self, set_test_backend, telescope_objective):
        dist = analysis.Distortion(telescope_objective)
        fig, ax = dist.view(figsize=(12.4, 10))
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)


class TestTelescopeTripletGridDistortion:
    def test_grid_distortion_values(self, set_test_backend, telescope_objective):
        dist = analysis.GridDistortion(telescope_objective)
        assert_allclose(dist.data["max_distortion"], 0.005785718069180374, atol=1e-9)

        assert dist.data["xr"].shape == (10, 10)
        assert dist.data["yr"].shape == (10, 10)
        assert dist.data["xp"].shape == (10, 10)
        assert dist.data["yp"].shape == (10, 10)

        assert_allclose(dist.data["xr"][0, 0], -1.2342622299776145, atol=1e-9)
        assert_allclose(dist.data["xr"][4, 6], 0.41137984374933073, atol=1e-9)

        assert_allclose(dist.data["yr"][1, 0], -0.959951505834632, atol=1e-9)
        assert_allclose(dist.data["yr"][2, 6], -0.6856458243955965, atol=1e-9)

        assert_allclose(dist.data["xp"][0, 2], -0.6856375010477692, atol=1e-9)
        assert_allclose(dist.data["xp"][4, 4], -0.13712543741510327, atol=1e-9)

        assert_allclose(dist.data["yp"][-1, 0], 1.2341908231761498, atol=1e-9)
        assert_allclose(dist.data["yp"][1, 5], -0.9599069415493584, atol=1e-9)

    def test_f_theta_distortion(self, set_test_backend, telescope_objective):
        dist = analysis.GridDistortion(telescope_objective, distortion_type="f-theta")

        assert_allclose(dist.data["max_distortion"], 0.010863278146924825, atol=1e-9)

        assert dist.data["xr"].shape == (10, 10)
        assert dist.data["yr"].shape == (10, 10)
        assert dist.data["xp"].shape == (10, 10)
        assert dist.data["yp"].shape == (10, 10)

        assert_allclose(dist.data["xr"][0, 0], -1.2342622299776145, atol=1e-9)
        assert_allclose(dist.data["xr"][4, 6], 0.41137984374933073, atol=1e-9)

        assert_allclose(dist.data["yr"][1, 0], -0.959951505834632, atol=1e-9)
        assert_allclose(dist.data["yr"][2, 6], -0.6856458243955965, atol=1e-9)

        assert_allclose(dist.data["xp"][0, 2], -0.6856267573347536, atol=1e-9)
        assert_allclose(dist.data["xp"][4, 4], -0.13712535146695065, atol=1e-9)

        assert_allclose(dist.data["yp"][-1, 0], 1.2341281632025562, atol=1e-9)
        assert_allclose(dist.data["yp"][1, 5], -0.9598774602686547, atol=1e-9)

    def test_invalid_distortion_type(self, set_test_backend, telescope_objective):
        with pytest.raises(ValueError):
            analysis.GridDistortion(telescope_objective, distortion_type="invalid")

    def test_view_grid_distortion(self, set_test_backend, telescope_objective):
        dist = analysis.GridDistortion(telescope_objective)
        fig, ax = dist.view()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_view_grid_distortion_larger_fig(
        self, set_test_backend, telescope_objective
    ):
        dist = analysis.GridDistortion(telescope_objective)
        fig, ax = dist.view(figsize=(12.4, 10))
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)


class TestTelescopeTripletFieldCurvature:
    def test_field_curvature_init(self, set_test_backend, telescope_objective):
        field_curvature = analysis.FieldCurvature(telescope_objective)
        assert field_curvature.optic == telescope_objective
        assert (
            [wp.value for wp in field_curvature.wavelengths]
            == telescope_objective.wavelengths.get_wavelengths()
        )
        assert field_curvature.num_points == 128

    def test_field_curvature_init_with_wavelength(
        self, set_test_backend, telescope_objective
    ):
        field_curvature = analysis.FieldCurvature(
            telescope_objective,
            wavelengths=[0.5, 0.6],
        )
        assert field_curvature.optic == telescope_objective
        assert [wp.value for wp in field_curvature.wavelengths] == [0.5, 0.6]
        assert field_curvature.num_points == 128

    def test_field_curvature_init_with_num_points(
        self, set_test_backend, telescope_objective
    ):
        num_points = 256
        field_curvature = analysis.FieldCurvature(
            telescope_objective,
            num_points=num_points,
        )
        assert field_curvature.optic == telescope_objective
        assert (
            [wp.value for wp in field_curvature.wavelengths]
            == telescope_objective.wavelengths.get_wavelengths()
        )
        assert field_curvature.num_points == num_points

    def test_field_curvature_init_with_all_parameters(
        self, set_test_backend, telescope_objective
    ):
        num_points = 256
        field_curvature = analysis.FieldCurvature(
            telescope_objective,
            wavelengths=[0.55],
            num_points=num_points,
        )
        assert field_curvature.optic == telescope_objective
        assert [wp.value for wp in field_curvature.wavelengths] == [0.55]
        assert field_curvature.num_points == num_points

    def test_field_curvature_view(self, set_test_backend, telescope_objective):
        field_curvature = analysis.FieldCurvature(telescope_objective)
        fig, ax = field_curvature.view()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_field_curvature_generate_data(self, set_test_backend, telescope_objective):
        f = analysis.FieldCurvature(telescope_objective)

        assert_allclose(f.data[0][0][89], -0.0013062169220806206, atol=1e-9)
        assert_allclose(f.data[0][1][40], 0.03435268469825703, atol=1e-9)
        assert_allclose(f.data[0][1][112], 0.012502083379998098, atol=1e-9)
        assert_allclose(f.data[0][0][81], 0.005363808856891348, atol=1e-9)
        assert_allclose(f.data[0][0][127], -0.041553105637156224, atol=1e-9)
        assert_allclose(f.data[0][0][40], 0.02969815644838593, atol=1e-9)
        assert_allclose(f.data[0][0][57], 0.021608994058848974, atol=1e-9)
        assert_allclose(f.data[0][1][45], 0.03350406866891282, atol=1e-9)
        assert_allclose(f.data[0][1][74], 0.026613511090172324, atol=1e-9)
        assert_allclose(f.data[0][1][94], 0.01990500178194723, atol=1e-9)

        assert_allclose(f.data[1][1][55], -0.004469963728211546, atol=1e-9)
        assert_allclose(f.data[1][1][19], 0.0008003571732224457, atol=1e-9)
        assert_allclose(f.data[1][1][93], -0.015595499139883678, atol=1e-9)
        assert_allclose(f.data[1][0][15], 0.0004226818372030349, atol=1e-9)
        assert_allclose(f.data[1][1][50], -0.0034313474749693047, atol=1e-9)
        assert_allclose(f.data[1][0][110], -0.05718858127937811, atol=1e-9)
        assert_allclose(f.data[1][0][89], -0.036917737894907106, atol=1e-9)
        assert_allclose(f.data[1][1][75], -0.00961346547634129, atol=1e-9)
        assert_allclose(f.data[1][0][69], -0.021587199726177217, atol=1e-9)

        assert_allclose(f.data[2][1][62], 0.059485399479466794, atol=1e-9)
        assert_allclose(f.data[2][0][103], 0.015768399161337723, atol=1e-9)
        assert_allclose(f.data[2][0][0], 0.06707048647659668, atol=1e-9)
        assert_allclose(f.data[2][1][68], 0.05794633552031286, atol=1e-9)
        assert_allclose(f.data[2][0][6], 0.06689636005684219, atol=1e-9)
        assert_allclose(f.data[2][1][40], 0.06391326892594748, atol=1e-9)
        assert_allclose(f.data[2][0][88], 0.029620344519446916, atol=1e-9)
        assert_allclose(f.data[2][0][5], 0.06694956529269887, atol=1e-9)
        assert_allclose(f.data[2][1][98], 0.048120430272662294, atol=1e-9)
        assert_allclose(f.data[2][0][5], 0.06694956529269887, atol=1e-9)


class TestSpotVsField:
    def test_rms_spot_size_vs_field_initialization(
        self, set_test_backend, telescope_objective
    ):
        spot_vs_field = analysis.RmsSpotSizeVsField(telescope_objective)
        assert spot_vs_field.num_fields == 64
        assert be.array_equal(spot_vs_field._field[:, 1], be.linspace(0, 1, 64))

    def test_rms_spot_radius(self, set_test_backend, telescope_objective):
        spot_vs_field = analysis.RmsSpotSizeVsField(telescope_objective)
        spot_size = spot_vs_field._spot_size
        assert spot_size.shape == (
            64,
            len(telescope_objective.wavelengths.get_wavelengths()),
        )

    def test_view_spot_vs_field(self, set_test_backend, telescope_objective):
        spot_vs_field = analysis.RmsSpotSizeVsField(telescope_objective)
        fig, ax = spot_vs_field.view()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_view_spot_vs_field_larger_fig(self, set_test_backend, telescope_objective):
        spot_vs_field = analysis.RmsSpotSizeVsField(telescope_objective)
        fig, ax = spot_vs_field.view(figsize=(12.4, 10))
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)


class TestWavefrontErrorVsField:
    def test_rms_wave_init(self, set_test_backend, telescope_objective):
        wavefront_error_vs_field = analysis.RmsWavefrontErrorVsField(
            telescope_objective,
        )
        assert wavefront_error_vs_field.num_fields == 32
        assert be.array_equal(
            wavefront_error_vs_field._field[:, 1],
            be.linspace(0, 1, 32),
        )

    def test_rms_wave(self, set_test_backend, telescope_objective):
        wavefront_error_vs_field = analysis.RmsWavefrontErrorVsField(
            telescope_objective,
        )
        wavefront_error = wavefront_error_vs_field._wavefront_error
        assert wavefront_error.shape == (
            32,
            len(telescope_objective.wavelengths.get_wavelengths()),
        )

    def test_view_wave(self, set_test_backend, telescope_objective):
        wavefront_error_vs_field = analysis.RmsWavefrontErrorVsField(
            telescope_objective,
        )
        fig, ax = wavefront_error_vs_field.view()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_view_wave_larger_fig(self, set_test_backend, telescope_objective):
        wavefront_error_vs_field = analysis.RmsWavefrontErrorVsField(
            telescope_objective,
        )
        fig, ax = wavefront_error_vs_field.view(figsize=(12.4, 10))
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)


class TestPupilAberration:
    def test_initialization(self, set_test_backend, telescope_objective):
        pupil_ab = analysis.PupilAberration(telescope_objective)
        assert pupil_ab.optic == telescope_objective
        assert [fp.coord for fp in pupil_ab.fields] == [(0.0, 0.0), (0.0, 0.7), (0.0, 1.0)]
        assert [wp.value for wp in pupil_ab.wavelengths] == [0.4861, 0.5876, 0.6563]
        assert pupil_ab.num_points == 257  # num_points is forced to be odd

    def test_generate_data(self, set_test_backend, telescope_objective):
        pupil_ab = analysis.PupilAberration(telescope_objective)
        data = pupil_ab._generate_data()
        assert "Px" in data
        assert "Py" in data
        assert "(0.0, 0.0)" in data
        assert "(0.0, 0.7)" in data
        assert "(0.0, 1.0)" in data
        assert "0.4861" in data["(0.0, 0.0)"]
        assert "0.5876" in data["(0.0, 0.0)"]
        assert "0.6563" in data["(0.0, 0.0)"]
        assert "x" in data["(0.0, 0.0)"]["0.4861"]
        assert "y" in data["(0.0, 0.0)"]["0.4861"]

    def test_view(self, set_test_backend, telescope_objective):
        pupil_ab = analysis.PupilAberration(telescope_objective)
        fig, axes = pupil_ab.view()
        assert fig is not None
        assert axes is not None
        assert len(axes) == 3
        assert isinstance(fig, Figure)
        plt.close(fig)


def test_spotdiagram_invalid_coordinates(cooke_triplet):
    with pytest.raises(ValueError) as excinfo:
        analysis.SpotDiagram(cooke_triplet, coordinates="invalid")
    assert str(excinfo.value) == "Coordinates must be 'global' or 'local'."


def test_generate_field_data_local(set_test_backend, cooke_triplet):
    spot = analysis.SpotDiagram(cooke_triplet, coordinates="local")

    # Pick the first field and wavelength
    field = spot.fields[0].coord
    wavelength = spot.wavelengths[0].value

    data = spot._generate_field_data(
        field,
        wavelength,
        num_rays=10,
        distribution="hexapolar",
        coordinates="local",
    )

    plot_x = data.x
    plot_y = data.y
    # intensity = data.intensity # if needed for future assertions
    global_x = spot.optic.surfaces.x[-1, :]
    global_y = spot.optic.surfaces.y[-1, :]
    assert_allclose(plot_x, global_x)
    assert_allclose(plot_y, global_y)


def test_generate_field_data_global(set_test_backend, cooke_triplet):
    spot = analysis.SpotDiagram(cooke_triplet, coordinates="global")

    # Pick the first field and wavelength
    field = spot.fields[0].coord
    wavelength = spot.wavelengths[0].value

    data = spot._generate_field_data(
        field,
        wavelength,
        num_rays=3,
        distribution="hexapolar",
        coordinates="global",
    )

    plot_x = data.x
    plot_y = data.y
    # intensity = data.intensity # if needed for future assertions
    global_x = spot.optic.surfaces.x[-1, :]
    global_y = spot.optic.surfaces.y[-1, :]
    assert_allclose(plot_x, global_x)
    assert_allclose(plot_y, global_y)


@pytest.fixture
def test_system_irradiance_v1():
    class TestSystemIrradianceV1(Optic):
        def __init__(self):
            super().__init__()
            self.surfaces.add(index=0, thickness=be.inf)
            self.surfaces.add(index=1, thickness=0, is_stop=True)
            self.surfaces.add(index=2, thickness=10)
            self.surfaces.add(index=3)  # image
            detector_size = RectangularAperture(
                x_max=2.5, x_min=-2.5, y_max=2.5, y_min=-2.5
            )
            self.surfaces[-1].aperture = detector_size
            self.wavelengths.add(0.55)
            self.fields.set_type("angle")
            self.fields.add(y=0)
            self.set_aperture("EPD", 5.0)

    return TestSystemIrradianceV1()


@pytest.fixture
def perfect_mirror_system():
    class PerfectMirror(Optic):
        def __init__(self):
            super().__init__()
            self.surfaces.add(index=0, thickness=be.inf)
            self.surfaces.add(index=1, thickness=50)
            self.surfaces.add(
                index=2,
                thickness=-25,
                radius=-50,
                conic=-1.0,
                material="mirror",
                is_stop=True,
            )
            self.surfaces.add(index=3)  # image
            detector_size = RectangularAperture(
                x_max=2.5, x_min=-2.5, y_max=2.5, y_min=-2.5
            )
            self.surfaces[-1].aperture = detector_size
            self.wavelengths.add(0.55)
            self.fields.set_type("angle")
            self.fields.add(y=0)
            self.set_aperture("EPD", 5.0)

    return PerfectMirror()


def _create_square_grid_rays(num_rays_edge, min_coord, max_coord, wavelength_val=0.55):
    x_rays_np = be.linspace(min_coord, max_coord, num_rays_edge)
    x_np, y_np = be.meshgrid(x_rays_np, x_rays_np)
    x_be_flat = be.array(x_np.flatten())
    y_be_flat = be.array(y_np.flatten())
    num_rays = x_be_flat.shape[0]

    z_be = be.zeros((num_rays,))  # pass shape as tuple because of torch backend
    L_be = be.zeros((num_rays,))
    M_be = be.zeros((num_rays,))
    N_be = be.ones((num_rays,))
    I_be = be.ones((num_rays,))
    W_be = be.full((num_rays,), wavelength_val)
    return RealRays(x_be_flat, y_be_flat, z_be, L_be, M_be, N_be, I_be, W_be)


def _apply_gaussian_apodization(
    x_coords_flat, y_coords_flat, sigma_x, sigma_y, peak_intensity=1.0
):
    # x_coords_flat and y_coords_flat are expected to be 1D backend arrays
    x_np = be.to_numpy(x_coords_flat)
    y_np = be.to_numpy(y_coords_flat)
    exponent = -(((x_np**2) / (2 * sigma_x**2)) + ((y_np**2) / (2 * sigma_y**2)))
    intensities_np = peak_intensity * be.exp(exponent)
    return be.array(intensities_np)


class TestIncoherentIrradiance:
    def test_irradiance_v1_uniform_and_user_defined_rays(
        self, set_test_backend, test_system_irradiance_v1
    ):
        optic_sys = test_system_irradiance_v1
        res = (5, 5)

        irr_uniform = analysis.IncoherentIrradiance(
            optic_sys, num_rays=500, distribution="uniform", res=res
        )
        irr_map_uniform, _, _ = irr_uniform.data[0][0]

        assert be.sum(irr_map_uniform) > 0
        assert be.max(irr_map_uniform) > 0
        fig, axes = irr_uniform.view()
        assert fig is not None
        assert axes is not None
        assert isinstance(fig, Figure)
        assert len(axes) > 0
        plt.close(fig)

        num_rays_edge = 5
        user_rays = _create_square_grid_rays(
            num_rays_edge=num_rays_edge, min_coord=-2.25, max_coord=2.25
        )
        irr_user = analysis.IncoherentIrradiance(
            optic_sys, res=res, user_initial_rays=user_rays
        )
        irr_map_user, _, _ = irr_user.data[0][0]

        pixel_area = ((2.5 - (-2.5)) / res[0]) * ((2.5 - (-2.5)) / res[1])
        total_power_on_detector = be.sum(irr_map_user) * pixel_area
        initial_total_power = be.sum(user_rays.i)

        assert_allclose(total_power_on_detector, initial_total_power, atol=1e-5)
        irr_user.view()

    def test_irradiance_v1_one_ray_per_other_pixel(
        self, set_test_backend, test_system_irradiance_v1
    ):
        optic_sys = test_system_irradiance_v1
        res_val = (10, 10)

        x_centers_np = be.linspace(-2.25, 2.25, 10)
        selected_x_np = x_centers_np[::2]
        selected_y_np = x_centers_np[::2]
        x_np_mesh, y_np_mesh = be.meshgrid(selected_x_np, selected_y_np)

        x_be_flat = be.array(x_np_mesh.flatten())
        y_be_flat = be.array(y_np_mesh.flatten())
        num_rays_flat = x_be_flat.shape[0]

        user_rays = RealRays(
            x=x_be_flat,
            y=y_be_flat,
            z=be.zeros((num_rays_flat,)),
            L=be.zeros((num_rays_flat,)),
            M=be.zeros((num_rays_flat,)),
            N=be.ones((num_rays_flat,)),
            intensity=be.ones((num_rays_flat,)),
            wavelength=be.full((num_rays_flat,), 0.55),
        )

        irr_analysis = analysis.IncoherentIrradiance(
            optic_sys, res=res_val, user_initial_rays=user_rays
        )
        irr_map_be, _, _ = irr_analysis.data[0][0]

        # Check for energy conservation
        pixel_area = ((2.5 - (-2.5)) / res_val[0]) * ((2.5 - (-2.5)) / res_val[1])
        total_power_on_detector = be.sum(irr_map_be) * pixel_area
        initial_total_power = be.sum(user_rays.i)

        assert_allclose(total_power_on_detector, initial_total_power, atol=1e-5)
        fig, axes = irr_analysis.view()
        assert fig is not None
        assert axes is not None
        assert isinstance(fig, Figure)
        assert len(axes) > 0

        plt.close(fig)

    def test_irradiance_gaussian_apodization(
        self, set_test_backend, test_system_irradiance_v1
    ):
        optic_sys = test_system_irradiance_v1
        res_val = (50, 50)
        num_rays_edge = 100
        sigma_val = 0.5

        x_rays_np = be.linspace(-2.5, 2.5, num_rays_edge)
        x_np_mesh, y_np_mesh = be.meshgrid(x_rays_np, x_rays_np)
        x_be_flat = be.array(x_np_mesh.flatten())
        y_be_flat = be.array(y_np_mesh.flatten())
        num_rays_flat = x_be_flat.shape[0]

        gaussian_intensities = _apply_gaussian_apodization(
            x_be_flat, y_be_flat, sigma_x=sigma_val, sigma_y=sigma_val
        )

        user_rays_apodized = RealRays(
            x=x_be_flat,
            y=y_be_flat,
            z=be.zeros((num_rays_flat,)),
            L=be.zeros((num_rays_flat,)),
            M=be.zeros((num_rays_flat,)),
            N=be.ones((num_rays_flat,)),
            intensity=gaussian_intensities,
            wavelength=be.full((num_rays_flat,), 0.55),
        )

        irr_apodized = analysis.IncoherentIrradiance(
            optic_sys, res=res_val, user_initial_rays=user_rays_apodized
        )
        irr_map_apodized_be, x_edges, y_edges = irr_apodized.data[0][0]

        center_idx_x = res_val[0] // 2
        center_idx_y = res_val[1] // 2

        val_center = irr_map_apodized_be[center_idx_x, center_idx_y]
        val_corner1 = irr_map_apodized_be[0, 0]
        val_corner2 = irr_map_apodized_be[res_val[0] - 1, res_val[1] - 1]
        max_val_be = be.max(irr_map_apodized_be)

        assert be.to_numpy(val_center) > be.to_numpy(val_corner1)
        assert be.to_numpy(val_center) > be.to_numpy(val_corner2)
        assert_allclose(val_center, max_val_be, rtol=1e-3)

        fig, axes = irr_apodized.view()
        assert fig is not None
        assert axes is not None
        assert isinstance(fig, Figure)
        assert len(axes) > 0
        plt.close(fig)

    def test_irradiance_perfect_mirror_focus(
        self, set_test_backend, perfect_mirror_system
    ):
        optic_sys = perfect_mirror_system
        res_val = (21, 21)
        num_rays_epd = 51

        user_rays_grid = _create_square_grid_rays(
            num_rays_edge=num_rays_epd, min_coord=-2.5, max_coord=2.5
        )

        irr_perfect = analysis.IncoherentIrradiance(
            optic_sys, res=res_val, user_initial_rays=user_rays_grid
        )
        irr_map_be, _, _ = irr_perfect.data[0][0]
        irr_map_np = be.to_numpy(irr_map_be)

        # With bilinear interpolation, a perfect focus on a grid vertex
        # distributes power among the 4 adjacent pixels

        total_sum_map = be.to_numpy(be.sum(irr_map_be))
        assert total_sum_map > 1e-9

        pixel_area = ((2.5 - (-2.5)) / res_val[0]) * ((2.5 - (-2.5)) / res_val[1])
        total_power_on_detector = total_sum_map * pixel_area
        initial_total_power = be.sum(user_rays_grid.i)

        assert_allclose(total_power_on_detector, initial_total_power, atol=1e-5)
        center_x_idx = res_val[0] // 2
        center_y_idx = res_val[1] // 2

        p1 = irr_map_np[center_x_idx, center_y_idx]
        p2 = irr_map_np[center_x_idx - 1, center_y_idx]
        p3 = irr_map_np[center_x_idx, center_y_idx - 1]
        p4 = irr_map_np[center_x_idx - 1, center_y_idx - 1]
        central_four_pixel_sum = p1 + p2 + p3 + p4

        # The sum of these four pixels should equal the total power in the map
        assert_allclose(central_four_pixel_sum, total_sum_map, atol=1e-5)

        mask = np.ones_like(irr_map_np, dtype=bool)
        mask[center_x_idx, center_y_idx] = False
        mask[center_x_idx - 1, center_y_idx] = False
        mask[center_x_idx, center_y_idx - 1] = False
        mask[center_x_idx - 1, center_y_idx - 1] = False
        assert_allclose(np.sum(irr_map_np[mask]), 0.0, atol=1e-5)

        fig, axes = irr_perfect.view()
        assert fig is not None
        assert axes is not None
        assert isinstance(fig, Figure)
        assert len(axes) > 0

        plt.close(fig)

    def test_irradiance_plot_cross_section_gaussian(
        self, set_test_backend, test_system_irradiance_v1
    ):
        optic_sys = test_system_irradiance_v1
        res_val = (50, 50)
        num_rays_edge = 100
        sigma_val = 0.5

        x_rays_np = np.linspace(-2.5, 2.5, num_rays_edge)
        x_np_mesh, y_np_mesh = np.meshgrid(x_rays_np, x_rays_np)
        x_be_flat = be.array(x_np_mesh.flatten())
        y_be_flat = be.array(y_np_mesh.flatten())
        num_rays_flat = x_be_flat.shape[0]

        gaussian_intensities = _apply_gaussian_apodization(
            x_be_flat, y_be_flat, sigma_x=sigma_val, sigma_y=sigma_val
        )
        user_rays_apodized = RealRays(
            x=x_be_flat,
            y=y_be_flat,
            z=be.zeros((num_rays_flat,)),
            L=be.zeros((num_rays_flat,)),
            M=be.zeros((num_rays_flat,)),
            N=be.ones((num_rays_flat,)),
            intensity=gaussian_intensities,
            wavelength=be.full((num_rays_flat,), 0.55),
        )
        irr_apodized = analysis.IncoherentIrradiance(
            optic_sys, res=res_val, user_initial_rays=user_rays_apodized
        )

        # Test cross-X plot at center
        fig, axes = irr_apodized.view(cross_section=("cross-x", res_val[0] // 2))
        assert fig is not None
        assert axes is not None
        assert isinstance(fig, Figure)
        assert len(axes) > 0

        plt.close(fig)
        # Test cross-Y plot at default middle slice
        fig, axes = irr_apodized.view(cross_section=("cross-y", None))
        assert fig is not None
        assert axes is not None
        assert isinstance(fig, Figure)
        assert len(axes) > 0

        plt.close(fig)
        # Test cross-X plot with normalize=False
        fig, axes = irr_apodized.view(
            cross_section=("cross-x", res_val[0] // 2), normalize=False
        )
        assert fig is not None
        assert axes is not None
        assert isinstance(fig, Figure)
        assert len(axes) > 0

        plt.close(fig)

    def test_irradiance_peak_irradiance(
        self, set_test_backend, test_system_irradiance_v1
    ):
        optic_sys = test_system_irradiance_v1
        irr = analysis.IncoherentIrradiance(optic_sys, num_rays=20, res=(10, 10))
        peaks = irr.peak_irradiance()
        assert len(peaks) == len(irr.fields)
        assert len(peaks[0]) == len(irr.wavelengths)
        assert be.to_numpy(peaks[0][0]) >= 0

    def test_detector_surface_no_aperture(self, set_test_backend):
        class TestSystemNoAperture(Optic):
            def __init__(self):
                super().__init__()
                self.surfaces.add(index=0, thickness=be.inf)
                self.surfaces.add(index=1, thickness=0, is_stop=True)
                self.surfaces.add(index=2, thickness=10)
                self.surfaces.add(index=3)  # Image plane, no aperture
                self.wavelengths.add(0.55)
                self.fields.set_type("angle")
                self.fields.add(y=0)
                self.set_aperture("EPD", 5.0)

        optic_no_ap = TestSystemNoAperture()
        with pytest.raises(
            ValueError, match="Detector surface has no physical aperture"
        ):
            analysis.IncoherentIrradiance(optic_no_ap, num_rays=5, res=(5, 5))

    @patch("builtins.print")
    def test_px_size_overrides_res_warning(
        self, mock_print, set_test_backend, test_system_irradiance_v1
    ):
        optic_sys = test_system_irradiance_v1
        # Detector aperture is 5mm x 5mm. px_size of (1.0, 1.0) should result in 5x5 pixels.
        # res of (10,10) should be ignored and a warning printed.
        irr = analysis.IncoherentIrradiance(
            optic_sys, res=(10, 10), px_size=(1.0, 1.0), num_rays=10
        )
        irr_map_be, x_edges, y_edges = irr.data[0][0]

        # Check that the effective resolution is 5x5
        assert irr_map_be.shape == (5, 5)
        assert len(x_edges) == 5 + 1  # x_edges has N+1 elements for N pixels
        assert len(y_edges) == 5 + 1

        # Check that the warning was printed
        mock_print.assert_any_call(
            "[IncoherentIrradiance] Warning: res parameter ignored - derived from px_size instead → (5,5) pixels"
        )
        fig, axes = irr.view()
        assert fig is not None
        assert axes is not None
        assert isinstance(fig, Figure)
        assert len(axes) > 0

        plt.close(fig)

    def test_irradiance_view_options(self, set_test_backend, test_system_irradiance_v1):
        optic_sys = test_system_irradiance_v1
        irr = analysis.IncoherentIrradiance(optic_sys, num_rays=10, res=(10, 10))

        # Test with different cmap and normalize=False
        fig, axes = irr.view(cmap="viridis", normalize=False)
        assert fig is not None
        assert axes is not None
        assert isinstance(fig, Figure)
        assert len(axes) > 0

        plt.close(fig)

        # Test with cross-section, normalize=True, and different cmap
        fig, axes = irr.view(cross_section=("cross-x", 0), normalize=True, cmap="magma")
        assert fig is not None
        assert axes is not None
        assert isinstance(fig, Figure)
        assert len(axes) > 0

        plt.close(fig)

    @patch("builtins.print")
    def test_irradiance_cross_section_invalid_slice(
        self, mock_print, set_test_backend, test_system_irradiance_v1
    ):
        optic_sys = test_system_irradiance_v1
        res_val = (5, 5)
        irr = analysis.IncoherentIrradiance(optic_sys, num_rays=10, res=res_val)

        # Invalid slice index for cross-x
        fig, axes = irr.view(
            cross_section=("cross-x", res_val[0] + 5)
        )  # Index out of bounds
        assert fig is not None
        assert axes is not None
        assert isinstance(fig, Figure)
        assert len(axes) > 0
        plt.close(fig)
        mock_print.assert_any_call(
            f"[IncoherentIrradiance] Warning: X-slice index {res_val[0] + 5} is out of bounds. Skipping."
        )

        # Invalid slice index for cross-y
        fig, axes = irr.view(
            cross_section=("cross-y", res_val[1] + 5)
        )  # Index out of bounds
        assert fig is not None
        assert axes is not None
        assert isinstance(fig, Figure)
        assert len(axes) > 0
        plt.close(fig)
        mock_print.assert_any_call(
            f"[IncoherentIrradiance] Warning: Y-slice index {res_val[1] + 5} is out of bounds. Skipping."
        )

        # Invalid cross_section_info format (not tuple)
        fig, axes = irr.view(cross_section="invalid")
        assert fig is not None
        assert axes is not None
        assert isinstance(fig, Figure)
        assert len(axes) > 0
        plt.close(fig)
        mock_print.assert_any_call(
            "[IncoherentIrradiance] Warning: Invalid cross_section type. Expected tuple. Defaulting to 2D plot."
        )
        # Invalid cross_section_info format (tuple wrong length)
        fig, axes = irr.view(cross_section=("cross-x",))
        assert fig is not None
        assert axes is not None
        assert isinstance(fig, Figure)
        assert len(axes) > 0
        plt.close(fig)
        mock_print.assert_any_call(
            "[IncoherentIrradiance] Warning: Invalid cross_section type. Expected tuple. Defaulting to 2D plot."
        )
        # Invalid cross_section_info format (tuple wrong types)
        fig, axes = irr.view(cross_section=(123, "cross-y"))
        assert fig is not None
        assert axes is not None
        assert isinstance(fig, Figure)
        assert len(axes) > 0
        plt.close(fig)
        mock_print.assert_any_call(
            "[IncoherentIrradiance] Warning: Invalid cross_section format. Expected ('cross-x' or 'cross-y', int). Defaulting to 2D plot."
        )

    def test_irradiance_view_no_data(
        self, capsys, set_test_backend, test_system_irradiance_v1
    ):
        irr = analysis.IncoherentIrradiance(
            test_system_irradiance_v1, num_rays=1, res=(2, 2)
        )  # Minimal rays to get some data
        irr.data = []  # Force no data
        res = irr.view()
        assert res is None  # Should return None if no data
        captured = capsys.readouterr()
        assert "No irradiance data to display." in captured.out

    def test_vmin_vmax_equal_case(self, set_test_backend, test_system_irradiance_v1):
        optic_sys = test_system_irradiance_v1
        num_rays = 10
        user_rays = RealRays(
            x=be.full((num_rays,), 0.0),
            y=be.full((num_rays,), 0.0),
            z=be.zeros((num_rays,)),
            L=be.zeros((num_rays,)),
            M=be.zeros((num_rays,)),
            N=be.ones((num_rays,)),
            intensity=be.ones((num_rays,)),
            wavelength=be.full((num_rays,), 0.55),
        )
        res_tuple = (5, 5)
        irr = analysis.IncoherentIrradiance(
            optic_sys, res=res_tuple, user_initial_rays=user_rays
        )
        dummy_edges = np.array(
            [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
        )  # numpy array for dummy edges

        # Case 1: All pixels have same non-zero irradiance
        irr.data = [[(be.full(res_tuple, 10.0), dummy_edges, dummy_edges)]]
        fig, axes = irr.view(normalize=False)  # Test the vmin=vmax branch in plotting
        assert fig is not None
        assert axes is not None
        assert isinstance(fig, Figure)
        assert len(axes) > 0
        plt.close(fig)

        # Case 2: All pixels are zero
        irr.data = [[(be.zeros(res_tuple), dummy_edges, dummy_edges)]]
        fig, axes = irr.view(normalize=False)
        assert fig is not None
        assert axes is not None
        assert isinstance(fig, Figure)
        assert len(axes) > 0
        plt.close(fig)

        # Case 3: Normalization active, vmin/vmax will be 0 and 1
        irr.data = [[(be.full(res_tuple, 10.0), dummy_edges, dummy_edges)]]
        fig, axes = irr.view(normalize=True)
        assert fig is not None
        assert axes is not None
        assert isinstance(fig, Figure)
        assert len(axes) > 0
        plt.close(fig)

    def test_peak_irradiance_empty_data(
        self, set_test_backend, test_system_irradiance_v1
    ):
        irr = analysis.IncoherentIrradiance(
            test_system_irradiance_v1, num_rays=1, res=(2, 2)
        )
        irr.data = []
        assert irr.peak_irradiance() == []

        irr.data = [[]]
        assert irr.peak_irradiance() == [[]]

        dummy_edges = be.array([0.0, 1.0])  # numpy array for dummy edges
        # Ensure float values for irradiance maps for consistent type with backend
        irr.data = [
            [
                (be.array([[1.0, 2.0], [3.0, 4.0]]), dummy_edges, dummy_edges),
                (be.array([[5.0, 6.0], [7.0, 8.0]]), dummy_edges, dummy_edges),
            ]
        ]
        peaks = irr.peak_irradiance()  # list of lists of backend scalars
        assert_allclose(
            peaks[0][0], be.array(4.0)
        )  # Compare backend scalar with backend scalar
        assert_allclose(peaks[0][1], be.array(8.0))

    def test_irradiance_autodiff(self, set_test_backend):
        if be.get_backend() != "torch":
            pytest.skip("Autodiff test only runs for torch backend")

        be.grad_mode.enable()
        # Create a simple system with a parameter that requires gradients
        optic_sys = Optic()
        # Make RADIUS a tensor that requires gradients, as changing it will
        # affect the final irradiance.
        radius_tensor = be.array(20.0)
        radius_tensor.requires_grad = True
        optic_sys.surfaces.add(index=0, thickness=be.inf)
        optic_sys.surfaces.add(
            index=1, thickness=7, radius=radius_tensor, is_stop=True, material="bk7"
        )
        optic_sys.surfaces.add(index=2, thickness=10)
        optic_sys.surfaces.add(index=3)
        detector_size = RectangularAperture(
            x_max=2.5, x_min=-2.5, y_max=2.5, y_min=-2.5
        )
        optic_sys.surfaces[-1].aperture = detector_size
        optic_sys.wavelengths.add(0.55)
        optic_sys.fields.set_type("angle")
        optic_sys.fields.add(y=0)
        optic_sys.set_aperture("EPD", 5.0)

        # Perform analysis
        irr_analysis = analysis.IncoherentIrradiance(
            optic_sys, num_rays=100, res=(10, 10)
        )
        irr_map, _, _ = irr_analysis.data[0][0]

        # Define a simple loss and backpropagate
        loss = be.sum(irr_map**2)
        loss.backward()

        grad = optic_sys.surfaces[1].geometry.radius.grad
        assert grad is not None
        assert be.to_numpy(grad) != 0


def test_incoherent_irradiance_initialization(
    set_test_backend, test_system_irradiance_v1
):
    optic = test_system_irradiance_v1
    irr = analysis.IncoherentIrradiance(
        optic,
        num_rays=10,
        res=(64, 64),
        px_size=(0.1, 0.1),
        detector_surface=-1,
        fields="all",
        wavelengths="all",
        distribution="random",
        user_initial_rays=None,
    )
    assert irr.optic == optic
    assert irr.num_rays == 10
    # npix_x/y get overridden by px_size if px_size implies a different resolution
    # Detector is 5mm wide. 0.1mm pixels -> 50 pixels.
    assert irr.npix_x == 50
    assert irr.npix_y == 50
    assert irr.px_size == (0.1, 0.1)
    assert irr.detector_surface == -1
    assert irr.fields == optic.fields.get_field_coords()
    assert [wp.value for wp in irr.wavelengths] == optic.wavelengths.get_wavelengths()
    assert irr.user_initial_rays is None
    assert len(irr.data) == len(optic.fields.get_field_coords())
    assert len(irr.data[0]) == len(optic.wavelengths.get_wavelengths())


def test_view_normalize_true_peak_zero(set_test_backend, test_system_irradiance_v1):
    optic = test_system_irradiance_v1
    irr = analysis.IncoherentIrradiance(optic, num_rays=1, res=(5, 5))
    dummy_edges = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])  # numpy array for dummy
    irr.data = [
        [(be.zeros((5, 5)), dummy_edges, dummy_edges)]
    ]  # All zero irradiance map

    fig, axes = irr.view(normalize=True)  # Should handle peak_val = 0
    assert fig is not None
    assert axes is not None
    assert isinstance(fig, Figure)
    assert len(axes) > 0
    plt.close(fig)


@patch("builtins.print")
def test_cross_section_plot_helper_out_of_bounds(
    mock_print, set_test_backend, test_system_irradiance_v1
):
    optic = test_system_irradiance_v1
    irr = analysis.IncoherentIrradiance(optic, num_rays=5, res=(5, 5))
    irr_map_be, x_edges, y_edges = irr.data[0][0]  # x_edges, y_edges are numpy arrays
    _, ax = plt.subplots()
    # Test cross-x out of bounds
    irr._plot_cross_section(
        ax, irr_map_be, x_edges, y_edges, "cross-x", 10, "Test", True
    )
    mock_print.assert_any_call(
        "[IncoherentIrradiance] Warning: X-slice index 10 is out of bounds. Skipping."
    )

    # Test cross-y out of bounds
    irr._plot_cross_section(
        ax, irr_map_be, x_edges, y_edges, "cross-y", 10, "Test", True
    )
    mock_print.assert_any_call(
        "[IncoherentIrradiance] Warning: Y-slice index 10 is out of bounds. Skipping."
    )

    # Test invalid axis type
    irr._plot_cross_section(
        ax, irr_map_be, x_edges, y_edges, "invalid-axis", 0, "Test", True
    )


class TestThroughFocusSpotDiagram:
    def test_init(self, set_test_backend, cooke_triplet):
        optic = cooke_triplet
        delta_focus = 0.05
        num_steps = 3
        num_rings = 4
        distribution = "random"
        coordinates = "global"

        tf_spot = analysis.ThroughFocusSpotDiagram(
            optic,
            delta_focus=delta_focus,
            num_steps=num_steps,
            num_rings=num_rings,
            distribution=distribution,
            coordinates=coordinates,
            fields="all",  # explicitly pass to ensure it's resolved
            wavelengths="all",  # explicitly pass to ensure it's resolved
        )

        assert tf_spot.delta_focus == delta_focus
        assert tf_spot.num_steps == num_steps
        assert tf_spot.num_rings == num_rings
        assert tf_spot.distribution == distribution
        assert tf_spot.coordinates == coordinates

        # Check that fields and wavelengths are resolved
        assert isinstance(tf_spot.fields, list)
        assert len(tf_spot.fields) > 0
        assert tf_spot.fields != "all"
        assert isinstance(tf_spot.wavelengths, list)
        assert len(tf_spot.wavelengths) > 0
        assert tf_spot.wavelengths != "all"

        expected_results_len = 2 * num_steps + 1
        assert len(tf_spot.results) == expected_results_len

        # Check structure of one result item
        result_item = tf_spot.results[0]  # First focal step
        assert isinstance(result_item, dict)
        assert len(result_item.keys()) == 1

        # Key should be the delta_focus value for that step
        focus_key = list(result_item.keys())[0]
        expected_focus_key = -num_steps * delta_focus
        assert_allclose(focus_key, expected_focus_key)

        rms_values_list = list(result_item.values())[0]
        assert isinstance(rms_values_list, list)
        assert len(rms_values_list) == len(tf_spot.fields)
        for rms_val in rms_values_list:
            # RMS value should be a float or a backend scalar that can be converted
            assert isinstance(be.to_numpy(rms_val).item(), float)

    def test_image_surface_z_restoration(self, set_test_backend, cooke_triplet):
        optic = cooke_triplet
        original_z = optic.image_surface.geometry.cs.z

        # Ensure original_z is a concrete value if it's a backend tensor
        if hasattr(original_z, "item"):
            original_z_val = original_z.item()
        else:
            original_z_val = float(original_z)

        tf_spot = analysis.ThroughFocusSpotDiagram(optic, delta_focus=0.05, num_steps=1)

        current_z = optic.image_surface.geometry.cs.z
        if hasattr(current_z, "item"):
            current_z_val = current_z.item()
        else:
            current_z_val = float(current_z)

        assert_allclose(current_z_val, original_z_val)

    def test_analysis_results_content(self, set_test_backend, cooke_triplet):
        optic = cooke_triplet
        delta_focus = 0.05
        num_steps = 1  # Results for -0.05, 0, +0.05

        tf_spot = analysis.ThroughFocusSpotDiagram(
            optic, delta_focus=delta_focus, num_steps=num_steps
        )

        assert len(tf_spot.results) == 3

        # Check results for nominal focus (delta_focus = 0)
        nominal_results_dict = None
        for res_dict in tf_spot.results:
            if be.isclose(list(res_dict.keys())[0], 0.0):
                nominal_results_dict = res_dict
                break

        assert nominal_results_dict is not None, "Nominal focus results not found."
        rms_values_at_nominal = list(nominal_results_dict.values())[0]

        # Compare with direct SpotDiagram calculation
        spot_direct = analysis.SpotDiagram(optic)  # Optic z should be at nominal here
        rms_direct_all_wl = spot_direct.rms_spot_radius()
        primary_wl_idx = optic.wavelengths.primary_index

        expected_rms_at_nominal = []
        for field_idx in range(len(tf_spot.fields)):
            expected_rms_at_nominal.append(rms_direct_all_wl[field_idx][primary_wl_idx])

        for i in range(len(tf_spot.fields)):
            assert_allclose(
                be.to_numpy(rms_values_at_nominal[i]),
                be.to_numpy(expected_rms_at_nominal[i]),
            )

        # Check other focal planes for plausibility (positive RMS)
        for i, res_dict in enumerate(tf_spot.results):
            # Skip nominal as it's already checked in detail
            if be.isclose(list(res_dict.keys())[0], 0.0):
                continue

            rms_list = list(res_dict.values())[0]
            current_df = list(res_dict.keys())[0]
            # print(f"Checking delta_f: {current_df}, RMS list: {rms_list}") # For debugging if needed
            for rms_val in rms_list:
                rms_float = be.to_numpy(rms_val).item()
                assert rms_float >= 0.0, (
                    f"RMS value {rms_float} is negative for delta_focus {current_df}"
                )
                assert not be.isnan(rms_val), (
                    f"RMS value is NaN for delta_focus {current_df}"
                )

    @patch("builtins.print")
    def test_view_method(self, mock_print, set_test_backend, cooke_triplet):
        optic = cooke_triplet
        tf_spot = analysis.ThroughFocusSpotDiagram(optic, delta_focus=0.1, num_steps=2)
        tf_spot.view()

        mock_print.assert_called()

        # Check for some expected output patterns
        # Get all calls to print in a single list of strings
        print_calls = [args[0] for args, kwargs in mock_print.call_args_list]

        assert any("Through-Focus Spot Diagram Results" in call for call in print_calls)
        assert any("Delta Focus:" in call for call in print_calls)
        assert any("Field (" in call for call in print_calls)

        # Check that the number of "Delta Focus:" lines matches num_steps
        delta_focus_lines = [call for call in print_calls if "Delta Focus:" in call]
        assert len(delta_focus_lines) == (2 * tf_spot.num_steps + 1)

        # Check that the number of "Field (" lines matches num_fields * num_delta_focus_steps
        field_lines = [call for call in print_calls if "Field (" in call]
        expected_field_lines = len(tf_spot.fields) * (2 * tf_spot.num_steps + 1)
        assert len(field_lines) == expected_field_lines


class TestThroughFocusSpotDiagram:
    @pytest.fixture
    def tf_spot(self, set_test_backend):
        """Creates a ThroughFocusSpotDiagram instance after setting backend."""
        optic = CookeTriplet()  # Construct after backend is set
        return ThroughFocusSpotDiagram(
            optic,
            delta_focus=0.05,
            num_steps=3,
            num_rings=3,
            fields="all",
            wavelengths="all",
            coordinates="local",
        )

    def test_init_valid(self, set_test_backend):
        optic = CookeTriplet()
        tf = ThroughFocusSpotDiagram(optic)
        assert tf.coordinates == "local"
        assert tf.distribution == "hexapolar"

    def test_init_invalid_coordinates(self, set_test_backend):
        optic = CookeTriplet()
        with pytest.raises(ValueError, match="Coordinates must be 'global' or 'local'"):
            ThroughFocusSpotDiagram(optic, coordinates="invalid")

    def test_perform_analysis_at_focus_returns_data(self, tf_spot):
        data = tf_spot._perform_analysis_at_focus()
        assert isinstance(data, list)
        assert all(isinstance(item, list) for item in data)
        assert all(hasattr(spot, "x") for field in data for spot in field)

    def test_validate_view_prerequisites_failure_empty_results(self, tf_spot):
        tf_spot.results = []
        assert not tf_spot._validate_view_prerequisites()

    def test_validate_view_prerequisites_failure_empty_fields(self, tf_spot):
        tf_spot.results = [[[]]]
        tf_spot.fields = []
        assert not tf_spot._validate_view_prerequisites()

    def test_validate_view_prerequisites_success(self, tf_spot):
        assert tf_spot._validate_view_prerequisites()

    def test_get_plot_axis_labels_orientation_0(self, tf_spot):
        x_label, y_label = tf_spot._get_plot_axis_labels()
        assert x_label in ("X (mm)", "U (mm)")
        assert y_label in ("Y (mm)", "V (mm)")

    def test_get_spot_centroid_and_axis_limit(self, tf_spot):
        data = tf_spot.results[0][0]
        cx, cy = tf_spot._get_spot_centroid(data)
        assert isinstance(cx, float)
        assert isinstance(cy, float)
        limit = tf_spot._compute_global_axis_limit(buffer=1.05)
        assert isinstance(limit, float)
        assert limit > 0

    def test_view_full(self, tf_spot):
        fig, axes = tf_spot.view()
        assert fig is not None
        assert axes is not None
        assert isinstance(fig, Figure)
        assert len(axes) > 0
        assert all(isinstance(ax, Axes) for ax in axes)
        plt.close(fig)

    def test_view_with_all_zero_intensities(self, tf_spot):
        # Zero all intensities manually
        for defocus_data in tf_spot.results:
            for field_data in defocus_data:
                for spot in field_data:
                    spot.intensity = spot.intensity * 0
        fig, axes = tf_spot.view()

        assert fig is not None
        assert axes is not None
        assert isinstance(fig, Figure)
        assert len(axes) > 0
        assert all(isinstance(ax, Axes) for ax in axes)
        plt.close(fig)


def read_zmx_file(file_path, skip_lines, cols=(0, 1)):
    import os

    if not os.path.exists(file_path):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        if file_path.startswith("tests/") or file_path.startswith("tests\\"):
            candidate = os.path.join(this_dir, file_path[6:])
        else:
            candidate = os.path.join(this_dir, file_path)
        if os.path.exists(candidate):
            file_path = candidate

    try:
        data = np.loadtxt(
            file_path, skiprows=skip_lines, usecols=cols, encoding="utf-16"
        )
        if data.ndim == 1 and len(cols) == 2:
            data = data.reshape(1, -1)
        if data.shape[0] == 0:
            return None, None
        return data[:, 0], data[:, 1]
    except Exception:
        return None, None


@pytest.fixture
def extended_source():
    """Fixture to provide an instance of SMFSource."""
    # Using default behavior: divergence calculated from MFD and Wavelength
    # This matches the diffraction-limited physics of a perfect Gaussian beam.
    return SMFSource(
        mfd_um=10.4,
        wavelength_um=1.55,
        total_power=1.0,
        # divergence_deg_1e2 is calculated automatically
    )


@pytest.fixture
def system_1():
    """A simple placeholder test system for RadiantIntensity analysis."""

    class TestSystemForIntensity(Optic):
        def __init__(self):
            from optiland.materials import Material

            super().__init__(name="System 1 for Intensity")

            self.set_aperture(aperture_type="objectNA", value=0.095)

            self.fields.set_type(field_type="angle")
            self.fields.add(y=0)

            self.wavelengths.add(value=1.55, is_primary=True)

            H_K3 = Material("H-K3", reference="cdgm")

            apt_detector = RectangularAperture(-30, 30, -30, 30)

            self.surfaces.add(index=0, thickness=0)
            self.surfaces.add(index=1, thickness=0.01)
            self.surfaces.add(index=2, thickness=129.6554)
            self.surfaces.add(
                index=3, thickness=4, radius=131.9743, is_stop=True, material=H_K3
            )
            self.surfaces.add(index=4, thickness=10.0, radius=-131.9743)
            self.surfaces.add(index=5, aperture=apt_detector)

    return TestSystemForIntensity()


class TestRadiantIntensity:
    @pytest.mark.parametrize(
        "reference_surface_index, filename, max_angle",
        [
            (1, r"tests/zemax_files/sph_lens_coll_intensity_free_prop.txt", 12),
        ],
    )  # (-1, r"tests/zemax_files/sph_lens_coll_intensity_img.txt", 0.5),
    def test_intensity_output_values(
        self,
        set_test_backend,
        system_1,
        extended_source,
        filename,
        reference_surface_index,
        max_angle,
    ):
        rays_to_trace = extended_source.generate_rays(num_rays=1_000_000)

        analysis_angle_max = max_angle
        intensity_analysis = analysis.RadiantIntensity(
            optic=system_1,
            user_initial_rays=rays_to_trace,
            num_angular_bins_X=101,
            num_angular_bins_Y=101,
            angle_X_min=-analysis_angle_max,
            angle_X_max=analysis_angle_max,
            angle_Y_min=-analysis_angle_max,
            angle_Y_max=analysis_angle_max,
            reference_surface_index=reference_surface_index,
            use_absolute_units=True,
        )

        optiland_map_be, _, _, optiland_angle_X_be, optiland_angle_Y_be = (
            intensity_analysis.data[0][0]
        )
        optiland_map = be.to_numpy(optiland_map_be)
        optiland_angles_x = be.to_numpy(optiland_angle_X_be)
        optiland_angles_y = be.to_numpy(optiland_angle_Y_be)

        central_y_idx = np.argmin(np.abs(optiland_angles_y))
        optiland_cross_section = optiland_map[:, central_y_idx]

        zemax_angles, zemax_intensity = read_zmx_file(
            filename, skip_lines=1, cols=(0, 1)
        )

        print(f"Zemax angles: {zemax_angles}")
        print(f"Zemax intensity: {zemax_intensity}")

        assert zemax_intensity is not None, (
            f"Failed to load intensity data from Zemax file: {filename}"
        )
        assert zemax_angles is not None, (
            f"Failed to load angle data from Zemax file: {filename}"
        )

        # compare the two datasets
        # normalize both datasets to their peak for shape comparison
        if np.max(optiland_cross_section) > 0:
            optiland_cross_section /= np.max(optiland_cross_section)

        if np.max(zemax_intensity) > 0:
            zemax_intensity /= np.max(zemax_intensity)

        assert_allclose(optiland_cross_section, zemax_intensity, atol=0.1, rtol=0.1)

    def test_view_intensity(self, set_test_backend, system_1):
        """
        Tests the view() method of RadiantIntensity to ensure it runs without error.
        """
        radiant_intensity = analysis.RadiantIntensity(
            system_1, num_rays=50, num_angular_bins_X=11, num_angular_bins_Y=11
        )
        fig, axes = radiant_intensity.view()
        assert fig is not None
        assert axes is not None
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        plt.close(fig)

    def test_view_intensity_cross_section(self, set_test_backend, system_1):
        """
        Tests the cross-section plotting in RadiantIntensity.view().
        """
        radiant_intensity = analysis.RadiantIntensity(
            system_1, num_rays=50, num_angular_bins_X=11, num_angular_bins_Y=11
        )
        fig, axes = radiant_intensity.view(cross_section=("cross-x", None))
        assert fig is not None
        assert axes is not None
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        plt.close(fig)

    def test_view_intensity_normalize(self, set_test_backend, system_1):
        """
        Tests normalized plotting in RadiantIntensity.view().
        """
        radiant_intensity = analysis.RadiantIntensity(
            system_1,
            num_rays=50,
            num_angular_bins_X=11,
            num_angular_bins_Y=11,
            use_absolute_units=True,
        )
        # Test with normalize=True (should override use_absolute_units)
        fig, axes = radiant_intensity.view(normalize=True)
        plt.close(fig)

        # Test with normalize=False
        fig, axes = radiant_intensity.view(normalize=False)
        plt.close(fig)

    def test_intensity_autodiff(self, set_test_backend):
        if be.get_backend() != "torch":
            pytest.skip("Autodiff test only runs for torch backend")

        be.grad_mode.enable()

        optic_sys = Optic()

        radius_tensor = be.array(20.0)
        radius_tensor.requires_grad = True
        optic_sys.surfaces.add(index=0, thickness=be.inf)
        optic_sys.surfaces.add(
            index=1, thickness=7, radius=radius_tensor, is_stop=True, material="bk7"
        )
        optic_sys.surfaces.add(index=2, thickness=10)
        optic_sys.surfaces.add(index=3)
        detector_size = RectangularAperture(
            x_max=2.5, x_min=-2.5, y_max=2.5, y_min=-2.5
        )
        optic_sys.surfaces[-1].aperture = detector_size
        optic_sys.wavelengths.add(0.55)
        optic_sys.fields.set_type("angle")
        optic_sys.fields.add(y=0)
        optic_sys.set_aperture("EPD", 5.0)

        user_rays = _create_square_grid_rays(10, -2.5, 2.5)

        int_analysis = analysis.RadiantIntensity(
            optic=optic_sys,
            user_initial_rays=user_rays,
            num_angular_bins_X=11,
            num_angular_bins_Y=11,
            angle_X_min=-5,
            angle_X_max=5,
            angle_Y_min=-5,
            angle_Y_max=5,
            reference_surface_index=-1,
        )
        int_map, _, _, _, _ = int_analysis.data[0][0]

        # Define a simple loss and backpropagate
        loss = be.sum(int_map**2)
        loss.backward()

        grad = optic_sys.surfaces[1].geometry.radius.grad
        assert grad is not None
        assert be.to_numpy(grad) != 0


class TestCookeTripletBestFitRayFan:
    """Test suite for the BestFitRayFan analysis class using a real optical system."""

    def test_initialization(self, set_test_backend):
        """
        Tests that the BestFitRayFan class initializes correctly,
        setting its own attributes and calling its parent's __init__.
        """
        cooke_triplet = CookeTriplet()
        # Test with default parameters
        fan = analysis.BestFitRayFan(cooke_triplet)
        assert fan.num_rays_for_fit == 15
        assert fan.num_points == 257  # 256 becomes 257 to be odd
        assert fan.optic == cooke_triplet

        # Test with custom parameters
        fan_custom = analysis.BestFitRayFan(
            cooke_triplet, num_points=100, num_rays_for_fit=20
        )
        assert fan_custom.num_rays_for_fit == 20
        assert fan_custom.num_points == 101  # 100 becomes 101 to be odd

    def test_best_fit_ray_fan_data(self, set_test_backend):
        """
        Tests the BestFitRayFan class with a real optical system to ensure
        it runs without errors and that results is similar to standard RayFan.
        """
        cooke_triplet = CookeTriplet()
        num_points = 33

        # Perform both standard and best-fit analyses for comparison
        fan_best_fit = analysis.BestFitRayFan(
            cooke_triplet, num_points=num_points, num_rays_for_fit=5
        )
        fan_standard = analysis.RayFan(cooke_triplet, num_points=num_points)

        data_best_fit = fan_best_fit.data
        data_standard = fan_standard.data

        # Check top-level data structure
        assert "Px" in data_best_fit
        assert "Py" in data_best_fit
        assert len(data_best_fit["Px"]) == num_points
        assert len(data_best_fit["Py"]) == num_points

        # Check data for an off-axis field and primary wavelength
        field_key = (
            f"{cooke_triplet.fields.get_field_coords()[1]}"  # e.g., "(0.0, 0.7)"
        )
        wave_key = f"{cooke_triplet.primary_wavelength}"

        x_best_fit = data_best_fit[field_key][wave_key]["x"]
        x_standard = data_standard[field_key][wave_key]["x"]

        y_best_fit = data_best_fit[field_key][wave_key]["y"]
        y_standard = data_standard[field_key][wave_key]["y"]

        # Assert that the data is valid (not all NaN)
        assert not be.all(be.isnan(x_best_fit))
        assert not be.all(be.isnan(y_best_fit))

        # Crucially, assert that the best-fit data is similar to the standard ray fan data
        assert_allclose(x_best_fit, x_standard)
        assert_allclose(y_best_fit[0], -0.0268906245)
        assert_allclose(y_best_fit[4], -0.01640633)

    def test_view_best_fit_ray_fan(self, set_test_backend, cooke_triplet):
        ray_fan = analysis.BestFitRayFan(cooke_triplet)
        fig, axes = ray_fan.view()
        assert fig is not None
        assert len(axes) > 0
        assert isinstance(fig, Figure)
        assert all(isinstance(ax, Axes) for ax in axes)
        plt.close(fig)

    def test_remove_distortion_with_invalid_central_ray(self, set_test_backend):
        """
        Tests the _remove_distortion logic when the central ray is obscured.
        It verifies that the function falls back to using the mean of the
        valid rays as the reference offset.
        """
        # 1. Mock the RayFan instance and its essential attributes
        # We don't need a real optic, just the structure to call the method
        mock_optic = MagicMock()
        mock_optic.primary_wavelength = 0.55

        # Instantiate RayFan, it will be our object under test
        fan = analysis.BestFitRayFan(mock_optic, num_points=5)
        fan.fields = [FieldPoint(coord=(0.0, 0.7), weight=1.0)]
        fan.wavelengths = [WavelengthPoint(value=0.55, weight=1.0)]

        # 2. Manually construct the input data dictionary
        center_idx = fan.num_points // 2  # This will be index 2 for num_points=5

        # Create intensity arrays where the central ray has zero intensity
        intensity_with_obscuration = be.array([1.0, 1.0, 0.0, 1.0, 1.0])

        # Create coordinate arrays with known values
        x_coords = be.array(
            [10.0, 20.0, 999.0, 30.0, 40.0]
        )  # 999 is a dummy for the invalid ray
        y_coords = be.array(
            [-4.0, -2.0, 888.0, 2.0, 4.0]
        )  # 888 is a dummy for the invalid ray

        fan.data = {
            "(0.0, 0.7)": {
                "0.55": {
                    "x": be.copy(x_coords),
                    "y": be.copy(y_coords),
                    "intensity_x": be.copy(intensity_with_obscuration),
                    "intensity_y": be.copy(intensity_with_obscuration),
                }
            }
        }

        # 3. Call the method we want to test
        processed_data = fan._remove_distortion(fan.data)

        # 4. Define the expected outcome
        # The offset should be the mean of the valid points, ignoring the central one.
        # Expected x_offset = mean([10, 20, 30, 40]) = 25
        # Expected y_offset = mean([-4, -2, 2, 4]) = 0
        expected_x_offset = 25.0
        expected_y_offset = 0.0

        expected_x_final = x_coords - expected_x_offset
        expected_y_final = y_coords - expected_y_offset

        # 5. Assert that the data was processed correctly
        output_data = processed_data["(0.0, 0.7)"]["0.55"]
        assert_allclose(output_data["x"], expected_x_final)
        assert_allclose(output_data["y"], expected_y_final)
