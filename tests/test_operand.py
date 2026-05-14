from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.optic import Optic
from optiland.optimization import operand
from optiland.optimization.operand import LensOperand, RayOperand
from optiland.samples.objectives import CookeTriplet
from optiland.samples.telescopes import HubbleTelescope

from .utils import assert_allclose


@pytest.fixture
def hubble():
    return HubbleTelescope()


@pytest.fixture
def cooke_triplet():
    return CookeTriplet()


class TestParaxialOperand:
    def test_f1(self, set_test_backend, hubble):
        assert_allclose(operand.ParaxialOperand.f1(hubble), -57600.08099840297)

    def test_f2(self, set_test_backend, hubble):
        assert_allclose(operand.ParaxialOperand.f2(hubble), 57600.080998403595)

    def test_F1(self, set_test_backend, hubble):
        assert_allclose(operand.ParaxialOperand.F1(hubble), -471891.9369498712)

    def test_F2(self, set_test_backend, hubble):
        assert_allclose(operand.ParaxialOperand.F2(hubble), 0.016864237529848742)

    def test_P1(self, set_test_backend, hubble):
        assert_allclose(operand.ParaxialOperand.P1(hubble), -414291.8559514682)

    def test_P2(self, set_test_backend, hubble):
        assert_allclose(operand.ParaxialOperand.P2(hubble), -57600.06413416607)

    def test_N1(self, set_test_backend, hubble):
        assert_allclose(operand.ParaxialOperand.N1(hubble), -414291.8559514676)

    def test_N2(self, set_test_backend, hubble):
        assert_allclose(operand.ParaxialOperand.N2(hubble), -57600.06413416545)

    def test_EPL(self, set_test_backend, hubble):
        assert_allclose(operand.ParaxialOperand.EPL(hubble), 4910.01016)

    def test_EPD(self, set_test_backend, hubble):
        assert_allclose(operand.ParaxialOperand.EPD(hubble), 2400)

    def test_XPL(self, set_test_backend, hubble):
        assert_allclose(operand.ParaxialOperand.XPL(hubble), -6958.363551642134)

    def test_XPD(self, set_test_backend, hubble):
        assert_allclose(operand.ParaxialOperand.XPD(hubble), 289.93210961932573)

    def test_magnification(self, set_test_backend, hubble):
        assert_allclose(operand.ParaxialOperand.magnification(hubble), 0.0)

    def test_magnification(self, set_test_backend, hubble):
        assert_allclose(operand.ParaxialOperand.total_track(hubble), 6365.20955)


class TestAberrationOperand:
    def test_seidel(self, set_test_backend, hubble):
        assert_allclose(
            operand.AberrationOperand.seidels(hubble, 0),
            0.0014539022855417389,
        )

    def test_TSC(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.TSC(hubble, 2), 0.0)

    def test_SC(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.SC(hubble, 2), 0.0)

    def test_CC(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.CC(hubble, 2), 0.0)

    def test_TCC(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.TCC(hubble, 1), 0.0)

    def test_TAC(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.TAC(hubble, 1), 0.0)

    def test_AC(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.AC(hubble, 1), 0.0)

    def test_TPC(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.TPC(hubble, 1), 0.0)

    def test_PC(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.PC(hubble, 1), 0.0)

    def test_DC(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.DC(hubble, 1), 0.0)

    def test_TAchC(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.TAchC(hubble, 1), 0.0)

    def test_LchC(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.LchC(hubble, 1), 0.0)

    def test_TchC(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.TchC(hubble, 1), 0.0)

    def test_TSC_sum(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.TSC_sum(hubble), 0.0)

    def test_SC_sum(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.SC_sum(hubble), 0.0)

    def test_CC_sum(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.CC_sum(hubble), 0.0)

    def test_TCC_sum(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.TCC_sum(hubble), 0.0)

    def test_TAC_sum(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.TAC_sum(hubble), 0.0)

    def test_AC_sum(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.AC_sum(hubble), 0.0)

    def test_TPC_sum(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.TPC_sum(hubble), 0.0)

    def test_PC_sum(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.PC_sum(hubble), 0.0)

    def test_DC_sum(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.DC_sum(hubble), 0.03489370392123652)

    def test_TAchC_sum(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.TAchC_sum(hubble), 0.0)

    def test_LchC_sum(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.LchC_sum(hubble), 0.0)

    def test_TchC_sum(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.TchC_sum(hubble), 0.0)


class TestRayOperand:
    def test_x_intercept(self, set_test_backend, hubble):
        data = {
            "optic": hubble,
            "surface_number": -1,
            "Hx": 1.0,
            "Hy": 0.0,
            "Px": 0.0,
            "Py": 0.0,
            "wavelength": 0.55,
        }
        assert_allclose(operand.RayOperand.x_intercept(**data), 150.42338010762842)

    def test_y_intercept(self, set_test_backend, hubble):
        data = {
            "optic": hubble,
            "surface_number": -1,
            "Hx": 0.0,
            "Hy": 1.0,
            "Px": 0.0,
            "Py": 0.0,
            "wavelength": 0.55,
        }
        assert_allclose(operand.RayOperand.y_intercept(**data), 150.42338010762842)

    def test_z_intercept(self, set_test_backend, hubble):
        data = {
            "optic": hubble,
            "surface_number": -1,
            "Hx": 0.0,
            "Hy": 1.0,
            "Px": 0.0,
            "Py": 0.0,
            "wavelength": 0.55,
        }
        assert_allclose(operand.RayOperand.z_intercept(**data), 6347.146837237045)

    def test_x_intercept_lcs(self, set_test_backend, hubble):
        data = {
            "optic": hubble,
            "surface_number": -1,
            "Hx": 1.0,
            "Hy": 0.0,
            "Px": 0.0,
            "Py": 0.0,
            "wavelength": 0.55,
        }
        assert_allclose(
            operand.RayOperand.x_intercept_lcs(**data),
            150.42338010762842,
        )

    def test_y_intercept_lcs(self, set_test_backend, hubble):
        data = {
            "optic": hubble,
            "surface_number": -1,
            "Hx": 0.0,
            "Hy": 1.0,
            "Px": 0.0,
            "Py": 0.0,
            "wavelength": 0.55,
        }
        assert_allclose(
            operand.RayOperand.y_intercept_lcs(**data),
            150.42338010762842,
        )

    def test_z_intercept_lcs(self, set_test_backend, hubble):
        data = {
            "optic": hubble,
            "surface_number": -1,
            "Hx": 0.0,
            "Hy": 1.0,
            "Px": 0.0,
            "Py": 0.0,
            "wavelength": 0.55,
        }
        assert_allclose(
            operand.RayOperand.z_intercept_lcs(**data),
            -18.062712762936826,
        )  # Because Hubble's image is curved, otherwise it would be 0

    def test_L(self, set_test_backend, hubble):
        data = {
            "optic": hubble,
            "surface_number": -1,
            "Hx": 0.0,
            "Hy": 1.0,
            "Px": 0.0,
            "Py": 0.0,
            "wavelength": 0.55,
        }
        assert_allclose(operand.RayOperand.L(**data), 0.0)

    def test_M(self, set_test_backend, hubble):
        data = {
            "optic": hubble,
            "surface_number": -1,
            "Hx": 0.0,
            "Hy": 1.0,
            "Px": 0.0,
            "Py": 0.0,
            "wavelength": 0.55,
        }
        assert_allclose(operand.RayOperand.M(**data), 0.021668820646914833)

    def test_N(self, set_test_backend, hubble):
        data = {
            "optic": hubble,
            "surface_number": -1,
            "Hx": 0.0,
            "Hy": 1.0,
            "Px": 0.0,
            "Py": 0.0,
            "wavelength": 0.55,
        }
        assert_allclose(operand.RayOperand.N(**data), 0.9997652035411982)

    def test_rms_spot_size(self, set_test_backend, hubble):
        data = {
            "optic": hubble,
            "surface_number": -1,
            "Hx": 0.0,
            "Hy": 1.0,
            "wavelength": 0.55,
            "num_rays": 100,
        }
        assert_allclose(
            operand.RayOperand.rms_spot_size(**data),
            0.025626727777956947,
        )

    def test_rms_spot_size_all_wave(self, set_test_backend, hubble):
        data = {
            "optic": hubble,
            "surface_number": -1,
            "Hx": 0.0,
            "Hy": 1.0,
            "wavelength": "all",
            "num_rays": 100,
        }
        assert_allclose(
            operand.RayOperand.rms_spot_size(**data),
            0.025626727777956947,
        )

    def test_opd_diff(self, set_test_backend, hubble):
        data = {
            "optic": hubble,
            "Hx": 0.0,
            "Hy": 1.0,
            "wavelength": 0.55,
            "num_rays": 3,
        }
        assert_allclose(operand.RayOperand.OPD_difference(**data),  0.22240780)

    def test_opd_diff_on_axis(self, set_test_backend, hubble):
        data = {
            "optic": hubble,
            "Hx": 0.0,
            "Hy": 0.0,
            "wavelength": 0.55,
            "num_rays": 3,
        }
        assert_allclose(
            operand.RayOperand.OPD_difference(**data),
             0.001328702368213,
        )

    def test_opd_diff_new_dist(self, set_test_backend, hubble):
        data = {
            "optic": hubble,
            "Hx": 0.0,
            "Hy": 1.0,
            "wavelength": 0.55,
            "num_rays": 100,
            "distribution": "hexapolar",
        }
        assert_allclose(operand.RayOperand.OPD_difference(**data), 0.2211995620762635)

    def create_test_optic(self):
        lens = Optic(name="TMA")
        lens.set_aperture(aperture_type="EPD", value=10)
        lens.fields.set_type(field_type="angle")
        lens.fields.add(y=0)
        lens.fields.add(y=+1.5)
        lens.fields.add(y=-1.5)
        lens.wavelengths.add(value=0.55, is_primary=True)

        lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        lens.surfaces.add(
            index=1,
            radius=-100,
            thickness=-20,
            conic=0,
            material="mirror",
            rx=be.radians(-15.0),
            is_stop=True,
        )
        lens.surfaces.add(
            index=2,
            radius=-100,
            thickness=+20,
            conic=0,
            material="mirror",
            rx=be.radians(-10.0),
            dy=-11.5,
        )
        lens.surfaces.add(
            index=3,
            radius=-100,
            thickness=-19,
            conic=0,
            material="mirror",
            rx=be.radians(-1.0),
            dy=-15,
        )
        lens.surfaces.add(index=4, dy=-19.3)
        return lens

    def test_clearance(self, set_test_backend):
        optic = self.create_test_optic()
        wavelength = 0.55

        dist1 = RayOperand.clearance(
            optic=optic,
            line_ray_surface_idx=1,
            line_ray_field_coords=(0.0, 0.0),
            line_ray_pupil_coords=(0.0, -1.0),
            point_ray_surface_idx=3,
            point_ray_field_coords=(0.0, 1.0),
            point_ray_pupil_coords=(0.0, 0.0),
            wavelength=wavelength,
        )
        assert_allclose(dist1, -7.412094834746042)

        dist2 = RayOperand.clearance(
            optic=optic,
            line_ray_surface_idx=1,
            line_ray_field_coords=(0.0, 0.0),
            line_ray_pupil_coords=(0.0, 0.0),
            point_ray_surface_idx=3,
            point_ray_field_coords=(0.0, 0.0),
            point_ray_pupil_coords=(0.0, 0.0),
            wavelength=wavelength,
        )
        assert_allclose(dist2, -13.065596389231768)

        dist3 = RayOperand.clearance(
            optic=optic,
            line_ray_surface_idx=1,
            line_ray_field_coords=(0.0, -1.0),
            line_ray_pupil_coords=(0.0, 1.0),
            point_ray_surface_idx=3,
            point_ray_field_coords=(0.0, 1.0),
            point_ray_pupil_coords=(0.0, 0.0),
            wavelength=wavelength,
        )
        assert_allclose(dist3, -15.730530102711754)

    def test_AOI(self, set_test_backend, cooke_triplet):
        """Test the angle of incidence operand using CookeTriplet."""

        # Test AOI on first surface (spherical)
        data_1 = {
            "optic": cooke_triplet,
            "surface_number": 1,
            "Hx": 0.0,
            "Hy": 0.5,
            "Px": 0.0,
            "Py": 0.0,
            "wavelength": 0.55,
        }
        aoi_1 = operand.RayOperand.AOI(**data_1)

        data_2 = {
            "optic": cooke_triplet,
            "surface_number": 7,
            "Hx": 0.0,
            "Hy": 1.0,  # Maximum field
            "Px": 0.0,
            "Py": 0.0,
            "wavelength": 0.55,
        }
        aoi_2 = operand.RayOperand.AOI(**data_2)

        data_3 = {
            "optic": cooke_triplet,
            "surface_number": 7,
            "Hx": 0.0,
            "Hy": 0.0,
            "Px": 0.0,
            "Py": 1.0,  # Marginal ray
            "wavelength": 0.55,
        }
        aoi_3 = operand.RayOperand.AOI(**data_3)

        assert_allclose(aoi_1, 4.751694988282298e000)
        assert_allclose(aoi_2, 1.898548831778711e001)
        assert_allclose(aoi_3, 5.764367266821943e000)

        assert aoi_1
        aoi_2
        aoi_3 > 0.0


class TestLensOperand:
    def setup_method(self):
        self.optic = Optic()
        self.optic.set_aperture(aperture_type="EPD", value=20.0)
        self.optic.surfaces.add(index=0, thickness=be.inf)
        self.optic.surfaces.add(
            index=1, thickness=10.0, is_stop=True
        )  # dummy, stop surface
        self.optic.wavelengths.add(value=0.550, is_primary=True)
        self.optic.fields.set_type(field_type="angle")
        self.optic.fields.add(y=0.0)

    def test_edge_thickness_simple_biconvex(self, set_test_backend):
        """
        Test the edge thickness calculation for a simple biconvex lens
        with identical semi-apertures.
        """

        self.optic.surfaces.add(index=2, radius=50.0, thickness=5.0, material="N-BK7")
        self.optic.surfaces.add(index=3, radius=-50.0, thickness=100.0, material="air")
        self.optic.surfaces.add(index=4)

        self.optic.updater.update_paraxial()

        sa1 = self.optic.surfaces[2].semi_aperture
        sa2 = self.optic.surfaces[3].semi_aperture
        semi_aperture = be.maximum(sa1, sa2)

        r1 = 50.0
        r2 = -50.0
        t = 5.0
        r_sq = semi_aperture**2

        sag1 = r_sq / (r1 * (1 + be.sqrt(1 - r_sq / r1**2)))
        sag2 = r_sq / (r2 * (1 + be.sqrt(1 - r_sq / r2**2)))
        expected_edge_thickness = t - sag1 + sag2

        edge_thickness = LensOperand.edge_thickness(self.optic, surface_number=2)
        assert_allclose(edge_thickness, expected_edge_thickness)

    def test_edge_thickness_auto_initializes_semi_apertures(self, set_test_backend):
        self.optic.surfaces.add(index=2, radius=50.0, thickness=5.0, material="N-BK7")
        self.optic.surfaces.add(index=3, radius=-50.0, thickness=100.0, material="air")
        self.optic.surfaces.add(index=4)

        # No explicit updater.update_paraxial() call on purpose.
        assert self.optic.surfaces[2].semi_aperture is None
        assert self.optic.surfaces[3].semi_aperture is None

        edge_thickness = LensOperand.edge_thickness(self.optic, surface_number=2)

        assert self.optic.surfaces[2].semi_aperture is not None
        assert self.optic.surfaces[3].semi_aperture is not None
        assert be.size(edge_thickness) == 1

    def test_edge_thickness_plano_concave(self, set_test_backend):
        self.optic.surfaces.add(index=2, radius=be.inf, thickness=2.0, material="N-BK7")
        self.optic.surfaces.add(index=3, radius=50.0, thickness=100.0, material="air")
        self.optic.surfaces.add(index=4)

        self.optic.updater.update_paraxial()

        r2 = 50.0
        t = 2.0
        semi_aperture = 10.0

        sag1 = 0.0
        sag2 = r2 - be.sqrt(r2**2 - semi_aperture**2)
        expected_edge_thickness = t - sag1 + sag2

        edge_thickness = LensOperand.edge_thickness(self.optic, surface_number=2)

        assert_allclose(edge_thickness, expected_edge_thickness)

    def test_edge_thickness_different_semi_apertures(self, set_test_backend):
        # add a lens with two surfaces having different semi apts
        self.optic.surfaces.add(index=2, radius=50.0, thickness=5.0, material="N-BK7")
        self.optic.surfaces.add(index=3, radius=-50.0, thickness=100.0, material="air")
        self.optic.surfaces.add(index=4)

        self.optic.updater.update_paraxial()
        self.optic.surfaces[2].semi_aperture = 10.0
        self.optic.surfaces[3].semi_aperture = 8.0

        # expected value calculation
        r1 = 50.0
        r2 = -50.0
        t = 5.0
        semi_aperture = 10.0

        sag1 = r1 - be.sqrt(r1**2 - semi_aperture**2)
        sag2 = r2 + be.sqrt(r2**2 - semi_aperture**2)
        expected_edge_thickness = t - sag1 + sag2

        edge_thickness = LensOperand.edge_thickness(self.optic, surface_number=2)

        assert_allclose(edge_thickness, expected_edge_thickness)

    def test_edge_thickness_aspheric_surface(self, set_test_backend):
        # add an aspheric surface to the lens
        coeffs = [1e-5]
        self.optic.surfaces.add(
            index=2,
            surface_type="even_asphere",
            radius=50.0,
            thickness=5.0,
            material="N-BK7",
            coefficients=coeffs,
        )
        self.optic.surfaces.add(index=3, radius=-50.0, thickness=100.0, material="air")
        self.optic.surfaces.add(index=4)

        self.optic.updater.update_paraxial()

        sa1 = self.optic.surfaces[2].semi_aperture
        sa2 = self.optic.surfaces[3].semi_aperture
        semi_aperture = be.maximum(sa1, sa2)

        r1 = 50.0
        r2 = -50.0
        t = 5.0

        r2_val = semi_aperture**2
        sag1_base = r2_val / (r1 * (1 + be.sqrt(1 - r2_val / r1**2)))
        sag1_asphere = coeffs[0] * r2_val
        sag1 = sag1_base + sag1_asphere

        sag2 = r2 + be.sqrt(r2**2 - semi_aperture**2)
        expected_edge_thickness = t - sag1 + sag2

        edge_thickness = LensOperand.edge_thickness(self.optic, surface_number=2)

        assert_allclose(edge_thickness, expected_edge_thickness)


class TestOperand:
    def test_get_value(self, set_test_backend, hubble):
        input_data = {"optic": hubble}
        op = operand.Operand(
            operand_type="f2",
            target=1,
            weight=1,
            input_data=input_data,
        )
        assert_allclose(op.value, 57600.080998403595)

    def test_invalid_operand(self, set_test_backend, hubble):
        input_data = {"optic": hubble}
        op = operand.Operand(
            operand_type="f3",
            target=1,
            weight=1,
            input_data=input_data,
        )
        with pytest.raises(ValueError):
            op.value

    def test_delta(self, set_test_backend, hubble):
        input_data = {"optic": hubble}
        op = operand.Operand(
            operand_type="f2",
            target=5000,
            weight=1,
            input_data=input_data,
        )
        assert_allclose(op.delta(), 52600.080998403595)

    def test_fun(self, set_test_backend, hubble):
        input_data = {"optic": hubble}
        op = operand.Operand(
            operand_type="f2",
            target=1e5,
            weight=1.5,
            input_data=input_data,
        )
        assert_allclose(op.fun(), -63599.87850239461)

    def test_reregister_operand(self, set_test_backend):
        try:
            operand.operand_registry._registry.pop("f3")
        except KeyError:
            pass

        operand.operand_registry.register("f3", lambda x: x)
        with pytest.raises(ValueError):
            operand.operand_registry.register("f3", lambda x: x)

    def test_contains(self, set_test_backend):
        assert "f1" in operand.operand_registry
        assert "invalid" not in operand.operand_registry

    def test_repr(self, set_test_backend):
        assert isinstance(repr(operand.operand_registry), str)

    def test_invalid_bounds(self, set_test_backend):
        with pytest.raises(ValueError):
            operand.Operand(operand_type="f1", min_val=1, max_val=0)

    def test_invalid_bounds_and_target(self, set_test_backend):
        with pytest.raises(ValueError):
            operand.Operand(operand_type="f1", min_val=1, max_val=0, target=1)

    def test_delta_ineq(self, set_test_backend):
        input_data = {"optic": HubbleTelescope()}
        op = operand.Operand(
            operand_type="f2",
            min_val=1e6,
            max_val=1e11,
            weight=1,
            input_data=input_data,
        )
        hubble = HubbleTelescope()
        f2 = hubble.paraxial.f2()

        # check on the lower bound
        assert_allclose(op.delta_ineq(), 1e6 - f2)

        op = operand.Operand(
            operand_type="f2",
            min_val=0,
            max_val=1,
            weight=1,
            input_data=input_data,
        )

        # check on the upper bound
        assert_allclose(op.delta_ineq(), f2 - 1)

    def test_delta_ineq_no_bounds(self, set_test_backend):
        input_data = {"optic": HubbleTelescope()}
        op = operand.Operand(operand_type="f2", weight=1, input_data=input_data)
        assert_allclose(op.delta_ineq(), 0)

    def test_delta_with_bounds(self, set_test_backend):
        input_data = {"optic": HubbleTelescope()}
        op = operand.Operand(
            operand_type="f2",
            min_val=1e6,
            max_val=1e11,
            weight=1,
            input_data=input_data,
        )
        hubble = HubbleTelescope()
        f2 = hubble.paraxial.f2()

        # check on the lower bound
        assert_allclose(op.delta(), 1e6 - f2)

        op = operand.Operand(
            operand_type="f2",
            min_val=0,
            max_val=1,
            weight=1,
            input_data=input_data,
        )

        # check on the upper bound
        assert_allclose(op.delta(), f2 - 1)

    def test_invalid_delta(self, set_test_backend):
        input_data = {"optic": HubbleTelescope()}
        op = operand.Operand(operand_type="f2", input_data=input_data)
        # manually remove target to make the operand invalid
        op.target = None
        with pytest.raises(ValueError):
            op.delta()


class TestOperandManager:
    def test_add(self, set_test_backend):
        manager = operand.OperandManager()
        manager.add("f1", 1)
        assert len(manager) == 1

    def test_clear(self, set_test_backend):
        manager = operand.OperandManager()
        manager.add("f1", 1)
        manager.clear()
        assert len(manager) == 0

    def test_iter(self, set_test_backend):
        manager = operand.OperandManager()
        manager.add("f1", 1)
        manager.add("f2", 1)
        for op in manager:
            assert isinstance(op, operand.Operand)

    def test_getitem(self, set_test_backend):
        manager = operand.OperandManager()
        manager.add("f1", 1)
        assert isinstance(manager[0], operand.Operand)

    def test_setitem(self, set_test_backend):
        manager = operand.OperandManager()
        manager.add("f1", 1)
        manager[0] = operand.Operand(
            operand_type="f2",
            target=1,
            weight=1,
            input_data={},
        )
        assert len(manager) == 1

    def test_len(self, set_test_backend):
        manager = operand.OperandManager()
        manager.add("f1", 1)
        manager.add("f2", 1)
        assert len(manager) == 2

    def test_getitem_error(self, set_test_backend):
        manager = operand.OperandManager()
        with pytest.raises(IndexError):
            manager[0]

    def test_setitem_invalid_type(self, set_test_backend):
        manager = operand.OperandManager()
        manager.add("f1", 1)
        with pytest.raises(ValueError):
            manager[0] = 1

    def test_delitem(self, set_test_backend):
        manager = operand.OperandManager()
        manager.add("f1", 1)
        del manager[0]
        assert len(manager) == 0
