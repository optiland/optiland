from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.aberrations import Aberrations
from optiland.optic import Optic
from optiland.samples.objectives import DoubleGauss
from optiland.samples.simple import Edmund_49_847, SingletStopSurf2

from .utils import assert_allclose


@pytest.fixture
def double_gauss():
    return DoubleGauss()


@pytest.fixture
def edmund_singlet():
    return Edmund_49_847()


@pytest.fixture
def singlet_stop_surf_two():
    return SingletStopSurf2()


@pytest.fixture
def simple_singlet():
    """Singlet with single field and wavelength"""
    lens = Optic()

    # add surfaces
    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    lens.surfaces.add(
        index=1,
        thickness=7,
        radius=19.93,
        is_stop=True,
        material="N-SF11",
    )
    lens.surfaces.add(index=2, thickness=21.48)
    lens.surfaces.add(index=3)

    # add aperture
    lens.set_aperture(aperture_type="EPD", value=20.0)

    # add field
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0)

    # add wavelength
    lens.wavelengths.add(value=0.55, is_primary=True)

    lens.updater.update_paraxial()
    lens.updater.image_solve()
    return lens


class TestDoubleGaussAberrations:
    def test_init(self, set_test_backend, double_gauss):
        aberrations = Aberrations(double_gauss)
        assert aberrations.optic == double_gauss

    def test_seidels(self, set_test_backend, double_gauss):
        S = double_gauss.aberrations.seidels()
        assert_allclose(
            S,
            [
                -0.003929457875534847,
                0.0003954597633218682,
                0.0034239055031729947,
                -0.016264753735226404,
                -0.046484107476755930,
            ],
        )

    def test_third_order(self, set_test_backend, double_gauss):
        data = double_gauss.aberrations.third_order()
        TSC, SC, CC, TCC, TAC, AC, TPC, PC, DC, TAchC, LchC, TchC, S = data
        assert_allclose(be.sum(TSC), -0.01964728937767421)
        assert_allclose(be.sum(SC), -0.19647289377674193)
        assert_allclose(be.sum(CC), 0.0019772988166093623)
        assert_allclose(be.sum(TCC), 0.005931896449828042)
        assert_allclose(be.sum(TAC), 0.017119527515864978)
        assert_allclose(be.sum(AC), 0.17119527515864985)
        assert_allclose(be.sum(TPC), -0.08132376867613199)
        assert_allclose(be.sum(PC), -0.8132376867613212)
        assert_allclose(be.sum(DC), -0.2324205373837797)
        assert_allclose(be.sum(TAchC), 0.0295705512988189)
        assert_allclose(be.sum(LchC), 0.2957055129881888)
        assert_allclose(be.sum(TchC), -0.01804376318260833)
        assert_allclose(S[0], -0.003929457875534847)
        assert_allclose(S[1], 0.0003954597633218682)
        assert_allclose(S[2], 0.0034239055031729947)
        assert_allclose(S[3], -0.016264753735226404)
        assert_allclose(S[4], -0.046484107476755930)

    def test_third_order_all_functions(self, set_test_backend, double_gauss):
        TSC = double_gauss.aberrations.TSC()
        SC = double_gauss.aberrations.SC()
        CC = double_gauss.aberrations.CC()
        TCC = double_gauss.aberrations.TCC()
        TAC = double_gauss.aberrations.TAC()
        AC = double_gauss.aberrations.AC()
        TPC = double_gauss.aberrations.TPC()
        PC = double_gauss.aberrations.PC()
        DC = double_gauss.aberrations.DC()
        TAchC = double_gauss.aberrations.TAchC()
        LchC = double_gauss.aberrations.LchC()
        TchC = double_gauss.aberrations.TchC()

        assert_allclose(be.sum(TSC), -0.01964728937767421)
        assert_allclose(be.sum(SC), -0.19647289377674193)
        assert_allclose(be.sum(CC), 0.0019772988166093623)
        assert_allclose(be.sum(TCC), 0.005931896449828042)
        assert_allclose(be.sum(TAC), 0.017119527515864978)
        assert_allclose(be.sum(AC), 0.17119527515864985)
        assert_allclose(be.sum(TPC), -0.08132376867613199)
        assert_allclose(be.sum(PC), -0.8132376867613212)
        assert_allclose(be.sum(DC), -0.2324205373837797)
        assert_allclose(be.sum(TAchC), 0.0295705512988189)
        assert_allclose(be.sum(LchC), 0.2957055129881888)
        assert_allclose(be.sum(TchC), -0.01804376318260833)


class TestEdmundSinglet:
    def test_init(self, set_test_backend, edmund_singlet):
        aberrations = Aberrations(edmund_singlet)
        assert aberrations.optic == edmund_singlet

    def test_seidels(self, set_test_backend, edmund_singlet):
        S = edmund_singlet.aberrations.seidels()
        assert_allclose(S[0], -1.730769175588275)
        assert_allclose(S[1], 0.14253720449059704)
        assert_allclose(S[2], -0.352955446544233)
        assert_allclose(S[3], -0.22120089147910937)
        assert_allclose(S[4], -0.020854909613614383)

    def test_third_order(self, set_test_backend, edmund_singlet):
        data = edmund_singlet.aberrations.third_order()
        TSC, SC, CC, TCC, TAC, AC, TPC, PC, DC, TAchC, LchC, TchC, S = data
        assert_allclose(be.sum(TSC), -1.7306053598822728)
        assert_allclose(be.sum(SC), -3.460883119362552)
        assert_allclose(be.sum(CC), 0.14252371347566878)
        assert_allclose(be.sum(TCC), 0.42757114042700617)
        assert_allclose(be.sum(TAC), -0.35292203963678487)
        assert_allclose(be.sum(AC), -0.7057772717825394)
        assert_allclose(be.sum(TPC), -0.2211799550187673)
        assert_allclose(be.sum(PC), -0.4423180410800838)
        assert_allclose(be.sum(DC), -0.020852935715656093)
        assert_allclose(be.sum(TAchC), -0.4947549112756089)
        assert_allclose(be.sum(LchC), -0.9894161663592405)
        assert_allclose(be.sum(TchC), 0.0)
        assert_allclose(S[0], -1.730769175588275)
        assert_allclose(S[1], 0.14253720449059704)
        assert_allclose(S[2], -0.352955446544233)
        assert_allclose(S[3], -0.22120089147910937)
        assert_allclose(S[4], -0.020854909613614383)

    def test_third_order_all_functions(self, set_test_backend, edmund_singlet):
        TSC = edmund_singlet.aberrations.TSC()
        SC = edmund_singlet.aberrations.SC()
        CC = edmund_singlet.aberrations.CC()
        TCC = edmund_singlet.aberrations.TCC()
        TAC = edmund_singlet.aberrations.TAC()
        AC = edmund_singlet.aberrations.AC()
        TPC = edmund_singlet.aberrations.TPC()
        PC = edmund_singlet.aberrations.PC()
        DC = edmund_singlet.aberrations.DC()
        TAchC = edmund_singlet.aberrations.TAchC()
        LchC = edmund_singlet.aberrations.LchC()
        TchC = edmund_singlet.aberrations.TchC()

        assert_allclose(be.sum(TSC), -1.7306053598822728)
        assert_allclose(be.sum(SC), -3.460883119362552)
        assert_allclose(be.sum(CC), 0.14252371347566878)
        assert_allclose(be.sum(TCC), 0.42757114042700617)
        assert_allclose(be.sum(TAC), -0.35292203963678487)
        assert_allclose(be.sum(AC), -0.7057772717825394)
        assert_allclose(be.sum(TPC), -0.2211799550187673)
        assert_allclose(be.sum(PC), -0.4423180410800838)
        assert_allclose(be.sum(DC), -0.020852935715656093)
        assert_allclose(be.sum(TAchC), -0.4947549112756089)
        assert_allclose(be.sum(LchC), -0.9894161663592405)
        assert_allclose(be.sum(TchC), 0.0)


class TestSingletStopTwo:
    def test_init(self, set_test_backend, singlet_stop_surf_two):
        aberrations = Aberrations(singlet_stop_surf_two)
        assert aberrations.optic == singlet_stop_surf_two

    def test_seidels(self, set_test_backend, singlet_stop_surf_two):
        S = singlet_stop_surf_two.aberrations.seidels()
        assert_allclose(S[0], -0.0326050034268675)
        assert_allclose(S[1], -0.0004386784359568394)
        assert_allclose(S[2], -0.01142479550599207)
        assert_allclose(S[3], -0.00692002070366785)
        assert_allclose(S[4], 0.0016544791002946339)

    def test_third_order(self, set_test_backend, singlet_stop_surf_two):
        data = singlet_stop_surf_two.aberrations.third_order()
        TSC, SC, CC, TCC, TAC, AC, TPC, PC, DC, TAchC, LchC, TchC, S = data
        assert_allclose(be.sum(TSC), -0.1323786953158293)
        assert_allclose(be.sum(SC), -1.074934343302707)
        assert_allclose(be.sum(CC), -0.0017810664901602852)
        assert_allclose(be.sum(TCC), -0.0053431994704808555)
        assert_allclose(be.sum(TAC), -0.0463855041980188)
        assert_allclose(be.sum(AC), -0.3766570698925757)
        assert_allclose(be.sum(TPC), -0.028095789481046744)
        assert_allclose(be.sum(PC), -0.22814191470407064)
        assert_allclose(be.sum(DC), 0.006717305986964978)
        assert_allclose(be.sum(TAchC), -0.2286190741226864)
        assert_allclose(be.sum(LchC), -1.8564202776151473)
        assert_allclose(be.sum(TchC), 0.011126795737552403)
        assert_allclose(S[0], -0.0326050034268675)
        assert_allclose(S[1], -0.0004386784359568394)
        assert_allclose(S[2], -0.01142479550599207)
        assert_allclose(S[3], -0.00692002070366785)
        assert_allclose(S[4], 0.0016544791002946339)

    def test_third_order_all_functions(self, set_test_backend, singlet_stop_surf_two):
        TSC = singlet_stop_surf_two.aberrations.TSC()
        SC = singlet_stop_surf_two.aberrations.SC()
        CC = singlet_stop_surf_two.aberrations.CC()
        TCC = singlet_stop_surf_two.aberrations.TCC()
        TAC = singlet_stop_surf_two.aberrations.TAC()
        AC = singlet_stop_surf_two.aberrations.AC()
        TPC = singlet_stop_surf_two.aberrations.TPC()
        PC = singlet_stop_surf_two.aberrations.PC()
        DC = singlet_stop_surf_two.aberrations.DC()
        TAchC = singlet_stop_surf_two.aberrations.TAchC()
        LchC = singlet_stop_surf_two.aberrations.LchC()
        TchC = singlet_stop_surf_two.aberrations.TchC()

        assert_allclose(be.sum(TSC), -0.1323786953158293)
        assert_allclose(be.sum(SC), -1.074934343302707)
        assert_allclose(be.sum(CC), -0.0017810664901602852)
        assert_allclose(be.sum(TCC), -0.0053431994704808555)
        assert_allclose(be.sum(TAC), -0.0463855041980188)
        assert_allclose(be.sum(AC), -0.3766570698925757)
        assert_allclose(be.sum(TPC), -0.028095789481046744)
        assert_allclose(be.sum(PC), -0.22814191470407064)
        assert_allclose(be.sum(DC), 0.006717305986964978)
        assert_allclose(be.sum(TAchC), -0.2286190741226864)
        assert_allclose(be.sum(LchC), -1.8564202776151473)
        assert_allclose(be.sum(TchC), 0.011126795737552403)


class TestSimpleSinglet:
    def test_on_axis_seidels_are_not_zero(self, set_test_backend, simple_singlet):
        """Test that Seidel coefficients are computed correctly for on-axis field"""
        S = simple_singlet.aberrations.seidels()
        # Spherical aberration should be non-zero
        assert not be.isclose(S[0], be.array(0.0))
        assert_allclose(S[0], -0.675281089)

        # Other Seidel coefficients are expected to be zero for on-axis field
        assert_allclose(S[1:], [0, 0, 0, 0], atol=1e-8)


class TestReflectiveSystemSeidels:
    """Regression test for upstream issue #347.

    For systems containing mirrors, the post-mirror refractive index must be
    treated as sign-flipped (Welford / Smith convention). Without this, every
    `(n[k] - n[k-1])` term in the Seidel formulas evaluates to zero across
    reflective surfaces, collapsing the first four Seidel sums to zero — even
    though mirrors do contribute aberrations.

    This also captures the addition of the conic constants to the Seidel
    aberrations via the `_get_conic_term` method.

    These values were compared to analytical solution derived by hand. Comparison
    to ZOS shows a sign difference (expected due to convention) and small
    deviation for spherical only. This is likely due to numerical methods used
    in ZOS, but further investigation required to confirm this.

    TODO: Add another test using a system with larger aberrations. Consider
    adding an objective test using an independent ray tracer.
    """

    def test_hubble_seidels_are_not_all_zero(self, set_test_backend):
        from optiland.samples.telescopes import HubbleTelescope

        S = HubbleTelescope().aberrations.seidels()
        assert_allclose(
            S,
            [
                -2.80642537e-06,
                3.37323248e-04,
                -1.45177194e-03,
                -1.28411858e-02,
                7.59541065e-04,
            ],
        )
