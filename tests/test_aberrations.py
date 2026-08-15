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


@pytest.fixture
def finite_conjugate_singlet():
    """Finite-conjugate singlet from issue #683."""
    lens = Optic(name="Convex/Plano")
    lens.surfaces.add(index=0, thickness=50.0)
    lens.surfaces.add(
        index=1,
        thickness=7.0,
        radius=20.0,
        is_stop=True,
        material="N-SF11",
    )
    lens.surfaces.add(index=2, thickness=43.0)
    lens.surfaces.add(index=3)

    lens.wavelengths.add(value=0.55, is_primary=True)
    lens.set_aperture(aperture_type="EPD", value=10.0)
    lens.fields.set_type(field_type="object_height")
    lens.fields.add(x=0, y=0)
    lens.fields.add(x=0, y=5)
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
        assert_allclose(be.sum(TAchC), -0.01400173301485554)
        assert_allclose(be.sum(LchC), -0.1400173301485556)
        assert_allclose(be.sum(TchC), 0.01227340687764573)
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
        assert_allclose(be.sum(TAchC), -0.01400173301485554)
        assert_allclose(be.sum(LchC), -0.1400173301485556)
        assert_allclose(be.sum(TchC), 0.01227340687764573)


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
        assert_allclose(be.sum(TAchC), -0.4609677086420541)
        assert_allclose(be.sum(LchC), -0.9218481569472585)
        assert_allclose(be.sum(TchC), -0.01674359274970534)
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
        assert_allclose(be.sum(TAchC), -0.4609677086420541)
        assert_allclose(be.sum(LchC), -0.9218481569472585)
        assert_allclose(be.sum(TchC), -0.01674359274970534)


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
        assert_allclose(be.sum(TAchC), -0.2234106023457104)
        assert_allclose(be.sum(LchC), -1.8141267259538556)
        assert_allclose(be.sum(TchC), 0.006577385169475487)
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
        assert_allclose(be.sum(TAchC), -0.2234106023457104)
        assert_allclose(be.sum(LchC), -1.8141267259538556)
        assert_allclose(be.sum(TchC), 0.006577385169475487)


class TestSimpleSinglet:
    def test_on_axis_seidels_are_not_zero(self, set_test_backend, simple_singlet):
        """Test that Seidel coefficients are computed correctly for on-axis field"""
        S = simple_singlet.aberrations.seidels()
        # Spherical aberration should be non-zero
        assert not be.isclose(S[0], be.array(0.0))
        assert_allclose(S[0], -0.675281089)

        # Other Seidel coefficients are expected to be zero for on-axis field
        assert_allclose(S[1:], [0, 0, 0, 0], atol=1e-8)


class TestFiniteConjugateAberrations:
    """Regression coverage for issue #683."""

    def test_chief_ray_is_finite(self, set_test_backend, finite_conjugate_singlet):
        y, u = finite_conjugate_singlet.paraxial.chief_ray()

        assert not be.any(~be.isfinite(y))
        assert not be.any(~be.isfinite(u))
        assert_allclose(y[0], -5.0)
        assert_allclose(y[1], 0.0, atol=1e-12)

    def test_third_order_coefficients_are_finite(
        self, set_test_backend, finite_conjugate_singlet
    ):
        coefficients = finite_conjugate_singlet.aberrations.third_order()

        for coefficient in coefficients:
            assert not be.any(~be.isfinite(coefficient))


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


class TestChromaticDispersionWavelengths:
    """The chromatic terms must differentiate over the system's own wavelengths.

    Regression: ``_precalculations`` evaluated the dispersion as
    ``n(0.4861) - n(0.6563)`` regardless of what the system defined. That is
    silently wrong for anything not specified on the visible F and C lines, and
    for an infrared design it is worse than wrong — most infrared glasses carry
    no refractive index data at 0.4861 µm to evaluate at all.

    ``CookeTriplet`` is specified at 0.48 / 0.55 / 0.65, close enough to the
    hardcoded lines to look plausible and far enough to matter: its ``LchC``
    moved by 7.9% when the hardcoding was removed.
    """

    def _sum(self, value):
        return float(be.to_numpy(value).reshape(-1).sum())

    def test_uses_defined_wavelengths_not_fc_lines(self, set_test_backend):
        """Shifting the wavelength set must move the chromatic terms."""
        from optiland.samples.objectives import CookeTriplet

        wide = CookeTriplet()
        narrow = CookeTriplet()
        narrow.wavelengths.wavelengths = []
        for value, primary in ((0.53, False), (0.55, True), (0.57, False)):
            narrow.add_wavelength(value=value, is_primary=primary)

        wide_lchc = self._sum(wide.aberrations.LchC())
        narrow_lchc = self._sum(narrow.aberrations.LchC())

        # A quarter of the spectral interval must not give the same answer.
        assert abs(narrow_lchc) < abs(wide_lchc) / 2, (
            f"LchC barely moved when the wavelength range shrank from "
            f"0.48-0.65 to 0.53-0.57: {wide_lchc} vs {narrow_lchc}. The "
            f"dispersion is probably being evaluated at fixed wavelengths "
            f"again."
        )

    def test_monochromatic_system_has_no_chromatic_aberration(self, set_test_backend):
        """One wavelength means no dispersion to difference across."""
        from optiland.samples.telescopes import HubbleTelescope

        hubble = HubbleTelescope()  # single wavelength, 0.55 µm
        assert self._sum(hubble.aberrations.LchC()) == pytest.approx(0.0, abs=1e-12)
        assert self._sum(hubble.aberrations.TchC()) == pytest.approx(0.0, abs=1e-12)

    def test_infrared_system_is_not_evaluated_on_visible_lines(self, set_test_backend):
        """A long-wave infrared system must use its own wavelengths.

        Germanium is essentially non-dispersive over 8-12 um in the shipped
        data, so the correct axial colour for this lens is zero. Evaluating the
        dispersion on the visible F and C lines instead borrows a refractive
        index swing the system never sees and reports 5.76 mm of axial colour
        -- with no error and no warning, which is the dangerous part.
        """
        from optiland.optic import Optic

        lens = Optic()
        lens.add_surface(index=0, thickness=be.inf)
        lens.add_surface(
            index=1, radius=100.0, thickness=6.0, material="germanium", is_stop=True
        )
        lens.add_surface(index=2, radius=-300.0, thickness=95.0)
        lens.add_surface(index=3)
        lens.set_aperture(aperture_type="EPD", value=20)
        lens.set_field_type(field_type="angle")
        lens.add_field(y=0)
        lens.add_field(y=2)
        lens.add_wavelength(value=8.0)
        lens.add_wavelength(value=10.0, is_primary=True)
        lens.add_wavelength(value=12.0)

        assert self._sum(lens.aberrations.LchC()) == pytest.approx(0.0, abs=1e-9)


class TestChromaticRayHeightIndex:
    """The chromatic terms must use the ray height *at* the surface.

    Regression: ``_TAchC_term`` and ``_TchC_term`` indexed the marginal ray as
    ``_ya[k - 1]`` while indexing the angles of incidence as ``_i[k - 1]``.
    Those two have different bases — ``_ya`` is the full per-surface array with
    ``_ya[0]`` at the object, while ``_i`` is a list built over
    ``range(1, N - 1)`` — so every surface was weighted by the ray height at the
    *previous* surface.

    Surface 1 came out right by accident whenever the object is at infinity,
    since the marginal ray is then parallel and ``_ya[0] == _ya[1]``, which is
    why the totals looked plausible rather than obviously broken.

    Rather than pin another golden value, these check the coefficients against
    what the aberrations physically are: the shift in focus and in chief-ray
    image height between the extreme wavelengths. Third-order theory is an
    approximation, so a few percent is expected; the pre-fix values were out by
    a factor of two, and in one case reported exactly zero for an aberration
    the system genuinely has.
    """

    def _sum(self, value):
        return float(be.to_numpy(value).reshape(-1).sum())

    def _at_wavelength(self, factory, wavelength):
        optic = factory()
        optic.wavelengths.wavelengths = []
        optic.wavelengths.add(value=wavelength, is_primary=True)
        return optic

    def test_axial_colour_matches_the_focus_shift(self, set_test_backend):
        """``LchC`` is the longitudinal shift of focus between F and C."""
        from optiland.samples.objectives import DoubleGauss

        short, long = 0.4861, 0.6563
        focus_short = float(
            be.to_numpy(self._at_wavelength(DoubleGauss, short).paraxial.F2()).reshape(
                -1
            )[-1]
        )
        focus_long = float(
            be.to_numpy(self._at_wavelength(DoubleGauss, long).paraxial.F2()).reshape(
                -1
            )[-1]
        )
        measured = focus_long - focus_short

        predicted = self._sum(DoubleGauss().aberrations.LchC())
        assert abs(predicted) == pytest.approx(abs(measured), rel=0.05), (
            f"LchC is {predicted} but the focus moves {measured} between "
            f"{short} and {long} um. Before the ray-height fix this ratio was "
            f"2.03."
        )

    def test_lateral_colour_matches_the_chief_ray_shift(self, set_test_backend):
        """``TchC`` is the transverse shift of the chief ray between F and C."""
        from optiland.samples.simple import Edmund_49_847

        short, long = 0.4861327, 0.6562725
        heights = []
        for wavelength in (short, long):
            _, _ = None, None
            chief_y, _ = self._at_wavelength(
                Edmund_49_847, wavelength
            ).paraxial.chief_ray()
            heights.append(float(be.to_numpy(chief_y).reshape(-1)[-1]))
        measured = heights[1] - heights[0]

        predicted = self._sum(Edmund_49_847().aberrations.TchC())
        assert abs(predicted) == pytest.approx(abs(measured), rel=0.05), (
            f"TchC is {predicted} but the chief ray moves {measured} between "
            f"{short} and {long} um. Before the ray-height fix this was "
            f"reported as exactly zero."
        )
