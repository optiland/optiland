from __future__ import annotations

from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

import optiland.backend as be
from optiland import distribution
from optiland.optic import Optic
from optiland.samples.eyepieces import EyepieceErfle
from optiland.samples.objectives import CookeTriplet, DoubleGauss
from optiland.wavefront import OPD, OPDFan, Wavefront, WavefrontData, ZernikeOPD
from tests.utils import assert_allclose

matplotlib.use("Agg")  # use non-interactive backend for testing


class TestWavefront:
    @pytest.mark.parametrize("OpticClass", [CookeTriplet, DoubleGauss, EyepieceErfle])
    def test_wavefront_initialization(self, OpticClass, set_test_backend):
        optic = OpticClass()
        w = Wavefront(optic)
        assert w.num_rays == 12
        assert [fp.coord for fp in w.fields] == optic.fields.get_field_coords()
        assert [wp.value for wp in w.wavelengths] == optic.wavelengths.get_wavelengths()
        assert isinstance(w.distribution, distribution.HexagonalDistribution)

    def test_wavefront_init_custom(self, set_test_backend):
        optic = DoubleGauss()
        w = Wavefront(
            optic,
            num_rays=100,
            distribution="random",
            wavelengths="primary",
        )
        assert w.num_rays == 100
        assert isinstance(w.distribution, distribution.RandomDistribution)
        assert [wp.value for wp in w.wavelengths] == [optic.primary_wavelength]

    def test_generate_data(self, set_test_backend):
        optic = EyepieceErfle()
        w = Wavefront(optic)
        assert isinstance(w.data, dict)
        assert isinstance(w.data[((0.0, 0.0), 0.5876)], WavefrontData)
        assert isinstance(w.data[((0.0, 0.0), 0.5876)].intensity, be.ndarray)
        assert isinstance(w.data[((0.0, 0.7), 0.5876)].opd, be.ndarray)
        assert isinstance(w.data[((0.0, 0.0), 0.5876)].pupil_x, be.ndarray)
        assert (
            be.size(w.data[((0.0, 1.0), 0.6563)].opd) == 469
        )  # num points in the pupil


class TestOPDFan:
    def test_opd_fan_initialization(self, set_test_backend):
        optic = DoubleGauss()
        opd_fan = OPDFan(optic)
        assert opd_fan.num_rays == 100
        assert [fp.coord for fp in opd_fan.fields] == optic.fields.get_field_coords()
        assert [
            wp.value for wp in opd_fan.wavelengths
        ] == optic.wavelengths.get_wavelengths()
        assert isinstance(opd_fan.distribution, distribution.CrossDistribution)
        arr = be.linspace(-1, 1, opd_fan.num_rays)
        assert be.all(opd_fan.pupil_coord == arr)

    @patch("matplotlib.pyplot.show")
    def test_opd_fan_view(self, moch_show, set_test_backend):
        optic = DoubleGauss()
        opd_fan = OPDFan(optic)
        fig, ax = opd_fan.view()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_opd_fan_view_large(self, moch_show, set_test_backend):
        optic = DoubleGauss()
        opd_fan = OPDFan(optic)
        fig, ax = opd_fan.view(figsize=(20, 20))
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestOPD:
    def test_opd_initialization(self, set_test_backend):
        optic = EyepieceErfle()
        opd = OPD(optic, (0, 1), 0.55)
        assert opd.num_rays == 15
        assert [fp.coord for fp in opd.fields] == [(0, 1)]
        assert [wp.value for wp in opd.wavelengths] == [0.55]
        assert isinstance(opd.distribution, distribution.HexagonalDistribution)

    @patch("matplotlib.pyplot.show")
    def test_opd_view(self, moch_show, set_test_backend):
        optic = DoubleGauss()
        opd = OPD(optic, (0, 1), 0.55)
        fig, ax = opd.view()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_opd_view_large(self, moch_show, set_test_backend):
        optic = DoubleGauss()
        opd = OPD(optic, (0, 1), 0.55)
        fig, ax = opd.view(figsize=(20, 20))
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_opd_view_3d(self, moch_show, set_test_backend):
        optic = DoubleGauss()
        opd = OPD(optic, (0, 1), 0.55)
        fig, ax = opd.view(projection="3d")
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes3D)
        plt.close(fig)

    def test_old_invalid_projection(self, set_test_backend):
        optic = EyepieceErfle()
        opd = OPD(optic, (0, 1), 0.55)
        with pytest.raises(ValueError):
            opd.view(projection="invalid")

    def test_opd_rms(self, set_test_backend):
        optic = CookeTriplet()
        opd = OPD(optic, (0, 1), 0.55)
        rms = opd.rms()
        assert_allclose(rms, 0.9709788038168692)


class TestZernikeOPD:
    def test_zernike_opd_initialization(self, set_test_backend):
        optic = DoubleGauss()
        zernike_opd = ZernikeOPD(optic, (0, 1), 0.55)
        assert zernike_opd.num_rays == 15
        assert [fp.coord for fp in zernike_opd.fields] == [(0, 1)]
        assert [wp.value for wp in zernike_opd.wavelengths] == [0.55]
        assert isinstance(zernike_opd.distribution, distribution.HexagonalDistribution)
        assert be.allclose(zernike_opd.x, zernike_opd.distribution.x)
        assert be.allclose(zernike_opd.y, zernike_opd.distribution.y)
        assert be.allclose(zernike_opd.z, zernike_opd.data[((0, 1), 0.55)].opd)
        assert zernike_opd.zernike_type == "fringe"
        assert zernike_opd.num_terms == 37

    @patch("matplotlib.pyplot.show")
    def test_zernike_opd_view(self, moch_show, set_test_backend):
        optic = DoubleGauss()
        zernike_opd = ZernikeOPD(optic, (0, 1), 0.55)
        fig, ax = zernike_opd.view()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_zernike_opd_view_large(self, moch_show, set_test_backend):
        optic = DoubleGauss()
        zernike_opd = ZernikeOPD(optic, (0, 1), 0.55)
        fig, ax = zernike_opd.view(figsize=(20, 20))
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_zernike_opd_view_3d(self, moch_show, set_test_backend):
        optic = DoubleGauss()
        zernike_opd = ZernikeOPD(optic, (0, 1), 0.55)
        fig, ax = zernike_opd.view(projection="3d")
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes3D)
        plt.close(fig)

    def test_zernike_opd_rms(self, set_test_backend):
        optic = CookeTriplet()
        zernike_opd = ZernikeOPD(optic, (0, 1), 0.55)
        rms = zernike_opd.rms()
        assert_allclose(rms, 0.9709788038168692)

    def test_zernike_opd_fit(self, set_test_backend):
        optic = CookeTriplet()
        zernike_opd = ZernikeOPD(optic, (0, 1), 0.55)
        c = zernike_opd.zernike.coeffs
        assert_allclose(c[0], 0.8430890395012354)
        assert_allclose(c[1], 6.863699034904449e-13)
        assert_allclose(c[2], 0.14504379704525455)
        assert_allclose(c[6], -1.160298338689596e-13)
        assert_allclose(c[24], -0.0007283668376039182)

    def test_zernike_xy_symmetry(self, set_test_backend):
        optic = CookeTriplet()
        zernike_opd0 = ZernikeOPD(optic, (0, 1), 0.55)
        c0 = zernike_opd0.zernike.coeffs

        # swap x and y fields
        optic.fields.fields[0].x = 0
        optic.fields.fields[0].y = 0
        optic.fields.fields[1].x = 14
        optic.fields.fields[1].y = 0
        optic.fields.fields[2].x = 20
        optic.fields.fields[2].y = 0

        # run at max y field (should be the same field)
        zernike_opd1 = ZernikeOPD(optic, (0, 1), 0.55)
        c1 = zernike_opd1.zernike.coeffs
        assert be.allclose(c0, c1)

    def test_zernike_xy_axis_swap(self, set_test_backend):
        optic = CookeTriplet()
        zernike_opd0 = ZernikeOPD(optic, (0, 1), 0.55)
        c0 = zernike_opd0.zernike.coeffs

        # swap x and y fields
        optic.fields.fields[0].x = 0
        optic.fields.fields[0].y = 0
        optic.fields.fields[1].x = 0
        optic.fields.fields[1].y = 14
        optic.fields.fields[2].x = 0
        optic.fields.fields[2].y = 20

        # run at max x field
        zernike_opd1 = ZernikeOPD(optic, (1, 0), 0.55)
        c1 = zernike_opd1.zernike.coeffs

        # x and y tilts swapped
        assert be.isclose(c0[1], c1[2])
        assert be.isclose(c0[2], c1[1])


def test_finite_conjugate_angle_field_opd(set_test_backend):
    """
    Test that OPD for finite conjugate systems with AngleField is correctly
    calculated without artificial tilt.

    This was added after a bug was found where the OPD was calculated with an
    artificial tilt for finite conjugate systems with AngleField.
    """
    # Create a simple finite conjugate system
    optic = Optic()
    optic.surfaces.add(index=0, thickness=100.0)  # Object to lens
    optic.surfaces.add(
        index=1, radius=50.0, thickness=5.0, material="BK7", is_stop=True
    )
    optic.surfaces.add(index=2, radius=-50.0, thickness=95.0)  # lens to image

    optic.set_aperture("EPD", 10.0)
    optic.wavelengths.add(0.55, is_primary=True)

    # 1. Set field type to angle
    optic.fields.set_type("angle")
    optic.fields.add(y=5.0)  # 5 degrees

    opd_angle = OPD(
        optic,
        field=(0, 1),
        wavelength="primary",
        num_rays=10,
        distribution="line_y",
        remove_tilt=False,
    )
    data_angle = opd_angle.get_data((0, 1), optic.primary_wavelength)

    # 2. Set field type to object height
    # EPL is at surface 1 (z=0). Object is at z=-100.
    EPL = optic.paraxial.EPL()
    obj_z = optic.surfaces.positions[0].item()
    dist = EPL - obj_z
    h_obj = dist * be.tan(be.radians(5.0))

    optic2 = Optic()
    optic2.surfaces.add(index=0, thickness=100.0)
    optic2.surfaces.add(
        index=1, radius=50.0, thickness=5.0, material="BK7", is_stop=True
    )
    optic2.surfaces.add(index=2, radius=-50.0, thickness=95.0)
    optic2.set_aperture("EPD", 10.0)
    optic2.wavelengths.add(0.55, is_primary=True)

    optic2.fields.set_type("object_height")
    optic2.fields.add(y=h_obj)

    opd_height = OPD(
        optic2,
        field=(0, 1),
        wavelength="primary",
        num_rays=10,
        distribution="line_y",
        remove_tilt=False,
    )
    data_height = opd_height.get_data((0, 1), optic2.primary_wavelength)

    # Verify that OPD values are close
    assert_allclose(data_angle.opd, data_height.opd, atol=1.0)

    # Check RMS
    assert_allclose(opd_angle.rms(), opd_height.rms(), rtol=0.1)
