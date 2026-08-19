from __future__ import annotations

from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import optiland.backend as be
from optiland import zernike

from .utils import assert_allclose

matplotlib.use("Agg")  # use non-interactive backend for testing


ZERNIKE_STANDARD_INDICES = np.array(
    [
        (0, 0),
        (1, -1),
        (1, 1),
        (2, -2),
        (2, 0),
        (2, 2),
        (3, -3),
        (3, -1),
        (3, 1),
        (3, 3),
        (4, -4),
        (4, -2),
        (4, 0),
        (4, 2),
        (4, 4),
        (5, -5),
        (5, -3),
        (5, -1),
        (5, 1),
        (5, 3),
        (5, 5),
        (6, -6),
        (6, -4),
        (6, -2),
        (6, 0),
        (6, 2),
        (6, 4),
        (6, 6),
        (7, -7),
        (7, -5),
        (7, -3),
        (7, -1),
        (7, 1),
        (7, 3),
        (7, 5),
        (7, 7),
        (8, -8),
        (8, -6),
        (8, -4),
        (8, -2),
        (8, 0),
        (8, 2),
        (8, 4),
        (8, 6),
        (8, 8),
        (9, -9),
        (9, -7),
        (9, -5),
        (9, -3),
        (9, -1),
        (9, 1),
        (9, 3),
        (9, 5),
        (9, 7),
        (9, 9),
        (10, -10),
        (10, -8),
        (10, -6),
        (10, -4),
        (10, -2),
        (10, 0),
        (10, 2),
        (10, 4),
        (10, 6),
        (10, 8),
        (10, 10),
        (11, -11),
        (11, -9),
        (11, -7),
        (11, -5),
        (11, -3),
        (11, -1),
        (11, 1),
        (11, 3),
        (11, 5),
        (11, 7),
        (11, 9),
        (11, 11),
        (12, -12),
        (12, -10),
        (12, -8),
        (12, -6),
        (12, -4),
        (12, -2),
        (12, 0),
        (12, 2),
        (12, 4),
        (12, 6),
        (12, 8),
        (12, 10),
        (12, 12),
        (13, -13),
        (13, -11),
        (13, -9),
        (13, -7),
        (13, -5),
        (13, -3),
        (13, -1),
        (13, 1),
        (13, 3),
        (13, 5),
        (13, 7),
        (13, 9),
        (13, 11),
        (13, 13),
        (14, -14),
        (14, -12),
        (14, -10),
        (14, -8),
        (14, -6),
        (14, -4),
        (14, -2),
        (14, 0),
        (14, 2),
        (14, 4),
        (14, 6),
        (14, 8),
        (14, 10),
        (14, 12),
        (14, 14),
    ],
    dtype=[("n", int), ("m", int)],
)


class TestZernikeStandard:
    def test_get_term(self, set_test_backend):
        z = zernike.ZernikeStandard()
        coeff = 1.0
        n = 3
        m = -2
        r = 0.5
        phi = be.pi / 4

        term = z.get_term(coeff, n, m, r, phi)

        assert_allclose(term, 1.0606601717798214)

    def test_terms(self, set_test_backend):
        z = zernike.ZernikeStandard(coeffs=[0.2, 0.8, 0.4, -0.8, -0.1])
        r = be.array([0.1, 0.2, 0.3])
        phi = be.array([0.0, be.pi / 2, be.pi])

        terms = z.terms(r, phi)
        assert_allclose(terms[0], be.array([0.2, 0.2, 0.2]))
        assert_allclose(terms[1], be.array([0.0, 0.32, 0.0]))
        assert_allclose(terms[2], be.array([0.08, 0.0, -0.24]))
        assert_allclose(terms[3], be.array([0.0, 0.0, 0.0]))
        assert_allclose(terms[4], be.array([0.16974098, 0.15934867, 0.14202817]))

    def test_poly(self, set_test_backend):
        z = zernike.ZernikeStandard(coeffs=[0.5, -0.5, 0.3, -0.8, 1.0])
        r = be.array([0.1, 0.2, 0.3])
        phi = be.array([0.0, be.pi / 2, be.pi])

        poly = z.poly(r, phi)
        assert_allclose(poly, be.array([-1.13740979, -1.29348674, -1.10028166]))

    def test_radial_term(self, set_test_backend):
        z = zernike.ZernikeStandard()
        n = 3
        m = -2
        r = 0.5

        radial_term = z._radial_term(n, m, r)
        assert_allclose(radial_term, 0.375)

    def test_azimuthal_term(self, set_test_backend):
        z = zernike.ZernikeStandard()
        m = -2
        phi = be.pi / 4.5

        azimuthal_term = z._azimuthal_term(m, phi)
        assert_allclose(azimuthal_term, 0.984807753012208)

    def test_norm_constant(self, set_test_backend):
        z = zernike.ZernikeStandard()
        n = 3
        m = -2

        norm_constant = z._norm_constant(n, m)
        assert_allclose(norm_constant, 2.8284271247461903)

    def test_generate_indices(self, set_test_backend):
        z = zernike.ZernikeStandard()

        indices = z._generate_indices(120)

        assert np.all(indices == ZERNIKE_STANDARD_INDICES)

    def test_index_to_number(self, set_test_backend):
        z = zernike.ZernikeStandard()

        for i, idx in enumerate(ZERNIKE_STANDARD_INDICES):
            assert z._index_to_number(*idx) == i


ZERNIKE_FRINGE_INDICES = np.array(
    [
        (0, 0),
        (1, 1),
        (1, -1),
        (2, 0),
        (2, 2),
        (2, -2),
        (3, 1),
        (3, -1),
        (4, 0),
        (3, 3),
        (3, -3),
        (4, 2),
        (4, -2),
        (5, 1),
        (5, -1),
        (6, 0),
        (4, 4),
        (4, -4),
        (5, 3),
        (5, -3),
        (6, 2),
        (6, -2),
        (7, 1),
        (7, -1),
        (8, 0),
        (5, 5),
        (5, -5),
        (6, 4),
        (6, -4),
        (7, 3),
        (7, -3),
        (8, 2),
        (8, -2),
        (9, 1),
        (9, -1),
        (10, 0),
        (6, 6),
        (6, -6),
        (7, 5),
        (7, -5),
        (8, 4),
        (8, -4),
        (9, 3),
        (9, -3),
        (10, 2),
        (10, -2),
        (11, 1),
        (11, -1),
        (12, 0),
        (7, 7),
        (7, -7),
        (8, 6),
        (8, -6),
        (9, 5),
        (9, -5),
        (10, 4),
        (10, -4),
        (11, 3),
        (11, -3),
        (12, 2),
        (12, -2),
        (13, 1),
        (13, -1),
        (14, 0),
        (8, 8),
        (8, -8),
        (9, 7),
        (9, -7),
        (10, 6),
        (10, -6),
        (11, 5),
        (11, -5),
        (12, 4),
        (12, -4),
        (13, 3),
        (13, -3),
        (14, 2),
        (14, -2),
        (15, 1),
        (15, -1),
        (16, 0),
        (9, 9),
        (9, -9),
        (10, 8),
        (10, -8),
        (11, 7),
        (11, -7),
        (12, 6),
        (12, -6),
        (13, 5),
        (13, -5),
        (14, 4),
        (14, -4),
        (15, 3),
        (15, -3),
        (16, 2),
        (16, -2),
        (17, 1),
        (17, -1),
        (18, 0),
        (10, 10),
        (10, -10),
        (11, 9),
        (11, -9),
        (12, 8),
        (12, -8),
        (13, 7),
        (13, -7),
        (14, 6),
        (14, -6),
        (15, 5),
        (15, -5),
        (16, 4),
        (16, -4),
        (17, 3),
        (17, -3),
        (18, 2),
        (18, -2),
        (19, 1),
        (19, -1),
    ],
    dtype=[("n", int), ("m", int)],
)


class TestZernikeFringe:
    def test_get_term(self, set_test_backend):
        z = zernike.ZernikeFringe()
        coeff = 1.0
        n = 3
        m = -2
        r = 0.5
        phi = be.pi / 4

        term = z.get_term(coeff, n, m, r, phi)

        assert_allclose(term, 0.375)

    def test_terms(self, set_test_backend):
        z = zernike.ZernikeFringe(coeffs=[0.2, 0.8, 0.4, -0.8, -0.1])
        r = be.array([0.1, 0.2, 0.3])
        phi = be.array([0.0, be.pi / 2, be.pi])

        terms = z.terms(r, phi)
        assert_allclose(terms[0], be.array([0.2, 0.2, 0.2]))
        assert_allclose(terms[1], be.array([0.08, 0.0, -0.24]))
        assert_allclose(terms[2], be.array([0.0, 0.08, 0.0]))
        assert_allclose(terms[3], be.array([0.784, 0.736, 0.656]))
        assert_allclose(terms[4], be.array([-0.001, 0.004, -0.009]))

    def test_poly(self, set_test_backend):
        z = zernike.ZernikeFringe(coeffs=[0.5, -0.5, 0.3, -0.8, 1.0])
        r = be.array([0.1, 0.2, 0.3])
        phi = be.array([0.0, be.pi / 2, be.pi])

        poly = z.poly(r, phi)
        assert_allclose(poly, be.array([1.244, 1.256, 1.396]))

    def test_radial_term(self, set_test_backend):
        z = zernike.ZernikeFringe()
        n = 3
        m = -2
        r = 0.5

        radial_term = z._radial_term(n, m, r)
        assert_allclose(radial_term, 0.375)

    def test_azimuthal_term(self, set_test_backend):
        z = zernike.ZernikeFringe()
        m = -2
        phi = be.pi / 4.5

        azimuthal_term = z._azimuthal_term(m, phi)
        assert_allclose(azimuthal_term, 0.984807753012208)

    def test_norm_constant(self, set_test_backend):
        z = zernike.ZernikeFringe()
        n = 3
        m = -2

        norm_constant = z._norm_constant(n, m)
        assert_allclose(norm_constant, 1)

    def test_generate_indices(self, set_test_backend):
        z = zernike.ZernikeFringe()

        indices = z._generate_indices(120)

        assert np.all(indices == ZERNIKE_FRINGE_INDICES)

    def test_index_to_number(self, set_test_backend):
        z = zernike.ZernikeFringe()

        for i, idx in enumerate(ZERNIKE_FRINGE_INDICES, start=1):
            assert z._index_to_number(*idx) == i


ZERNIKE_NOLL_INDICES = np.array(
    [
        (0, 0),
        (1, 1),
        (1, -1),
        (2, 0),
        (2, -2),
        (2, 2),
        (3, -1),
        (3, 1),
        (3, -3),
        (3, 3),
        (4, 0),
        (4, 2),
        (4, -2),
        (4, 4),
        (4, -4),
        (5, 1),
        (5, -1),
        (5, 3),
        (5, -3),
        (5, 5),
        (5, -5),
        (6, 0),
        (6, -2),
        (6, 2),
        (6, -4),
        (6, 4),
        (6, -6),
        (6, 6),
        (7, -1),
        (7, 1),
        (7, -3),
        (7, 3),
        (7, -5),
        (7, 5),
        (7, -7),
        (7, 7),
        (8, 0),
        (8, 2),
        (8, -2),
        (8, 4),
        (8, -4),
        (8, 6),
        (8, -6),
        (8, 8),
        (8, -8),
        (9, 1),
        (9, -1),
        (9, 3),
        (9, -3),
        (9, 5),
        (9, -5),
        (9, 7),
        (9, -7),
        (9, 9),
        (9, -9),
        (10, 0),
        (10, -2),
        (10, 2),
        (10, -4),
        (10, 4),
        (10, -6),
        (10, 6),
        (10, -8),
        (10, 8),
        (10, -10),
        (10, 10),
        (11, -1),
        (11, 1),
        (11, -3),
        (11, 3),
        (11, -5),
        (11, 5),
        (11, -7),
        (11, 7),
        (11, -9),
        (11, 9),
        (11, -11),
        (11, 11),
        (12, 0),
        (12, 2),
        (12, -2),
        (12, 4),
        (12, -4),
        (12, 6),
        (12, -6),
        (12, 8),
        (12, -8),
        (12, 10),
        (12, -10),
        (12, 12),
        (12, -12),
        (13, 1),
        (13, -1),
        (13, 3),
        (13, -3),
        (13, 5),
        (13, -5),
        (13, 7),
        (13, -7),
        (13, 9),
        (13, -9),
        (13, 11),
        (13, -11),
        (13, 13),
        (13, -13),
        (14, 0),
        (14, -2),
        (14, 2),
        (14, -4),
        (14, 4),
        (14, -6),
        (14, 6),
        (14, -8),
        (14, 8),
        (14, -10),
        (14, 10),
        (14, -12),
        (14, 12),
        (14, -14),
        (14, 14),
    ],
    dtype=[("n", int), ("m", int)],
)


class TestZernikeNoll:
    def test_get_term(self, set_test_backend):
        z = zernike.ZernikeNoll()
        coeff = 1.0
        n = 3
        m = -2
        r = 0.5
        phi = be.pi / 4

        term = z.get_term(coeff, n, m, r, phi)

        assert_allclose(term, 1.0606601717798214)

    def test_terms(self, set_test_backend):
        z = zernike.ZernikeNoll(coeffs=[0.2, 0.8, 0.4, -0.8, -0.1])
        r = be.array([0.1, 0.2, 0.3])
        phi = be.array([0.0, be.pi / 2, be.pi])

        terms = z.terms(r, phi)
        assert_allclose(terms[0], be.array([0.2, 0.2, 0.2]))
        assert_allclose(terms[1], be.array([0.16, 0.0, -0.48]))
        assert_allclose(terms[2], be.array([0.0, 0.16, 0.0]))
        assert_allclose(terms[3], be.array([1.35792783, 1.27478939, 1.13622533]))
        assert_allclose(terms[4], be.array([0.0, 0.0, 0.0]))

    def test_poly(self, set_test_backend):
        z = zernike.ZernikeNoll(coeffs=[0.5, -0.5, 0.3, -0.8, 1.0])
        r = be.array([0.1, 0.2, 0.3])
        phi = be.array([0.0, be.pi / 2, be.pi])

        poly = z.poly(r, phi)
        assert_allclose(poly, be.array([1.75792783, 1.89478939, 1.93622533]))

    def test_radial_term(self, set_test_backend):
        z = zernike.ZernikeNoll()
        n = 3
        m = -2
        r = 0.5

        radial_term = z._radial_term(n, m, r)
        assert_allclose(radial_term, 0.375)

    def test_azimuthal_term(self, set_test_backend):
        z = zernike.ZernikeNoll()
        m = -2
        phi = be.pi / 4.5

        azimuthal_term = z._azimuthal_term(m, phi)
        assert_allclose(azimuthal_term, 0.984807753012208)

    def test_norm_constant(self, set_test_backend):
        z = zernike.ZernikeNoll()
        n = 3
        m = -2

        norm_constant = z._norm_constant(n, m)
        assert_allclose(norm_constant, 2.8284271247461903)

    def test_generate_indices(self, set_test_backend):
        z = zernike.ZernikeNoll()

        indices = z._generate_indices(120)

        assert np.all(indices == ZERNIKE_NOLL_INDICES)

    def test_index_to_number(self, set_test_backend):
        z = zernike.ZernikeNoll()

        for i, idx in enumerate(ZERNIKE_NOLL_INDICES, start=1):
            assert z._index_to_number(*idx) == i


class TestZernikeFit:
    def setup_method(self):
        coeffs = [0.1, 0.2, 0.8, -0.2, 0.4, 0.6, 0.0, 0.0, 0.5, 0.1, -0.3]

        z = zernike.ZernikeStandard(coeffs=coeffs)

        x = be.linspace(-1, 1, 10)
        x, y = be.meshgrid(x, x)
        x = be.ravel(x)
        y = be.ravel(y)
        r = be.sqrt(x**2 + y**2)
        cond = r <= 1
        x = x[cond]
        y = y[cond]
        r = r[cond]
        theta = be.arctan2(y, x)

        self.x = x
        self.y = y
        self.z = z.poly(r, theta)

    def test_init(self, set_test_backend):
        zernike_fit = zernike.ZernikeFit(self.x, self.y, self.z)
        assert_allclose(zernike_fit.x, self.x)
        assert_allclose(zernike_fit.y, self.y)
        assert_allclose(zernike_fit.z, self.z)
        assert zernike_fit.zernike_type == "fringe"
        assert zernike_fit.num_terms == 36

    def test_init_standard(self, set_test_backend):
        zernike_fit_standard = zernike.ZernikeFit(
            self.x,
            self.y,
            self.z,
            zernike_type="standard",
        )
        assert_allclose(zernike_fit_standard.x, self.x)
        assert_allclose(zernike_fit_standard.y, self.y)
        assert_allclose(zernike_fit_standard.z, self.z)
        assert zernike_fit_standard.zernike_type == "standard"
        assert zernike_fit_standard.num_terms == 36

    def test_init_noll(self, set_test_backend):
        zernike_fit_noll = zernike.ZernikeFit(
            self.x,
            self.y,
            self.z,
            zernike_type="noll",
        )
        assert_allclose(zernike_fit_noll.x, self.x)
        assert_allclose(zernike_fit_noll.y, self.y)
        assert_allclose(zernike_fit_noll.z, self.z)
        assert zernike_fit_noll.zernike_type == "noll"
        assert zernike_fit_noll.num_terms == 36

    def test_invalid_zernike_type(self, set_test_backend):
        with pytest.raises(ValueError):
            zernike.ZernikeFit(self.x, self.y, self.z, zernike_type="invalid")

    def test_coeffs(self, set_test_backend):
        zernike_fit = zernike.ZernikeFit(self.x, self.y, self.z)
        assert len(zernike_fit.coeffs) == 36

        arr = be.array([0.1, 1.6, 0.4, 0.69282032, 1.46969385])
        assert_allclose(zernike_fit.coeffs[:5], arr)

    def test_coeffs_standard(self, set_test_backend):
        zernike_fit_standard = zernike.ZernikeFit(
            self.x,
            self.y,
            self.z,
            zernike_type="standard",
        )
        assert len(zernike_fit_standard.coeffs) == 36

        arr = be.array([0.1, 0.2, 0.8, -0.2, 0.4])
        assert_allclose(zernike_fit_standard.coeffs[:5], arr)

    def test_coeffs_noll(self, set_test_backend):
        zernike_fit_noll = zernike.ZernikeFit(
            self.x,
            self.y,
            self.z,
            zernike_type="noll",
        )
        assert len(zernike_fit_noll.coeffs) == 36

        arr = be.array([0.1, 0.8, 0.2, 0.4, -0.2])
        assert_allclose(zernike_fit_noll.coeffs[:5], arr)

    def test_invalid_view_projection(self, set_test_backend):
        zernike_fit = zernike.ZernikeFit(self.x, self.y, self.z)
        with pytest.raises(ValueError):
            zernike_fit.view(projection="invalid")

    @patch("matplotlib.pyplot.show")
    def test_view_standard(self, mock_show, set_test_backend):
        zernike_fit_standard = zernike.ZernikeFit(
            self.x,
            self.y,
            self.z,
            zernike_type="standard",
        )
        fig, ax = zernike_fit_standard.view(projection="2d")
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_view_standard_3d(self, mock_show, set_test_backend):
        zernike_fit_standard = zernike.ZernikeFit(
            self.x,
            self.y,
            self.z,
            zernike_type="standard",
        )
        fig, ax = zernike_fit_standard.view(projection="3d")
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_view_noll(self, mock_show, set_test_backend):
        zernike_fit_noll = zernike.ZernikeFit(
            self.x,
            self.y,
            self.z,
            zernike_type="noll",
        )
        fig, ax = zernike_fit_noll.view(projection="2d")
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_view_noll_3d(self, mock_show, set_test_backend):
        zernike_fit_noll = zernike.ZernikeFit(
            self.x,
            self.y,
            self.z,
            zernike_type="noll",
        )
        fig, ax = zernike_fit_noll.view(projection="3d")
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_view_residual_standard(self, mock_show, set_test_backend):
        zernike_fit_standard = zernike.ZernikeFit(
            self.x,
            self.y,
            self.z,
            zernike_type="standard",
        )
        fig, ax = zernike_fit_standard.view_residual()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_view_residual_noll(self, mock_show, set_test_backend):
        zernike_fit_noll = zernike.ZernikeFit(
            self.x,
            self.y,
            self.z,
            zernike_type="noll",
        )
        fig, ax = zernike_fit_noll.view_residual()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_view(self, mock_show, set_test_backend):
        zernike_fit = zernike.ZernikeFit(self.x, self.y, self.z)
        fig, ax = zernike_fit.view(projection="2d")
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_view_3d(self, mock_show, set_test_backend):
        zernike_fit = zernike.ZernikeFit(self.x, self.y, self.z)
        fig, ax = zernike_fit.view(projection="3d")
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_view_residual(self, mock_show, set_test_backend):
        zernike_fit = zernike.ZernikeFit(self.x, self.y, self.z)
        fig, ax = zernike_fit.view_residual()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)
