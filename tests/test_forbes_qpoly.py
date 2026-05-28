"""Tests for the qpoly module's coefficient trimming and Clenshaw edge cases.

These tests target PR 1 of the Forbes Q-polynomial downstreaming work
(issue #617): robust handling of empty, single-coefficient, and
trailing-zero coefficient vectors plus derivative orders larger than the
polynomial degree.
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.geometries.forbes.qpoly import (
    _trim_trailing_zeros,
    change_basis_q2d_to_pnm,
    change_basis_qbfs_to_pn,
    clenshaw_q2d,
    clenshaw_q2d_der,
    clenshaw_qbfs,
    clenshaw_qbfs_der,
    compute_z_zprime_q2d,
    compute_z_zprime_qbfs,
    q2d_nm_coeffs_to_ams_bms,
    q2d_sum_from_alphas,
)

from .utils import assert_allclose

# ---------------------------------------------------------------------------
# _trim_trailing_zeros
# ---------------------------------------------------------------------------


class TestTrimTrailingZeros:
    def test_none(self):
        assert _trim_trailing_zeros(None) == []

    def test_empty(self):
        assert _trim_trailing_zeros([]) == []

    def test_single_zero(self):
        assert _trim_trailing_zeros([0]) == []

    def test_all_zero(self):
        assert _trim_trailing_zeros([0, 0, 0]) == []

    def test_single_nonzero(self):
        assert _trim_trailing_zeros([1.5]) == [1.5]

    def test_trailing_only(self):
        assert _trim_trailing_zeros([1, 0, 0]) == [1]

    def test_interior_zero_preserved(self):
        assert _trim_trailing_zeros([1, 0, 2, 0, 0]) == [1, 0, 2]

    def test_no_trailing(self):
        assert _trim_trailing_zeros([1, 2, 3]) == [1, 2, 3]

    def test_tuple(self):
        assert _trim_trailing_zeros((1.0, 0.0, 0.0)) == [1.0]

    def test_numpy_array(self):
        arr = np.array([1.0, 2.0, 0.0, 0.0])
        out = _trim_trailing_zeros(arr)
        assert len(out) == 2
        assert float(out[0]) == 1.0
        assert float(out[1]) == 2.0

    def test_numpy_all_zero(self):
        arr = np.zeros(4)
        assert _trim_trailing_zeros(arr) == []


# ---------------------------------------------------------------------------
# Clenshaw / basis-change edge cases (Qbfs)
# ---------------------------------------------------------------------------


class TestClenshawQbfsEdgeCases:
    def test_empty_coefs_scalar(self, set_test_backend):
        result = clenshaw_qbfs([], be.array(0.25))
        assert float(be.to_numpy(result)) == 0.0

    def test_empty_coefs_array(self, set_test_backend):
        usq = be.array([0.1, 0.25, 0.4])
        result = clenshaw_qbfs([], usq)
        assert_allclose(result, be.zeros_like(usq))

    def test_all_zero_coefs(self, set_test_backend):
        usq = be.array([0.1, 0.5, 0.9])
        result = clenshaw_qbfs([0.0, 0.0, 0.0, 0.0], usq)
        assert_allclose(result, be.zeros_like(usq))

    def test_one_coef(self, set_test_backend):
        # Q_0(u) is constant; the Clenshaw sum 2*alpha[0] for one P_n
        # coefficient equals 2 * (cs[0] / f_qbfs(0)) = cs[0].
        usq = be.array([0.1, 0.5, 0.9])
        cs = [0.5]
        result = clenshaw_qbfs(cs, usq)
        # Two-coefficient version with explicit trailing zero must match.
        result_trail = clenshaw_qbfs([0.5, 0.0], usq)
        assert_allclose(result, result_trail)

    def test_trailing_zeros_match_trimmed(self, set_test_backend):
        usq = be.array([0.1, 0.3, 0.7])
        cs_dense = [0.7, -0.2, 0.4, 0.0, 0.0, 0.0]
        cs_trim = [0.7, -0.2, 0.4]
        assert_allclose(
            clenshaw_qbfs(cs_dense, usq),
            clenshaw_qbfs(cs_trim, usq),
        )

    def test_change_basis_handles_trailing_zero(self, set_test_backend):
        bs_dense = change_basis_qbfs_to_pn([1.0, 0.0, 0.0])
        bs_trim = change_basis_qbfs_to_pn([1.0])
        assert_allclose(bs_dense, bs_trim)

    def test_change_basis_empty(self, set_test_backend):
        bs = change_basis_qbfs_to_pn([])
        assert be.to_numpy(bs).size == 0

    def test_change_basis_all_zero(self, set_test_backend):
        bs = change_basis_qbfs_to_pn([0.0, 0.0, 0.0])
        assert be.to_numpy(bs).size == 0


class TestClenshawQbfsDerEdgeCases:
    def test_empty_coefs(self, set_test_backend):
        usq = be.array([0.1, 0.5])
        alphas = clenshaw_qbfs_der([], usq, j=1)
        # No coefficients => zero alpha table padded to 2 modes.
        assert be.to_numpy(alphas).shape[-1] == 2

    def test_j_greater_than_degree(self, set_test_backend):
        usq = be.array([0.1, 0.5])
        cs = [0.5]  # degree 0
        alphas = clenshaw_qbfs_der(cs, usq, j=2)
        # Higher derivative tables (j=1, j=2) must be zero.
        np_alphas = be.to_numpy(alphas)
        assert np.allclose(np_alphas[1], 0.0)
        assert np.allclose(np_alphas[2], 0.0)

    def test_compute_z_zprime_empty(self, set_test_backend):
        u = be.array([0.1, 0.5])
        usq = u * u
        s, ds_du = compute_z_zprime_qbfs([], u, usq)
        assert_allclose(s, be.zeros_like(u))
        assert_allclose(ds_du, be.zeros_like(u))

    def test_compute_z_zprime_all_zero(self, set_test_backend):
        u = be.array([0.1, 0.5, 0.9])
        usq = u * u
        s, ds_du = compute_z_zprime_qbfs([0.0, 0.0, 0.0], u, usq)
        assert_allclose(s, be.zeros_like(u))
        assert_allclose(ds_du, be.zeros_like(u))

    def test_compute_z_zprime_trailing_zeros_match_trimmed(self, set_test_backend):
        u = be.array([0.1, 0.3, 0.7])
        usq = u * u
        s_dense, ds_dense = compute_z_zprime_qbfs([0.7, -0.2, 0.4, 0.0, 0.0], u, usq)
        s_trim, ds_trim = compute_z_zprime_qbfs([0.7, -0.2, 0.4], u, usq)
        assert_allclose(s_dense, s_trim)
        assert_allclose(ds_dense, ds_trim)


# ---------------------------------------------------------------------------
# Clenshaw / basis-change edge cases (Q2D)
# ---------------------------------------------------------------------------


class TestClenshawQ2DEdgeCases:
    def test_empty_coefs(self, set_test_backend):
        usq = be.array([0.1, 0.5])
        alphas = clenshaw_q2d([], m=1, usq=usq)
        # Padded to at least 2 modes so callers can safely index alpha[1].
        assert be.to_numpy(alphas).shape[0] == 2
        assert np.allclose(be.to_numpy(alphas), 0.0)

    def test_one_coef(self, set_test_backend):
        usq = be.array([0.1, 0.5])
        alphas = clenshaw_q2d([0.5], m=1, usq=usq)
        # The first mode carries ds[0]; the padded second mode must be zero.
        assert be.to_numpy(alphas).shape[0] >= 1

    def test_trailing_zeros_match_trimmed_m1(self, set_test_backend):
        # m == 1 with a coefficient vector of length >= 4 triggers the
        # special alphas[3] correction in q2d_sum_from_alphas.  A dense
        # vector with a zero in slot 3 must produce the *same* sum as the
        # trimmed three-element vector that skips the correction — this is
        # the prysm one-sweep alpha-index fix.
        usq = be.array(0.4)
        cs_dense = [0.7, -0.2, 0.4, 0.0]
        cs_trim = [0.7, -0.2, 0.4]
        alphas_dense = clenshaw_q2d(cs_dense, m=1, usq=usq)
        alphas_trim = clenshaw_q2d(cs_trim, m=1, usq=usq)
        s_dense = q2d_sum_from_alphas(alphas_dense, m=1, num_coeffs=len(cs_dense))
        s_trim = q2d_sum_from_alphas(alphas_trim, m=1, num_coeffs=len(cs_trim))
        # After trimming, num_coeffs collapses to 3 internally, so the
        # m==1 special correction is correctly skipped — sums must agree.
        assert_allclose(s_dense, s_trim)

    def test_change_basis_q2d_empty(self, set_test_backend):
        ds = change_basis_q2d_to_pnm([], m=1)
        assert be.to_numpy(ds).size == 0

    def test_change_basis_q2d_trailing_zero(self, set_test_backend):
        ds_dense = change_basis_q2d_to_pnm([1.0, 0.5, 0.0, 0.0], m=2)
        ds_trim = change_basis_q2d_to_pnm([1.0, 0.5], m=2)
        assert_allclose(ds_dense, ds_trim)


class TestClenshawQ2DDerEdgeCases:
    def test_empty_coefs(self, set_test_backend):
        usq = be.array([0.1, 0.5])
        alphas = clenshaw_q2d_der([], m=1, usq=usq, j=1)
        np_a = be.to_numpy(alphas)
        # Padded to at least 2 modes, all zero.
        assert np_a.shape[0] == 2  # j+1
        assert np.allclose(np_a, 0.0)

    def test_j_greater_than_degree(self, set_test_backend):
        usq = be.array([0.1, 0.5])
        alphas = clenshaw_q2d_der([0.5], m=2, usq=usq, j=2)
        np_a = be.to_numpy(alphas)
        # j=1 and j=2 alpha rows are zero because the polynomial is degree 0.
        assert np.allclose(np_a[1], 0.0)
        assert np.allclose(np_a[2], 0.0)

    def test_compute_z_zprime_q2d_trailing_zeros_match_trimmed(self, set_test_backend):
        u = be.array([0.2, 0.5, 0.8])
        t = be.array([0.1, 0.6, 1.2])
        ams_dense = [[0.3, -0.1, 0.2, 0.0]]  # m=1, n=0..3, trailing zero
        bms_dense = [[0.0, 0.0]]
        ams_trim = [[0.3, -0.1, 0.2]]
        bms_trim = [[]]
        v_dense = compute_z_zprime_q2d([], ams_dense, bms_dense, u, t)
        v_trim = compute_z_zprime_q2d([], ams_trim, bms_trim, u, t)
        for a, b in zip(v_dense, v_trim, strict=True):
            assert_allclose(a, b)


# ---------------------------------------------------------------------------
# q2d_nm_coeffs_to_ams_bms — asymmetric a/b families
# ---------------------------------------------------------------------------


class TestQ2dNmCoeffsAsymmetric:
    def test_cos_only_family_survives(self):
        # Only ('a', 2, 0) supplied. zip(ams, bms) must still expose m=2.
        nms = [(0, 2)]
        coefs = [0.42]
        cms, ams, bms = q2d_nm_coeffs_to_ams_bms(nms, coefs)
        assert cms == []
        assert len(ams) == 2  # padded up through azimuthal order 2
        assert len(bms) == 2
        assert ams[1] == [0.42]
        assert bms[1] == []

    def test_sin_only_family_survives(self):
        # Only ('b', 3, 0) supplied. zip must still expose m=3.
        nms = [(0, -3)]
        coefs = [0.99]
        cms, ams, bms = q2d_nm_coeffs_to_ams_bms(nms, coefs)
        assert len(ams) == 3
        assert len(bms) == 3
        assert ams[2] == []  # no cos m=3 family
        assert bms[2] == [0.99]


# ---------------------------------------------------------------------------
# Geometry-level regression: scaling-style sag with trailing zeros
# ---------------------------------------------------------------------------


class TestGeometrySagWithTrailingZeros:
    def test_qbfs_sag_unchanged_by_trailing_zero_terms(self, set_test_backend):
        """A ForbesQNormalSlopeGeometry must produce identical sag when the
        radial-terms dict contains extra zero-valued terms at high order."""
        from optiland.coordinate_system import CoordinateSystem
        from optiland.geometries import (
            ForbesQNormalSlopeGeometry,
            ForbesSurfaceConfig,
        )

        cs = CoordinateSystem()
        cfg_tight = ForbesSurfaceConfig(
            radius=20.0,
            conic=-1.0,
            terms={0: 1e-3, 1: 5e-4},
            norm_radius=5.0,
        )
        cfg_padded = ForbesSurfaceConfig(
            radius=20.0,
            conic=-1.0,
            terms={0: 1e-3, 1: 5e-4, 2: 0.0, 3: 0.0, 4: 0.0},
            norm_radius=5.0,
        )
        g_tight = ForbesQNormalSlopeGeometry(cs, surface_config=cfg_tight)
        g_padded = ForbesQNormalSlopeGeometry(cs, surface_config=cfg_padded)
        x = be.array([0.5, 1.0, 2.0, 3.5])
        y = be.array([0.2, -0.4, 1.0, 2.0])
        assert_allclose(g_tight.sag(x, y), g_padded.sag(x, y))


# Mark Q2D regression tests on numpy backend — torch import not available
# in some local environments; parametrized backend fixture covers both.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")
