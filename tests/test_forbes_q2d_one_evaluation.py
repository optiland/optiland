"""Forbes Q-2D surface: the sag and its slopes share one evaluation.

The Newton-Raphson loop evaluates the sag at each iterate and its slopes at
the same point. The Q-2D geometry now gives both from one evaluation
(``_sag_and_terms`` and ``_slopes_from_terms``): r**2, u**2, the conic
correction factor and one evaluation of the polynomial, with its slopes.
The slopes are functions of r**2 times x or y, so they need no division by
r and no special case at the vertex. The sag and the slopes use the same
r**2 and the same rule at u = 1.

The changes of values are small. The Q-2D sag has no epsilon in its radius
any more (it was sqrt(r**2 + 1e-12)), so it changes near the vertex, and
the slopes change in the last bits.
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland import _host_values as host_values
from optiland.geometries.forbes import ForbesQ2dGeometry, qpoly
from optiland.geometries.forbes import geometry as forbes_geometry
from optiland.geometries.newton_raphson import _shares_terms
from optiland.rays import RealRays

from .forbes_evaluation_test_utils import (
    NORM,
    Q2D,
    X,
    Y,
    _assert_bits,
    _geometry,
    _slopes,
)
from .nr_implicit_test_utils import backend_state

CASES = [Q2D]
IDS = ["q2d"]


class TestSharedTerms:
    @pytest.mark.parametrize("terms", CASES, ids=IDS)
    def test_newton_loop_shares_the_terms(self, terms):
        """The Q-2D surface shares its terms; the polar fallback does not."""
        with backend_state("numpy", "float64"):
            geometry = _geometry(terms)
            assert _shares_terms(geometry)
            if isinstance(geometry, ForbesQ2dGeometry):
                geometry._use_cartesian_normal = False
                assert not _shares_terms(geometry)

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("terms", CASES, ids=IDS)
    def test_terms_give_the_sag_and_the_slopes(self, terms, backend):
        """_sag_and_terms has the bits of sag; the slopes those of the normal."""
        if backend == "torch":
            pytest.importorskip("torch")
        with backend_state(backend, "float64"):
            geometry = _geometry(terms)
            x, y = be.array(X), be.array(Y)
            sag, shared = geometry._sag_and_terms(x, y)
            _assert_bits(sag, geometry.sag(x, y))
            slopes = geometry._slopes_from_terms(x, y, shared)
            expected = geometry._surface_normal_analytical(x, y)
            for value, reference in zip(slopes, expected, strict=True):
                _assert_bits(value, reference)

    @pytest.mark.parametrize("terms", CASES, ids=IDS)
    def test_one_polynomial_evaluation_per_iterate(self, terms, monkeypatch):
        """The slopes of an iterate take the evaluation of its sag."""
        names = ("clenshaw_qbfs", "compute_z_zprime_qbfs", "_q2d_cartesian_eval")
        calls = []
        for name in names:
            original = getattr(forbes_geometry, name)

            def counted(*args, original=original, name=name, **kwargs):
                calls.append(name)
                return original(*args, **kwargs)

            monkeypatch.setattr(forbes_geometry, name, counted)
        with backend_state("numpy", "float64"):
            geometry = _geometry(terms)
            rays = RealRays(
                x=[0.0, 2.5, -4.0],
                y=[0.0, -1.6, 3.0],
                z=[-5.0, -5.0, -5.0],
                L=[0.0, 0.08, -0.05],
                M=[0.0, -0.05, 0.02],
                N=[1.0, (1 - 0.08**2 - 0.05**2) ** 0.5, (1 - 0.05**2 - 0.02**2) ** 0.5],
                intensity=[1.0, 1.0, 1.0],
                wavelength=[0.587, 0.587, 0.587],
            )
            result = geometry._solve_distance_primal(rays)
        assert result.iterations > 0
        assert len(calls) == result.iterations + 1
        assert "clenshaw_qbfs" not in calls  # no evaluation of the sag alone


class TestValues:
    @pytest.mark.parametrize("terms", CASES, ids=IDS)
    def test_slopes_are_those_of_the_sag(self, terms):
        """Central differences of the sag give the slopes, also near the vertex."""
        with backend_state("numpy", "float64"):
            geometry = _geometry(terms)
            h = 1.0e-5
            points = [(0.0, 0.0), (1.0e-6, -2.0e-6), (3.0, 1.0), (-5.0, 4.0)]
            for x0, y0 in points:
                x, y = be.array([x0]), be.array([y0])
                dfdx, dfdy = _slopes(geometry, x, y)
                sag = geometry.sag
                fd_x = (sag(x + h, y) - sag(x - h, y)) / (2 * h)
                fd_y = (sag(x, y + h) - sag(x, y - h)) / (2 * h)
                np.testing.assert_allclose(dfdx, fd_x, rtol=1e-7, atol=1e-11)
                np.testing.assert_allclose(dfdy, fd_y, rtol=1e-7, atol=1e-11)

    @pytest.mark.parametrize("terms", CASES, ids=IDS)
    def test_one_rule_at_the_rim(self, terms):
        """At u = 1 the slopes are those of the inside; outside, of the base."""
        with backend_state("numpy", "float64"):
            geometry = _geometry(terms)
            plane = _geometry({key: 0.0 for key in terms})
            h = 1.0e-6
            x, y = be.array([NORM]), be.array([0.0])
            dfdx, _ = _slopes(geometry, x, y)
            inside = (geometry.sag(x, y) - geometry.sag(x - h, y)) / h
            np.testing.assert_allclose(dfdx, inside, rtol=1e-5)
            base, _ = _slopes(plane, x, y)
            assert abs(float(dfdx[0] - base[0])) > 1e-6
            outside = be.array([NORM + 1.0e-9])
            _assert_bits(
                geometry._surface_normal(outside, y)[0],
                plane._surface_normal(outside, y)[0],
            )

    def test_q2d_sag_has_no_epsilon(self):
        """The Q-2D sag is the polar sum with rho = sqrt(r**2), also at the vertex."""
        with backend_state("numpy", "float64"):
            geometry = _geometry(Q2D)
            x, y = be.array(X[:5]), be.array(Y[:5])
            r2 = x**2 + y**2
            u = np.sqrt(r2) / NORM
            theta = np.arctan2(y, x)
            s_m0, s_mgt0 = qpoly.compute_z_q2d(
                geometry.cm0_coeffs, geometry.ams_coeffs, geometry.bms_coeffs, u, theta
            )
            factor, _ = geometry._conic_correction_factor(r2)
            usq = u**2
            departure = factor * (usq * (1 - usq) * s_m0 + s_mgt0)
            expected = geometry._base_sag(r2) + departure
            np.testing.assert_allclose(geometry.sag(x, y), expected, rtol=1e-14, atol=0)
            assert float(geometry.sag(be.array([0.0]), be.array([0.0]))[0]) == 0.0


class TestHostReads:
    @pytest.mark.parametrize("terms", CASES, ids=IDS)
    def test_evaluation_reads_no_tensor_value(self, terms, monkeypatch):
        """The radius checks read the host copy."""
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(host_values, "_DEVICE_TYPES", frozenset({"cuda", "cpu"}))
        with backend_state("torch", "float64"):
            geometry = _geometry(terms)
            x = torch.tensor(X, dtype=torch.float64)
            y = torch.tensor(Y, dtype=torch.float64)
            geometry.sag(x, y)  # prepare the coefficients
            reads = []
            for name in ("item", "__bool__"):
                original = getattr(torch.Tensor, name)

                def counted(tensor, original=original, name=name):
                    reads.append(name)
                    return original(tensor)

                monkeypatch.setattr(torch.Tensor, name, counted)
            with host_values.host_scope([geometry.radius]):
                geometry.sag(x, y)
                geometry._surface_normal(x, y)
                sag, shared = geometry._sag_and_terms(x, y)
                geometry._slopes_from_terms(x, y, shared)
            assert reads == []
