"""Chebyshev surface-derivative consistency and implicit-diff regressions.

The Chebyshev sag uses normalized coordinates ``x / norm_x`` and
``y / norm_y``; the chain rule therefore puts a ``1 / norm`` factor in every
slope. These tests pin down:

- ``sag()`` and ``_surface_normal()`` describe the same surface for unequal
  normalizations (slopes match central differences of the sag);
- a single analytic term has the exact textbook derivative;
- values and derivatives are finite and exact at the normalized endpoints
  ``x = +-norm`` where the old ``sqrt(1 - x**2)`` form produced ``0/0``;
- the implicit ray-intersection gradient matches central differences for a
  coefficient and for a ray-origin coordinate.
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import ChebyshevPolynomialGeometry
from optiland.rays import RealRays

from .nr_implicit_test_utils import backend_state

torch = pytest.importorskip("torch")

NORM_X = 5.0
NORM_Y = 3.0

# Multiple nonzero terms including cross terms.
COEFFS = [
    [0.0, 8.0e-3, -2.0e-3],
    [1.5e-2, -4.0e-3, 1.0e-3],
    [5.0e-3, 2.0e-3, 0.0],
]

BACKEND_PRECISION = [
    ("numpy", "float64"),
    ("numpy", "float32"),
    ("torch", "float64"),
    ("torch", "float32"),
]


def build_chebyshev(coefficients=None, radius=25.0):
    return ChebyshevPolynomialGeometry(
        CoordinateSystem(),
        radius=radius,
        conic=0.0,
        coefficients=be.array(coefficients if coefficients is not None else COEFFS),
        norm_x=NORM_X,
        norm_y=NORM_Y,
    )


def _slopes(geometry, x, y):
    nx, ny, nz = geometry._surface_normal(be.array([x]), be.array([y]))
    sx = float(be.to_numpy(-nx / nz).ravel()[0])
    sy = float(be.to_numpy(-ny / nz).ravel()[0])
    return sx, sy


class TestSlopeMatchesSagDerivative:
    @pytest.mark.parametrize("backend,precision", BACKEND_PRECISION)
    @pytest.mark.parametrize(
        "x,y", [(1.2, -0.8), (-3.9, 2.4), (0.0, 0.0), (4.5, -2.7)]
    )
    def test_normal_matches_central_fd_of_sag(self, backend, precision, x, y):
        with backend_state(backend, precision):
            geometry = build_chebyshev()

            if precision == "float64":
                h, tol = 1.0e-6, 1.0e-6
            else:
                h, tol = 1.0e-2, 2.0e-3

            def sag(px, py):
                value = geometry.sag(be.array([px]), be.array([py]))
                return float(be.to_numpy(value).ravel()[0])

            # Step-size stability: two perturbation sizes must agree before
            # the FD value is treated as a reference.
            fd_x = [
                (sag(x + hh, y) - sag(x - hh, y)) / (2.0 * hh) for hh in (h, 2 * h)
            ]
            fd_y = [
                (sag(x, y + hh) - sag(x, y - hh)) / (2.0 * hh) for hh in (h, 2 * h)
            ]
            assert abs(fd_x[0] - fd_x[1]) <= max(tol, tol * abs(fd_x[0]))
            assert abs(fd_y[0] - fd_y[1]) <= max(tol, tol * abs(fd_y[0]))

            sx, sy = _slopes(geometry, x, y)
            assert sx == pytest.approx(fd_x[0], abs=tol, rel=tol)
            assert sy == pytest.approx(fd_y[0], abs=tol, rel=tol)


class TestAnalyticTerm:
    @pytest.mark.parametrize("backend,precision", BACKEND_PRECISION)
    def test_single_term_derivative_includes_norm_factor(self, backend, precision):
        """0.7 * T_2(x/5): dz/dx = 0.7 * 4 * (x/5) / 5. Catches a missing 1/5."""
        with backend_state(backend, precision):
            geometry = ChebyshevPolynomialGeometry(
                CoordinateSystem(),
                radius=be.inf,
                conic=0.0,
                coefficients=be.array([[0.0], [0.0], [0.7]]),
                norm_x=5.0,
                norm_y=3.0,
            )
            x = 2.0
            expected = 0.7 * 4.0 * (x / 5.0) / 5.0  # 0.224

            sx, sy = _slopes(geometry, x, 0.5)
            tol = 1e-10 if precision == "float64" else 1e-5
            assert sx == pytest.approx(expected, abs=tol)
            assert sy == pytest.approx(0.0, abs=tol)


class TestEndpointBehavior:
    @pytest.mark.parametrize("backend,precision", BACKEND_PRECISION)
    @pytest.mark.parametrize(
        "x,y",
        [(NORM_X, 0.0), (-NORM_X, 0.0), (0.0, NORM_Y), (0.0, -NORM_Y),
         (NORM_X, NORM_Y), (-NORM_X, -NORM_Y)],
    )
    def test_sag_and_normal_finite_at_endpoints(self, backend, precision, x, y):
        with backend_state(backend, precision):
            geometry = build_chebyshev()
            sag = geometry.sag(be.array([x]), be.array([y]))
            assert bool(be.all(be.isfinite(sag)))

            nx, ny, nz = geometry._surface_normal(be.array([x]), be.array([y]))
            for comp in (nx, ny, nz):
                assert bool(be.all(be.isfinite(comp)))

    @pytest.mark.parametrize("backend,precision", BACKEND_PRECISION)
    def test_endpoint_slope_matches_exact_limit(self, backend, precision):
        """T_n'(+-1) = (+-1)^(n-1) n^2: for 0.7*T_2(x/5), dz/dx(+-5) = +-0.56."""
        with backend_state(backend, precision):
            geometry = ChebyshevPolynomialGeometry(
                CoordinateSystem(),
                radius=be.inf,
                conic=0.0,
                coefficients=be.array([[0.0], [0.0], [0.7]]),
                norm_x=5.0,
                norm_y=3.0,
            )
            tol = 1e-10 if precision == "float64" else 1e-5

            sx_pos, _ = _slopes(geometry, 5.0, 0.0)
            sx_neg, _ = _slopes(geometry, -5.0, 0.0)
            expected = 0.7 * (2.0**2) / 5.0  # 0.56
            assert sx_pos == pytest.approx(expected, abs=tol)
            assert sx_neg == pytest.approx(-expected, abs=tol)

    @pytest.mark.parametrize("precision", ["float64", "float32"])
    def test_torch_coordinate_gradients_finite_at_endpoints(self, precision):
        with backend_state("torch", precision):
            dtype = torch.float64 if precision == "float64" else torch.float32
            geometry = build_chebyshev()

            x = torch.tensor([NORM_X], dtype=dtype, requires_grad=True)
            y = torch.tensor([NORM_Y], dtype=dtype, requires_grad=True)
            sag = geometry.sag(x, y).sum()
            gx, gy = torch.autograd.grad(sag, (x, y))
            assert bool(torch.isfinite(gx).all()), "d(sag)/dx is not finite at x=norm_x"
            assert bool(torch.isfinite(gy).all()), "d(sag)/dy is not finite at y=norm_y"


class TestRayIntersectionImplicitGradients:
    """AD-vs-FD through the full implicit distance() path, norm_x != norm_y."""

    @staticmethod
    def _rays(dtype, x_override=None):
        x0 = 0.9 if x_override is None else x_override
        L, M = 0.08, -0.05
        N = (1.0 - L * L - M * M) ** 0.5
        return RealRays(
            x=x0,
            y=-0.6,
            z=-5.0,
            L=L,
            M=M,
            N=N,
            intensity=1.0,
            wavelength=0.587,
        )

    @classmethod
    def _distance_for_coeff(cls, coeff_value, dtype):
        coeffs = torch.zeros((3, 3), dtype=dtype)
        base = torch.tensor(COEFFS, dtype=dtype)
        coeffs = base.clone()
        coeffs[1, 0] = coeff_value
        geometry = ChebyshevPolynomialGeometry(
            CoordinateSystem(),
            radius=25.0,
            conic=0.0,
            coefficients=coeffs,
            norm_x=NORM_X,
            norm_y=NORM_Y,
        )
        return geometry.distance(cls._rays(dtype))[0]

    @classmethod
    def _distance_for_ray_x(cls, x_value, dtype):
        geometry = ChebyshevPolynomialGeometry(
            CoordinateSystem(),
            radius=25.0,
            conic=0.0,
            coefficients=be.array(COEFFS),
            norm_x=NORM_X,
            norm_y=NORM_Y,
        )
        return geometry.distance(cls._rays(dtype, x_override=x_value))[0]

    @pytest.mark.parametrize("precision", ["float64", "float32"])
    def test_coefficient_gradient_matches_fd(self, precision):
        with backend_state("torch", precision):
            dtype = torch.float64 if precision == "float64" else torch.float32
            base = float(COEFFS[1][0])

            param = torch.tensor(base, dtype=dtype, requires_grad=True)
            t = self._distance_for_coeff(param, dtype)
            assert t.requires_grad
            (ad,) = torch.autograd.grad(t, param)
            ad = float(ad)

            steps = (2e-6, 1e-6) if precision == "float64" else (2e-2, 1e-2)
            fd_values = []
            for h in steps:
                with torch.no_grad():
                    plus = float(
                        self._distance_for_coeff(
                            torch.tensor(base + h, dtype=dtype), dtype
                        )
                    )
                    minus = float(
                        self._distance_for_coeff(
                            torch.tensor(base - h, dtype=dtype), dtype
                        )
                    )
                fd_values.append((plus - minus) / (2.0 * h))
            fd = fd_values[-1]

            fd_tol = 1e-6 if precision == "float64" else 5e-2
            assert abs(fd_values[0] - fd_values[1]) <= max(fd_tol, fd_tol * abs(fd)), (
                f"FD unstable across steps: {fd_values}"
            )
            tol = 1e-6 if precision == "float64" else 5e-2
            assert abs(ad - fd) <= max(tol, tol * abs(fd)), (
                f"coefficient AD/FD mismatch: AD={ad:.9e} FD={fd:.9e}"
            )

    @pytest.mark.parametrize("precision", ["float64", "float32"])
    def test_ray_origin_gradient_matches_fd(self, precision):
        with backend_state("torch", precision):
            dtype = torch.float64 if precision == "float64" else torch.float32
            x0 = 0.9

            param = torch.tensor(x0, dtype=dtype, requires_grad=True)
            t = self._distance_for_ray_x(param, dtype)
            assert t.requires_grad
            (ad,) = torch.autograd.grad(t, param)
            ad = float(ad)

            steps = (2e-6, 1e-6) if precision == "float64" else (2e-2, 1e-2)
            fd_values = []
            for h in steps:
                with torch.no_grad():
                    plus = float(self._distance_for_ray_x(x0 + h, dtype))
                    minus = float(self._distance_for_ray_x(x0 - h, dtype))
                fd_values.append((plus - minus) / (2.0 * h))
            fd = fd_values[-1]

            tol = 1e-6 if precision == "float64" else 5e-2
            assert abs(fd_values[0] - fd_values[1]) <= max(tol, tol * abs(fd)), (
                f"FD unstable across steps: {fd_values}"
            )
            assert abs(ad - fd) <= max(tol, tol * abs(fd)), (
                f"ray-origin AD/FD mismatch: AD={ad:.9e} FD={fd:.9e}"
            )


class TestNumpyTorchForwardConsistency:
    def test_sag_and_normal_agree_across_backends(self):
        with backend_state("numpy", "float64"):
            geometry = build_chebyshev()
            sag_np = np.asarray(
                be.to_numpy(geometry.sag(be.array([1.2, -3.9]), be.array([-0.8, 2.4])))
            )
            n_np = [
                np.asarray(be.to_numpy(c))
                for c in geometry._surface_normal(
                    be.array([1.2, -3.9]), be.array([-0.8, 2.4])
                )
            ]

        with backend_state("torch", "float64"):
            geometry = build_chebyshev()
            sag_t = np.asarray(
                be.to_numpy(geometry.sag(be.array([1.2, -3.9]), be.array([-0.8, 2.4])))
            )
            n_t = [
                np.asarray(be.to_numpy(c))
                for c in geometry._surface_normal(
                    be.array([1.2, -3.9]), be.array([-0.8, 2.4])
                )
            ]

        np.testing.assert_allclose(sag_np, sag_t, rtol=1e-12, atol=1e-12)
        for a, b in zip(n_np, n_t, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
