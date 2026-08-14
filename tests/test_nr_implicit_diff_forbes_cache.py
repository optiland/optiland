"""Forbes coefficient-cache interaction with the implicit Newton solve.

The Newton-Raphson ``distance()`` solve runs its primal iteration inside
``torch.no_grad()``. Forbes geometries cache derived coefficient containers
(``_prepared_*``) on first use, so the *first* cache build can happen while
grad recording is disabled. A container stacked under ``no_grad`` is
permanently detached: the later differentiable correction would then reuse it,
the residual would still depend on the ray coordinates, and the gradient with
respect to the Forbes coefficients would silently collapse to zero.

``ForbesGeometryBase._ensure_coeffs`` builds the cache under an explicit
``torch.enable_grad()`` so it stays differentiable regardless of the caller's
ambient grad context. These tests pin that behavior down.
"""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.forbes.geometry import (
    ForbesQ2dGeometry,
    ForbesQNormalSlopeGeometry,
    ForbesSurfaceConfig,
)
from optiland.rays import RealRays

from .nr_implicit_test_utils import backend_state

torch = pytest.importorskip("torch")


def _qbfs_geometry() -> ForbesQNormalSlopeGeometry:
    cfg = ForbesSurfaceConfig(
        radius=25.0,
        conic=-0.5,
        norm_radius=8.0,
        terms={0: 1.0e-3, 1: -5.0e-4, 2: 2.0e-4, 3: -8.0e-5},
    )
    return ForbesQNormalSlopeGeometry(CoordinateSystem(), cfg)


def _q2d_geometry() -> ForbesQ2dGeometry:
    cfg = ForbesSurfaceConfig(
        radius=25.0,
        conic=-0.5,
        norm_radius=8.0,
        terms={
            ("a", 0, 0): 5.0e-4,
            ("a", 0, 1): -2.0e-4,
            ("a", 1, 0): 1.0e-4,
            ("b", 1, 0): 8.0e-5,
        },
    )
    return ForbesQ2dGeometry(CoordinateSystem(), cfg)


def _rays(n: int = 7) -> RealRays:
    dtype = torch.float64
    return RealRays(
        x=torch.linspace(-3.0, 3.0, n, dtype=dtype),
        y=torch.linspace(-2.0, 2.0, n, dtype=dtype),
        z=torch.full((n,), -10.0, dtype=dtype),
        L=torch.zeros(n, dtype=dtype),
        M=torch.zeros(n, dtype=dtype),
        N=torch.ones(n, dtype=dtype),
        intensity=torch.ones(n, dtype=dtype),
        wavelength=torch.full((n,), 0.55, dtype=dtype),
    )


def _prime_cache_under_no_grad(geometry) -> None:
    """Force the *first* cache build to happen with grad recording disabled."""
    probe = torch.tensor([1.0], dtype=torch.float64)
    with torch.no_grad():
        geometry.sag(probe, probe)
    assert not geometry._coeffs_dirty, "cache should have been built by the probe"


class TestNoGradPrimingKeepsCoefficientsTrainable:
    """A cache built under no_grad must not detach the coefficients."""

    def test_q_normal_slope_no_grad_priming(self):
        with backend_state("torch"):
            geometry = _qbfs_geometry()
            coeff = torch.tensor(1.0e-3, dtype=torch.float64, requires_grad=True)
            geometry.radial_terms[0] = coeff

            _prime_cache_under_no_grad(geometry)

            t = geometry.distance(_rays())
            t.sum().backward()

            assert coeff.grad is not None
            assert torch.isfinite(coeff.grad)
            assert float(coeff.grad.abs()) > 0.0

    def test_q2d_no_grad_priming(self):
        with backend_state("torch"):
            geometry = _q2d_geometry()
            coeff = torch.tensor(5.0e-4, dtype=torch.float64, requires_grad=True)
            geometry.freeform_coeffs[("a", 0, 0)] = coeff

            _prime_cache_under_no_grad(geometry)

            t = geometry.distance(_rays())
            t.sum().backward()

            assert coeff.grad is not None
            assert torch.isfinite(coeff.grad)
            assert float(coeff.grad.abs()) > 0.0

    def test_priming_does_not_change_the_gradient(self):
        """The primed and unprimed paths must agree, not merely both be nonzero."""

        def gradient(prime: bool) -> float:
            with backend_state("torch"):
                geometry = _qbfs_geometry()
                coeff = torch.tensor(1.0e-3, dtype=torch.float64, requires_grad=True)
                geometry.radial_terms[0] = coeff
                if prime:
                    _prime_cache_under_no_grad(geometry)
                geometry.distance(_rays()).sum().backward()
                return float(coeff.grad)

        primed = gradient(prime=True)
        unprimed = gradient(prime=False)
        assert primed == pytest.approx(unprimed, rel=1e-12, abs=1e-15)


class TestForbesCoefficientGradientAccuracy:
    """AD gradients must match stable central finite differences."""

    def test_q_normal_slope_ad_matches_central_fd(self):
        with backend_state("torch"):

            def distance_sum(value: float) -> float:
                geometry = _qbfs_geometry()
                geometry.radial_terms[0] = be.array(value)
                with torch.no_grad():
                    return float(geometry.distance(_rays()).sum())

            geometry = _qbfs_geometry()
            coeff = torch.tensor(1.0e-3, dtype=torch.float64, requires_grad=True)
            geometry.radial_terms[0] = coeff
            geometry.distance(_rays()).sum().backward()
            ad = float(coeff.grad)

            fd_values = [
                (distance_sum(1.0e-3 + h) - distance_sum(1.0e-3 - h)) / (2.0 * h)
                for h in (1.0e-6, 1.0e-7)
            ]
            spread = abs(fd_values[0] - fd_values[1])
            assert spread <= max(1e-8, 1e-3 * abs(fd_values[-1])), (
                f"FD estimates unstable across steps: {fd_values}"
            )

            fd = fd_values[-1]
            assert abs(ad - fd) <= max(1e-8, 1e-2 * abs(fd)), (
                f"AD/FD mismatch: AD={ad:.12e}, FD={fd:.12e}"
            )


class TestRepeatedBackward:
    """Two forward/backward passes on the same geometry must both work."""

    def test_two_consecutive_backwards_without_retain_graph(self):
        with backend_state("torch"):
            geometry = _qbfs_geometry()
            coeff = torch.tensor(1.0e-3, dtype=torch.float64, requires_grad=True)
            geometry.radial_terms[0] = coeff

            geometry.distance(_rays()).sum().backward()
            first = float(coeff.grad)

            # Zero the gradient and run again on the *same* geometry object, so
            # a cached tensor whose grad_fn was freed by the first backward
            # would surface as a stale-graph RuntimeError here.
            coeff.grad = None
            geometry.distance(_rays()).sum().backward()
            second = float(coeff.grad)

            assert first == pytest.approx(second, rel=1e-12, abs=1e-15)
            assert abs(second) > 0.0

    def test_repeated_backward_after_no_grad_priming(self):
        with backend_state("torch"):
            geometry = _q2d_geometry()
            coeff = torch.tensor(5.0e-4, dtype=torch.float64, requires_grad=True)
            geometry.freeform_coeffs[("a", 0, 0)] = coeff

            _prime_cache_under_no_grad(geometry)

            for _ in range(2):
                coeff.grad = None
                geometry.distance(_rays()).sum().backward()
                assert coeff.grad is not None
                assert torch.isfinite(coeff.grad)
                assert float(coeff.grad.abs()) > 0.0
