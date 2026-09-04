"""Singular-root policy tests for Newton-Raphson implicit differentiation.

These tests pin down the scientific contract of ``distance()``:

- regular rays receive the exact first-order implicit derivative;
- rejected rays (non-finite, non-converged, tangent, near-vertical surface)
  keep their detached primal forward value, are never evaluated through a
  grad-attached residual branch, and contribute zero gradient to shared
  trainable parameters;
- rejections are reported through one grouped ``RuntimeWarning``.
"""

from __future__ import annotations

import warnings

import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.newton_raphson import NewtonRaphsonGeometry
from optiland.rays import RealRays

from .nr_implicit_test_utils import backend_state

torch = pytest.importorskip("torch")


class _SqrtHeightGeometry(NewtonRaphsonGeometry):
    """Planar-base surface ``z = c * sqrt(x)``.

    The residual is non-finite for any ray whose intersection has ``x < 0``,
    which makes this a deterministic probe for the invalid-branch policy.
    """

    def __init__(self, c, tol=1e-10, max_iter=25):
        super().__init__(
            CoordinateSystem(), radius=be.inf, conic=0.0, tol=tol, max_iter=max_iter
        )
        self.c = c

    def sag(self, x=0, y=0):
        return self.c * be.sqrt(x)

    def _surface_normal(self, x, y):
        dzdx = self.c / (2.0 * be.sqrt(x))
        dzdy = be.zeros_like(dzdx)
        norm = be.sqrt(dzdx**2 + dzdy**2 + 1.0)
        return dzdx / norm, dzdy / norm, -1.0 / norm


class _CountingSqrtGeometry(_SqrtHeightGeometry):
    """Counts sag() evaluations made with autograd recording enabled."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_attached_sag_calls = 0

    def sag(self, x=0, y=0):
        if torch.is_grad_enabled():
            self.grad_attached_sag_calls += 1
        return super().sag(x, y)


class _ParabolaHeightGeometry(NewtonRaphsonGeometry):
    """Planar-base surface ``z = a * x**2`` for exact tangent-root tests."""

    def __init__(self, a, tol=1e-10, max_iter=25):
        super().__init__(
            CoordinateSystem(), radius=be.inf, conic=0.0, tol=tol, max_iter=max_iter
        )
        self.a = a

    def sag(self, x=0, y=0):
        return self.a * x**2

    def _surface_normal(self, x, y):
        dzdx = 2.0 * self.a * x
        dzdy = be.zeros_like(dzdx)
        norm = be.sqrt(dzdx**2 + dzdy**2 + 1.0)
        return dzdx / norm, dzdy / norm, -1.0 / norm


class _RampHeightGeometry(NewtonRaphsonGeometry):
    """Planar-base surface ``z = s * x``.

    For a ray parallel to the z axis at ``x0``, the exact root is
    ``t* = s * x0 - z0`` with ``dt*/ds = x0``, while the surface normal has
    ``|n_z| = 1 / sqrt(1 + s**2)``. Choosing ``s`` therefore dials ``n_z``
    through the dtype-aware validity threshold without affecting the
    (perfectly regular) Newton denominator ``dF/dt = -N``.
    """

    def __init__(self, s, tol=1e-10, max_iter=25):
        super().__init__(
            CoordinateSystem(), radius=be.inf, conic=0.0, tol=tol, max_iter=max_iter
        )
        self.s = s

    def sag(self, x=0, y=0):
        return self.s * x

    def _surface_normal(self, x, y):
        dzdx = self.s * be.ones_like(x)
        dzdy = be.zeros_like(x)
        norm = be.sqrt(dzdx**2 + dzdy**2 + 1.0)
        return dzdx / norm, dzdy / norm, -1.0 / norm


def _axial_rays(x_values, z0=-1.0):
    """Rays parallel to the z axis at the given x offsets."""
    n = len(x_values)
    return RealRays(
        x=be.array(x_values),
        y=be.zeros(n),
        z=be.full((n,), z0),
        L=be.zeros(n),
        M=be.zeros(n),
        N=be.ones(n),
        intensity=be.ones(n),
        wavelength=be.full((n,), 0.587),
    )


def _dtype(precision: str):
    return torch.float64 if precision == "float64" else torch.float32


# ----------------------------------------------------------------------
# A. Mixed valid / non-finite batch
# ----------------------------------------------------------------------


class TestMixedValidInvalidBatch:
    @pytest.mark.parametrize("precision", ["float64", "float32"])
    def test_invalid_ray_contributes_zero_gradient(self, precision):
        with backend_state("torch", precision):
            dtype = _dtype(precision)

            # Reference: ray 0 alone. t* = 1 + c * sqrt(x0), dt*/dc = 1.
            c_ref = torch.nn.Parameter(torch.tensor(0.7, dtype=dtype))
            geometry = _SqrtHeightGeometry(c_ref)
            rays = _axial_rays([1.0])
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                t_ref = geometry.distance(rays)
            t_ref.sum().backward()
            grad_ref = float(c_ref.grad.detach().item())
            assert grad_ref == pytest.approx(1.0, abs=1e-5)

            # Mixed batch: ray 0 regular, ray 1 hits the non-finite domain.
            c = torch.nn.Parameter(torch.tensor(0.7, dtype=dtype))
            geometry = _SqrtHeightGeometry(c)
            rays = _axial_rays([1.0, -1.0])

            with pytest.warns(RuntimeWarning, match="rejected 1 of 2"):
                t = geometry.distance(rays)

            assert bool(torch.isfinite(t[0]))
            t.sum().backward()

            grad = c.grad.detach()
            assert bool(torch.isfinite(grad)), (
                f"invalid ray contaminated the shared gradient: {grad}"
            )
            assert float(grad.item()) == pytest.approx(grad_ref, abs=1e-5), (
                "mixed-batch gradient must equal the valid-ray-only gradient"
            )

    def test_valid_ray_forward_value_is_preserved(self):
        with backend_state("torch", "float64"):
            c = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
            geometry = _SqrtHeightGeometry(c)
            rays = _axial_rays([1.0, -1.0])

            with pytest.warns(RuntimeWarning):
                t = geometry.distance(rays)

            assert float(t[0].detach()) == pytest.approx(1.7, abs=1e-10)
            # The invalid ray keeps its (non-finite) detached primal value
            # rather than a fabricated finite one.
            assert not bool(torch.isfinite(t[1]))


# ----------------------------------------------------------------------
# B. All-invalid batch
# ----------------------------------------------------------------------


class TestAllInvalidBatch:
    def test_returns_detached_primal_without_grad_attached_residual(self):
        with backend_state("torch", "float64"):
            c = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
            geometry = _CountingSqrtGeometry(c)
            rays = _axial_rays([-1.0, -2.0])

            with pytest.warns(RuntimeWarning, match="rejected 2 of 2"):
                t = geometry.distance(rays)

            # No grad-attached residual evaluation was attempted, the result
            # is fully detached and no implicit gradient was manufactured.
            assert geometry.grad_attached_sag_calls == 0
            assert not t.requires_grad
            assert c.grad is None


# ----------------------------------------------------------------------
# C. Exact tangent root
# ----------------------------------------------------------------------


class TestExactTangentRoot:
    @pytest.mark.parametrize("precision", ["float64", "float32"])
    def test_tangent_root_is_detached_and_warned(self, precision):
        with backend_state("torch", precision):
            dtype = _dtype(precision)
            a = torch.nn.Parameter(torch.tensor(1.0, dtype=dtype))
            geometry = _ParabolaHeightGeometry(a)

            # Ray along x, tangent to z = x**2 at the vertex: F(t) = t**2,
            # so t* = 0 is an exact double root with F_t(0) = 0.
            rays = RealRays(
                x=be.array([0.0]),
                y=be.array([0.0]),
                z=be.array([0.0]),
                L=be.array([1.0]),
                M=be.array([0.0]),
                N=be.array([0.0]),
                intensity=be.array([1.0]),
                wavelength=be.array([0.587]),
            )

            with pytest.warns(RuntimeWarning, match="tangent/grazing"):
                t = geometry.distance(rays)

            # The forward root remains available; the implicit path is
            # rejected: no clipped finite derivative is presented as exact.
            assert float(be.to_numpy(t)[0]) == pytest.approx(0.0, abs=1e-8)
            assert not t.requires_grad
            assert a.grad is None

    @pytest.mark.parametrize("precision", ["float64", "float32"])
    def test_tangent_forward_root_in_numpy(self, precision):
        with backend_state("numpy", precision):
            geometry = _ParabolaHeightGeometry(be.array(1.0))
            rays = RealRays(
                x=be.array([0.0]),
                y=be.array([0.0]),
                z=be.array([0.0]),
                L=be.array([1.0]),
                M=be.array([0.0]),
                N=be.array([0.0]),
                intensity=be.array([1.0]),
                wavelength=be.array([0.587]),
            )
            t = geometry.distance(rays)
            assert float(be.to_numpy(t)[0]) == pytest.approx(0.0, abs=1e-8)


# ----------------------------------------------------------------------
# D. Ordinary regular root regression (both dtypes)
# ----------------------------------------------------------------------


class TestRegularRootRegression:
    @pytest.mark.parametrize("precision", ["float64", "float32"])
    def test_regular_root_gradient_matches_analytic(self, precision):
        """For z = c*sqrt(x) and an axial ray, dt*/dc = sqrt(x0) exactly."""
        with backend_state("torch", precision):
            dtype = _dtype(precision)
            c = torch.nn.Parameter(torch.tensor(0.7, dtype=dtype))
            geometry = _SqrtHeightGeometry(c)
            rays = _axial_rays([4.0])

            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                t = geometry.distance(rays)

            assert t.requires_grad
            t.sum().backward()
            tol = 1e-10 if precision == "float64" else 1e-4
            assert float(c.grad.detach().item()) == pytest.approx(2.0, abs=tol)


# ----------------------------------------------------------------------
# P0.2 -- dtype-aware |n_z| validity
# ----------------------------------------------------------------------


class TestNzThreshold:
    X0 = 0.5  # dt*/ds = x0 for the ramp surface

    def _trace_ramp(self, slope_value, dtype):
        s = torch.nn.Parameter(torch.tensor(slope_value, dtype=dtype))
        geometry = _RampHeightGeometry(s)
        rays = _axial_rays([self.X0])
        return s, geometry.distance(rays)

    @pytest.mark.parametrize("precision", ["float64", "float32"])
    def test_steep_regular_surface_stays_differentiable(self, precision):
        with backend_state("torch", precision):
            dtype = _dtype(precision)
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                s, t = self._trace_ramp(100.0, dtype)

            assert t.requires_grad
            t.sum().backward()
            tol = 1e-8 if precision == "float64" else 1e-3
            assert float(s.grad.detach().item()) == pytest.approx(self.X0, abs=tol)

    @pytest.mark.parametrize(
        "precision,slope",
        [
            # |n_z| ~ 1e-7: below the float32 threshold (~3.8e-6), far above
            # the float64 one (~7.1e-15).
            ("float32", 1.0e7),
            # |n_z| ~ 1e-16: below the float64 threshold.
            ("float64", 1.0e16),
        ],
    )
    def test_locally_vertical_surface_is_rejected_without_nan(
        self, precision, slope
    ):
        with backend_state("torch", precision):
            dtype = _dtype(precision)
            with pytest.warns(RuntimeWarning, match="n_z"):
                s, t = self._trace_ramp(slope, dtype)

            assert bool(torch.isfinite(t).all())
            assert not t.requires_grad
            assert s.grad is None

    def test_same_slope_is_dtype_dependent(self):
        """|n_z| ~ 1e-7 is regular in float64 but invalid in float32."""
        slope = 1.0e7
        with backend_state("torch", "float64"):
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                s, t = self._trace_ramp(slope, torch.float64)
            t.sum().backward()
            assert float(s.grad.detach().item()) == pytest.approx(self.X0, abs=1e-6)

        with backend_state("torch", "float32"):
            with pytest.warns(RuntimeWarning, match="n_z"):
                _, t = self._trace_ramp(slope, torch.float32)
            assert not t.requires_grad
