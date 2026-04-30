"""Direct tests of Newton-Raphson implicit-diff distance() behavior.

These tests target the primitive that was changed:
``NewtonRaphsonGeometry.distance(rays)``.
"""

from __future__ import annotations

import pytest

import optiland.backend as be

from .nr_implicit_test_utils import (
    assert_fd_is_stable,
    backend_state,
    build_reference_even_asphere,
    build_reference_rays,
    finite_diff_reference,
)
from .utils import assert_allclose

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _torch_backend_float64():
    with backend_state("torch", precision="float64"):
        yield


def _distance_for_surface_param(param_name: str, param_value, *, off_axis: bool) -> torch.Tensor:
    geometry = build_reference_even_asphere()

    if param_name == "radius":
        geometry.radius = param_value
    elif param_name == "conic":
        geometry.k = param_value
    elif param_name == "asphere_coeff":
        coeffs = list(geometry.coefficients)
        coeffs[0] = param_value
        geometry.coefficients = coeffs
    else:
        raise ValueError(f"Unknown parameter name: {param_name}")

    rays = build_reference_rays(off_axis=off_axis)
    return geometry.distance(rays)[0]


def _distance_for_ray_x(x0, *, off_axis: bool) -> torch.Tensor:
    geometry = build_reference_even_asphere()
    rays = build_reference_rays(off_axis=off_axis, x_override=x0)
    return geometry.distance(rays)[0]


class TestDistanceSurfaceParameterGradients:
    """AD-vs-FD checks for distance() w.r.t. surface parameters."""

    @pytest.mark.parametrize(
        "param_name,base_value,grad_abs_tol,grad_rel_tol,fd_abs_tol,fd_rel_tol",
        [
            ("radius", 20.0, 1e-9, 5e-3, 1e-9, 1e-2),
            # Conic derivative is very small in this configuration, so use a
            # meaningful absolute tolerance in addition to relative tolerance.
            ("conic", -0.35, 8e-10, 2e-1, 5e-10, 8e-2),
            ("asphere_coeff", -2.248851e-4, 1e-8, 5e-3, 1e-8, 1e-2),
        ],
    )
    def test_distance_grad_matches_fd_on_axis(
        self,
        param_name,
        base_value,
        grad_abs_tol,
        grad_rel_tol,
        fd_abs_tol,
        fd_rel_tol,
    ):
        param = torch.nn.Parameter(torch.tensor(base_value, dtype=torch.float64))
        t = _distance_for_surface_param(param_name, param, off_axis=False)
        assert t.requires_grad
        t.backward()
        ad_grad = float(param.grad.detach().item())

        def scalar_distance(value: float) -> float:
            with torch.no_grad():
                return float(
                    _distance_for_surface_param(param_name, value, off_axis=False)
                    .detach()
                    .item()
                )

        # Two epsilons reduce the risk of cancellation or solver-noise artifacts.
        fd_ref, fd_values = finite_diff_reference(
            scalar_distance,
            base_value,
            epsilons=(2e-6, 1e-6),
        )
        assert_fd_is_stable(fd_values, abs_tol=fd_abs_tol, rel_tol=fd_rel_tol)

        assert abs(ad_grad - fd_ref) <= max(grad_abs_tol, grad_rel_tol * abs(fd_ref)), (
            f"AD/FD mismatch for {param_name}: AD={ad_grad:.12e}, "
            f"FD={fd_ref:.12e}, FD samples={fd_values}"
        )

    def test_distance_grad_matches_fd_off_axis_for_radius(self):
        base_value = 20.0
        param = torch.nn.Parameter(torch.tensor(base_value, dtype=torch.float64))
        t = _distance_for_surface_param("radius", param, off_axis=True)
        t.backward()
        ad_grad = float(param.grad.detach().item())

        def scalar_distance(value: float) -> float:
            with torch.no_grad():
                return float(
                    _distance_for_surface_param("radius", value, off_axis=True)
                    .detach()
                    .item()
                )

        fd_ref, fd_values = finite_diff_reference(
            scalar_distance,
            base_value,
            epsilons=(2e-6, 1e-6),
        )
        assert_fd_is_stable(fd_values, abs_tol=1e-8, rel_tol=1e-2)

        assert abs(ad_grad - fd_ref) <= max(1e-8, 1e-2 * abs(fd_ref)), (
            f"Off-axis AD/FD mismatch for radius: AD={ad_grad:.12e}, "
            f"FD={fd_ref:.12e}, FD samples={fd_values}"
        )


class TestDistanceRayStateGradients:
    """AD-vs-FD checks for distance() w.r.t. ray-state variables."""

    def test_distance_grad_matches_fd_wrt_initial_x(self):
        x0 = 0.31
        x_param = torch.nn.Parameter(torch.tensor(x0, dtype=torch.float64))
        t = _distance_for_ray_x(x_param, off_axis=True)
        t.backward()
        ad_grad = float(x_param.grad.detach().item())

        def scalar_distance(value: float) -> float:
            with torch.no_grad():
                return float(_distance_for_ray_x(value, off_axis=True).detach().item())

        fd_ref, fd_values = finite_diff_reference(
            scalar_distance,
            x0,
            epsilons=(2e-6, 1e-6),
        )
        assert_fd_is_stable(fd_values, abs_tol=1e-8, rel_tol=1e-2)

        assert abs(ad_grad - fd_ref) <= max(1e-8, 1e-2 * abs(fd_ref)), (
            f"AD/FD mismatch for ray x0: AD={ad_grad:.12e}, "
            f"FD={fd_ref:.12e}, FD samples={fd_values}"
        )


class TestForwardConsistency:
    """Differentiable distance() should preserve the primal converged root."""

    @pytest.mark.parametrize("off_axis", [False, True])
    def test_diff_forward_matches_primal_root(self, off_axis):
        radius_param = torch.nn.Parameter(torch.tensor(20.0, dtype=torch.float64))
        geometry = build_reference_even_asphere()
        geometry.radius = radius_param
        rays = build_reference_rays(off_axis=off_axis)

        with torch.no_grad():
            t_primal = geometry._solve_distance_primal(rays)

        t_diff = geometry.distance(rays)
        assert t_diff.requires_grad
        assert_allclose(t_diff.detach(), t_primal, rtol=0.0, atol=1e-12)


class TestHigherOrderContract:
    """The implicit correction contract is first-order gradient accuracy only."""

    def test_second_derivative_not_contractually_matched(self):
        # This oblique configuration accentuates the difference between
        # double-backward through the implicit-correction graph and a finite-
        # difference estimate of d/dR(dt/dR). We keep this as an explicit
        # first-order-only contract test.

        def distance_from_radius(radius_value):
            geometry = build_reference_even_asphere()
            geometry.radius = radius_value
            rays = build_reference_rays(
                off_axis=True,
                x_override=1.0,
                y_override=-0.8,
                L_override=0.28,
                M_override=-0.18,
            )
            return geometry.distance(rays)[0]

        radius_param = torch.nn.Parameter(torch.tensor(20.0, dtype=torch.float64))
        t = distance_from_radius(radius_param)
        d1 = torch.autograd.grad(t, radius_param, create_graph=True)[0]
        d2_autograd = float(torch.autograd.grad(d1, radius_param)[0].detach().item())

        def first_derivative_at(radius_value: float) -> float:
            p = torch.nn.Parameter(torch.tensor(radius_value, dtype=torch.float64))
            t_local = distance_from_radius(p)
            d1_local = torch.autograd.grad(t_local, p)[0]
            return float(d1_local.detach().item())

        d2_fd = (first_derivative_at(20.0 + 5e-5) - first_derivative_at(20.0 - 5e-5)) / (
            2.0 * 5e-5
        )

        rel_err = abs(d2_autograd - d2_fd) / (abs(d2_fd) + 1e-15)
        assert rel_err > 2e-2, (
            "Second-order behavior appears too close to finite differences for a "
            "first-order-only contract. If higher-order support was intentionally "
            "added, update this test and the Newton-Raphson distance contract."
        )
