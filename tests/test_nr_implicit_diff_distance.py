"""Direct tests of Newton-Raphson implicit-diff distance() behavior.

These tests target the primitive that was changed:
``NewtonRaphsonGeometry.distance(rays)``.
"""

from __future__ import annotations

import warnings

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
            primal = geometry._solve_distance_primal(rays)

        t_diff = geometry.distance(rays)
        assert t_diff.requires_grad
        # The implicit path preserves the converged primal solution within
        # solver tolerance and applies one final Newton correction -- it is
        # not bitwise identical to the primal root.
        assert_allclose(t_diff.detach(), primal.t, rtol=0.0, atol=1e-12)

    def test_primal_result_reports_convergence_state(self):
        geometry = build_reference_even_asphere()
        rays = build_reference_rays(off_axis=True)

        with torch.no_grad():
            primal = geometry._solve_distance_primal(rays)

        assert bool(primal.converged.all())
        assert primal.iterations >= 1
        assert float(primal.residual.abs().max()) < geometry.tol

    @pytest.mark.parametrize("precision", ["float64", "float32"])
    def test_converges_in_both_precisions(self, precision):
        """The default tol=1e-10 sits below float32 round-off.

        Without a dtype-aware floor under the tolerance, every ray would be
        classified non-converged in float32 -- suppressing its implicit
        gradient and warning on every call -- even though the root is as good
        as float32 allows.
        """
        with backend_state("torch", precision):
            geometry = build_reference_even_asphere()
            rays = build_reference_rays(off_axis=True)

            with torch.no_grad():
                primal = geometry._solve_distance_primal(rays)

            assert bool(primal.converged.all()), (
                f"{precision} solve reported non-convergence at residual "
                f"{float(primal.residual.abs().max()):.3e}"
            )
            assert bool(torch.isfinite(primal.t).all())

    def test_float32_does_not_warn_about_convergence(self):
        with backend_state("torch", "float32"):
            radius_param = torch.nn.Parameter(
                torch.tensor(20.0, dtype=torch.float32)
            )
            geometry = build_reference_even_asphere()
            geometry.radius = radius_param
            rays = build_reference_rays(off_axis=True)

            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                t = geometry.distance(rays)

            assert bool(torch.isfinite(t).all())


class TestHigherOrderContract:
    """The implicit correction contract is first-order gradient accuracy only.

    The supported contract is: ``distance()`` returns a value whose *first*
    derivative matches the implicit-function-theorem result. Double backward
    is permitted to run (so composing the geometry into a larger graph does
    not raise), but its value is not part of the contract because the
    denominator ``dF/dt`` is intentionally detached.

    This deliberately does **not** assert that the second derivative is
    *wrong*: encoding "must stay inaccurate" would force the implementation to
    remain limited if higher-order support is added later.
    """

    def _distance_from_radius(self, radius_value):
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

    def test_first_derivative_is_the_supported_contract(self):
        radius_param = torch.nn.Parameter(torch.tensor(20.0, dtype=torch.float64))
        t = self._distance_from_radius(radius_param)
        (d1,) = torch.autograd.grad(t, radius_param, create_graph=True)

        assert torch.isfinite(d1)
        assert float(d1.detach().abs()) > 0.0

        def distance_at(radius_value: float) -> float:
            with torch.no_grad():
                return float(
                    self._distance_from_radius(
                        torch.tensor(radius_value, dtype=torch.float64)
                    )
                )

        # First order is validated against central differences of the *value* --
        # this is the part of the contract the implementation guarantees.
        h = 5e-6
        fd = (distance_at(20.0 + h) - distance_at(20.0 - h)) / (2.0 * h)
        ad = float(d1.detach())
        assert abs(ad - fd) <= max(1e-8, 1e-2 * abs(fd)), (
            f"first-order AD/FD mismatch: AD={ad:.12e}, FD={fd:.12e}"
        )

    def test_double_backward_runs_and_stays_finite(self):
        # Higher-order derivatives are not guaranteed to match the unrolled
        # Newton system, but they must not raise or produce NaN/Inf.
        radius_param = torch.nn.Parameter(torch.tensor(20.0, dtype=torch.float64))
        t = self._distance_from_radius(radius_param)
        (d1,) = torch.autograd.grad(t, radius_param, create_graph=True)
        (d2,) = torch.autograd.grad(d1, radius_param)
        assert torch.isfinite(d2)
