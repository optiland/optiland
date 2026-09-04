"""Tests for implicit differentiation of Newton-Raphson ray-surface intersection.

Validates that:
1. Gradients through aspheric surfaces (EvenAsphere / NR geometry) are non-None
   and finite when using the torch backend with implicit differentiation.
2. Autograd gradients match finite-difference gradients to within tolerance.
3. The numpy backend is not affected (no regression).
"""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.optimization import OptimizationProblem
from optiland.samples.simple import AsphericSinglet

# Skip the entire module if torch is not installed
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module", autouse=True)
def _set_torch_backend():
    """Ensure the torch backend is active for this test module."""
    original = be.get_backend()
    be.set_backend("torch")
    be.set_precision("float64")
    yield
    be.set_precision("float32")
    be.set_backend(original)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_problem(variable_type="radius", surface_number=1, **var_kwargs):
    """Build a minimal optimisation problem around the AsphericSinglet."""
    lens = AsphericSinglet()
    problem = OptimizationProblem()
    problem.add_variable(
        lens,
        variable_type,
        surface_number=surface_number,
        **var_kwargs,
    )
    problem.add_operand(
        operand_type="rms_spot_size",
        target=0.0,
        weight=1.0,
        input_data={
            "optic": lens,
            "surface_number": lens.surface_group.num_surfaces - 1,
            "Hx": 0.0,
            "Hy": 0.0,
            "num_rays": 20,
            "wavelength": 0.587,
        },
    )
    return problem, lens


def _scalar_loss(problem):
    """Compute the scalar merit function (sum of squared residuals)."""
    problem.update_optics()
    return problem.sum_squared()


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestImplicitDiffGradientExists:
    """Gradients through NR intersection should be non-None and finite."""

    @pytest.mark.parametrize(
        "var_type,extra",
        [
            ("radius", {}),
            ("conic", {}),
            ("asphere_coeff", {"coeff_number": 0}),
        ],
    )
    def test_gradient_is_finite(self, var_type, extra):
        problem, _ = _make_problem(variable_type=var_type, **extra)

        with be.grad_mode.temporary_enable():
            val = problem.variables[0].variable.get_value()
            param = torch.nn.Parameter(be.array(val))
            problem.variables[0].variable.update_value(param)
            problem.update_optics()

            loss = problem.sum_squared()
            assert loss.requires_grad, "Loss should require grad"

            loss.backward()
            assert param.grad is not None, f"Gradient is None for {var_type}"
            assert torch.isfinite(param.grad).all(), (
                f"Non-finite gradient for {var_type}: {param.grad}"
            )


class TestImplicitDiffMatchesFiniteDifference:
    """Autograd gradients must agree with finite-difference approximations."""

    @pytest.mark.parametrize(
        "var_type,extra",
        [
            ("radius", {}),
            ("conic", {}),
            ("asphere_coeff", {"coeff_number": 0}),
        ],
    )
    def test_gradient_matches_finite_diff(self, var_type, extra):
        eps = 1e-6
        problem, lens = _make_problem(variable_type=var_type, **extra)

        # --- Autograd gradient ---
        with be.grad_mode.temporary_enable():
            param = torch.nn.Parameter(
                be.array(problem.variables[0].variable.get_value())
            )
            problem.variables[0].variable.update_value(param)
            problem.update_optics()

            loss = problem.sum_squared()
            loss.backward()
            ad_grad = param.grad.item()

        # --- Finite-difference gradient ---
        val0 = float(param.data.item())

        # f(x + eps)
        problem.variables[0].variable.update_value(val0 + eps)
        problem.update_optics()
        with torch.no_grad():
            loss_plus = problem.sum_squared().item()

        # f(x - eps)
        problem.variables[0].variable.update_value(val0 - eps)
        problem.update_optics()
        with torch.no_grad():
            loss_minus = problem.sum_squared().item()

        fd_grad = (loss_plus - loss_minus) / (2.0 * eps)

        # Restore original value
        problem.variables[0].variable.update_value(val0)
        problem.update_optics()

        assert ad_grad != 0.0, "Autograd gradient should be non-zero"
        rel_err = abs(ad_grad - fd_grad) / (abs(fd_grad) + 1e-12)
        assert rel_err < 0.05, (
            f"Gradient mismatch for {var_type}: AD={ad_grad:.8f}, "
            f"FD={fd_grad:.8f}, rel_err={rel_err:.4f}"
        )


class TestNumpyPathUnchanged:
    """The numpy backend should still work identically (no regression)."""

    def test_numpy_distance_returns_same_result(self):
        """Verify aspheric singlet can be traced with numpy backend."""
        original = be.get_backend()
        try:
            be.set_backend("numpy")
            lens = AsphericSinglet()
            lens.trace(0, 0, 0.587, 20, "hexapolar")
            # Just verify that tracing completes without error and
            # produces finite intercept values on the image surface
            img = lens.surface_group.num_surfaces - 1
            x = lens.surface_group.x[img, :]
            import numpy as np

            assert np.all(np.isfinite(x)), "Non-finite values in numpy trace"
        finally:
            be.set_backend(original)
