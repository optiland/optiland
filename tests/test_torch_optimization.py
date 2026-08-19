from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.optimization import (
    OptimizationProblem,
    TorchAdamOptimizer,
    TorchSGDOptimizer,
)
from optiland.samples.objectives import CookeTriplet


def setup_problem(
    add_variable=True,
    add_operand=True,
    min_val=1.0,
    max_val=10.0,
    target=12.0,
    initial_value=5.0,
):
    """
    Helper function to set up a standard optimization problem for testing.

    Args:
        add_variable (bool): Whether to add a variable to the problem.
        add_operand (bool): Whether to add an operand to the problem.
        min_val (float, optional): The minimum bound for the variable.
        max_val (float, optional): The maximum bound for the variable.
        target (float): The target value for the operand.
        initial_value (float): The initial value for the variable.

    Returns:
        tuple: A tuple containing the OptimizationProblem and the lens object.
    """
    lens = CookeTriplet()
    problem = OptimizationProblem()

    # Set the initial state before adding variables to ensure it's captured
    if add_variable:
        lens.surfaces[1].radius = initial_value
        problem.add_variable(
            lens,
            "radius",
            surface_number=1,
            min_val=min_val,
            max_val=max_val,
        )

    if add_operand:
        problem.add_operand(
            operand_type="f2",
            target=target,
            weight=1.0,
            input_data={"optic": lens},
        )

    # Manually update optics to reflect initial state
    problem.update_optics()
    return problem, lens


@pytest.fixture(scope="module", autouse=True)
def set_torch_backend():
    """
    Fixture to ensure the torch backend is set for all tests in this file.
    It will set the backend to torch before the tests run and revert to the
    original backend after all tests in the module are completed.
    """
    original_backend = be.get_backend()
    be.set_backend("torch")
    yield
    be.set_backend(original_backend)


class TestTorchBaseOptimizerSetup:
    """
    Tests focused on the setup and edge cases of the TorchBaseOptimizer,
    which are independent of the specific optimizer algorithm used.
    """

    def test_init_raises_error_if_backend_not_torch(self):
        """
        Ensures that initializing a Torch optimizer without the 'torch'
        backend raises a RuntimeError.
        """
        # This test temporarily switches the backend and must restore it
        # to not affect other tests, as the module-scoped fixture will
        # not run between tests.
        original_backend = be.get_backend()
        try:
            be.set_backend("numpy")
            # We need to re-run setup_problem under the numpy backend
            problem, _ = setup_problem()
            with pytest.raises(RuntimeError, match="requires the 'torch' backend"):
                TorchAdamOptimizer(problem)
        finally:
            # Restore the backend for subsequent tests
            be.set_backend(original_backend)

    def test_init_enables_gradient_tracking_with_warning(self):
        """
        Tests that a warning is issued if gradient tracking is disabled
        when the optimizer is initialized, and that it gets enabled.
        """
        # Create the problem first. This may enable grad mode.
        problem, _ = setup_problem()

        # Now, explicitly disable grad mode to test the optimizer's __init__.
        if hasattr(be.grad_mode, "disable"):
            be.grad_mode.disable()
        assert not be.grad_mode.requires_grad

        # The optimizer's __init__ should now issue a warning and enable grad mode.
        with pytest.warns(
            UserWarning, match="Gradient tracking is enabled for PyTorch"
        ):
            optimizer = TorchAdamOptimizer(problem)

        # Check that gradients are now enabled by the optimizer
        assert be.grad_mode.requires_grad

        # Cleanup: Ensure grad mode is enabled for subsequent tests
        if hasattr(be.grad_mode, "enable"):
            be.grad_mode.enable()


@pytest.mark.parametrize("optimizer_class", [TorchAdamOptimizer, TorchSGDOptimizer])
class TestTorchOptimizers:
    """
    A parametrized test suite for all concrete Torch optimizer implementations.
    """

    def test_optimize_successfully_reduces_loss(self, optimizer_class):
        """
        Verifies that the optimizer successfully reduces the loss function value
        over a number of steps.
        """
        problem, _ = setup_problem()
        initial_loss = problem.sum_squared().item()

        optimizer = optimizer_class(problem)
        result = optimizer.optimize(n_steps=50, disp=False)

        assert result.fun < initial_loss
        assert be.isclose(be.array(result.fun), be.array(problem.sum_squared().item()))

    def test_optimize_no_operands_returns_zero_loss(self, optimizer_class):
        """
        Tests that optimization with no operands results in a loss of zero,
        as there is nothing to optimize.
        """
        problem, _ = setup_problem(add_operand=False)
        optimizer = optimizer_class(problem)
        result = optimizer.optimize(n_steps=10, disp=False)
        assert be.isclose(be.array(result.fun), be.array(0.0))

    def test_optimize_no_variables_raises_error(self, optimizer_class):
        """
        Tests that attempting to optimize a problem with no variables raises
        a ValueError, as there are no parameters for the optimizer to act on.
        """
        problem, _ = setup_problem(add_variable=False)
        optimizer = optimizer_class(problem)
        with pytest.raises(ValueError):
            optimizer.optimize(n_steps=10)

    def test_optimize_with_zero_steps_makes_no_change(self, optimizer_class):
        """
        Tests that running the optimizer for zero steps results in no change
        to the loss function or the variable values.
        """
        problem, _ = setup_problem(initial_value=5.0)
        initial_loss = problem.sum_squared().item()
        initial_param_val = problem.variables[0].variable.get_value()

        optimizer = optimizer_class(problem)
        result = optimizer.optimize(n_steps=0, disp=False)

        assert be.isclose(be.array(result.fun), be.array(initial_loss))
        final_param_val = problem.variables[0].variable.get_value()
        assert be.isclose(initial_param_val, final_param_val)

    def test_callback_is_invoked_at_each_step(self, optimizer_class):
        """
        Ensures the callback function is called exactly n_steps times with the
        correct step index and a float loss value.
        """
        problem, _ = setup_problem()
        optimizer = optimizer_class(problem)
        n_steps = 10
        history = []

        def callback_fn(step, loss):
            history.append((step, loss))

        optimizer.optimize(n_steps=n_steps, callback=callback_fn, disp=False)

        assert len(history) == n_steps
        for i, (step, loss) in enumerate(history):
            assert step == i
            assert isinstance(loss, float)
            if (
                i > 0
            ):  # Loss is not guaranteed to decrease every step for all optimizers
                assert loss >= 0.0

    @pytest.mark.parametrize("disp, should_have_output", [(True, True), (False, False)])
    def test_display_output_controlled_by_disp_flag(
        self, capsys, optimizer_class, disp, should_have_output
    ):
        """
        Tests that console output is correctly controlled by the 'disp' flag.
        """
        problem, _ = setup_problem()
        optimizer = optimizer_class(problem)
        optimizer.optimize(n_steps=1, disp=disp)
        captured = capsys.readouterr()

        if should_have_output:
            assert "Loss" in captured.out
        else:
            assert "Loss" not in captured.out


class TestTorchOptimizerScaledSpace:
    """
    Tests that verify the Torch optimizers work in scaled parameter space,
    consistent with the bounds from var.bounds.

    This prevents the bug where raw parameters (e.g. radius=15.0) were clamped
    by scaled bounds (e.g. [-0.9, -0.7]), corrupting the value and producing
    NaN loss.
    """

    def test_params_initialized_in_scaled_space(self):
        """
        Optimizer params must match var.value (scaled), not
        var.variable.get_value() (raw).

        For a RadiusVariable with LinearScaler(factor=1/100, offset=-1.0),
        raw=5.0 should become scaled = 5.0 * 0.01 - 1.0 = -0.95.
        """
        problem, _ = setup_problem(initial_value=5.0, min_val=1.0, max_val=10.0)
        optimizer = TorchAdamOptimizer(problem)

        # The optimizer param should be the scaled value, not the raw value
        scaled_param = optimizer.params[0].item()
        raw_value = problem.variables[0].variable.get_value()
        expected_scaled = problem.variables[0].value

        assert abs(scaled_param - expected_scaled) < 1e-10, (
            f"Param {scaled_param} should equal scaled value {expected_scaled}, "
            f"not raw value {raw_value}"
        )
        # Sanity: scaled and raw should differ for this scaler
        assert abs(scaled_param - raw_value) > 0.1, (
            "Scaled and raw values should be different"
        )

    def test_bounds_consistent_with_params(self):
        """
        After _apply_bounds(), the parameter must remain in the valid scaled
        range — not be corrupted by a space mismatch.
        """
        problem, _ = setup_problem(initial_value=5.0, min_val=1.0, max_val=10.0)
        optimizer = TorchAdamOptimizer(problem)

        min_bound, max_bound = problem.variables[0].bounds

        # Apply bounds and verify param stays in range (with float tolerance)
        optimizer._apply_bounds()
        param_val = optimizer.params[0].item()

        tol = 1e-6
        assert min_bound - tol <= param_val <= max_bound + tol, (
            f"Param {param_val} outside scaled bounds [{min_bound}, {max_bound}]"
        )

    def test_bounded_optimization_no_nan(self):
        """
        Regression test: optimization with bounded variables must not produce
        NaN loss. This was the exact failure mode when raw params were clamped
        by scaled bounds.
        """
        import math

        problem, lens = setup_problem(
            initial_value=5.0,
            min_val=1.0,
            max_val=10.0,
            target=12.0,
        )
        optimizer = TorchAdamOptimizer(problem)
        result = optimizer.optimize(n_steps=20, disp=False)

        assert not math.isnan(result.fun), f"Loss should not be NaN, got {result.fun}"
        assert result.fun >= 0.0, f"Loss should be non-negative, got {result.fun}"

    def test_bounded_variable_stays_in_physical_range(self):
        """
        After optimization with bounds [1, 10], the actual radius on the
        optic must be within that physical range — not corrupted to a scaled
        value like -0.7.
        """
        problem, lens = setup_problem(
            initial_value=5.0,
            min_val=1.0,
            max_val=10.0,
            target=12.0,
        )
        optimizer = TorchAdamOptimizer(problem)
        optimizer.optimize(n_steps=10, disp=False)

        # Get the actual physical radius from the optic
        raw_radius = problem.variables[0].variable.get_value()
        if hasattr(raw_radius, "item"):
            raw_radius = raw_radius.item()

        # Allow small floating-point overshoot from inverse-scaling
        tol = 1e-4
        assert 1.0 - tol <= raw_radius <= 10.0 + tol, (
            f"Physical radius {raw_radius} outside bounds [1.0, 10.0]. "
            "This suggests a scaled/unscaled space mismatch."
        )

    def test_optimizer_with_real_material_no_nan(self):
        """
        End-to-end regression test using Material("N-BK7") with bounded
        variables — the exact combination that triggered both the NaN bug
        (scaled/unscaled mismatch) and the RuntimeError (stale graph in
        cached material tensors).
        """
        import math

        from optiland.optic import Optic

        lens = Optic()
        lens.surfaces.add(index=0, thickness=be.inf)
        lens.surfaces.add(
            index=1,
            thickness=7,
            radius=15,
            material="N-BK7",
            is_stop=True,
        )
        lens.surfaces.add(index=2, thickness=30, radius=-1000)
        lens.surfaces.add(index=3)
        lens.set_aperture(aperture_type="EPD", value=15)
        lens.fields.set_type(field_type="angle")
        lens.fields.add(y=0)
        lens.wavelengths.add(value=0.55, is_primary=True)

        problem = OptimizationProblem()
        problem.add_operand(
            operand_type="f2",
            target=50,
            weight=1,
            input_data={"optic": lens},
        )
        problem.add_variable(
            lens,
            "radius",
            surface_number=1,
            min_val=10,
            max_val=30,
        )
        problem.add_variable(lens, "thickness", surface_number=2)

        initial_loss = problem.sum_squared().item()
        optimizer = TorchAdamOptimizer(problem)
        result = optimizer.optimize(n_steps=10, disp=False)

        assert not math.isnan(result.fun), (
            f"Loss should not be NaN with Material('N-BK7') and bounded "
            f"variables, got {result.fun}"
        )
        assert result.fun < initial_loss, (
            "Optimizer should reduce loss for the real-material bounded case; "
            "otherwise gradients may be disconnected."
        )

    def test_legacy_array_behavior_breaks_gradient_flow(self, monkeypatch):
        """Regression guard for the historical graph-break behavior.

        The legacy TorchBackend.array implementation converted lists containing
        tensors with torch.tensor(...), which detached them from autograd.
        This test monkeypatches that legacy behavior and verifies gradients on
        optimizer parameters become None, proving learning was broken.
        """
        import numpy as np
        import torch

        from optiland.backend.torch_backend import TorchBackend
        from optiland.optic import Optic

        def _legacy_array(self, x):
            if isinstance(x, torch.Tensor):
                return x
            if (
                isinstance(x, (list, tuple))
                and len(x) > 0
                and isinstance(x[0], np.ndarray)
            ):
                x = np.array(x)
            return torch.tensor(
                x,
                device=self._device(),
                dtype=self._dtype(),
                requires_grad=self._grad(),
            )

        monkeypatch.setattr(TorchBackend, "array", _legacy_array)

        lens = Optic()
        lens.add_surface(index=0, thickness=be.inf)
        lens.add_surface(
            index=1,
            thickness=7,
            radius=15,
            material="N-BK7",
            is_stop=True,
        )
        lens.add_surface(index=2, thickness=30, radius=-1000)
        lens.add_surface(index=3)
        lens.set_aperture(aperture_type="EPD", value=15)
        lens.set_field_type(field_type="angle")
        lens.add_field(y=0)
        lens.add_wavelength(value=0.55, is_primary=True)

        problem = OptimizationProblem()
        problem.add_operand(
            operand_type="f2",
            target=50,
            weight=1,
            input_data={"optic": lens},
        )
        problem.add_variable(
            lens,
            "radius",
            surface_number=1,
            min_val=10,
            max_val=30,
        )
        problem.add_variable(lens, "thickness", surface_number=2)

        optimizer = TorchAdamOptimizer(problem)
        torch_opt, _ = optimizer._create_optimizer_and_scheduler(lr=1e-2, gamma=0.99)

        torch_opt.zero_grad()
        for k, param in enumerate(optimizer.params):
            problem.variables[k].update(param)
        problem.update_optics()
        loss = problem.sum_squared()
        loss.backward()

        assert all(param.grad is None for param in optimizer.params), (
            "Legacy array behavior should detach tensors and break gradients; "
            "expected all parameter grads to be None."
        )
