from __future__ import annotations

import pytest

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None
    nn = None
    optim = None


import optiland.backend as be
from optiland.ml.wrappers import OpticalSystemModule
from optiland.optimization import OptimizationProblem
from optiland.samples.objectives import CookeTriplet


def setup_problem(
    add_variable=True,
    add_operand=True,
    min_val=1.0,
    max_val=10.0,
    target=12.0,
):
    """
    Helper function to set up a standard optimization problem.
    This is borrowed from the existing test suite for consistency.
    """
    lens = CookeTriplet()
    problem = OptimizationProblem()
    if add_variable:
        problem.add_variable(
            lens,
            "thickness",
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
    return problem, lens


@pytest.fixture(scope="module", autouse=True)
def set_torch_backend():
    """
    Fixture to ensure the torch backend is set for all tests in this file.
    It will set the backend to torch before the tests run and revert to the
    original backend after all tests are completed.
    """
    original_backend = be.get_backend()
    be.set_backend("torch")
    yield
    be.set_backend(original_backend)


class TestOpticalSystemModule:
    """
    Tests for the OpticalSystemModule wrapper class.
    """

    def test_init_runtime_error_wrong_backend(self):
        """
        Test that a RuntimeError is raised if the backend is not 'torch'.
        """
        original_backend = be.get_backend()
        be.set_backend("numpy")
        problem, lens = setup_problem()
        with pytest.raises(RuntimeError) as e:
            _ = OpticalSystemModule(lens, problem)
        assert "requires the 'torch' backend" in str(e.value)
        be.set_backend(original_backend)  # Reset backend for other tests

    def test_init_enables_gradients(self):
        """
        Test that a warning is issued and gradients are enabled if they are
        initially disabled.
        """
        problem, lens = setup_problem()
        be.grad_mode.disable()
        assert not be.grad_mode.requires_grad

        with pytest.warns(UserWarning, match="Gradient tracking is enabled"):
            _ = OpticalSystemModule(lens, problem)

        assert be.grad_mode.requires_grad
        be.grad_mode.enable()  # Ensure it's enabled for subsequent tests

    def test_init_parameter_creation(self):
        """
        Test that the nn.ParameterList is created correctly upon initialization.
        """
        problem, lens = setup_problem()
        module = OpticalSystemModule(lens, problem)

        assert isinstance(module.params, nn.ParameterList)
        assert len(module.params) == len(problem.variables)

        initial_val = problem.variables[0].value
        assert be.isclose(module.params[0].data, be.array(initial_val))

    def test_default_loss_function(self):
        """
        Test that the default loss function correctly computes the sum of squares.
        """
        problem, lens = setup_problem()
        module = OpticalSystemModule(lens, problem)

        expected_loss = problem.sum_squared()
        actual_loss = module._default_loss()

        assert be.isclose(expected_loss, actual_loss)

    def test_sync_params_to_problem_and_bounds(self):
        """
        Test that _sync_params_to_problem correctly updates the underlying
        problem variables and applies bounds.
        """
        min_b, max_b = 5.0, 15.0
        problem, lens = setup_problem(min_val=min_b, max_val=max_b)
        module = OpticalSystemModule(lens, problem)

        # Test clamping to the maximum bound
        with torch.no_grad():
            module.params[0].data.fill_(20.0)

        module._sync_params_to_problem()
        module.apply_bounds()

        assert be.isclose(module.params[0].data, be.array(max_b))

        # Test clamping to the minimum bound
        with torch.no_grad():
            module.params[0].data.fill_(1.0)

        module._sync_params_to_problem()
        module.apply_bounds()

        assert be.isclose(module.params[0].data, be.array(min_b))

    def test_forward_pass_and_optimization(self):
        """
        Test the forward pass and ensure that gradients are computed, allowing
        for optimization.
        """
        problem, lens = setup_problem()
        module = OpticalSystemModule(lens, problem)

        initial_loss = module.forward()
        assert isinstance(initial_loss, torch.Tensor)
        initial_loss = be.copy(initial_loss)

        # Use a simple optimizer to check if loss decreases
        optimizer = optim.Adam(module.parameters(), lr=10)

        for _ in range(10):
            optimizer.zero_grad()
            loss = module.forward()
            loss.backward()
            optimizer.step()
            module.apply_bounds()

        final_loss = module.forward()
        assert final_loss.item() < initial_loss

    def test_custom_objective_function(self):
        """
        Test that a custom objective function is used correctly in the forward pass.
        """
        problem, lens = setup_problem()

        # A simple custom objective that returns a constant
        custom_fn = lambda: torch.tensor(123.45, dtype=torch.float64)

        module = OpticalSystemModule(lens, problem, objective_fn=custom_fn)

        loss = module.forward()
        assert be.isclose(loss, torch.tensor(123.45, dtype=torch.float64))

    def test_multiple_thickness_variables_all_get_gradients(self):
        """
        Regression test for issue #569: when more than one thickness variable
        is optimized simultaneously, every thickness must receive a non-zero
        gradient. Previously only the last thickness variable's gradient
        survived because ``set_thickness`` called ``positions.detach()`` on
        every invocation, which also severed the in-iteration gradient path
        introduced by earlier thickness updates in the same forward pass.
        """
        from optiland import optic

        be.grad_mode.enable()
        lens = optic.Optic()
        lens.surfaces.add(index=0, thickness=be.inf)
        lens.surfaces.add(
            index=1, thickness=7, radius=1000, material="N-SF11", is_stop=True,
        )
        lens.surfaces.add(index=2, thickness=30, radius=-1000)
        lens.surfaces.add(index=3)
        lens.set_aperture(aperture_type="EPD", value=15)
        lens.fields.set_type(field_type="angle")
        lens.fields.add(y=0)
        lens.wavelengths.add(value=0.55, is_primary=True)

        problem = OptimizationProblem()
        input_data = {
            "optic": lens,
            "surface_number": -1,
            "Hx": 0, "Hy": 0,
            "num_rays": 5,
            "wavelength": 0.55,
            "distribution": "hexapolar",
        }
        problem.add_operand(
            "rms_spot_size", target=0, weight=1, input_data=input_data,
        )
        problem.add_variable(lens, "radius", surface_number=1)
        problem.add_variable(lens, "thickness", surface_number=1)
        problem.add_variable(lens, "radius", surface_number=2)
        problem.add_variable(lens, "thickness", surface_number=2)

        module = OpticalSystemModule(lens, problem)
        params = list(module.parameters())
        # Ordering matches add_variable above.
        thick1_param = params[1]
        thick2_param = params[3]

        loss = module()
        loss.backward()

        # Both thickness gradients must exist and be non-zero.
        assert thick1_param.grad is not None, (
            "thickness1 grad is None — issue #569 not fixed"
        )
        assert thick2_param.grad is not None
        assert thick1_param.grad.item() != 0.0, (
            "thickness1 grad is zero — issue #569 not fixed"
        )
        assert thick2_param.grad.item() != 0.0
