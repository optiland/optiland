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
from optiland.optimization import OptimizationProblem, TorchAdamOptimizer
from optiland.optimization.scaling import (
    IdentityScaler,
    LogScaler,
    PowerScaler,
    ReciprocalScaler,
)
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


def test_init_runtime_error_wrong_backend():
    original_backend = be.get_backend()
    try:
        be.set_backend("numpy")
        problem, lens = setup_problem()
        requirement = "package" if torch is None else "backend"
        with pytest.raises(RuntimeError, match=f"requires the 'torch' {requirement}"):
            OpticalSystemModule(lens, problem)
    finally:
        be.set_backend(original_backend)


def _physical_value(problem, index=0):
    """Return the unscaled value of a problem variable as a float."""
    return float(be.to_numpy(problem.variables[index].variable.get_value()))


@pytest.fixture(scope="module")
def set_torch_backend():
    """
    Configure Torch for the tests that require the optional backend.
    It will set the backend to torch before the tests run and revert to the
    original backend after all tests are completed.
    """
    pytest.importorskip("torch")
    original_backend = be.get_backend()
    be.set_backend("torch")
    yield
    be.set_backend(original_backend)


@pytest.mark.usefixtures("set_torch_backend")
class TestOpticalSystemModule:
    """
    Tests for the OpticalSystemModule wrapper class.
    """

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
        Params live in scaled space: apply_bounds clamps them to the scaled
        bounds, and syncing puts the physical bound on the surface.
        """
        min_b, max_b = 5.0, 15.0
        problem, lens = setup_problem(min_val=min_b, max_val=max_b)
        module = OpticalSystemModule(lens, problem)
        scaled_min, scaled_max = problem.variables[0].bounds

        # Test clamping to the maximum bound
        with torch.no_grad():
            module.params[0].data.fill_(scaled_max + 5.0)

        module.apply_bounds()
        module._sync_params_to_problem()

        assert be.isclose(module.params[0].data, be.array(scaled_max))
        assert _physical_value(problem) == pytest.approx(max_b, rel=1e-5)

        # Test clamping to the minimum bound
        with torch.no_grad():
            module.params[0].data.fill_(scaled_min - 5.0)

        module.apply_bounds()
        module._sync_params_to_problem()

        assert be.isclose(module.params[0].data, be.array(scaled_min))
        assert _physical_value(problem) == pytest.approx(min_b, rel=1e-5)

    def test_apply_bounds_keeps_value_inside_bounds(self):
        """
        Regression: a thickness of 3.25896 with bounds [1, 10] became 20.0
        after apply_bounds, because the already scaled bounds were
        inverse-scaled again before clamping the scaled parameter.
        """
        problem, lens = setup_problem(min_val=1.0, max_val=10.0)
        module = OpticalSystemModule(lens, problem)
        before = _physical_value(problem)

        module.apply_bounds()
        module()

        assert _physical_value(problem) == pytest.approx(before, rel=1e-5)

    @pytest.mark.parametrize(
        "variable_type, scaler, min_val, max_val",
        [
            ("thickness", None, 2.0, 8.0),
            ("radius", None, 15.0, 40.0),
            ("radius", IdentityScaler(), 15.0, 40.0),
            ("radius", ReciprocalScaler(), 15.0, 40.0),
            ("thickness", LogScaler(), 2.0, 8.0),
            ("thickness", PowerScaler(power=2.0), 2.0, 8.0),
        ],
        ids=[
            "thickness-default",
            "radius-default",
            "radius-identity",
            "radius-reciprocal",
            "thickness-log",
            "thickness-power",
        ],
    )
    def test_apply_bounds_respects_physical_bounds(
        self, variable_type, scaler, min_val, max_val
    ):
        """
        For every scaler, apply_bounds must leave an in-range value alone and
        pull an out-of-range value back to a physical bound. The reciprocal
        scaler reverses the bound order in scaled space.
        """
        lens = CookeTriplet()
        problem = OptimizationProblem()
        kwargs = {} if scaler is None else {"scaler": scaler}
        problem.add_variable(
            lens,
            variable_type,
            surface_number=1,
            min_val=min_val,
            max_val=max_val,
            **kwargs,
        )
        module = OpticalSystemModule(lens, problem)
        scaled_lo, scaled_hi = (float(b) for b in problem.variables[0].bounds)
        span = scaled_hi - scaled_lo

        inside = module.params[0].item()
        module.apply_bounds()
        assert module.params[0].item() == pytest.approx(inside)

        for overshoot in (scaled_hi + span, scaled_lo - span):
            with torch.no_grad():
                module.params[0].data.fill_(overshoot)
            module.apply_bounds()
            module._sync_params_to_problem()

            value = _physical_value(problem)
            assert value == pytest.approx(min_val, rel=1e-5) or value == pytest.approx(
                max_val, rel=1e-5
            )

    def test_apply_bounds_matches_torch_optimizer(self):
        """
        OpticalSystemModule and TorchAdamOptimizer must clamp the same
        parameter to the same value.
        """
        problem, lens = setup_problem(min_val=2.0, max_val=8.0)
        module = OpticalSystemModule(lens, problem)
        optimizer = TorchAdamOptimizer(problem)
        scaled_lo, scaled_hi = (float(b) for b in problem.variables[0].bounds)

        for value in (scaled_lo - 1.0, 0.5 * (scaled_lo + scaled_hi), scaled_hi + 1.0):
            with torch.no_grad():
                module.params[0].data.fill_(value)
                optimizer.params[0].data.fill_(value)
            module.apply_bounds()
            optimizer._apply_bounds()

            assert module.params[0].item() == pytest.approx(optimizer.params[0].item())

    def test_bounded_training_loop_stays_in_physical_bounds(self):
        """
        A user-owned Adam loop that calls apply_bounds after each step must
        keep the surface inside its physical bounds and the loss finite.
        """
        min_b, max_b = 2.0, 8.0
        problem, lens = setup_problem(min_val=min_b, max_val=max_b, target=12.0)
        module = OpticalSystemModule(lens, problem)
        optimizer = optim.Adam(module.parameters(), lr=0.1)

        for _ in range(10):
            optimizer.zero_grad()
            loss = module()
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            module.apply_bounds()
            module._sync_params_to_problem()

            assert min_b - 1e-4 <= _physical_value(problem) <= max_b + 1e-4

    def test_total_track_operand_gradient_matches_finite_difference(self):
        """
        Regression: under torch, total_track was a Python float, so the operand
        had no gradient and torch optimizers ignored track constraints.
        """
        lens = CookeTriplet()
        problem = OptimizationProblem()
        problem.add_variable(lens, "thickness", surface_number=1)
        problem.add_operand(
            operand_type="total_track",
            max_val=50.0,
            weight=1.0,
            input_data={"optic": lens},
        )
        module = OpticalSystemModule(lens, problem)

        loss = module()
        assert isinstance(lens.total_track, torch.Tensor)
        assert loss.requires_grad
        loss.backward()
        autograd = module.params[0].grad.item()

        # The loss is quadratic in the thickness, so a central difference is
        # exact up to rounding.
        h = 1e-2
        param = module.params[0]
        x0 = param.detach().clone()
        with torch.no_grad():
            param.copy_(x0 + h)
            loss_plus = module().item()
            param.copy_(x0 - h)
            loss_minus = module().item()
            param.copy_(x0)
        finite_diff = (loss_plus - loss_minus) / (2 * h)

        assert autograd != 0.0
        assert autograd == pytest.approx(finite_diff, rel=1e-3)

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
