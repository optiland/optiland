"""Regression tests for bounds in vectorizable ray residuals (issue #958)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import optiland.backend as be
from optiland.optimization import OptimizationProblem
from optiland.optimization.operand.operand import operand_registry
from optiland.samples.objectives import CookeTriplet

_VECTORIZED_OPERANDS = [
    "real_x_intercept",
    "real_y_intercept",
    "real_z_intercept",
    "real_L",
    "real_M",
    "real_N",
]


def _ray_data(lens, pupil_y=0.5, wavelength=0.55):
    return dict(
        optic=lens,
        surface_number=-1,
        Hx=0.0,
        Hy=0.0,
        Px=0.1,
        Py=pupil_y,
        wavelength=wavelength,
    )


def _scalar(value):
    return be.to_numpy(be.array(value)).item()


@pytest.mark.parametrize("operand_type", _VECTORIZED_OPERANDS)
@pytest.mark.parametrize(
    "offsets, delta",
    [
        ({"min_val": 0.25}, 0.25),
        ({"max_val": -0.25}, 0.25),
        ({"min_val": -0.5}, 0.0),
        ({"max_val": 0.5}, 0.0),
        ({"min_val": -0.5, "max_val": 0.5}, 0.0),
        ({"min_val": 0.25, "max_val": 0.5}, 0.25),
        ({"min_val": -0.5, "max_val": -0.25}, 0.25),
    ],
    ids=[
        "below-min",
        "above-max",
        "min-satisfied",
        "max-satisfied",
        "inside",
        "below-interval",
        "above-interval",
    ],
)
def test_bounded_ray_residual_matches_standard(
    set_test_backend, operand_type, offsets, delta
):
    lens = CookeTriplet()
    data = _ray_data(lens)
    raw = _scalar(operand_registry.get(operand_type)(**data))
    bounds = {key: raw + offset for key, offset in offsets.items()}

    for batching in (False, True):
        problem = OptimizationProblem(batching=batching)
        problem.add_operand(operand_type, weight=2.5, input_data=data, **bounds)
        residuals = problem.residual_vector()
        assert be.to_numpy(residuals) == pytest.approx([2.5 * delta], abs=1e-9)
        assert _scalar(problem.sum_squared()) == pytest.approx((2.5 * delta) ** 2)
        assert _scalar(be.sum(residuals**2)) == pytest.approx(
            _scalar(problem.sum_squared())
        )


@pytest.mark.parametrize("operand_type", _VECTORIZED_OPERANDS)
def test_mixed_ray_constraints_keep_order_and_share_traces(
    set_test_backend, operand_type, monkeypatch
):
    lens = CookeTriplet()
    problem = OptimizationProblem()
    specifications = [
        (2.5, {"min_val": 0.25}),
        (0.75, {"target": 0.3}),
        (0.0, {"max_val": -0.5}),
        (1.5, {"min_val": -0.2, "max_val": 0.2}),
    ]
    for index, (weight, offsets) in enumerate(specifications):
        data = _ray_data(lens, 0.2 * (index + 1), 0.65 if index == 3 else 0.55)
        raw = _scalar(operand_registry.get(operand_type)(**data))
        constraints = {key: raw + offset for key, offset in offsets.items()}
        problem.add_operand(operand_type, weight=weight, input_data=data, **constraints)
    problem.add_operand(
        "f2",
        target=_scalar(lens.paraxial.f2()) - 1.0,
        weight=0.4,
        input_data={"optic": lens},
    )

    trace_calls = []
    original_trace = lens.trace_generic

    def trace_generic(*args, **kwargs):
        trace_calls.append(1)
        return original_trace(*args, **kwargs)

    monkeypatch.setattr(lens, "trace_generic", trace_generic)
    expected = [0.625, -0.225, 0.0, 0.0, 0.4]
    residuals = problem.residual_vector()
    assert be.to_numpy(residuals) == pytest.approx(expected, abs=1e-9)
    assert len(trace_calls) == 2  # One shared trace per wavelength, not per operand.
    assert _scalar(problem.sum_squared()) == pytest.approx(sum(r**2 for r in expected))


@pytest.mark.parametrize(
    "bounds, bound_slope",
    [
        ({"min_val": 0.5}, -3.0),
        ({"max_val": -0.1}, 3.0),
        ({"min_val": -1.0, "max_val": 1.0}, 0.0),
    ],
)
def test_bounded_residual_keeps_torch_gradient(
    set_test_backend, bounds, bound_slope, monkeypatch
):
    if be.get_backend() != "torch":
        pytest.skip("Torch autograd test")
    import torch

    parameter = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    problem = OptimizationProblem()
    data = _ray_data(CookeTriplet())
    problem.add_operand("real_y_intercept", weight=3.0, input_data=data, **bounds)
    problem.add_operand("real_y_intercept", weight=0.5, target=1.1, input_data=data)
    # Controlled differentiable trace data gives an independent analytic derivative.
    traced = SimpleNamespace(y=torch.stack([parameter, 2 * parameter]).reshape(1, 2))
    monkeypatch.setattr(problem._batched_evaluator, "_safe_execute", lambda job: traced)

    residuals = problem.residual_vector()
    merit = problem.sum_squared()
    residual_gradient = torch.autograd.grad(
        (residuals**2).sum(), parameter, retain_graph=True
    )[0]
    merit_gradient = torch.autograd.grad(merit, parameter)[0]
    bound_residual = 0.0 if bound_slope == 0.0 else 0.9
    expected_gradient = 2 * bound_residual * bound_slope - 0.7
    assert be.to_numpy(residuals) == pytest.approx([bound_residual, -0.35])
    assert residual_gradient.item() == pytest.approx(expected_gradient)
    assert merit_gradient.item() == pytest.approx(expected_gradient)


@pytest.mark.parametrize("bound_name", ["min_val", "max_val"])
def test_bounded_real_ray_gradient_matches_finite_difference(
    set_test_backend, bound_name
):
    if be.get_backend() != "torch":
        pytest.skip("Torch autograd test")
    import torch

    lens = CookeTriplet()
    data = _ray_data(lens)
    raw = _scalar(operand_registry.get("real_y_intercept")(**data))
    bound = raw + (0.25 if bound_name == "min_val" else -0.25)
    problem = OptimizationProblem()
    problem.add_variable(lens, "radius", surface_number=1)
    problem.add_operand(
        "real_y_intercept", weight=2.5, input_data=data, **{bound_name: bound}
    )
    variable = problem.variables[0].variable
    radius = _scalar(variable.get_value())

    def loss(value):
        variable.update_value(value)
        problem.update_optics()
        return be.sum(problem.residual_vector() ** 2)

    parameter = torch.tensor(radius, dtype=torch.float64, requires_grad=True)
    gradient = torch.autograd.grad(loss(parameter), parameter)[0].item()
    step = 1e-4
    finite_difference = (
        _scalar(loss(be.array(radius + step))) - _scalar(loss(be.array(radius - step)))
    ) / (2 * step)
    assert abs(finite_difference) > 1e-8
    assert gradient == pytest.approx(finite_difference, rel=1e-4, abs=1e-7)
