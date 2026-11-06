"""Scalar AOI values and differentiable optimization regressions."""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.optimization import OptimizationProblem
from optiland.optimization.operand.ray import RayOperand, _angle_of_incidence
from optiland.samples import CookeTriplet
from tests.utils import assert_allclose


@pytest.mark.parametrize("degrees", [0.0, 1e-8, 30.0, 89.999, 90.0, 120.0, 180.0])
@pytest.mark.parametrize("normal_sign", [-1.0, 1.0])
def test_aoi_unsigned_angle(set_test_backend, degrees, normal_sign):
    radians = be.array(degrees * be.pi / 180.0)
    zero = be.zeros_like(radians)
    angle = _angle_of_incidence(
        be.sin(radians),
        zero,
        be.cos(radians),
        zero,
        zero,
        be.ones_like(radians) * normal_sign,
    )
    assert_allclose(angle, min(degrees, 180.0 - degrees), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("batching", [False, True])
@pytest.mark.parametrize("single_element_normal", [False, True])
def test_aoi_scalar_mixed_residuals(
    set_test_backend, batching, single_element_normal, monkeypatch
):
    lens = CookeTriplet()
    data = dict(
        optic=lens,
        surface_number=1,
        Hx=0.0,
        Hy=0.0,
        Px=0.0,
        Py=0.5,
        wavelength=lens.primary_wavelength,
    )
    geometry = lens.surfaces[1].geometry
    normal = geometry.surface_normal
    shape = (1,) if single_element_normal else ()
    monkeypatch.setattr(
        geometry,
        "surface_normal",
        lambda rays: tuple(be.reshape(v, shape) for v in normal(rays)),
    )
    angle = RayOperand.AOI(**data)
    assert angle.shape == ()
    assert_allclose(angle, 6.520933186552002)
    problem = OptimizationProblem(batching=batching)
    problem.add_operand("AOI", target=5.0, weight=2.0, input_data=data)
    problem.add_operand("f2", target=50.0, input_data={"optic": lens})
    residuals = problem.residual_vector()
    assert residuals.shape == (2,)
    assert_allclose(residuals[0], 2.0 * (angle - 5.0))
    assert_allclose(problem.sum_squared(), be.sum(residuals**2))


@pytest.mark.parametrize("batching", [False, True])
@pytest.mark.parametrize("pupil_y", [0.0, 0.5])
def test_aoi_merit_gradient(set_test_backend, batching, pupil_y):
    if be.get_backend() != "torch":
        pytest.skip("Torch autograd regression")
    import torch

    lens = CookeTriplet()
    problem = OptimizationProblem(batching=batching)
    problem.add_variable(lens, "radius", surface_number=1)
    variable = problem.variables[0].variable
    radius = variable.get_value().item()
    data = dict(
        optic=lens,
        surface_number=1,
        Hx=0.0,
        Hy=0.0,
        Px=0.0,
        Py=pupil_y,
        wavelength=lens.primary_wavelength,
    )
    target = 0.0 if pupil_y == 0.0 else 5.0
    problem.add_operand("AOI", target=target, weight=2.0, input_data=data)

    def evaluate(value):
        variable.update_value(value)
        problem.update_optics()
        return problem.sum_squared()

    parameter = torch.tensor(radius, dtype=torch.float64, requires_grad=True)
    merit = evaluate(parameter)
    assert merit.requires_grad
    gradient = torch.autograd.grad(merit, parameter)[0].item()
    residuals = problem.residual_vector()
    assert residuals.requires_grad
    residual_gradient = torch.autograd.grad(residuals[0], parameter)[0].item()
    assert torch.isfinite(torch.tensor(gradient))
    assert torch.isfinite(torch.tensor(residual_gradient))
    step = 1e-4
    numerical = (
        evaluate(be.array(radius + step)).item()
        - evaluate(be.array(radius - step)).item()
    ) / (2 * step)
    assert gradient == pytest.approx(numerical, rel=1e-5, abs=1e-9)
    if pupil_y == 0.0:
        assert merit.item() == 0.0
        assert gradient == 0.0
    else:
        assert abs(numerical) > 1e-3
        assert gradient == pytest.approx(
            2 * residuals[0].item() * residual_gradient, rel=1e-10
        )
