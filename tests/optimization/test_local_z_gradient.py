"""Keep surface-position derivatives in local z-intercept operands."""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.optic import Optic
from optiland.optimization import OptimizationProblem
from optiland.optimization.operand.ray import RayOperand
from optiland.samples import CookeTriplet
from tests.utils import assert_allclose


def _plane_system():
    lens = Optic()
    lens.surfaces.add(index=0, thickness=be.inf)
    lens.surfaces.add(index=1, thickness=10.0, is_stop=True)
    lens.surfaces.add(index=2)
    lens.set_aperture(aperture_type="EPD", value=10.0)
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0.0)
    lens.wavelengths.add(value=0.55, is_primary=True)
    return lens


def _ray_data(lens):
    return dict(
        optic=lens,
        surface_number=2,
        Hx=0.0,
        Hy=0.0,
        Px=0.0,
        Py=0.5,
        wavelength=lens.primary_wavelength,
    )


@pytest.mark.parametrize("batching", [False, True])
@pytest.mark.parametrize("origin_shape", [(), (1,)])
def test_local_z_scalar_stacks_with_global_z(set_test_backend, batching, origin_shape):
    lens = _plane_system()
    lens.surfaces[2].geometry.cs.z = be.reshape(be.array(10.0), origin_shape)
    data = _ray_data(lens)
    local = RayOperand.z_intercept_lcs(**data)
    assert local.shape == ()
    assert_allclose(local, 0.0)
    problem = OptimizationProblem(batching=batching)
    problem.add_operand("real_z_intercept_lcs", target=1.0, weight=2.0, input_data=data)
    problem.add_operand("real_z_intercept", target=0.0, input_data=data)
    assert_allclose(problem.residual_vector(), [-2.0, 10.0])
    assert problem.residual_vector().shape == (2,)
    assert_allclose(problem.sum_squared(), 104.0)


@pytest.mark.parametrize("batching", [False, True])
@pytest.mark.parametrize("origin_shape", [(), (1,)])
@pytest.mark.parametrize("curved", [False, True])
def test_local_z_thickness_gradient(set_test_backend, batching, origin_shape, curved):
    if be.get_backend() != "torch":
        pytest.skip("Torch autograd regression")
    import torch

    lens = CookeTriplet() if curved else _plane_system()
    problem = OptimizationProblem(batching=batching)
    problem.add_variable(lens, "thickness", surface_number=1)
    variable = problem.variables[0].variable
    thickness = variable.get_value().item()
    problem.add_operand(
        "real_z_intercept_lcs", target=1.0, weight=2.0, input_data=_ray_data(lens)
    )

    def evaluate(value):
        variable.update_value(value)
        problem.update_optics()
        cs = lens.surfaces[2].geometry.cs
        cs.z = be.reshape(cs.z, origin_shape)
        return problem.residual_vector()[0]

    parameter = torch.tensor(thickness, dtype=torch.float64, requires_grad=True)
    residual = evaluate(parameter)
    derivative = torch.autograd.grad(residual, parameter, retain_graph=True)[0].item()
    merit = problem.sum_squared()
    merit_derivative = torch.autograd.grad(merit, parameter)[0].item()
    step = 1e-4
    above = evaluate(be.array(thickness + step)).item()
    below = evaluate(be.array(thickness - step)).item()
    numerical = (above - below) / (2 * step)
    merit_numerical = (above**2 - below**2) / (2 * step)
    assert derivative == pytest.approx(numerical, rel=1e-5, abs=1e-8)
    assert merit_derivative == pytest.approx(merit_numerical, rel=1e-5, abs=1e-8)
    if curved:
        assert abs(numerical) > 1e-7
    else:
        # Translating a plane cannot change its own local z coordinate.
        assert residual.item() == -2.0
        assert derivative == 0.0
        assert merit_derivative == 0.0
