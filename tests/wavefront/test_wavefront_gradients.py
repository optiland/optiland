"""Autograd gradients of off-axis wavefront OPD agree with finite differences.

The reference geometry (sphere center and radius, or plane point and normal)
depends on the traced rays, so the OPD gradient must include that dependence.
On axis the chief-ray image point does not move with a radius change, which
hides a detached reference; off axis it moves and the gradient is wrong.
"""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.optimization.operand.ray import RayOperand
from optiland.samples.objectives import CookeTriplet
from optiland.wavefront import Wavefront
from tests.utils import assert_allclose

WAVELENGTH = 0.55
SAMPLE = 5  # an off-axis pupil sample of the 8-ring hexapolar grid
STEP = 1e-4  # central finite-difference step on the radius, in mm


def _nominal_radius() -> float:
    return float(be.to_numpy(CookeTriplet().surfaces[1].geometry.radius))


def _opd_sample(radius, field, strategy: str, afocal: bool):
    lens = CookeTriplet()
    lens.surfaces[1].geometry.radius = radius
    wavefront = Wavefront(
        lens,
        [field],
        [WAVELENGTH],
        num_rays=8,
        distribution="hexapolar",
        strategy=strategy,
        afocal=afocal,
    )
    return wavefront.get_data(field, WAVELENGTH).opd[SAMPLE]


def _opd_difference(radius, field):
    lens = CookeTriplet()
    lens.surfaces[1].geometry.radius = radius
    return RayOperand.OPD_difference(lens, *field, 8, WAVELENGTH)


def _autograd_and_fd(function):
    import torch

    r0 = _nominal_radius()
    radius = torch.tensor(r0, dtype=torch.float64, requires_grad=True)
    value = function(radius)
    assert value.requires_grad
    gradient = torch.autograd.grad(value, radius)[0]
    with torch.no_grad():
        upper = function(torch.tensor(r0 + STEP, dtype=torch.float64))
        lower = function(torch.tensor(r0 - STEP, dtype=torch.float64))
    return gradient, (upper - lower) / (2 * STEP)


@pytest.mark.parametrize("field", [(0.0, 1.0), (0.7, 0.7)])
@pytest.mark.parametrize(
    ("strategy", "afocal"),
    [
        ("chief_ray", False),
        ("chief_ray", True),
        ("centroid", False),
        ("best_fit", False),
    ],
)
def test_off_axis_opd_gradient_matches_finite_difference(
    set_test_backend, field, strategy, afocal
):
    if be.get_backend() != "torch":
        pytest.skip("Torch autograd regression")

    gradient, finite_difference = _autograd_and_fd(
        lambda r: _opd_sample(r, field, strategy, afocal)
    )
    assert_allclose(gradient, finite_difference, rtol=1e-4, atol=1e-6)


def test_off_axis_opd_difference_gradient_matches_finite_difference(
    set_test_backend,
):
    if be.get_backend() != "torch":
        pytest.skip("Torch autograd regression")

    gradient, finite_difference = _autograd_and_fd(
        lambda r: _opd_difference(r, (0.0, 1.0))
    )
    assert_allclose(gradient, finite_difference, rtol=1e-4, atol=1e-6)
