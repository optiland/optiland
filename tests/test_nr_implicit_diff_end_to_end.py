"""End-to-end implicit-diff tests through OptimizationProblem objectives."""

from __future__ import annotations

import numpy as np
import pytest

from optiland.optimization import OptimizationProblem
from optiland.samples.simple import AsphericSinglet

from .nr_implicit_test_utils import (
    assert_fd_is_stable,
    backend_state,
    finite_diff_reference,
)

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _torch_backend_float64():
    with backend_state("torch", precision="float64"):
        yield


def _make_problem(*, Hx: float, Hy: float):
    lens = AsphericSinglet()
    # Add a nonzero configured field so normalized off-axis coordinates map to
    # a physically off-axis trace rather than the degenerate single-field case.
    lens.fields.add(y=5.0)

    problem = OptimizationProblem()
    problem.add_variable(lens, "radius", surface_number=1)
    problem.add_operand(
        operand_type="rms_spot_size",
        target=0.0,
        weight=1.0,
        input_data={
            "optic": lens,
            "surface_number": lens.surfaces.num_surfaces - 1,
            "Hx": Hx,
            "Hy": Hy,
            "num_rays": 24,
            "wavelength": 0.587,
        },
    )
    return problem, lens


@pytest.mark.parametrize("Hx,Hy", [(0.0, 0.0), (0.2, 0.3)])
def test_merit_gradient_matches_fd_on_and_off_axis(Hx, Hy):
    problem, lens = _make_problem(Hx=Hx, Hy=Hy)

    base_raw = problem.variables[0].variable.get_value()
    if torch.is_tensor(base_raw):
        base_value = float(base_raw.detach().item())
    else:
        base_value = float(base_raw)
    param = torch.nn.Parameter(torch.tensor(base_value, dtype=torch.float64))
    problem.variables[0].variable.update_value(param)

    problem.update_optics()
    loss = problem.sum_squared()
    loss.backward()
    ad_grad = float(param.grad.detach().item())

    assert np.isfinite(ad_grad), f"AD gradient is non-finite for field ({Hx}, {Hy})"

    def scalar_loss(radius_value: float) -> float:
        fd_problem, _ = _make_problem(Hx=Hx, Hy=Hy)
        fd_problem.variables[0].variable.update_value(radius_value)
        fd_problem.update_optics()
        with torch.no_grad():
            return float(fd_problem.sum_squared().item())

    # Two-epsilon FD check to reduce risk of accidental cancellation.
    fd_ref, fd_values = finite_diff_reference(
        scalar_loss,
        base_value,
        epsilons=(2e-6, 1e-6),
    )
    assert_fd_is_stable(fd_values, abs_tol=1e-8, rel_tol=2e-2)

    assert abs(ad_grad - fd_ref) <= max(1e-7, 2e-2 * abs(fd_ref)), (
        f"Merit AD/FD mismatch at field ({Hx}, {Hy}): "
        f"AD={ad_grad:.12e}, FD={fd_ref:.12e}, FD samples={fd_values}"
    )

    if (Hx, Hy) != (0.0, 0.0):
        # Explicitly verify this is genuinely off-axis ray geometry.
        lens.trace(Hx, Hy, 0.587, 24, "hexapolar")
        img = lens.surfaces.num_surfaces - 1
        x = np.asarray(lens.surfaces.x[img, :].detach().cpu())
        y = np.asarray(lens.surfaces.y[img, :].detach().cpu())
        assert abs(x.mean()) > 1e-3 or abs(y.mean()) > 1e-3
