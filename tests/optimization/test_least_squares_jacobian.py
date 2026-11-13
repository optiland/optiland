"""Native finite-difference Jacobians behind SciPy ``LeastSquares``.

``LeastSquares`` passes SciPy a ``jac`` backed by the finite-difference
evaluator, with the steps of SciPy's own '2-point' scheme, so runs follow the
path of SciPy's internal differences while the columns get the evaluator's
savings (baseline reuse, partial trace, aim policy).

Kramer Harrison, 2026
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import optimize
from scipy.optimize._numdiff import approx_derivative

import optiland.backend as be
from optiland.optimization import LeastSquares, OptimizationProblem
from optiland.optimization.failure import ConfigurationError
from optiland.optimization.optimizer.scipy import least_squares as ls_module
from optiland.samples.objectives import CookeTriplet

from ..utils import assert_allclose

pytestmark = pytest.mark.filterwarnings("ignore")


def _problem() -> OptimizationProblem:
    """Cooke triplet: three radii (one negative) and four spot operands."""
    lens = CookeTriplet()
    problem = OptimizationProblem()
    for surface in (1, 2, 4):
        problem.add_variable(lens, "radius", surface_number=surface)
    for hy in (0.0, 0.5, 0.7, 1.0):
        problem.add_operand(
            operand_type="rms_spot_size",
            target=0.0,
            weight=1.0,
            input_data={
                "optic": lens,
                "surface_number": -1,
                "Hx": 0.0,
                "Hy": hy,
                "num_rays": 3,
                "wavelength": 0.55,
                "distribution": "hexapolar",
            },
        )
    return problem


def _x(problem: OptimizationProblem) -> np.ndarray:
    return np.array([float(be.to_numpy(v.value)) for v in problem.variables])


def test_jacobian_matches_scipy_two_point(set_test_backend):
    problem = _problem()
    optimizer = LeastSquares(problem)
    x = _x(problem)
    assert np.any(x < 0)  # covers SciPy's step away from zero
    r0 = optimizer._compute_residuals_vector(x)
    expected = approx_derivative(
        optimizer._compute_residuals_vector, x, method="2-point", f0=r0
    )
    assert_allclose(optimizer._compute_jacobian(x), expected, rtol=1e-12, atol=0)


def test_jacobian_flips_at_bounds_like_scipy(set_test_backend):
    problem = _problem()
    # Negative variables on a bound: 0 on its lower one, 1 on its upper one
    first, second = problem.variables[0], problem.variables[1]
    first.min_val = float(be.to_numpy(first.variable.get_value()))
    second.max_val = float(be.to_numpy(second.variable.get_value()))
    lower, upper = first.bounds[0], second.bounds[1]
    x = _x(problem)
    optimizer = LeastSquares(problem)
    r0 = optimizer._compute_residuals_vector(x)
    expected = approx_derivative(
        optimizer._compute_residuals_vector,
        x,
        method="2-point",
        f0=r0,
        bounds=([lower, -np.inf, -np.inf], [np.inf, upper, np.inf]),
    )
    assert_allclose(optimizer._compute_jacobian(x), expected, rtol=1e-12, atol=0)


def test_lm_steps_ignore_bounds(set_test_backend):
    problem = _problem()
    # An 'lm' iterate may leave the box it ignores
    third = problem.variables[2]
    raw = float(be.to_numpy(third.variable.get_value()))
    third.min_val, third.max_val = raw + 1.0, raw + 2.0
    x = _x(problem)
    optimizer = LeastSquares(problem)
    optimizer._evaluator = ls_module._SciPyFiniteDiffEvaluator(problem, bounded=False)
    r0 = optimizer._compute_residuals_vector(x)
    expected = approx_derivative(
        optimizer._compute_residuals_vector, x, method="2-point", f0=r0
    )
    assert_allclose(optimizer._compute_jacobian(x), expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("method", ["lm", "trf"])
def test_run_follows_scipy_two_point_path(set_test_backend, method):
    reference = _problem()
    x0 = _x(reference)
    plain = LeastSquares(reference)
    plain._compute_residuals_vector(x0)
    expected = optimize.least_squares(
        plain._compute_residuals_vector, x0, jac="2-point", method=method, ftol=1e-3
    )

    problem = _problem()
    result = LeastSquares(problem).optimize(method_choice=method, tol=1e-3)
    assert result.nfev == expected.nfev
    assert_allclose(result.x, expected.x, rtol=1e-10, atol=0)


def _captured_least_squares_kwargs(monkeypatch, **optimize_kwargs) -> dict:
    captured = {}
    real = optimize.least_squares

    def spy(fun, x0, **kwargs):
        captured.update(kwargs)
        return real(fun, x0, **kwargs)

    monkeypatch.setattr(ls_module.optimize, "least_squares", spy)
    LeastSquares(_problem()).optimize(**optimize_kwargs)
    return captured


def test_scipy_receives_native_jacobian(set_test_backend, monkeypatch):
    kwargs = _captured_least_squares_kwargs(monkeypatch, maxiter=4, method_choice="trf")
    assert callable(kwargs["jac"])
    assert kwargs["jac"].__func__ is LeastSquares._compute_jacobian
    assert kwargs["max_nfev"] == 4


@pytest.mark.parametrize("counts_columns", [False, True])
def test_lm_keeps_its_iteration_budget(set_test_backend, monkeypatch, counts_columns):
    monkeypatch.setattr(ls_module, "_LM_COUNTS_FD_COLUMNS", counts_columns)
    kwargs = _captured_least_squares_kwargs(monkeypatch, maxiter=20, method_choice="lm")
    n = 3
    assert kwargs["max_nfev"] == (math.ceil(20 / (n + 1)) if counts_columns else 20)


def test_frozen_aim_policy_improves_merit(set_test_backend):
    problem = _problem()
    start = float(be.to_numpy(problem.sum_squared()))
    optimizer = LeastSquares(problem)
    optimizer.optimize(maxiter=10, method_choice="trf", aim_policy="frozen")
    assert optimizer._evaluator.aim_policy.name == "frozen"
    assert float(be.to_numpy(problem.sum_squared())) < start


def test_unknown_aim_policy_raises(set_test_backend):
    with pytest.raises(ConfigurationError, match="aim_policy"):
        LeastSquares(_problem()).optimize(maxiter=2, aim_policy="loose")
