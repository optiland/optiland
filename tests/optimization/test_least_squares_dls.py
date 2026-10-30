"""``LeastSquares.optimize(method_choice="dls")`` (opt v2 PR4.3).

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import optiland.backend as be
from optiland.optimization import LeastSquares, OptimizationProblem
from optiland.samples.objectives import CookeTriplet

pytestmark = pytest.mark.filterwarnings("ignore")


def _problem(min_val=None, max_val=None):
    optic = CookeTriplet()
    problem = OptimizationProblem()
    problem.add_operand("f2", target=50.0, weight=1.0, input_data={"optic": optic})
    for hy in (0.0, 0.7, 1.0):
        problem.add_operand(
            "rms_spot_size",
            target=0.0,
            weight=10.0,
            input_data={
                "optic": optic,
                "surface_number": -1,
                "Hx": 0.0,
                "Hy": hy,
                "num_rays": 4,
                "wavelength": optic.primary_wavelength,
                "distribution": "hexapolar",
            },
        )
    for s in (1, 2, 3, 4):
        problem.add_variable(
            optic, "radius", surface_number=s, min_val=min_val, max_val=max_val
        )
    return optic, problem


def _merit(problem):
    return float(be.to_numpy(problem.sum_squared()))


def test_dls_returns_optimize_result(set_test_backend):
    _, problem = _problem()
    f0 = _merit(problem)
    result = LeastSquares(problem).optimize(method_choice="dls", maxiter=200)
    assert isinstance(result, OptimizeResult)
    for key in ("x", "fun", "cost", "nfev", "njev", "status", "message", "success"):
        assert key in result
    f1 = _merit(problem)
    assert f1 < f0
    # The optic holds the returned point; cost follows SciPy's 0.5 * sum(r^2).
    assert f1 == pytest.approx(2.0 * result.cost, rel=1e-9)
    x_now = np.array([float(be.to_numpy(v.value)) for v in problem.variables])
    np.testing.assert_allclose(x_now, result.x, rtol=1e-12)


def test_dls_maxiter_counts_jacobian_columns():
    _, problem = _problem()
    result = LeastSquares(problem).optimize(method_choice="dls", maxiter=12, tol=0.0)
    n = len(problem.variables)
    assert result.status == 0
    assert result.nfev <= 12 + n


def test_dls_honors_bounds():
    optic, problem = _problem()
    x0 = np.array([float(be.to_numpy(v.value)) for v in problem.variables])
    # Tight box in the unscaled radius around the start
    for var in problem.variables:
        r = float(be.to_numpy(var.variable.get_value()))
        var.min_val, var.max_val = sorted((r * 0.98, r * 1.02))
    LeastSquares(problem).optimize(method_choice="dls", maxiter=300)
    for var in problem.variables:
        r = float(be.to_numpy(var.variable.get_value()))
        assert var.min_val - 1e-12 <= r <= var.max_val + 1e-12
    assert x0.shape[0] == len(problem.variables)


def test_dls_callback_forms():
    _, problem = _problem()
    xs = []
    LeastSquares(problem).optimize(method_choice="dls", maxiter=60, callback=xs.append)
    assert xs and all(np.asarray(x).shape == (4,) for x in xs)

    _, problem = _problem()
    results = []

    def cb(intermediate_result):
        results.append(intermediate_result)

    LeastSquares(problem).optimize(method_choice="dls", maxiter=60, callback=cb)
    assert results and all(isinstance(r, OptimizeResult) for r in results)


def test_dls_disp(capsys):
    _, problem = _problem()
    LeastSquares(problem).optimize(method_choice="dls", maxiter=40, disp=True)
    assert "Function evaluations" in capsys.readouterr().out


def test_dls_x_scale_accepted():
    _, problem = _problem()
    result = LeastSquares(problem).optimize(
        method_choice="dls", maxiter=60, x_scale=0.5
    )
    assert np.all(np.isfinite(result.x))


def test_dls_untraceable_start_raises(monkeypatch):
    from optiland.optimization.failure import ConfigurationError

    _, problem = _problem()
    optimizer = LeastSquares(problem)
    real = problem.residual_vector

    def failing():
        r = be.to_numpy(real()).copy()
        r[2] = np.nan  # the 0.7 field fails to trace
        return be.array(r)

    monkeypatch.setattr(problem, "residual_vector", failing)
    with pytest.raises(ConfigurationError, match="#2 rms_spot_size"):
        optimizer.optimize(method_choice="dls")


def test_other_methods_unchanged():
    _, problem = _problem()
    result = LeastSquares(problem).optimize(method_choice="trf", maxiter=20)
    assert result.njev is not None or result.nfev > 0
