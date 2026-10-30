"""``minimize()`` facade and OptimizationResult (opt v2 PR5.2).

Kramer Harrison, 2026
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import optiland.backend as be
from optiland.optimization import (
    CancelToken,
    ConfigurationError,
    HistoryObserver,
    MaxIter,
    OptimizationProblem,
    OptimizationResult,
    minimize,
)
from optiland.optimization.failure import OptimizationFailure
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


def _x(problem):
    return np.array([float(be.to_numpy(v.value)) for v in problem.variables])


def test_signature_is_the_lean_v1_signature():
    params = inspect.signature(minimize).parameters
    assert list(params) == [
        "problem",
        "method",
        "maxiter",
        "tol",
        "stop",
        "observers",
        "cancel_token",
        "disp",
        "options",
    ]
    assert params["method"].default == "auto"
    assert params["maxiter"].default is None
    assert params["tol"].default == 1e-3
    assert params["disp"].default is False
    for name in ("maxiter", "tol", "stop", "observers", "cancel_token", "disp"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["options"].kind is inspect.Parameter.VAR_KEYWORD


@pytest.mark.parametrize("method", ["auto", "dls"])
def test_auto_resolves_to_dls_and_mutates_optic(set_test_backend, method):
    _, problem = _problem()
    f0 = _merit(problem)
    result = minimize(problem, method, maxiter=200)
    assert isinstance(result, OptimizationResult)
    assert isinstance(result, OptimizeResult)
    assert result.method == "dls"
    assert result.history is None
    assert result.stop_reason == result.message
    assert result.n_evals == result.nfev
    # In-place: the optic holds result.x and fun is the merit there
    f1 = _merit(problem)
    assert f1 < f0
    assert result.fun == pytest.approx(f1, rel=1e-9)
    np.testing.assert_allclose(_x(problem), result.x, rtol=1e-12)
    assert result.residuals.shape == (len(problem.operands),)


def test_dls_result_matches_legacy_dls():
    from optiland.optimization import LeastSquares

    _, p1 = _problem()
    _, p2 = _problem()
    a = minimize(p1, "dls", maxiter=100)
    b = LeastSquares(p2).optimize(method_choice="dls", maxiter=100)
    np.testing.assert_allclose(a.x, b.x, rtol=1e-12)
    assert a.nfev == b.nfev


def test_auto_picks_lbfgsb_for_material_variables():
    optic, problem = _problem()
    problem.add_variable(optic, "material", surface_number=1, glass_selection=["N-BK7"])
    assert any(isinstance(v.value, str) for v in problem.variables)
    with pytest.raises(ConfigurationError, match="Glass"):
        minimize(problem)


@pytest.mark.parametrize("method", ["l-bfgs-b", "slsqp", "powell", "least_squares"])
def test_scipy_methods_reduce_merit(method):
    _, problem = _problem()
    f0 = _merit(problem)
    result = minimize(problem, method, maxiter=60)
    assert result.method == method
    assert _merit(problem) < f0
    assert result.fun == pytest.approx(_merit(problem), rel=1e-6)
    assert result.n_evals == result.nfev > 0
    assert np.isfinite(result.x).all()


def test_least_squares_method_choice_and_bounds():
    _, problem = _problem()
    for var in problem.variables:
        r = float(be.to_numpy(var.variable.get_value()))
        var.min_val, var.max_val = sorted((r * 0.97, r * 1.03))
    f0 = _merit(problem)
    # "lm" cannot handle bounds: the core switches to "trf" and respects them
    minimize(problem, "least_squares", maxiter=60, method_choice="lm")
    assert _merit(problem) < f0
    for var in problem.variables:
        r = float(be.to_numpy(var.variable.get_value()))
        assert var.min_val - 1e-9 <= r <= var.max_val + 1e-9
    with pytest.raises(ConfigurationError, match="method_choice"):
        minimize(problem, "least_squares", method_choice="nope")


def test_unknown_method_and_option_raise_before_any_work():
    _, problem = _problem()
    f0 = _merit(problem)
    x0 = _x(problem)
    with pytest.raises(ConfigurationError, match="Unknown method"):
        minimize(problem, "simulated_annealing")
    with pytest.raises(ConfigurationError, match="Unknown option"):
        minimize(problem, "dls", bogus=1)
    with pytest.raises(ConfigurationError, match="jacobian_update"):
        minimize(problem, "dls", jacobian_update="nope")
    # Options of another method are unknown here
    with pytest.raises(ConfigurationError, match="scheme"):
        minimize(problem, "l-bfgs-b", scheme="central")
    with pytest.raises(ConfigurationError, match="method_choice"):
        minimize(problem, "dls", method_choice="trf")
    with pytest.raises(ConfigurationError, match="torch backend"):
        minimize(problem, "adam")
    with pytest.raises(ConfigurationError, match="lr"):
        minimize(problem, "dls", lr=0.1)
    assert _merit(problem) == f0
    np.testing.assert_array_equal(_x(problem), x0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"maxiter": 0}, "maxiter"), ({"tol": 0.0}, "tol")],
)
def test_argument_validation(kwargs, match):
    _, problem = _problem()
    with pytest.raises(ConfigurationError, match=match):
        minimize(problem, **kwargs)


def test_no_variables_raises():
    problem = OptimizationProblem()
    with pytest.raises(ConfigurationError, match="no variables"):
        minimize(problem)


def test_dls_options_are_forwarded():
    _, problem = _problem()
    result = minimize(
        problem,
        "dls",
        maxiter=100,
        scheme="central",
        rel_step=1e-6,
        aim_policy="frozen",
        jacobian_update="auto",
        on_failure="raise",
    )
    assert result.nfev > 0


def test_history_stop_and_cancel_on_dls():
    _, problem = _problem()
    history = HistoryObserver()
    result = minimize(
        problem, "dls", tol=1e-12, observers=[history], stop=MaxIter(2), maxiter=500
    )
    assert result.status == -4
    assert result.nit == 2
    assert result.history == history.records
    assert len(result.history) == 3

    _, problem = _problem()
    token = CancelToken()
    token.cancel()
    result = minimize(problem, "dls", cancel_token=token)
    assert result.nit == 1
    assert "cancelled" in result.stop_reason


@pytest.mark.parametrize("method", ["l-bfgs-b", "bfgs"])
def test_observers_and_stop_on_scipy_local(method):
    _, problem = _problem()
    history = HistoryObserver()
    result = minimize(problem, method, tol=1e-14, observers=[history], stop=MaxIter(2))
    assert result.nit == 2
    assert "max_iter=2" in result.stop_reason
    assert result.success
    assert len(result.history) == 3
    assert result.history[-1]["value"] > 0
    steps = [r["step_norm"] for r in result.history[1:]]
    assert all(s > 0 for s in steps)
    # The optic holds the returned point
    np.testing.assert_allclose(_x(problem), result.x, rtol=1e-12)


class _CancelOnFirstStep:
    def __init__(self, token):
        self.token = token
        self.seen = []

    def on_start(self, state):
        pass

    def on_step(self, state):
        self.seen.append(state.iteration)
        self.token.cancel()

    def on_end(self, state, result):
        pass


@pytest.mark.parametrize("method", ["l-bfgs-b", "least_squares"])
def test_cancel_scipy_local_and_least_squares(method):
    _, problem = _problem()
    token = CancelToken()
    canceller = _CancelOnFirstStep(token)
    result = minimize(problem, method, cancel_token=token, observers=[canceller])
    assert np.isfinite(result.fun)
    assert np.isfinite(_merit(problem))
    if method == "l-bfgs-b":
        assert canceller.seen == [1]
        assert "cancelled" in result.stop_reason


def test_cancel_before_start_least_squares():
    _, problem = _problem()
    x0 = _x(problem)
    token = CancelToken()
    token.cancel()
    result = minimize(problem, "least_squares", cancel_token=token)
    assert result.stop_reason == "cancelled by user"
    assert not result.success
    np.testing.assert_allclose(_x(problem), x0, rtol=1e-12)


def test_disp_prints(capsys):
    _, problem = _problem()
    minimize(problem, "dls", maxiter=40, disp=True)
    out = capsys.readouterr().out
    assert "start: merit" in out
    assert "end: merit" in out


def _poison(problem, predicate):
    """Make ``residual_vector`` return NaN in row 2 where ``predicate(x)``."""
    original = problem.residual_vector

    def residual_vector():
        r = be.to_numpy(original()).copy()
        x = float(be.to_numpy(problem.variables[0].value))
        if predicate(x):
            r[2] = np.nan
        return be.array(r)

    problem.residual_vector = residual_vector


@pytest.mark.parametrize("method", ["dls", "l-bfgs-b", "least_squares"])
def test_untraceable_start_raises_configuration_error(method):
    _, problem = _problem()
    _poison(problem, lambda x: True)
    with pytest.raises(ConfigurationError, match=r"untraceable.*#2"):
        minimize(problem, method, maxiter=5)


@pytest.mark.parametrize("method", ["dls", "l-bfgs-b", "least_squares"])
def test_failed_trials_do_not_break_the_run(method):
    _, problem = _problem()
    x0 = float(be.to_numpy(problem.variables[0].value))
    _poison(problem, lambda x: abs(x - x0) > 1e-3 * abs(x0))
    result = minimize(problem, method, maxiter=40)
    assert np.isfinite(result.fun)
    assert np.isfinite(result.x).all()


@pytest.mark.parametrize("method", ["l-bfgs-b", "least_squares"])
def test_on_failure_raise(method):
    _, problem = _problem()
    x0 = float(be.to_numpy(problem.variables[0].value))
    _poison(problem, lambda x: abs(x - x0) > 1e-3 * abs(x0))
    with pytest.raises(OptimizationFailure):
        minimize(problem, method, maxiter=40, on_failure="raise")


def test_fun_is_scalar_merit_for_every_method():
    for method in ("dls", "l-bfgs-b", "least_squares"):
        _, problem = _problem()
        result = minimize(problem, method, maxiter=30)
        assert np.ndim(result.fun) == 0


def test_exported_from_package():
    import optiland.optimization as opt

    for name in (
        "minimize",
        "OptimizationResult",
        "MaxIter",
        "MaxEvals",
        "CostTolerance",
        "GradNormTolerance",
        "StepStall",
        "ConsoleObserver",
        "HistoryObserver",
        "CheckpointObserver",
        "CancelToken",
        "ConfigurationError",
    ):
        assert hasattr(opt, name)


@pytest.fixture
def torch_backend():
    torch = pytest.importorskip("torch")  # noqa: F841
    be.set_backend("torch")
    be.set_device("cpu")
    be.grad_mode.enable()
    be.set_precision("float64")
    yield
    be.set_backend("numpy")


@pytest.mark.parametrize("method", ["adam", "sgd"])
def test_torch_first_order(torch_backend, method):
    _, problem = _problem()
    f0 = _merit(problem)
    history = HistoryObserver()
    result = minimize(
        problem, method, maxiter=15, lr=1e-6, gamma=0.9, observers=[history]
    )
    assert result.method == method
    assert result.nit == result.n_evals == 15
    assert len(result.history) == 16
    assert result.stop_reason == "15 steps completed"
    assert result.fun == pytest.approx(_merit(problem), rel=1e-9)
    assert _merit(problem) < f0
    np.testing.assert_allclose(_x(problem), result.x, rtol=1e-9)


@pytest.mark.parametrize("method", ["adam", "sgd"])
def test_torch_stop_and_cancel(torch_backend, method):
    _, problem = _problem()
    result = minimize(problem, method, maxiter=50, lr=1e-6, stop=MaxIter(3))
    assert result.nit == 3
    assert "max_iter=3" in result.stop_reason
    assert np.isfinite(_merit(problem))
    np.testing.assert_allclose(_x(problem), result.x, rtol=1e-9)

    _, problem = _problem()
    token = CancelToken()
    token.cancel()
    result = minimize(problem, method, maxiter=50, lr=1e-6, cancel_token=token)
    assert result.nit == 1
    assert "cancelled" in result.stop_reason
