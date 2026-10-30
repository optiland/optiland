"""Native LM / DLS core tests (opt v2 PR4.2).

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.optimization import OptimizationProblem
from optiland.optimization.control.levenberg import LevenbergController
from optiland.optimization.evaluators import FiniteDiffEvaluator
from optiland.optimization.failure import ConfigurationError
from optiland.optimization.native import dls
from optiland.samples.objectives import CookeTriplet

from .validation.problems import PROBLEMS, CallableEvaluator

pytestmark = pytest.mark.filterwarnings("ignore")


def _rosenbrock_evaluator(x0=(-1.2, 1.0)):
    return CallableEvaluator(PROBLEMS["rosenbrock"].residuals, np.array(x0))


# ---------------------------------------------------------------------------
# LevenbergController
# ---------------------------------------------------------------------------


def test_more_scaling_is_monotone():
    c = LevenbergController()
    d1 = c.rescale(np.array([[3.0, 0.0], [4.0, 1.0]])).copy()
    np.testing.assert_allclose(d1, [5.0, 1.0])
    d2 = c.rescale(np.array([[1.0, 0.0], [0.0, 2.0]]))
    np.testing.assert_allclose(d2, [5.0, 2.0])


def test_fixed_scaling():
    # D = 1 / x_scale, independent of the Jacobian (SciPy trf semantics).
    c = LevenbergController(scaling="fixed")
    np.testing.assert_allclose(
        c.rescale(np.array([[3.0, 0.0], [4.0, 1.0]])), [1.0, 1.0]
    )
    c = LevenbergController(scaling="fixed", weights=np.array([1.0, 10.0]))
    np.testing.assert_allclose(
        c.rescale(np.array([[3.0, 0.0], [4.0, 1.0]])), [1.0, 10.0]
    )
    with pytest.raises(ValueError):
        LevenbergController(scaling="bogus")


def _controller(J, radius, **kw):
    c = LevenbergController(**kw)
    c.rescale(J)
    c.radius = radius
    return c


def test_step_inside_region_is_gauss_newton():
    rng = np.random.default_rng(0)
    J = rng.normal(size=(6, 3))
    r = rng.normal(size=6)
    g, A = J.T @ r, J.T @ J
    c = _controller(J, radius=1e6)
    dx = c.solve(A, g)
    np.testing.assert_allclose(dx, np.linalg.lstsq(J, -r, rcond=None)[0], atol=1e-10)
    assert c.lam == 0.0
    assert not c.on_boundary


@pytest.mark.parametrize("scaling", ["more", "fixed"])
def test_step_on_boundary_solves_damped_equations(scaling):
    rng = np.random.default_rng(1)
    J = rng.normal(size=(4, 6))  # m < n: J^T J is singular
    r = rng.normal(size=4)
    g, A = J.T @ r, J.T @ J
    c = _controller(J, radius=0.05, scaling=scaling)
    dx = c.solve(A, g)
    assert c.on_boundary and c.lam > 0
    assert np.linalg.norm(c.d * dx) == pytest.approx(0.05, rel=0.1)
    np.testing.assert_allclose((A + c.lam * np.diag(c.d**2)) @ dx, -g, atol=1e-10)
    # The model reduction is in units of sum(r^2) and positive for the step.
    model = r @ r - np.sum((r + J @ dx) ** 2)
    assert c.model_reduction(dx, g, A) == pytest.approx(model, rel=1e-10)
    assert model > 0


def test_radius_updates():
    J = np.eye(2)
    c = _controller(J, radius=1.0)
    c.update(0.1, np.array([0.4, 0.0]))  # poor fit: shrink to |D dx| / 4
    assert c.radius == pytest.approx(0.1)
    c.update(0.9, np.array([0.1, 0.0]))  # good fit on the boundary: grow
    assert c.radius == pytest.approx(0.2)
    c.update(0.9, np.array([0.01, 0.0]))  # good fit inside: unchanged
    assert c.radius == pytest.approx(0.2)
    c.update(-np.inf, np.array([0.2, 0.0]))  # non-finite trial
    assert c.radius == pytest.approx(0.05)


def test_init_radius_and_collapse():
    c = _controller(np.eye(2), radius=None)
    assert c.init_radius(np.array([3.0, 4.0])) == pytest.approx(5.0)
    assert c.init_radius(np.zeros(2)) == 1.0
    c.radius = 1e-20
    assert c.collapsed(np.ones(2))


def test_singular_system_falls_back():
    c = LevenbergController(d_min=0.0)
    c.d = np.array([1.0, 1.0])
    c.radius = 1.0
    dx = c.solve(np.zeros((2, 2)), np.array([1.0, 0.0]))
    assert np.all(np.isfinite(dx))
    assert np.linalg.norm(dx) == pytest.approx(1.0, rel=0.1)


# ---------------------------------------------------------------------------
# dls driver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["exact", "auto", "broyden"])
def test_rosenbrock_converges(mode):
    ev = _rosenbrock_evaluator()
    res = dls(ev, ftol=1e-12, xtol=1e-12, gtol=1e-10, jacobian_update=mode)
    assert res.success
    np.testing.assert_allclose(res.x, [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(ev.x, res.x)  # evaluator left at x
    for key in (
        "x",
        "fun",
        "cost",
        "jac",
        "grad",
        "nfev",
        "njev",
        "nit",
        "status",
        "message",
    ):
        assert key in res
    assert res.cost == pytest.approx(0.5 * res.fun @ res.fun)


def test_unknown_jacobian_update():
    with pytest.raises(ConfigurationError):
        dls(_rosenbrock_evaluator(), jacobian_update="sometimes")


def test_max_nfev_counts_jacobian_columns():
    ev = _rosenbrock_evaluator()
    res = dls(ev, max_nfev=10, jacobian_update="exact", ftol=0, xtol=0, gtol=0)
    assert res.status == 0
    assert not res.success
    # 1 baseline + 2 columns per Jacobian + trials; never far over the budget
    assert res.nfev == ev.nfev
    assert res.nfev <= 10 + ev.n_vars


def test_initial_nonfinite_raises():
    ev = CallableEvaluator(lambda x: np.array([np.nan, x[0]]), np.array([1.0]))
    with pytest.raises(ConfigurationError, match="untraceable"):
        dls(ev)


def test_nonfinite_trial_is_rejected():
    # Residuals are NaN for x > 0.5: the optimizer must stay in the valid region.
    def fun(x):
        return (
            np.array([x[0] - 2.0, 0.1 * x[0]])
            if x[0] <= 0.5
            else np.array([np.nan, np.nan])
        )

    ev = CallableEvaluator(fun, np.array([0.0]))
    res = dls(ev, max_nfev=200)
    assert np.all(np.isfinite(res.fun))
    assert res.x[0] <= 0.5
    assert res.x[0] > 0.4


def test_bounds_projected_and_resynced():
    lower, upper = np.array([-2.0, -2.0]), np.array([0.5, 2.0])
    ev = _rosenbrock_evaluator()
    evaluated = []
    real_residuals, real_jacobian = ev.residuals, ev.jacobian
    in_jacobian = [False]

    def spy(x):
        if not in_jacobian[0]:  # trial points only, not difference columns
            evaluated.append(np.array(x))
        return real_residuals(x)

    def jacobian(x, r0=None):
        in_jacobian[0] = True
        try:
            return real_jacobian(x, r0)
        finally:
            in_jacobian[0] = False

    ev.residuals, ev.jacobian = spy, jacobian
    res = dls(ev, bounds=(lower, upper), ftol=1e-12, xtol=1e-12)
    assert all(np.all(x >= lower) and np.all(x <= upper) for x in evaluated)
    assert res.x[0] == pytest.approx(0.5)
    assert res.active_mask[0] == 1
    # Constrained minimum of Rosenbrock on x1 <= 0.5 is at (0.5, 0.25).
    np.testing.assert_allclose(res.x, [0.5, 0.25], atol=1e-6)
    np.testing.assert_allclose(res.fun, PROBLEMS["rosenbrock"].residuals(res.x))


def test_callback_signatures_and_stop():
    seen = []
    dls(_rosenbrock_evaluator(), callback=lambda x: seen.append(np.array(x)))
    assert seen

    results = []

    def cb(intermediate_result):
        results.append(intermediate_result)
        if intermediate_result.nit >= 2:
            raise StopIteration

    res = dls(_rosenbrock_evaluator(), callback=cb)
    assert res.status == -2
    assert res.nit == 2
    assert results[-1].cost == pytest.approx(res.cost)


def test_stale_broyden_refreshed_on_rejection():
    ev = _rosenbrock_evaluator()
    res = dls(ev, jacobian_update="broyden", ftol=1e-12, xtol=1e-12, gtol=1e-10)
    assert res.success
    # Broyden saves Jacobians: fewer than one per accepted step.
    assert res.njev < res.nit + 1


def test_x_scale_selects_scaling():
    kw = {"max_nfev": 12, "ftol": 0, "xtol": 0, "gtol": 0}
    more = dls(_rosenbrock_evaluator(), x_scale="jac", **kw)
    fixed = dls(_rosenbrock_evaluator(), x_scale=1.0, **kw)
    weighted = dls(_rosenbrock_evaluator(), x_scale=[1.0, 1e-2], **kw)
    assert not np.allclose(more.x, fixed.x)
    assert not np.allclose(fixed.x, weighted.x)
    with pytest.raises(ConfigurationError):
        dls(_rosenbrock_evaluator(), x_scale="auto")
    with pytest.raises(ConfigurationError):
        dls(_rosenbrock_evaluator(), x_scale=0.0)


def test_lens_problem_improves(set_test_backend):
    optic = CookeTriplet()
    problem = OptimizationProblem()
    problem.add_operand("f2", target=50.0, weight=1.0, input_data={"optic": optic})
    problem.add_operand(
        "rms_spot_size",
        target=0.0,
        weight=10.0,
        input_data={
            "optic": optic,
            "surface_number": -1,
            "Hx": 0.0,
            "Hy": 0.0,
            "num_rays": 5,
            "wavelength": optic.primary_wavelength,
            "distribution": "hexapolar",
        },
    )
    problem.add_variable(optic, "radius", surface_number=1)
    problem.add_variable(optic, "radius", surface_number=2)
    problem.add_variable(optic, "radius", surface_number=6)
    f0 = float(be.to_numpy(problem.sum_squared()))
    ev = FiniteDiffEvaluator(problem)
    res = dls(ev, max_nfev=200, ftol=1e-6)
    f1 = float(be.to_numpy(problem.sum_squared()))
    assert f1 < f0
    assert f1 == pytest.approx(2.0 * res.cost, rel=1e-9)  # optic left at res.x
