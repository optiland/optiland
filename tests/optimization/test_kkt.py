"""Hard constraints and the KKT controller of native DLS (opt v2 PR7.3).

Known-KKT problems first: linear residuals and linear rows, whose solution and
multipliers are known in closed form, through a minimal analytic evaluator.
Then the same machinery through ``OptimizationProblem`` on real lenses.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.optimization import OptimizationProblem, minimize
from optiland.optimization.control import kkt
from optiland.optimization.control.kkt import KKTController, solve_dual_qp
from optiland.optimization.evaluators import FiniteDiffEvaluator
from optiland.optimization.failure import ConfigurationError
from optiland.optimization.native import dls
from optiland.optimization.operand.operand import operand_registry
from optiland.optimization.optimizer.custom.cma_es import CMAES
from optiland.optimization.optimizer.scipy.base import OptimizerGeneric
from optiland.optimization.optimizer.scipy.dual_annealing import DualAnnealing
from optiland.optimization.optimizer.scipy.least_squares import LeastSquares
from optiland.optimization.optimizer.scipy.orthogonal_descent import OrthogonalDescent
from optiland.samples.objectives import CookeTriplet

from .validation.problems import CallableEvaluator

pytestmark = pytest.mark.filterwarnings("ignore")


class AnalyticEvaluator:
    """Residuals ``M x - b`` with rows ``C x - d`` and exact Jacobians."""

    def __init__(self, M, b, C, d, equality, x0, tol=1e-9, rows=None):
        self.M, self.b = np.asarray(M, float), np.asarray(b, float)
        self.C, self.d = np.asarray(C, float), np.asarray(d, float)
        self.equality = np.asarray(equality, dtype=bool)
        self.x = np.asarray(x0, float).copy()
        self.n_vars = self.x.size
        self.n_constraints = self.d.size
        self.backend = "numpy"
        self.tol = tol
        self.rows = rows or (lambda x: self.C @ x - self.d)

    def constraint_rows(self):
        labels = [f"row{i}" for i in range(self.n_constraints)]
        return self.equality, np.full(self.n_constraints, self.tol), labels

    def read_x(self):
        return self.x.copy()

    def write_x(self, x):
        self.x = np.asarray(x, float).copy()

    def residuals(self, x):
        return self.M @ x - self.b

    def value(self, x):
        r = self.residuals(x)
        return float(r @ r)

    def jacobian(self, x, r0=None):
        return self.M.copy()

    def constraints(self, x):
        return self.rows(x)

    def constraint_jacobian(self, x):
        return self.C.copy()


def _assert_kkt(res, ev, mu):
    """Stationarity, feasibility, sign and complementarity of a result."""
    x = res.x
    mu = np.asarray(mu, float)
    s = ev.rows(x)
    grad = 2.0 * ev.M.T @ ev.residuals(x)
    bound_free = grad + ev.C.T @ mu
    np.testing.assert_allclose(bound_free, 0.0, atol=1e-6)
    assert np.all(np.abs(s[ev.equality]) <= 1e-7)
    assert np.all(s[~ev.equality] <= 1e-7)
    assert np.all(mu[~ev.equality] >= -1e-7)
    np.testing.assert_allclose(mu[~ev.equality] * s[~ev.equality], 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Dual QP
# ---------------------------------------------------------------------------


def test_dual_qp_releases_on_multiplier_sign():
    # min 1/2 |d|^2 - g.d s.t. rows: the corner of two rows seen from far away
    # makes the first row added lose its multiplier once the second enters.
    A = np.array([[1.0, 0.0], [1.0, 1.0]])
    g = np.array([-3.0, 1.0])
    b = np.array([1.0, 0.0])
    mu, active, solved = solve_dual_qp(A @ A.T, b + A @ g, np.zeros(2, bool), 1e-12)
    assert solved
    d = -(g + A.T @ mu)
    # Direct enumeration of the four working sets gives the same point.
    best = None
    for rows in ([], [0], [1], [0, 1]):
        S = A[rows] @ A[rows].T
        m = np.zeros(2)
        if rows:
            m[rows] = np.linalg.solve(S, -(b + A @ g)[rows])
        cand = -(g + A.T @ m)
        if np.all(m >= -1e-12) and np.all(A @ cand <= b + 1e-12):
            best = cand
    np.testing.assert_allclose(d, best, atol=1e-10)
    assert np.all(mu >= -1e-12)


def test_dual_qp_inconsistent_rows_are_reported():
    A = np.array([[1.0], [-1.0]])
    b = np.array([-1.0, -1.0])  # x <= -1 and x >= 1
    _, _, solved = solve_dual_qp(A @ A.T, b, np.zeros(2, bool), 1e-12)
    assert not solved


def test_dual_qp_inconsistent_equalities_are_reported():
    A = np.array([[1.0], [1.0]])
    b = np.array([1.0, 2.0])
    _, _, solved = solve_dual_qp(A @ A.T, b, np.ones(2, bool), 1e-12)
    assert not solved


# ---------------------------------------------------------------------------
# Known-KKT QPs through dls
# ---------------------------------------------------------------------------


def test_equality_qp():
    # min |x - (1, 2)|^2 s.t. x0 + x1 = 1: x = (0, 1), mu = 2.
    ev = AnalyticEvaluator(
        np.eye(2), [1.0, 2.0], [[1.0, 1.0]], [1.0], [True], [5.0, -3.0]
    )
    res = dls(ev, ftol=1e-14, xtol=1e-14)
    assert res.success
    np.testing.assert_allclose(res.x, [0.0, 1.0], atol=1e-8)
    assert res.multipliers["row0"] == pytest.approx(2.0, abs=1e-6)
    assert res.active_set == ["row0"]
    assert res.max_constraint_violation <= 1e-8
    _assert_kkt(res, ev, [res.multipliers["row0"]])


def test_active_inequality_qp():
    # min |x - (2, 2)|^2 s.t. x0 + x1 <= 2: x = (1, 1), mu = 2.
    ev = AnalyticEvaluator(
        np.eye(2), [2.0, 2.0], [[1.0, 1.0]], [2.0], [False], [0.0, 0.0]
    )
    res = dls(ev, ftol=1e-14, xtol=1e-14)
    assert res.success
    np.testing.assert_allclose(res.x, [1.0, 1.0], atol=1e-8)
    assert res.multipliers["row0"] == pytest.approx(2.0, abs=1e-6)
    assert res.active_set == ["row0"]


def test_inactive_inequality_has_zero_multiplier():
    ev = AnalyticEvaluator(
        np.eye(2), [0.25, 0.25], [[1.0, 1.0]], [2.0], [False], [0.0, 0.0]
    )
    res = dls(ev, ftol=1e-14, xtol=1e-14)
    assert res.success
    np.testing.assert_allclose(res.x, [0.25, 0.25], atol=1e-8)
    assert res.multipliers["row0"] == 0.0
    assert res.active_set == []


def test_mixed_rows_with_a_released_start():
    # min |x - (3, 0, -1)|^2 s.t. x0 + x1 + x2 = 1, x0 <= 1, -x2 <= 0.
    # Solution x = (1, 0, 0): mu_eq = 0, mu_x0 = 4, mu_x2 = 2.
    M = np.eye(3)
    C = [[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
    ev = AnalyticEvaluator(
        M, [3.0, 0.0, -1.0], C, [1.0, 1.0, 0.0], [True, False, False], [0.0, 1.0, 0.0]
    )
    res = dls(ev, ftol=1e-14, xtol=1e-14)
    assert res.success
    np.testing.assert_allclose(res.x, [1.0, 0.0, 0.0], atol=1e-7)
    mu = [res.multipliers[f"row{i}"] for i in range(3)]
    np.testing.assert_allclose(mu, [0.0, 4.0, 2.0], atol=1e-5)
    _assert_kkt(res, ev, mu)


@pytest.mark.parametrize("seed", range(5))
def test_random_qp_matches_slsqp(seed):
    from scipy.optimize import minimize as scipy_minimize

    rng = np.random.default_rng(seed)
    M = rng.normal(size=(8, 5))
    b = rng.normal(size=8)
    C = rng.normal(size=(4, 5))
    d = rng.normal(size=4) + 0.5
    equality = np.array([True, False, False, False])
    # Rows hold at the origin, except for the equality.
    d[1:] = np.abs(d[1:])
    ev = AnalyticEvaluator(M, b, C, d, equality, np.zeros(5))
    res = dls(ev, ftol=1e-14, xtol=1e-14, gtol=1e-10)
    assert res.success

    ref = scipy_minimize(
        lambda x: np.sum((M @ x - b) ** 2),
        np.zeros(5),
        jac=lambda x: 2 * M.T @ (M @ x - b),
        method="SLSQP",
        constraints=[
            {"type": "eq", "fun": lambda x: C[0] @ x - d[0]},
            {"type": "ineq", "fun": lambda x: d[1:] - C[1:] @ x},
        ],
        options={"ftol": 1e-14, "maxiter": 500},
    )
    np.testing.assert_allclose(res.x, ref.x, atol=1e-5)
    _assert_kkt(res, ev, [res.multipliers[f"row{i}"] for i in range(4)])


def test_bounds_are_inequality_rows():
    # min |x - (2, -2)|^2 on [0, 1]^2: x = (1, 0), bound multipliers 2 and 4.
    ev = CallableEvaluator(lambda x: x - np.array([2.0, -2.0]), np.array([0.5, 0.5]))
    lower, upper = np.zeros(2), np.ones(2)
    res = dls(ev, bounds=(lower, upper), ftol=1e-14, xtol=1e-14)
    assert res.success
    np.testing.assert_allclose(res.x, [1.0, 0.0], atol=1e-9)
    assert res.active_mask.tolist() == [1, -1]
    assert res.multipliers["x[0] <= 1"] == pytest.approx(2.0, abs=1e-5)
    assert res.multipliers["x[1] >= 0"] == pytest.approx(4.0, abs=1e-5)
    assert sorted(res.active_set) == ["x[0] <= 1", "x[1] >= 0"]


def test_row_and_bound_active_together():
    # min |x - (2, 2)|^2 s.t. x0 + x1 <= 1.5, x0 <= 0.5: x = (0.5, 1.0).
    # Stationarity: mu_row = 2, mu_bound = 1.
    ev = AnalyticEvaluator(
        np.eye(2), [2.0, 2.0], [[1.0, 1.0]], [1.5], [False], [0.0, 0.0]
    )
    res = dls(ev, bounds=([-np.inf, -np.inf], [0.5, np.inf]), ftol=1e-14, xtol=1e-14)
    assert res.success
    np.testing.assert_allclose(res.x, [0.5, 1.0], atol=1e-8)
    assert res.multipliers["row0"] == pytest.approx(2.0, abs=1e-5)
    assert res.multipliers["x[0] <= 0.5"] == pytest.approx(1.0, abs=1e-5)


def test_no_merit_rows_finds_a_feasible_point():
    ev = AnalyticEvaluator(
        np.zeros((0, 2)), np.zeros(0), [[1.0, 2.0]], [3.0], [True], [0.0, 0.0]
    )
    res = dls(ev, ftol=1e-12, xtol=1e-12)
    assert res.success
    assert res.max_constraint_violation <= 1e-8
    # Minimum-norm correction of the origin.
    np.testing.assert_allclose(res.x, [0.6, 1.2], atol=1e-8)


def test_cholesky_once_per_lambda(monkeypatch):
    rng = np.random.default_rng(3)
    M, b = rng.normal(size=(6, 4)), rng.normal(size=6)
    C, d = np.eye(2, 4), np.array([-0.1, 0.1])
    ev = AnalyticEvaluator(M, b, C, d, [True, False], np.ones(4))
    batches = [[]]
    real_cholesky, real_linearize = kkt._cholesky, KKTController.linearize

    def cholesky(H):
        batches[-1].append(H.tobytes())
        return real_cholesky(H)

    def linearize(self, *args):
        batches.append([])
        real_linearize(self, *args)

    monkeypatch.setattr(kkt, "_cholesky", cholesky)
    monkeypatch.setattr(KKTController, "linearize", linearize)
    res = dls(ev, ftol=1e-12, xtol=1e-12)
    assert res.success
    assert any(batches)
    # Within one linearization no lambda is factored twice.
    assert all(len(keys) == len(set(keys)) for keys in batches)


def test_no_cholesky_when_no_row_binds(monkeypatch):
    ev = CallableEvaluator(lambda x: x - 0.25, np.array([0.5, 0.5]))
    monkeypatch.setattr(
        kkt, "_cholesky", lambda H: pytest.fail("J^T J must not be factored")
    )
    res = dls(ev, bounds=(np.zeros(2), np.ones(2)), ftol=1e-12, xtol=1e-12)
    assert res.success
    np.testing.assert_allclose(res.x, 0.25, atol=1e-8)
    assert res.active_set == []


def test_failed_cholesky_raises_lambda(monkeypatch):
    # A rank-deficient J at the smallest lambda: the factorization may fail.
    ev = AnalyticEvaluator(
        [[1.0, 1.0], [1.0, 1.0]], [1.0, 1.0], [[1.0, -1.0]], [0.0], [True], [2.0, -2.0]
    )
    real = kkt._cholesky
    calls = [0]

    def flaky(H):
        calls[0] += 1
        if calls[0] <= 2:
            raise be.linalg.LinAlgError("not positive definite")
        return real(H)

    monkeypatch.setattr(kkt, "_cholesky", flaky)
    res = dls(ev, ftol=1e-12, xtol=1e-12)
    assert res.success
    np.testing.assert_allclose(res.x, [0.5, 0.5], atol=1e-6)


# ---------------------------------------------------------------------------
# Restoration and graceful degrade
# ---------------------------------------------------------------------------


def test_nonlinear_equality_restored():
    # Circle row: min (x0 - 2)^2 + x1^2 s.t. x0^2 + x1^2 = 1: x = (1, 0).
    ev = AnalyticEvaluator(
        np.eye(2),
        [2.0, 0.0],
        np.zeros((1, 2)),
        [0.0],
        [True],
        [0.1, 0.5],
        rows=lambda x: np.array([x @ x - 1.0]),
    )
    ev.constraint_jacobian = lambda x: 2.0 * x.reshape(1, 2)
    # J^T J omits the row curvature, so this is a linear rate: ask for more.
    res = dls(ev, ftol=1e-15, xtol=1e-12)
    assert res.success
    np.testing.assert_allclose(res.x, [1.0, 0.0], atol=1e-5)
    # 2 (x - a) + mu 2 x = 0 at (1, 0): mu = 1.
    assert res.multipliers["row0"] == pytest.approx(1.0, abs=1e-3)


def test_inconsistent_rows_degrade_gracefully():
    # x0 <= -1 and x0 >= 1 cannot both hold.
    ev = AnalyticEvaluator(
        np.eye(2),
        [0.0, 0.0],
        [[1.0, 0.0], [-1.0, 0.0]],
        [-1.0, -1.0],
        [False, False],
        [0.0, 1.0],
    )
    res = dls(ev, max_nfev=200)
    assert not res.success
    assert res.status == -5
    assert res.max_constraint_violation == pytest.approx(1.0, abs=1e-6)
    assert "row" in res.message
    assert [row["feasible"] for row in res.constraint_report] == [False, False]


def test_collinear_rows_with_the_same_limit_are_fine():
    ev = AnalyticEvaluator(
        np.eye(2),
        [2.0, 2.0],
        [[1.0, 1.0], [2.0, 2.0]],
        [2.0, 4.0],
        [False, False],
        [0.0, 0.0],
    )
    res = dls(ev, ftol=1e-14, xtol=1e-14)
    assert res.success
    np.testing.assert_allclose(res.x, [1.0, 1.0], atol=1e-7)


def test_non_finite_constraint_rejects_the_trial():
    # sqrt row: undefined below 0.
    ev = AnalyticEvaluator(
        np.eye(1),
        [-4.0],
        np.zeros((1, 1)),
        [0.0],
        [False],
        [1.0],
        rows=lambda x: (
            np.array([np.sqrt(x[0]) - 2.0]) if x[0] >= 0 else np.array([np.nan])
        ),
    )
    ev.constraint_jacobian = lambda x: np.array([[0.5 / np.sqrt(max(x[0], 1e-6))]])
    res = dls(ev, ftol=1e-10, xtol=1e-10)
    assert np.all(np.isfinite(res.fun))
    assert res.x[0] >= 0.0


def test_non_finite_start_row_is_a_configuration_error():
    ev = AnalyticEvaluator(
        np.eye(1),
        [0.0],
        [[1.0]],
        [0.0],
        [True],
        [1.0],
        rows=lambda x: np.array([np.nan]),
    )
    with pytest.raises(ConfigurationError):
        dls(ev)


def test_broyden_keeps_rows_exact_enough():
    rng = np.random.default_rng(7)
    M, b = rng.normal(size=(7, 4)), rng.normal(size=7)
    C, d = rng.normal(size=(2, 4)), np.array([0.5, 2.0])
    ev = AnalyticEvaluator(M, b, C, d, [True, False], np.zeros(4))
    exact = dls(ev, ftol=1e-13, xtol=1e-13)
    res = dls(ev, jacobian_update="broyden", ftol=1e-13, xtol=1e-13)
    assert res.success
    np.testing.assert_allclose(res.x, exact.x, atol=1e-6)


def test_loose_ftol_ends_feasible_through_restoration():
    # Curved row, loose ftol: the merit stalls long before the row is met.
    ev = AnalyticEvaluator(
        np.eye(2),
        [2.0, 0.0],
        np.zeros((1, 2)),
        [0.0],
        [True],
        [0.1, 0.5],
        rows=lambda x: np.array([x @ x - 1.0]),
    )
    ev.constraint_jacobian = lambda x: 2.0 * x.reshape(1, 2)
    res = dls(ev, ftol=0.5, xtol=1e-12)
    assert res.success
    assert res.max_constraint_violation <= 1e-9


def test_collapsed_region_restores_then_finishes(monkeypatch):
    # Trials never lower the merit at first: the radius collapses, restoration
    # reaches the row, and the run finishes from the restored point.
    ev = AnalyticEvaluator(
        np.eye(2), [0.0, 0.0], [[1.0, 1.0]], [1.0], [True], [3.0, 3.0]
    )
    real_merit = KKTController.merit
    calls = [0]

    def merit(self, F, s):
        calls[0] += 1
        return real_merit(self, F, s) if calls[0] > 400 else 1.0 + calls[0]

    monkeypatch.setattr(KKTController, "merit", merit)
    res = dls(ev, ftol=1e-12, xtol=1e-12)
    assert res.max_constraint_violation <= 1e-8
    np.testing.assert_allclose(ev.rows(res.x), 0.0, atol=1e-8)


def test_failed_restoration_reports_failure(monkeypatch):
    ev = AnalyticEvaluator(
        np.eye(1), [0.0], np.zeros((1, 1)), [1.0], [True], [0.0]
    )  # the row does not depend on x
    res = dls(ev, ftol=1e-12, xtol=1e-12, max_nfev=100)
    assert not res.success
    assert res.status == -5
    assert res.max_constraint_violation == pytest.approx(1.0)


def test_merit_and_sigma():
    c = KKTController(
        np.array([True, False]),
        np.array([1e-8, 1e-8]),
        ["c", "g"],
        np.full(1, -np.inf),
        np.full(1, np.inf),
    )
    assert c.feasible(np.array([0.0, -5.0]))
    assert not c.feasible(np.array([1e-3, 0.0]))
    assert c.merit(2.0, np.array([0.5, 0.25])) == pytest.approx(2.0 + 0.75 * c.sigma)
    assert c.merit(np.inf, np.zeros(2)) == np.inf


def test_levenberg_hooks_are_neutral():
    from optiland.optimization.control.levenberg import LevenbergController

    c = LevenbergController()
    assert c.feasible(np.zeros(0))
    assert c.merit(3.0, np.zeros(0)) == 3.0
    g = np.ones(2)
    assert c.stationarity(g) is g
    assert c.diagnostics(np.zeros(0))["multipliers"] == {}


def test_unconstrained_run_reports_empty_fields():
    ev = CallableEvaluator(lambda x: x - 1.0, np.zeros(2))
    res = dls(ev)
    assert res.multipliers == {}
    assert res.active_set == []
    assert res.max_constraint_violation == 0.0
    assert res.constraint_report == []


# ---------------------------------------------------------------------------
# Real lenses
# ---------------------------------------------------------------------------


def _cooke(set_bounds=False):
    lens = CookeTriplet()
    problem = OptimizationProblem()
    for field in lens.fields.get_field_coords():
        for py in (-1.0, 0.0, 1.0):
            problem.add_operand(
                "real_y_intercept",
                target=0.0,
                input_data={
                    "optic": lens,
                    "surface_number": -1,
                    "Hx": 0.0,
                    "Hy": float(field[1]),
                    "Px": 0.0,
                    "Py": py,
                    "wavelength": 0.55,
                },
            )
    for s in (1, 2, 3, 4, 5, 6):
        kw = {"min_val": -1e3, "max_val": 1e3} if set_bounds else {}
        problem.add_variable(lens, "radius", surface_number=s, **kw)
    return lens, problem


def test_lens_equality_and_inequality(set_test_backend):
    lens, problem = _cooke()
    f0 = float(be.to_numpy(lens.paraxial.f2()))
    problem.add_constraint("f2", target=f0 * 1.02, input_data={"optic": lens})
    ray = {
        "optic": lens,
        "surface_number": 3,
        "Hx": 0.0,
        "Hy": 1.0,
        "Px": 0.0,
        "Py": 1.0,
        "wavelength": 0.55,
    }
    limit = float(be.to_numpy(operand_registry.get("real_y_intercept")(**ray)))
    problem.add_constraint("real_y_intercept", max_val=limit + 2.0, input_data=ray)

    start = float(be.to_numpy(problem.sum_squared()))
    # A long valley: a loose tol still has to reach feasibility before stopping.
    res = minimize(problem, "dls", tol=1e-2, maxiter=800)
    assert res.success, res.message
    assert res.max_constraint_violation <= 1e-7
    assert float(be.to_numpy(lens.paraxial.f2())) == pytest.approx(f0 * 1.02, abs=1e-6)
    assert float(be.to_numpy(problem.sum_squared())) < start
    assert any(label.startswith("#0") for label in res.active_set)


def test_lens_bounds_hold_every_trial(set_test_backend):
    lens, problem = _cooke()
    for var in problem.variables:
        value = float(be.to_numpy(var.variable.get_value()))
        var.min_val, var.max_val = sorted((value * 0.99, value * 1.01))
    res = minimize(problem, "dls", maxiter=300)
    for var in problem.variables:
        value = float(be.to_numpy(var.variable.get_value()))
        assert var.min_val - 1e-12 <= value <= var.max_val + 1e-12
    assert res.active_set


def test_constraints_add_no_design_evaluations():
    def run(constrained):
        lens, problem = _cooke()
        if constrained:
            problem.add_constraint(
                "f2", min_val=-1e6, max_val=1e6, input_data={"optic": lens}
            )
        ev = FiniteDiffEvaluator(problem)
        r = ev.residuals(ev.read_x())
        ev.jacobian(ev.read_x(), r0=r)
        ev.constraints(ev.read_x())
        ev.constraint_jacobian(ev.read_x())
        return ev.nfev, ev.njev

    assert run(True) == run(False)


def test_failing_rows_name_the_constraint():
    from optiland.optimization.failure import describe_rows

    lens, problem = _cooke()
    problem.add_constraint("f2", min_val=1.0, input_data={"optic": lens})
    names = describe_rows(problem, [len(problem.operands)])
    assert names.endswith("f2")


# ---------------------------------------------------------------------------
# Legacy paths
# ---------------------------------------------------------------------------


def _constrained():
    lens, problem = _cooke()
    problem.add_constraint("f2", target=100.0, input_data={"optic": lens})
    return problem


@pytest.mark.parametrize("method_choice", ["lm", "trf", "dogbox"])
def test_scipy_least_squares_raises(method_choice):
    with pytest.raises(ConfigurationError, match="penalty operands"):
        LeastSquares(_constrained()).optimize(method_choice=method_choice)


def test_generic_and_global_optimizers_raise():
    with pytest.raises(ConfigurationError, match="hard constraints"):
        OptimizerGeneric(_constrained()).optimize()
    with pytest.raises(ConfigurationError, match="hard constraints"):
        DualAnnealing(_constrained()).optimize()
    with pytest.raises(ConfigurationError, match="hard constraints"):
        CMAES(_constrained()).optimize()
    with pytest.raises(ConfigurationError, match="hard constraints"):
        OrthogonalDescent(_constrained()).optimize()


@pytest.mark.parametrize("method", ["l-bfgs-b", "slsqp", "least_squares"])
def test_minimize_non_dls_raises(method):
    with pytest.raises(ConfigurationError, match="hard constraints"):
        minimize(_constrained(), method)


def test_legacy_dls_runs_constraints(set_test_backend):
    lens, problem = _cooke()
    f0 = float(be.to_numpy(lens.paraxial.f2()))
    problem.add_constraint("f2", target=f0, input_data={"optic": lens})
    res = LeastSquares(problem).optimize(method_choice="dls", tol=1e-5, maxiter=400)
    assert res.max_constraint_violation <= 1e-7


def test_penalty_operands_still_work_everywhere():
    lens, problem = _cooke()
    problem.add_operand("f2", min_val=1.0, max_val=1e6, input_data={"optic": lens})
    LeastSquares(problem).optimize(method_choice="trf", maxiter=5)
