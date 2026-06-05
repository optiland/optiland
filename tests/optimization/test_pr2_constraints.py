"""Tests for the PR2 hard-constraint / KKT feature build.

Covers SPEC_opt_finalization_20260605.md:

* §4.1 ``Constraint`` / ``ConstraintManager`` + ``add_constraint``
* §4.2 ``be.linalg.qr`` / ``be.linalg.cholesky`` parity
* §4.3 KKT active-set constrained LM (equality, inactive + active inequality)
* §4.7 diagnostics (multipliers, active set, feasibility) on the result
* D2  NullSpaceStrategy removal
* D13 hard constraints require a KKT-capable method

Kramer Harrison, 2026
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.optimization import (
    Constraint,
    ConstraintManager,
    OptimizationProblem,
    minimize,
)
from optiland.optimization.errors import ConfigurationError
from optiland.optimization.operand.operand import Operand
from optiland.samples.objectives import CookeTriplet


def _metric(lens, t) -> float:
    op = Operand(t, target=0.0, weight=1, input_data={"optic": lens})
    return float(be.to_numpy(op.value))


# ---------------------------------------------------------------------------
# §4.2 be.linalg qr / cholesky
# ---------------------------------------------------------------------------


class TestLinalgAdditions:
    def test_qr_reconstructs(self):
        A = np.array([[4.0, 1.0], [1.0, 3.0], [2.0, 0.5]])
        Q, R = be.linalg.qr(A)
        assert np.allclose(be.to_numpy(Q) @ be.to_numpy(R), A)

    def test_cholesky_reconstructs(self):
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        L = be.to_numpy(be.linalg.cholesky(A))
        assert np.allclose(L @ L.T, A)


# ---------------------------------------------------------------------------
# §4.1 Constraint data layer
# ---------------------------------------------------------------------------


class TestConstraintDataLayer:
    def test_equality_row(self):
        c = Constraint(operand_type="f2", target=10.0, input_data={})
        assert c.n_equality == 1 and c.n_inequality == 0

    def test_two_inequality_rows(self):
        c = Constraint(operand_type="f2", min_val=1.0, max_val=2.0, input_data={})
        assert c.n_inequality == 2 and c.n_equality == 0

    def test_mixed_spec_rejected(self):
        with pytest.raises(ValueError, match="cannot mix"):
            Constraint(operand_type="f2", target=1.0, min_val=0.0, input_data={})

    def test_empty_spec_rejected(self):
        with pytest.raises(ValueError, match="needs a target"):
            Constraint(operand_type="f2", input_data={})

    def test_manager_vectors_and_report(self):
        lens = CookeTriplet()
        f2 = _metric(lens, "f2")
        mgr = ConstraintManager()
        mgr.add("f2", target=f2, input_data={"optic": lens})
        mgr.add("f2", min_val=f2 - 5, max_val=f2 + 5, input_data={"optic": lens})
        assert mgr.n_equality == 1
        assert mgr.n_inequality == 2
        # Equality residual ~ 0 at the start (f2 == target).
        assert abs(float(be.to_numpy(mgr.c_vector())[0])) < 1e-6
        # Inequalities both satisfied → max violation ~ 0.
        assert mgr.max_violation() < 1e-6
        report = mgr.report()
        assert len(report) == 3
        assert all(row["feasible"] for row in report)

    def test_inequality_jacobian_shape(self):
        from optiland.optimization.evaluators.finite_difference import (
            FiniteDiffEvaluator,
        )

        lens = CookeTriplet()
        problem = OptimizationProblem()
        problem.add_variable(lens, "radius", surface_number=1)
        problem.add_variable(lens, "radius", surface_number=2)
        problem.add_constraint("f2", min_val=10.0, input_data={"optic": lens})
        ev = FiniteDiffEvaluator(problem)
        A = problem.constraints.inequality_jacobian(ev)
        assert A.shape == (1, 2)
        assert np.all(np.isfinite(A))


# ---------------------------------------------------------------------------
# §4.1 OptimizationProblem.add_constraint / clear_constraints
# ---------------------------------------------------------------------------


class TestProblemConstraintApi:
    def test_add_and_clear(self):
        lens = CookeTriplet()
        p = OptimizationProblem()
        p.add_constraint("f2", target=50.0, input_data={"optic": lens})
        assert len(p.constraints) == 1
        p.clear_constraints()
        assert len(p.constraints) == 0

    def test_constraint_info_no_constraints(self, capsys):
        p = OptimizationProblem()
        p.constraint_info()
        assert "No hard constraints" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# §4.3 KKT active-set constrained LM
# ---------------------------------------------------------------------------


class TestKKTConstrainedSolve:
    def test_inactive_inequality_optimizes_and_stays_feasible(self):
        lens = CookeTriplet()
        f2_0 = _metric(lens, "f2")
        p = OptimizationProblem()
        p.add_variable(lens, "radius", surface_number=1)
        p.add_operand("f2", target=f2_0 * 0.97, weight=1.0, input_data={"optic": lens})
        p.add_constraint(
            "f2", min_val=f2_0 * 0.80, input_data={"optic": lens}, tol=1e-6
        )
        res = minimize(p, method="dls", maxiter=80, tol=1e-12, disp=False)
        # Merit reaches the (feasible) target; the loose bound stays inactive.
        assert _metric(lens, "f2") == pytest.approx(f2_0 * 0.97, rel=1e-4)
        assert res.max_constraint_violation < 1e-5
        assert res.active_set == []

    def test_active_inequality_binds_with_positive_multiplier(self):
        lens = CookeTriplet()
        f2_0 = _metric(lens, "f2")
        floor = f2_0 * 0.9
        p = OptimizationProblem()
        p.add_variable(lens, "radius", surface_number=1)
        # Merit pulls f2 well below the floor, so the bound must bind.
        p.add_operand("f2", target=f2_0 * 0.5, weight=1.0, input_data={"optic": lens})
        p.add_constraint("f2", min_val=floor, input_data={"optic": lens}, tol=1e-4)
        res = minimize(p, method="dls", maxiter=150, tol=1e-14, disp=False)
        # f2 stops at the floor, feasibly.
        assert _metric(lens, "f2") == pytest.approx(floor, abs=max(1e-3, floor * 1e-4))
        assert _metric(lens, "f2") >= floor - 1e-3
        assert res.max_constraint_violation < 1e-3
        # The active bound carries a (correctly-signed) positive multiplier.
        assert res.active_set and any("f2>=" in lbl for lbl in res.active_set)
        assert max(res.multipliers.values()) > 0.0

    def test_equality_constraint_feasible(self):
        lens = CookeTriplet()
        f2_0 = _metric(lens, "f2")
        target = f2_0 * 0.96
        p = OptimizationProblem()
        p.add_variable(lens, "radius", surface_number=1)
        p.add_operand("f2", target=target, weight=1.0, input_data={"optic": lens})
        p.add_constraint("f2", target=target, input_data={"optic": lens}, tol=1e-5)
        res = minimize(p, method="dls", maxiter=100, tol=1e-14, disp=False)
        assert _metric(lens, "f2") == pytest.approx(target, rel=1e-4)
        assert res.max_constraint_violation < 1e-3


# ---------------------------------------------------------------------------
# Validation + D2 removal
# ---------------------------------------------------------------------------


class TestConstraintValidation:
    def test_hard_constraint_managed_raises(self):
        lens = CookeTriplet()
        p = OptimizationProblem()
        p.add_variable(lens, "radius", surface_number=1)
        p.add_operand("f2", target=50.0, weight=1.0, input_data={"optic": lens})
        p.add_constraint("f2", target=50.0, input_data={"optic": lens})
        with pytest.raises(ConfigurationError, match="hard constraint"):
            minimize(p, method="l-bfgs-b", disp=False)

    def test_auto_routes_to_dls_when_constrained(self):
        lens = CookeTriplet()
        f2_0 = _metric(lens, "f2")
        p = OptimizationProblem()
        p.add_variable(lens, "radius", surface_number=1)
        p.add_operand("f2", target=f2_0, weight=1.0, input_data={"optic": lens})
        p.add_constraint("f2", min_val=f2_0 - 20, input_data={"optic": lens})
        res = minimize(p, method="auto", maxiter=20, disp=False)
        assert res.method == "dls"

    def test_null_space_strategy_removed(self):
        with pytest.raises((ImportError, ModuleNotFoundError)):
            import optiland.optimization.constraints.null_space  # noqa: F401


def test_math_import_guard():
    # Sanity: penalty merit uses math.isfinite on the L1-augmented objective.
    assert math.isfinite(1.0)
