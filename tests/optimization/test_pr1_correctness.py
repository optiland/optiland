"""Regression tests for the PR1 correctness fixes.

Covers SPEC_opt_finalization_20260605.md §3.1–§3.8:

* §3.1 state/optic resync after a bounded (clamped) step
* §3.2 cross-solver objective consistency on a multi-weight problem
* §3.3 Gauss-Newton backend safety + Armijo descent guard
* §3.4 torch first-order: gradient populated + post-step merit recomputed
* §3.5 unified non-finite / trace-failure policy (reject / raise / penalty)
* §3.6 latching ``AndCriterion`` + ``success`` semantics + intrinsic reason
* §3.7 scaler domain guards

Kramer Harrison, 2026
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.optimization import minimize, optimization
from optiland.optimization.drivers import _derive_success
from optiland.optimization.errors import ConfigurationError, OptimizationFailure
from optiland.optimization.failure import (
    is_nonfinite,
    normalize_failure_mode,
    penalty_merit,
    penalty_residual,
)
from optiland.optimization.operand.operand import Operand
from optiland.optimization.scaling.linear import LinearScaler
from optiland.optimization.scaling.log import LogScaler
from optiland.optimization.scaling.power import PowerScaler
from optiland.optimization.stopping.composite import AndCriterion
from optiland.samples.objectives import CookeTriplet


def _f2(lens) -> float:
    """Measured paraxial focal length of ``lens`` (backend-agnostic float)."""
    op = Operand(operand_type="f2", target=0.0, weight=1.0, input_data={"optic": lens})
    return float(be.to_numpy(op.value))


# ---------------------------------------------------------------------------
# §3.7 Scaler domain guards
# ---------------------------------------------------------------------------


class TestScalerGuards:
    def test_linear_zero_factor_rejected(self):
        with pytest.raises(ConfigurationError):
            LinearScaler(factor=0.0)

    def test_log_nonpositive_rejected(self):
        scaler = LogScaler()
        with pytest.raises(ConfigurationError):
            scaler.scale(-1.0)
        with pytest.raises(ConfigurationError):
            scaler.scale(0.0)

    def test_log_positive_ok(self):
        scaler = LogScaler()
        assert math.isclose(float(be.to_numpy(scaler.scale(1.0))), 0.0, abs_tol=1e-12)

    def test_power_zero_rejected(self):
        with pytest.raises(ConfigurationError):
            PowerScaler(power=0.0)

    def test_power_even_on_negative_rejected(self):
        with pytest.raises(ConfigurationError):
            PowerScaler(power=2.0).scale(-3.0)

    def test_power_fractional_on_negative_rejected(self):
        with pytest.raises(ConfigurationError):
            PowerScaler(power=0.5).scale(-3.0)

    def test_power_odd_on_negative_ok(self):
        val = float(be.to_numpy(PowerScaler(power=3.0).scale(-2.0)))
        assert math.isclose(val, -8.0, rel_tol=1e-9)

    def test_power_monotonicity_flag(self):
        assert PowerScaler(power=2.0).monotonic is True
        assert PowerScaler(power=-1.0).monotonic is False


# ---------------------------------------------------------------------------
# §3.5 Failure-policy helpers + evaluator integration
# ---------------------------------------------------------------------------


class TestFailurePolicyHelpers:
    def test_normalize_modes(self):
        assert normalize_failure_mode(None) == "reject"
        assert normalize_failure_mode("RAISE") == "raise"
        assert normalize_failure_mode("penalty") == "penalty"
        with pytest.raises(ConfigurationError):
            normalize_failure_mode("nonsense")

    def test_is_nonfinite(self):
        assert is_nonfinite(float("nan"))
        assert is_nonfinite(float("inf"))
        assert not is_nonfinite(1.5)

    def test_penalty_is_scale_proportional(self):
        # Penalty scales with the reference merit, never a hardcoded 1e10.
        assert penalty_merit(1.0) < penalty_merit(1e3)
        assert penalty_merit(1.0) == pytest.approx(1e6)
        # penalty_residual squares-sum back to penalty_merit.
        m = 4
        r = penalty_residual(10.0, m)
        assert m * r**2 == pytest.approx(penalty_merit(10.0))


class _NaNOperand:
    """Operand stub that always evaluates to NaN."""

    operand_type = "nan_stub"
    target = 0.0
    min_val = None
    max_val = None
    weight = 1.0
    input_data: dict = {}

    def effective_weight(self, optic=None):
        return 1.0

    def delta(self):
        return be.array(float("nan"))


def _nan_problem():
    lens = CookeTriplet()
    problem = optimization.OptimizationProblem(batching=False)
    problem.add_variable(lens, "radius", surface_number=1)
    problem.operands.operands.append(_NaNOperand())
    return problem


class TestEvaluatorFailurePolicy:
    def test_reject_returns_nonfinite(self):
        from optiland.optimization.evaluators.finite_difference import (
            FiniteDiffEvaluator,
        )

        ev = FiniteDiffEvaluator(_nan_problem(), on_failure="reject")
        x = ev.read_x()
        assert not math.isfinite(ev.value(x))
        assert np.all(~np.isfinite(ev.residuals(x)))

    def test_raise_raises(self):
        from optiland.optimization.evaluators.finite_difference import (
            FiniteDiffEvaluator,
        )

        ev = FiniteDiffEvaluator(_nan_problem(), on_failure="raise")
        x = ev.read_x()
        with pytest.raises(OptimizationFailure):
            ev.value(x)

    def test_penalty_returns_finite(self):
        from optiland.optimization.evaluators.finite_difference import (
            FiniteDiffEvaluator,
        )

        ev = FiniteDiffEvaluator(_nan_problem(), on_failure="penalty")
        x = ev.read_x()
        v = ev.value(x)
        assert math.isfinite(v) and v > 0
        r = ev.residuals(x)
        assert np.all(np.isfinite(r))

    def test_managed_does_not_crash_on_trace_failure(self):
        # A run whose merit goes non-finite must complete, not raise (§3.5).
        problem = _nan_problem()
        result = minimize(problem, method="l-bfgs-b", maxiter=3, disp=False)
        assert result is not None


# ---------------------------------------------------------------------------
# §3.6 Stopping / driver semantics
# ---------------------------------------------------------------------------


class _FixedCriterion:
    """Fires (stops) exactly on the iteration matching ``fire_on``."""

    def __init__(self, fire_on: int, reason: str):
        self._fire_on = fire_on
        self._reason = reason

    def reset(self, state):
        pass

    def should_stop(self, state):
        if state.iteration == self._fire_on:
            return True, self._reason
        return False, None


class _StateStub:
    def __init__(self, iteration):
        self.iteration = iteration
        self.value = 0.0


class TestAndCriterionLatching:
    def test_latches_until_both_have_fired(self):
        # ``a`` fires at iter 1, ``b`` fires at iter 3; they never fire on the
        # same iteration, so a non-latching AND would never stop.
        crit = AndCriterion(
            _FixedCriterion(1, "cost_tol"), _FixedCriterion(3, "max_iter")
        )
        crit.reset(_StateStub(0))
        assert crit.should_stop(_StateStub(0)) == (False, None)
        assert crit.should_stop(_StateStub(1))[0] is False  # only a fired
        assert crit.should_stop(_StateStub(2))[0] is False  # still latched
        stop, reason = crit.should_stop(_StateStub(3))  # b fires → both latched
        assert stop is True
        assert "cost_tol" in reason and "max_iter" in reason

    def test_reset_clears_latches(self):
        crit = AndCriterion(
            _FixedCriterion(1, "cost_tol"), _FixedCriterion(1, "max_iter")
        )
        crit.should_stop(_StateStub(1))
        crit.reset(_StateStub(0))
        # After reset both latches cleared; nothing fired at iteration 0.
        assert crit.should_stop(_StateStub(0)) == (False, None)


class TestSuccessSemantics:
    def test_convergence_reasons_are_success(self):
        assert _derive_success("cost_tol=1e-03 (rel_change=1e-04)")
        assert _derive_success("grad_norm=1e-07 < tol=1e-06")
        assert _derive_success("step_stall: |Δx|=1e-09 < tol=1e-08")

    def test_maxiter_alone_is_not_success(self):
        assert not _derive_success("max_iter=1000 reached")

    def test_hard_failure_is_not_success(self):
        assert not _derive_success("lm_lambda_max: all trials rejected")
        assert not _derive_success("gauss_newton_singular: JᵀJ is singular")
        assert not _derive_success("infeasible: ‖c‖=1e-2")

    def test_latched_and_with_convergence_is_success(self):
        assert _derive_success("(max_iter=1000 reached) and (cost_tol=1e-03)")


# ---------------------------------------------------------------------------
# §3.2 Cross-solver objective consistency (multi-weight)
# ---------------------------------------------------------------------------


class TestCrossSolverAgreement:
    def _build(self):
        """Conflicting, differently-weighted operands on one variable.

        The weighted least-squares optimum depends on the weights; before the
        §3.2 fix scipy ``least_squares`` minimized ``Σ(weight·δ)²`` (i.e.
        ``weight²`` weighting) and diverged from ``dls`` / ``l-bfgs-b``.
        """
        lens = CookeTriplet()
        f0 = _f2(lens)
        problem = optimization.OptimizationProblem()
        problem.add_variable(lens, "radius", surface_number=0)
        problem.add_operand(
            operand_type="f2", target=f0 * 0.9, weight=1.0, input_data={"optic": lens}
        )
        problem.add_operand(
            operand_type="f2", target=f0 * 1.1, weight=9.0, input_data={"optic": lens}
        )
        return problem

    def test_dls_ls_lbfgs_agree(self):
        merits = {}
        for method in ("dls", "least_squares", "l-bfgs-b"):
            problem = self._build()
            result = minimize(
                problem, method=method, maxiter=200, tol=1e-10, disp=False
            )
            merits[method] = result.value
        ref = merits["dls"]
        for method, m in merits.items():
            assert m == pytest.approx(ref, rel=5e-2), (method, merits)


# ---------------------------------------------------------------------------
# §3.1 Bounded LM resync
# ---------------------------------------------------------------------------


class TestBoundedResync:
    def test_optic_state_matches_result_and_bounds(self):
        lens = CookeTriplet()
        f0 = _f2(lens)
        problem = optimization.OptimizationProblem()
        # Tight bounds around the start so an unreachable target drives the
        # variable hard against a bound.
        r0 = float(be.to_numpy(lens.surfaces.surfaces[1].geometry.radius))
        problem.add_variable(
            lens, "radius", surface_number=1, min_val=r0 - 1.0, max_val=r0 + 1.0
        )
        problem.add_operand(
            operand_type="f2",
            target=f0 * 5.0,  # unreachable within the bound box
            weight=1.0,
            input_data={"optic": lens},
        )
        result = minimize(problem, method="dls", maxiter=80, tol=1e-12, disp=False)

        var = problem.variables[0]
        val = float(be.to_numpy(var.value))
        lo, hi = var.bounds
        # Bounds honored …
        assert lo - 1e-9 <= val <= hi + 1e-9
        # … and the live optic state equals the reported solution (no desync).
        assert val == pytest.approx(
            float(np.atleast_1d(be.to_numpy(result.x))[0]), abs=1e-9
        )


# ---------------------------------------------------------------------------
# §3.3 Gauss-Newton descent guard
# ---------------------------------------------------------------------------


class TestGaussNewtonGuard:
    def test_gn_reduces_merit(self):
        lens = CookeTriplet()
        f0 = _f2(lens)
        problem = optimization.OptimizationProblem()
        problem.add_variable(lens, "radius", surface_number=0)
        problem.add_variable(lens, "radius", surface_number=1)
        problem.add_operand(
            operand_type="f2", target=f0 * 1.05, weight=1.0, input_data={"optic": lens}
        )
        result = minimize(
            problem, method="gauss_newton", maxiter=60, tol=1e-12, disp=False
        )
        # The descent guard must never let the merit increase past the start.
        assert result.value <= result.initial_value + 1e-9


# ---------------------------------------------------------------------------
# §3.4 Torch first-order convergence plumbing
# ---------------------------------------------------------------------------


class TestTorchGradientPlumbing:
    def test_gradient_and_post_step_merit_populated(self):
        torch = pytest.importorskip("torch")  # noqa: F841
        be.set_backend("torch")
        try:
            from optiland.optimization.evaluators.autograd import AutogradEvaluator
            from optiland.optimization.native.torch_opt import TorchOptimizer

            lens = CookeTriplet()
            f0 = _f2(lens)
            problem = optimization.OptimizationProblem()
            problem.add_variable(lens, "radius", surface_number=0)
            problem.add_operand(
                operand_type="f2",
                target=f0 * 1.02,
                weight=1.0,
                input_data={"optic": lens},
            )
            ev = AutogradEvaluator(problem)
            opt = TorchOptimizer(optimizer_name="adam", lr=1e-2)
            from optiland.optimization.control.identity import IdentityController

            state = opt.initialize(
                ev, ev.read_x(), controller=IdentityController(), constraints=None
            )
            state = opt.step(state)
            assert state.gradient is not None
            assert "grad_norm" in state.scratch
            assert math.isfinite(state.scratch["grad_norm"])
            assert math.isfinite(float(be.to_numpy(state.value)))
        finally:
            be.set_backend("numpy")
