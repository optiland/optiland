"""Phase 1 optimization rework tests.

Covers:
  - PR1a: state, errors, evaluators, stopping, observers, drivers, api.minimize()
  - PR1b: weighted_residuals() correctness (Σ r² == sum_squared())
  - PR1c: variable_vector() / set_variable_vector() round-trips
  - PR1d: HistoryObserver, ProgressObserver, CancelToken, BoxBoundsStrategy,
          ScipyNativeStrategy, StepStall, GradNormTolerance, MaxEvals, & / |
  - PR1e: ConfigurationError for torch methods under numpy, unknown methods

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.optimization import optimization
from optiland.optimization.errors import ConfigurationError
from optiland.optimization.state import (
    Capability,
    EvalCapability,
    OptimizationResult,
    OptimizationState,
)
from optiland.samples.objectives import CookeTriplet

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_problem():
    """A small doublet-style problem with two equality operands."""
    lens = CookeTriplet()
    p = optimization.OptimizationProblem(batching=False)
    p.add_operand(
        operand_type="f2",
        target=50.0,
        weight=1.0,
        input_data={"optic": lens},
    )
    p.add_operand(
        operand_type="total_track",
        target=100.0,
        weight=2.0,
        input_data={"optic": lens},
    )
    return p


@pytest.fixture()
def problem_with_variable(simple_problem):
    """Simple problem + one radius variable."""
    lens = simple_problem.operands[0].input_data["optic"]
    simple_problem.add_variable(lens, "radius", surface_number=1)
    return simple_problem


# ---------------------------------------------------------------------------
# PR1a — state / errors / evaluators
# ---------------------------------------------------------------------------


class TestState:
    def test_optimization_state_defaults(self):
        state = OptimizationState(x=np.zeros(3), value=5.0)
        assert state.iteration == 0
        assert state.n_value_evals == 0
        assert state.converged is False
        assert state.stop_reason is None
        assert state.scratch == {}

    def test_optimization_result_aliases(self):
        r = OptimizationResult(
            x=np.zeros(2),
            success=True,
            status="max_iter reached",
            value=0.5,
            initial_value=1.0,
            improvement_pct=50.0,
            iterations=10,
            n_value_evals=100,
            n_jac_evals=0,
            wall_time_s=0.1,
            history=None,
            backend="numpy",
            method="l-bfgs-b",
        )
        assert r.fun == r.value
        assert r.message == r.status

    def test_capability_enum(self):
        assert Capability.OBSERVE in frozenset({Capability.OBSERVE})

    def test_eval_capability_enum(self):
        assert EvalCapability.VALUE in frozenset(
            {EvalCapability.VALUE, EvalCapability.GRADIENT}
        )


class TestConfigurationError:
    def test_is_value_error(self):
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("test")

    def test_dls_not_implemented(self):
        lens = CookeTriplet()
        p = optimization.OptimizationProblem()
        p.add_operand("f2", target=50.0, weight=1.0, input_data={"optic": lens})
        with pytest.raises(ConfigurationError, match="Phase 2"):
            from optiland.optimization.api import minimize

            minimize(p, method="dls")

    def test_adam_under_numpy_raises(self):
        assert be.get_backend() == "numpy"
        lens = CookeTriplet()
        p = optimization.OptimizationProblem()
        p.add_operand("f2", target=50.0, weight=1.0, input_data={"optic": lens})
        with pytest.raises(ConfigurationError, match="torch"):
            from optiland.optimization.api import minimize

            minimize(p, method="adam")

    def test_unknown_method_raises(self):
        lens = CookeTriplet()
        p = optimization.OptimizationProblem()
        p.add_operand("f2", target=50.0, weight=1.0, input_data={"optic": lens})
        with pytest.raises(ConfigurationError, match="Unknown method"):
            from optiland.optimization.api import minimize

            minimize(p, method="nonexistent_method")


class TestFiniteDiffEvaluator:
    def test_read_write_round_trip(self, problem_with_variable):
        from optiland.optimization.evaluators.finite_difference import (
            FiniteDiffEvaluator,
        )

        ev = FiniteDiffEvaluator(problem_with_variable)
        x0 = ev.read_x()
        assert len(x0) == 1
        ev.write_x(x0)
        x1 = ev.read_x()
        np.testing.assert_allclose(x0, x1)

    def test_value_returns_float(self, problem_with_variable):
        from optiland.optimization.evaluators.finite_difference import (
            FiniteDiffEvaluator,
        )

        ev = FiniteDiffEvaluator(problem_with_variable)
        x0 = ev.read_x()
        v = ev.value(x0)
        assert isinstance(v, float)

    def test_residuals_shape(self, problem_with_variable):
        from optiland.optimization.evaluators.finite_difference import (
            FiniteDiffEvaluator,
        )

        ev = FiniteDiffEvaluator(problem_with_variable)
        x0 = ev.read_x()
        r = ev.residuals(x0)
        assert r.ndim == 1
        assert r.shape[0] == 2  # two operands

    def test_gradient_shape(self, problem_with_variable):
        from optiland.optimization.evaluators.finite_difference import (
            FiniteDiffEvaluator,
        )

        ev = FiniteDiffEvaluator(problem_with_variable)
        x0 = ev.read_x()
        g = ev.gradient(x0)
        assert g.shape == (1,)

    def test_jacobian_shape(self, problem_with_variable):
        from optiland.optimization.evaluators.finite_difference import (
            FiniteDiffEvaluator,
        )

        ev = FiniteDiffEvaluator(problem_with_variable)
        x0 = ev.read_x()
        J = ev.jacobian(x0)
        assert J.shape == (2, 1)

    def test_provides_all_capabilities(self, problem_with_variable):
        from optiland.optimization.evaluators.finite_difference import (
            FiniteDiffEvaluator,
        )

        ev = FiniteDiffEvaluator(problem_with_variable)
        assert EvalCapability.VALUE in ev.provides
        assert EvalCapability.JACOBIAN in ev.provides

    def test_central_scheme(self, problem_with_variable):
        from optiland.optimization.evaluators.finite_difference import (
            FiniteDiffEvaluator,
        )

        ev = FiniteDiffEvaluator(problem_with_variable, scheme="central")
        x0 = ev.read_x()
        g = ev.gradient(x0)
        assert g.shape == (1,)


# ---------------------------------------------------------------------------
# PR1b — weighted_residuals correctness
# ---------------------------------------------------------------------------


class TestWeightedResiduals:
    def test_sum_equals_sum_squared_single_weight(self):
        """Σ r² == sum_squared() with uniform weights."""
        lens = CookeTriplet()
        p = optimization.OptimizationProblem(batching=False)
        p.add_operand("f2", target=45.0, weight=1.0, input_data={"optic": lens})
        p.add_operand(
            "total_track", target=80.0, weight=3.0, input_data={"optic": lens}
        )

        r = p.weighted_residuals()
        s = float(be.to_numpy(p.sum_squared()))
        r_sum = float(be.to_numpy(be.sum(r**2)))
        assert abs(r_sum - s) < 1e-10, f"Σr²={r_sum} != sum_squared={s}"

    def test_sum_equals_sum_squared_batched(self):
        """Same identity holds when batching is enabled."""
        lens = CookeTriplet()
        p = optimization.OptimizationProblem(batching=True)
        p.add_operand(
            operand_type="real_y_intercept",
            target=0.0,
            weight=2.0,
            input_data={
                "optic": lens,
                "surface_number": -1,
                "Hx": 0.0,
                "Hy": 0.7,
                "Px": 0.0,
                "Py": 0.0,
                "wavelength": 0.55,
            },
        )
        p.add_operand("f2", target=45.0, weight=1.5, input_data={"optic": lens})

        r = p.weighted_residuals()
        s = float(be.to_numpy(p.sum_squared()))
        r_sum = float(be.to_numpy(be.sum(r**2)))
        assert abs(r_sum - s) < 1e-10, f"Σr²={r_sum} != sum_squared={s}"

    def test_empty_operands(self):
        p = optimization.OptimizationProblem(batching=False)
        r = p.weighted_residuals()
        assert len(r) == 0

    def test_residual_vector_alias(self):
        """residual_vector() should return the same as weighted_residuals()."""
        lens = CookeTriplet()
        p = optimization.OptimizationProblem(batching=False)
        p.add_operand("f2", target=45.0, weight=1.0, input_data={"optic": lens})

        r_wr = be.to_numpy(p.weighted_residuals())
        r_rv = be.to_numpy(p.residual_vector())
        np.testing.assert_allclose(r_wr, r_rv)


# ---------------------------------------------------------------------------
# PR1c — variable_vector / set_variable_vector
# ---------------------------------------------------------------------------


class TestVariableVector:
    def test_variable_vector_length(self, problem_with_variable):
        x = problem_with_variable.variable_vector()
        assert len(x) == 1

    def test_set_variable_vector_round_trip(self, problem_with_variable):
        p = problem_with_variable
        x0 = np.array(p.variable_vector())
        p.set_variable_vector(x0 * 1.0)
        x1 = np.array(p.variable_vector())
        np.testing.assert_allclose(x0, x1)

    def test_empty_variables(self):
        p = optimization.OptimizationProblem(batching=False)
        x = p.variable_vector()
        assert len(x) == 0


# ---------------------------------------------------------------------------
# PR1d — stopping criteria
# ---------------------------------------------------------------------------


class TestStoppingCriteria:
    def _make_state(self, iteration=0, value=1.0, n_evals=0, step_norm=None, grad=None):
        return OptimizationState(
            x=np.zeros(2),
            value=value,
            gradient=grad,
            iteration=iteration,
            n_value_evals=n_evals,
            step_norm=step_norm,
        )

    def test_max_iter_fires(self):
        from optiland.optimization.stopping.criteria import MaxIter

        c = MaxIter(5)
        s = self._make_state(iteration=5)
        stop, reason = c.should_stop(s)
        assert stop
        assert "5" in reason

    def test_max_iter_not_fires(self):
        from optiland.optimization.stopping.criteria import MaxIter

        c = MaxIter(5)
        s = self._make_state(iteration=4)
        stop, _ = c.should_stop(s)
        assert not stop

    def test_cost_tolerance(self):
        from optiland.optimization.stopping.criteria import CostTolerance

        c = CostTolerance(tol=0.1)
        s0 = self._make_state(value=1.0)
        c.reset(s0)
        s1 = self._make_state(value=1.0 + 1e-12)
        stop, reason = c.should_stop(s1)
        assert stop
        assert "cost_tol" in reason

    def test_cost_tolerance_not_fires_on_large_change(self):
        from optiland.optimization.stopping.criteria import CostTolerance

        c = CostTolerance(tol=0.01)
        s0 = self._make_state(value=1.0)
        c.reset(s0)
        s1 = self._make_state(value=0.5)
        stop, _ = c.should_stop(s1)
        assert not stop

    def test_max_evals(self):
        from optiland.optimization.stopping.criteria import MaxEvals

        c = MaxEvals(10)
        stop, _ = c.should_stop(self._make_state(n_evals=10))
        assert stop
        stop, _ = c.should_stop(self._make_state(n_evals=9))
        assert not stop

    def test_step_stall(self):
        from optiland.optimization.stopping.criteria import StepStall

        c = StepStall(tol=1e-6)
        stop, _ = c.should_stop(self._make_state(step_norm=1e-7))
        assert stop
        stop, _ = c.should_stop(self._make_state(step_norm=1e-5))
        assert not stop
        stop, _ = c.should_stop(self._make_state(step_norm=None))
        assert not stop

    def test_grad_norm_tolerance(self):
        from optiland.optimization.stopping.criteria import GradNormTolerance

        c = GradNormTolerance(tol=1e-5)
        g = np.array([1e-8, 1e-8])
        stop, _ = c.should_stop(self._make_state(grad=g))
        assert stop
        g_large = np.array([1.0, 0.0])
        stop, _ = c.should_stop(self._make_state(grad=g_large))
        assert not stop

    def test_or_composition(self):
        from optiland.optimization.stopping.criteria import CostTolerance, MaxIter

        c = MaxIter(100) | CostTolerance(0.1)
        s0 = self._make_state(value=1.0)
        c.reset(s0)
        s1 = self._make_state(iteration=1, value=1.0)
        stop, _ = c.should_stop(s1)
        assert stop  # CostTolerance fires (tiny change)

    def test_and_composition_requires_both(self):
        from optiland.optimization.stopping.criteria import CostTolerance, MaxIter

        c = MaxIter(5) & CostTolerance(0.1)
        s0 = self._make_state(value=1.0)
        c.reset(s0)
        # MaxIter NOT fires (iter=2 < 5), CostTolerance fires
        s1 = self._make_state(iteration=2, value=1.0)
        stop, _ = c.should_stop(s1)
        assert not stop  # Need BOTH

    def test_criteria_reset(self):
        from optiland.optimization.stopping.criteria import CostTolerance

        c = CostTolerance(tol=0.1)
        s0 = self._make_state(value=1.0)
        c.reset(s0)
        assert c._prev == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# PR1d — observers
# ---------------------------------------------------------------------------


class TestObservers:
    def _make_result(self):
        return OptimizationResult(
            x=np.zeros(1),
            success=True,
            status="test",
            value=0.5,
            initial_value=1.0,
            improvement_pct=50.0,
            iterations=5,
            n_value_evals=10,
            n_jac_evals=0,
            wall_time_s=0.01,
            history=None,
            backend="numpy",
            method="l-bfgs-b",
        )

    def test_history_observer(self):
        from optiland.optimization.observers.history import HistoryObserver

        obs = HistoryObserver()
        s0 = OptimizationState(x=np.zeros(1), value=1.0)
        obs.on_start(s0)
        s1 = OptimizationState(x=np.zeros(1), value=0.8, iteration=1)
        obs.on_step(s1)
        s2 = OptimizationState(x=np.zeros(1), value=0.6, iteration=2)
        obs.on_step(s2)
        assert len(obs.records) == 3  # start + 2 steps
        assert obs.records[0]["iteration"] == 0
        assert obs.records[1]["value"] == pytest.approx(0.8)

    def test_progress_observer_throttle(self):
        from optiland.optimization.observers.logging import ProgressObserver

        obs = ProgressObserver(log_every=5)
        s0 = OptimizationState(x=np.zeros(1), value=1.0)
        obs.on_start(s0)
        for i in range(1, 8):
            si = OptimizationState(x=np.zeros(1), value=1.0 - i * 0.1, iteration=i)
            obs.on_step(si)
        # Only emits at iter 5
        assert obs.step_iteration == 5
        assert obs.step_value == pytest.approx(0.5)

    def test_cancel_token(self):
        from optiland.optimization.observers.cancel import CancelToken

        tok = CancelToken()
        assert not tok.cancelled
        tok.cancel()
        assert tok.cancelled
        tok.reset()
        assert not tok.cancelled

    def test_cancel_observer_raises(self):
        from optiland.optimization.observers.cancel import CancelObserver, CancelToken

        tok = CancelToken()
        tok.cancel()
        obs = CancelObserver(tok)
        s = OptimizationState(x=np.zeros(1), value=0.5, iteration=1)
        with pytest.raises(StopIteration):
            obs.on_step(s)

    def test_console_observer_runs(self, capsys):
        from optiland.optimization.observers.logging import ConsoleObserver

        obs = ConsoleObserver()
        s0 = OptimizationState(x=np.zeros(1), value=1.0)
        obs.on_start(s0)
        s1 = OptimizationState(x=np.zeros(1), value=0.8, iteration=1)
        obs.on_step(s1)
        result = self._make_result()
        obs.on_end(s1, result)
        out = capsys.readouterr().out
        assert "Optimization started" in out
        assert "Optimization ended" in out


# ---------------------------------------------------------------------------
# PR1d — constraints
# ---------------------------------------------------------------------------


class TestConstraints:
    def test_box_bounds_clamp(self):
        from optiland.optimization.constraints.bounds import BoxBoundsStrategy

        strat = BoxBoundsStrategy()
        # Manually inject bounds
        strat._bounds = [(0.0, 10.0), (None, 5.0)]
        x = np.array([12.0, 7.0])
        state = OptimizationState(x=x, value=0.0)
        x_clamped = strat.apply_to_step(x, state)
        assert x_clamped[0] == pytest.approx(10.0)
        assert x_clamped[1] == pytest.approx(5.0)

    def test_box_bounds_to_scipy(self):
        from optiland.optimization.constraints.bounds import BoxBoundsStrategy

        strat = BoxBoundsStrategy()
        strat._bounds = [(0.0, 10.0), (None, None)]
        result = strat.to_scipy()
        assert result[0] == (0.0, 10.0)
        assert result[1][0] == -np.inf

    def test_box_bounds_is_feasible(self):
        from optiland.optimization.constraints.bounds import BoxBoundsStrategy

        strat = BoxBoundsStrategy()
        strat._bounds = [(0.0, 10.0)]
        assert strat.is_feasible(np.array([5.0]))
        assert not strat.is_feasible(np.array([-1.0]))

    def test_composite_strategy(self):
        from optiland.optimization.constraints.base import CompositeStrategy
        from optiland.optimization.constraints.bounds import BoxBoundsStrategy
        from optiland.optimization.constraints.scipy_native import ScipyNativeStrategy

        b = BoxBoundsStrategy()
        b._bounds = [(0.0, 5.0)]
        n = ScipyNativeStrategy()
        comp = CompositeStrategy([b, n])
        x = np.array([7.0])
        state = OptimizationState(x=x, value=0.0)
        x_clamped = comp.apply_to_step(x, state)
        assert x_clamped[0] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# PR1a — api.minimize() end-to-end (numpy path)
# ---------------------------------------------------------------------------


class TestMinimizeAPI:
    def _make_simple_problem(self):
        lens = CookeTriplet()
        p = optimization.OptimizationProblem()
        p.add_operand("f2", target=50.0, weight=1.0, input_data={"optic": lens})
        p.add_operand(
            "total_track", target=100.0, weight=1.0, input_data={"optic": lens}
        )
        p.add_variable(lens, "radius", surface_number=1)
        return p

    def test_minimize_lbfgsb(self):
        from optiland.optimization.api import minimize

        p = self._make_simple_problem()
        result = minimize(p, method="l-bfgs-b", maxiter=5, tol=1e-2, disp=False)
        assert isinstance(result, OptimizationResult)
        assert result.backend == "numpy"
        assert result.method == "l-bfgs-b"
        assert len(result.x) == 1

    def test_minimize_bfgs(self):
        from optiland.optimization.api import minimize

        p = self._make_simple_problem()
        result = minimize(p, method="bfgs", maxiter=3, tol=1e-2, disp=False)
        assert isinstance(result, OptimizationResult)

    def test_minimize_least_squares(self):
        from optiland.optimization.api import minimize

        p = self._make_simple_problem()
        result = minimize(p, method="least_squares", maxiter=5, tol=1e-2, disp=False)
        assert isinstance(result, OptimizationResult)

    def test_minimize_auto_numpy(self):
        from optiland.optimization.api import minimize

        p = self._make_simple_problem()
        # auto with numpy + all-equality + m >= n -> least_squares
        result = minimize(p, method="auto", maxiter=5, disp=False)
        assert isinstance(result, OptimizationResult)

    def test_minimize_with_history_observer(self):
        from optiland.optimization.api import minimize
        from optiland.optimization.observers.history import HistoryObserver

        p = self._make_simple_problem()
        hist = HistoryObserver()
        result = minimize(p, method="l-bfgs-b", maxiter=3, disp=False, observers=[hist])
        assert isinstance(result, OptimizationResult)

    def test_minimize_with_cancel_token_numpy(self):
        from optiland.optimization.api import minimize
        from optiland.optimization.observers.cancel import CancelToken

        p = self._make_simple_problem()
        tok = CancelToken()
        # Cancel immediately — optimizer should still return cleanly
        tok.cancel()
        result = minimize(
            p, method="l-bfgs-b", maxiter=100, disp=False, cancel_token=tok
        )
        assert isinstance(result, OptimizationResult)

    def test_minimize_result_duck_compat(self):
        """OptimizationResult is duck-compatible with scipy.optimize.OptimizeResult."""
        from optiland.optimization.api import minimize

        p = self._make_simple_problem()
        result = minimize(p, method="l-bfgs-b", maxiter=3, disp=False)
        assert hasattr(result, "x")
        assert hasattr(result, "fun")
        assert hasattr(result, "success")
        assert hasattr(result, "message")
        assert result.fun == result.value

    def test_minimize_lm_phase2_raises(self):
        from optiland.optimization.api import minimize

        p = self._make_simple_problem()
        with pytest.raises(ConfigurationError, match="Phase 2"):
            minimize(p, method="lm", disp=False)

    def test_minimize_slsqp(self):
        from optiland.optimization.api import minimize

        p = self._make_simple_problem()
        result = minimize(p, method="slsqp", maxiter=3, disp=False)
        assert isinstance(result, OptimizationResult)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_evaluator_protocol(self, problem_with_variable):
        from optiland.optimization.evaluators.base import Evaluator
        from optiland.optimization.evaluators.finite_difference import (
            FiniteDiffEvaluator,
        )

        ev = FiniteDiffEvaluator(problem_with_variable)
        assert isinstance(ev, Evaluator)

    def test_stopping_criterion_protocol(self):
        from optiland.optimization.stopping.base import StoppingCriterion
        from optiland.optimization.stopping.criteria import MaxIter

        c = MaxIter(10)
        assert isinstance(c, StoppingCriterion)

    def test_observer_protocol(self):
        from optiland.optimization.observers.base import Observer
        from optiland.optimization.observers.history import HistoryObserver

        obs = HistoryObserver()
        assert isinstance(obs, Observer)
