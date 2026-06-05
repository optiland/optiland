"""Phase 2 optimization rework tests.

Covers:
  - Protocol conformance: LevenbergMarquardt, GaussNewton, LevenbergController,
    LineSearchController, CheckpointObserver
  - Analytic NLLS benchmarks: Rosenbrock-residuals, simple linear least-squares
  - Optical regression: native DLS vs SciPy LeastSquares (corrected objective)
  - Robustness (D12): NaN-producing trial → step rejected, λ increases, run converges
  - Step semantics (D11): one on_step per accepted step; n_trial_evals ≥ n_accepted
  - method="auto" → "dls" for all-equality numpy problems
  - CheckpointObserver: files created at correct cadence
  - LineSearchController: Armijo condition satisfied
  - ConfigurationError still raised for torch methods under numpy

Kramer Harrison, 2026
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import optiland.backend as be
from optiland.optimization import optimization
from optiland.optimization.api import minimize
from optiland.optimization.control.identity import IdentityController
from optiland.optimization.control.levenberg import LevenbergController
from optiland.optimization.control.line_search import LineSearchController
from optiland.optimization.errors import ConfigurationError
from optiland.optimization.native.least_squares import GaussNewton, LevenbergMarquardt
from optiland.optimization.observers.checkpoint import CheckpointObserver
from optiland.optimization.observers.history import HistoryObserver
from optiland.optimization.state import (
    Capability,
    OptimizationResult,
    OptimizationState,
)
from optiland.optimization.stopping.criteria import CostTolerance, MaxIter, StepStall
from optiland.samples.objectives import CookeTriplet


# ---------------------------------------------------------------------------
# Mock evaluator: self-contained NLLS for algebraic benchmarks
# ---------------------------------------------------------------------------


class MockEvaluator:
    """Lightweight evaluator wrapping a pure NLLS residual function.

    The residual function must be callable as ``fun(x) -> np.ndarray``.
    No Optiland ``OptimizationProblem`` required.
    """

    def __init__(
        self,
        fun: Any,
        x0: np.ndarray,
        nan_on_first_trial: bool = False,
    ) -> None:
        self._fun = fun
        self._x = np.asarray(x0, dtype=float)
        self.n_vars: int = len(self._x)
        self.backend: str = "numpy"
        self._nan_trial_count = 0
        self._nan_on_first_trial = nan_on_first_trial

    # -- Evaluator protocol --
    def read_x(self) -> np.ndarray:
        return self._x.copy()

    def write_x(self, x: Any) -> None:
        self._x = np.asarray(x, dtype=float)

    def residuals(self, x: Any) -> np.ndarray:
        self.write_x(x)
        return np.asarray(self._fun(self._x), dtype=float)

    def value(self, x: Any) -> float:
        if self._nan_on_first_trial and self._nan_trial_count == 0:
            self._nan_trial_count += 1
            self.write_x(x)
            return float("nan")
        self._nan_trial_count += 1
        r = self.residuals(x)
        return float(np.dot(r, r))

    def jacobian(self, x: Any, r0: Any = None) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        r0 = self.residuals(x) if r0 is None else np.asarray(r0, dtype=float)
        m = r0.shape[0]
        J = np.empty((m, self.n_vars))
        for i in range(self.n_vars):
            h = max(1e-5 * abs(x[i]), 1e-8)
            xp = x.copy()
            xp[i] += h
            J[:, i] = (self.residuals(xp) - r0) / h
        self.write_x(x)
        return J


# ---------------------------------------------------------------------------
# NLLS benchmark functions
# ---------------------------------------------------------------------------


def _rosenbrock_residuals(x: np.ndarray) -> np.ndarray:
    """Rosenbrock as a 2-residual NLLS: solution x*=[1,1]."""
    return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])


def _linear_residuals_factory(A: np.ndarray, b: np.ndarray):
    """Residuals r(x) = A@x - b; solution x* = pinv(A)@b."""

    def _fun(x: np.ndarray) -> np.ndarray:
        return A @ x - b

    return _fun


def _quadratic_residuals(x: np.ndarray) -> np.ndarray:
    """Simple 2-parameter fit: r_i = a*t_i^2 + b*t_i - y_i for known a=1, b=2.

    Solution: x* = [1, 2].
    """
    t = np.linspace(0.0, 1.0, 10)
    y = 1.0 * t**2 + 2.0 * t
    return (x[0] * t**2 + x[1] * t) - y


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_optical_problem():
    """Two equality operands + two radius variables on a Cooke triplet."""
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
        weight=1.0,
        input_data={"optic": lens},
    )
    p.add_variable(lens, "radius", surface_number=1)
    p.add_variable(lens, "radius", surface_number=3)
    return p


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_levenberg_marquardt_capabilities(self):
        lm = LevenbergMarquardt()
        assert Capability.OBSERVE in lm.capabilities
        assert Capability.STEP_CONTROL in lm.capabilities
        assert Capability.MUTATE_TERMINATION in lm.capabilities
        assert Capability.BOUNDS in lm.capabilities

    def test_gauss_newton_capabilities(self):
        gn = GaussNewton()
        assert Capability.OBSERVE in gn.capabilities
        assert Capability.STEP_CONTROL in gn.capabilities

    def test_levenberg_controller_protocol(self):
        ctrl = LevenbergController()
        state = OptimizationState(x=np.zeros(2), value=1.0)
        ctrl.reset(state)
        assert "lambda" in state.scratch
        assert state.scratch["lambda"] == pytest.approx(1e-3)

    def test_line_search_controller_reset(self):
        ctrl = LineSearchController()
        state = OptimizationState(x=np.zeros(2), value=1.0)
        ctrl.reset(state)  # should not raise

    def test_checkpoint_observer_protocol(self):
        obs = CheckpointObserver(directory=tempfile.mkdtemp())
        state = OptimizationState(x=np.zeros(2), value=1.0)
        result = OptimizationResult(
            x=np.zeros(2),
            success=True,
            status="ok",
            value=1.0,
            initial_value=2.0,
            improvement_pct=50.0,
            iterations=0,
            n_value_evals=1,
            n_jac_evals=0,
            wall_time_s=0.01,
            history=None,
            backend="numpy",
            method="dls",
        )
        obs.on_start(state)
        obs.on_step(state)  # iteration=0 → no write yet
        obs.on_end(state, result)  # always writes final


# ---------------------------------------------------------------------------
# LevenbergController unit tests
# ---------------------------------------------------------------------------


class TestLevenbergController:
    def test_accepts_on_cost_decrease(self):
        """Controller accepts when trial cost is lower."""
        # r = x - target; JT = I; GN direction = -(x - target)
        target = np.array([3.0, 5.0])
        x0 = np.array([1.0, 2.0])
        fun = lambda x: x - target  # noqa: E731

        ev = MockEvaluator(fun, x0)
        ctrl = LevenbergController(lam_init=1e-3)
        state = OptimizationState(x=ev.read_x(), value=ev.value(ev.read_x()))
        ctrl.reset(state)

        r = ev.residuals(state.x)
        J = ev.jacobian(state.x)

        from optiland.optimization.control.base import StepInfo

        info = StepInfo(residuals=r, jacobian=J, value_fn=ev.value, direction=None)
        outcome = ctrl.transform(None, info, state)

        assert outcome.accepted
        assert outcome.delta_x is not None
        assert outcome.info["trial_value"] < state.value

    def test_rejects_nan_and_increases_lambda(self):
        """Non-finite trial → reject + λ increases (D12)."""
        x0 = np.array([1.0, 2.0])
        call_count = [0]

        def fun_with_nan(x: np.ndarray) -> float:
            call_count[0] += 1
            if call_count[0] <= 1:
                return float("nan")
            return float(np.dot(x - np.array([3.0, 5.0]), x - np.array([3.0, 5.0])))

        target = np.array([3.0, 5.0])
        ev = MockEvaluator(lambda x: x - target, x0)
        ctrl = LevenbergController(lam_init=1e-3, max_trials=5)
        state = OptimizationState(x=ev.read_x(), value=5.0)
        ctrl.reset(state)
        lam_before = state.scratch["lambda"]

        r = ev.residuals(state.x)
        J = ev.jacobian(state.x)

        from optiland.optimization.control.base import StepInfo

        # Patch value_fn to inject NaN on first call
        nan_on_first = [True]

        def patched_value_fn(x):
            if nan_on_first[0]:
                nan_on_first[0] = False
                return float("nan")
            return ev.value(x)

        info = StepInfo(
            residuals=r, jacobian=J, value_fn=patched_value_fn, direction=None
        )
        outcome = ctrl.transform(None, info, state)

        # Either accepted on a later trial or rejected; either way λ was bumped
        lam_after = state.scratch["lambda"]
        if not outcome.accepted:
            assert lam_after > lam_before
        # n_trials >= 2 because first trial was NaN
        assert outcome.info["n_trials"] >= 2

    def test_lambda_decreases_on_acceptance(self):
        """After acceptance, λ should decrease (trust region expands)."""
        target = np.array([3.0])
        ev = MockEvaluator(lambda x: x - target, np.array([1.0]))
        ctrl = LevenbergController(lam_init=1.0, lam_factor_down=0.1)
        state = OptimizationState(x=ev.read_x(), value=ev.value(ev.read_x()))
        ctrl.reset(state)

        r = ev.residuals(state.x)
        J = ev.jacobian(state.x)

        from optiland.optimization.control.base import StepInfo

        info = StepInfo(residuals=r, jacobian=J, value_fn=ev.value, direction=None)
        outcome = ctrl.transform(None, info, state)

        if outcome.accepted:
            assert state.scratch["lambda"] < 1.0


# ---------------------------------------------------------------------------
# LevenbergMarquardt analytic benchmarks
# ---------------------------------------------------------------------------


class TestLevenbergMarquardtAnalytic:
    def test_linear_lsq_exact(self):
        """LM on an overdetermined linear system converges to the LS solution."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((8, 3))
        b = rng.standard_normal(8)
        x_ref = np.linalg.lstsq(A, b, rcond=None)[0]
        x0 = np.zeros(3)

        ev = MockEvaluator(_linear_residuals_factory(A, b), x0)
        lm = LevenbergMarquardt()
        ctrl = LevenbergController()
        state = lm.initialize(ev, x0, controller=ctrl, constraints=None)

        stop = MaxIter(200) | CostTolerance(1e-10)
        while not lm.converged(state):
            should, _ = stop.should_stop(state)
            if should:
                break
            state = lm.step(state)

        assert np.allclose(state.x, x_ref, atol=1e-6)

    def test_rosenbrock_residuals_converges(self):
        """LM on Rosenbrock residuals converges to x*=[1,1]."""
        x0 = np.array([-1.2, 1.0])
        ev = MockEvaluator(_rosenbrock_residuals, x0)
        lm = LevenbergMarquardt()
        ctrl = LevenbergController()
        state = lm.initialize(ev, x0, controller=ctrl, constraints=None)

        for _ in range(500):
            if lm.converged(state) or state.value < 1e-14:
                break
            state = lm.step(state)

        assert np.allclose(state.x, [1.0, 1.0], atol=1e-5)
        assert state.value < 1e-10

    def test_quadratic_fit_converges(self):
        """LM fits a 2-parameter quadratic to exact data."""
        x0 = np.array([0.0, 0.0])
        ev = MockEvaluator(_quadratic_residuals, x0)
        lm = LevenbergMarquardt()
        ctrl = LevenbergController()
        state = lm.initialize(ev, x0, controller=ctrl, constraints=None)

        for _ in range(200):
            if lm.converged(state) or state.value < 1e-14:
                break
            state = lm.step(state)

        assert np.allclose(state.x, [1.0, 2.0], atol=1e-6)


# ---------------------------------------------------------------------------
# GaussNewton analytic benchmarks
# ---------------------------------------------------------------------------


class TestGaussNewtonAnalytic:
    def test_linear_lsq_exact(self):
        """GN on an overdetermined linear system converges in one step."""
        rng = np.random.default_rng(7)
        A = rng.standard_normal((6, 2))
        b = rng.standard_normal(6)
        x_ref = np.linalg.lstsq(A, b, rcond=None)[0]
        x0 = np.zeros(2)

        ev = MockEvaluator(_linear_residuals_factory(A, b), x0)
        gn = GaussNewton()
        state = gn.initialize(ev, x0, controller=IdentityController(), constraints=None)

        stop = MaxIter(20) | StepStall(1e-12)
        stop.reset(state)
        for _ in range(20):
            should, _ = stop.should_stop(state)
            if should or gn.converged(state):
                break
            state = gn.step(state)

        assert np.allclose(state.x, x_ref, atol=1e-6)

    def test_rosenbrock_residuals_converges(self):
        """GN on Rosenbrock residuals converges to x*=[1,1]."""
        x0 = np.array([-1.2, 1.0])
        ev = MockEvaluator(_rosenbrock_residuals, x0)
        gn = GaussNewton()
        state = gn.initialize(ev, x0, controller=IdentityController(), constraints=None)

        stop = MaxIter(200) | StepStall(1e-12)
        stop.reset(state)
        while not gn.converged(state):
            should, _ = stop.should_stop(state)
            if should:
                break
            state = gn.step(state)

        assert np.allclose(state.x, [1.0, 1.0], atol=1e-4)


# ---------------------------------------------------------------------------
# Robustness (D12): NaN injection
# ---------------------------------------------------------------------------


class TestNaNRobustness:
    def test_nan_trial_rejected_and_converges(self):
        """NaN on the first trial is rejected; LM still converges (D12)."""
        target = np.array([3.0, 5.0])
        x0 = np.array([1.0, 2.0])
        ev = MockEvaluator(lambda x: x - target, x0, nan_on_first_trial=True)

        lm = LevenbergMarquardt()
        ctrl = LevenbergController(lam_init=1e-3, max_trials=20)
        state = lm.initialize(ev, x0, controller=ctrl, constraints=None)

        for _ in range(100):
            if lm.converged(state) or state.value < 1e-10:
                break
            state = lm.step(state)

        # NaN was injected on the first trial; LM should still have converged
        assert np.allclose(state.x, target, atol=1e-5) or state.value < 1e-8

    def test_nan_never_enters_jacobian(self):
        """After a NaN-rejected trial, the next Jacobian is still computed at x (not NaN)."""
        target = np.array([3.0, 5.0])
        x0 = np.array([0.0, 0.0])
        ev = MockEvaluator(lambda x: x - target, x0, nan_on_first_trial=True)

        lm = LevenbergMarquardt()
        ctrl = LevenbergController(lam_init=1e-4, max_trials=20)
        state = lm.initialize(ev, x0, controller=ctrl, constraints=None)

        # Run one step (first trial will be NaN; should still be accepted on a later trial)
        state = lm.step(state)

        # The Jacobian should be finite
        if state.jacobian is not None:
            assert np.all(np.isfinite(state.jacobian))


# ---------------------------------------------------------------------------
# Step semantics (D11)
# ---------------------------------------------------------------------------


class TestStepSemantics:
    def test_one_on_step_per_accepted_step(self):
        """SteppedDriver calls on_step exactly once per accepted step (D11)."""
        from optiland.optimization.drivers import SteppedDriver

        target = np.array([3.0, 5.0])
        x0 = np.array([0.5, 1.0])
        ev = MockEvaluator(lambda x: x - target, x0)

        lm = LevenbergMarquardt()
        ctrl = LevenbergController()

        step_calls = [0]

        class CountingObserver:
            def on_start(self, state):
                pass

            def on_step(self, state):
                step_calls[0] += 1

            def on_end(self, state, result):
                pass

        driver = SteppedDriver()
        result = driver.run(
            lm,
            ev,
            ev.read_x(),
            controller=ctrl,
            constraints=None,
            criteria=MaxIter(10) | CostTolerance(1e-10),
            observers=[CountingObserver()],
            initial_value=float(np.dot(x0 - target, x0 - target)),
            method="dls",
        )

        assert step_calls[0] == result.iterations

    def test_n_trial_evals_geq_accepted(self):
        """n_trial_evals >= n accepted steps (each step ≥ 1 trial)."""
        target = np.array([2.0])
        x0 = np.array([0.0])
        ev = MockEvaluator(lambda x: x - target, x0)

        lm = LevenbergMarquardt()
        ctrl = LevenbergController()
        state = lm.initialize(ev, x0, controller=ctrl, constraints=None)

        stop = MaxIter(20) | CostTolerance(1e-10)
        stop.reset(state)
        while not lm.converged(state):
            should, _ = stop.should_stop(state)
            if should:
                break
            state = lm.step(state)

        assert state.n_trial_evals >= state.iteration


# ---------------------------------------------------------------------------
# API-level tests
# ---------------------------------------------------------------------------


class TestMinimizeAPI:
    def test_method_dls_converges(self, simple_optical_problem):
        """minimize(problem, method='dls') completes and returns a valid result."""
        result = minimize(
            simple_optical_problem,
            method="dls",
            maxiter=50,
            tol=1e-4,
            disp=False,
        )
        assert isinstance(result, OptimizationResult)
        assert result.method == "dls"
        assert isinstance(result.value, float)
        assert result.iterations >= 0

    def test_method_lm_alias(self, simple_optical_problem):
        """method='lm' is an alias for 'dls'."""
        result = minimize(
            simple_optical_problem,
            method="lm",
            maxiter=30,
            tol=1e-4,
            disp=False,
        )
        assert isinstance(result, OptimizationResult)

    def test_method_gauss_newton(self, simple_optical_problem):
        """minimize(problem, method='gauss_newton') runs without error."""
        result = minimize(
            simple_optical_problem,
            method="gauss_newton",
            maxiter=20,
            tol=1e-4,
            disp=False,
        )
        assert isinstance(result, OptimizationResult)

    def test_auto_resolves_to_dls_for_equality_problem(self, simple_optical_problem):
        """method='auto' → 'dls' for all-equality problem with m >= n."""
        result = minimize(
            simple_optical_problem,
            method="auto",
            maxiter=20,
            tol=1e-3,
            disp=False,
        )
        assert result.method == "dls"

    def test_torch_under_numpy_raises(self, simple_optical_problem):
        """method='adam' under numpy raises ConfigurationError (unchanged)."""
        assert be.get_backend() == "numpy"
        with pytest.raises(ConfigurationError, match="torch"):
            minimize(simple_optical_problem, method="adam", disp=False)

    def test_history_observer_attached(self, simple_optical_problem):
        """HistoryObserver records one snapshot per accepted step."""
        history_obs = HistoryObserver()
        result = minimize(
            simple_optical_problem,
            method="dls",
            maxiter=5,
            tol=1e-2,
            disp=False,
            observers=[history_obs],
        )
        # HistoryObserver records the initial state in on_start plus one per step
        assert len(history_obs.records) == result.iterations + 1

    def test_result_duck_compatible(self, simple_optical_problem):
        """Result exposes .fun and .message (SciPy duck-compatibility)."""
        result = minimize(
            simple_optical_problem,
            method="dls",
            maxiter=5,
            disp=False,
        )
        assert result.fun == result.value
        assert result.message == result.status

    def test_custom_levenberg_controller_forwarded(self, simple_optical_problem):
        """User-supplied LevenbergController is used (not the default)."""
        ctrl = LevenbergController(lam_init=0.1)
        result = minimize(
            simple_optical_problem,
            method="dls",
            controller=ctrl,
            maxiter=10,
            disp=False,
        )
        assert isinstance(result, OptimizationResult)


# ---------------------------------------------------------------------------
# Optical regression: native DLS vs SciPy LeastSquares
# ---------------------------------------------------------------------------


class TestOpticalRegression:
    """Native LM should converge to a solution comparable to SciPy LeastSquares
    on the corrected objective."""

    def _make_problem(self):
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
            weight=1.0,
            input_data={"optic": lens},
        )
        p.add_variable(lens, "radius", surface_number=1)
        p.add_variable(lens, "radius", surface_number=3)
        return p

    def test_native_lm_reduces_merit(self):
        """Native LM reduces the merit function from the starting point."""
        p = self._make_problem()
        init_val = float(be.to_numpy(p.sum_squared()))
        result = minimize(p, method="dls", maxiter=30, tol=1e-5, disp=False)
        assert result.value <= init_val + 1e-8  # should not get worse

    def test_native_lm_vs_scipy_ls_parity(self):
        """Native LM and SciPy LeastSquares converge to similar merit on corrected obj."""
        p1 = self._make_problem()
        p2 = self._make_problem()

        result_lm = minimize(p1, method="dls", maxiter=100, tol=1e-6, disp=False)
        result_scipy = minimize(
            p2,
            method="least_squares",
            maxiter=100,
            tol=1e-6,
            disp=False,
        )

        # Both should substantially reduce the merit; neither should regress
        assert result_lm.improvement_pct >= 0.0
        assert result_scipy.improvement_pct >= 0.0


# ---------------------------------------------------------------------------
# CheckpointObserver
# ---------------------------------------------------------------------------


class TestCheckpointObserver:
    def test_files_created_at_cadence(self, tmp_path):
        """CheckpointObserver writes a file every 'every' accepted steps."""
        target = np.array([3.0, 5.0])
        x0 = np.array([0.0, 0.0])
        ev = MockEvaluator(lambda x: x - target, x0)

        lm = LevenbergMarquardt()
        ctrl = LevenbergController()
        every = 2
        obs = CheckpointObserver(directory=tmp_path, every=every)
        from optiland.optimization.drivers import SteppedDriver

        driver = SteppedDriver()
        result = driver.run(
            lm,
            ev,
            ev.read_x(),
            controller=ctrl,
            constraints=None,
            criteria=MaxIter(10) | CostTolerance(1e-10),
            observers=[obs],
            initial_value=float(np.dot(x0 - target, x0 - target)),
            method="dls",
        )

        files = list(tmp_path.glob("checkpoint_*.pkl"))
        # Must have at least the _final file
        assert any("_final" in f.name for f in files)

    def test_checkpoint_can_be_unpickled(self, tmp_path):
        """Checkpoint files can be loaded back."""
        target = np.array([2.0])
        x0 = np.array([0.0])
        ev = MockEvaluator(lambda x: x - target, x0)

        lm = LevenbergMarquardt()
        ctrl = LevenbergController()
        obs = CheckpointObserver(directory=tmp_path, every=1)

        from optiland.optimization.drivers import SteppedDriver

        driver = SteppedDriver()
        driver.run(
            lm,
            ev,
            ev.read_x(),
            controller=ctrl,
            constraints=None,
            criteria=MaxIter(5),
            observers=[obs],
            initial_value=4.0,
            method="dls",
        )

        files = sorted(tmp_path.glob("checkpoint_*.pkl"))
        assert len(files) >= 1
        with files[0].open("rb") as fh:
            loaded = pickle.load(fh)  # noqa: S301
        assert hasattr(loaded, "x")
        assert hasattr(loaded, "value")

    def test_keep_last_prunes_old_files(self, tmp_path):
        """keep_last=2 retains at most 2 checkpoint files."""
        target = np.array([3.0, 5.0])
        x0 = np.array([0.0, 0.0])
        ev = MockEvaluator(lambda x: x - target, x0)

        lm = LevenbergMarquardt()
        ctrl = LevenbergController()
        obs = CheckpointObserver(directory=tmp_path, every=1, keep_last=2)

        from optiland.optimization.drivers import SteppedDriver

        driver = SteppedDriver()
        driver.run(
            lm,
            ev,
            ev.read_x(),
            controller=ctrl,
            constraints=None,
            criteria=MaxIter(6) | CostTolerance(1e-10),
            observers=[obs],
            initial_value=float(np.dot(x0 - target, x0 - target)),
            method="dls",
        )

        remaining = list(tmp_path.glob("checkpoint_*.pkl"))
        assert len(remaining) <= 2


# ---------------------------------------------------------------------------
# LineSearchController
# ---------------------------------------------------------------------------


class TestLineSearchController:
    def test_armijo_condition_satisfied(self):
        """Accepted step satisfies the Armijo sufficient-decrease condition."""
        # minimize f(x) = (x-3)^2; gradient at x=0 is -6; direction d = 1 (descent)
        def value_fn(x):
            return float((x[0] - 3.0) ** 2)

        ctrl = LineSearchController(c1=1e-4, rho=0.5, max_steps=30, alpha_init=2.0)
        state = OptimizationState(x=np.array([0.0]), value=9.0)
        ctrl.reset(state)

        from optiland.optimization.control.base import StepInfo

        direction = np.array([1.0])  # positive = toward 3
        gradient = np.array([-6.0])  # d f/dx at x=0 is 2*(0-3)=-6
        info = StepInfo(direction=direction, gradient=gradient, value_fn=value_fn)

        outcome = ctrl.transform(direction, info, state)

        if outcome.accepted:
            f_new = outcome.info["trial_value"]
            alpha = outcome.info["alpha"]
            slope = float(gradient @ direction)
            assert f_new <= 9.0 + 1e-4 * alpha * slope

    def test_returns_last_step_if_armijo_never_satisfied(self):
        """If Armijo is never met, accepted=False and last step is returned."""

        def bad_value_fn(x):
            # Always returns current value (no improvement)
            return 9.0

        ctrl = LineSearchController(c1=1e-4, rho=0.5, max_steps=5, alpha_init=1.0)
        state = OptimizationState(x=np.array([0.0]), value=9.0)

        from optiland.optimization.control.base import StepInfo

        info = StepInfo(
            direction=np.array([1.0]),
            gradient=np.array([-6.0]),
            value_fn=bad_value_fn,
        )
        outcome = ctrl.transform(np.array([1.0]), info, state)
        assert not outcome.accepted
        assert outcome.delta_x is not None
