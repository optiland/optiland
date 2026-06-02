"""Tests for Phase 3: NullSpaceStrategy (exact CODE V-style projection).

Covers:
- Protocol conformance (ConstraintStrategy)
- Null-space projection correctness for a linear equality constraint
- Newton restoration to feasibility
- Active-set detection for inequality constraints
- Rank-deficiency handling (redundant / zero constraints)
- ConfigurationError for NullSpaceStrategy with managed optimizers
- Integration: minimize a quadratic subject to a linear equality constraint

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import pytest

from optiland.optimization.constraints.base import ConstraintStrategy
from optiland.optimization.constraints.null_space import (
    NullSpaceStrategy,
    _null_space_basis,
)
from optiland.optimization.errors import ConfigurationError
from optiland.optimization.state import OptimizationState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(x: np.ndarray) -> OptimizationState:
    """Build a minimal OptimizationState at x with a dummy value."""
    return OptimizationState(x=x.copy(), value=float(np.dot(x, x)))


# ---------------------------------------------------------------------------
# 1. Protocol conformance
# ---------------------------------------------------------------------------


def test_null_space_satisfies_constraint_strategy_protocol():
    def h(x):
        return np.array([x[0]])
    strat = NullSpaceStrategy(equality_fn=h)
    assert isinstance(strat, ConstraintStrategy)


def test_null_space_protocol_methods_exist():
    def h(x):
        return np.array([x[0]])
    strat = NullSpaceStrategy(equality_fn=h)
    assert hasattr(strat, "prepare")
    assert hasattr(strat, "apply_to_step")
    assert hasattr(strat, "to_scipy")
    assert hasattr(strat, "is_feasible")


def test_to_scipy_returns_none():
    def h(x):
        return np.array([x[0]])
    strat = NullSpaceStrategy(equality_fn=h)
    assert strat.to_scipy() is None


def test_constructor_raises_without_constraints():
    with pytest.raises(ValueError, match="at least one"):
        NullSpaceStrategy()


# ---------------------------------------------------------------------------
# 2. Null-space basis helper
# ---------------------------------------------------------------------------


def test_null_space_basis_1d_constraint():
    """For A = [[1, 0, 0]], null space should be the yz-plane."""
    A = np.array([[1.0, 0.0, 0.0]])
    Z = _null_space_basis(A)
    assert Z.shape == (3, 2)
    # AZ = 0
    assert np.allclose(A @ Z, 0, atol=1e-12)
    # Z orthonormal
    assert np.allclose(Z.T @ Z, np.eye(2), atol=1e-12)


def test_null_space_basis_full_rank():
    """For a full-rank square A, null space is empty."""
    A = np.eye(3)
    Z = _null_space_basis(A)
    assert Z.shape[1] == 0


def test_null_space_basis_zero_matrix():
    """Zero matrix: null space is all of R^n."""
    A = np.zeros((2, 4))
    Z = _null_space_basis(A)
    assert Z.shape == (4, 4)


def test_null_space_basis_rank_deficient():
    """Repeated rows should give rank 1, null space of dimension n-1."""
    A = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    Z = _null_space_basis(A, rank_tol=1e-8)
    # rank = 1, null space dim = 2
    assert Z.shape == (3, 2)
    assert np.allclose(A @ Z, 0, atol=1e-10)


# ---------------------------------------------------------------------------
# 3. Projection correctness (linear equality constraint)
# ---------------------------------------------------------------------------


def test_projection_satisfies_linear_constraint():
    """x[0] = 0: the projected step must not change x[0]."""
    def h(x):
        return np.array([x[0]])
    strat = NullSpaceStrategy(equality_fn=h, restore_tol=1e-10)

    # Start feasible at x = [0, 1, 2]
    x_curr = np.array([0.0, 1.0, 2.0])
    state = _make_state(x_curr)

    # Proposed step violates constraint: x[0] would become 5
    x_proposed = np.array([5.0, 3.0, -1.0])
    x_new = strat.apply_to_step(x_proposed, state)

    assert np.isclose(x_new[0], 0.0, atol=1e-8), f"x[0]={x_new[0]} should be 0"


def test_projection_preserves_null_space_component():
    """Component along the null space should be preserved."""
    def h(x):
        return np.array([x[0]])
    strat = NullSpaceStrategy(equality_fn=h, restore_tol=1e-10)

    x_curr = np.array([0.0, 0.0, 0.0])
    state = _make_state(x_curr)

    x_proposed = np.array([3.0, 2.0, 1.0])
    x_new = strat.apply_to_step(x_proposed, state)

    # x[1] and x[2] should be approximately preserved (null-space components)
    assert np.isclose(x_new[1], 2.0, atol=1e-7)
    assert np.isclose(x_new[2], 1.0, atol=1e-7)


def test_projection_two_equality_constraints():
    """h(x) = [x[0], x[1]]: null space is the x[2] axis."""
    def h(x):
        return x[:2]
    strat = NullSpaceStrategy(equality_fn=h, restore_tol=1e-10)

    x_curr = np.array([0.0, 0.0, 5.0])
    state = _make_state(x_curr)

    # Proposed step moves x[0] and x[1]; only x[2] change should survive
    x_proposed = np.array([3.0, -2.0, 8.0])
    x_new = strat.apply_to_step(x_proposed, state)

    assert np.isclose(x_new[0], 0.0, atol=1e-8)
    assert np.isclose(x_new[1], 0.0, atol=1e-8)
    # x[2] change should be preserved: Δx[2] = 8-5 = 3
    assert np.isclose(x_new[2], 8.0, atol=1e-7)


# ---------------------------------------------------------------------------
# 4. Newton restoration
# ---------------------------------------------------------------------------


def test_restoration_from_infeasible_start():
    """If x_proposed already violates h(x)=0, restoration should fix it."""
    def h(x):
        return np.array([x[0] - 1.0])  # constraint: x[0] = 1
    strat = NullSpaceStrategy(equality_fn=h, restore_tol=1e-10, restore_max_iter=20)

    # Start near-feasible
    x_curr = np.array([1.0, 0.0, 0.0])
    state = _make_state(x_curr)

    # Large violation
    x_proposed = np.array([5.0, 2.0, -1.0])
    x_new = strat.apply_to_step(x_proposed, state)

    assert np.isclose(x_new[0], 1.0, atol=1e-7), f"x[0]={x_new[0]} should be ~1"


def test_is_feasible():
    def h(x):
        return np.array([x[0]])
    strat = NullSpaceStrategy(equality_fn=h)

    assert strat.is_feasible(np.array([0.0, 1.0]))
    assert not strat.is_feasible(np.array([0.1, 1.0]))


# ---------------------------------------------------------------------------
# 5. Active-set detection for inequalities
# ---------------------------------------------------------------------------


def test_inactive_inequality_ignored():
    """Inactive inequality (g << 0) should not constrain the step."""
    def g(x):
        return x[0] - 10.0  # g(x) = x[0] - 10 <= 0
    strat = NullSpaceStrategy(
        inequality_fns=[g],
        active_tol=1e-6,
    )

    x_curr = np.array([0.0, 0.0])  # g = -10: inactive
    state = _make_state(x_curr)

    x_proposed = np.array([2.0, 3.0])
    x_new = strat.apply_to_step(x_proposed, state)

    # No constraint active → step passes through unchanged
    np.testing.assert_array_almost_equal(x_new, x_proposed)


def test_active_inequality_projected():
    """Active inequality (g ≈ 0) constrains the step."""
    def g(x):
        return x[0]  # g(x) = x[0] <= 0
    strat = NullSpaceStrategy(
        inequality_fns=[g],
        active_tol=1e-6,
        restore_tol=1e-9,
        restore_max_iter=20,
    )

    x_curr = np.array([0.0, 1.0])  # g = 0: active
    state = _make_state(x_curr)

    # Proposed step would push x[0] positive (infeasible)
    x_proposed = np.array([3.0, 2.0])
    x_new = strat.apply_to_step(x_proposed, state)

    # After projection, x[0] should be driven back to ~0
    assert x_new[0] <= 1e-7, f"x[0]={x_new[0]} should be ≤ 0"


def test_violated_inequality_restored():
    """Violated inequality (g > 0) should be restored to the boundary."""
    def g(x):
        return x[0]  # x[0] <= 0
    strat = NullSpaceStrategy(
        inequality_fns=[g],
        active_tol=0.5,  # g(x_curr) = -0.1 < 0.5 so active
        restore_tol=1e-9,
        restore_max_iter=20,
    )

    x_curr = np.array([-0.1, 1.0])  # g = -0.1 < active_tol=0.5 → active
    state = _make_state(x_curr)

    x_proposed = np.array([2.0, 2.0])
    x_new = strat.apply_to_step(x_proposed, state)

    assert x_new[0] <= 1e-7


# ---------------------------------------------------------------------------
# 6. Rank deficiency
# ---------------------------------------------------------------------------


def test_redundant_constraints_handled():
    """Two identical equality constraints: rank=1, treated as one."""
    # h(x) = [x[0], x[0]] — redundant rows
    def h(x):
        return np.array([x[0], x[0]])
    strat = NullSpaceStrategy(equality_fn=h, rank_tol=1e-8, restore_tol=1e-10)

    x_curr = np.array([0.0, 1.0, 2.0])
    state = _make_state(x_curr)

    x_proposed = np.array([5.0, 3.0, -1.0])
    x_new = strat.apply_to_step(x_proposed, state)

    # x[0] should still be driven to 0
    assert np.isclose(x_new[0], 0.0, atol=1e-7)


def test_no_active_constraints_passthrough():
    """When there are no active constraints at all, pass x_proposed unchanged."""
    # equality_fn returns empty (this is constructed indirectly via inequality only)
    def g(x):
        return x[0] - 100.0  # g(x) = x[0] - 100 <= 0
    strat = NullSpaceStrategy(inequality_fns=[g], active_tol=1e-6)

    x_curr = np.array([0.0, 1.0])  # g = -100: inactive
    state = _make_state(x_curr)

    x_proposed = np.array([2.0, 3.0])
    x_new = strat.apply_to_step(x_proposed, state)
    np.testing.assert_array_almost_equal(x_new, x_proposed)


# ---------------------------------------------------------------------------
# 7. ConfigurationError for managed optimizer
# ---------------------------------------------------------------------------


def test_null_space_with_managed_raises_configuration_error():
    """Using NullSpaceStrategy with a managed method must raise ConfigurationError."""
    from unittest.mock import MagicMock

    import optiland.backend as be
    from optiland.optimization.api import minimize

    # Build a minimal mock problem that looks like a managed-optimizer target
    problem = MagicMock()
    problem.operands = []
    problem.variables = []
    problem.initial_value = 0.0
    problem.sum_squared.return_value = be.array(0.0)

    def h(x):
        return np.array([x[0]])
    strat = NullSpaceStrategy(equality_fn=h)

    msg = "NullSpaceStrategy requires a stepped"
    with pytest.raises(ConfigurationError, match=msg):
        minimize(problem, method="l-bfgs-b", constraints=strat, disp=False)


# ---------------------------------------------------------------------------
# 8. Integration: constrained quadratic minimisation
# ---------------------------------------------------------------------------


def test_lm_with_null_space_equality_constraint():
    """LM + NullSpaceStrategy: minimise ||x||² subject to sum(x) = 1 (n=3).

    Unconstrained optimum: x=0.  Constrained optimum: x = [1/3, 1/3, 1/3].
    """
    from unittest.mock import MagicMock


    # ---- Build a synthetic OptimizationProblem ----
    # residuals: r_i = x_i  (merit = ||x||^2)
    # constraint: sum(x) = 1

    _x = np.array([1.0, 0.0, 0.0])  # feasible start: sum=1

    class FakeVariable:
        def __init__(self, idx: int):
            self._idx = idx
            self._val = _x[idx]
            self.bounds = (None, None)

        def get_value(self) -> float:
            return float(_x[self._idx])

        def update_value(self, v: float) -> None:
            _x[self._idx] = v

        @property
        def value(self) -> float:
            return self._val

        def update(self, v: float) -> None:
            _x[self._idx] = v

    variables = [FakeVariable(i) for i in range(3)]

    problem = MagicMock()
    problem.variables = variables
    problem.initial_value = 1.0

    def _sum_sq() -> float:
        return float(np.dot(_x, _x))

    def _weighted_residuals() -> np.ndarray:
        return _x.copy()

    problem.sum_squared.side_effect = _sum_sq
    problem.weighted_residuals.side_effect = _weighted_residuals
    problem.operands = []
    problem.update_optics = MagicMock()

    # Equality constraint: sum(x) = 1
    def h(x):
        return np.array([np.sum(x) - 1.0])
    strat = NullSpaceStrategy(equality_fn=h, restore_tol=1e-9, restore_max_iter=20)

    # Run a few LM steps manually rather than through the full facade to avoid
    # the evaluator's dependency on the full optic stack.
    from optiland.optimization.control.levenberg import LevenbergController
    from optiland.optimization.native.least_squares import LevenbergMarquardt
    from optiland.optimization.stopping.criteria import MaxIter

    # Patch FiniteDiffEvaluator to use our fake problem directly
    evaluator = MagicMock()
    evaluator.n_vars = 3
    evaluator.backend = "numpy"
    evaluator.problem = problem

    def _read_x() -> np.ndarray:
        return _x.copy()

    def _write_x(x: np.ndarray) -> None:
        _x[:] = x

    def _residuals(x: np.ndarray) -> np.ndarray:
        _write_x(x)
        return _x.copy()

    def _jacobian(x: np.ndarray) -> np.ndarray:
        _write_x(x)
        return np.eye(3)

    def _value(x: np.ndarray) -> float:
        _write_x(x)
        return float(np.dot(_x, _x))

    evaluator.read_x.side_effect = _read_x
    evaluator.write_x.side_effect = _write_x
    evaluator.residuals.side_effect = _residuals
    evaluator.jacobian.side_effect = _jacobian
    evaluator.value.side_effect = _value

    ctrl = LevenbergController()
    opt = LevenbergMarquardt()
    state = opt.initialize(evaluator, _x, controller=ctrl, constraints=strat)

    stop = MaxIter(50)
    stop.reset(state)
    for _ in range(50):
        if opt.converged(state):
            break
        _stop, _ = stop.should_stop(state)
        if _stop:
            break
        state = opt.step(state)
        # apply constraint after each step
        if strat is not None:
            state.x = strat.apply_to_step(state.x, state)

    x_final = np.asarray(state.x, dtype=float)

    # The constrained optimum is x = [1/3, 1/3, 1/3]
    assert np.isclose(np.sum(x_final), 1.0, atol=1e-5), (
        f"sum(x)={np.sum(x_final):.6f} should be ~1"
    )
    expected = np.ones(3) / 3
    assert np.allclose(x_final, expected, atol=0.05), (
        f"x={x_final} should be near [1/3, 1/3, 1/3]"
    )
