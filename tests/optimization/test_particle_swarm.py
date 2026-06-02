from __future__ import annotations

import numpy as np
import pytest
from scipy import optimize

import optiland.backend as be
from optiland.optimization.optimizer.custom.particle_swarm import ParticleSwarm


class MockVariable:
    """Mock variable class for testing particle swarm optimizer."""

    def __init__(self, value: float, bounds: tuple[float, float] | None) -> None:
        self.value = value
        self.bounds = bounds
        self.history: list[float] = []

    def update(self, val: float) -> None:
        """Update the variable value."""
        self.value = val
        self.history.append(val)


class MockProblem:
    """Mock problem class for testing particle swarm optimizer."""

    def __init__(self, variables: list[MockVariable], target_fun=None) -> None:
        self.variables = variables
        self.initial_value = 0.0
        self.optics_updated = False
        self.target_fun = target_fun

    def update_optics(self) -> None:
        """Update optics state."""
        self.optics_updated = True

    def sum_squared(self) -> float:
        """Calculate the sum of squares of the variable values."""
        if self.target_fun:
            return self.target_fun([v.value for v in self.variables])
        return sum(be.to_numpy(v.value).item() ** 2 for v in self.variables)


class TestParticleSwarm:
    """Unit tests for ParticleSwarm optimizer."""

    def test_particle_swarm_init(self, set_test_backend) -> None:
        """Test ParticleSwarm initialization."""
        variables = [MockVariable(1.0, (-5.0, 5.0))]
        problem = MockProblem(variables)
        optimizer = ParticleSwarm(problem)

        assert optimizer.problem == problem
        assert len(optimizer._x) == 0

    def test_particle_swarm_simple_optimization(self, set_test_backend) -> None:
        """Test standard optimization flow on a simple quadratic problem."""
        # Objective: minimize f(x, y) = x^2 + (y - 1)^2
        # Minimum is at (0, 1)
        variables = [MockVariable(3.0, (-5.0, 5.0)), MockVariable(-2.0, (-5.0, 5.0))]

        def target_fun(vals):
            return vals[0] ** 2 + (vals[1] - 1.0) ** 2

        problem = MockProblem(variables, target_fun=target_fun)
        optimizer = ParticleSwarm(problem)

        callback_called = False

        def callback():
            nonlocal callback_called
            callback_called = True

        result = optimizer.optimize(
            maxiter=50,
            swarm_size=20,
            inertia=0.6,
            individual=1.5,
            social=1.5,
            tol=1e-6,
            stall_iterations=10,
            seed=42,
            disp=True,
            callback=callback,
        )

        assert isinstance(result, optimize.OptimizeResult)
        assert result.success is True
        assert np.isclose(result.x[0], 0.0, atol=1e-2)
        assert np.isclose(result.x[1], 1.0, atol=1e-2)
        assert np.isclose(result.fun, 0.0, atol=1e-3)
        assert result.swarm_size == 20
        assert result.inertia == 0.6
        assert result.individual == 1.5
        assert result.social == 1.5
        assert callback_called is True
        assert problem.optics_updated is True

        # Verify that variables in the problem were updated to the best solution
        assert np.isclose(problem.variables[0].value, 0.0, atol=1e-2)
        assert np.isclose(problem.variables[1].value, 1.0, atol=1e-2)

    def test_particle_swarm_default_swarm_size(self, set_test_backend) -> None:
        """Test that swarm size defaults to max(20, 10 * ndim)."""
        variables = [
            MockVariable(1.0, (-5.0, 5.0)),
            MockVariable(1.0, (-5.0, 5.0)),
            MockVariable(1.0, (-5.0, 5.0)),
        ]
        problem = MockProblem(variables)
        optimizer = ParticleSwarm(problem)

        result = optimizer.optimize(maxiter=5, swarm_size=None, disp=False, seed=42)
        assert result.swarm_size == 30  # max(20, 10 * 3) = 30

        # Test with 1 variable
        variables_1 = [MockVariable(1.0, (-5.0, 5.0))]
        problem_1 = MockProblem(variables_1)
        optimizer_1 = ParticleSwarm(problem_1)

        result_1 = optimizer_1.optimize(maxiter=5, swarm_size=None, disp=False, seed=42)
        assert result_1.swarm_size == 20  # max(20, 10 * 1) = 20

    def test_particle_swarm_invalid_swarm_size(self, set_test_backend) -> None:
        """Test that invalid swarm size raises ValueError."""
        variables = [MockVariable(1.0, (-5.0, 5.0))]
        problem = MockProblem(variables)
        optimizer = ParticleSwarm(problem)

        with pytest.raises(ValueError, match="swarm_size must be at least 2."):
            optimizer.optimize(swarm_size=1)

    def test_particle_swarm_missing_bounds(self, set_test_backend) -> None:
        """Test that missing bounds raise ValueError."""
        variables = [MockVariable(1.0, None)]
        problem = MockProblem(variables)
        optimizer = ParticleSwarm(problem)

        with pytest.raises(
            ValueError, match="PSO requires all variables to have finite bounds."
        ):
            optimizer.optimize()

    def test_particle_swarm_invalid_bounds_length(self, set_test_backend) -> None:
        """Test that invalid bounds length raises ValueError."""
        # bounds list not of length 2
        variables = [MockVariable(1.0, (1.0,))]  # type: ignore
        problem = MockProblem(variables)
        optimizer = ParticleSwarm(problem)

        with pytest.raises(
            ValueError, match="PSO requires all variables to have finite bounds."
        ):
            optimizer.optimize()

    def test_particle_swarm_infinite_bounds(self, set_test_backend) -> None:
        """Test that infinite bounds raise ValueError."""
        variables = [MockVariable(1.0, (-np.inf, 5.0))]
        problem = MockProblem(variables)
        optimizer = ParticleSwarm(problem)

        with pytest.raises(
            ValueError, match="PSO requires all variables to have finite bounds."
        ):
            optimizer.optimize()

        variables_upper = [MockVariable(1.0, (-5.0, np.inf))]
        problem_upper = MockProblem(variables_upper)
        optimizer_upper = ParticleSwarm(problem_upper)

        with pytest.raises(
            ValueError, match="PSO requires all variables to have finite bounds."
        ):
            optimizer_upper.optimize()

    def test_particle_swarm_inverted_bounds(self, set_test_backend) -> None:
        """Test that upper bound less than lower bound raises ValueError."""
        variables = [MockVariable(1.0, (5.0, -5.0))]
        problem = MockProblem(variables)
        optimizer = ParticleSwarm(problem)

        with pytest.raises(
            ValueError, match="Each variable bound must satisfy lower <= upper."
        ):
            optimizer.optimize()

    def test_particle_swarm_fixed_variables(self, set_test_backend) -> None:
        """Test optimization when some variables are fixed (lower == upper)."""
        # x is movable, y is fixed at 2.0
        variables = [MockVariable(3.0, (-5.0, 5.0)), MockVariable(2.0, (2.0, 2.0))]

        def target_fun(vals):
            return vals[0] ** 2 + (vals[1] - 1.0) ** 2

        problem = MockProblem(variables, target_fun=target_fun)
        optimizer = ParticleSwarm(problem)

        result = optimizer.optimize(
            maxiter=50, swarm_size=20, stall_iterations=5, seed=42, disp=False
        )

        assert result.success is True
        # x should converge to 0
        assert np.isclose(result.x[0], 0.0, atol=5e-2)
        # y should remain fixed at 2.0
        assert result.x[1] == 2.0
        assert problem.variables[1].value == 2.0

    def test_particle_swarm_all_fixed_variables(self, set_test_backend) -> None:
        """Test optimization when all variables are fixed."""
        variables = [MockVariable(1.0, (1.0, 1.0)), MockVariable(2.0, (2.0, 2.0))]
        problem = MockProblem(variables)
        optimizer = ParticleSwarm(problem)

        result = optimizer.optimize(
            maxiter=10, swarm_size=10, stall_iterations=2, seed=42, disp=False
        )

        assert result.success is True
        assert result.x[0] == 1.0
        assert result.x[1] == 2.0

    def test_particle_swarm_stall_detection(self, set_test_backend) -> None:
        """Test that optimization stops early when stalling is detected."""
        variables = [MockVariable(1.0, (-5.0, 5.0)), MockVariable(1.0, (-5.0, 5.0))]
        # Using a flat function so it will stall immediately
        problem = MockProblem(variables, target_fun=lambda x: 42.0)
        optimizer = ParticleSwarm(problem)

        result = optimizer.optimize(
            maxiter=100,
            swarm_size=20,
            tol=10.0,
            stall_iterations=2,
            seed=42,
            disp=False,
        )

        assert result.success is True
        assert "global best stalled" in result.message
        assert result.nit < 100
