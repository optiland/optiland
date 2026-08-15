from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from optiland.optimization import OrthogonalDescent
from optiland.optimization.problem import OptimizationProblem


class MockVariable:
    def __init__(self, value=0.0):
        self._value = value
        self.min_val = -10.0
        self.max_val = 10.0
        self.value = value  # This is the scaled value property in real GenericVariable

    def update(self, val):
        self.value = val
        self._value = val

    def apply(self):
        pass


def test_orthogonal_descent_simple_quadratic():
    # Minimizing f(x, y) = (x-1)^2 + (y+2)^2
    # Solution: x=1, y=-2

    # Mock Problem
    problem = MagicMock(spec=OptimizationProblem)
    problem.variables = []

    # Mock Variables
    var_x = MagicMock()
    var_x.value = np.array(0.0)  # Initial value
    var_x.min_val = -5.0
    var_x.max_val = 5.0
    var_x.bounds = (-5.0, 5.0)

    var_y = MagicMock()
    var_y.value = np.array(0.0)
    var_y.min_val = -5.0
    var_y.max_val = 5.0
    var_y.bounds = (-5.0, 5.0)

    def update_x(val):
        var_x.value = np.array(val)

    def update_y(val):
        var_y.value = np.array(val)

    var_x.update = SideEffect(update_x)
    var_y.update = SideEffect(update_y)

    problem.variables = [var_x, var_y]
    problem.initial_value = 5.0  # (0-1)^2 + (0+2)^2 = 1 + 4 = 5

    def calculate_rss():
        vx = var_x.value.item()
        vy = var_y.value.item()
        return MagicMock(item=lambda: (vx - 1) ** 2 + (vy + 2) ** 2)

    problem.rss.side_effect = calculate_rss

    optimizer = OrthogonalDescent(problem)
    optimizer.optimize(max_iter=10, tol=1e-6)

    final_x = var_x.value.item()
    final_y = var_y.value.item()

    assert np.isclose(final_x, 1.0, atol=1e-4)
    assert np.isclose(final_y, -2.0, atol=1e-4)


def test_orthogonal_descent_uses_scaled_variable_bounds():
    """Search in the same scaled coordinate system used by Variable.update."""
    problem = MagicMock(spec=OptimizationProblem)
    variable = MagicMock()
    variable.value = np.array(-0.5)
    variable.min_val = 1.0
    variable.max_val = 10.0
    variable.bounds = (-0.9, 0.0)

    def update(value):
        variable.value = np.array(value)

    variable.update.side_effect = update
    problem.rss.side_effect = lambda: MagicMock(
        item=lambda: (variable.value.item() + 0.7) ** 2
    )

    optimizer = OrthogonalDescent(problem)
    optimizer._optimize_variable(variable)

    assert np.isclose(variable.value.item(), -0.7, atol=1e-4)


class SideEffect:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
