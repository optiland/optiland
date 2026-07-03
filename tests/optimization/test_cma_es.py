from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy import optimize

if TYPE_CHECKING:
    from collections.abc import Callable

import optiland.backend as be
from optiland import optimization
from optiland.optimization.optimizer.custom import CMAES
from optiland.optimization.optimizer.custom.cma_es import CMAES as CMAESDirect


class MockVariable:
    """Mock variable for CMA-ES tests."""

    def __init__(self, value: float, bounds: tuple[float, float] | None) -> None:
        self.value = value
        self.bounds = bounds

    def update(self, value: float) -> None:
        """Update the variable value."""
        self.value = value


class MockProblem:
    """Mock optimization problem with a configurable objective."""

    def __init__(
        self,
        variables: list[MockVariable],
        target_fun: Callable[[list[float]], float] | None = None,
    ) -> None:
        self.variables = variables
        self.initial_value = 0.0
        self.optics_updated = False
        self.evaluation_count = 0
        self.target_fun = target_fun

    def update_optics(self) -> None:
        """Record that the problem state was refreshed."""
        self.optics_updated = True

    def sum_squared(self) -> float:
        """Evaluate the configured objective."""
        self.evaluation_count += 1
        values = [be.to_numpy(variable.value).item() for variable in self.variables]
        if self.target_fun is not None:
            return self.target_fun(values)
        return sum(value**2 for value in values)


class TestCMAES:
    """Unit tests for the bounded CMA-ES optimizer."""

    def test_public_exports(self, set_test_backend) -> None:
        """Test that CMAES is available from public modules."""
        assert CMAES is CMAESDirect
        assert optimization.CMAES is CMAESDirect

    def test_init(self, set_test_backend) -> None:
        """Test CMAES initialization."""
        problem = MockProblem([MockVariable(1.0, (-5.0, 5.0))])

        optimizer = CMAES(problem)

        assert optimizer.problem is problem
        assert optimizer._x == []

    def test_default_population_size_uses_movable_dimensions(
        self, set_test_backend
    ) -> None:
        """Test the classic default population size formula."""
        problem = MockProblem(
            [
                MockVariable(1.0, (-5.0, 5.0)),
                MockVariable(2.0, (-5.0, 5.0)),
                MockVariable(3.0, (-5.0, 5.0)),
                MockVariable(4.0, (4.0, 4.0)),
            ]
        )

        result = CMAES(problem).optimize(
            maxiter=1,
            population_size=None,
            seed=1,
            disp=False,
        )

        assert result.population_size == 7

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"maxiter": 0}, "maxiter must be at least 1."),
            ({"population_size": 1}, "population_size must be at least 2."),
            ({"population_size": 0}, "population_size must be at least 2."),
            ({"sigma0": 0.0}, "sigma0 must be positive."),
            ({"sigma0": -0.1}, "sigma0 must be positive."),
            ({"sigma0": np.nan}, "sigma0 must be positive."),
            ({"tolx": -1.0}, "tolx must be non-negative."),
            ({"tolx": np.nan}, "tolx must be non-negative."),
            ({"tolfun": -1.0}, "tolfun must be non-negative."),
            ({"tolfun": np.nan}, "tolfun must be non-negative."),
        ],
    )
    def test_invalid_parameters_fail_before_evaluation(
        self,
        set_test_backend,
        kwargs: dict[str, float | int],
        message: str,
    ) -> None:
        """Test parameter validation before objective evaluation."""
        problem = MockProblem([MockVariable(1.0, (-5.0, 5.0))])
        optimizer = CMAES(problem)
        evaluations_before = problem.evaluation_count

        with pytest.raises(ValueError, match=message):
            optimizer.optimize(disp=False, **kwargs)

        assert problem.evaluation_count == evaluations_before

    @pytest.mark.parametrize(
        ("bounds", "message"),
        [
            (None, "CMAES requires all variables to have finite bounds."),
            ((1.0,), "CMAES requires all variables to have finite bounds."),
            ((-np.inf, 5.0), "CMAES requires all variables to have finite bounds."),
            ((-5.0, np.inf), "CMAES requires all variables to have finite bounds."),
            ((5.0, -5.0), "Each variable bound must satisfy lower <= upper."),
        ],
    )
    def test_invalid_bounds(
        self,
        set_test_backend,
        bounds,
        message: str,
    ) -> None:
        """Test rejection of unsupported bounds."""
        problem = MockProblem([MockVariable(1.0, bounds)])

        with pytest.raises(ValueError, match=message):
            CMAES(problem).optimize(disp=False)

    def test_current_value_outside_bounds(self, set_test_backend) -> None:
        """Test rejection of an infeasible current design."""
        problem = MockProblem([MockVariable(6.0, (-5.0, 5.0))])

        with pytest.raises(
            ValueError,
            match="Current variable values must lie within their bounds.",
        ):
            CMAES(problem).optimize(disp=False)

    def test_non_finite_current_value(self, set_test_backend) -> None:
        """Test rejection of a non-finite current design."""
        problem = MockProblem([MockVariable(np.nan, (-5.0, 5.0))])

        with pytest.raises(
            ValueError,
            match="Current variable values must lie within their bounds.",
        ):
            CMAES(problem).optimize(disp=False)

    def test_current_design_is_preserved_as_baseline(self, set_test_backend) -> None:
        """Test that random sampling cannot lose a superior current design."""
        variables = [
            MockVariable(0.0, (-5.0, 5.0)),
            MockVariable(1.0, (-5.0, 5.0)),
        ]
        problem = MockProblem(
            variables,
            target_fun=lambda values: values[0] ** 2 + (values[1] - 1.0) ** 2,
        )

        result = CMAES(problem).optimize(
            maxiter=1,
            population_size=6,
            sigma0=0.5,
            seed=2,
            disp=False,
        )

        assert result.fun == 0.0
        assert np.array_equal(result.x, [0.0, 1.0])
        assert result.nfev == 7

    def test_fixed_variable_is_excluded_from_search(self, set_test_backend) -> None:
        """Test that fixed dimensions remain unchanged."""
        variables = [
            MockVariable(3.0, (-5.0, 5.0)),
            MockVariable(2.0, (2.0, 2.0)),
        ]
        problem = MockProblem(
            variables,
            target_fun=lambda values: values[0] ** 2 + (values[1] - 1.0) ** 2,
        )

        result = CMAES(problem).optimize(
            maxiter=150,
            sigma0=0.25,
            tolx=1e-7,
            tolfun=1e-12,
            seed=3,
            disp=False,
        )

        assert np.isclose(result.x[0], 0.0, atol=2e-2)
        assert result.x[1] == 2.0
        assert result.covariance.shape == (1, 1)

    def test_all_fixed_variables(self, set_test_backend) -> None:
        """Test optimization when every variable is fixed."""
        variables = [
            MockVariable(1.0, (1.0, 1.0)),
            MockVariable(2.0, (2.0, 2.0)),
        ]
        problem = MockProblem(variables)

        result = CMAES(problem).optimize(seed=4, disp=False)

        assert result.success is True
        assert result.nit == 0
        assert result.nfev == 1
        assert np.array_equal(result.x, [1.0, 2.0])
        assert result.covariance.shape == (0, 0)
        assert "No movable variables" in result.message
        assert problem.optics_updated is True

    def test_invalid_population_size_with_all_fixed_variables(
        self, set_test_backend
    ) -> None:
        """Test explicit population validation even without movable dimensions."""
        problem = MockProblem([MockVariable(1.0, (1.0, 1.0))])

        with pytest.raises(
            ValueError,
            match="population_size must be at least 2.",
        ):
            CMAES(problem).optimize(population_size=1, disp=False)

    def test_sphere_convergence(self, set_test_backend) -> None:
        """Test convergence on a sphere function."""
        problem = MockProblem(
            [
                MockVariable(3.0, (-5.0, 5.0)),
                MockVariable(-2.0, (-5.0, 5.0)),
                MockVariable(4.0, (-5.0, 5.0)),
            ]
        )

        result = CMAES(problem).optimize(
            maxiter=250,
            sigma0=0.3,
            tolx=1e-8,
            tolfun=1e-14,
            seed=5,
            disp=False,
        )

        assert result.fun < 1e-8
        assert np.linalg.norm(result.x) < 1e-3
        assert result.nfev == 1 + result.population_size * result.nit

    def test_axis_scaled_ellipsoid_convergence(self, set_test_backend) -> None:
        """Test adaptation to unequal coordinate scales."""
        problem = MockProblem(
            [
                MockVariable(4.0, (-5.0, 5.0)),
                MockVariable(-4.0, (-5.0, 5.0)),
            ],
            target_fun=lambda values: values[0] ** 2 + 100.0 * values[1] ** 2,
        )

        result = CMAES(problem).optimize(
            maxiter=300,
            population_size=10,
            sigma0=0.3,
            seed=6,
            disp=False,
        )

        assert result.fun < 1e-7
        assert np.linalg.norm(result.x) < 1e-3

    def test_different_physical_variable_scales(self, set_test_backend) -> None:
        """Test normalization across variables with different physical scales."""
        problem = MockProblem(
            [
                MockVariable(500.0, (-1000.0, 1000.0)),
                MockVariable(-0.0005, (-0.001, 0.001)),
            ],
            target_fun=lambda values: (
                (values[0] / 1000.0) ** 2 + (values[1] / 0.001) ** 2
            ),
        )

        result = CMAES(problem).optimize(
            maxiter=250,
            population_size=10,
            sigma0=0.3,
            seed=15,
            disp=False,
        )

        assert result.fun < 1e-8
        assert abs(result.x[0]) < 0.1
        assert abs(result.x[1]) < 1e-7

    def test_covariance_is_returned_in_physical_units(self, set_test_backend) -> None:
        """Test covariance scaling from normalized to physical coordinates."""

        def run(spans: np.ndarray) -> optimize.OptimizeResult:
            problem = MockProblem(
                [MockVariable(0.5 * span, (0.0, span)) for span in spans],
                target_fun=lambda values: sum(
                    (value / span) ** 2
                    for value, span in zip(values, spans, strict=True)
                ),
            )
            return CMAES(problem).optimize(
                maxiter=5,
                population_size=6,
                sigma0=0.1,
                tolx=0.0,
                tolfun=0.0,
                seed=17,
                disp=False,
            )

        unit_spans = np.ones(2)
        physical_spans = np.array([10.0, 100.0])
        unit_result = run(unit_spans)
        physical_result = run(physical_spans)

        np.testing.assert_allclose(
            physical_result.covariance,
            unit_result.covariance * np.outer(physical_spans, physical_spans),
        )

    def test_rotated_ellipsoid_convergence(self, set_test_backend) -> None:
        """Test learning of correlation between variables."""
        rotation = np.array(
            [
                [np.cos(np.pi / 4), -np.sin(np.pi / 4)],
                [np.sin(np.pi / 4), np.cos(np.pi / 4)],
            ]
        )

        def objective(values: list[float]) -> float:
            rotated = rotation @ np.asarray(values)
            return float(rotated[0] ** 2 + 100.0 * rotated[1] ** 2)

        problem = MockProblem(
            [
                MockVariable(4.0, (-5.0, 5.0)),
                MockVariable(-3.0, (-5.0, 5.0)),
            ],
            target_fun=objective,
        )

        result = CMAES(problem).optimize(
            maxiter=350,
            population_size=12,
            sigma0=0.3,
            seed=7,
            disp=False,
        )

        assert result.fun < 1e-7
        assert np.linalg.norm(result.x) < 1e-3
        correlation = result.covariance[0, 1] / np.sqrt(
            result.covariance[0, 0] * result.covariance[1, 1]
        )
        assert abs(correlation) > 0.5

    def test_reproducible_with_seed(self, set_test_backend) -> None:
        """Test deterministic results for identical seeds."""

        def run():
            problem = MockProblem(
                [
                    MockVariable(3.0, (-5.0, 5.0)),
                    MockVariable(-2.0, (-5.0, 5.0)),
                ]
            )
            return CMAES(problem).optimize(
                maxiter=20,
                population_size=8,
                seed=8,
                disp=False,
            )

        first = run()
        second = run()

        assert np.array_equal(first.x, second.x)
        assert first.fun == second.fun
        assert first.nit == second.nit
        assert first.nfev == second.nfev
        assert np.array_equal(first.covariance, second.covariance)

    def test_tolfun_stopping(self, set_test_backend) -> None:
        """Test stopping when recent objective values are flat."""
        problem = MockProblem(
            [MockVariable(0.5, (0.0, 1.0))],
            target_fun=lambda values: 42.0,
        )

        result = CMAES(problem).optimize(
            maxiter=100,
            population_size=4,
            sigma0=0.1,
            tolx=0.0,
            tolfun=0.0,
            seed=9,
            disp=False,
        )

        assert result.success is True
        assert result.nit == 10
        assert "objective values changed by at most" in result.message

    def test_tolfun_uses_generation_best_history(self, set_test_backend) -> None:
        """Test convergence after an obsolete global best leaves history."""
        evaluations = 0

        def shifted_flat_objective(values: list[float]) -> float:
            nonlocal evaluations
            evaluations += 1
            return 0.0 if evaluations <= 6 else 100.0

        problem = MockProblem(
            [MockVariable(0.5, (0.0, 1.0))],
            target_fun=shifted_flat_objective,
        )

        result = CMAES(problem).optimize(
            maxiter=12,
            population_size=4,
            sigma0=0.05,
            tolx=0.0,
            tolfun=0.0,
            seed=18,
            disp=False,
        )

        assert result.success is True
        assert result.nit == 11
        assert "objective values changed by at most" in result.message

    def test_tolx_stopping(self, set_test_backend) -> None:
        """Test stopping when the search distribution is sufficiently small."""
        problem = MockProblem([MockVariable(0.5, (0.0, 1.0))])

        result = CMAES(problem).optimize(
            maxiter=100,
            sigma0=0.01,
            tolx=1.0,
            tolfun=0.0,
            seed=10,
            disp=False,
        )

        assert result.success is True
        assert result.nit == 1
        assert "search scale fell below" in result.message

    def test_callback_stopping(self, set_test_backend) -> None:
        """Test callback data and callback-requested stopping."""
        problem = MockProblem([MockVariable(0.5, (0.0, 1.0))])
        calls: list[tuple[int, np.ndarray, float]] = []

        def callback(
            generation: int,
            best_position: np.ndarray,
            best_value: float,
        ) -> bool:
            calls.append((generation, best_position.copy(), best_value))
            return True

        result = CMAES(problem).optimize(
            maxiter=100,
            seed=11,
            disp=False,
            callback=callback,
        )

        assert result.success is True
        assert result.nit == 1
        assert result.message == "Optimization stopped by callback."
        assert calls[0][0] == 1
        assert np.array_equal(calls[0][1], result.x)
        assert calls[0][2] == result.fun

    def test_maximum_generations_result(self, set_test_backend) -> None:
        """Test result status when the generation limit is reached."""
        problem = MockProblem(
            [MockVariable(0.5, (0.0, 1.0))],
            target_fun=lambda values: values[0] ** 2,
        )

        result = CMAES(problem).optimize(
            maxiter=1,
            sigma0=0.1,
            tolx=0.0,
            tolfun=0.0,
            seed=12,
            disp=False,
        )

        assert result.success is False
        assert result.nit == 1
        assert result.message == "Maximum number of generations reached."

    def test_condition_number_safeguard(self, set_test_backend, monkeypatch) -> None:
        """Test termination when the covariance matrix becomes ill-conditioned."""
        problem = MockProblem(
            [
                MockVariable(0.5, (0.0, 1.0)),
                MockVariable(0.5, (0.0, 1.0)),
            ]
        )

        monkeypatch.setattr(
            np.linalg,
            "eigh",
            lambda covariance: (
                np.array([1e-15, 1.0]),
                np.eye(2),
            ),
        )

        result = CMAES(problem).optimize(
            maxiter=10,
            population_size=6,
            sigma0=0.1,
            tolx=0.0,
            tolfun=0.0,
            seed=16,
            disp=False,
        )

        assert result.success is False
        assert result.nit == 1
        assert result.condition_number > 1e14
        assert result.message == "Covariance matrix condition number exceeded 1e14."

    def test_covariance_psd_safeguard(self, set_test_backend, monkeypatch) -> None:
        """Test termination when covariance loses positive semidefiniteness."""
        problem = MockProblem(
            [
                MockVariable(0.5, (0.0, 1.0)),
                MockVariable(0.5, (0.0, 1.0)),
            ]
        )
        monkeypatch.setattr(
            np.linalg,
            "eigh",
            lambda covariance: (
                np.array([-1.0, 1.0]),
                np.eye(2),
            ),
        )

        result = CMAES(problem).optimize(
            maxiter=10,
            population_size=6,
            sigma0=0.1,
            seed=19,
            disp=False,
        )

        assert result.success is False
        assert result.nit == 1
        assert result.message == "Covariance matrix lost positive semidefiniteness."

    def test_large_steps_are_reflected_into_bounds(self, set_test_backend) -> None:
        """Test feasible sampling for large steps in a bounded search space."""
        evaluated: list[list[float]] = []

        def objective(values: list[float]) -> float:
            evaluated.append(values.copy())
            return sum(value**2 for value in values)

        problem = MockProblem(
            [MockVariable(0.5, (0.0, 1.0)) for _ in range(8)],
            target_fun=objective,
        )

        result = CMAES(problem).optimize(
            maxiter=1,
            population_size=10,
            sigma0=100.0,
            tolx=0.0,
            tolfun=0.0,
            seed=13,
            disp=False,
        )

        assert result.nit == 1
        assert result.nfev == 11
        assert all(
            0.0 <= value <= 1.0 for candidate in evaluated for value in candidate
        )

    def test_result_updates_problem_state(self, set_test_backend) -> None:
        """Test that the final problem state matches the best result."""
        variables = [
            MockVariable(3.0, (-5.0, 5.0)),
            MockVariable(-2.0, (-5.0, 5.0)),
        ]
        problem = MockProblem(variables)

        result = CMAES(problem).optimize(
            maxiter=100,
            seed=14,
            disp=False,
        )

        final_values = [
            be.to_numpy(variable.value).item() for variable in problem.variables
        ]
        assert np.array_equal(final_values, result.x)
        assert problem.optics_updated is True
        assert isinstance(result, optimize.OptimizeResult)
