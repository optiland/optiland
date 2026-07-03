from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np
from scipy import optimize

import optiland.backend as be

from ..scipy.base import OptimizerGeneric

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...problem import OptimizationProblem


class CMAES(OptimizerGeneric):
    """Bounded covariance matrix adaptation evolution strategy optimizer.

    The optimizer runs in a normalized search space and implements weighted
    recombination, cumulative step-size adaptation, and rank-one plus rank-mu
    covariance updates. Fixed variables are excluded from the internal search
    space.

    Args:
        problem: The optimization problem to solve.
    """

    def __init__(self, problem: OptimizationProblem) -> None:
        super().__init__(problem)

    def optimize(
        self,
        maxiter: int = 500,
        population_size: int | None = None,
        sigma0: float = 0.3,
        tolx: float = 1e-8,
        tolfun: float = 1e-12,
        seed: int | None = None,
        disp: bool = True,
        callback: Callable[[int, np.ndarray, float], bool | None] | None = None,
    ) -> optimize.OptimizeResult:
        """Run CMA-ES.

        Args:
            maxiter: Maximum number of generations.
            population_size: Number of candidates per generation. If ``None``,
                uses ``4 + floor(3 * log(n))`` for ``n`` movable variables.
            sigma0: Initial global step size in normalized coordinates.
            tolx: Stop when the largest coordinate search scale falls below
                this value.
            tolfun: Stop when recent best objective values vary by no more than
                this value.
            seed: Random seed for reproducibility.
            disp: Whether to print generation progress.
            callback: Optional callback called as
                ``callback(generation, best_position, best_value)``. Returning
                ``True`` stops the optimization.

        Returns:
            A SciPy-style optimization result.

        Raises:
            ValueError: If an algorithm parameter, bound, or current variable
                value is invalid.
        """
        x0_backend = [variable.value for variable in self.problem.variables]
        x0 = np.asarray(be.to_numpy(x0_backend), dtype=float)
        lower, upper = self._validate_bounds(x0)
        movable = upper > lower
        dimension = int(np.count_nonzero(movable))

        if population_size is None and dimension > 0:
            population_size = 4 + int(np.floor(3.0 * np.log(dimension)))
        elif population_size is None:
            population_size = 0

        self._validate_parameters(
            maxiter=maxiter,
            population_size=population_size,
            sigma0=sigma0,
            tolx=tolx,
            tolfun=tolfun,
            has_movable_variables=dimension > 0,
        )

        self._x.append(x0_backend)
        best_position = x0.copy()
        best_value = float(self._fun(best_position))
        nfev = 1

        if dimension == 0:
            self._set_problem_state(best_position)
            return optimize.OptimizeResult(
                x=best_position,
                fun=best_value,
                nit=0,
                nfev=nfev,
                success=True,
                message="No movable variables to optimize.",
                population_size=0,
                sigma=0.0,
                mean=best_position.copy(),
                covariance=np.empty((0, 0), dtype=float),
                condition_number=1.0,
            )

        movable_lower = lower[movable]
        movable_span = upper[movable] - movable_lower
        mean = (x0[movable] - movable_lower) / movable_span
        sigma = float(sigma0)
        covariance = np.eye(dimension)
        eigenvectors = np.eye(dimension)
        deviations = np.ones(dimension)
        path_sigma = np.zeros(dimension)
        path_covariance = np.zeros(dimension)
        condition_number = 1.0

        mu = population_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)

        c_sigma = (mu_eff + 2.0) / (dimension + mu_eff + 5.0)
        d_sigma = (
            1.0
            + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (dimension + 1.0)) - 1.0)
            + c_sigma
        )
        c_covariance = (4.0 + mu_eff / dimension) / (
            dimension + 4.0 + 2.0 * mu_eff / dimension
        )
        c1 = 2.0 / ((dimension + 1.3) ** 2 + mu_eff)
        c_mu = min(
            1.0 - c1,
            2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((dimension + 2.0) ** 2 + mu_eff),
        )
        expected_norm = np.sqrt(dimension) * (
            1.0 - 1.0 / (4.0 * dimension) + 1.0 / (21.0 * dimension**2)
        )
        history_size = max(
            10,
            int(np.ceil(30.0 * dimension / population_size)),
        )
        best_history: deque[float] = deque(maxlen=history_size)
        rng = np.random.default_rng(seed)

        success = False
        message = "Maximum number of generations reached."
        completed_generations = 0

        if disp:
            print(
                f"CMA-ES start: best merit = {best_value:.12g}, "
                f"population_size = {population_size}, ndim = {dimension}"
            )

        for generation in range(1, maxiter + 1):
            sampled = self._sample_population(
                mean=mean,
                sigma=sigma,
                eigenvectors=eigenvectors,
                deviations=deviations,
                population_size=population_size,
                rng=rng,
            )
            candidates, steps = sampled
            fitness = np.empty(population_size, dtype=float)
            for index, candidate in enumerate(candidates):
                full_candidate = self._to_physical(
                    candidate,
                    x0,
                    movable,
                    movable_lower,
                    movable_span,
                )
                fitness[index] = float(self._fun(full_candidate))
            nfev += population_size

            order = np.argsort(fitness)
            candidates = candidates[order]
            steps = steps[order]
            fitness = fitness[order]

            if fitness[0] < best_value:
                best_value = float(fitness[0])
                best_position = self._to_physical(
                    candidates[0],
                    x0,
                    movable,
                    movable_lower,
                    movable_span,
                )

            old_mean = mean.copy()
            selected_candidates = candidates[:mu]
            selected_steps = steps[:mu]
            mean = np.sum(weights[:, np.newaxis] * selected_candidates, axis=0)
            weighted_step = np.sum(
                weights[:, np.newaxis] * selected_steps,
                axis=0,
            )

            inverse_sqrt_step = eigenvectors @ (
                (eigenvectors.T @ weighted_step) / deviations
            )
            path_sigma = (1.0 - c_sigma) * path_sigma + np.sqrt(
                c_sigma * (2.0 - c_sigma) * mu_eff
            ) * inverse_sqrt_step

            path_sigma_norm = np.linalg.norm(path_sigma)
            normalized_path = path_sigma_norm / np.sqrt(
                1.0 - (1.0 - c_sigma) ** (2 * generation)
            )
            h_sigma = float(
                normalized_path / expected_norm < 1.4 + 2.0 / (dimension + 1.0)
            )

            path_covariance = (
                1.0 - c_covariance
            ) * path_covariance + h_sigma * np.sqrt(
                c_covariance * (2.0 - c_covariance) * mu_eff
            ) * weighted_step

            rank_mu = np.zeros_like(covariance)
            for weight, step in zip(weights, selected_steps, strict=True):
                rank_mu += weight * np.outer(step, step)

            covariance = (
                (
                    1.0
                    - c1
                    - c_mu
                    + c1 * (1.0 - h_sigma) * c_covariance * (2.0 - c_covariance)
                )
                * covariance
                + c1 * np.outer(path_covariance, path_covariance)
                + c_mu * rank_mu
            )
            covariance = 0.5 * (covariance + covariance.T)

            sigma *= np.exp(
                (c_sigma / d_sigma) * (path_sigma_norm / expected_norm - 1.0)
            )

            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            largest_eigenvalue = float(np.max(eigenvalues))
            negative_tolerance = -1e-12 * max(1.0, largest_eigenvalue)
            if float(np.min(eigenvalues)) < negative_tolerance:
                message = "Covariance matrix lost positive semidefiniteness."
                completed_generations = generation
                break

            eigenvalues = np.maximum(eigenvalues, np.finfo(float).eps)
            deviations = np.sqrt(eigenvalues)
            condition_number = float(np.max(eigenvalues) / np.min(eigenvalues))
            completed_generations = generation
            best_history.append(float(fitness[0]))

            if disp:
                mean_shift = np.linalg.norm(mean - old_mean)
                print(
                    f"Generation {generation:4d} | "
                    f"best merit = {best_value:.12g} | "
                    f"sigma = {sigma:.6g} | mean shift = {mean_shift:.6g}"
                )

            if callback is not None and callback(
                generation,
                best_position.copy(),
                best_value,
            ):
                success = True
                message = "Optimization stopped by callback."
                break

            coordinate_scale = sigma * float(np.max(np.sqrt(np.diag(covariance))))
            if coordinate_scale <= tolx:
                success = True
                message = (
                    f"Optimization converged: search scale fell below tolx={tolx}."
                )
                break

            current_fitness_range = float(np.max(fitness) - np.min(fitness))
            historic_fitness_range = max(best_history) - min(best_history)
            if len(best_history) == history_size and (
                current_fitness_range <= tolfun and historic_fitness_range <= tolfun
            ):
                success = True
                message = (
                    "Optimization converged: recent objective values changed "
                    f"by at most tolfun={tolfun}."
                )
                break

            if condition_number > 1e14:
                message = "Covariance matrix condition number exceeded 1e14."
                break

        self._set_problem_state(best_position)
        physical_mean = self._to_physical(
            mean,
            x0,
            movable,
            movable_lower,
            movable_span,
        )
        physical_covariance = covariance * np.outer(movable_span, movable_span)
        return optimize.OptimizeResult(
            x=best_position.copy(),
            fun=best_value,
            nit=completed_generations,
            nfev=nfev,
            success=success,
            message=message,
            population_size=population_size,
            sigma=sigma,
            mean=physical_mean,
            covariance=physical_covariance,
            condition_number=condition_number,
        )

    def _validate_bounds(self, x0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Validate and return finite variable bounds."""
        bounds = [variable.bounds for variable in self.problem.variables]
        if any(bound is None or len(bound) != 2 for bound in bounds):
            raise ValueError("CMAES requires all variables to have finite bounds.")

        lower = np.asarray([bound[0] for bound in bounds], dtype=float)
        upper = np.asarray([bound[1] for bound in bounds], dtype=float)
        if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
            raise ValueError("CMAES requires all variables to have finite bounds.")
        if np.any(upper < lower):
            raise ValueError("Each variable bound must satisfy lower <= upper.")
        if np.any(~np.isfinite(x0)) or np.any(x0 < lower) or np.any(x0 > upper):
            raise ValueError("Current variable values must lie within their bounds.")
        return lower, upper

    @staticmethod
    def _validate_parameters(
        *,
        maxiter: int,
        population_size: int,
        sigma0: float,
        tolx: float,
        tolfun: float,
        has_movable_variables: bool,
    ) -> None:
        """Validate CMA-ES configuration."""
        if maxiter < 1:
            raise ValueError("maxiter must be at least 1.")
        if population_size != 0 and population_size < 2:
            raise ValueError("population_size must be at least 2.")
        if has_movable_variables and population_size == 0:
            raise ValueError("population_size must be at least 2.")
        if not np.isfinite(sigma0) or sigma0 <= 0.0:
            raise ValueError("sigma0 must be positive.")
        if not np.isfinite(tolx) or tolx < 0.0:
            raise ValueError("tolx must be non-negative.")
        if not np.isfinite(tolfun) or tolfun < 0.0:
            raise ValueError("tolfun must be non-negative.")

    @staticmethod
    def _sample_population(
        *,
        mean: np.ndarray,
        sigma: float,
        eigenvectors: np.ndarray,
        deviations: np.ndarray,
        population_size: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample a population and reflect candidates into the unit box."""
        dimension = mean.size
        candidates = np.empty((population_size, dimension), dtype=float)
        steps = np.empty_like(candidates)

        for candidate_index in range(population_size):
            normal = rng.standard_normal(dimension)
            step = eigenvectors @ (deviations * normal)
            candidate = mean + sigma * step
            candidate = np.mod(candidate, 2.0)
            candidate = np.where(candidate <= 1.0, candidate, 2.0 - candidate)
            candidates[candidate_index] = candidate
            steps[candidate_index] = (candidate - mean) / sigma

        return candidates, steps

    @staticmethod
    def _to_physical(
        normalized: np.ndarray,
        template: np.ndarray,
        movable: np.ndarray,
        lower: np.ndarray,
        span: np.ndarray,
    ) -> np.ndarray:
        """Expand normalized movable coordinates into a physical vector."""
        physical = template.copy()
        physical[movable] = lower + normalized * span
        return physical

    def _set_problem_state(self, position: np.ndarray) -> None:
        """Update the problem to the supplied complete variable vector."""
        for index, variable in enumerate(self.problem.variables):
            variable.update(position[index])
        self.problem.update_optics()
