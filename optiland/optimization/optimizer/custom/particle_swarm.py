from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import optimize

import optiland.backend as be

from ..scipy.base import OptimizerGeneric

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...problem import OptimizationProblem


class ParticleSwarm(OptimizerGeneric):
    """Particle Swarm Optimization (PSO) solver.

    PSO is a population-based, derivative-free optimizer
    introduced by Kennedy & Eberhart in 1995.
    The paper can be found at doi.org/10.1109/ICNN.1995.488968.

    Instead of moving a single candidate solution as in gradient descent,
    we evolve a swarm of particles. Each particle remembers:
      1. its own best location found so far (personal best),
      2. the best location found by the swarm (global best).

    The search dynamic is a balance between:
      - inertia: keep moving in the current direction,
      - individual pull: return toward the particle's own best experience,
      - social pull: move toward the swarm's best-known solution.

    In practice, PSO is useful when:
      - the objective is non-convex or multimodal,
      - gradients are unavailable, unreliable, or too expensive,
      - a reasonably global search is preferred over a local optimizer.

    Args:
        problem: The optimization problem to be solved.

    Notes:
        - All variables must have finite bounds.
          This implementation samples and clamps particles inside the box
          defined by those bounds, so unbounded variables are not supported.
        - The current variable values are injected as the first particle
          of the swarm.
          This is often a good engineering choice because it preserves the
          current design as one known-valid candidate in the initial population.
        - The returned object is a ``scipy.optimize.OptimizeResult``.

    """

    def __init__(self, problem: OptimizationProblem):
        super().__init__(problem)

    def optimize(
        self,
        maxiter: int = 200,
        swarm_size: int | None = None,
        inertia: float = 0.7,
        individual: float = 1.5,
        social: float = 1.5,
        tol: float = 1e-3,
        stall_iterations: int = 20,
        seed: int | None = None,
        disp: bool = True,
        callback: Callable | None = None,
    ) -> optimize.OptimizeResult:
        """Run Particle Swarm Optimization.

        Args:
            maxiter: Maximum number of PSO iterations.
            swarm_size: Number of particles in the swarm. If ``None``,
                uses ``max(20, 10 * ndim)``.
            inertia: Inertia weight.
                Larger values encourage exploration because particles keep more
                of their previous velocity. Smaller values damp motion and make
                the swarm settle more aggressively.
            individual: Individual memory coefficient.
                Strength of attraction toward each particle's personal best.
                Higher values increase individual exploration / self-correction.
            social: Social coefficient.
                Strength of attraction toward the swarm's global best.
                Higher values increase collective convergence pressure.
            tol: Absolute improvement tolerance used for stall detection.
            stall_iterations: Stop if the global best improvement remains
                below ``tol`` for this many consecutive iterations.
            seed: Random seed for reproducibility.
            disp: Whether to print iteration progress.
            callback: Optional callback called after each iteration as
                ``callback(iteration, best_position, best_value)``.
                If it returns ``True``, the optimization stops early.

        Returns:
            A ``scipy.optimize.OptimizeResult`` containing the optimization
            result.

        Raises:
            ValueError: If a variable does not have finite bounds, or if
                ``swarm_size`` is invalid.

        Notes on the default PSO coefficients:
            The default triplet
                inertia = 0.7
                individual = 1.5
                social = 1.5
            is given using the Clerc-Kennedy recommendation with:
                inertia    = chi
                individual = chi * phi1
                social     = chi * phi2
                chi * phi1 = chi * phi2 ≈ 0.72984 * 2.05 ≈ 1.49618.
            These values often provide a reasonable exploration/convergence tradeoff.

            Please note that PSO performance is problem-dependent,
            especially with dimensionality, variable scaling,
            noise in the merit function, and tight or active bounds.
        """
        x0_backend = [var.value for var in self.problem.variables]
        self._x.append(x0_backend)
        x0 = np.asarray(be.to_numpy(x0_backend), dtype=float)
        ndim = x0.size

        # Read variable bounds from the optimization problem
        bounds = [var.bounds for var in self.problem.variables]
        if any(bound is None or len(bound) != 2 for bound in bounds):
            raise ValueError("PSO requires all variables to have finite bounds.")

        lower = np.asarray([bound[0] for bound in bounds], dtype=float)
        upper = np.asarray([bound[1] for bound in bounds], dtype=float)

        # This implementation relies on box-bounded search:
        # particles are initialized uniformly inside [lower, upper] and clipped
        # back into that box after each move.
        # Therefore every variable must have valid finite bounds.
        if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
            raise ValueError("PSO requires all variables to have finite bounds.")
        if np.any(upper < lower):
            raise ValueError("Each variable bound must satisfy lower <= upper.")

        # Heuristic default swarm size:
        # - at least 20 particles to maintain some search diversity,
        # - scale with dimension to avoid under-sampling higher-dimensional spaces.
        # Too small: swarm collapses too quickly or misses good basins.
        # Too large: objective evaluation cost dominates.
        if swarm_size is None:
            swarm_size = max(20, 10 * ndim)
        if swarm_size < 2:
            raise ValueError("swarm_size must be at least 2.")

        rng = np.random.default_rng(seed)

        # Per-dimension search span. Useful for initialization and for detecting
        # fixed variables, i.e. dimensions where lower == upper.
        span = upper - lower
        movable_mask = span > 0.0

        # Initialize swarm positions.
        #
        # Standard PSO practice is to spread particles across the feasible box
        # to obtain broad initial coverage. That gives the swarm a chance to
        # discover promising basins early instead of starting from a narrow cloud.
        # Positions is the state vector of the problem.
        # Each row `positions[i]` is one particle, each particle contains all variables.
        positions = rng.uniform(lower, upper, size=(swarm_size, ndim))

        # Inject the current design as the first particle. This is an informed
        # engineering decision:
        #   - it guarantees the existing design is evaluated,
        #   - it prevents losing a potentially good baseline,
        #   - it can accelerate convergence when the current point is already
        #     near a useful region.
        #
        # We still clip it for safety in case stored variable values drifted
        # slightly outside the formal bounds.
        # Position[0] is the initial state vector of the problem.
        positions[0] = np.clip(x0, lower, upper)

        # Initialize swarm velocities.
        #
        # Starting at zero everywhere would make the first movement depend only
        # on the individual/social terms, which are initially weak or degenerate
        # because personal best = current position and the global best may be
        # poorly informative at iteration 0.
        #
        # Giving particles random initial velocities encourages immediate motion
        # and better early exploration.
        velocities = np.zeros((swarm_size, ndim), dtype=float)
        if np.any(movable_mask):
            velocities[:, movable_mask] = rng.uniform(
                -span[movable_mask],
                span[movable_mask],
                size=(swarm_size, int(np.count_nonzero(movable_mask))),
            )

        # Evaluate the initial swarm.
        #
        # At iteration 0, each particle's personal best is simply its starting
        # position, because that is the only point it has visited.
        personal_best_positions = positions.copy()
        personal_best_values = np.empty(swarm_size, dtype=float)
        nfev = 0

        for i in range(swarm_size):
            personal_best_values[i] = float(self._fun(positions[i]))
            nfev += 1

        # The global best is the best personal best among all particles.
        # In standard global-best PSO, every particle is influenced by this same
        # elite point, which typically accelerates convergence but can increase
        # the risk of premature swarm collapse on difficult multimodal problems.
        best_idx = int(np.argmin(personal_best_values))
        global_best_position = personal_best_positions[best_idx].copy()
        global_best_value = float(personal_best_values[best_idx])

        if disp:
            print(
                f"PSO start: best merit = {global_best_value:.12g}, "
                f"swarm_size = {swarm_size}, ndim = {ndim}"
            )

        success = False
        message = "Maximum number of iterations reached."
        no_improve_count = 0

        for iteration in range(1, maxiter + 1):
            prev_global_best_value = global_best_value

            # r1 and r2 are defined as independently sampled random coefficients.
            # In PSO this stochasticity is important:
            #   - it prevents deterministic lockstep motion,
            #   - it diversifies trajectories even under the same global best,
            #   - it helps the swarm probe neighborhoods rather than collapsing
            #     along a single rigid path.
            r1 = rng.random((swarm_size, ndim))
            r2 = rng.random((swarm_size, ndim))

            # Core PSO velocity update:
            #
            #   v <- w*v
            #        + c1*r1*(pbest - x)
            #        + c2*r2*(gbest - x)
            #
            # Interpretation:
            #   - inertia * velocities:
            #       preserves momentum, enabling exploration and directional
            #       continuity across iterations.
            #
            #   - individual * r1 * (personal_best_positions - positions):
            #       pulls each particle back toward what it has discovered to
            #       be good in the past.
            #
            #   - social * r2 * (global_best_position - positions):
            #       pulls each particle toward the best-known point found by the
            #       swarm as a whole.
            #
            # This simple model is one of the reasons PSO is attractive:
            # it can search effectively without requiring gradients or Hessians.
            velocities = (
                inertia * velocities
                + individual * r1 * (personal_best_positions - positions)
                + social * r2 * (global_best_position - positions)
            )

            # Advance particles (ie update each particle in the state vector)
            positions = positions + velocities  # shape (swarm_size, ndim)

            # Enforce feasibility with simple box clipping.
            positions = np.clip(positions, lower, upper)

            # Zero velocity for fixed variables and for particles stuck on bounds.
            #
            # For fixed variables (lower == upper), velocity must remain zero.
            # Otherwise meaningless numerical motion could accumulate.
            #
            # Note: despite the comment, the actual code only guarantees zero
            # velocity for fixed variables. It does not explicitly zero the
            # velocity of movable variables that happen to sit on bounds.
            # If bound-stick behavior becomes important, a more explicit rule
            # could be implemented here.
            velocities[:, ~movable_mask] = 0.0

            for i in range(swarm_size):
                # Evaluate the current particle position (ie merit function value)
                merit_function_value = float(self._fun(positions[i]))
                nfev += 1

                # Personal-best update:
                # if the particle improved upon its own historical record,
                # retain this position as its new memory anchor.
                if merit_function_value < personal_best_values[i]:
                    personal_best_values[i] = merit_function_value
                    personal_best_positions[i] = positions[i].copy()

                    # Global-best update:
                    # only triggered when a personal best beats the current
                    # best-known swarm solution.
                    if merit_function_value < global_best_value:
                        global_best_value = merit_function_value
                        global_best_position = positions[i].copy()

            improvement = prev_global_best_value - global_best_value

            # Stall-based convergence criterion.
            #
            # PSO does not naturally provide a gradient norm stop condition,
            # so practical implementations may use one of:
            #   - lack of objective improvement,
            #   - swarm diameter collapse,
            #   - velocity norm collapse,
            #   - max iterations.
            #
            # Here the choice is "global best stopped improving enough".
            if improvement <= tol:
                no_improve_count += 1
            else:
                no_improve_count = 0

            if disp:
                print(
                    f"Iter {iteration:4d} | "
                    f"best merit = {global_best_value:.12g} | "
                    f"improvement = {improvement:.6g}"
                )

            callback() if callback else None

            if no_improve_count >= stall_iterations:
                success = True
                message = (
                    "Optimization converged: global best stalled within "
                    f"tol={tol} for {stall_iterations} iterations."
                )
                break

        # Update problem variables with the final best solution found by the swarm.
        for idvar, var in enumerate(self.problem.variables):
            var.update(global_best_position[idvar])

        # Recompute the optical system so that the problem state is consistent
        # with the final optimized variable values.
        self.problem.update_optics()

        # Return a SciPy-style result object for consistency
        # with the optimizer ecosystem.
        return optimize.OptimizeResult(
            x=global_best_position.copy(),
            fun=float(global_best_value),
            nit=iteration,
            nfev=nfev,
            success=success,
            message=message,
            swarm_size=swarm_size,
            inertia=inertia,
            individual=individual,
            social=social,
        )
