"""Batched Ray Evaluator Module

This module provides the BatchedRayEvaluator class which accelerates
optimization by eliminating redundant ray tracing through intelligent batching
of operand evaluations.

The evaluator analyzes optimization problems to group operands that require
ray tracing with the same optic and wavelength, then executes a minimal number
of traces. Each operand extracts its value from the shared trace results.

This approach works with both NumPy and PyTorch backends and preserves
automatic differentiation (autograd) by indexing directly into the traced
tensor data.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.optimization.operand.operand import operand_registry

if TYPE_CHECKING:
    from optiland.optimization.problem import OptimizationProblem


# Operand types that call optic.trace_generic (single-ray operands)
_TRACE_GENERIC_OPERANDS = frozenset(
    {
        "real_x_intercept",
        "real_y_intercept",
        "real_z_intercept",
        "real_x_intercept_lcs",
        "real_y_intercept_lcs",
        "real_z_intercept_lcs",
        "real_L",
        "real_M",
        "real_N",
        "AOI",
    }
)

# Operand types that call optic.trace (distribution-of-rays operands)
_TRACE_OPERANDS = frozenset(
    {
        "rms_spot_size",
    }
)


def _make_generic_trace_key(optic, wavelength):
    """Create a grouping key for trace_generic operands.

    All single-ray operands sharing the same optic and wavelength can be
    batched into a single ``optic.trace_generic`` call.
    """
    return (id(optic), float(wavelength))


def _make_trace_key(optic, Hx, Hy, wavelength, num_rays, distribution):
    """Create a grouping key for trace (distribution) operands.

    ``rms_spot_size`` operands with identical parameters share a single
    ``optic.trace`` call.
    """
    return (
        id(optic),
        float(Hx),
        float(Hy),
        float(wavelength) if wavelength != "all" else "all",
        int(num_rays),
        str(distribution),
    )


class _GenericTraceJob:
    """Batched trace_generic job for single-ray operands.

    Accumulates (Hx, Hy, Px, Py) pairs and traces them all in one call.
    After execution, operands extract their values by ray index.
    """

    __slots__ = ("optic", "wavelength", "ray_params", "operand_indices")

    def __init__(self, optic, wavelength):
        self.optic = optic
        self.wavelength = wavelength
        self.ray_params: list[dict[str, float]] = []
        self.operand_indices: list[int] = []

    def add_operand(self, operand_idx: int, Hx: float, Hy: float, Px: float, Py: float):
        """Register an operand and its ray parameters."""
        ray_index = len(self.ray_params)
        self.ray_params.append({"Hx": Hx, "Hy": Hy, "Px": Px, "Py": Py})
        self.operand_indices.append(operand_idx)
        return ray_index

    def execute(self):
        """Execute the batched trace_generic call.

        Returns the optic's surface_group (with traced data stored on
        surfaces).
        """
        if not self.ray_params:
            return None

        Hx = be.array([p["Hx"] for p in self.ray_params])
        Hy = be.array([p["Hy"] for p in self.ray_params])
        Px = be.array([p["Px"] for p in self.ray_params])
        Py = be.array([p["Py"] for p in self.ray_params])

        self.optic.trace_generic(Hx, Hy, Px, Py, self.wavelength)
        return self.optic.surface_group


class _DistributionTraceJob:
    """Shared trace job for distribution-based operands (``rms_spot_size``).

    Operands with identical trace parameters share a single ``optic.trace``
    call.
    """

    __slots__ = (
        "optic",
        "Hx",
        "Hy",
        "wavelength",
        "num_rays",
        "distribution",
        "operand_indices",
    )

    def __init__(self, optic, Hx, Hy, wavelength, num_rays, distribution):
        self.optic = optic
        self.Hx = Hx
        self.Hy = Hy
        self.wavelength = wavelength
        self.num_rays = num_rays
        self.distribution = distribution
        self.operand_indices: list[int] = []

    def add_operand(self, operand_idx: int):
        """Register an operand that uses this trace."""
        self.operand_indices.append(operand_idx)

    def execute(self):
        """Execute the trace call.

        Returns the optic's surface_group.
        """
        if self.wavelength == "all":
            # Multi-wavelength rms_spot_size — let the operand handle it
            # directly since it does multiple traces internally.
            return None

        self.optic.trace(
            self.Hx,
            self.Hy,
            self.wavelength,
            self.num_rays,
            self.distribution,
        )
        return self.optic.surface_group


def _extract_value_generic(operand_type, surface_group, input_data, ray_index):
    """Extract a single operand value from a batched trace_generic result.

    This reads from the surface_group arrays at the correct ``ray_index``
    position, preserving the autograd computation graph.

    Args:
        operand_type: The type string of the operand.
        surface_group: The optic's surface group (post-trace).
        input_data: The operand's ``input_data`` dict.
        ray_index: The index of this operand's ray in the batch.

    Returns:
        The scalar value for the operand.
    """
    surface_number = input_data["surface_number"]

    if operand_type == "real_x_intercept":
        return surface_group.x[surface_number, ray_index]

    if operand_type == "real_y_intercept":
        return surface_group.y[surface_number, ray_index]

    if operand_type == "real_z_intercept":
        return surface_group.z[surface_number, ray_index]

    if operand_type == "real_x_intercept_lcs":
        intercept = surface_group.x[surface_number, ray_index]
        decenter = surface_group.surfaces[surface_number].geometry.cs.x
        return intercept - decenter

    if operand_type == "real_y_intercept_lcs":
        intercept = surface_group.y[surface_number, ray_index]
        decenter = surface_group.surfaces[surface_number].geometry.cs.y
        return intercept - decenter

    if operand_type == "real_z_intercept_lcs":
        intercept = surface_group.z[surface_number, ray_index]
        decenter = surface_group.surfaces[surface_number].geometry.cs.z
        if be.is_array_like(decenter):
            decenter = decenter.item()
        return intercept - decenter

    if operand_type == "real_L":
        return surface_group.L[surface_number, ray_index]

    if operand_type == "real_M":
        return surface_group.M[surface_number, ray_index]

    if operand_type == "real_N":
        return surface_group.N[surface_number, ray_index]

    if operand_type == "AOI":
        return _extract_aoi(surface_group, input_data, ray_index)

    raise ValueError(f"Unknown trace_generic operand type: {operand_type}")


def _extract_aoi(surface_group, input_data, ray_index):
    """Extract angle of incidence from batched trace_generic result."""
    from optiland.rays import RealRays

    surface_number = input_data["surface_number"]
    wavelength = input_data["wavelength"]

    surface = surface_group.surfaces[surface_number]
    geometry = surface.geometry

    # Incident direction cosines (from previous surface)
    L_inc = surface_group.L[surface_number - 1, ray_index]
    M_inc = surface_group.M[surface_number - 1, ray_index]
    N_inc = surface_group.N[surface_number - 1, ray_index]

    rays_at_surface = RealRays(
        x=surface_group.x[surface_number, ray_index],
        y=surface_group.y[surface_number, ray_index],
        z=surface_group.z[surface_number, ray_index],
        L=L_inc,
        M=M_inc,
        N=N_inc,
        intensity=1.0,
        wavelength=wavelength,
    )

    nx, ny, nz = geometry.surface_normal(rays=rays_at_surface)
    dot_product = be.abs(L_inc * nx + M_inc * ny + N_inc * nz)
    dot_product_clip = be.minimum(dot_product, be.array(1.0))
    angle_rad = be.arccos(dot_product_clip)
    angle_deg = be.rad2deg(angle_rad)

    if be.is_array_like(angle_deg):
        angle_deg = angle_deg.item()

    return angle_deg


def _extract_rms_spot(surface_group, input_data):
    """Extract rms_spot_size from a shared trace result.

    This replicates the calculation from ``RayOperand.rms_spot_size`` but
    reads from the already-traced surface_group data, preserving autograd.
    """
    surface_number = input_data["surface_number"]
    x = surface_group.x[surface_number, :].flatten()
    y = surface_group.y[surface_number, :].flatten()
    r2 = (x - be.mean(x)) ** 2 + (y - be.mean(y)) ** 2
    return be.sqrt(be.mean(r2))


class BatchedRayEvaluator:
    """High-performance evaluator for optimization problems using batched
    ray tracing.

    This evaluator analyses the optimization problem to identify operands that
    can share ray traces, groups them into minimal trace jobs, and provides
    ``fun_array`` / ``sum_squared`` methods that compute all operand values
    efficiently.

    The evaluator is fully compatible with both NumPy and PyTorch backends.
    When using PyTorch, the autograd computation graph is preserved because
    operand values are extracted by indexing into the traced tensor arrays
    (no detach / clone operations).

    Usage::

        problem = OptimizationProblem()
        # ... add operands and variables ...

        evaluator = BatchedRayEvaluator(problem)

        # Use instead of problem.sum_squared()
        loss = evaluator.sum_squared()

    Args:
        problem: The optimization problem to evaluate.
    """

    def __init__(self, problem: OptimizationProblem):
        self.problem = problem

        # Jobs populated by _analyze
        self._generic_jobs: list[_GenericTraceJob] = []
        self._distribution_jobs: list[_DistributionTraceJob] = []

        # Per-operand evaluation strategy:
        #   ("generic", job_idx, ray_index)
        #   ("distribution", job_idx, None)
        #   ("direct", None, None)   -- fallback to standard evaluation
        self._operand_plan: list[tuple[str, Any, Any]] = []

        self._analyze()

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _analyze(self):
        """Analyse the problem and create batching plan."""
        self._generic_jobs.clear()
        self._distribution_jobs.clear()
        self._operand_plan.clear()

        # ---- trace_generic grouping ----
        # key -> job index
        generic_key_to_idx: dict[tuple, int] = {}

        # ---- distribution trace grouping ----
        dist_key_to_idx: dict[tuple, int] = {}

        for i, operand in enumerate(self.problem.operands):
            op_type = operand.operand_type
            data = operand.input_data

            if op_type in _TRACE_GENERIC_OPERANDS:
                wl = data["wavelength"]
                optic = data["optic"]
                key = _make_generic_trace_key(optic, wl)

                if key not in generic_key_to_idx:
                    job = _GenericTraceJob(optic, wl)
                    generic_key_to_idx[key] = len(self._generic_jobs)
                    self._generic_jobs.append(job)

                job_idx = generic_key_to_idx[key]
                job = self._generic_jobs[job_idx]
                ray_idx = job.add_operand(
                    i,
                    data["Hx"],
                    data["Hy"],
                    data["Px"],
                    data["Py"],
                )
                self._operand_plan.append(("generic", job_idx, ray_idx))

            elif op_type in _TRACE_OPERANDS:
                wl = data.get("wavelength", 0.587)
                optic = data["optic"]

                # Multi-wavelength rms_spot_size must be evaluated directly
                if wl == "all":
                    self._operand_plan.append(("direct", None, None))
                    continue

                num_rays = data.get("num_rays", 100)
                dist = data.get("distribution", "hexapolar")
                key = _make_trace_key(
                    optic,
                    data["Hx"],
                    data["Hy"],
                    wl,
                    num_rays,
                    dist,
                )

                if key not in dist_key_to_idx:
                    job = _DistributionTraceJob(
                        optic,
                        data["Hx"],
                        data["Hy"],
                        wl,
                        num_rays,
                        dist,
                    )
                    dist_key_to_idx[key] = len(self._distribution_jobs)
                    self._distribution_jobs.append(job)

                job_idx = dist_key_to_idx[key]
                self._distribution_jobs[job_idx].add_operand(i)
                self._operand_plan.append(("distribution", job_idx, None))

            else:
                # Non-ray operand (paraxial, aberration, lens, etc.)
                self._operand_plan.append(("direct", None, None))

    # ------------------------------------------------------------------
    # Re-analysis (if the problem structure has changed)
    # ------------------------------------------------------------------

    def refresh(self):
        """Re-analyse the problem.

        Call this if operands or variables have been added/removed since
        the evaluator was created.
        """
        self._analyze()

    def _ensure_plan_current(self) -> None:
        """Rebuild the operand plan when the problem size has changed."""
        if len(self._operand_plan) != len(self.problem.operands):
            self._analyze()

    @staticmethod
    def _safe_execute(job):
        """Execute a trace job and return ``None`` on trace failures."""
        try:
            return job.execute()
        except Exception:
            return None

    def _evaluate_generic_jobs(self, raw_values: list[Any]) -> None:
        """Populate values for operands covered by ``trace_generic`` jobs."""
        num_operands = len(self.problem.operands)
        for job_idx, job in enumerate(self._generic_jobs):
            sg = self._safe_execute(job)
            if sg is None:
                continue

            for i in range(num_operands):
                plan_type, pj, ray_idx = self._operand_plan[i]
                if plan_type == "generic" and pj == job_idx:
                    raw_values[i] = _extract_value_generic(
                        self.problem.operands[i].operand_type,
                        sg,
                        self.problem.operands[i].input_data,
                        ray_idx,
                    )

    def _evaluate_distribution_jobs(self, raw_values: list[Any]) -> None:
        """Populate values for operands covered by distribution trace jobs."""
        num_operands = len(self.problem.operands)
        for job_idx, job in enumerate(self._distribution_jobs):
            sg = self._safe_execute(job)

            for i in range(num_operands):
                plan_type, pj, _ = self._operand_plan[i]
                if plan_type == "distribution" and pj == job_idx:
                    operand = self.problem.operands[i]
                    if sg is not None:
                        raw_values[i] = _extract_rms_spot(
                            sg,
                            operand.input_data,
                        )
                    else:
                        # Fallback (e.g. multi-wavelength)
                        metric_fn = operand_registry.get(operand.operand_type)
                        raw_values[i] = metric_fn(**operand.input_data)

    def _evaluate_direct_operands(self, raw_values: list[Any]) -> None:
        """Populate values for non-ray operands evaluated directly."""
        num_operands = len(self.problem.operands)
        for i in range(num_operands):
            if raw_values[i] is not None:
                continue
            plan_type, _, _ = self._operand_plan[i]
            if plan_type == "direct":
                operand = self.problem.operands[i]
                # Match OptimizationProblem.fun_array semantics: zero-effective-
                # weight operands are excluded and should not be evaluated.
                if operand.effective_weight() == 0.0:
                    continue
                metric_fn = operand_registry.get(operand.operand_type)
                if metric_fn is None:
                    raise ValueError(f"Unknown operand type: {operand.operand_type}")
                raw_values[i] = metric_fn(**operand.input_data)

    def _build_contribution_terms(self, raw_values: list[Any]) -> list[Any]:
        """Convert raw operand values into merit contribution terms."""
        terms = []
        for i, operand in enumerate(self.problem.operands):
            ew = operand.effective_weight()
            if ew == 0.0:
                continue
            value = raw_values[i]
            if value is None:
                raise RuntimeError(
                    f"Operand {i} ({operand.operand_type}) was not evaluated"
                )
            delta = self._compute_delta(operand, value)
            terms.append((ew * delta) ** 2)
        return terms

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def fun_array(self):
        """Compute contribution terms for each active operand.

        This is the batched equivalent of
        ``OptimizationProblem.fun_array()``.

        Trace jobs are executed one at a time and operand values are
        extracted immediately after each trace, before the next trace
        overwrites the shared ``surface_group`` state.
        """
        self._ensure_plan_current()

        num_operands = len(self.problem.operands)

        # Pre-allocate operand values — None means "not yet computed"
        raw_values: list[Any] = [None] * num_operands

        # Three-stage evaluation: generic traces, distribution traces, and
        # direct/non-ray metrics.
        self._evaluate_generic_jobs(raw_values)
        self._evaluate_distribution_jobs(raw_values)
        self._evaluate_direct_operands(raw_values)

        terms = self._build_contribution_terms(raw_values)

        if not terms:
            return be.array([0.0])
        return be.stack(terms)

    def residual_vector(self):
        """Compute the vector of weighted operand deltas (unsquared).

        Returns a 1-D array whose *i*-th element is ``weight_i * delta_i``
        for each operand. This is the residual vector **r** needed by
        least-squares algorithms such as Levenberg-Marquardt.

        Trace jobs are executed one at a time and operand values are
        extracted immediately after each trace, before the next trace
        overwrites the shared ``surface_group`` state.
        """
        self._ensure_plan_current()

        num_operands = len(self.problem.operands)

        # Use a list to hold the computed tensor node for EACH operand.
        # This avoids PyTorch in-place modification errors, allowing us
        # to safely use be.stack() at the end to fuse the graph.
        computed_residuals = [None] * num_operands

        # --- 1. Process Batched Ray Traces (Vectorized) ---
        for _job_idx, job in enumerate(self._generic_jobs):
            sg = self._safe_execute(job)

            if sg is None:
                continue

            op_indices = job.operand_indices

            # Map operand types to surface_group tensor attributes
            attr_map = {
                "real_x_intercept": "x",
                "real_y_intercept": "y",
                "real_z_intercept": "z",
                "real_L": "L",
                "real_M": "M",
                "real_N": "N",
            }

            # Vectorize extraction by grouping operands of the same type
            for op_type, attr in attr_map.items():
                mask = [
                    self.problem.operands[i].operand_type == op_type for i in op_indices
                ]
                if not any(mask):
                    continue

                matching_op_indices = [
                    op_indices[idx] for idx, is_match in enumerate(mask) if is_match
                ]
                ray_indices = [idx for idx, is_match in enumerate(mask) if is_match]
                surfs = [
                    self.problem.operands[i].input_data["surface_number"]
                    for i in matching_op_indices
                ]

                # THE VECTORIZED EXTRACTION —
                # Pulls all needed ray data in ONE clean autograd operation
                data_tensor = getattr(sg, attr)
                extracted_vals = data_tensor[surfs, ray_indices]

                # Vectorized target and weight arrays
                targets = be.array(
                    [self.problem.operands[i].target for i in matching_op_indices]
                )
                weights = be.array(
                    [
                        self.problem.operands[i].effective_weight()
                        for i in matching_op_indices
                    ]
                )

                # Compute all deltas for this group simultaneously
                deltas = weights * (extracted_vals - targets)

                # Assign the resulting scalar tensors to our list
                for local_idx, global_op_idx in enumerate(matching_op_indices):
                    computed_residuals[global_op_idx] = deltas[local_idx]

            # Handle non-vectorizable generic operands (AOI, local coords) via
            # scalar fallback
            for local_idx, global_op_idx in enumerate(op_indices):
                if computed_residuals[global_op_idx] is not None:
                    continue  # Already processed vectorially above

                operand = self.problem.operands[global_op_idx]
                val = _extract_value_generic(
                    operand.operand_type, sg, operand.input_data, local_idx
                )
                delta = self._compute_delta(operand, val)
                ew = operand.effective_weight()
                if ew == 0.0:
                    computed_residuals[global_op_idx] = be.array(0.0)
                else:
                    computed_residuals[global_op_idx] = ew * delta

        # --- 2. Process Distribution Jobs ---
        for job_idx, job in enumerate(self._distribution_jobs):
            sg = self._safe_execute(job)
            for i in range(num_operands):
                plan_type, pj, _ = self._operand_plan[i]
                if plan_type == "distribution" and pj == job_idx:
                    operand = self.problem.operands[i]
                    if sg is not None:
                        val = _extract_rms_spot(sg, operand.input_data)
                    else:
                        metric_fn = operand_registry.get(operand.operand_type)
                        val = metric_fn(**operand.input_data)

                    delta = self._compute_delta(operand, val)
                    ew = operand.effective_weight()
                    if ew == 0.0:
                        computed_residuals[i] = be.array(0.0)
                    else:
                        computed_residuals[i] = ew * delta

        # --- 3. Process Direct Jobs ---
        for i in range(num_operands):
            if computed_residuals[i] is not None:
                continue
            plan_type, _, _ = self._operand_plan[i]
            if plan_type == "direct":
                operand = self.problem.operands[i]
                ew = operand.effective_weight()
                if ew == 0.0:
                    computed_residuals[i] = be.array(0.0)
                    continue
                metric_fn = operand_registry.get(operand.operand_type)
                val = metric_fn(**operand.input_data)

                delta = self._compute_delta(operand, val)
                computed_residuals[i] = ew * delta

        # --- Final Graph Assembly ---
        for i, res in enumerate(computed_residuals):
            if res is None:
                raise RuntimeError(
                    f"Operand {i} ({self.problem.operands[i].operand_type}) "
                    "was not evaluated"
                )

        if not computed_residuals:
            return be.array([])

        return be.stack(computed_residuals)

    def sum_squared(self):
        """Compute the sum of squared weighted deltas.

        This is the batched equivalent of
        ``OptimizationProblem.sum_squared()``.
        """
        return be.sum(self.fun_array())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_delta(operand, value):
        """Compute the delta between the operand value and its target/bounds.

        This replicates the logic from ``Operand.delta()`` but accepts a
        pre-computed *value* so we avoid re-evaluating the operand.
        """
        # Some direct operands naturally return shape-(1,) tensors/arrays.
        # Standard (non-batched) inequality logic may collapse these to
        # scalars via Python max(...), so mirror that behavior here to avoid
        # mixed scalar/(1,) terms that cannot be stacked together.
        if be.size(value) == 1:
            value = be.ravel(value)[0]

        if operand.target is not None:
            return value - operand.target

        if operand.min_val is not None or operand.max_val is not None:
            lower_penalty = (
                be.maximum(be.array(0.0), be.array(operand.min_val) - value)
                if operand.min_val is not None
                else be.array(0.0)
            )
            upper_penalty = (
                be.maximum(be.array(0.0), value - be.array(operand.max_val))
                if operand.max_val is not None
                else be.array(0.0)
            )
            return lower_penalty + upper_penalty

        raise ValueError(f"Operand '{operand.operand_type}' has no target or bounds")
