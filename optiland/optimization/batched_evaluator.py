"""Batched Ray Evaluator Module

This module provides the BatchedRayEvaluator class which accelerates
optimization by eliminating redundant ray tracing through intelligent batching
of operand evaluations.

The evaluator analyzes optimization problems to group every operand that
requires real rays with the same optic and wavelength into one super-bundle:
single-ray operands and the pupil samples of all ``rms_spot_size`` operands are
traced together in one ``optic.trace_generic`` call, and each operand reads its
slice of the shared result. ``wavelength="all"`` spot operands are split into
one sub-job per wavelength, merged into the matching bundles. Each evaluation
runs inside one evaluation epoch, so per-design work is done once.

This approach works with both NumPy and PyTorch backends and preserves
automatic differentiation (autograd) by indexing directly into the traced
tensor data.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland import _epoch
from optiland.distribution import create_distribution
from optiland.optimization.operand.operand import operand_registry
from optiland.optimization.operand.ray import _angle_of_incidence

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

# Single-ray operands that read one per-surface ray attribute directly. With a
# scalar target, their residuals are computed in one vectorized step.
_GENERIC_ATTRS = {
    "real_x_intercept": "x",
    "real_y_intercept": "y",
    "real_z_intercept": "z",
    "real_L": "L",
    "real_M": "M",
    "real_N": "N",
}

_NAN_POLICIES = ("propagate", "omit", "raise")

# Per-surface ray attributes read by the single-ray and spot operands
_RAY_ATTRS = ("x", "y", "z", "L", "M", "N")
_SPOT_ATTRS = ("x", "y")


def _make_generic_trace_key(optic, wavelength):
    """Create a grouping key for a super-bundle.

    All ray operands sharing the same optic and wavelength are batched into a
    single ``optic.trace_generic`` call.
    """
    return (id(optic), float(wavelength))


def _make_trace_key(optic, Hx, Hy, wavelength, num_rays, distribution):
    """Create a grouping key for a pupil sample (distribution) segment.

    ``rms_spot_size`` operands with identical parameters share one segment of
    the bundle.
    """
    return (
        id(optic),
        float(Hx),
        float(Hy),
        float(wavelength) if wavelength != "all" else "all",
        int(num_rays),
        str(distribution),
    )


class _PupilSegment:
    """Pupil samples of one distribution at one field point, inside a bundle."""

    __slots__ = ("Hx", "Hy", "num_rays", "distribution")

    def __init__(self, Hx, Hy, num_rays, distribution):
        self.Hx = Hx
        self.Hy = Hy
        self.num_rays = num_rays
        self.distribution = distribution

    def pupil(self, samples: dict) -> tuple[Any, Any]:
        """Pupil coordinates, generated as ``optic.trace`` would generate them.

        A distribution named by string is generated once per bundle execution
        and shared by every segment that names it with the same ``num_rays``.
        """
        distribution = self.distribution
        if not isinstance(distribution, str):
            return distribution.x, distribution.y
        key = (distribution, self.num_rays)
        if key not in samples:
            generated = create_distribution(distribution)
            generated.generate_points(self.num_rays)
            samples[key] = (generated.x, generated.y)
        return samples[key]


class _BundleJob:
    """Super-bundle of all rays traced for one (optic, wavelength).

    The bundle is made of parts traced together in one ``trace_generic`` call.
    Part 0 holds the single rays of the single-ray operands, in registration
    order, so a single ray's index in the bundle is its index in
    ``ray_params``. Part ``k >= 1`` holds the pupil samples of
    ``segments[k - 1]``.
    """

    __slots__ = (
        "optic",
        "wavelength",
        "ray_params",
        "operand_indices",
        "segments",
        "offsets",
        "_segment_index",
    )

    def __init__(self, optic, wavelength):
        self.optic = optic
        self.wavelength = wavelength
        self.ray_params: list[tuple[float, float, float, float]] = []
        self.operand_indices: list[int] = []
        self.segments: list[_PupilSegment] = []
        # part -> (first ray, end ray) of the last execution
        self.offsets: dict[int, tuple[int, int]] = {}
        self._segment_index: dict[tuple, int] = {}

    def add_operand(self, operand_idx: int, Hx: float, Hy: float, Px: float, Py: float):
        """Register a single-ray operand; return its ray index in the bundle."""
        ray_index = len(self.ray_params)
        self.ray_params.append((Hx, Hy, Px, Py))
        self.operand_indices.append(operand_idx)
        return ray_index

    def add_segment(self, key: tuple, Hx, Hy, num_rays, distribution) -> int:
        """Register a pupil sample segment (shared by key); return its index."""
        if key not in self._segment_index:
            self._segment_index[key] = len(self.segments)
            self.segments.append(_PupilSegment(Hx, Hy, num_rays, distribution))
        return self._segment_index[key]

    @property
    def parts(self) -> list[int]:
        """The parts of the bundle, in ray order."""
        first = [0] if self.ray_params else []
        return first + list(range(1, len(self.segments) + 1))

    def execute(self, parts: list[int] | None = None):
        """Trace the given parts (all by default) in one ``trace_generic`` call.

        Records the ray range of each traced part in ``offsets``.

        Returns the optic's surface_group (with traced data stored on
        surfaces).
        """
        if parts is None:
            parts = self.parts
        if not parts:
            return None

        columns: list[list[Any]] = [[], [], [], []]
        samples: dict = {}
        offsets: dict[int, tuple[int, int]] = {}
        start = 0
        for part in parts:
            if part == 0:
                rays = zip(*self.ray_params, strict=True)
                for column, values in zip(columns, rays, strict=True):
                    column.append(be.array(list(values)))
                count = len(self.ray_params)
            else:
                segment = self.segments[part - 1]
                Px, Py = segment.pupil(samples)
                count = be.size(Px)
                columns[0].append(be.full((count,), float(segment.Hx)))
                columns[1].append(be.full((count,), float(segment.Hy)))
                columns[2].append(Px)
                columns[3].append(Py)
            offsets[part] = (start, start + count)
            start += count

        Hx, Hy, Px, Py = (
            column[0] if len(column) == 1 else be.concatenate(column)
            for column in columns
        )
        self.offsets = offsets
        self.optic.trace_generic(Hx, Hy, Px, Py, self.wavelength)
        return self.optic.surfaces


class _BundlePart:
    """One part of a bundle traced on its own, used to isolate a failure."""

    __slots__ = ("job", "part")

    def __init__(self, job: _BundleJob, part: int):
        self.job = job
        self.part = part

    def execute(self):
        """Trace only this part of the bundle."""
        return self.job.execute([self.part])


class _TraceView:
    """Read-once view of a traced surface group.

    ``SurfaceGroup.x`` and friends stack every surface on each access, so each
    attribute is read once per trace and shared by all operands reading it.
    Attributes are read lazily: those needed after the next trace overwrites
    the surface group must be listed in ``prefetch``.
    """

    __slots__ = ("_surface_group", "_cache")

    def __init__(self, surface_group, prefetch: tuple[str, ...] = ()):
        self._surface_group = surface_group
        self._cache: dict[str, Any] = {
            name: getattr(surface_group, name) for name in prefetch
        }

    def __getattr__(self, name: str):
        cache = self._cache
        if name not in cache:
            cache[name] = getattr(self._surface_group, name)
        return cache[name]


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
    return _angle_of_incidence(L_inc, M_inc, N_inc, nx, ny, nz)


def _spot_surface(input_data):
    """The surface an ``rms_spot_size`` operand is measured on."""
    try:
        return input_data["surface_number"]
    except KeyError:
        raise KeyError(
            "The 'rms_spot_size' operand requires 'surface_number' in its "
            "input_data, naming the surface the spot is measured on. Use -1 "
            "for the image surface, e.g. input_data={'optic': lens, 'Hx': 0, "
            "'Hy': 0, 'num_rings': 6, 'wavelength': 0.55, "
            "'surface_number': -1}."
        ) from None


def _extract_rms_spot(xs, ys, input_data, primary_index=None):
    """RMS spot size from sliced bundle data.

    This replicates the calculation of ``RayOperand.rms_spot_size`` exactly,
    including its ``nan_policy``, but reads from already-traced data,
    preserving autograd.

    Args:
        xs: Per-wavelength x intercepts on the spot surface (one entry unless
            the operand uses ``wavelength="all"``).
        ys: Per-wavelength y intercepts, matching ``xs``.
        input_data: The operand's ``input_data`` dict.
        primary_index: Index of the primary wavelength in ``xs`` for a
            ``wavelength="all"`` operand, else None.

    Returns:
        The RMS spot size.
    """
    nan_policy = input_data.get("nan_policy", "propagate")
    if nan_policy not in _NAN_POLICIES:
        raise ValueError(
            f"Invalid nan_policy '{nan_policy}'. Must be one of {_NAN_POLICIES}."
        )

    has_nan = any(be.any(be.isnan(a)) for a in (*xs, *ys))
    if primary_index is None:
        x, y = xs[0], ys[0]
        r2 = (x - be.nanmean(x)) ** 2 + (y - be.nanmean(y)) ** 2
        rms = be.sqrt(be.nanmean(r2))
    else:
        mean_x = be.nanmean(xs[primary_index])
        mean_y = be.nanmean(ys[primary_index])
        r2 = [
            (x - mean_x) ** 2 + (y - mean_y) ** 2 for x, y in zip(xs, ys, strict=True)
        ]
        rms = be.sqrt(be.nanmean(be.concatenate(r2)))

    if has_nan:
        if nan_policy == "raise":
            raise ValueError(
                "rms_spot_size encountered a NaN ray intersection on "
                f"surface {_spot_surface(input_data)}. This typically "
                "indicates total internal reflection or a vignetted ray. Use "
                "nan_policy='omit' to compute the RMS from valid rays "
                "only, or leave nan_policy='propagate' (default) to "
                "return NaN."
            )
        if nan_policy == "propagate":
            return rms * be.nan
    return rms


class BatchedRayEvaluator:
    """High-performance evaluator for optimization problems using batched
    ray tracing.

    This evaluator analyses the optimization problem to identify operands that
    can share ray traces, groups them into one super-bundle per optic and
    wavelength, and provides ``fun_array`` / ``sum_squared`` methods that
    compute all operand values efficiently.

    A bundle whose trace fails is retraced part by part, so a failure marks
    only the operands of the failing part as non-finite (NaN); it never raises.

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

        # Jobs populated by _analyze: one super-bundle per (optic, wavelength)
        self._bundles: list[_BundleJob] = []

        # Per-operand evaluation strategy:
        #   ("generic", bundle_idx, ray_index)
        #   ("distribution", primary_index, reads) -- reads: [(bundle, part)];
        #       primary_index is None unless wavelength="all"
        #   ("direct", None, None)   -- standard evaluation
        self._operand_plan: list[tuple[str, Any, Any]] = []

        # Per bundle: {attr: (operand indices, surfaces, ray indices)} of the
        # single-ray operands read straight from one ray attribute.
        self._attr_groups: list[dict[str, tuple[list, list, list]]] = []

        # Optics referenced by the operands (epoch debug guard), optics with
        # wavelength="all" operands, and the wavelength sets they expanded to.
        self._optics: tuple = ()
        self._all_optics: tuple = ()
        self._wavelength_signature: tuple = ()

        self._analyze()

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _analyze(self):
        """Analyse the problem and create batching plan."""
        self._bundles = []
        self._operand_plan = []
        bundle_index: dict[tuple, int] = {}
        optics: dict[int, Any] = {}
        all_optics: dict[int, Any] = {}

        def bundle_for(optic, wavelength) -> int:
            key = _make_generic_trace_key(optic, wavelength)
            if key not in bundle_index:
                bundle_index[key] = len(self._bundles)
                self._bundles.append(_BundleJob(optic, wavelength))
            return bundle_index[key]

        for i, operand in enumerate(self.problem.operands):
            op_type = operand.operand_type
            data = operand.input_data
            if data and data.get("optic") is not None:
                optics.setdefault(id(data["optic"]), data["optic"])

            if op_type in _TRACE_GENERIC_OPERANDS:
                b = bundle_for(data["optic"], data["wavelength"])
                ray_idx = self._bundles[b].add_operand(
                    i,
                    data["Hx"],
                    data["Hy"],
                    data["Px"],
                    data["Py"],
                )
                self._operand_plan.append(("generic", b, ray_idx))

            elif op_type in _TRACE_OPERANDS:
                wl = data.get("wavelength", 0.587)
                optic = data["optic"]
                num_rays = data.get("num_rays", 100)
                dist = data.get("distribution", "hexapolar")

                if wl == "all":
                    # One sub-job per wavelength, merged into its bundle
                    all_optics.setdefault(id(optic), optic)
                    wavelengths = optic.wavelengths.get_wavelengths()
                    primary_index = optic.wavelengths.primary_index
                else:
                    wavelengths = [wl]
                    primary_index = None

                reads = []
                for wave in wavelengths:
                    b = bundle_for(optic, wave)
                    key = _make_trace_key(
                        optic, data["Hx"], data["Hy"], wave, num_rays, dist
                    )
                    s = self._bundles[b].add_segment(
                        key, data["Hx"], data["Hy"], num_rays, dist
                    )
                    reads.append((b, s + 1))
                self._operand_plan.append(("distribution", primary_index, reads))

            else:
                # Non-ray operand (paraxial, aberration, lens, etc.)
                self._operand_plan.append(("direct", None, None))

        self._optics = tuple(optics.values())
        self._all_optics = tuple(all_optics.values())
        self._wavelength_signature = self._current_wavelength_signature()
        self._build_index_arrays()

    def _build_index_arrays(self) -> None:
        """Precompute per-bundle index arrays of the single-ray operands."""
        self._attr_groups = []
        operands = self.problem.operands
        for job in self._bundles:
            groups: dict[str, tuple[list, list, list]] = {}
            for ray_idx, i in enumerate(job.operand_indices):
                operand = operands[i]
                attr = _GENERIC_ATTRS.get(operand.operand_type)
                if attr is None:
                    continue
                idx, surfs, rays = groups.setdefault(attr, ([], [], []))
                idx.append(i)
                surfs.append(operand.input_data["surface_number"])
                rays.append(ray_idx)
            self._attr_groups.append(groups)

    def _current_wavelength_signature(self) -> tuple:
        """Wavelength sets of the optics with ``wavelength="all"`` operands."""
        return tuple(
            (
                tuple(optic.wavelengths.get_wavelengths()),
                optic.wavelengths.primary_index,
            )
            for optic in self._all_optics
        )

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
        """Rebuild the plan when the problem size or a wavelength set changed."""
        if len(self._operand_plan) != len(self.problem.operands) or (
            self._all_optics
            and self._current_wavelength_signature() != self._wavelength_signature
        ):
            self._analyze()

    @staticmethod
    def _safe_execute(job):
        """Execute a trace job and return ``None`` on trace failures."""
        try:
            return job.execute()
        except Exception:
            return None

    def _trace_bundle(self, job: _BundleJob) -> dict[int, tuple[Any, int, int]]:
        """Trace a bundle; return ``{part: (view, start, stop)}`` per traced part.

        A bundle that fails as a whole is retraced part by part, so only the
        parts that fail on their own are missing from the result.

        Single rays are read before the next bundle is traced; spot data is
        read after all bundles, so it is prefetched here.
        """
        parts = job.parts
        surface_group = self._safe_execute(job)
        if surface_group is not None:
            view = _TraceView(surface_group, _SPOT_ATTRS if job.segments else ())
            # Part 0 (the single rays) always starts the bundle
            return {
                part: (view, *job.offsets.get(part, (0, len(job.ray_params))))
                for part in parts
            }

        reads: dict[int, tuple[Any, int, int]] = {}
        if len(parts) > 1:
            for part in parts:
                surface_group = self._safe_execute(_BundlePart(job, part))
                if surface_group is not None:
                    # Later parts overwrite the surface group: read now
                    prefetch = _RAY_ATTRS if part == 0 else _SPOT_ATTRS
                    view = _TraceView(surface_group, prefetch)
                    reads[part] = (view, *job.offsets[part])
        return reads

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

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

    def _residuals(self, include_zero: bool):
        """Residuals ``effective_weight * delta`` of the operands, in order.

        Args:
            include_zero: Keep zero-effective-weight operands as exact zeros
                (``residual_vector``) instead of dropping them
                (``fun_array``).

        Returns:
            A 1-D array, or None when no operand contributes.
        """
        self._ensure_plan_current()
        operands = list(self.problem.operands)
        weights = [operand.effective_weight() for operand in operands]
        raw_values: list[Any] = [None] * len(operands)
        covered = [False] * len(operands)
        chunks: list[tuple[list[int], Any]] = []
        traced: list[dict[int, tuple[Any, int, int]]] = []

        # --- 1. Super-bundles: one trace per (optic, wavelength) ---
        for b, job in enumerate(self._bundles):
            reads = self._trace_bundle(job)
            traced.append(reads)
            if not job.ray_params:
                continue
            read = reads.get(0)

            # Vectorized extraction and residuals of scalar-target operands
            for attr, (idx, surfs, rays) in self._attr_groups[b].items():
                if read is None:
                    break
                # Zero-weight operands are exact zeros (or dropped) below
                keep = [
                    k
                    for k, i in enumerate(idx)
                    if operands[i].target is not None and weights[i] != 0.0
                ]
                if not keep:
                    continue
                if len(keep) < len(idx):
                    idx = [idx[k] for k in keep]
                    surfs = [surfs[k] for k in keep]
                    rays = [rays[k] for k in keep]
                values = getattr(read[0], attr)[surfs, rays]
                targets = be.array([float(operands[i].target) for i in idx])
                w = be.array([weights[i] for i in idx])
                chunks.append((idx, w * (values - targets)))
                for i in idx:
                    covered[i] = True

            # Bounded, local-coordinate and AOI operands, one by one
            for ray_idx, i in enumerate(job.operand_indices):
                if covered[i] or weights[i] == 0.0:
                    continue
                if read is None:
                    raw_values[i] = be.nan
                    continue
                operand = operands[i]
                raw_values[i] = _extract_value_generic(
                    operand.operand_type, read[0], operand.input_data, ray_idx
                )

        # --- 2. Spot operands: slices of the bundles ---
        for i, (plan_type, primary_index, reads) in enumerate(self._operand_plan):
            if plan_type != "distribution" or weights[i] == 0.0:
                continue
            data = operands[i].input_data
            segments = [traced[b].get(part) for b, part in reads]
            if any(segment is None for segment in segments):
                raw_values[i] = be.nan  # failure isolation
                continue
            surface_number = _spot_surface(data)
            xs = [view.x[surface_number, lo:hi] for view, lo, hi in segments]
            ys = [view.y[surface_number, lo:hi] for view, lo, hi in segments]
            raw_values[i] = _extract_rms_spot(xs, ys, data, primary_index)

        # --- 3. Direct operands ---
        self._evaluate_direct_operands(raw_values)

        return self._assemble(raw_values, chunks, covered, weights, include_zero)

    def _assemble(
        self,
        raw_values: list[Any],
        chunks: list[tuple[list[int], Any]],
        covered: list[bool],
        weights: list[float],
        include_zero: bool,
    ):
        """Join vectorized and scalar residuals into one vector in operand order.

        Returns:
            A 1-D array, or None when no operand contributes.
        """
        operands = self.problem.operands
        eq_idx, eq_values, eq_targets = [], [], []
        other_idx, other_residuals = [], []
        zero_idx = []
        for i, operand in enumerate(operands):
            if covered[i]:
                continue
            if weights[i] == 0.0:
                if include_zero:
                    zero_idx.append(i)
                continue
            value = raw_values[i]
            if value is None:
                raise RuntimeError(
                    f"Operand {i} ({operand.operand_type}) was not evaluated"
                )
            if operand.target is not None:
                if be.size(value) == 1:
                    value = be.ravel(value)[0]
                eq_idx.append(i)
                eq_values.append(value)
                eq_targets.append(float(operand.target))
            else:
                other_idx.append(i)
                other_residuals.append(weights[i] * self._compute_delta(operand, value))

        if eq_idx:
            w = be.array([weights[i] for i in eq_idx])
            chunks.append((eq_idx, w * (be.stack(eq_values) - be.array(eq_targets))))
        if other_idx:
            chunks.append((other_idx, be.stack(other_residuals)))
        if zero_idx:
            chunks.append((zero_idx, be.zeros((len(zero_idx),))))
        if not chunks:
            return None

        order = [i for idx, _ in chunks for i in idx]
        residuals = (
            chunks[0][1]
            if len(chunks) == 1
            else be.concatenate([values for _, values in chunks])
        )
        permutation = sorted(range(len(order)), key=order.__getitem__)
        if permutation != list(range(len(order))):
            residuals = residuals[permutation]
        return residuals

    def fun_array(self):
        """Compute contribution terms for each active operand.

        This is the batched equivalent of
        ``OptimizationProblem.fun_array()``.

        All ray operands of one (optic, wavelength) share one trace; operand
        values are extracted right after each trace, before the next trace
        overwrites the shared ``surface_group`` state.
        """
        with _epoch.evaluation_epoch(self._optics):
            residuals = self._residuals(include_zero=False)
        if residuals is None:
            return be.array([0.0])
        return residuals**2

    def residual_vector(self):
        """Compute the vector of weighted operand deltas (unsquared).

        Returns a 1-D array whose *i*-th element is ``weight_i * delta_i``
        for each operand. This is the residual vector **r** needed by
        least-squares algorithms such as Levenberg-Marquardt.

        All ray operands of one (optic, wavelength) share one trace; operand
        values are extracted right after each trace, before the next trace
        overwrites the shared ``surface_group`` state.
        """
        with _epoch.evaluation_epoch(self._optics):
            residuals = self._residuals(include_zero=True)
        if residuals is None:
            return be.array([])
        return residuals

    def sum_squared(self):
        """Compute the sum of squared weighted deltas.

        This is the batched equivalent of
        ``OptimizationProblem.sum_squared()``.
        """
        with _epoch.evaluation_epoch(self._optics):
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
