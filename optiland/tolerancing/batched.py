"""Batched tolerancing consumers (design 6.5, plan WP7).

``monte_carlo_batched`` and ``sensitivity_batched`` are drop-in twins of
``MonteCarlo.run`` and ``SensitivityAnalysis.run`` that trace every sample of a
chunk in **one** fused launch instead of one sequential ``optic.trace`` per
sample.  They return the same ``pandas.DataFrame`` that the sequential analysis
returns, with the same columns in the same order.

How the two paths stay the same computation
-------------------------------------------

``RayOperand.*`` re-traces before it reads records (``operand/ray.py:56-57,
355-367``), so ``BatchTraceResult.install()`` alone cannot feed an operand: the
operand would just throw the installed rows away and trace again.  Plan 3.9
therefore fixes the contract used here:

* only operands whose ``operand_type`` is in :data:`BATCHABLE_OPERANDS` are
  evaluated on the batched path;
* each reader mirrors its operand's **post-trace** expression with the same
  ``be.*`` calls, in the same order, on ``optic.surfaces.<q>[surface_number, :]``
  after ``result.install(optic, b)``;
* everything else -- an operand outside the map, ``wavelength="all"``, a
  compensator, a sampler the sequential analysis does not support -- falls back
  to the existing loop, which runs completely unchanged.  The returned frame
  then carries ``df.attrs["batched"] is False`` and a ``reason``.

How close "the same computation" is, exactly
--------------------------------------------

The reader applies the same backend ops to the rows ``install()`` writes, so
the batched frame is as close to the sequential frame as those rows are -- and
that is plan 7.1's **two-tier** rule, not one blanket promise.  The tier is
decided by the operand's bundle size, and both frames carry it in
``df.attrs["tier"]`` beside ``df.attrs["num_rays"]``:

* **tier A**, every operand bundle strictly larger than
  :data:`TIER_A_MIN_RAYS` (1024) rays: the frames are equal **cell for cell**.
  ``test_monte_carlo_batched_matches_loop[19 rings]`` and
  ``test_sensitivity_batched_matches_loop[19 rings]`` assert exactly that with
  an exact float64 comparison, at N = 1,141.
* **tier B**, any operand bundle of 1024 rays or fewer: equal to
  ``64 * eps_mode * scale`` per cell, and in df64 **not** bit for bit.  This is
  plan 7.1's tier-B **site 1** and it is not a kernel defect: below
  ``materials/base.py``'s ``_MAX_VALUE_KEY_ARRAY_SIZE = 1024`` the per-op path
  evaluates a dispersive index on the whole ``w`` array, above it on one
  uniform representative -- which is what the kernel always uses (plan 3.8).
  Measured (finding R2-V1-10): a Cooke triplet's ``rms_spot_size`` at 12
  hexapolar rings (N = 469) differs from ``MonteCarlo.run`` on 3-4 of 4 samples
  in df64, by at most 6.5e-16 against a bound of 2.3e-13; at 19 rings
  (N = 1,141) and at 27 rings (N = 2,269) it is exact, in sf64 it is exact at
  every size, and with ``IdealMaterial`` glasses it is exact at N = 469 too.
  ``test_monte_carlo_batched_matches_loop[12 rings]``,
  ``test_sensitivity_batched_matches_loop[12 rings]`` and the lock test
  ``tests/metal/test_trace_adversarial_round2.py::test_r2v110_locked`` pin it;
  the entry is in ``NOTES/fused-trace-research/documented-limits.md``.

Nothing here refuses a small bundle: the batch gate deliberately skips the
bundle-size checks (design 6.3), and refusing 256 < N <= 1024 would not help
anyway -- the hook path fuses that range too and is tier B there for the same
reason.  Tolerancing operands are exactly where small bundles live, so the
qualification is stated rather than hidden.

Deviations, stated here and in place
------------------------------------

1. **``record=True`` (every surface), not just the rows the operands read.**
   Design 6.5 says "``record`` = the rows the operands read".  That cannot be
   honoured while the reader mirrors the operand expression, because
   ``SurfaceGroup.x`` *drops surfaces with empty records*
   (``surface_group.py:194-195``): recording a subset would silently shift the
   ``[surface_number, :]`` index the operand uses.  Recording everything keeps
   the index identical to a real trace.  Cost at the default chunk of 64
   designs, 8 surfaces and 1,141 rays: ~37 MB.
2. **The optic is left nominal.**  ``MonteCarlo.run`` leaves the system at the
   last iteration's perturbation; ``trace_batch`` restores every variable in a
   ``finally``, so these helpers return with the system nominal (which is what
   ``SensitivityAnalysis.run`` does anyway).  Only the *records* on
   ``optic.surfaces`` are left behind, holding the last evaluated design.
3. **Operands are grouped by launch configuration.**  Operands that ask for
   different ``Hx``/``Hy``/``num_rays``/``distribution`` need different launch
   sets, so each group costs one ``trace_batch`` per chunk.  All operands must
   still share one wavelength (plan 3.9); a mixed set falls back.
4. **The batch is evaluated with autograd off.**  ``Tolerancing.__init__``
   builds a ``CompensatorOptimizer``, and ``OptimizationProblem.__init__``
   enables ``be.grad_mode`` on the torch backend
   (``optimization/problem.py:63-66``).  A grad-enabled bundle is a
   *structural* gate refusal (plan 1.2), so without :func:`_forward_only`
   every tolerancing problem would fall back and the helper would be a slower
   spelling of the loop it replaces.  Nothing here differentiates and the
   caller's setting is restored.
   This is not cosmetic: **measured**, the per-op Metal path is not
   grad-invariant in df64 (and on a Newton geometry it is not
   ``torch.is_grad_enabled()``-invariant either -- finding R2-V1-04).  Same
   optic, same radii, same operand call, kernel off throughout, only
   ``be.grad_mode`` different -- 53 of 1,141 image-row
   ``x`` low words and 35 ``y`` low words change, moving the RMS spot by 1.7e-13
   relative; sf64 is unaffected.  The fused path's contract is therefore with
   the **grad-off** per-op path (plan section 7, and the harness's
   ``OPTILAND_TEST_MPS_GRAD=0``), and that is what
   ``test_monte_carlo_batched_matches_loop`` compares against.

No torch and no ``optiland.backend.torch_backend.metal.*`` module is imported at
module scope (plan 0.2.4); ``optiland.raytrace.batch_trace`` is itself free of
both.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

import optiland.backend as be
from optiland.raytrace.batch_trace import trace_batch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Iterator, Sequence

__all__ = [
    "BATCHABLE_OPERANDS",
    "Reader",
    "monte_carlo_batched",
    "sensitivity_batched",
]

#: Designs per fused launch when the caller does not say otherwise.
DEFAULT_CHUNK = 64

#: The bundle size above which the batched frame equals the sequential frame
#: cell for cell (plan 7.1 tier A).  It is ``materials/base.py``'s
#: ``_MAX_VALUE_KEY_ARRAY_SIZE``: at or below it the per-op path evaluates a
#: dispersive index on the whole wavelength array instead of on one uniform
#: representative, which is plan 7.1's tier-B site 1.  See the module
#: docstring and finding R2-V1-10.
TIER_A_MIN_RAYS = 1024


# ---------------------------------------------------------------------------
# Readers: one operand's post-trace expression, on installed records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Reader:
    """How one operand type is evaluated from installed records.

    Attributes:
        operand_type: the ``METRIC_DICT`` key this reader stands in for.
        read: ``read(optic, input_data) -> value``.  Runs *after*
            ``BatchTraceResult.install(optic, b)`` and must reproduce the
            operand function's post-trace expression with the same ``be.*``
            calls in the same order.
        launch: ``launch(input_data) -> tuple``.  The launch configuration the
            operand needs, as ``(Hx, Hy, num_rays, distribution)``; operands
            sharing one tuple share one ``trace_batch`` call.
        wavelength: ``wavelength(input_data) -> float``.
        check: ``check(input_data) -> str | None``.  A refusal reason when this
            operand cannot be batched, None when it can.
    """

    operand_type: str
    read: Callable[[Any, dict], Any]
    launch: Callable[[dict], tuple]
    wavelength: Callable[[dict], Any]
    check: Callable[[dict], str | None]


def _read_rms_spot_size(optic: Any, input_data: dict) -> Any:
    """``RayOperand.rms_spot_size``' post-trace expression, verbatim.

    Mirrors ``operand/ray.py:364-386`` for the scalar-wavelength branch: the
    NaN-omitting means, the centroid-relative square radius, the square root,
    and the ``nan_policy`` handling -- the same ``be.*`` calls in the same
    order, on the records ``install()`` just wrote.  Nothing is re-traced.
    """
    surface_number = input_data["surface_number"]
    nan_policy = input_data.get("nan_policy", "propagate")

    valid_nan_policies = ("propagate", "omit", "raise")
    if nan_policy not in valid_nan_policies:
        raise ValueError(
            f"Invalid nan_policy '{nan_policy}'. Must be one of {valid_nan_policies}."
        )

    def _has_nan(*arrays):
        return any(be.any(be.isnan(a)) for a in arrays)

    x = optic.surfaces.x[surface_number, :].flatten()
    y = optic.surfaces.y[surface_number, :].flatten()
    has_nan = _has_nan(x, y)
    r2 = (x - be.nanmean(x)) ** 2 + (y - be.nanmean(y)) ** 2
    rms = be.sqrt(be.nanmean(r2))

    if has_nan:
        if nan_policy == "raise":
            raise ValueError(
                "rms_spot_size encountered a NaN ray intersection on "
                f"surface {surface_number}. This typically indicates "
                "total internal reflection or a vignetted ray. Use "
                "nan_policy='omit' to compute the RMS from valid rays "
                "only, or leave nan_policy='propagate' (default) to "
                "return NaN."
            )
        if nan_policy == "propagate":
            return rms * be.nan
    return rms


def _launch_rms_spot_size(input_data: dict) -> tuple:
    """``rms_spot_size``' launch configuration."""
    return (
        input_data["Hx"],
        input_data["Hy"],
        input_data["num_rays"],
        input_data.get("distribution", "hexapolar"),
    )


def _check_rms_spot_size(input_data: dict) -> str | None:
    """Why ``rms_spot_size`` cannot be batched, or None."""
    if input_data.get("wavelength") == "all":
        # The "all" branch concatenates one trace per wavelength and centres
        # every one on the primary wavelength's centroid (operand/ray.py:
        # 352-363).  That is a multi-wavelength axis, which v1 does not have.
        return "wavelength_all"
    for key in ("optic", "surface_number", "Hx", "Hy", "num_rays", "wavelength"):
        if key not in input_data:
            return f"missing_input:{key}"
    return None


#: Operand type -> reader (plan 3.9).  v1 is ``rms_spot_size`` with a scalar
#: wavelength; anything else runs on the existing sequential loop.
BATCHABLE_OPERANDS: dict[str, Reader] = {
    "rms_spot_size": Reader(
        operand_type="rms_spot_size",
        read=_read_rms_spot_size,
        launch=_launch_rms_spot_size,
        wavelength=lambda data: data["wavelength"],
        check=_check_rms_spot_size,
    ),
}


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------
def _refusal_reason(tolerancing: Any) -> str | None:
    """Why this tolerancing problem cannot be batched, or None (plan 3.9)."""
    if tolerancing.compensator.has_variables:
        # A compensator runs a nested scipy optimisation per sample
        # (core.py:143-158); the samples are not independent designs.
        return "compensator_variables"

    optic = tolerancing.optic
    wavelengths: list[Any] = []
    for operand in tolerancing.operands:
        reader = BATCHABLE_OPERANDS.get(operand.operand_type)
        if reader is None:
            return f"operand_type:{operand.operand_type}"
        input_data = operand.input_data or {}
        if input_data.get("optic") is not optic:
            return f"operand_optic:{operand.operand_type}"
        why = reader.check(input_data)
        if why is not None:
            return why
        wavelengths.append(reader.wavelength(input_data))

    if not _single_wavelength(wavelengths):
        return "mixed_wavelength"
    return None


def _single_wavelength(wavelengths: Sequence[Any]) -> bool:
    """True when every operand asks for the same wavelength."""
    try:
        distinct = {float(w) for w in wavelengths}
    except (TypeError, ValueError):
        return False
    return len(distinct) <= 1


def _sampler_reason(tolerancing: Any) -> str | None:
    """``SensitivityAnalysis`` only supports ``RangeSampler`` (its own rule)."""
    from optiland.tolerancing.perturbation import RangeSampler

    for perturbation in tolerancing.perturbations:
        if not isinstance(perturbation.sampler, RangeSampler):
            return "sampler"
    return None


# ---------------------------------------------------------------------------
# The batched evaluation
# ---------------------------------------------------------------------------
def _operand_groups(tolerancing: Any) -> dict[tuple, list[tuple[int, Any, Reader]]]:
    """Operands grouped by the launch configuration they need."""
    groups: dict[tuple, list[tuple[int, Any, Reader]]] = {}
    for index, operand in enumerate(tolerancing.operands):
        reader = BATCHABLE_OPERANDS[operand.operand_type]
        key = reader.launch(operand.input_data or {})
        groups.setdefault(key, []).append((index, operand, reader))
    return groups


def _values_table(rows: Sequence[Sequence[Any]], n_vars: int) -> np.ndarray:
    """The design table, object dtype so a sampler's own type survives.

    ``trace_batch`` narrows it to float64 when every entry converts; the raw
    objects are kept here so a non-numeric perturbation (a glass name) would
    reach ``Variable.update`` unchanged.
    """
    table = np.empty((len(rows), n_vars), dtype=object)
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            table[i, j] = value
    return table


@contextmanager
def _forward_only() -> Iterator[None]:
    """Evaluate the batch with autograd off, then restore the caller's setting.

    Deviation 4, measured.  ``Tolerancing.__init__`` builds a
    ``CompensatorOptimizer``, and ``OptimizationProblem.__init__`` calls
    ``be.grad_mode.enable()`` on the torch backend
    (``optimization/problem.py:63-66``, with a ``UserWarning``).  *Every*
    tolerancing problem therefore arrives with autograd on, and plan 1.2 makes
    a grad-enabled bundle a **structural** gate refusal -- so without this the
    batched path could never fuse, on any problem, and would silently be a slow
    spelling of the sequential loop.

    Nothing here differentiates: the perturbations are sampled, the trace is
    forward, the operand is a reduction.

    Two qualifications, both measured, on "autograd only records a graph":

    * ``be.grad_mode.disable()`` does NOT clear ``torch.is_grad_enabled()``
      (round-2 observation 3).  It controls which leaves carry
      ``requires_grad``, which is what the gate reads, and that is exactly what
      this context manager is for.
    * ``torch.is_grad_enabled()`` DOES change the forward arithmetic on a
      Newton geometry: ``NewtonRaphsonGeometry.distance`` returns the DiffOptics
      one-step correction instead of the primal solve (round-2 finding
      R2-V1-04).  ``trace_batch`` therefore traces both of its legs inside
      ``torch.no_grad()`` and this manager does not have to.  The systems this
      module tolerances today are conic (``rms_spot_size`` on a Cooke triplet),
      where the branch does not exist, so
      ``test_monte_carlo_batched_matches_loop`` still compares against a
      grad-**on** sequential run and demands equality; a Newton system would
      need that reference under ``torch.no_grad()`` too.
    """
    if be.get_backend() != "torch":
        yield
        return
    grad_mode = be.grad_mode
    previous = bool(grad_mode.requires_grad)
    grad_mode.disable()
    try:
        yield
    finally:
        if previous:
            grad_mode.enable()
        else:
            grad_mode.disable()


def _evaluate_batched(
    tolerancing: Any,
    variables: Sequence[Any],
    table: np.ndarray,
    chunk: int,
) -> tuple[list[list[float]], bool, str | None, int]:
    """Every design's operand values, in ``chunk``-sized fused launches.

    Returns:
        (values, fused, reason, n_rays) -- ``values[b][i]`` is operand ``i`` of
        design ``b``; ``fused`` is True only when every launch was produced by
        the kernel; ``reason`` names the first gate refusal when it is not (the
        fallback is still correct, just sequential, and a silently unfused run
        would otherwise look exactly like a fused one); ``n_rays`` is the
        SMALLEST bundle any launch used, which is what decides the frame's tier
        (module docstring, :data:`TIER_A_MIN_RAYS`, finding R2-V1-10) -- the
        smallest, because one tier-B operand qualifies the whole frame.

    A refusal worth knowing about: an optic **constructed** while
    ``be.grad_mode`` was enabled holds ``requires_grad`` leaves, which plan 1.2
    refuses structurally, and :func:`_forward_only` cannot undo that without
    detaching the caller's own tensors.  Build the optic before the
    ``Tolerancing`` (the natural order) or with grad off.
    """
    optic = tolerancing.optic
    groups = _operand_groups(tolerancing)
    wavelength = BATCHABLE_OPERANDS[tolerancing.operands[0].operand_type].wavelength(
        tolerancing.operands[0].input_data or {}
    )
    n_designs = int(table.shape[0])
    n_operands = len(tolerancing.operands)
    values: list[list[float]] = [[0.0] * n_operands for _ in range(n_designs)]
    fused = True
    reason: str | None = None
    n_rays = 0

    with _forward_only():
        for start in range(0, n_designs, chunk):
            rows = table[start : start + chunk]
            for (Hx, Hy, num_rays, distribution), members in groups.items():
                result = trace_batch(
                    optic,
                    variables,
                    rows,
                    Hx=Hx,
                    Hy=Hy,
                    wavelength=wavelength,
                    num_rays=num_rays,
                    distribution=distribution,
                    # Deviation 1 of the module docstring: every surface,
                    # because SurfaceGroup.x drops empty records and would
                    # shift the operand's [surface_number, :] index.
                    record=True,
                )
                fused = fused and bool(result.fused)
                launched = int(result.n_rays)
                n_rays = launched if n_rays == 0 else min(n_rays, launched)
                if not result.fused and reason is None:
                    skips = [
                        k.split(":", 1)[1]
                        for k in result.stats
                        if k.startswith("fused_trace_skip:")
                    ]
                    reason = f"gate:{','.join(sorted(skips))}" if skips else "gate"
                for b in range(int(rows.shape[0])):
                    result.install(optic, b)
                    for index, operand, reader in members:
                        values[start + b][index] = float(
                            reader.read(optic, operand.input_data or {})
                        )
    return values, fused, reason, n_rays


def _label(
    frame: Any, *, batched: bool, fused: bool, reason: str | None, n_rays: int | None
) -> Any:
    """Stamp a frame with which path produced it and which tier applies.

    ``tier`` is "A" when every operand bundle was larger than
    :data:`TIER_A_MIN_RAYS`, "B" otherwise (module docstring, R2-V1-10).  A
    frame the sequential loop produced is trivially tier A: it *is* the
    reference, and ``num_rays`` is None because no launch was made here.
    """
    frame.attrs["batched"] = batched
    frame.attrs["fused"] = fused
    frame.attrs["reason"] = reason
    frame.attrs["num_rays"] = n_rays
    frame.attrs["tier"] = "A" if n_rays is None or n_rays > TIER_A_MIN_RAYS else "B"
    return frame


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
def monte_carlo_batched(
    tolerancing: Any, num_iterations: int, *, chunk: int = DEFAULT_CHUNK
):
    """``MonteCarlo.run`` with every sample of a chunk in one fused launch.

    The perturbation values are drawn up front through the existing samplers,
    in the order ``MonteCarlo.run`` draws them (sample ``i`` takes one draw
    from every perturbation, in perturbation order), so a seeded sampler gives
    the same sequence either way.

    Args:
        tolerancing: the ``Tolerancing`` problem.
        num_iterations: how many samples to draw.
        chunk: designs per fused launch.

    Returns:
        pandas.DataFrame: the same frame ``MonteCarlo.run`` produces -- one
        column per perturbation (its ``str(variable)``) and one per operand
        (``"<i>: <operand>"``).  ``df.attrs["batched"]`` says which path ran,
        ``df.attrs["fused"]`` whether the kernel produced the rows, and
        ``df.attrs["reason"]`` names the refusal when it did not.
        ``df.attrs["tier"]`` is "A" when the operand column equals
        ``MonteCarlo.run``'s **cell for cell** and "B" when it equals it only
        to ``64 * eps_mode * scale``; "B" is what a bundle of
        :data:`TIER_A_MIN_RAYS` rays or fewer can deliver, and
        ``df.attrs["num_rays"]`` is the bundle that decided it (module
        docstring, finding R2-V1-10).  The perturbation columns are the
        samplers' draws and are exact either way.

    Raises:
        ValueError: whatever ``MonteCarlo`` raises for an invalid problem (no
            operands, no perturbations) -- the analysis object is constructed
            first precisely so the validation is identical.
    """
    from optiland.tolerancing.monte_carlo import MonteCarlo

    analysis = MonteCarlo(tolerancing)
    reason = _refusal_reason(tolerancing)
    if reason is None and int(num_iterations) <= 0:
        reason = "no_iterations"
    if reason is not None:
        return _loop_result(analysis, reason, num_iterations)

    perturbations = list(tolerancing.perturbations)
    variables = [perturbation.variable for perturbation in perturbations]

    tolerancing.reset()
    draws = [
        [perturbation.sampler.sample() for perturbation in perturbations]
        for _ in range(int(num_iterations))
    ]
    table = _values_table(draws, len(variables))

    values, fused, gate_reason, n_rays = _evaluate_batched(
        tolerancing, variables, table, int(chunk)
    )

    records = []
    for i, draw in enumerate(draws):
        record: dict[str, Any] = {}
        for perturbation, value in zip(perturbations, draw, strict=True):
            record[str(perturbation.variable)] = float(value)
        record.update(
            {
                f"{name}": value
                for name, value in zip(analysis.operand_names, values[i], strict=False)
            },
        )
        records.append(record)

    frame = pd.DataFrame(records)
    _label(frame, batched=True, fused=fused, reason=gate_reason, n_rays=n_rays)
    return frame


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------
def sensitivity_batched(tolerancing: Any, *, chunk: int = DEFAULT_CHUNK):
    """``SensitivityAnalysis.run`` with the whole sweep in fused launches.

    One design per (perturbation, sample) pair: the swept variable takes the
    sampled value and every other perturbed variable stays nominal, exactly as
    the sequential sweep's ``reset()`` + single ``apply()`` leaves the system.

    Args:
        tolerancing: the ``Tolerancing`` problem.
        chunk: designs per fused launch.

    Returns:
        pandas.DataFrame: the same frame ``SensitivityAnalysis.run`` produces,
        with ``perturbation_type`` and ``perturbation_value`` columns first.
        ``df.attrs`` carries ``batched``, ``fused``, ``reason``, ``num_rays``
        and ``tier`` as in :func:`monte_carlo_batched` -- including the tier-B
        qualification for an operand bundle of :data:`TIER_A_MIN_RAYS` rays or
        fewer (finding R2-V1-10).
    """
    from optiland.tolerancing.sensitivity_analysis import SensitivityAnalysis

    analysis = SensitivityAnalysis(tolerancing)
    reason = _refusal_reason(tolerancing) or _sampler_reason(tolerancing)
    if reason is not None:
        return _loop_result(analysis, reason)

    perturbations = list(tolerancing.perturbations)
    variables = [perturbation.variable for perturbation in perturbations]

    tolerancing.reset()
    nominal = [variable.value for variable in variables]

    rows: list[list[Any]] = []
    labels: list[tuple[str, Any]] = []
    for k, perturbation in enumerate(perturbations):
        for _ in range(int(perturbation.sampler.size)):
            value = perturbation.sampler.sample()
            row = list(nominal)
            row[k] = value
            rows.append(row)
            labels.append((str(perturbation.variable), value))

    table = _values_table(rows, len(variables))
    values, fused, gate_reason, n_rays = _evaluate_batched(
        tolerancing, variables, table, int(chunk)
    )

    records = []
    for i, (name, value) in enumerate(labels):
        record: dict[str, Any] = {
            "perturbation_type": name,
            "perturbation_value": value,
        }
        record.update(
            {
                f"{operand}": operand_value
                for operand, operand_value in zip(
                    analysis.operand_names, values[i], strict=False
                )
            },
        )
        records.append(record)

    frame = pd.DataFrame(records)
    tolerancing.reset()
    _label(frame, batched=True, fused=fused, reason=gate_reason, n_rays=n_rays)
    return frame


# ---------------------------------------------------------------------------
# The refusal path: the existing analysis, untouched
# ---------------------------------------------------------------------------
def _loop_result(analysis: Any, reason: str, *run_args: Any):
    """Run the sequential analysis and label its frame."""
    analysis.run(*run_args)
    frame = analysis.get_results()
    return _label(frame, batched=False, fused=False, reason=reason, n_rays=None)
