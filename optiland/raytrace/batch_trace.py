"""Batched multi-design real ray tracing (design 6, plan WP7).

``trace_batch`` traces ``B`` *designs* of one optical system in a single fused
Metal launch: row ``b`` of ``values`` is fed through the ordinary
``Variable.update`` / ``optic.updater.update()`` mutation path, the per-surface
tables are compiled per design, and the kernel walks the surface list for every
``(design, ray)`` thread.  Nothing re-implements updater semantics: the contract
is the live optic, always.

Three properties are load-bearing and each has a named test in
``tests/metal/test_trace_batch.py``:

* **The contract loop is the reference.**  On NumPy, on torch-CPU, with
  ``OPTILAND_METAL_FUSED_TRACE=0``, or whenever the gate refuses, ``trace_batch``
  runs the same designs through ``optic.trace`` and returns the same
  ``BatchTraceResult`` shape.  The fused result must equal it component for
  component (plan 7.1 tier A).
* **Tier 1 is a cache, not a second implementation.**  With no pickups and no
  solves, only the rows a variable touches are re-read per design; every call
  re-compiles one or two designs through tier 0 and compares the tables with
  ``np.array_equal`` (the canary of plan 3.9).  A mismatch counts
  ``fused_trace:tier1_canary_mismatch`` and the whole batch is recompiled
  through tier 0.
* **Units are the optimizer's units.**  ``values[b, j]`` is in variable ``j``'s
  *scaled* units, exactly what ``Variable.update`` receives from scipy
  (plan 3.9 [fix: L1.10]).  Callers holding physical values convert with
  ``var.variable.scale(physical)``.

No torch, and no ``optiland.backend.torch_backend.metal.*`` module, is imported
at module scope (plan 0.2.4): every Metal import below is inside a function, so
importing this module on the NumPy path pulls in nothing GPU-related.

Deviations from design 6, stated in place:

* The mirror-drift refusal of plan 3.7 applies to the batch path too: when
  ``trace_mirror.check_all()`` reports drift, ``trace_batch`` warns once and
  runs the contract loop.  The check is the driver's own cached one, so the
  hook and the batch path never disagree about it.
* ``tol_floor_scale`` is refused.  Day-1 question Q3 decided the
  ``TOL_CROSSOVER`` policy in favour of the per-design per-op re-trace, and the
  frozen ``launch_trace`` signature (plan 3.3) carries no tolerance-floor
  argument, so the alternative cannot be honoured silently.
* The per-design late fallback does **not** raise under
  ``OPTILAND_METAL_FUSED_TRACE=require``: re-tracing the affected design on the
  per-op path *is* the batch contract (design 6.3), not a silent degradation.
  Every such design is still counted in ``fused_trace:late_fallback`` and named
  in ``BatchTraceResult.late_fallback_designs``.
* Counters are per *call*, not per design: one ``fused_trace:candidates`` and
  one ``fused_trace:traces`` per fused ``trace_batch``, ``fused_trace:designs``
  and ``fused_trace:surface_steps`` by design count.  The census identity of
  plan 1.3 is a hook-path identity; the batch path is never exercised by the
  suite census.
"""

from __future__ import annotations

import math
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

import optiland.backend as be
from optiland.distribution import create_distribution
from optiland.rays import RealRays

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Iterator, Sequence

__all__ = [
    "BatchTraceResult",
    "FDResult",
    "fd_jacobian",
    "fd_reference_cpu",
    "trace_batch",
]

#: ``final`` plane index -> the ``RealRays`` attribute it carries.
_FINAL_ATTRS: tuple[str, ...] = (
    "x",
    "y",
    "z",
    "L",
    "M",
    "N",
    "i",
    "opd",
    "L0",
    "M0",
    "N0",
)

#: The attributes ``BatchTraceResult`` exposes, in ``snap`` plane order
#: (``trace_layout.S_*``): surface attribute per plane.
_RESULT_ATTRS: tuple[str, ...] = (
    "x",
    "y",
    "z",
    "L",
    "M",
    "N",
    "intensity",
    "opd",
)


# ---------------------------------------------------------------------------
# Small backend-agnostic helpers
# ---------------------------------------------------------------------------
def _is_metal(a: Any) -> bool:
    """True for a ``MetalFloat64`` without importing the Metal package."""
    return type(a).__name__ == "MetalFloat64"


def _switch() -> str:
    """``OPTILAND_METAL_FUSED_TRACE``: ``"1"`` (default), ``"0"`` or ``"require"``."""
    return os.environ.get("OPTILAND_METAL_FUSED_TRACE", "1")


@contextmanager
def _fused_disabled() -> Iterator[None]:
    """Force the per-op path inside the block (the per-design re-trace)."""
    previous = os.environ.get("OPTILAND_METAL_FUSED_TRACE")
    os.environ["OPTILAND_METAL_FUSED_TRACE"] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("OPTILAND_METAL_FUSED_TRACE", None)
        else:
            os.environ["OPTILAND_METAL_FUSED_TRACE"] = previous


def _metal_mode() -> str | None:
    """The active Metal representation, or None when the backend is not Metal.

    The probe is one tiny ``be.array``: the backend itself decides whether the
    result is a ``MetalFloat64``, so this never guesses from configuration.
    """
    probe = be.array(np.zeros(1, dtype=np.float64))
    if not _is_metal(probe):
        return None
    return str(probe.mode)


def _count(key: str, n: int = 1) -> None:
    """Count ``key`` through ``metal.tensor.count_event`` when Metal is present."""
    try:
        from optiland.backend.torch_backend.metal.tensor import count_event
    except ImportError:  # pragma: no cover - no torch / no Metal
        return
    count_event(key, n)


def _metal_stats() -> dict[str, int]:
    """``be.metal_stats()`` or an empty dict on a non-Metal backend."""
    try:
        return dict(be.metal_stats())
    except (AttributeError, RuntimeError):
        return {}


def _stats_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    """``after - before``, dropping unchanged keys."""
    delta: dict[str, int] = {}
    for key, value in after.items():
        change = int(value) - int(before.get(key, 0))
        if change:
            delta[key] = change
    return delta


def _clone_plane(a: Any) -> Any:
    """A private copy of one ray plane, component for component.

    A ``to_numpy`` / ``be.array`` round trip is NOT used: it would re-encode the
    value and can shift the df64 low word (day-1 P1).  Cloning the raw
    components keeps the copy bit-identical to the original.
    """
    if _is_metal(a):
        from optiland.backend.torch_backend.metal.tensor import wrap

        return wrap(tuple(c.clone() for c in a.components), a.mode)
    return be.copy(a)


def _clone_rays(rays: RealRays) -> RealRays:
    """A private copy of a launch bundle (``surfaces.trace`` mutates in place)."""
    clone = RealRays(
        _clone_plane(rays.x),
        _clone_plane(rays.y),
        _clone_plane(rays.z),
        _clone_plane(rays.L),
        _clone_plane(rays.M),
        _clone_plane(rays.N),
        _clone_plane(rays.i),
        _clone_plane(rays.w),
    )
    clone.opd = _clone_plane(rays.opd)
    return clone


def _select(a: Any, *index: int) -> Any:
    """``a[index]`` as a zero-copy view, for Metal and for backend arrays."""
    if a is None:
        return None
    if _is_metal(a):
        from optiland.backend.torch_backend.metal.tensor import wrap

        return wrap(tuple(c[index] for c in a.components), a.mode)
    return a[index]


def _row_slice(a: Any, row: int) -> Any:
    """``a[:, row, :]`` as a zero-copy view (``(B, n_rows, N) -> (B, N)``)."""
    if _is_metal(a):
        from optiland.backend.torch_backend.metal.tensor import wrap

        return wrap(tuple(c[:, row, :] for c in a.components), a.mode)
    return a[:, row, :]


def _to_numpy(a: Any) -> Any:
    """Decode any backend array (or torch tensor) to NumPy."""
    if a is None:
        return None
    if _is_metal(a):
        return a.to_numpy()
    if isinstance(a, np.ndarray):
        return a
    detach = getattr(a, "detach", None)
    if detach is not None:
        return detach().cpu().numpy()
    return np.asarray(a)


def _stack_designs(planes: list[Any]) -> Any:
    """Stack one per-design plane list into a leading design axis.

    For ``MetalFloat64`` inputs the stack happens on the raw components, so the
    result is bit-identical to the inputs (no decode / re-encode).
    """
    first = planes[0]
    if _is_metal(first):
        import torch

        from optiland.backend.torch_backend.metal.tensor import wrap

        ncomp = len(first.components)
        return wrap(
            tuple(
                torch.stack([p.components[k].contiguous() for p in planes])
                for k in range(ncomp)
            ),
            first.mode,
        )
    return be.stack(planes)


def _length(a: Any) -> int:
    """The number of rays in one plane."""
    return int(np.prod(tuple(a.shape))) if hasattr(a, "shape") else len(a)


# ---------------------------------------------------------------------------
# Values, variables and the contract mutation path
# ---------------------------------------------------------------------------
def _as_values(values: Any, n_vars: int) -> np.ndarray:
    """``values`` as a ``(B, n_vars)`` array, float64 when it can be.

    A ``material`` variable carries strings, so a float conversion is tried
    first and object dtype is the fallback; a mixed float / string table never
    silently becomes a string table.
    """
    try:
        table = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        table = np.asarray(values, dtype=object)
    if table.ndim == 1:
        table = table.reshape(-1, 1)
    if table.ndim != 2:
        raise ValueError(f"values must be 2-D (B, n_vars), got shape {table.shape}")
    if table.shape[1] != n_vars:
        raise ValueError(
            f"values has {table.shape[1]} columns but {n_vars} variables were given"
        )
    if table.shape[0] == 0:
        raise ValueError("values must contain at least one design row")
    return table


def _apply_design(optic: Any, variables: Sequence[Any], row: Any) -> None:
    """The contract mutation: ``Variable.update`` then ``optic.updater.update()``.

    Day-1 Q4 measured that ``Variable.update`` alone does **not** apply pickups
    or solves, so the ``update()`` is unconditional: compiling a record for a
    system nobody would ever trace is worse than the cost of the call.
    """
    for variable, value in zip(variables, row, strict=True):
        variable.update(value)
    optic.updater.update()


def _current_values(variables: Sequence[Any]) -> list[Any]:
    """The variables' values now, in the scaled units ``update`` consumes.

    Deviation from design 6.2, stated: the originals are captured at entry
    rather than restored with ``Variable.reset()``.  ``reset()`` restores the
    variable's *construction-time* value, which would silently move a system a
    caller had already perturbed; capturing here coincides with ``reset()``
    whenever the system is nominal, which is the case the design describes.
    """
    return [variable.value for variable in variables]


def _restore(optic: Any, variables: Sequence[Any], originals: Sequence[Any]) -> None:
    """Put every variable back and run one final ``optic.updater.update()``."""
    for variable, value in zip(variables, originals, strict=True):
        variable.update(value)
    optic.updater.update()


def _surface_number(variable: Any) -> int | None:
    """The surface a variable is bound to, or None when it is not bound."""
    index = getattr(variable.variable, "surface_number", None)
    return None if index is None else int(index)


def _touched_rows(variable: Any, n_surfaces: int) -> set[int]:
    """The record rows one variable can change (design 6.2, day-1 Q5).

    * ``thickness``: the surface itself and everything downstream, because
      ``OpticUpdater.set_thickness`` rebuilds every downstream ``cs.z`` from the
      cumulative thicknesses.  Surface 0 is the exception: it only moves the
      object surface (``optic_updater.py:76-82``).
    * ``index`` / ``material``: the surface and its successor, because
      ``set_material`` replaces one object that both ``s.material_post`` and
      ``s+1.material_pre`` reference (day-1 Q5 finding b).
    * everything else: the surface itself.
    """
    index = _surface_number(variable)
    if index is None:
        return set(range(n_surfaces))
    if variable.type == "thickness":
        if index == 0:
            return {0}
        return set(range(min(index, n_surfaces), n_surfaces))
    if variable.type in ("index", "material"):
        return {i for i in (index, index + 1) if 0 <= i < n_surfaces}
    return {index} if 0 <= index < n_surfaces else set()


def _row_cache_enabled(row_cache: Any, optic: Any, variables: Sequence[Any]) -> bool:
    """Whether tier 1 (the row cache) applies (design 6.2).

    ``"auto"`` enables it only when the optic has neither pickups nor solves:
    both can move a surface no variable names, and tier 1 would not re-read it.
    """
    if row_cache in (False, 0, "off", "none"):
        return False
    if any(_surface_number(v) is None for v in variables):
        return False
    if row_cache is True:
        return True
    if row_cache not in ("auto", None):
        raise ValueError(f"unknown row_cache policy: {row_cache!r}")
    return len(optic.pickups) == 0 and len(optic.solves) == 0


# ---------------------------------------------------------------------------
# Launch generation and the shared-launch rule
# ---------------------------------------------------------------------------
def _generate_bundle(
    optic: Any,
    Hx: Any,
    Hy: Any,
    wavelength: float,
    num_rays: int | None,
    distribution: Any,
) -> RealRays:
    """One launch set, in ``RealRayTracer.trace``'s order.

    Mirrors ``real_ray_tracer.py:100-128`` exactly, so the ray order inside
    ``N`` is field-major / pupil-minor and every downstream indexing convention
    holds (design 6.1).
    """
    if isinstance(distribution, str):
        dist = create_distribution(distribution)
        dist.generate_points(num_rays)
    else:
        dist = distribution
    px, py = dist.x, dist.y
    with be.no_grad_unless_enabled():
        hx = be.atleast_1d(Hx)
        hy = be.atleast_1d(Hy)
        num_fields = len(hx)
        num_pupil = len(px)
        return optic.ray_tracer.ray_generator.generate_rays(
            be.repeat(hx, num_pupil),
            be.repeat(hy, num_pupil),
            be.tile(px, num_fields),
            be.tile(py, num_fields),
            wavelength,
        )


#: Variable types that move the launch set even downstream of the stop.
#:
#: Deviation from design 6.1 condition 3, **measured**: a post-stop thickness
#: perturbation of a CookeTriplet moved ``optic.paraxial.EPL()`` by 3 ulp
#: (11.512158673746798 vs ...795) for one design of five, which moved the sf64
#: launch ``z`` of all 1,141 rays by 1 ulp and put the whole trace off the
#: contract loop.  Note 06's own dependency table (section 2.4) already lists
#: ``positions[...]`` as changing "for any pose/thickness change"; its section-4
#: condition list was the optimistic reading.  Sharing is an optimisation and
#: never the contract, so thickness simply does not qualify.
_LAUNCH_SENSITIVE_TYPES: frozenset[str] = frozenset({"thickness"})


def _shared_launch_reason(optic: Any, variables: Sequence[Any]) -> str | None:
    """None when one launch set is bit-identical for every design.

    The six conditions of note 06 section 4, in order.  Sharing is an
    optimisation and never the contract: when any of them fails, ``trace_batch``
    regenerates the launch set per design exactly as scipy's sequential ``_fun``
    calls do today.
    """
    config = getattr(optic.ray_tracer, "ray_aiming_config", {})
    mode = config.get("mode", "paraxial")
    if mode != "paraxial":
        return f"ray aiming mode is {mode!r}, not 'paraxial'"

    aperture = type(optic.aperture).__name__
    if aperture not in ("EPDAperture", "FloatByStopAperture"):
        return f"aperture {aperture} is a function of the whole prescription"

    field_type = type(optic.fields.field_definition).__name__
    if field_type not in ("AngleField", "ObjectHeightField"):
        return f"field type {field_type} is solved per call"

    stop = int(optic.surfaces.stop_index)
    # FloatByStop reads the stop's own semi-aperture, so a variable ON the stop
    # moves the EPD; an EPD aperture is a stored constant and tolerates it.
    first_free = stop + 1 if aperture == "FloatByStopAperture" else stop
    for variable in variables:
        index = _surface_number(variable)
        if index is None:
            return f"variable {variable.type!r} is not bound to a surface"
        if index < first_free:
            return (
                f"variable {variable.type!r} on surface {index} is at or before "
                f"the stop (index {stop}) and moves the launch set"
            )
        if variable.type in _LAUNCH_SENSITIVE_TYPES:
            return (
                f"variable {variable.type!r} changes surfaces.positions, and the "
                "paraxial quantities derived from it are not bit-invariant"
            )
    return None


# ---------------------------------------------------------------------------
# The feature gate for the batch path
# ---------------------------------------------------------------------------
def _feature_gate(optic: Any, w0: float, mode: str) -> Any:
    """Run ``can_fuse_trace``'s feature checks without its ray checks.

    Design 6.3: the batch path builds its own launch buffers, so the gate's
    bundle checks (host residency, ray count, dtype) do not apply -- a
    ``1e4 x 1e2`` batch is a legitimate fused workload even though a standalone
    100-ray bundle is host-resident and refused at the hook.  Everything else
    the gate decides (surface types, geometries, apertures, interaction models,
    poses, Newton parameters, finite indices at ``w0``, grad) still applies, and
    the reason reported is the gate's own: this function runs the gate, it does
    not re-implement it.
    """
    from optiland.backend.torch_backend.metal import trace_record
    from optiland.backend.torch_backend.metal.tensor import HOST_THRESHOLD

    n = max(512, int(HOST_THRESHOLD) + 1)
    zeros = np.zeros(n, dtype=np.float64)
    ones = np.ones(n, dtype=np.float64)
    probe = RealRays(
        be.array(zeros),
        be.array(zeros),
        be.array(zeros),
        be.array(zeros),
        be.array(zeros),
        be.array(ones),
        be.array(ones),
        be.array(np.full(n, float(w0))),
    )
    if not _is_metal(probe.x) or str(probe.x.mode) != mode:  # pragma: no cover
        raise RuntimeError("the probe bundle did not land on the Metal backend")
    return trace_record.can_fuse_trace(optic.surfaces, probe, 0, wavelength=w0)


# ---------------------------------------------------------------------------
# Record compilation: tier 0 (the contract) and tier 1 (the row cache)
# ---------------------------------------------------------------------------
def _fill_row(group: Any, records: Any, b: int, index: int, ctx: dict) -> None:
    """Re-read surface ``index`` into design ``b``'s row (tier 1).

    The body is ``compile_records``' per-surface body (``trace_record.py``)
    against the same public adapter registries and the same private pose /
    aperture-capability helpers, so there is exactly one implementation of each
    piece.  The row is zeroed first because the adapters OR their flags in.
    The standing guard against this drifting from ``compile_records`` is the
    per-call canary below plus ``test_tier1_equals_tier0_bitwise``.
    """
    from optiland.backend.torch_backend.metal import trace_layout as L
    from optiland.backend.torch_backend.metal.trace_adapters import (
        APERTURE_ADAPTERS,
        GEOMETRY_ADAPTERS,
        INTERACTION_ADAPTERS,
    )
    from optiland.backend.torch_backend.metal.trace_record import (
        _accepts_aperture,
        _fill_pose,
    )

    surface = group.surfaces[index]
    row_int = records.surf_int[b, index]
    row_real = records.surf_real[b, index]
    row_coef = records.coef[b, index]
    row_int[:] = 0
    row_real[:] = 0.0
    row_coef[:] = 0.0

    if index == 0:
        row_int[L.SI_GEOM] = L.GEOM_OBJECT
        row_int[L.SI_SNAPROW] = int(records.snap_rows[b, 0])
        row_int[L.SI_STEPCOST] = 1
        return

    ctx = dict(ctx, surface=surface, index=index)
    geometry = surface.geometry
    _fill_pose(geometry.cs, row_int, row_real)
    GEOMETRY_ADAPTERS[type(geometry)].fill(geometry, row_int, row_real, row_coef, ctx)
    aperture = surface.aperture
    if aperture is not None:
        row_int[L.SI_FLAGS] |= L.FL_HAS_APERTURE
        APERTURE_ADAPTERS[type(aperture)].fill(
            aperture, row_int, row_real, row_coef, ctx
        )
        if _accepts_aperture(geometry) and not (row_int[L.SI_FLAGS] & L.FL_RADIUS_INF):
            row_int[L.SI_FLAGS] |= L.FL_AP_IN_ROOT
    INTERACTION_ADAPTERS[type(surface.interaction_model)].fill(
        surface.interaction_model, row_int, row_real, row_coef, ctx
    )
    row_int[L.SI_SNAPROW] = int(records.snap_rows[b, index])
    step_cost = 1
    if int(row_int[L.SI_GEOM]) in (L.GEOM_EVEN, L.GEOM_ODD):
        step_cost = 1 + int(row_int[L.SI_MAXITER])
    row_int[L.SI_STEPCOST] = step_cost


def _copy_design(records: Any, b: int, single: Any) -> None:
    """Copy a one-design ``TraceRecords`` into design ``b`` of ``records``."""
    if single.C != records.C:
        raise ValueError(
            f"design {b} compiles {single.C} coefficient slots but design 0 "
            f"compiles {records.C}; that is a structural change, not a "
            "tolerancing perturbation"
        )
    records.surf_int[b] = single.surf_int[0]
    records.surf_real[b] = single.surf_real[0]
    records.coef[b] = single.coef[0]


def _structural_check(records: Any) -> None:
    """Every design must share design 0's structure (design 6.2).

    Pose flags (``HAS_RX/RY/RZ``) and ``ABSORBING`` may vary; geometry code,
    aperture code, reflectivity, snapshot row, Newton iteration cap, coefficient
    count and step cost may not.
    """
    from optiland.backend.torch_backend.metal import trace_layout as L

    slots = (
        (L.SI_GEOM, "geometry code"),
        (L.SI_NCOEFF, "coefficient count"),
        (L.SI_MAXITER, "Newton max_iter"),
        (L.SI_APCODE, "aperture code"),
        (L.SI_SNAPROW, "snapshot row"),
        (L.SI_STEPCOST, "step cost"),
    )
    reference = records.surf_int[0]
    for slot, what in slots:
        bad = np.nonzero(records.surf_int[:, :, slot] != reference[:, slot])
        if bad[0].size:
            b, s = int(bad[0][0]), int(bad[1][0])
            raise ValueError(
                f"design {b} changes the {what} of surface {s} "
                f"({int(records.surf_int[b, s, slot])} vs "
                f"{int(reference[s, slot])}); trace_batch takes tolerancing "
                "perturbations, not structural edits"
            )
    flipped = np.nonzero(
        (records.surf_int[:, :, L.SI_FLAGS] ^ reference[:, L.SI_FLAGS])
        & L.FL_REFLECTIVE
    )
    if flipped[0].size:
        b, s = int(flipped[0][0]), int(flipped[1][0])
        raise ValueError(
            f"design {b} flips the reflectivity of surface {s}; trace_batch "
            "takes tolerancing perturbations, not structural edits"
        )


#: What one surface contributes to a design's structural signature.
_SIGNATURE_FIELDS = (
    "surface type",
    "geometry type",
    "aperture type",
    "reflectivity",
    "coefficient count",
    "radius finiteness",
)


def _structural_signature(group: Any) -> list[tuple]:
    """The structure of one design, readable on every backend.

    The fused path compares the compiled ``surf_int`` tables, which is sharper
    (it also sees the snapshot row and the Newton iteration cap).  The contract
    loop has no tables, so it compares this signature instead: ``trace_batch``
    raises the same ``ValueError`` on NumPy as on Metal.
    """
    signature = []
    for surface in group.surfaces:
        geometry = getattr(surface, "geometry", None)
        model = getattr(surface, "interaction_model", None)
        radius = getattr(geometry, "radius", None)
        signature.append(
            (
                type(surface).__name__,
                type(geometry).__name__,
                type(getattr(surface, "aperture", None)).__name__,
                bool(getattr(model, "is_reflective", False)),
                len(getattr(geometry, "coefficients", ()) or ()),
                # An infinite radius is a different kernel geometry code
                # (GEOM_STD_INF vs GEOM_CONIC), so it is structural.
                None if radius is None else bool(math.isfinite(float(radius))),
            )
        )
    return signature


def _check_signature(reference: list[tuple], other: list[tuple], b: int) -> None:
    """Raise when design ``b``'s structure differs from design 0's."""
    if len(reference) != len(other):
        raise ValueError(
            f"design {b} has {len(other)} surfaces but design 0 has "
            f"{len(reference)}; trace_batch takes tolerancing perturbations, "
            "not structural edits"
        )
    for s, (want, got) in enumerate(zip(reference, other, strict=True)):
        for field_name, a, c in zip(_SIGNATURE_FIELDS, want, got, strict=True):
            if a != c:
                raise ValueError(
                    f"design {b} changes the {field_name} of surface {s} "
                    f"({c!r} vs {a!r}); trace_batch takes tolerancing "
                    "perturbations, not structural edits"
                )


def _canary_designs(b_count: int) -> list[int]:
    """The designs a tier-1 call re-compiles through tier 0 (plan 3.9)."""
    designs = [b_count - 1]
    if b_count > 2:
        designs.append(random.randrange(b_count - 1))
    return sorted(set(designs))


def _tables_equal(records: Any, b: int, single: Any) -> bool:
    """Tier 1's design ``b`` against a tier-0 compile of the same design."""
    return (
        single.C == records.C
        and np.array_equal(records.surf_int[b], single.surf_int[0])
        and np.array_equal(records.surf_real[b], single.surf_real[0], equal_nan=True)
        and np.array_equal(records.coef[b], single.coef[0], equal_nan=True)
    )


@dataclass
class _Compiled:
    """The per-design tables and launch sets one fused batch needs."""

    records: Any
    bundles: list[RealRays]
    row_cache_used: bool
    canary_mismatch: bool
    trailing: list[tuple[Any, Any]]


def _compile_designs(
    optic: Any,
    variables: Sequence[Any],
    values: np.ndarray,
    *,
    w0: float,
    mode: str,
    record: Any,
    tier1: bool,
    shared: bool,
    bundles: list[RealRays] | None,
    launch_kwargs: dict,
) -> _Compiled:
    """Walk every design through the contract path and compile its tables."""
    from optiland.backend.torch_backend.metal.trace_record import compile_records

    group = optic.surfaces
    b_count = int(values.shape[0])
    generate = bundles is None
    bundles = [] if generate else list(bundles)
    trailing: list[tuple[Any, Any]] = []

    _apply_design(optic, variables, values[0])
    records = compile_records(group, w0, mode, record=record, designs=b_count)
    rows: set[int] = set()
    if tier1:
        for variable in variables:
            rows |= _touched_rows(variable, records.S)

    ctx = {"mode": mode, "w0": w0, "group": group}
    for b in range(b_count):
        if b:
            _apply_design(optic, variables, values[b])
            if tier1:
                for index in sorted(rows):
                    _fill_row(group, records, b, index, ctx)
            else:
                _copy_design(
                    records, b, compile_records(group, w0, mode, record=record)
                )
        last = group.surfaces[-1]
        trailing.append((last.material_post.propagation_model, last.thickness))
        if generate and (b == 0 or not shared):
            bundles.append(_generate_bundle(optic, **launch_kwargs))

    mismatch = False
    if tier1:
        for d in _canary_designs(b_count):
            _apply_design(optic, variables, values[d])
            if not _tables_equal(
                records, d, compile_records(group, w0, mode, record=record)
            ):
                mismatch = True
                break

    return _Compiled(records, bundles, tier1, mismatch, trailing)


# ---------------------------------------------------------------------------
# Launch packing
# ---------------------------------------------------------------------------
def _pack_launch(bundles: Sequence[RealRays], mode: str, n: int) -> tuple[Any, ...]:
    """Pack one or ``B`` bundles into the kernel's ``R[9][Lb][N]`` layout.

    The per-bundle packing is the driver's own ``_pack_launch`` (design 5), so
    the plane order and the validation are shared with the hook path; stacking
    the ``(9, N)` blocks along the design axis is the only thing added here.
    """
    import torch

    from optiland.backend.torch_backend.metal import trace as driver
    from optiland.backend.torch_backend.metal import trace_layout as L

    packed = [driver._pack_launch(rays, mode, n) for rays in bundles]
    if len(packed) == 1:
        return packed[0]
    ncomp = len(packed[0])
    return tuple(
        torch.stack([p[k].reshape(L.Q_PLANES, n) for p in packed], dim=1)
        .reshape(-1)
        .contiguous()
        for k in range(ncomp)
    )


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------
@dataclass
class BatchTraceResult:
    """What one ``trace_batch`` produced (design 6.4).

    Attributes:
        x, y, z, L, M, N, intensity, opd: ``(B, n_rows, N)`` recorded rows, or
            None when ``record`` asked for nothing.  On the fused path these
            are zero-copy views of the kernel's ``snap`` buffer.
        rows: surface index -> row index inside ``n_rows``.
        final: the eleven ``(B, N)`` final planes, or None when ``write_final``
            was False (the default whenever the image row is recorded, day-1
            Q10).
        status: ``(B, S, N)`` uint8 status planes (zeros on the fallback path).
        iters: ``(B, S, N)`` uint8 Newton iteration counts (zeros on the
            fallback path).
        launch_shared: whether one launch set was shared by every design.
        stats: the ``be.metal_stats()`` delta over the call.
        fused: whether the kernel produced the rows.
        row_cache_used: whether tier 1 compiled the tables.
        canary_mismatch: whether the tier-1 canary fired and forced tier 0.
        late_fallback_designs: ``bool[B]``; True where the design was re-traced
            on the per-op path after ``ST_TOL_CROSSOVER``.
    """

    x: Any = None
    y: Any = None
    z: Any = None
    L: Any = None
    M: Any = None
    N: Any = None
    intensity: Any = None
    opd: Any = None
    rows: dict[int, int] = field(default_factory=dict)
    final: dict[str, Any] | None = None
    status: Any = None
    iters: Any = None
    launch_shared: bool = False
    stats: dict[str, int] = field(default_factory=dict)
    fused: bool = False
    row_cache_used: bool = False
    canary_mismatch: bool = False
    late_fallback_designs: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=bool)
    )
    #: (propagation model, thickness) of the last surface, per design.
    trailing: list[tuple[Any, Any]] = field(default_factory=list)
    #: The canonical wavelength every design was traced at.
    w0: float = 0.0
    #: Number of designs and rays.
    B: int = 0
    n_rays: int = 0

    def rays(self, b: int) -> RealRays:
        """Design ``b``'s rays at the image, trailing propagation applied.

        The trailing step is ``RealRayTracer.trace``'s own
        (``real_ray_tracer.py:132-137``): ``SurfaceGroup.trace`` ends at the
        image surface, and the propagation through the last surface's thickness
        happens afterwards.  ``L0/M0/N0`` are None unless ``write_final`` was on
        (day-1 Q10).
        """
        if self.final is not None:
            planes = {a: _select(self.final[a], b) for a in _FINAL_ATTRS}
        else:
            row = self.rows.get(self._image_index())
            if row is None:
                raise ValueError(
                    "rays(b) needs either write_final=True or the image row in "
                    f"`record`; recorded rows are {sorted(self.rows)}"
                )
            planes = {
                "x": _select(self.x, b, row),
                "y": _select(self.y, b, row),
                "z": _select(self.z, b, row),
                "L": _select(self.L, b, row),
                "M": _select(self.M, b, row),
                "N": _select(self.N, b, row),
                "i": _select(self.intensity, b, row),
                "opd": _select(self.opd, b, row),
                "L0": None,
                "M0": None,
                "N0": None,
            }
        out = RealRays(
            planes["x"],
            planes["y"],
            planes["z"],
            planes["L"],
            planes["M"],
            planes["N"],
            planes["i"],
            be.array(np.full(self.n_rays, float(self.w0))),
        )
        out.opd = planes["opd"]
        out.L0, out.M0, out.N0 = planes["L0"], planes["M0"], planes["N0"]
        model, thickness = self.trailing[b]
        if model is not None:
            model.propagate(out, thickness)
        return out

    def _image_index(self) -> int:
        return max(self.rows) if self.rows else -1

    def install(self, optic: Any, b: int) -> None:
        """Write design ``b``'s recorded rows onto ``optic.surfaces``.

        Nothing else is touched: no method is patched, no ray is re-traced, and
        surfaces that were not recorded keep the empty arrays ``reset()`` gives
        them.  An analysis that reads ``optic.surfaces`` therefore sees design
        ``b``; one that re-traces gets a real trace of whatever the optic
        currently holds.
        """
        if self.x is None:
            raise ValueError("install() needs recorded rows; record was 'none'")
        optic.surfaces.reset()
        for index, row in self.rows.items():
            surface = optic.surfaces[index]
            for attr in _RESULT_ATTRS:
                setattr(surface, attr, _select(getattr(self, attr), b, row))

    def to_numpy(self) -> dict[str, Any]:
        """One decode of everything, as plain NumPy arrays."""
        out: dict[str, Any] = {
            attr: _to_numpy(getattr(self, attr)) for attr in _RESULT_ATTRS
        }
        out["rows"] = dict(self.rows)
        out["status"] = _to_numpy(self.status)
        out["iters"] = _to_numpy(self.iters)
        if self.final is not None:
            out["final"] = {k: _to_numpy(v) for k, v in self.final.items()}
        else:
            out["final"] = None
        return out

    def rms_spot(self, row: int = -1) -> Any:
        """RMS spot size per design, over one recorded row.

        Mirrors ``RayOperand.rms_spot_size``' expression
        (``operand/ray.py:365-372``): NaN-omitting means, centroid-relative,
        square root of the mean square radius.  ``row`` indexes ``n_rows``.
        """
        if self.x is None:
            raise ValueError("rms_spot() needs recorded rows; record was 'none'")
        n_rows = int(self.x.shape[1])
        index = row + n_rows if row < 0 else row
        xs = _row_slice(self.x, index)
        ys = _row_slice(self.y, index)
        mean_x = be.reshape(be.nanmean(xs, axis=1), (-1, 1))
        mean_y = be.reshape(be.nanmean(ys, axis=1), (-1, 1))
        r2 = (xs - mean_x) ** 2 + (ys - mean_y) ** 2
        return be.sqrt(be.nanmean(r2, axis=1))


# ---------------------------------------------------------------------------
# trace_batch
# ---------------------------------------------------------------------------
def trace_batch(
    optic: Any,
    variables: Sequence[Any],
    values: Any,
    *,
    Hx: Any,
    Hy: Any,
    wavelength: float,
    num_rays: int | None = 100,
    distribution: Any = "hexapolar",
    record: Any = "image",
    shared_launch: Any = "auto",
    write_final: bool | None = None,
    row_cache: Any = "auto",
    tol_floor_scale: float | None = None,
) -> BatchTraceResult:
    """Trace ``B`` designs of ``optic`` in one launch (design 6.1).

    Args:
        optic: the optical system; it is mutated design by design and restored
            before the call returns.
        variables: ``optiland.optimization.variable.Variable`` objects -- the
            same mutation mechanism tolerancing perturbations and every
            optimizer use.
        values: ``(B, len(variables))`` in each variable's **scaled** units,
            exactly what ``Variable.update`` receives from an optimizer.  A
            caller holding physical values converts with
            ``var.variable.scale(physical)``.
        Hx: normalized x field coordinate(s).
        Hy: normalized y field coordinate(s).
        wavelength: the wavelength every design is traced at.
        num_rays: the distribution's sampling parameter.
        distribution: name or ``BaseDistribution`` instance.
        record: ``"image"`` (default), ``"all"``/True, ``"none"``/False,
            ``"stop"``, or an explicit sequence of surface indices.
        shared_launch: ``"auto"`` (share only when note 06 section 4's
            conditions hold), True (force) or False (one launch set per design).
        write_final: write the eleven final planes.  Defaults to False when the
            image row is recorded and True otherwise (day-1 Q10).
        row_cache: ``"auto"`` (tier 1 when the optic has no pickups and no
            solves), True (force) or False (tier 0 for every design).
        tol_floor_scale: refused; see the module docstring.

    Returns:
        BatchTraceResult: the recorded rows, final planes, diagnostics planes
        and counter delta.

    Raises:
        ValueError: ``values`` has the wrong shape, ``record`` names a surface
            twice or out of range, or a design changes the system's structure.
        NotImplementedError: ``tol_floor_scale`` was given.
    """
    if tol_floor_scale is not None:
        raise NotImplementedError(
            "tol_floor_scale is not implemented: day-1 question Q3 settled the "
            "TOL_CROSSOVER policy in favour of the per-design per-op re-trace "
            "(design 6.3), and launch_trace's frozen signature carries no "
            "tolerance-floor argument"
        )
    variables = list(variables)
    table = _as_values(values, len(variables))
    # ``compile_records``'s vocabulary is True / False / "image" / "stop" /
    # sequence (plan 3.3); design 6.1 spells the first two "all" and "none".
    if record in ("all", True):
        record = True
    elif record in ("none", False, None):
        record = False
    launch_kwargs = {
        "Hx": Hx,
        "Hy": Hy,
        "wavelength": wavelength,
        "num_rays": num_rays,
        "distribution": distribution,
    }

    before = _metal_stats()
    originals = _current_values(variables)
    try:
        mode = None if _switch() == "0" else _metal_mode()
        result = None
        if mode is not None:
            result = _trace_batch_fused(
                optic,
                variables,
                table,
                mode=mode,
                record=record,
                shared_launch=shared_launch,
                write_final=write_final,
                row_cache=row_cache,
                launch_kwargs=launch_kwargs,
            )
        if result is None:
            result = _trace_batch_loop(
                optic,
                variables,
                table,
                record=record,
                write_final=write_final,
                launch_kwargs=launch_kwargs,
            )
    finally:
        _restore(optic, variables, originals)
    result.stats = _stats_delta(before, _metal_stats())
    return result


def _resolve_record_rows(records: Any) -> dict[int, int]:
    """surface index -> snapshot row, from the compiled table."""
    rows = {}
    for index, row in enumerate(np.asarray(records.snap_rows[0]).tolist()):
        if int(row) >= 0:
            rows[index] = int(row)
    return rows


def _resolve_write_final(
    write_final: bool | None, rows: dict[int, int], s: int
) -> bool:
    """Day-1 Q10: ``write_final`` defaults to False when the image row is recorded."""
    if write_final is not None:
        return bool(write_final)
    return (s - 1) not in rows


def _trace_batch_fused(
    optic: Any,
    variables: Sequence[Any],
    values: np.ndarray,
    *,
    mode: str,
    record: Any,
    shared_launch: Any,
    write_final: bool | None,
    row_cache: Any,
    launch_kwargs: dict,
) -> BatchTraceResult | None:
    """The fused path; None means "refused, run the contract loop"."""
    import torch

    from optiland.backend.torch_backend.metal import trace as driver
    from optiland.backend.torch_backend.metal import trace_record as tr

    b_count = int(values.shape[0])
    w0 = tr.canonical_w0(launch_kwargs["wavelength"], mode)
    require = _switch() == "require"

    gate = _feature_gate(optic, w0, mode)
    if gate.ok or not gate.structural:
        _count("fused_trace:candidates")
    if not gate.ok:
        _count(f"fused_trace_skip:{gate.reason.value}")
        if require and not gate.structural:
            raise driver.MetalFallbackError(
                "OPTILAND_METAL_FUSED_TRACE=require: trace_batch was refused "
                f"for the feature reason {gate.reason.value!r}"
            )
        return None

    drift = driver._drift_qualnames()
    if drift:
        driver._warn_once(
            "drift",
            driver.FusedTraceDriftWarning,
            "fused trace: mirrored Python source drifted from the fingerprint "
            f"table for {', '.join(drift)}; re-verify with "
            "`python -m optiland.backend.torch_backend.metal.trace_mirror "
            "--update --verified <qualname>=<note>`",
        )
        if os.environ.get("OPTILAND_METAL_FUSED_TRACE_DRIFT", "refuse") != "warn":
            _count("fused_trace_skip:mirror_drift")
            if require:
                raise driver.MetalFallbackError(
                    "OPTILAND_METAL_FUSED_TRACE=require: trace_batch was "
                    "refused for the feature reason 'mirror_drift'"
                )
            return None

    if shared_launch == "auto":
        shared = _shared_launch_reason(optic, variables) is None
    else:
        shared = bool(shared_launch)

    tier1 = _row_cache_enabled(row_cache, optic, variables)
    compiled = _compile_designs(
        optic,
        variables,
        values,
        w0=w0,
        mode=mode,
        record=record,
        tier1=tier1,
        shared=shared,
        bundles=None,
        launch_kwargs=launch_kwargs,
    )
    if compiled.canary_mismatch:
        _count("fused_trace:tier1_canary_mismatch")
        compiled = _compile_designs(
            optic,
            variables,
            values,
            w0=w0,
            mode=mode,
            record=record,
            tier1=False,
            shared=shared,
            bundles=compiled.bundles,
            launch_kwargs=launch_kwargs,
        )
        compiled.canary_mismatch = True

    records = compiled.records
    _structural_check(records)

    rows = _resolve_record_rows(records)
    write_final = _resolve_write_final(write_final, rows, int(records.S))
    n = _length(compiled.bundles[0].x)

    fraction = float(
        os.environ.get(
            "OPTILAND_METAL_FUSED_TRACE_MEMORY_FRACTION", tr.DEFAULT_MEMORY_FRACTION
        )
    )
    budget = fraction * float(torch.mps.recommended_max_memory())
    if tr.memory_bytes(records, n, 1 if shared else b_count, write_final) > budget:
        _count("fused_trace_skip:memory")
        if require:
            raise driver.MetalFallbackError(
                "OPTILAND_METAL_FUSED_TRACE=require: trace_batch was refused "
                "for the feature reason 'memory'"
            )
        return None

    launch = _pack_launch(compiled.bundles, mode, n)
    result = driver.launch_trace(
        records,
        launch,
        launch_stride=0 if shared else 1,
        N=n,
        write_final=write_final,
        mode=mode,
    )

    late = np.asarray(result.late_fallback_designs, dtype=bool)
    if late.any():
        _count("fused_trace:late_fallback", int(late.sum()))
        _retrace_designs(
            optic,
            variables,
            values,
            np.nonzero(late)[0],
            compiled=compiled,
            result=result,
            rows=rows,
            shared=shared,
            write_final=write_final,
        )

    _count("fused_trace:traces")
    _count("fused_trace:designs", b_count)
    _count("fused_trace:surface_steps", b_count * n * (int(records.S) - 1))

    out = BatchTraceResult(
        rows=rows,
        status=result.status,
        iters=result.iters,
        launch_shared=shared,
        fused=True,
        row_cache_used=compiled.row_cache_used,
        canary_mismatch=compiled.canary_mismatch,
        late_fallback_designs=late,
        trailing=compiled.trailing,
        w0=w0,
        B=b_count,
        n_rays=n,
    )
    if result.snap is not None:
        from optiland.backend.torch_backend.metal.tensor import wrap

        for q, attr in enumerate(_RESULT_ATTRS):
            setattr(out, attr, wrap(tuple(c[q] for c in result.snap), mode))
    if result.final is not None:
        from optiland.backend.torch_backend.metal.tensor import wrap

        out.final = {
            attr: wrap(tuple(c[q] for c in result.final), mode)
            for q, attr in enumerate(_FINAL_ATTRS)
        }
    return out


def _retrace_designs(
    optic: Any,
    variables: Sequence[Any],
    values: np.ndarray,
    designs: Any,
    *,
    compiled: _Compiled,
    result: Any,
    rows: dict[int, int],
    shared: bool,
    write_final: bool,
) -> None:
    """Re-trace the ``TOL_CROSSOVER`` designs on the per-op path (design 6.3).

    Each affected design is traced through an ``Optic.from_dict`` copy, so the
    caller's optic keeps whatever the batch left on it and no other design is
    disturbed; the results overwrite that design's slice of the kernel buffers,
    so the returned ``BatchTraceResult`` stays one uniform object.
    """
    from optiland.optic import Optic

    for b in (int(d) for d in designs):
        _apply_design(optic, variables, values[b])
        copy = Optic.from_dict(optic.to_dict())
        rays = _clone_rays(compiled.bundles[0 if shared else b])
        with _fused_disabled():
            copy.surfaces.trace(rays, record=True)
        if result.snap is not None:
            for index, row in rows.items():
                surface = copy.surfaces[index]
                for q, attr in enumerate(_RESULT_ATTRS):
                    source = getattr(surface, attr)
                    for k, comp in enumerate(source.components):
                        result.snap[k][q, b, row].copy_(comp.reshape(-1))
        if write_final and result.final is not None:
            for q, attr in enumerate(_FINAL_ATTRS):
                source = getattr(rays, attr, None)
                if source is None:
                    continue
                for k, comp in enumerate(source.components):
                    result.final[k][q, b].copy_(comp.reshape(-1))


def _trace_batch_loop(
    optic: Any,
    variables: Sequence[Any],
    values: np.ndarray,
    *,
    record: Any,
    write_final: bool | None,
    launch_kwargs: dict,
) -> BatchTraceResult:
    """The contract loop: one ``optic.trace`` per design, on any backend.

    This is both the universal fallback (NumPy, torch-CPU, kill switch, gate
    refusal) and the reference the batch conformance tests compare against
    (design 6.5).
    """
    group = optic.surfaces
    b_count = int(values.shape[0])
    s = len(group.surfaces)
    rows = _loop_rows(group, record, s)
    write_final = _resolve_write_final(write_final, rows, s)

    per_design: dict[str, list[Any]] = {attr: [] for attr in _RESULT_ATTRS}
    finals: dict[str, list[Any]] = {attr: [] for attr in _FINAL_ATTRS}
    trailing: list[tuple[Any, Any]] = []
    n = 0
    signature: list[tuple] | None = None
    for b in range(b_count):
        _apply_design(optic, variables, values[b])
        if signature is None:
            signature = _structural_signature(group)
        else:
            _check_signature(signature, _structural_signature(group), b)
        rays = _generate_bundle(optic, **launch_kwargs)
        n = _length(rays.x)
        group.trace(rays, record=True)
        for index, row in sorted(rows.items(), key=lambda kv: kv[1]):
            del row
            surface = group.surfaces[index]
            for attr in _RESULT_ATTRS:
                per_design[attr].append(_clone_plane(getattr(surface, attr)))
        for attr in _FINAL_ATTRS:
            plane = getattr(rays, attr, None)
            finals[attr].append(None if plane is None else _clone_plane(plane))
        last = group.surfaces[-1]
        trailing.append((last.material_post.propagation_model, last.thickness))

    out = BatchTraceResult(
        rows=rows,
        status=np.zeros((b_count, s, n), dtype=np.uint8),
        iters=np.zeros((b_count, s, n), dtype=np.uint8),
        launch_shared=False,
        fused=False,
        row_cache_used=False,
        late_fallback_designs=np.zeros(b_count, dtype=bool),
        trailing=trailing,
        w0=_loop_w0(launch_kwargs["wavelength"]),
        B=b_count,
        n_rays=n,
    )
    n_rows = len(rows)
    if n_rows:
        for attr in _RESULT_ATTRS:
            planes = per_design[attr]
            stacked = _stack_designs(planes)
            setattr(out, attr, be.reshape(stacked, (b_count, n_rows, n)))
    if write_final and all(p is not None for p in finals["x"]):
        out.final = {}
        for attr in _FINAL_ATTRS:
            planes = finals[attr]
            if any(p is None for p in planes):
                out.final = None
                break
            out.final[attr] = _stack_designs(planes)
    return out


def _loop_w0(wavelength: float) -> float:
    """The canonical wavelength, when Metal is present; the raw value otherwise."""
    mode = _metal_mode()
    if mode is None:
        return float(wavelength)
    from optiland.backend.torch_backend.metal.trace_record import canonical_w0

    return canonical_w0(wavelength, mode)


def _loop_rows(group: Any, record: Any, s: int) -> dict[int, int]:
    """Resolve the ``record`` policy without the Metal record compiler."""
    if record in ("none", None, False):
        return {}
    if record in ("all", True):
        wanted = list(range(s))
    elif record == "image":
        wanted = [s - 1]
    elif record == "stop":
        wanted = [int(group.stop_index)]
    elif isinstance(record, str):
        raise ValueError(f"unknown record policy: {record!r}")
    else:
        wanted = [int(i) for i in record]
    rows: dict[int, int] = {}
    for row, index in enumerate(wanted):
        if not 0 <= index < s:
            raise ValueError(f"record index {index} is outside 0..{s - 1}")
        if index in rows:
            raise ValueError(f"record index {index} is listed twice")
        rows[index] = row
    return rows


# ---------------------------------------------------------------------------
# Finite differences (design 6.5)
# ---------------------------------------------------------------------------
@dataclass
class FDResult:
    """One batched finite-difference Jacobian.

    Attributes:
        jacobian: ``(n_vars,)`` float64 derivatives of ``merit`` with respect to
            each variable, in scaled units.
        merits: the merit value of every design row, float64.
        steps: the ACTUAL stencil width per variable after bound clipping --
            the denominator that was used, not the requested ``step``.
        clipped: True where a bound clipped one side of the stencil.
        base: the merit at the unperturbed design (forward mode only; NaN for
            central differences).
        central: whether central differences were used.
        result: the underlying ``BatchTraceResult``.
    """

    jacobian: np.ndarray
    merits: np.ndarray
    steps: np.ndarray
    clipped: np.ndarray
    base: float
    central: bool
    result: BatchTraceResult


def _clip(value: float, variable: Any) -> float:
    """Clip a scaled value into the variable's scaled bounds."""
    low, high = variable.bounds
    if low is not None and value < low:
        return float(low)
    if high is not None and value > high:
        return float(high)
    return float(value)


def _fd_rows(
    variables: Sequence[Any], step: Any, central: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The design table for a finite-difference Jacobian.

    Central: ``2n`` rows, minus/plus interleaved per variable, so both members
    of a pair come from the same launch.  Forward: the base row followed by
    ``n`` perturbed rows.
    """
    base = np.array([float(v.value) for v in variables], dtype=np.float64)
    n = len(variables)
    steps = (
        np.full(n, float(step), dtype=np.float64)
        if np.isscalar(step)
        else (np.asarray(step, dtype=np.float64))
    )
    if steps.shape != (n,):
        raise ValueError(f"step must be scalar or ({n},), got shape {steps.shape}")

    rows = np.repeat(base[None, :], 2 * n if central else n + 1, axis=0)
    actual = np.zeros(n, dtype=np.float64)
    clipped = np.zeros(n, dtype=bool)
    for j, variable in enumerate(variables):
        if central:
            minus = _clip(base[j] - steps[j], variable)
            plus = _clip(base[j] + steps[j], variable)
            rows[2 * j, j] = minus
            rows[2 * j + 1, j] = plus
        else:
            minus = base[j]
            plus = _clip(base[j] + steps[j], variable)
            rows[j + 1, j] = plus
        actual[j] = plus - minus
        wanted_plus = base[j] + steps[j]
        wanted_minus = base[j] - steps[j] if central else base[j]
        clipped[j] = bool(plus != wanted_plus or minus != wanted_minus)
    return rows, actual, clipped


def fd_jacobian(
    optic: Any,
    variables: Sequence[Any],
    merit: Callable[[BatchTraceResult, int], Any],
    step: Any,
    *,
    central: bool = True,
    **trace_kwargs: Any,
) -> FDResult:
    """A batched finite-difference Jacobian of ``merit`` (design 6.5).

    Every stencil point is one design of a single ``trace_batch`` call, so both
    members of a central pair come from the same launch.  The denominator is
    the ACTUAL clipped stencil width, never the requested step, and the merit
    values are decoded to float64 before they are subtracted.

    Args:
        optic: the optical system.
        variables: the variables to differentiate against.
        merit: ``merit(result, b) -> float``; anything a backend array can be
            reduced to, decoded here.
        step: scalar or ``(n_vars,)``, in the variables' scaled units.
        central: central differences (the default) or forward.
        **trace_kwargs: passed to ``trace_batch`` (``Hx``, ``Hy``,
            ``wavelength``, ``num_rays``, ...).

    Returns:
        FDResult: the derivatives and the stencil that produced them.
    """
    variables = list(variables)
    rows, actual, clipped = _fd_rows(variables, step, central)
    result = trace_batch(optic, variables, rows, **trace_kwargs)
    merits = np.array(
        [float(_to_numpy(merit(result, b))) for b in range(rows.shape[0])],
        dtype=np.float64,
    )
    n = len(variables)
    jacobian = np.zeros(n, dtype=np.float64)
    for j in range(n):
        if actual[j] == 0.0:
            jacobian[j] = np.nan
            continue
        if central:
            jacobian[j] = (merits[2 * j + 1] - merits[2 * j]) / actual[j]
        else:
            jacobian[j] = (merits[j + 1] - merits[0]) / actual[j]
    return FDResult(
        jacobian=jacobian,
        merits=merits,
        steps=actual,
        clipped=clipped,
        base=float(merits[0]) if not central else float("nan"),
        central=central,
        result=result,
    )


def fd_reference_cpu(
    optic: Any,
    variables: Sequence[Any],
    merit: Callable[[BatchTraceResult, int], Any],
    step_small: Any,
    **trace_kwargs: Any,
) -> np.ndarray:
    """A small-step central-difference reference on the NumPy backend.

    The point is to separate truncation error from GPU representation noise
    (design 6.5): this runs the same designs, through the same contract loop,
    on a NumPy copy of the system, at a step the caller chooses to be small
    enough that truncation dominates nothing.

    Args:
        optic: the optical system (never mutated; a dict copy is traced).
        variables: the variables to differentiate against.
        merit: ``merit(result, b) -> float``.
        step_small: scalar or ``(n_vars,)``, in scaled units.
        **trace_kwargs: passed to ``trace_batch``.

    Returns:
        numpy.ndarray: ``(n_vars,)`` float64 derivatives.
    """
    from optiland.optic import Optic
    from optiland.optimization.variable import Variable

    variables = list(variables)
    scaled = [v.value for v in variables]
    data = optic.to_dict()

    previous = be.get_backend()
    be.set_backend("numpy")
    try:
        cpu_optic = Optic.from_dict(data)
        cpu_vars = [
            Variable(
                cpu_optic,
                v.type,
                min_val=v.min_val,
                max_val=v.max_val,
                scaler=v.scaler,
                **v.kwargs,
            )
            for v in variables
        ]
        for variable, value in zip(cpu_vars, scaled, strict=True):
            variable.update(value)
        cpu_optic.updater.update()
        return fd_jacobian(
            cpu_optic, cpu_vars, merit, step_small, central=True, **trace_kwargs
        ).jacobian
    finally:
        be.set_backend(previous)
