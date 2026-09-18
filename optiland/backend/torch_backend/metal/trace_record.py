"""Eligibility gate and record compiler for the fused trace kernel.

Two pure host-side pieces sit between a :class:`SurfaceGroup` and the Metal
kernel:

``can_fuse_trace``
    the ordered predicate of the implementation plan's design 2.2.  Every free
    check runs before the single device readback that yields the bundle's
    canonical wavelength and its uniformity verdict.  The result is a
    :class:`GateResult`; a refusal names a closed-set :class:`FusedTraceSkip`
    reason and says whether the reason is *structural* (the bundle could never
    be a kernel candidate) or a *feature* the kernel does not implement yet.

``compile_records``
    the per-surface table builder.  Every real-valued slot is produced by the
    same host float64 op the per-op path runs -- ``be.cos`` on the live pose
    tensor, never ``math.cos`` on a Python float -- so the kernel mirrors the
    Python expression instead of approximating it (plan 0.2.1, design 3.4).

Deviation from design 2.2 recorded at WP2: the ``nonfinite_index`` check needs
``w0``, so it runs *after* the wavelength readback rather than before it.  Every
other reason keeps the design's order.  When the caller supplies ``wavelength=``
there is no readback and the order is the design's exactly.

Two checks were added by the round-1 fix lane, both free and both refusing a
configuration the kernel cannot mirror rather than tracing it (plan 1.2): a
bundle with ``rays.is_normalized`` clear (``propagation_model``, R1-V1-04) and
a df64 asphere coefficient stored as a backend tensor (``geometry_type``,
R1-V1-05).  Both are documented at the line they run on.

The round-3 fix lane added a third, for the same reason: the pose is
accepted only when ``type(geometry.cs) is CoordinateSystem``
(``reference_cs``, R3-V2-03).  A source digest cannot see a subclass that
overrides a mirrored method, so every mirrored family is keyed by exact
type; the pose was the one that was not.

Nothing here launches a kernel or allocates a device buffer.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.propagation.homogeneous import HomogeneousPropagation
from optiland.rays.real_rays import RealRays
from optiland.surfaces.image_surface import ImageSurface
from optiland.surfaces.object_surface import ObjectSurface
from optiland.surfaces.standard_surface import Surface
from optiland.surfaces.surface_group import SurfaceGroup

from . import encode
from . import trace_layout as L
from .trace_adapters import (
    APERTURE_ADAPTERS,
    GEOMETRY_ADAPTERS,
    INTERACTION_ADAPTERS,
    STRUCTURAL_REASONS,
    FusedTraceSkip,
    is_backend_scalar,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

__all__ = [
    "GateResult",
    "TraceRecords",
    "can_fuse_trace",
    "canonical_w0",
    "compile_records",
    "memory_bytes",
]

#: The nine ray planes the kernel launches with, in ``Q_*`` order.
RAY_ATTRS = ("x", "y", "z", "L", "M", "N", "i", "opd", "w")

#: Default allocation budget as a fraction of ``recommended_max_memory()``.
DEFAULT_MEMORY_FRACTION = 0.25

#: Largest ``max_iter`` the ``iters`` plane can carry: 255 collides with
#: ``ITERS_UNWRITTEN`` (plan 3.2), so the gate refuses it.
MAX_NEWTON_ITER = L.ITERS_UNWRITTEN - 1


def _count_event(key: str, n: int = 1) -> None:
    """Count ``key`` through ``tensor.count_event`` when WP3 has landed it.

    WP3 owns ``metal/tensor.py``; until its one-line ``count_event`` exists this
    writes the same counter directly.  Both paths mutate the same ``_STATS``
    counter that ``be.metal_stats()`` reports.
    """
    from . import tensor as _tensor

    fn = getattr(_tensor, "count_event", None)
    if fn is not None:
        fn(key, n)
    else:  # pragma: no cover - exercised until WP3 lands count_event
        _tensor._STATS[key] += n


# ---------------------------------------------------------------------------
# Canonical wavelength (plan 3.8)
# ---------------------------------------------------------------------------
def canonical_w0(wavelength: float, mode: str) -> float:
    """The wavelength a bundle generated at ``wavelength`` actually carries.

    In ``df64`` a wavelength is stored as a (hi, lo) float32 pair, so the value
    the rays carry is ``decode(encode(w))`` -- 14 of the 15 wavelengths used by
    the shipped samples are not df64-representable (day-1 probe P12).  In
    ``sf64`` the encoding is the IEEE binary64 bit pattern and the round trip is
    the identity.

    Args:
        wavelength: the requested wavelength in microns.
        mode: ``'df64'`` or ``'sf64'``.

    Returns:
        float: the canonical wavelength for ``mode``.
    """
    if mode == "sf64":
        return float(wavelength)
    if mode != "df64":
        raise ValueError(f"unknown metal mode: {mode!r}")
    hi, lo = encode.encode_df64(np.asarray(float(wavelength), dtype=np.float64))
    return float(np.ravel(encode.decode_df64(hi, lo))[0])


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------
@dataclass
class GateResult:
    """Verdict of :func:`can_fuse_trace`.

    Attributes:
        ok: True when the kernel may take this trace.
        reason: the refusal reason, or None when ``ok``.
        structural: True when ``reason`` is structural (never raises under
            ``OPTILAND_METAL_FUSED_TRACE=require``).
        mode: the representation of the ray tensors, when it could be read.
        w0: the canonical wavelength of the bundle, when it could be read.
        n: number of rays (0 when it could not be read).
        s: number of surfaces in the group (0 when it could not be read).
    """

    ok: bool
    reason: FusedTraceSkip | None
    structural: bool
    mode: str | None
    w0: float | None
    n: int
    s: int


def _refuse(reason: FusedTraceSkip, *, mode=None, n=0, s=0) -> GateResult:
    return GateResult(
        ok=False,
        reason=reason,
        structural=reason in STRUCTURAL_REASONS,
        mode=mode,
        w0=None,
        n=n,
        s=s,
    )


def _is_metal(t: Any) -> bool:
    return type(t).__name__ == "MetalFloat64"


def _scalar_float(value: Any) -> float | None:
    """``float(value)`` for a plain number or a 0-d backend array, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    ndim = getattr(value, "ndim", None)
    if ndim is None or ndim != 0:
        return None
    try:
        return float(value)
    except (TypeError, ValueError, RuntimeError):  # pragma: no cover - defensive
        return None


def _requires_grad(value: Any) -> bool:
    return bool(getattr(value, "requires_grad", False))


def _grad_tensors(group: SurfaceGroup) -> list[Any]:
    """Every tensor the adapters read, for the ``requires_grad`` check."""
    out: list[Any] = []
    for surface in group.surfaces[1:]:
        geometry = getattr(surface, "geometry", None)
        cs = getattr(geometry, "cs", None)
        if cs is not None:
            out.extend((cs.x, cs.y, cs.z, cs.rx, cs.ry, cs.rz))
        adapter = GEOMETRY_ADAPTERS.get(type(geometry))
        if adapter is not None:
            out.extend(adapter.grad_params(geometry))
        aperture = getattr(surface, "aperture", None)
        ap_adapter = APERTURE_ADAPTERS.get(type(aperture))
        if ap_adapter is not None:
            out.extend(ap_adapter.grad_params(aperture))
        model = getattr(surface, "interaction_model", None)
        im_adapter = INTERACTION_ADAPTERS.get(type(model))
        if im_adapter is not None:
            out.extend(im_adapter.grad_params(model))
    return out


def _uniform_wavelength(w: Any, mode: str) -> tuple[bool, float]:
    """One device->host copy: the bundle's ``w0`` and its uniformity verdict.

    Reduces on the component tensors (plain torch tensors, which never touch
    ``be.metal_stats()``) and issues a single ``.cpu()`` on the small probe
    (day-1 probe P10).
    """
    import torch

    comps = w._comps
    if mode == "df64":
        hi, lo = comps
        hi_flat = hi.reshape(-1)
        lo_flat = lo.reshape(-1)
        probe = torch.stack(
            (
                (hi_flat == hi_flat[0]).all().to(torch.float32),
                (lo_flat == lo_flat[0]).all().to(torch.float32),
                torch.isnan(hi_flat).any().to(torch.float32),
                hi_flat[0],
                lo_flat[0],
                torch.zeros((), dtype=torch.float32, device=hi.device),
            )
        )
        host = probe.cpu().numpy()
        uniform = bool(host[0]) and bool(host[1]) and not bool(host[2])
        w0 = float(
            np.ravel(encode.decode_df64(np.float32(host[3]), np.float32(host[4])))[0]
        )
    else:
        (bits,) = comps
        flat = bits.reshape(-1)
        probe = torch.stack(
            (
                (flat == flat[0]).all().to(torch.int64),
                flat[0],
                torch.zeros((), dtype=torch.int64, device=bits.device),
            )
        )
        host = probe.cpu().numpy()
        uniform = bool(host[0])
        w0 = float(np.ravel(encode.decode_sf64(np.asarray(host[1]).reshape(1)))[0])
    # An all-NaN sf64 bundle has a *uniform* bit pattern; P10 requires the
    # explicit finiteness test, and df64 gets it for free.
    return (uniform and math.isfinite(w0)), w0


def _memory_budget() -> float:
    try:
        import torch

        total = float(torch.mps.recommended_max_memory())
    except Exception:  # pragma: no cover - no Metal device
        return math.inf
    fraction = float(
        os.environ.get(
            "OPTILAND_METAL_FUSED_TRACE_MEMORY_FRACTION", DEFAULT_MEMORY_FRACTION
        )
    )
    return fraction * total


def _bytes(
    *,
    designs: int,
    surfaces: int,
    coefficients: int,
    n_rows: int,
    n_rays: int,
    launch_designs: int,
    write_final: bool,
) -> int:
    """Plan 3.6, verbatim: launch + snap + final + status/iters + tables."""
    b, s, c, n = designs, surfaces, coefficients, n_rays
    total = L.Q_PLANES * launch_designs * n * 8
    total += L.S_PLANES * b * n_rows * n * 8
    if write_final:
        total += L.F_PLANES * b * n * 8
    total += 2 * b * s * n
    total += b * s * (L.SR_STRIDE * 8 + L.SI_STRIDE * 4 + c * 8)
    return int(total)


def can_fuse_trace(
    group: Any,
    rays: Any,
    skip: int,
    *,
    wavelength: float | None = None,
) -> GateResult:
    """Decide whether the fused kernel may take this trace.

    Pure: nothing is mutated and no kernel is launched.  Exactly one device
    readback happens (counted as ``fused_trace:readback``) and only when
    ``wavelength`` is None.

    Args:
        group: the surface group ``SurfaceGroup.trace`` was called on.
        rays: the ray bundle.
        skip: the ``skip`` argument of ``SurfaceGroup.trace``.
        wavelength: the canonical wavelength, when the caller already knows it
            (the batch API does); None asks for the readback.

    Returns:
        GateResult: the verdict.
    """
    # -- whole-trace structural checks (design 2.2, steps 1-8) -------------
    if skip != 0:
        return _refuse(FusedTraceSkip.SKIP)
    if type(group) is not SurfaceGroup:
        return _refuse(FusedTraceSkip.GROUP_TYPE)
    if type(rays) is not RealRays:
        return _refuse(FusedTraceSkip.RAYS_TYPE)

    planes = [getattr(rays, name, None) for name in RAY_ATTRS]
    if not all(_is_metal(t) for t in planes):
        return _refuse(FusedTraceSkip.RAYS_SHAPE)
    modes = {t._mode for t in planes}
    shapes = {tuple(t.shape) for t in planes}
    if len(modes) != 1 or len(shapes) != 1:
        return _refuse(FusedTraceSkip.RAYS_SHAPE)
    mode = modes.pop()
    shape = shapes.pop()
    if len(shape) != 1 or shape[0] <= 0:
        return _refuse(FusedTraceSkip.RAYS_SHAPE, mode=mode)
    n = int(shape[0])

    from .tensor import HOST_THRESHOLD

    if n <= HOST_THRESHOLD or any(t.is_host_resident for t in planes):
        return _refuse(FusedTraceSkip.HOST_RESIDENT, mode=mode, n=n)

    import torch

    if torch.is_grad_enabled() and (
        any(_requires_grad(t) for t in planes)
        or any(_requires_grad(t) for t in _grad_tensors(group))
    ):
        return _refuse(FusedTraceSkip.REQUIRES_GRAD, mode=mode, n=n)

    # A bundle whose ``is_normalized`` flag is clear makes
    # ``HomogeneousPropagation.propagate`` re-normalise the directions after
    # every surface (propagation/homogeneous.py:56-57).  The kernel mirrors
    # that function WITHOUT the branch, so the trace it would run is not the
    # trace the per-op path runs -- measured at 4.9 mm on a bundle with
    # non-unit direction cosines and at 4.5e-13 mm on an ordinary pupil bundle
    # whose flag alone was cleared (finding R1-V1-04).  Plan 1.2: what the
    # kernel does not mirror is refused with a counted reason, never traced.
    # The reason is ``propagation_model`` -- the propagation the kernel mirrors
    # does not cover this bundle -- because the ``FusedTraceSkip`` set is
    # frozen by plan 3.3 and only the integrator may add a value; a dedicated
    # ``rays_not_normalized`` would read better and is requested in
    # ``NOTES/fused-trace-research/status.md``.
    if not getattr(rays, "is_normalized", True):
        return _refuse(FusedTraceSkip.PROPAGATION_MODEL, mode=mode, n=n)

    surfaces = list(group.surfaces)
    s = len(surfaces)
    if s < 2 or type(surfaces[0]) is not ObjectSurface:
        return _refuse(FusedTraceSkip.SURFACE0_NOT_OBJECT, mode=mode, n=n, s=s)
    if n > 2**30:
        return _refuse(FusedTraceSkip.TOO_MANY_RAYS, mode=mode, n=n, s=s)
    min_rays = int(os.environ.get("OPTILAND_METAL_FUSED_TRACE_MIN_RAYS", "0"))
    if n < min_rays:
        return _refuse(FusedTraceSkip.MIN_RAYS, mode=mode, n=n, s=s)

    # -- per-surface checks (design 2.2, steps 9-16, free part) ------------
    materials: list[tuple[Any, Any]] = []
    for surface in surfaces[1:]:
        reason = _check_surface(surface, mode, materials)
        if reason is not None:
            return _refuse(reason, mode=mode, n=n, s=s)

    # -- memory (step 17): the worst case the hook can ask for -------------
    if (
        _bytes(
            designs=1,
            surfaces=s,
            coefficients=1,
            n_rows=s,
            n_rays=n,
            launch_designs=1,
            write_final=True,
        )
        > _memory_budget()
    ):
        return _refuse(FusedTraceSkip.MEMORY, mode=mode, n=n, s=s)

    # -- the one readback (step 18) ----------------------------------------
    if wavelength is None:
        uniform, w0 = _uniform_wavelength(rays.w, mode)
        _count_event("fused_trace:readback")
        if not uniform:
            return _refuse(FusedTraceSkip.MIXED_WAVELENGTH, mode=mode, n=n, s=s)
    else:
        w0 = float(wavelength)

    # -- indices at w0 (step 16's second half; needs w0) -------------------
    for material_pre, material_post in materials:
        for value in (
            _index_at(material_pre, "n", w0),
            _index_at(material_post, "n", w0),
            _index_at(material_pre, "k", w0),
        ):
            if value is None or not math.isfinite(value):
                return _refuse(FusedTraceSkip.NONFINITE_INDEX, mode=mode, n=n, s=s)

    return GateResult(
        ok=True, reason=None, structural=False, mode=mode, w0=w0, n=n, s=s
    )


def _check_surface(
    surface: Any, mode: str, materials: list[tuple[Any, Any]]
) -> FusedTraceSkip | None:
    """Design 2.2 steps 9-16 for one surface, in order."""
    if type(surface) not in (Surface, ImageSurface):
        return FusedTraceSkip.SURFACE_TYPE

    geometry = getattr(surface, "geometry", None)
    adapter = GEOMETRY_ADAPTERS.get(type(geometry))
    if adapter is None:
        return FusedTraceSkip.GEOMETRY_TYPE
    reason = adapter.check(geometry)
    if reason is not None:
        return reason
    if mode == "df64" and adapter.code in (L.GEOM_EVEN, L.GEOM_ODD):
        # ``EvenAsphere.sag``'s ``Ci * r2 ** (i + 1)`` is a host-scalar df64 op
        # for a Python/NumPy scalar and an array-array df64 op for a backend
        # tensor; the two round differently (measured: 2 raw words on 1 ray of
        # 4096, finding R1-V1-05), and ``_fill_asphere`` stores ``float(Ci)``,
        # so the kernel can only ever mirror the host-scalar form.  Refuse the
        # form the record cannot represent instead of tracing it -- the same
        # df64-only asymmetry as the ``SR_AP0..SR_AP3`` denormal scan below.
        # sf64 is unaffected: both forms are correctly rounded binary64 there,
        # so the two expressions agree bit for bit (measured: 0 raw words).
        # The reason is ``geometry_type`` because ``_check_asphere`` already
        # returns it for a coefficient form the adapter cannot take, and the
        # ``FusedTraceSkip`` set is frozen by plan 3.3.
        for coefficient in geometry.coefficients:
            if is_backend_scalar(coefficient):
                return FusedTraceSkip.GEOMETRY_TYPE

    cs = geometry.cs
    if type(cs) is not CoordinateSystem or cs.reference_cs is not None:
        # R3-V2-03.  Every other mirrored family is keyed by EXACT type --
        # ``type(group)``, ``type(rays)``, ``type(surface)``,
        # ``GEOMETRY_ADAPTERS.get(type(geometry))``, ``APERTURE_ADAPTERS``,
        # ``INTERACTION_ADAPTERS``, ``type(material.propagation_model)`` -- and
        # the pose was the one exception: it checked only ``reference_cs``.
        # ``CoordinateSystem.localize``/``globalize`` are MIRRORED rows, but a
        # subclass overrides them without touching the base source the digest
        # is taken from, so ``trace_mirror.check_all()`` cannot see it: a
        # subclass whose ``localize`` adds a shift fused and disagreed with the
        # per-op path in both modes.  The reason is ``reference_cs`` because it
        # is the pose block's own counted reason and the ``FusedTraceSkip`` set
        # is frozen by plan 3.3 -- the same precedent as the ``geometry_type``
        # reuse above.
        return FusedTraceSkip.REFERENCE_CS
    for pose in (cs.x, cs.y, cs.z, cs.rx, cs.ry, cs.rz):
        value = _scalar_float(pose)
        if value is None or not math.isfinite(value):
            return FusedTraceSkip.POSE_NONFINITE

    model = getattr(surface, "interaction_model", None)
    im_adapter = INTERACTION_ADAPTERS.get(type(model))
    if im_adapter is None:
        return FusedTraceSkip.INTERACTION_TYPE
    reason = im_adapter.check(model)
    if reason is not None:
        return reason

    material_pre = surface.material_pre
    material_post = surface.material_post
    for material in (material_pre, material_post):
        if type(getattr(material, "propagation_model", None)) is not (
            HomogeneousPropagation
        ):
            return FusedTraceSkip.PROPAGATION_MODEL
    materials.append((material_pre, material_post))

    aperture = getattr(surface, "aperture", None)
    if aperture is not None:
        ap_adapter = APERTURE_ADAPTERS.get(type(aperture))
        if ap_adapter is None:
            return FusedTraceSkip.APERTURE_TYPE
        reason = ap_adapter.check(aperture)
        if reason is not None:
            return reason
        if mode == "df64":
            # The stored slots are what the encoder will see: the radial and
            # elliptical adapters square their parameters on the host, and a
            # square inside the float32 denormal band is flushed to zero, which
            # would turn an annulus into a disc (design 3.3).
            scratch = np.zeros(L.SR_STRIDE, dtype=np.float64)
            ap_adapter.fill(
                aperture,
                np.zeros(L.SI_STRIDE, dtype=np.int32),
                scratch,
                np.zeros(1, dtype=np.float64),
                {"mode": mode},
            )
            for param in scratch[L.SR_AP0 : L.SR_AP3 + 1]:
                if 0.0 < abs(param) < encode.DF64_FLT_MIN:
                    return FusedTraceSkip.APERTURE_PARAMS

    if adapter.code in (L.GEOM_EVEN, L.GEOM_ODD):
        max_iter = getattr(geometry, "max_iter", None)
        tol = getattr(geometry, "tol", None)
        if not isinstance(max_iter, (int, np.integer)) or isinstance(max_iter, bool):
            return FusedTraceSkip.NEWTON_PARAMS
        if not 0 <= int(max_iter) <= MAX_NEWTON_ITER:
            return FusedTraceSkip.NEWTON_PARAMS
        tol_value = _scalar_float(tol)
        if tol_value is None or not math.isfinite(tol_value):
            return FusedTraceSkip.NEWTON_PARAMS
    return None


def _index_at(material: Any, prop: str, w0: float) -> float | None:
    """``material.n``/``k`` at ``w0`` through a 1-element ``be.array`` (design 3.8)."""
    if material is None:
        return None
    try:
        value = getattr(material, prop)(be.array([w0]))
    except Exception:  # pragma: no cover - a material that cannot evaluate
        return None
    return float(np.ravel(be.to_numpy(value))[0])


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
@dataclass
class TraceRecords:
    """The per-surface tables one launch binds.

    Attributes:
        mode: ``'df64'`` or ``'sf64'``.
        B: number of designs (1 on the hook path).
        S: number of surfaces, including the object surface at row 0.
        C: coefficient stride, ``max(1, max n_coeff)``.
        w0: the canonical wavelength every material scalar was evaluated at.
        surf_int: int32 ``[B, S, SI_STRIDE]``.
        surf_real: float64 ``[B, S, SR_STRIDE]``.
        coef: float64 ``[B, S, C]``, zero padded, lowest order first.
        snap_rows: int32 ``[B, S]``; the snapshot row per surface, -1 when the
            surface is not recorded.
        step_cost: int32 ``[S]``; 1 per surface, ``1 + max_iter`` for a Newton
            geometry (plan 3.3).
        n_rows: number of recorded snapshot rows.
        has_newton: True when any surface is solved by Newton iteration.
        weighted_steps: ``int(step_cost[1:].sum())``; the chunker's unit.
    """

    mode: str
    B: int
    S: int
    C: int
    w0: float
    surf_int: np.ndarray
    surf_real: np.ndarray
    coef: np.ndarray
    snap_rows: np.ndarray
    step_cost: np.ndarray
    n_rows: int
    has_newton: bool
    weighted_steps: int = field(default=0)


def _snap_rows(group: SurfaceGroup, record: Any, s: int) -> tuple[list[int], int]:
    """Map the ``record`` policy onto one snapshot row per surface."""
    rows = [-1] * s
    if record is False or record is None:
        return rows, 0
    if record is True:
        return list(range(s)), s
    if isinstance(record, str):
        if record == "image":
            wanted = [s - 1]
        elif record == "stop":
            wanted = [int(group.stop_index)]
        else:
            raise ValueError(f"unknown record policy: {record!r}")
    else:
        wanted = [int(i) for i in record]
    for row, index in enumerate(wanted):
        if not 0 <= index < s:
            raise ValueError(f"record index {index} is outside 0..{s - 1}")
        if rows[index] != -1:
            raise ValueError(f"record index {index} is listed twice")
        rows[index] = row
    return rows, len(wanted)


def compile_records(
    group: Any,
    w0: float,
    mode: str,
    *,
    record: bool | str | Sequence[int] = True,
    designs: int = 1,
) -> TraceRecords:
    """Build the per-surface tables for ``group`` at wavelength ``w0``.

    Every real slot is read through ``be.*`` on the live tensors, so the value
    is the one the per-op path would compute (design 3.4).  With ``designs > 1``
    the leading axis is allocated and every design row is filled with the
    group's current state; the batch API overwrites the rows it varies.

    Args:
        group: the surface group.
        w0: the canonical wavelength (see :func:`canonical_w0`).
        mode: ``'df64'`` or ``'sf64'``.
        record: the snapshot policy -- True (every surface), False (none),
            ``'image'``, ``'stop'`` or an explicit sequence of surface indices.
        designs: the size of the leading design axis.

    Returns:
        TraceRecords: the compiled tables.
    """
    surfaces = list(group.surfaces)
    s = len(surfaces)
    b = int(designs)
    rows, n_rows = _snap_rows(group, record, s)

    n_coeff = [0] * s
    for index, surface in enumerate(surfaces[1:], start=1):
        adapter = GEOMETRY_ADAPTERS.get(type(surface.geometry))
        if adapter is None:
            raise ValueError(
                f"surface {index} has no geometry adapter: "
                f"{type(surface.geometry).__name__}"
            )
        n_coeff[index] = len(getattr(surface.geometry, "coefficients", ()))
    c = max(1, max(n_coeff))

    surf_int = np.zeros((b, s, L.SI_STRIDE), dtype=np.int32)
    surf_real = np.zeros((b, s, L.SR_STRIDE), dtype=np.float64)
    coef = np.zeros((b, s, c), dtype=np.float64)
    snap_rows = np.full((b, s), -1, dtype=np.int32)
    step_cost = np.ones(s, dtype=np.int32)

    row_int = surf_int[0]
    row_real = surf_real[0]
    row_coef = coef[0]

    # Row 0: the object surface records the launch state and nothing else.
    row_int[0, L.SI_GEOM] = L.GEOM_OBJECT
    row_int[0, L.SI_SNAPROW] = rows[0]
    row_int[0, L.SI_STEPCOST] = 1
    snap_rows[:, 0] = rows[0]

    has_newton = False
    ctx: dict[str, Any] = {"mode": mode, "w0": w0, "group": group}
    for index, surface in enumerate(surfaces[1:], start=1):
        ctx["surface"] = surface
        ctx["index"] = index
        geometry = surface.geometry
        _fill_pose(geometry.cs, row_int[index], row_real[index])
        GEOMETRY_ADAPTERS[type(geometry)].fill(
            geometry, row_int[index], row_real[index], row_coef[index], ctx
        )
        aperture = surface.aperture
        if aperture is not None:
            row_int[index, L.SI_FLAGS] |= L.FL_HAS_APERTURE
            APERTURE_ADAPTERS[type(aperture)].fill(
                aperture, row_int[index], row_real[index], row_coef[index], ctx
            )
            if _accepts_aperture(geometry) and not (
                row_int[index, L.SI_FLAGS] & L.FL_RADIUS_INF
            ):
                row_int[index, L.SI_FLAGS] |= L.FL_AP_IN_ROOT
        INTERACTION_ADAPTERS[type(surface.interaction_model)].fill(
            surface.interaction_model,
            row_int[index],
            row_real[index],
            row_coef[index],
            ctx,
        )
        row_int[index, L.SI_SNAPROW] = rows[index]
        snap_rows[:, index] = rows[index]
        geom_code = int(row_int[index, L.SI_GEOM])
        if geom_code in (L.GEOM_EVEN, L.GEOM_ODD):
            has_newton = True
            step_cost[index] = 1 + int(row_int[index, L.SI_MAXITER])
        row_int[index, L.SI_STEPCOST] = int(step_cost[index])

    if b > 1:
        surf_int[1:] = row_int
        surf_real[1:] = row_real
        coef[1:] = row_coef

    return TraceRecords(
        mode=mode,
        B=b,
        S=s,
        C=c,
        w0=float(w0),
        surf_int=surf_int,
        surf_real=surf_real,
        coef=coef,
        snap_rows=snap_rows,
        step_cost=step_cost,
        n_rows=n_rows,
        has_newton=has_newton,
        weighted_steps=int(step_cost[1:].sum()),
    )


def _accepts_aperture(geometry: Any) -> bool:
    """Mirror ``_aperture_aware_distance``'s signature test, without caching it.

    The per-op path writes the answer to ``surface._distance_capability``; the
    record compiler must not (design 2.6), so the test is repeated here.
    """
    import inspect

    parameters = inspect.signature(geometry.distance).parameters
    return "aperture" in parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )


def _fill_pose(cs: Any, row_int: np.ndarray, row_real: np.ndarray) -> None:
    """Slots 0-17 and the ``FL_HAS_R*`` bits (design 3.4).

    ``localize`` evaluates ``be.cos(-rz)`` and ``globalize`` ``be.cos(rz)``, so
    both are recorded: no ``cos(-a) == cos(a)`` libm assumption is made.  The
    ``if self.rx:`` truth tests of ``CoordinateSystem`` are mirrored exactly, so
    an exact-zero rotation skips its rotation in the kernel too.
    """
    row_real[L.SR_TX] = float(cs.x)
    row_real[L.SR_TY] = float(cs.y)
    row_real[L.SR_TZ] = float(cs.z)
    row_real[L.SR_NTX] = float(-cs.x)
    row_real[L.SR_NTY] = float(-cs.y)
    row_real[L.SR_NTZ] = float(-cs.z)

    row_real[L.SR_CNRZ] = float(be.cos(-cs.rz))
    row_real[L.SR_SNRZ] = float(be.sin(-cs.rz))
    row_real[L.SR_CNRY] = float(be.cos(-cs.ry))
    row_real[L.SR_SNRY] = float(be.sin(-cs.ry))
    row_real[L.SR_CNRX] = float(be.cos(-cs.rx))
    row_real[L.SR_SNRX] = float(be.sin(-cs.rx))

    row_real[L.SR_CRX] = float(be.cos(cs.rx))
    row_real[L.SR_SRX] = float(be.sin(cs.rx))
    row_real[L.SR_CRY] = float(be.cos(cs.ry))
    row_real[L.SR_SRY] = float(be.sin(cs.ry))
    row_real[L.SR_CRZ] = float(be.cos(cs.rz))
    row_real[L.SR_SRZ] = float(be.sin(cs.rz))

    flags = 0
    if cs.rx:
        flags |= L.FL_HAS_RX
    if cs.ry:
        flags |= L.FL_HAS_RY
    if cs.rz:
        flags |= L.FL_HAS_RZ
    row_int[L.SI_FLAGS] |= flags


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
def memory_bytes(
    records: TraceRecords, N: int, launch_designs: int, write_final: bool
) -> int:
    """Device bytes one launch of ``records`` over ``N`` rays allocates.

    Args:
        records: the compiled tables.
        N: rays per design.
        launch_designs: ``1`` for a shared launch buffer, ``B`` for per-design.
        write_final: whether the ``final`` buffer is allocated.

    Returns:
        int: the total, per plan 3.6 (accurate to 1-2%, day-1 probe P6).
    """
    return _bytes(
        designs=records.B,
        surfaces=records.S,
        coefficients=records.C,
        n_rows=records.n_rows,
        n_rays=int(N),
        launch_designs=int(launch_designs),
        write_final=bool(write_final),
    )
