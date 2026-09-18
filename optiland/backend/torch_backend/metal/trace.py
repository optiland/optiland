"""Host driver for the fused trace-interpreter kernel (``kernels/trace.metal``).

This module is the only thing between ``SurfaceGroup.trace``'s hook and the
kernel.  It owns, in order:

* the third per-mode Metal library (mode headers + ``conic.metal`` +
  ``trace_layout.h`` + ``trace.metal``), compiled once per process and warmed
  up because Metal builds pipeline states lazily (day-1 Q12);
* launch packing from ``rays.<attr>.detach().components`` with the component
  checks ``MetalLibrary._check_tensor`` applies (library.py:308);
* table encoding, the ten positional bindings of
  ``trace_layout.BUFFER_ORDER`` and the weighted slab plan (:func:`_slab_plan`);
* the unconditional write-completion sentinel: ``iters`` is pre-filled with
  ``ITERS_UNWRITTEN`` and checked after the last slab, because a command
  buffer that runs too long is aborted **silently** -- day-1 P9 measured a
  dispatch that returned in 0.51 s with 1,047,360 of 1,048,576 sentinels
  unwritten and ``torch.mps.synchronize()`` reporting success.  Reading the
  sentinel back is also what waits for the slabs: the driver makes no
  device-wide ``torch.mps.synchronize()`` call, because one thread doing that
  while another has an encoder open aborts the process (round-3 finding
  R3-V1-01);
* the late fallback: a Newton surface that crossed the tolerance crossover
  makes the whole trace go back to the per-op path with nothing written;
* writeback as zero-copy ``wrap`` views of ``aten.select`` slices (design 5,
  probed in day-1 P2: shared ``data_ptr``, one ``gpu:mul`` and zero ``host:*``
  for arithmetic on a view);
* the counters, the one-per-process notices and the DIAG planes.  None of
  the driver's notices can become an exception in the caller's process:
  they announce a fallback the caller never asked for, so a warning filter
  that promotes them to errors would make the fused path RAISE where the
  per-op path returns (round-3 finding R3-V1-06).  :func:`_emit` demotes a
  promoted warning to a ``logging`` record on this module's logger, which
  reaches stderr through ``logging.lastResort`` and cannot be promoted.

Nothing in here "improves" on the Python path: the kernel mirrors it and this
driver only moves bytes (plan 0.2.1).

Import cost: this module imports torch, so it is imported lazily by the hook
(plan 3.4) and never at module scope from an upstream file.
"""

from __future__ import annotations

import contextlib
import logging
import os
import warnings
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from optiland.backend.torch_backend.metal import compile as _compile
from optiland.backend.torch_backend.metal import encode, trace_layout, trace_mirror
from optiland.backend.torch_backend.metal.library import (
    DEFAULT_CHUNK,
    HEADER_PREREQUISITES,
    HEADERS,
    REQUIRED_HEADERS,
    get_library,
)
from optiland.backend.torch_backend.metal.tensor import (
    MACHINE_EPS,
    count_event,
    count_gpu,
    is_metal,
    wrap,
)
from optiland.backend.torch_backend.metal.trace_adapters import (
    STRUCTURAL_REASONS,
    FusedTraceSkip,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

__all__ = [
    "DEFAULT_MAX_STEPS",
    "ENTRY",
    "ENTRY_SPHERICAL",
    "entry_name",
    "FusedTraceDriftWarning",
    "FusedTraceUnavailableWarning",
    "LaunchResult",
    "MetalFallbackError",
    "N_FLOOR",
    "TRACE_FILES",
    "diag_from",
    "forget_diag",
    "fused_trace",
    "launch_trace",
    "reset_driver_state",
    "trace_source",
]


class MetalFallbackError(RuntimeError):
    """Raised when the fused path refuses a candidate under ``require``.

    Also raised, in **every** switch setting, when the write-completion
    sentinel shows that the GPU did not write every ``(b, s, i)`` cell: that
    is silent data corruption, not a fallback (day-1 P9).
    """


class FusedTraceUnavailableWarning(RuntimeWarning):
    """Metal or the trace library is unavailable; the per-op path runs."""


class FusedTraceDriftWarning(RuntimeWarning):
    """A MIRRORED Python source drifted from its fingerprint (plan 3.7)."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The amalgamation tail, after the mode's header stack (design 4.1).  The
#: order is pinned: ``conic.metal`` defines ``conic_ops<R>`` and
#: ``trace_layout.h`` the ``OT_*`` slots that ``trace.metal`` consumes
#: (WP1 ``test_include_order``).
TRACE_FILES: tuple[str, ...] = ("conic.metal", "trace_layout.h", "trace.metal")

#: Entry point per mode (design 4.3).
ENTRY: dict[str, str] = {
    "df64": "trace_surfaces_df64",
    "sf64": "trace_surfaces_sf64",
}

#: The spherical-only twin of each entry point: the same body instantiated
#: without the Newton branch.  Selected when ``records.has_newton`` is False.
ENTRY_SPHERICAL: dict[str, str] = {
    "df64": "trace_surfaces_df64_spherical",
    "sf64": "trace_surfaces_sf64_spherical",
}


def entry_name(mode: str, has_newton: bool) -> str:
    """The kernel entry point the driver dispatches for ``mode``."""
    return (ENTRY if has_newton else ENTRY_SPHERICAL)[mode]


#: Weighted surface-steps per command buffer (design 4.4, day-1 Q6: ~2 ms of
#: GPU time on this machine, three orders under the measured safe duration).
DEFAULT_MAX_STEPS = 2**26

#: ``consts[C_NFLOOR]``: the |N| floor of ``StandardGeometry.distance`` for an
#: infinite radius (``geometries/standard.py:90``).  Not float32-exact
#: (day-1 P7), so it is encoded on the host and never written as a ``lit()``.
N_FLOOR = 1e-14

_NCOMP: dict[str, int] = {"df64": 2, "sf64": 1}
_COMP_DTYPE: dict[str, Any] = {"df64": torch.float32, "sf64": torch.int64}

#: Status bit -> counter suffix, for ``OPTILAND_METAL_TRACE_DIAG=1``.
_DIAG_BITS: tuple[tuple[int, str], ...] = (
    (trace_layout.ST_MISS, "miss"),
    (trace_layout.ST_TIR, "tir"),
    (trace_layout.ST_CLIPPED, "clipped"),
    (trace_layout.ST_NEWTON_NOT_CONVERGED, "newton_not_converged"),
    (trace_layout.ST_TOL_CROSSOVER, "tol_crossover"),
    (trace_layout.ST_NZ_FLOORED, "nz_floored"),
    (trace_layout.ST_DF_FLOORED, "df_floored"),
    (trace_layout.ST_NONUNIFORM_W, "nonuniform_w"),
)

# ---------------------------------------------------------------------------
# Process state (all resettable for tests; see :func:`reset_driver_state`)
# ---------------------------------------------------------------------------

#: Where a notice goes when a warning filter refuses to let it be a warning
#: (R3-V1-06).  ``logging.lastResort`` prints WARNING and above to stderr when
#: the embedding application configures nothing, so the message is still seen.
_LOGGER = logging.getLogger(__name__)

_LIBS: dict[str, Any] = {}
_DRIFT: list[str] | None = None
_WARNED: set[str] = set()
#: group -> (status, iters) of its last fused trace, under DIAG only.  A weak
#: key so a traced group stays collectable and ``copy.deepcopy(optic)`` does
#: not drag the planes along (plan WP3 ``test_diag_not_deepcopied``).
_DIAG_PLANES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def reset_driver_state(*, libraries: bool = False) -> None:
    """Forget the cached drift result, the one-shot warnings and DIAG planes.

    Args:
        libraries: Also drop the compiled-library cache (the Metal libraries
            themselves stay cached by source hash in ``compile._LIBRARIES``).
    """
    global _DRIFT
    _DRIFT = None
    _WARNED.clear()
    _DIAG_PLANES.clear()
    if libraries:
        _LIBS.clear()


# ---------------------------------------------------------------------------
# Environment switches (plan 3.5, design 2.3).  Read per trace so a
# programmatic ``os.environ[...] = "0"`` is a valid rollback (plan 1.5).
# ---------------------------------------------------------------------------


def _switch() -> str:
    """``OPTILAND_METAL_FUSED_TRACE``: ``"1"`` (default), ``"0"`` or ``"require"``."""
    return os.environ.get("OPTILAND_METAL_FUSED_TRACE", "1")


def _max_steps() -> int:
    """``OPTILAND_METAL_FUSED_TRACE_MAX_STEPS``: weighted steps per slab."""
    raw = os.environ.get("OPTILAND_METAL_FUSED_TRACE_MAX_STEPS")
    if raw is None:
        return DEFAULT_MAX_STEPS
    value = int(raw)
    if value < 1:
        raise ValueError(
            f"OPTILAND_METAL_FUSED_TRACE_MAX_STEPS must be >= 1, got {value}"
        )
    return value


def _min_rays() -> int:
    """``OPTILAND_METAL_FUSED_TRACE_MIN_RAYS``: below this N the gate refuses."""
    return int(os.environ.get("OPTILAND_METAL_FUSED_TRACE_MIN_RAYS", "0"))


def _memory_fraction() -> float:
    """``OPTILAND_METAL_FUSED_TRACE_MEMORY_FRACTION`` (default 0.25)."""
    return float(os.environ.get("OPTILAND_METAL_FUSED_TRACE_MEMORY_FRACTION", "0.25"))


def _group_size() -> int | None:
    """``OPTILAND_METAL_FUSED_TRACE_GROUP``: threadgroup size, or None."""
    raw = os.environ.get("OPTILAND_METAL_FUSED_TRACE_GROUP")
    if raw in (None, ""):
        return None
    value = int(raw)
    if value < 1 or value > 256 or (value & (value - 1)):
        raise ValueError(
            "OPTILAND_METAL_FUSED_TRACE_GROUP must be a power of two <= 256, "
            f"got {value}"
        )
    return value


def _diag() -> bool:
    """``OPTILAND_METAL_TRACE_DIAG``: pull the status / iters planes."""
    return os.environ.get("OPTILAND_METAL_TRACE_DIAG", "0") == "1"


def _drift_policy() -> str:
    """``OPTILAND_METAL_FUSED_TRACE_DRIFT``: ``refuse`` (default) or ``warn``."""
    return os.environ.get("OPTILAND_METAL_FUSED_TRACE_DRIFT", "refuse")


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------


def _filtered_headers(mode: str) -> list[str]:
    """The mode's header stack, filtered by presence and prerequisites.

    Copies ``MetalLibrary.__init__``'s recipe (library.py:235-248) rather than
    reaching into it, so ``library.py`` stays unchanged (plan 3.1): a partial
    header set is a clean ``FileNotFoundError``, not a raw Metal error.
    """
    present = [h for h in HEADERS[mode] if (_compile._KERNEL_DIR / h).is_file()]
    present = [
        h for h in present if all(p in present for p in HEADER_PREREQUISITES.get(h, ()))
    ]
    missing = [h for h in REQUIRED_HEADERS[mode] if h not in present]
    if missing:
        raise FileNotFoundError(
            f"{mode}: required kernel header(s) missing from "
            f"{_compile._KERNEL_DIR}: {missing}"
        )
    return present


def trace_source(mode: str) -> str:
    """The amalgamated source of the trace library for ``mode``."""
    return _compile.kernel_source(*_filtered_headers(mode), *TRACE_FILES)


def _kernel_library(mode: str) -> Any:
    """Compile (once per process) and warm up the trace library for ``mode``.

    ``get_library(mode)`` runs first so the representation's df64 self-test has
    run in this process (library.py:268) before any fused trace is dispatched;
    the trace library is a separate, third library (design 4.1).
    """
    lib = _LIBS.get(mode)
    if lib is None:
        if mode not in ENTRY:
            raise ValueError(f"unknown mode {mode!r}")
        get_library(mode)
        lib = _compile.compile_library(trace_source(mode))
        _warm_up(lib, mode)
        _LIBS[mode] = lib
    return lib


def _warm_up(lib: Any, mode: str) -> None:
    """Dispatch the entry point once at B = N = S = 1 to force the pipeline.

    A Metal compile that succeeds can still fail at the first dispatch because
    pipeline states are built lazily (day-1 Q12); without this the failure
    would surface inside a user's trace.
    """
    surf_int = np.zeros((1, 1, trace_layout.SI_STRIDE), dtype=np.int32)
    surf_int[:, :, trace_layout.SI_SNAPROW] = -1
    records = _StubRecords(
        mode=mode,
        B=1,
        S=1,
        C=0,
        surf_int=surf_int,
        surf_real=np.zeros((1, 1, trace_layout.SR_STRIDE)),
        coef=np.zeros((1, 1, 0)),
        snap_rows=np.full((1, 1), -1, dtype=np.int32),
        n_rows=0,
        has_newton=False,
        weighted_steps=0,
    )
    launch = _encode_components(np.zeros(trace_layout.Q_PLANES), mode)
    for has_newton in (False, True):
        records.has_newton = has_newton
        _launch_slabs(
            lib,
            records,
            launch,
            launch_stride=0,
            N=1,
            write_final=False,
            mode=mode,
            count=False,
        )


@dataclass
class _StubRecords:
    """The subset of ``TraceRecords`` the launch path reads (warm-up only)."""

    mode: str
    B: int
    S: int
    C: int
    surf_int: np.ndarray
    surf_real: np.ndarray
    coef: np.ndarray
    snap_rows: np.ndarray
    n_rows: int
    has_newton: bool
    weighted_steps: int


# ---------------------------------------------------------------------------
# Buffer helpers
# ---------------------------------------------------------------------------


def _check_component(t: Any, dtype: Any, what: str) -> None:
    """Reject anything the kernel would read as raw storage.

    The same five checks ``MetalLibrary._check_tensor`` runs (library.py:308).
    The fused conic driver omits them; the omission is not copied (design 2.5).
    """
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{what}: expected a torch.Tensor, got {type(t).__name__}")
    if isinstance(t, torch.Tensor) and type(t) is not torch.Tensor:
        raise TypeError(f"{what}: expected a plain tensor, got {type(t).__name__}")
    if t.device.type != "mps":
        raise ValueError(f"{what}: expected an mps tensor, got device {t.device}")
    if t.dtype != dtype:
        raise TypeError(f"{what}: expected dtype {dtype}, got {t.dtype}")
    if not t.is_contiguous():
        raise ValueError(f"{what}: tensor must be contiguous (kernels ignore strides)")
    if t.is_neg() or t.is_conj():
        raise ValueError(
            f"{what}: lazily negated / conjugated views are not supported "
            "(the kernel would read the raw storage); call resolve_neg() / "
            "resolve_conj() first"
        )


def _encode_components(a: Any, mode: str) -> tuple[torch.Tensor, ...]:
    """Encode a float64 array into the mode's flat device component buffers."""
    flat = np.ascontiguousarray(np.asarray(a, dtype=np.float64).reshape(-1))
    if mode == "df64":
        return tuple(encode.to_mps_df64(flat))
    return (encode.to_mps_sf64(flat),)


def _empty_components(n: int, mode: str) -> tuple[torch.Tensor, ...]:
    """Allocate zeroed component buffers for ``n`` R elements."""
    dtype = _COMP_DTYPE[mode]
    return tuple(torch.zeros(n, dtype=dtype, device="mps") for _ in range(_NCOMP[mode]))


def _consts_array(mode: str) -> np.ndarray:
    """``consts R[C_SIZE]``: machine epsilon and the |N| floor."""
    c = np.zeros(trace_layout.C_SIZE, dtype=np.float64)
    c[trace_layout.C_EPS] = MACHINE_EPS[mode]
    c[trace_layout.C_NFLOOR] = N_FLOOR
    return c


def _dims_tensor(
    *,
    S: int,
    N: int,
    B: int,
    launch_stride: int,
    ray_base: int,
    n_rows: int,
    C: int,
    design_base: int,
    write_final: bool,
) -> torch.Tensor:
    """A fresh ``dims int32[D_SIZE]`` buffer for one slab (design 4.4)."""
    d = [0] * trace_layout.D_SIZE
    d[trace_layout.D_S] = S
    d[trace_layout.D_N] = N
    d[trace_layout.D_B] = B
    d[trace_layout.D_LAUNCH_STRIDE] = launch_stride
    d[trace_layout.D_RAY_BASE] = ray_base
    d[trace_layout.D_NROWS] = n_rows
    d[trace_layout.D_C] = C
    d[trace_layout.D_P] = trace_layout.SR_STRIDE
    d[trace_layout.D_DESIGN_BASE] = design_base
    d[trace_layout.D_WRITE_FINAL] = int(write_final)
    return torch.tensor(d, dtype=torch.int32, device="mps")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _slab_plan(
    B: int, N: int, weighted_steps: int, max_steps: int, default_chunk: int
) -> list[tuple[int, int, int, int]]:
    """The slab plan: ``(design_base, b_extent, ray_base, n_extent)`` per launch.

    Design 4.4 with ``weighted_steps`` in place of ``steps_per_path``
    (plan 3.5, [fix: L4.5]): an asphere row costs ``1 + max_iter`` steps
    because every thread may run the Newton loop, so a 44-row aspheric system
    is chunked far more finely than a 44-row spherical one.

    This function is the single source of the plan: tests predict chunk counts
    by calling it, and ``launch_trace`` dispatches exactly what it returns.

    Args:
        B: Number of designs.
        N: Rays per design.
        weighted_steps: ``records.weighted_steps`` (``step_cost[1:].sum()``).
        max_steps: Weighted steps per command buffer.
        default_chunk: Upper bound on the ray extent (``library.DEFAULT_CHUNK``).

    Returns:
        A list of slabs covering ``[0, B) x [0, N)`` exactly once.  Both
        extents are always >= 1.
    """
    if B < 0 or N < 0:
        raise ValueError(f"B and N must be non-negative, got B={B}, N={N}")
    if B == 0 or N == 0:
        return []
    w = max(1, int(weighted_steps))
    b_chunk = max(1, min(B, max_steps // max(1, w * N)))
    n_chunk = max(
        1,
        min(N, default_chunk // b_chunk, max_steps // max(1, w * b_chunk)),
    )
    return [
        (
            design_base,
            min(b_chunk, B - design_base),
            ray_base,
            min(n_chunk, N - ray_base),
        )
        for design_base in range(0, B, b_chunk)
        for ray_base in range(0, N, n_chunk)
    ]


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------


@dataclass
class LaunchResult:
    """What one fused trace produced, before writeback.

    Attributes:
        snap: Component buffers of ``snap`` reshaped
            ``(S_PLANES, B, n_rows, N)``, or None when nothing was recorded.
        final: Component buffers of ``final`` reshaped ``(F_PLANES, B, N)``,
            or None when ``write_final`` was False.
        status: ``uint8[B, S, N]`` status planes.
        iters: ``uint8[B, S, N]`` Newton iteration counts.
        launches: Number of slabs dispatched.
        late_fallback_designs: ``bool[B]``; True where a Newton surface set
            ``ST_TOL_CROSSOVER`` for at least one ray.
    """

    snap: tuple[torch.Tensor, ...] | None
    final: tuple[torch.Tensor, ...] | None
    status: torch.Tensor
    iters: torch.Tensor
    launches: int
    late_fallback_designs: np.ndarray


def _late_fallback_mask(status: torch.Tensor, has_newton: bool) -> np.ndarray:
    """Which designs crossed the Newton tolerance crossover.

    Args:
        status: The ``uint8[B, S, N]`` status planes.
        has_newton: Whether any surface is Newton-solved.  When it is False the
            reduction (and its sync) is skipped entirely: a system without an
            asphere never pays for the post-launch check (design 2.5 step 7).

    Returns:
        ``bool[B]``, all False when ``has_newton`` is False.
    """
    b = int(status.shape[0])
    if not has_newton:
        return np.zeros(b, dtype=bool)
    crossed = (status & trace_layout.ST_TOL_CROSSOVER) != 0
    return crossed.any(dim=2).any(dim=1).cpu().numpy().astype(bool)


def _launch_slabs(
    lib: Any,
    records: Any,
    launch: Sequence[torch.Tensor],
    *,
    launch_stride: int,
    N: int,
    write_final: bool,
    mode: str,
    count: bool,
) -> LaunchResult:
    """Allocate, bind and dispatch every slab; then check the sentinel."""
    B, S, C = int(records.B), int(records.S), int(records.C)
    n_rows = int(records.n_rows)
    ncomp = _NCOMP[mode]
    dtype = _COMP_DTYPE[mode]

    if launch_stride not in (0, 1):
        raise ValueError(
            "launch_stride must be 0 (shared launch, Lb = 1) or 1 (per design, "
            f"Lb = B); got {launch_stride}.  The kernel derives Lb from it "
            "(WP1 finding 2) and admits no third layout."
        )
    lb = B if launch_stride else 1
    if len(launch) != ncomp:
        raise ValueError(f"{mode} needs {ncomp} launch component(s), got {len(launch)}")
    want = trace_layout.Q_PLANES * lb * N
    for k, t in enumerate(launch):
        _check_component(t, dtype, f"launch component {k}")
        if t.numel() != want:
            raise ValueError(
                f"launch component {k}: expected {want} elements "
                f"(Q_PLANES * Lb * N = {trace_layout.Q_PLANES} * {lb} * {N}), "
                f"got {t.numel()}"
            )

    surf_int = torch.tensor(
        np.ascontiguousarray(records.surf_int.reshape(-1), dtype=np.int32),
        dtype=torch.int32,
        device="mps",
    )
    surf_real = _encode_components(records.surf_real, mode)
    # WP1 finding 3: bind a non-empty buffer even when no surface carries
    # coefficients, so the binding is always valid.
    coef_src = records.coef.reshape(-1) if C else np.zeros(1)
    coef = _encode_components(coef_src if coef_src.size else np.zeros(1), mode)
    consts = _encode_components(_consts_array(mode), mode)

    snap_n = trace_layout.S_PLANES * B * n_rows * N
    fin_n = trace_layout.F_PLANES * B * N if write_final else 0
    snap = _empty_components(max(1, snap_n), mode)
    fin = _empty_components(max(1, fin_n), mode)
    status = torch.zeros(B * S * N, dtype=torch.uint8, device="mps")
    iters = torch.full(
        (B * S * N,), trace_layout.ITERS_UNWRITTEN, dtype=torch.uint8, device="mps"
    )

    plan = _slab_plan(B, N, int(records.weighted_steps), _max_steps(), DEFAULT_CHUNK)
    entry = getattr(lib, entry_name(mode, bool(records.has_newton)))
    group = _group_size()
    for design_base, b_extent, ray_base, n_extent in plan:
        dims = _dims_tensor(
            S=S,
            N=N,
            B=B,
            launch_stride=launch_stride,
            ray_base=ray_base,
            n_rows=n_rows,
            C=C,
            design_base=design_base,
            write_final=write_final,
        )
        args = [
            *launch,
            surf_int,
            *surf_real,
            *coef,
            dims,
            *consts,
            *snap,
            *fin,
            status,
            iters,
        ]
        kwargs: dict[str, Any] = {"threads": [n_extent, b_extent, 1]}
        if group is not None:
            kwargs["group_size"] = [group, 1, 1]
        entry(*args, **kwargs)
        if count:
            count_gpu("fused_trace")
            count_event("fused_trace:chunks")

    # Write-completion sentinel, unconditionally and in every mode: a command
    # buffer the driver aborted reports success (day-1 P9).  One reduction --
    # and the ``.item()`` that reads its result is also what WAITS for every
    # slab dispatched above: a host readback commits this stream's command
    # buffer and blocks until it has completed, so nothing below can read a
    # half-written buffer.
    #
    # There is deliberately no ``torch.mps.synchronize()`` here (round-3
    # finding R3-V1-01).  That call commits the process-wide MPS command
    # buffer from the calling thread; when a second thread is tracing another
    # optic at the same time it has an encoder open on that same buffer, and
    # the process ABORTS -- SIGSEGV, or `failed assertion _status <
    # MTLCommandBufferStatusCommitted at line 323 in
    # -[IOGPUMetalCommandBuffer setCurrentCommandEncoder:]`.  Not a catchable
    # exception, and it was the only such call in the package: the per-op path
    # makes none and survives the same workload, so this additive path must
    # not lower the ceiling (upstream #833 runs analyses on worker jobs).  The
    # readback gives this trace exactly the ordering the synchronize gave it;
    # what it does not give is a device-wide barrier, and the driver never
    # needed one.
    if bool((iters == trace_layout.ITERS_UNWRITTEN).any().item()):
        if count:
            count_event("fused_trace:unvisited")
        raise MetalFallbackError(
            "fused trace: the GPU left write-completion sentinels in `iters` "
            f"({int((iters == trace_layout.ITERS_UNWRITTEN).sum().item())} of "
            f"{B * S * N} cells unwritten).  The command buffer was aborted "
            "silently; the results are discarded."
        )

    status3 = status.reshape(B, S, N)
    late = _late_fallback_mask(status3, bool(records.has_newton))

    return LaunchResult(
        snap=(
            tuple(c.reshape(trace_layout.S_PLANES, B, n_rows, N) for c in snap)
            if n_rows
            else None
        ),
        final=(
            tuple(c.reshape(trace_layout.F_PLANES, B, N) for c in fin)
            if write_final
            else None
        ),
        status=status3,
        iters=iters.reshape(B, S, N),
        launches=len(plan),
        late_fallback_designs=late,
    )


def launch_trace(
    records: Any,
    launch: Sequence[torch.Tensor],
    *,
    launch_stride: int,
    N: int,
    write_final: bool,
    mode: str,
) -> LaunchResult:
    """Run one fused trace: every slab of ``records`` over ``N`` rays.

    Args:
        records: A ``TraceRecords`` (``metal/trace_record.py``).
        launch: The packed launch buffers, ``Q_PLANES * Lb * N`` elements per
            component, with ``Lb = 1`` when ``launch_stride == 0`` and
            ``Lb = records.B`` when it is 1.
        launch_stride: 0 (one bundle shared by every design) or 1 (per design).
        N: Rays per design.
        write_final: Write the eleven ``final`` planes.
        mode: ``"df64"`` or ``"sf64"``.

    Returns:
        LaunchResult: buffers, planes, launch count and the per-design late
        fallback mask.

    Raises:
        MetalFallbackError: The write-completion sentinel survived a slab.
    """
    return _launch_slabs(
        _kernel_library(mode),
        records,
        launch,
        launch_stride=launch_stride,
        N=N,
        write_final=write_final,
        mode=mode,
        count=True,
    )


# ---------------------------------------------------------------------------
# Launch packing
# ---------------------------------------------------------------------------

#: ``rays`` attributes in launch-plane order (``trace_layout.Q_*``).
_LAUNCH_ATTRS: tuple[str, ...] = ("x", "y", "z", "L", "M", "N", "i", "opd", "w")


def _plane_components(value: Any, mode: str, n: int, what: str) -> list[torch.Tensor]:
    """The GPU component tensors of one launch plane, flat and length ``n``."""
    if not is_metal(value):
        raise TypeError(f"{what}: expected a MetalFloat64, got {type(value).__name__}")
    if value.mode != mode:
        raise TypeError(f"{what}: expected mode {mode!r}, got {value.mode!r}")
    out: list[torch.Tensor] = []
    for k, c in enumerate(value.detach().components):
        t = c.contiguous().reshape(-1)
        _check_component(t, _COMP_DTYPE[mode], f"{what} component {k}")
        if t.numel() == 1 and n != 1:
            # A uniform plane (``rays.w`` of a bundle generated at one
            # wavelength) is broadcast to the bundle; the gate has already
            # refused mixed wavelengths.
            t = t.expand(n).contiguous()
        if t.numel() != n:
            raise ValueError(f"{what}: expected {n} elements, got {t.numel()}")
        out.append(t)
    return out


def _pack_launch(rays: Any, mode: str, n: int) -> tuple[torch.Tensor, ...]:
    """Pack ``rays`` into one contiguous ``R[9][1][N]`` buffer per component."""
    planes = [
        _plane_components(getattr(rays, attr), mode, n, f"rays.{attr}")
        for attr in _LAUNCH_ATTRS
    ]
    return tuple(
        torch.cat([planes[q][k] for q in range(trace_layout.Q_PLANES)])
        for k in range(_NCOMP[mode])
    )


# ---------------------------------------------------------------------------
# Writeback (design 5)
# ---------------------------------------------------------------------------

#: ``snap`` plane -> the ``Surface`` attribute it feeds.
_SNAP_ATTRS: tuple[tuple[int, str], ...] = (
    (trace_layout.S_X, "x"),
    (trace_layout.S_Y, "y"),
    (trace_layout.S_Z, "z"),
    (trace_layout.S_L, "L"),
    (trace_layout.S_M, "M"),
    (trace_layout.S_N, "N"),
    (trace_layout.S_I, "intensity"),
    (trace_layout.S_OPD, "opd"),
)

#: ``final`` plane -> the ``RealRays`` attribute it feeds.
_FINAL_ATTRS: tuple[tuple[int, str], ...] = (
    (trace_layout.F_X, "x"),
    (trace_layout.F_Y, "y"),
    (trace_layout.F_Z, "z"),
    (trace_layout.F_L, "L"),
    (trace_layout.F_M, "M"),
    (trace_layout.F_N, "N"),
    (trace_layout.F_I, "i"),
    (trace_layout.F_OPD, "opd"),
    (trace_layout.F_L0, "L0"),
    (trace_layout.F_M0, "M0"),
    (trace_layout.F_N0, "N0"),
)


def _snap_view(
    comps: Sequence[torch.Tensor], q: int, b: int, row: int, mode: str
) -> Any:
    """A zero-copy ``MetalFloat64`` view of one ``snap`` row (day-1 P2)."""
    return wrap(tuple(c[q, b, row] for c in comps), mode)


def _final_view(comps: Sequence[torch.Tensor], q: int, b: int, mode: str) -> Any:
    """A zero-copy ``MetalFloat64`` view of one ``final`` plane."""
    return wrap(tuple(c[q, b] for c in comps), mode)


def _write_back(
    group: Any, rays: Any, records: Any, result: LaunchResult, mode: str, b: int = 0
) -> None:
    """Install design ``b``'s snapshots and final ray state as views.

    Every recorded surface gets a row, so the ``be.size(surf.x) > 0`` filter
    in ``SurfaceGroup.x/y/...`` never drops one and every absolute index stays
    right (design 5).  A partial writeback is never produced.
    """
    if result.snap is not None:
        rows = records.snap_rows
        surfaces = group.surfaces
        for s, surf in enumerate(surfaces):
            row = int(rows[b, s])
            if row < 0:
                continue
            for q, attr in _SNAP_ATTRS:
                setattr(surf, attr, _snap_view(result.snap, q, b, row, mode))
    if result.final is not None:
        for q, attr in _FINAL_ATTRS:
            setattr(rays, attr, _final_view(result.final, q, b, mode))


# ---------------------------------------------------------------------------
# Drift, warnings and refusals
# ---------------------------------------------------------------------------


def _drift_qualnames() -> list[str]:
    """``trace_mirror.check_all()``, cached for the process (plan 3.7)."""
    global _DRIFT
    if _DRIFT is None:
        _DRIFT = list(trace_mirror.check_all())
    return _DRIFT


def _emit(category: type[Warning], message: str, stacklevel: int) -> None:
    """Announce ``message`` in a way no warning filter can turn into a raise.

    Every notice this driver emits describes a *fallback the caller never asked
    for*: the trace library could not be built, a mirrored source drifted, or
    the kernel's Newton loop did not converge on some rays.  In each case the
    caller still gets an answer -- the per-op path's -- and the plan's
    transparency rule (preface, 1.2) is about that outcome, not only about the
    numbers.  A process that runs with ``warnings.simplefilter("error")``
    (``python -W error``, ``pytest -W error``, ``filterwarnings = error``)
    turns any warning into an exception, so emitting these through
    ``warnings.warn`` alone made the fused path RAISE where the per-op path
    returns: 4 of 4 checks in round-3 finding R3-V1-06, both modes.

    So the warning is still emitted -- ``pytest.warns`` and the ordinary
    filters keep working -- but if the filter promotes it, the exception is
    caught here and the same text goes to this module's logger instead.
    ``logging.lastResort`` writes WARNING and above to stderr when the
    application has configured no handler, so nothing is lost; it simply cannot
    become the caller's exception.

    The loud channel for "do not silently fall back" is unchanged and is not a
    warning: ``OPTILAND_METAL_FUSED_TRACE=require`` raises
    :class:`MetalFallbackError` from :func:`_refuse`, and every refusal is
    counted under ``fused_trace_skip:*`` whether or not anybody is listening.

    Args:
        category: The warning class to emit.
        message: The text, used for both channels.
        stacklevel: ``warnings.warn``'s ``stacklevel`` as seen by *the caller
            of this function* (this frame is added here).
    """
    try:
        warnings.warn(message, category, stacklevel=stacklevel + 1)
    except Exception:  # the filter promoted it; it must not reach the caller
        _LOGGER.warning("%s", message)


def _warn_once(key: str, category: type[Warning], message: str) -> None:
    """Emit ``message`` once per process (plan 1.5), never as an exception."""
    if key in _WARNED:
        return
    _WARNED.add(key)
    _emit(category, message, 3)


def _refuse(reason: FusedTraceSkip, raising: bool) -> bool:
    """Count a refusal and raise when ``require`` applies.  Always returns False."""
    count_event(f"fused_trace_skip:{reason.value}")
    if raising and reason not in STRUCTURAL_REASONS:
        raise MetalFallbackError(
            f"OPTILAND_METAL_FUSED_TRACE=require: a candidate bundle was "
            f"refused by the fused trace for the feature reason {reason.value!r}"
        )
    return False


def _trace_record() -> Any:
    """Import the record compiler (WP2's module) lazily.

    Raises:
        ImportError: The module is not in the tree yet.
    """
    from optiland.backend.torch_backend.metal import trace_record

    return trace_record


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def diag_from(group: Any) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The ``(status, iters)`` planes of ``group``'s LAST trace, or None.

    Populated only under ``OPTILAND_METAL_TRACE_DIAG=1``.  The planes live in a
    module-level ``WeakKeyDictionary`` keyed by the group, never on the group
    itself, so ``copy.deepcopy(optic)`` neither copies them nor keeps them
    alive ([fix: L1.8]).

    The planes are never stale (round-1 finding R1-V1-01).  ``fused_trace``
    drops whatever the group's previous trace left here before it does anything
    else, so exactly three outcomes are possible for the trace that just ran:

    * it completed on the kernel -- the planes are that trace's own;
    * it reached the kernel and the driver then discarded the result (the late
      fallback of ``ST_TOL_CROSSOVER``) -- the planes are that discarded
      launch's own, complete for every ``(b, s, i)``, and
      ``fused_trace:late_fallback`` says the rays the caller holds came from
      the per-op path instead (round-1 finding R1-V1-02);
    * it never reached the kernel (the hook is off, the gate refused, DIAG is
      off, or a launch raised) -- None.

    A consumer may therefore zip ``diag_from(group)[0][0]`` with the bundle it
    just traced without checking a ray count.

    Three paths rewrite a group's state without ever reaching ``fused_trace``,
    and each drops the planes itself through :func:`forget_diag` so the third
    outcome above is true for them too: ``SurfaceGroup._fused_metal_trace``'s
    early returns -- the plan 1.5 runtime rollback
    ``OPTILAND_METAL_FUSED_TRACE=0``, a bundle that is not a ``MetalFloat64``,
    and a driver that cannot be imported (round-3 finding R3-V1-04) -- and
    ``BatchTraceResult.install``, which writes one design's rows onto a group
    that the batch never traced through the hook (R3-V2-01).  A batch's own
    diagnostics are never installed as a group's: they are per design, in
    ``BatchTraceResult.status`` and ``.iters``.

    One caveat on the middle case, and it is not a defect: a discarded
    launch's planes record what the KERNEL did, and the crossover is by
    definition the point where the kernel's Newton iteration is allowed to
    diverge from Python's -- that divergence is the reason the launch is
    discarded.  So those planes carry the reason for the fallback, but their
    ``iters`` must not be compared against a per-op prediction the way a
    completed trace's are (measured: 11 of 4096 rays on the Newton row of a
    ``tol = 0`` singlet).  The rays the caller holds come from the per-op path.
    """
    return _DIAG_PLANES.get(group)


def _forget_diag(group: Any) -> None:
    """Drop ``group``'s DIAG planes, so a later trace cannot read them.

    Called at the top of every :func:`fused_trace`: from that moment the
    group's planes are either this trace's own or absent (R1-V1-01).

    ``group`` is whatever the hook was called with, so it need not be weakly
    referenceable; such a key can never have been stored, and forgetting it is
    a no-op rather than a ``TypeError``.
    """
    if not _DIAG_PLANES:
        return
    # A key `_record_diag` could never have stored anyway.
    with contextlib.suppress(TypeError):
        _DIAG_PLANES.pop(group, None)


def forget_diag(group: Any) -> None:
    """Drop ``group``'s DIAG planes from outside the driver.

    :func:`fused_trace` already does this for every trace that reaches it.
    This is the entry point for the callers that rewrite a group's state
    WITHOUT reaching it, and it exists so that neither of them has to import
    this module to do it -- both look it up in ``sys.modules`` and skip the
    call when it is absent, which is what keeps the NumPy and torch-CPU paths
    free of every ``optiland.backend.torch_backend.metal.*`` import
    (plan 0.2.4, ``test_trace_gate.py::test_non_metal_paths_untouched``):

    * ``SurfaceGroup._fused_metal_trace``'s three early returns -- the hook is
      off (the plan 1.5 runtime rollback), the bundle is not a
      ``MetalFloat64``, or the driver cannot be imported.  The upstream per-op
      loop then runs, and an older fused trace's planes must not survive it
      describing a ray count that is no longer the caller's (R3-V1-04);
    * ``BatchTraceResult.install`` -- it writes one design's recorded rows onto
      a group that the batch never traced through the hook, so the planes of
      whatever that group was traced with before would outlive the rows they
      claim to describe (R3-V2-01).
    """
    _forget_diag(group)


def _record_diag(group: Any, result: LaunchResult, *, kept: bool = True) -> None:
    """Store the planes, count ``fused_trace:diag:<bit>`` and warn on Newton.

    Args:
        group: The traced ``SurfaceGroup``.
        result: The launch whose planes these are.
        kept: Whether the driver kept this launch's results.  A discarded
            launch (the late fallback) still records its planes and counts its
            bits -- that is the only place ``ST_TOL_CROSSOVER`` is ever
            observable, and without it nothing in the counters says WHY
            ``fused_trace:late_fallback`` fired (R1-V1-02).  It does not warn
            about Newton: nobody will see the numbers that did not converge,
            because the per-op path is about to recompute the whole trace.
    """
    _DIAG_PLANES[group] = (result.status, result.iters)
    status = result.status.cpu().numpy()
    for bit, name in _DIAG_BITS:
        n = int(np.count_nonzero(status & bit))
        if n:
            count_event(f"fused_trace:diag:{name}", n)
            if kept and bit == trace_layout.ST_NEWTON_NOT_CONVERGED:
                # Through ``_emit``, not ``warnings.warn``: under
                # ``-W error`` this notice used to be the exception that a
                # DIAG=1 trace of a non-converging system raised where the
                # per-op path returns (R3-V1-06).  The per-op path hides
                # non-convergence entirely, so the fused path must not be
                # able to fail on having noticed it.
                _emit(
                    RuntimeWarning,
                    f"fused trace: Newton did not converge for {n} "
                    "(surface, ray) pairs; the per-op path hides this "
                    "(newton_raphson.py:580-581)",
                    3,
                )


# ---------------------------------------------------------------------------
# The hook target
# ---------------------------------------------------------------------------


def fused_trace(group: Any, rays: Any, skip: int, record: bool) -> bool:
    """Trace ``rays`` through ``group`` on the GPU in one kernel.

    This is what ``SurfaceGroup._fused_metal_trace`` calls (plan 3.4).  It
    gates, counts, launches and writes back; ``group.reset()`` has already run
    in the hook and is never called here.

    Args:
        group: The ``SurfaceGroup``.
        rays: The ``RealRays`` bundle (mutated in place on success).
        skip: ``SurfaceGroup.trace``'s ``skip``; anything but 0 is refused.
        record: Whether per-surface snapshots are wanted.  Taken for its truth
            value only, exactly as the per-op path takes it (``if record:``),
            so ``1``, ``numpy.True_`` and ``"image"`` all mean "record every
            surface" here as they do there (R3-V1-02, R3-V1-03).  The richer
            ``bool | str | Sequence[int]`` policy of ``compile_records`` is the
            batch API's, not the hook's.

    Returns:
        bool: True when the kernel handled the trace and ``rays`` and the
        surfaces carry its results; False when the caller's upstream loop must
        run (a refusal or the late fallback, with nothing written).

    Raises:
        MetalFallbackError: Under ``require``, when a candidate bundle is
            refused for a feature reason or takes the late fallback; and, in
            every setting, when the write-completion sentinel survives.
    """
    # R1-V1-01: the previous trace's DIAG planes stop being readable here, at
    # the top, before any early return can leave them behind.  Only this
    # trace's own launch puts planes back.
    _forget_diag(group)
    switch = _switch()
    if switch == "0":
        return False
    require = switch == "require"

    try:
        tr = _trace_record()
    except ImportError as exc:  # pragma: no cover - only before WP2 lands
        # Candidacy cannot be evaluated without the gate, so this never raises
        # under ``require``: a structural non-candidate must stay silent
        # (plan 1.3).  It is still counted and warned about once.
        _warn_once(
            "unavailable",
            FusedTraceUnavailableWarning,
            "fused trace unavailable: the record compiler "
            f"(metal/trace_record.py) could not be imported ({exc}); every "
            "trace runs on the per-op path",
        )
        return _refuse(FusedTraceSkip.UNAVAILABLE, False)

    gate = tr.can_fuse_trace(group, rays, skip)
    candidate = gate.ok or not gate.structural
    if candidate:
        count_event("fused_trace:candidates")
    if not gate.ok:
        return _refuse(gate.reason, require and candidate)

    drift = _drift_qualnames()
    if drift:
        _warn_once(
            "drift",
            FusedTraceDriftWarning,
            "fused trace: mirrored Python source drifted from the fingerprint "
            f"table for {', '.join(drift)}; re-verify with "
            "`python -m optiland.backend.torch_backend.metal.trace_mirror "
            "--update --verified <qualname>=<note>`",
        )
        if _drift_policy() != "warn":
            return _refuse(FusedTraceSkip.MIRROR_DRIFT, require)

    mode, w0, n = gate.mode, gate.w0, gate.n
    try:
        lib = _kernel_library(mode)
    except Exception as exc:
        _warn_once(
            "unavailable",
            FusedTraceUnavailableWarning,
            f"fused trace unavailable: the {mode} trace library could not be "
            f"built ({type(exc).__name__}: {exc}); every trace runs on the "
            "per-op path",
        )
        return _refuse(FusedTraceSkip.UNAVAILABLE, require)

    # R3-V1-02 / R3-V1-03: ``record`` comes from ``SurfaceGroup.trace``, whose
    # upstream contract is a bool that is only ever used for its truth value
    # (``surface.trace(rays, record=record)`` -> ``if record:``).  The hook may
    # not narrow that domain to ``compile_records``' richer
    # ``bool | str | Sequence[int]``: with the hook on, ``record=1`` and
    # ``record=np.True_`` used to raise ``TypeError`` where the per-op path
    # returns, and ``record="image"`` / ``record=[1, 3]`` used to record one or
    # two of the eight Cooke rows where the per-op path records all eight --
    # silently, so flipping OPTILAND_METAL_FUSED_TRACE changed the answer.
    # The batch API keeps the rich domain; it calls ``compile_records``
    # directly.  A value that cannot be taken for truth at all (a multi-element
    # array) raises here exactly what ``if record:`` raises on the per-op path.
    records = tr.compile_records(group, w0, mode, record=bool(record))

    budget = _memory_fraction() * float(torch.mps.recommended_max_memory())
    if tr.memory_bytes(records, n, 1, True) > budget:
        return _refuse(FusedTraceSkip.MEMORY, require)

    launch = _pack_launch(rays, mode, n)
    result = _launch_slabs(
        lib,
        records,
        launch,
        launch_stride=0,
        N=n,
        write_final=True,
        mode=mode,
        count=True,
    )

    if bool(result.late_fallback_designs.any()):
        # Nothing has been written into ``rays`` or the surfaces; the caller's
        # upstream loop runs on the same bundle.  The planes of the launch that
        # is being discarded are still recorded and counted (R1-V1-01,
        # R1-V1-02): they are complete -- every slab ran and the sentinel check
        # passed above -- and they carry the ``ST_TOL_CROSSOVER`` bit that is
        # the reason for this fallback.
        count_event("fused_trace:late_fallback")
        if _diag():
            _record_diag(group, result, kept=False)
        if require:
            raise MetalFallbackError(
                "OPTILAND_METAL_FUSED_TRACE=require: a Newton seed crossed the "
                "tolerance crossover (ST_TOL_CROSSOVER), so the trace falls "
                "back to the per-op path"
            )
        return False

    _write_back(group, rays, records, result, mode)
    count_event("fused_trace:traces")
    count_event("fused_trace:designs", int(records.B))
    count_event("fused_trace:surface_steps", int(records.B) * n * (int(records.S) - 1))
    if _diag():
        _record_diag(group, result)
    return True
