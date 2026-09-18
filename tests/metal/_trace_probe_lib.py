"""Compile and drive ``kernels/trace.metal`` from the WP1 unit tests.

This is the WP1 lane's own harness: it builds the third per-mode Metal library
(mode headers + ``conic.metal`` + ``trace_layout.h`` + ``trace.metal``) exactly
as design 4.1 specifies, allocates the ten buffers of
``trace_layout.BUFFER_ORDER`` and dispatches an entry point.  It deliberately
does NOT import ``metal/trace.py`` (WP3's driver, which does not exist yet), so
the kernel can be unit-tested before the driver lands and, afterwards, without
the gate or the record compiler in the way.

Everything a test compares is a RAW COMPONENT (``hi``/``lo`` float32 arrays in
df64, int64 bit patterns in sf64), never a decoded float64: ``decode(encode(x))
!= x`` at the 2**-48 level, and plan 7.1 tier A is component equality (day-1
finding P1).
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence  # noqa: TC003
from pathlib import Path  # noqa: TC003
from typing import Any

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import numpy as np  # noqa: E402

from optiland.backend.torch_backend.metal import compile as _compile  # noqa: E402
from optiland.backend.torch_backend.metal import encode, trace_layout  # noqa: E402
from optiland.backend.torch_backend.metal.library import (  # noqa: E402
    HEADER_PREREQUISITES,
    HEADERS,
    REQUIRED_HEADERS,
)

MODES: tuple[str, ...] = ("df64", "sf64")

#: The amalgamation tail, after the mode's header stack (design 4.1).
TRACE_FILES: tuple[str, ...] = ("conic.metal", "trace_layout.h", "trace.metal")

KERNEL_DIR: Path = _compile._KERNEL_DIR
TRACE_METAL: Path = KERNEL_DIR / "trace.metal"

ENTRY: dict[str, str] = {
    "df64": "trace_surfaces_df64",
    "sf64": "trace_surfaces_sf64",
}

#: Machine epsilon of each representation (``metal/tensor.py`` MACHINE_EPS).
MACHINE_EPS: dict[str, float] = {"df64": 2.0**-48, "sf64": 2.0**-53}

#: Pre-fill for the distance probes' ``iters`` byte, so an unvisited thread
#: is visible there exactly as ``ITERS_UNWRITTEN`` makes it visible in a trace.
ITERS_PROBE_UNWRITTEN: int = trace_layout.ITERS_UNWRITTEN

_LIBS: dict[tuple[str, bool, tuple[str, ...]], Any] = {}


# ---------------------------------------------------------------------------
# Source amalgamation
# ---------------------------------------------------------------------------


def headers(mode: str) -> list[str]:
    """The mode's header stack, filtered by presence and prerequisites.

    Mirrors ``MetalLibrary.__init__`` (library.py:235-248), the stricter
    recipe design 4.1 asks for: a partial header set raises
    ``FileNotFoundError`` instead of producing a raw Metal error.
    """
    present = [h for h in HEADERS[mode] if (KERNEL_DIR / h).is_file()]
    present = [
        h for h in present if all(p in present for p in HEADER_PREREQUISITES.get(h, ()))
    ]
    missing = [h for h in REQUIRED_HEADERS[mode] if h not in present]
    if missing:
        raise FileNotFoundError(
            f"{mode}: required kernel header(s) missing from {KERNEL_DIR}: {missing}"
        )
    return present


#: The round-0 divergence-injection sites of plan section 6.  Each is an
#: ``#ifdef OPTILAND_TRACE_BREAK_<name>`` in ``trace.metal`` that replaces one
#: mirrored expression with the "obvious" rewrite, so a conformance test can be
#: shown to FAIL when the kernel stops mirroring.  ``modes`` is the set of
#: modes in which the rewrite is arithmetically visible.
BREAK_SITES: dict[str, tuple[str, ...]] = {
    # 48-bit df64 division against the host float64 value in the record; sf64
    # division is correctly rounded, so only df64 must diverge.
    "U_INKERNEL": ("df64",),
    # Horner instead of the term-by-term Python sum: operand side (df64) and,
    # from two coefficients on, association (both modes).
    "HORNER": ("df64", "sf64"),
    # One composed rotation matrix instead of three sequential rotations.
    "COMPOSED_ROTATION": ("df64", "sf64"),
    # `df::sqr` instead of `mul(x, x)`; sf64's `sqr` IS `mul(a, a)`.
    "SQR": ("df64",),
}


def trace_source(mode: str, *, probes: bool = False, breaks: Sequence[str] = ()) -> str:
    """The amalgamated trace-library source for ``mode``.

    Args:
        mode: ``"df64"`` or ``"sf64"``.
        probes: Prepend ``#define OPTILAND_TRACE_PROBES 1`` so the probe
            kernels are compiled in (design 4.13).
        breaks: Round-0 injection sites to enable (keys of
            :data:`BREAK_SITES`), each prepended as
            ``#define OPTILAND_TRACE_BREAK_<name> 1``.
    """
    prologue = "#define OPTILAND_TRACE_PROBES 1\n" if probes else ""
    for name in breaks:
        if name not in BREAK_SITES:
            raise KeyError(f"unknown break site {name!r}; have {sorted(BREAK_SITES)}")
        prologue += f"#define OPTILAND_TRACE_BREAK_{name} 1\n"
    return prologue + _compile.kernel_source(*headers(mode), *TRACE_FILES)


def trace_library(
    mode: str, *, probes: bool = False, breaks: Sequence[str] = ()
) -> Any:
    """Compile (once per process) and warm up the trace library for ``mode``.

    Metal builds pipeline states lazily, so a compile that "succeeds" can still
    fail at the first dispatch (day-1 Q12 finding).  Every entry point is
    therefore dispatched once, on a one-thread problem, at build time.
    """
    key = (mode, probes, tuple(breaks))
    lib = _LIBS.get(key)
    if lib is None:
        lib = _compile.compile_library(trace_source(mode, probes=probes, breaks=breaks))
        _warm_up(lib, mode)
        _LIBS[key] = lib
    return lib


def _warm_up(lib: Any, mode: str) -> None:
    """Dispatch the entry point once at B = N = 1, S = 1 to force the pipeline."""
    run_trace(
        lib,
        mode,
        launch=np.zeros((trace_layout.Q_PLANES, 1, 1)),
        surf_int=np.zeros((1, 1, trace_layout.SI_STRIDE), dtype=np.int32),
        surf_real=np.zeros((1, 1, trace_layout.SR_STRIDE)),
        coef=np.zeros((1, 1, 0)),
        n_rows=0,
        write_final=False,
    )


# ---------------------------------------------------------------------------
# Buffer plumbing
# ---------------------------------------------------------------------------


def _torch() -> Any:
    import torch

    return torch


def to_device(array: np.ndarray, mode: str) -> list[Any]:
    """Encode a float64 array into the mode's flat device component buffers."""
    flat = np.ascontiguousarray(np.asarray(array, dtype=np.float64).reshape(-1))
    if mode == "df64":
        hi, lo = encode.to_mps_df64(flat)
        return [hi, lo]
    return [encode.to_mps_sf64(flat)]


def empty_device(n: int, mode: str) -> list[Any]:
    """Allocate the mode's component buffers for ``n`` R elements, zero-filled."""
    torch = _torch()
    if mode == "df64":
        return [torch.zeros(n, dtype=torch.float32, device="mps") for _ in range(2)]
    return [torch.zeros(n, dtype=torch.int64, device="mps")]


def raw_components(buffers: list[Any], shape: tuple[int, ...]) -> list[np.ndarray]:
    """Host copies of the raw components, reshaped; never decoded."""
    return [b.cpu().numpy().reshape(shape) for b in buffers]


def dims_buffer(
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
) -> Any:
    """The ``dims int32[D_SIZE]`` buffer, slots per ``trace_layout``."""
    torch = _torch()
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


#: The |N| floor of the infinite-radius branch of ``_conic_intersection_distance``
#: (``standard.py:88``: ``be.where(be.abs(rays.N) > 1e-14, rays.N, 1e-14)``).  It is
#: a fixed literal in the Python source, NOT a machine epsilon, and it is not
#: float32-exact, so it reaches the kernel encoded in ``consts[C_NFLOOR]``.
#: ``test_plane_and_std_inf_differ_at_zero_slope`` pins that WP3's driver binds
#: the same value.
N_FLOOR: float = 1e-14


def consts_array(mode: str, *, n_floor: float | None = None) -> np.ndarray:
    """The ``consts R[C_SIZE]`` array: machine epsilon and the |N| floor."""
    c = np.zeros(trace_layout.C_SIZE, dtype=np.float64)
    c[trace_layout.C_EPS] = MACHINE_EPS[mode]
    c[trace_layout.C_NFLOOR] = N_FLOOR if n_floor is None else float(n_floor)
    return c


def pack_rays(rays: Any, mode: str, n: int) -> tuple[list[Any], tuple[int, int, int]]:
    """Pack a live ``RealRays`` bundle into launch buffers, WITHOUT re-encoding.

    The nine launch planes are concatenated straight out of the bundle's
    component tensors, so the kernel sees the very words the per-op path reads:
    a ``to_numpy`` / ``be.array`` round trip would re-encode and lose the
    sub-2**-48 part of every df64 value, which raw-component equality then
    reports as a kernel bug.  A uniform plane (``rays.w``) is broadcast.
    """
    torch = _torch()
    planes: list[list[Any]] = []
    for attr in ("x", "y", "z", "L", "M", "N", "i", "opd", "w"):
        value = getattr(rays, attr)
        comps = []
        for c in value.detach().components:
            t = c.contiguous().reshape(-1)
            if t.numel() == 1 and n != 1:
                t = t.expand(n).contiguous()
            if t.numel() != n:
                raise ValueError(f"rays.{attr}: expected {n} elements, got {t.numel()}")
            comps.append(t)
        planes.append(comps)
    ncomp = 2 if mode == "df64" else 1
    bufs = [
        torch.cat([planes[q][k] for q in range(trace_layout.Q_PLANES)])
        for k in range(ncomp)
    ]
    return bufs, (trace_layout.Q_PLANES, 1, n)


def run_trace(
    lib: Any,
    mode: str,
    *,
    launch: np.ndarray | None = None,
    launch_bufs: tuple[list[Any], tuple[int, int, int]] | None = None,
    surf_int: np.ndarray,
    surf_real: np.ndarray,
    coef: np.ndarray,
    n_rows: int,
    write_final: bool,
    consts: np.ndarray | None = None,
    ray_base: int = 0,
    design_base: int = 0,
    threads: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Allocate, bind and dispatch one slab of the trace kernel.

    Args:
        lib: A library from :func:`trace_library`.
        mode: ``"df64"`` or ``"sf64"``.
        launch: ``float64[Q_PLANES, Lb, N]``; ``Lb`` is 1 (shared launch,
            ``launch_stride = 0``) or ``B`` (per design, ``launch_stride = 1``).
        surf_int: ``int32[B, S, SI_STRIDE]``.
        surf_real: ``float64[B, S, SR_STRIDE]``.
        coef: ``float64[B, S, C]`` (``C`` may be 0).
        n_rows: Number of recorded snapshot rows.
        write_final: Write the ``final`` plane set.
        consts: ``float64[C_SIZE]``; defaults to :func:`consts_array`.
        ray_base: ``dims[D_RAY_BASE]``; the thread's ray is ``ray_base + g.x``.
        design_base: ``dims[D_DESIGN_BASE]``; the design is ``design_base + g.y``.
        threads: ``(n_extent, b_extent)``; defaults to the full problem minus
            the bases.

    Returns:
        dict with the raw component buffers of ``snap`` / ``final`` (reshaped,
        or ``None`` when not written) and the ``status`` / ``iters`` planes as
        ``uint8[B, S, N]`` host arrays.
    """
    torch = _torch()
    if (launch is None) == (launch_bufs is None):
        raise ValueError("pass exactly one of launch / launch_bufs")
    Q, Lb, N = launch.shape if launch is not None else launch_bufs[1]
    B, S, P = surf_real.shape
    C = coef.shape[2]
    if Q != trace_layout.Q_PLANES:
        raise ValueError(f"launch must have {trace_layout.Q_PLANES} planes, got {Q}")
    if P != trace_layout.SR_STRIDE:
        raise ValueError(f"surf_real stride must be {trace_layout.SR_STRIDE}, got {P}")
    if surf_int.shape != (B, S, trace_layout.SI_STRIDE):
        raise ValueError(f"surf_int must be {(B, S, trace_layout.SI_STRIDE)}")
    if Lb not in (1, B):
        raise ValueError(f"launch second axis must be 1 or B={B}, got {Lb}")
    launch_stride = 0 if Lb == 1 else 1

    launch_components = (
        to_device(launch, mode) if launch is not None else list(launch_bufs[0])
    )
    bufs: list[Any] = list(launch_components)
    bufs.append(
        torch.tensor(
            np.ascontiguousarray(surf_int.reshape(-1), dtype=np.int32),
            dtype=torch.int32,
            device="mps",
        )
    )
    bufs += to_device(surf_real, mode)
    bufs += to_device(coef if C else np.zeros(1), mode)
    bufs.append(
        dims_buffer(
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
    )
    bufs += to_device(consts_array(mode) if consts is None else consts, mode)

    snap_shape = (trace_layout.S_PLANES, B, n_rows, N)
    fin_shape = (trace_layout.F_PLANES, B, N)
    snap_bufs = empty_device(max(1, int(np.prod(snap_shape))), mode)
    fin_bufs = empty_device(int(np.prod(fin_shape)), mode)
    bufs += snap_bufs
    bufs += fin_bufs

    status = torch.zeros(B * S * N, dtype=torch.uint8, device="mps")
    iters = torch.full(
        (B * S * N,), trace_layout.ITERS_UNWRITTEN, dtype=torch.uint8, device="mps"
    )
    bufs += [status, iters]

    nx, ny = threads if threads is not None else (N - ray_base, B - design_base)
    getattr(lib, ENTRY[mode])(*bufs, threads=[nx, ny, 1])
    torch.mps.synchronize()

    return {
        "snap": raw_components(snap_bufs, snap_shape) if n_rows else None,
        "final": raw_components(fin_bufs, fin_shape) if write_final else None,
        "status": status.cpu().numpy().reshape(B, S, N),
        "iters": iters.cpu().numpy().reshape(B, S, N),
        "launch_raw": raw_components(launch_components, (Q, Lb, N)),
        "bindings": len(bufs),
    }


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------


def blank_tables(
    B: int, S: int, *, C: int = 0, snap_rows: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Zeroed ``(surf_int, surf_real, coef, n_rows)`` with every row recorded.

    ``snap_rows=False`` sets every ``SI_SNAPROW`` to -1 (nothing recorded), the
    ``record=False`` shape of the contract.
    """
    surf_int = np.zeros((B, S, trace_layout.SI_STRIDE), dtype=np.int32)
    surf_real = np.zeros((B, S, trace_layout.SR_STRIDE), dtype=np.float64)
    coef = np.zeros((B, S, C), dtype=np.float64)
    surf_int[:, 0, trace_layout.SI_GEOM] = trace_layout.GEOM_OBJECT
    surf_int[:, 1:, trace_layout.SI_GEOM] = trace_layout.GEOM_PLANE
    if snap_rows:
        surf_int[:, :, trace_layout.SI_SNAPROW] = np.arange(S, dtype=np.int32)
        n_rows = S
    else:
        surf_int[:, :, trace_layout.SI_SNAPROW] = -1
        n_rows = 0
    return surf_int, surf_real, coef, n_rows


def random_launch(
    N: int, *, Lb: int = 1, seed: int = 11, wavelength: float = 0.55
) -> np.ndarray:
    """A plausible ``float64[Q_PLANES, Lb, N]`` launch bundle.

    Directions are normalised, intensity is 1, OPD is 0 and the wavelength
    plane is uniform, as ``RayGenerator.generate_rays`` produces.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros((trace_layout.Q_PLANES, Lb, N), dtype=np.float64)
    for b in range(Lb):
        out[trace_layout.Q_X, b] = rng.uniform(-8, 8, N)
        out[trace_layout.Q_Y, b] = rng.uniform(-8, 8, N)
        out[trace_layout.Q_Z, b] = rng.uniform(-40, -1, N)
        ell = rng.uniform(-0.3, 0.3, N)
        em = rng.uniform(-0.3, 0.3, N)
        out[trace_layout.Q_L, b] = ell
        out[trace_layout.Q_M, b] = em
        out[trace_layout.Q_N, b] = np.sqrt(1 - ell * ell - em * em)
        out[trace_layout.Q_I, b] = 1.0
        out[trace_layout.Q_OPD, b] = 0.0
        out[trace_layout.Q_W, b] = wavelength
    return out


def mirror_tables(
    B: int, S: int, *, snap_rows: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Tables whose every surface row is an EXACT sign flip of ``N``.

    Rows ``s >= 1`` are reflective planes at the origin with a zero pose, no
    aperture, no absorption and ``n_pre = 0``.  Fed :func:`mirror_launch`
    (every ray exactly on the vertex plane) the whole surface step is exact and
    predictable:

    * ``t = -z / N = -0.0`` and ``advance`` leaves ``x, y, z`` untouched;
    * ``opd + t * 0`` leaves the OPD untouched and the intensity is untouched;
    * the plane normal is ``(0, 0, 1)``, so ``dot = N`` and ``reflect`` gives
      ``L, M, -N`` -- ``N - (2|N|) * sign(N)`` is exact in both
      representations (a power-of-two scaling and a cancellation whose result
      is representable).

    So after ``k`` surface rows the state is the launch state with
    ``N * (-1)**k``, which is a per-row DISTINGUISHABLE value: an addressing
    test on these tables cannot pass by writing the same thing everywhere.
    """
    surf_int = np.zeros((B, S, trace_layout.SI_STRIDE), dtype=np.int32)
    surf_real = np.zeros((B, S, trace_layout.SR_STRIDE), dtype=np.float64)
    coef = np.zeros((B, S, 0), dtype=np.float64)
    surf_int[:, 0, trace_layout.SI_GEOM] = trace_layout.GEOM_OBJECT
    surf_int[:, 1:, trace_layout.SI_GEOM] = trace_layout.GEOM_PLANE
    surf_int[:, 1:, trace_layout.SI_FLAGS] = trace_layout.FL_REFLECTIVE
    if snap_rows:
        surf_int[:, :, trace_layout.SI_SNAPROW] = np.arange(S, dtype=np.int32)
        n_rows = S
    else:
        surf_int[:, :, trace_layout.SI_SNAPROW] = -1
        n_rows = 0
    return surf_int, surf_real, coef, n_rows


def mirror_launch(
    N: int, *, Lb: int = 1, seed: int = 11, wavelength: float = 0.55
) -> np.ndarray:
    """A launch bundle for :func:`mirror_tables`: every ray on ``z = 0``.

    No component is zero apart from ``z`` and the OPD, so no addition in the
    surface step can flip the sign of a zero, and each design row carries a
    different bundle so the design axis is observable.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros((trace_layout.Q_PLANES, Lb, N), dtype=np.float64)
    for b in range(Lb):
        out[trace_layout.Q_X, b] = rng.uniform(1.0, 8.0, N) * (1.0 + b)
        out[trace_layout.Q_Y, b] = rng.uniform(1.0, 8.0, N) * (1.0 + b)
        out[trace_layout.Q_Z, b] = 0.0
        ell = rng.uniform(0.05, 0.3, N)
        em = rng.uniform(0.05, 0.3, N)
        out[trace_layout.Q_L, b] = ell
        out[trace_layout.Q_M, b] = em
        out[trace_layout.Q_N, b] = np.sqrt(1 - ell * ell - em * em)
        out[trace_layout.Q_I, b] = 1.0 - 0.1 * b
        out[trace_layout.Q_OPD, b] = 0.0
        out[trace_layout.Q_W, b] = wavelength
    return out


# ---------------------------------------------------------------------------
# Probe drivers (design 4.13).  Every probe takes its ray data as a plane-major
# ``float64[planes, n]`` array (plane ``k`` of ray ``i`` at ``k * n + i``), its
# real parameters as a full ``surf_real`` ROW of ``SR_STRIDE`` slots (so the
# probe exercises the same slot mapping the trace loop reads) and its integer
# parameters in one ``int32`` buffer ``[n, param]``.  Results come back as RAW
# COMPONENTS, never decoded.
# ---------------------------------------------------------------------------

#: ``ip`` slots, mirroring the ``OT_PROBE_*`` defines in ``trace.metal``.
PROBE_N, PROBE_PARAM = 0, 1


def int_params(n: int, param: int = 0) -> Any:
    """The probes' ``const device int*`` buffer."""
    torch = _torch()
    return torch.tensor([int(n), int(param)], dtype=torch.int32, device="mps")


def param_row(**slots: float) -> np.ndarray:
    """A ``surf_real`` row with the named ``trace_layout.SR_*`` slots set.

    Example: ``param_row(AP0=12.5**2, AP1=0.0)``.
    """
    row = np.zeros(trace_layout.SR_STRIDE, dtype=np.float64)
    for name, value in slots.items():
        row[getattr(trace_layout, f"SR_{name}")] = value
    return row


def _run(lib: Any, name: str, bufs: list[Any], n: int) -> None:
    torch = _torch()
    getattr(lib, name)(*bufs, threads=[n, 1, 1])
    torch.mps.synchronize()


def run_probe_pose(
    lib: Any, mode: str, *, which: str, planes: np.ndarray, pose: np.ndarray, flags: int
) -> list[np.ndarray]:
    """``probe_localize`` / ``probe_globalize``: 6 planes in, 6 planes out.

    Args:
        which: ``"localize"`` or ``"globalize"``.
        planes: ``float64[6, n]`` -- x, y, z, L, M, N.
        pose: ``float64[SR_STRIDE]`` -- the surface's real row.
        flags: ``SI_FLAGS`` (only the ``FL_HAS_R*`` bits are read).

    Returns:
        The raw components of the ``float64[6, n]`` result.
    """
    if which not in ("localize", "globalize"):
        raise ValueError(f"which must be localize or globalize, got {which!r}")
    n = int(planes.shape[1])
    in_bufs = to_device(planes, mode)
    pose_bufs = to_device(pose, mode)
    out_bufs = empty_device(6 * n, mode)
    bufs = [*in_bufs, *pose_bufs, int_params(n, flags), *out_bufs]
    _run(lib, f"probe_{which}_{mode}", bufs, n)
    return raw_components(out_bufs, (6, n))


def run_probe_contains(
    lib: Any, mode: str, *, x: np.ndarray, y: np.ndarray, params: np.ndarray, code: int
) -> np.ndarray:
    """``probe_contains``: a ``uint8`` inside-mask, one byte per point."""
    torch = _torch()
    n = int(np.asarray(x).size)
    planes = np.stack(
        [np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)]
    )
    in_bufs = to_device(planes, mode)
    par_bufs = to_device(params, mode)
    out = torch.zeros(n, dtype=torch.uint8, device="mps")
    bufs = [*in_bufs, *par_bufs, int_params(n, code), out]
    _run(lib, f"probe_contains_{mode}", bufs, n)
    return out.cpu().numpy()


def run_probe_interact(
    lib: Any, mode: str, *, planes: np.ndarray, params: np.ndarray, reflective: bool
) -> tuple[list[np.ndarray], np.ndarray]:
    """``probe_interact``: 6 planes in (L0, M0, N0, nx, ny, nz), 3 out + status.

    Returns:
        ``(raw components of float64[3, n], status uint8[n])``.
    """
    torch = _torch()
    n = int(planes.shape[1])
    in_bufs = to_device(planes, mode)
    par_bufs = to_device(params, mode)
    out_bufs = empty_device(3 * n, mode)
    status = torch.zeros(n, dtype=torch.uint8, device="mps")
    flags = trace_layout.FL_REFLECTIVE if reflective else 0
    bufs = [*in_bufs, *par_bufs, int_params(n, flags), *out_bufs, status]
    _run(lib, f"probe_interact_{mode}", bufs, n)
    return raw_components(out_bufs, (3, n)), status.cpu().numpy()


def run_probe_pow_scalar(
    lib: Any, mode: str, *, values: np.ndarray, e: float
) -> list[np.ndarray]:
    """``probe_pow_scalar``: ``pow_scalar(x, e)`` over a flat array."""
    torch = _torch()
    n = int(np.asarray(values).size)
    in_bufs = to_device(values, mode)
    e_buf = torch.tensor([float(e)], dtype=torch.float32, device="mps")
    out_bufs = empty_device(n, mode)
    bufs = [*in_bufs, e_buf, int_params(n), *out_bufs]
    _run(lib, f"probe_pow_scalar_{mode}", bufs, n)
    return raw_components(out_bufs, (n,))


def run_probe_propagate(
    lib: Any, mode: str, *, planes: np.ndarray, params: np.ndarray, absorbing: bool
) -> list[np.ndarray]:
    """``probe_propagate``: 10 planes in (x, y, z, L, M, N, i, opd, w, t).

    Returns:
        The raw components of the ``float64[5, n]`` result
        (x, y, z, intensity, opd).
    """
    n = int(planes.shape[1])
    in_bufs = to_device(planes, mode)
    par_bufs = to_device(params, mode)
    out_bufs = empty_device(5 * n, mode)
    flags = trace_layout.FL_ABSORBING if absorbing else 0
    bufs = [*in_bufs, *par_bufs, int_params(n, flags), *out_bufs]
    _run(lib, f"probe_propagate_{mode}", bufs, n)
    return raw_components(out_bufs, (5, n))


# ---------------------------------------------------------------------------
# Source scanners (the literal discipline of design 4.2, rules 2 and 5)
# ---------------------------------------------------------------------------

_LIT_RE = re.compile(r"\blit\(\s*([^()]*?)\s*\)")
_FMA_RE = re.compile(r"\b(?:df::mul_add|sf::fma|metal::fma|fma)\s*\(")
#: ``lit``'s own declaration, ``static inline df64 lit(float v) { ... }``.
_LIT_DECL_RE = re.compile(r"^(?:const\s+)?(?:float|int|uint)\s+\w+$")


def lit_arguments(source: str) -> list[str]:
    """Every textual argument of a ``lit(...)`` CALL in ``source``.

    Comments are stripped first (a comment naming ``lit(1e-14)`` is prose, not
    a constant the kernel uses) and ``lit``'s own parameter declaration is not
    a call site.
    """
    return [
        m.group(1)
        for m in _LIT_RE.finditer(_COMMENT_RE.sub(_blank, source))
        if not _LIT_DECL_RE.match(m.group(1))
    ]


def lit_value(arg: str) -> float:
    """Parse a ``lit()`` argument (``1e3``, ``-0.5f``, ``2.0f``) as a float.

    Raises:
        ValueError: If the argument is not a plain numeric literal, which is
            itself a violation of design 4.2 rule 2.
    """
    return float(arg.strip().rstrip("fF"))


def non_float32_exact_lits(source: str) -> list[tuple[str, float]]:
    """``lit()`` arguments that are not exactly representable as float32."""
    bad: list[tuple[str, float]] = []
    for arg in lit_arguments(source):
        try:
            value = lit_value(arg)
        except ValueError:
            continue  # a runtime float; see dynamic_lit_arguments()
        if float(np.float32(value)) != value:
            bad.append((arg, value))
    return bad


def fma_call_lines(source: str) -> list[tuple[int, str]]:
    """Lines calling a fused multiply-add (forbidden: design 4.2 rule 5)."""
    return [
        (n, line)
        for n, line in enumerate(source.splitlines(), start=1)
        if _FMA_RE.search(line) and not line.lstrip().startswith("//")
    ]


def dynamic_lit_arguments(source: str) -> list[str]:
    """``lit()`` arguments that are not numeric literals at all.

    Design 4.2 rule 2 is about *literals*; exactly one call site converts a
    runtime float -- ``pow_scalar``'s ``O::pow(x, O::lit(e))`` fallback, whose
    ``e`` is a SIMD-uniform exponent taken from a coefficient loop index.  The
    unit test asserts this list is exactly that one name, so a second dynamic
    conversion cannot be added without a review.
    """
    dynamic: list[str] = []
    for arg in lit_arguments(source):
        try:
            lit_value(arg)
        except ValueError:
            dynamic.append(arg)
    return dynamic


_FLOAT_TOKEN = (
    r"(?:\d+\.\d*(?:[eE][-+]?\d+)?[fF]?"
    r"|\.\d+(?:[eE][-+]?\d+)?[fF]?"
    r"|\d+[eE][-+]?\d+[fF]?"
    r"|\d+[fF])"
)
_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_BARE_FLOAT_RE = re.compile(
    rf"(?:[\w\)]\s*[-+*/]\s*{_FLOAT_TOKEN}(?![\w.])"
    rf"|(?<![\w.]){_FLOAT_TOKEN}\s*[-+*/]\s*[\w\(])"
)


def _blank(match: re.Match[str]) -> str:
    """Replace a match with spaces, keeping newlines so line numbers hold."""
    return "".join("\n" if ch == "\n" else " " for ch in match.group(0))


def _without_comments_and_lits(source: str) -> str:
    """``source`` with comments and every ``lit(...)`` call blanked out."""
    return _LIT_RE.sub(_blank, _COMMENT_RE.sub(_blank, source))


def bare_float_literal_lines(source: str) -> list[tuple[int, str]]:
    """Lines using a float literal as an operand of ``+ - * /`` outside ``lit()``.

    Design 4.2 rule 2: every constant that takes part in the mirrored
    arithmetic goes through ``lit()`` (float32-exact) or an encoded table slot,
    so a bare ``x * 2.0f`` would be a df64-vs-float32 mismatch waiting to
    happen.  Exponent-selector comparisons (``e == -1.0f``) are not arithmetic
    and are not flagged; nor are integer indices and ``ulong`` suffixes.
    """
    stripped = _without_comments_and_lits(source)
    lines = source.splitlines()
    return [
        (n, lines[n - 1])
        for n, line in enumerate(stripped.splitlines(), start=1)
        if _BARE_FLOAT_RE.search(line)
    ]


# ---------------------------------------------------------------------------
# Distance / sag / normal probe drivers (WP1 part 2, design 4.6-4.7)
#
# These four probes need more than the two ``ip`` slots the earlier ones use:
# the geometry code, the aperture code, the coefficient count and ``max_iter``
# all select a branch inside the functions they drive.  ``int_params_geom``
# renders the longer buffer; the older probes read only slots 0-1, so the two
# layouts coexist.
# ---------------------------------------------------------------------------

#: ``ip`` slots, mirroring the ``OT_PROBE_*`` defines in ``trace.metal``.
PROBE_GEOM, PROBE_APCODE, PROBE_NCOEF, PROBE_MAXITER = 2, 3, 4, 5
PROBE_IP_SLOTS = 6


def int_params_geom(
    n: int,
    *,
    flags: int = 0,
    geom: int = 0,
    apcode: int = 0,
    ncoef: int = 0,
    max_iter: int = 0,
) -> Any:
    """The six-slot ``const device int*`` buffer of the distance probes."""
    torch = _torch()
    slots = [0] * PROBE_IP_SLOTS
    slots[PROBE_N] = int(n)
    slots[PROBE_PARAM] = int(flags)
    slots[PROBE_GEOM] = int(geom)
    slots[PROBE_APCODE] = int(apcode)
    slots[PROBE_NCOEF] = int(ncoef)
    slots[PROBE_MAXITER] = int(max_iter)
    return torch.tensor(slots, dtype=torch.int32, device="mps")


def _coef_buffers(coefficients: np.ndarray | None, mode: str) -> tuple[list[Any], int]:
    """Device buffers for a coefficient list (never zero-length) and its count."""
    values = np.asarray(
        [] if coefficients is None else coefficients, dtype=np.float64
    ).reshape(-1)
    payload = values if values.size else np.zeros(1, dtype=np.float64)
    return to_device(payload, mode), int(values.size)


def run_probe_sag(
    lib: Any,
    mode: str,
    *,
    x: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    geom: int,
    coefficients: np.ndarray | None = None,
    flags: int = 0,
) -> list[np.ndarray]:
    """``probe_sag``: ``sag_of(geom, x, y)`` over a flat array of points.

    ``flags`` reaches ``sag_of`` as the row's ``SI_FLAGS``; only the operand-side
    bits ``FL_K1_ON_RIGHT`` / ``FL_R_ON_RIGHT`` are read there (R2-V1-03).
    """
    n = int(np.asarray(x).size)
    planes = np.stack(
        [np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)]
    )
    in_bufs = to_device(planes, mode)
    par_bufs = to_device(params, mode)
    coef_bufs, ncoef = _coef_buffers(coefficients, mode)
    out_bufs = empty_device(n, mode)
    ip = int_params_geom(n, flags=flags, geom=geom, ncoef=ncoef)
    bufs = [*in_bufs, *par_bufs, *coef_bufs, ip, *out_bufs]
    _run(lib, f"probe_sag_{mode}", bufs, n)
    return raw_components(out_bufs, (n,))


def run_probe_normal(
    lib: Any,
    mode: str,
    *,
    x: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    geom: int,
    coefficients: np.ndarray | None = None,
    flags: int = 0,
) -> list[np.ndarray]:
    """``probe_normal``: the raw components of ``float64[3, n]`` (nx, ny, nz).

    ``flags`` reaches ``normal_of`` as the row's ``SI_FLAGS``; only the
    operand-side bits are read there (R2-V1-03).
    """
    n = int(np.asarray(x).size)
    planes = np.stack(
        [np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)]
    )
    in_bufs = to_device(planes, mode)
    par_bufs = to_device(params, mode)
    coef_bufs, ncoef = _coef_buffers(coefficients, mode)
    out_bufs = empty_device(3 * n, mode)
    ip = int_params_geom(n, flags=flags, geom=geom, ncoef=ncoef)
    bufs = [*in_bufs, *par_bufs, *coef_bufs, ip, *out_bufs]
    _run(lib, f"probe_normal_{mode}", bufs, n)
    return raw_components(out_bufs, (3, n))


def run_probe_distance(
    lib: Any,
    mode: str,
    *,
    planes: np.ndarray,
    params: np.ndarray,
    geom: int,
    flags: int = 0,
    apcode: int = 0,
    coefficients: np.ndarray | None = None,
    max_iter: int = 0,
    consts: np.ndarray | None = None,
    newton: bool = False,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """``probe_distance`` / ``probe_newton``: 6 planes in, ``t`` + status + iters.

    Args:
        planes: ``float64[6, n]`` -- x, y, z, L, M, N in the LOCAL frame.
        params: ``float64[SR_STRIDE]`` -- the surface's real row.
        geom: ``SI_GEOM``.
        flags: ``SI_FLAGS`` (``FL_RADIUS_INF`` / ``FL_AP_IN_ROOT`` are read).
        apcode: ``SI_APCODE``, used only under ``FL_AP_IN_ROOT``.
        coefficients: the aspheric coefficient list, lowest order first.
        max_iter: ``SI_MAXITER`` (Newton only).
        consts: ``float64[C_SIZE]``; defaults to :func:`consts_array`.
        newton: drive ``newton_distance`` directly instead of the SI_GEOM
            switch (so no ``MISS`` bit is raised).

    Returns:
        ``(raw components of float64[n], status uint8[n], iters uint8[n])``.
    """
    torch = _torch()
    n = int(planes.shape[1])
    in_bufs = to_device(planes, mode)
    par_bufs = to_device(params, mode)
    coef_bufs, ncoef = _coef_buffers(coefficients, mode)
    con_bufs = to_device(consts_array(mode) if consts is None else consts, mode)
    out_bufs = empty_device(n, mode)
    status = torch.zeros(n, dtype=torch.uint8, device="mps")
    iters = torch.full((n,), ITERS_PROBE_UNWRITTEN, dtype=torch.uint8, device="mps")
    ip = int_params_geom(
        n, flags=flags, geom=geom, apcode=apcode, ncoef=ncoef, max_iter=max_iter
    )
    bufs = [*in_bufs, *par_bufs, *coef_bufs, *con_bufs, ip, *out_bufs, status, iters]
    _run(lib, f"probe_newton_{mode}" if newton else f"probe_distance_{mode}", bufs, n)
    return (
        raw_components(out_bufs, (n,)),
        status.cpu().numpy(),
        iters.cpu().numpy(),
    )
