"""NumPy-only fast path of ``SurfaceGroup.trace``. The generic trace
(``Surface.trace``) is the reference: it runs on every backend, and this
module must give its results bit for bit.

This module holds the compiled kernel of the trace plan. The builder,
``optiland.surfaces._trace_plan``, writes one row per surface into two
tables and calls ``trace_plan``. The kernel traces the rays in blocks of
``_BLOCK`` rays. For each block, each surface runs short loops over the rays
of the block, one function per step of the generic trace. These loops
vectorize, and the buffers of a block stay in the L1 or L2 cache while the
block goes through all surfaces. Numba inlines the step functions into
``trace_plan`` (``inline="always"``), so the kernel compiles to one loop
nest. Functions that call ``_roots`` are the exception (see
``_conic_distance``).

Each function repeats the float64 expressions of the generic NumPy trace in
the same order. No function uses fastmath, so LLVM keeps the order of the
operations and does not fuse a multiplication and an addition. With
``error_model="numpy"``, a division by zero gives inf or NaN, as in NumPy.
A NaN is at the same position as in the generic trace, but compiled code can
give it another sign bit.

Table layout
------------
``ints`` (int64) and ``floats`` (float64) have one row per surface. The
builder starts each row from the defaults, so a column that the steps of a
row do not read holds its default. In the column "Valid when", "traced"
means that KIND is not OBJECT, "absorbs" that the medium before the surface
absorbs (ABSORB_ROW is a row of ``exponents``), "refracts" that REFLECT is
0, and "aperture" that APERTURE is not NO_APERTURE.

======================  ============================  ====  ===========  ===========
Column of ``ints``      Meaning                       Unit  Valid when   Default
======================  ============================  ====  ===========  ===========
KIND                    OBJECT, PLANE, CONIC or FLAT  -     always       OBJECT
ABSORB_ROW              row of ``exponents``          -     absorbs      NO_ROW
ROTATE                  ROTATE_X, _Y and _Z flags     -     traced       0
APERTURE                RADIAL or OFFSET_RADIAL       -     traced       NO_APERTURE
REFLECT                 1 for a mirror                -     traced       0
CLIP_ROW                row of ``clipped``            -     aperture     NO_ROW
======================  ============================  ====  ===========  ===========

======================  ============================  ====  ===========  ===========
Column of ``floats``    Meaning                       Unit  Valid when   Default
======================  ============================  ====  ===========  ===========
TX_IN, TY_IN, TZ_IN     -x, -y, -z (localize)         mm    traced       0.0
TX_OUT, TY_OUT, TZ_OUT  x, y, z (globalize)           mm    traced       0.0
RADIUS                  radius of curvature R         mm    CONIC, FLAT  0.0
CONIC_K                 conic constant k              -     CONIC        0.0
ONE_PLUS_K              1 + k                         -     CONIC, FLAT  0.0
RADIUS_SQUARED          R * R                         mm2   CONIC, FLAT  0.0
INDEX_RATIO             u = n1 / n2                   -     refracts     0.0
INDEX_RATIO_SQUARED     u**2 (see the builder)        -     refracts     0.0
OPD_INDEX               n before the surface          -     traced       0.0
ALPHA                   4 * pi * k / wavelength       1/um  absorbs      0.0
COS_X_IN, SIN_X_IN      cos, sin of -rx (localize)    -     ROTATE_X     0.0
COS_Y_IN, SIN_Y_IN      cos, sin of -ry               -     ROTATE_Y     0.0
COS_Z_IN, SIN_Z_IN      cos, sin of -rz               -     ROTATE_Z     0.0
COS_X_OUT, SIN_X_OUT    cos, sin of rx (globalize)    -     ROTATE_X     0.0
COS_Y_OUT, SIN_Y_OUT    cos, sin of ry                -     ROTATE_Y     0.0
COS_Z_OUT, SIN_Z_OUT    cos, sin of rz                -     ROTATE_Z     0.0
R_MAX_SQUARED           r_max**2 of the aperture      mm2   aperture     0.0
R_MIN_SQUARED           r_min**2 of the aperture      mm2   aperture     0.0
OFFSET_X, OFFSET_Y      offset of the aperture        mm    aperture     0.0
======================  ============================  ====  ===========  ===========

x, y, z, rx, ry and rz are the values of the ``CoordinateSystem`` of the
surface. n1 and n2 are the refractive indices before and after the surface.
``clipped`` has one row per aperture: the mask of the rays that the aperture
clips, for the intensity pass of a system that absorbs.

Manuel Fragata Mendes, 2026
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from numba import njit, types
from numba.extending import intrinsic

from optiland.backend._conic import _conic_candidates
from optiland.backend.numpy_backend.conic import _float64_eps, _scalar_where

if TYPE_CHECKING:
    from collections.abc import Callable

    from numba.core.typing.templates import Signature

# Rays per block. A block has 15 buffers of 256 float64 values (2 KiB each,
# 30 KiB in all), so it stays in the L1 or L2 cache.
_BLOCK = 256

# The default of each *_ROW column: the surface has no row in that table.
NO_ROW = -1

# Step kinds (column KIND).
OBJECT = 0  # ObjectSurface: record only
PLANE = 1  # Plane
CONIC = 2  # StandardGeometry with a finite radius
FLAT = 3  # StandardGeometry with an infinite radius

# Flags of the rotations (column ROTATE).
ROTATE_X, ROTATE_Y, ROTATE_Z = 1, 2, 4

# Aperture kinds (column APERTURE).
NO_APERTURE = 0
RADIAL = 1  # RadialAperture
OFFSET_RADIAL = 2  # OffsetRadialAperture

# Columns of ``ints``.
KIND = 0
ABSORB_ROW = 1
ROTATE = 2
APERTURE = 3
REFLECT = 4
CLIP_ROW = 5
INT_COLUMNS = 6

# Columns of ``floats``.
TX_IN, TY_IN, TZ_IN = 0, 1, 2
TX_OUT, TY_OUT, TZ_OUT = 3, 4, 5
RADIUS, CONIC_K, ONE_PLUS_K, RADIUS_SQUARED = 6, 7, 8, 9
INDEX_RATIO, INDEX_RATIO_SQUARED, OPD_INDEX, ALPHA = 10, 11, 12, 13
# The IN columns hold cos and sin of the negated angle, as localize computes
# them. Do not derive them from the OUT columns.
COS_X_IN, SIN_X_IN, COS_Y_IN, SIN_Y_IN, COS_Z_IN, SIN_Z_IN = range(14, 20)
COS_X_OUT, SIN_X_OUT, COS_Y_OUT, SIN_Y_OUT, COS_Z_OUT, SIN_Z_OUT = range(20, 26)
R_MAX_SQUARED, R_MIN_SQUARED, OFFSET_X, OFFSET_Y = 26, 27, 28, 29
FLOAT_COLUMNS = 30

_N_MIN = 1e-14  # N_safe of _conic_intersection_distance


@intrinsic
def _llvm_sqrt(
    typingctx: object, value: types.Type
) -> tuple[Signature, Callable] | None:
    """Square root as the LLVM intrinsic ``llvm.sqrt.f64``."""
    if value != types.float64:
        return None

    def codegen(context, builder, signature, args):
        function = builder.module.declare_intrinsic("llvm.sqrt", [args[0].type])
        return builder.call(function, args)

    return types.float64(types.float64), codegen


@njit(inline="always", error_model="numpy", cache=True)
def _sqrt(value: float) -> float:
    """Correctly rounded square root (IEEE 754), as ``numpy.sqrt``.

    Numba lowers ``math.sqrt`` to a call of the C library, which can set
    errno, so a loop that contains it does not vectorize. The intrinsic gives
    the same value, and NaN for a negative value.
    """
    return _llvm_sqrt(value)


@njit(inline="always", error_model="numpy", cache=True)
def _sign(value: float) -> float:
    """``numpy.sign`` of a float64: 1, -1, +0 for a zero, and NaN for NaN."""
    if value > 0:
        return 1.0
    if value < 0:
        return -1.0
    if value == 0:
        return 0.0
    return value


@njit(inline="always", error_model="numpy", cache=True)
def _aligned_normal(
    L0: float, M0: float, N0: float, nx: float, ny: float, nz: float
) -> tuple[float, float, float, float]:
    """``RealRays._align_surface_normal`` for one ray.

    Returns:
        The normal turned to the side of the incident direction, and the
        cosine of the angle of incidence.
    """
    dot = L0 * nx + M0 * ny + N0 * nz
    sign = _sign(dot)
    return nx * sign, ny * sign, nz * sign, abs(dot)


@njit(inline="always", error_model="numpy", cache=True)
def _load_block(rays_in: tuple, state: tuple, start: int, m: int) -> None:
    """Copy rays ``start`` to ``start + m`` into the buffers of a block.

    The buffers of L0, M0 and N0 start at zero.
    """
    x, y, z, L, M, N, i, opd = rays_in
    bx, by, bz, bL, bM, bN, bi, bopd, bL0, bM0, bN0 = state
    stop = start + m
    bx[:m] = x[start:stop]
    by[:m] = y[start:stop]
    bz[:m] = z[start:stop]
    bL[:m] = L[start:stop]
    bM[:m] = M[start:stop]
    bN[:m] = N[start:stop]
    bi[:m] = i[start:stop]
    bopd[:m] = opd[start:stop]
    bL0[:m] = 0.0
    bM0[:m] = 0.0
    bN0[:m] = 0.0


@njit(inline="always", error_model="numpy", cache=True)
def _translate(position: tuple, dx: float, dy: float, dz: float, m: int) -> None:
    """``RealRays.translate``: add (dx, dy, dz) to the position of each ray."""
    bx, by, bz = position
    for j in range(m):
        bx[j] = bx[j] + dx
        by[j] = by[j] + dy
        bz[j] = bz[j] + dz


@njit(inline="always", error_model="numpy", cache=True)
def _rotate_x(position: tuple, direction: tuple, c: float, s: float, m: int) -> None:
    """``RealRays.rotate_x`` with c = cos(rx) and s = sin(rx)."""
    _, by, bz = position
    _, bM, bN = direction
    for j in range(m):
        yj, zj, Mj, Nj = by[j], bz[j], bM[j], bN[j]
        by[j] = yj * c - zj * s
        bz[j] = yj * s + zj * c
        bM[j] = Mj * c - Nj * s
        bN[j] = Mj * s + Nj * c


@njit(inline="always", error_model="numpy", cache=True)
def _rotate_y(position: tuple, direction: tuple, c: float, s: float, m: int) -> None:
    """``RealRays.rotate_y`` with c = cos(ry) and s = sin(ry)."""
    bx, _, bz = position
    bL, _, bN = direction
    for j in range(m):
        xj, zj, Lj, Nj = bx[j], bz[j], bL[j], bN[j]
        bx[j] = xj * c + zj * s
        bz[j] = -xj * s + zj * c
        bL[j] = Lj * c + Nj * s
        bN[j] = -Lj * s + Nj * c


@njit(inline="always", error_model="numpy", cache=True)
def _rotate_z(position: tuple, direction: tuple, c: float, s: float, m: int) -> None:
    """``RealRays.rotate_z`` with c = cos(rz) and s = sin(rz)."""
    bx, by, _ = position
    bL, bM, _ = direction
    for j in range(m):
        xj, yj, Lj, Mj = bx[j], by[j], bL[j], bM[j]
        bx[j] = xj * c - yj * s
        by[j] = xj * s + yj * c
        bL[j] = Lj * c - Mj * s
        bM[j] = Lj * s + Mj * c


@njit(inline="always", error_model="numpy", cache=True)
def _localize(
    position: tuple, direction: tuple, row: np.ndarray, rotate: int, m: int
) -> None:
    """``CoordinateSystem.localize``: translate, then rotate about z, y, x."""
    _translate(position, row[TX_IN], row[TY_IN], row[TZ_IN], m)
    if rotate & ROTATE_Z:
        _rotate_z(position, direction, row[COS_Z_IN], row[SIN_Z_IN], m)
    if rotate & ROTATE_Y:
        _rotate_y(position, direction, row[COS_Y_IN], row[SIN_Y_IN], m)
    if rotate & ROTATE_X:
        _rotate_x(position, direction, row[COS_X_IN], row[SIN_X_IN], m)


@njit(inline="always", error_model="numpy", cache=True)
def _globalize(
    position: tuple, direction: tuple, row: np.ndarray, rotate: int, m: int
) -> None:
    """``CoordinateSystem.globalize``: rotate about x, y, z, then translate."""
    if rotate & ROTATE_X:
        _rotate_x(position, direction, row[COS_X_OUT], row[SIN_X_OUT], m)
    if rotate & ROTATE_Y:
        _rotate_y(position, direction, row[COS_Y_OUT], row[SIN_Y_OUT], m)
    if rotate & ROTATE_Z:
        _rotate_z(position, direction, row[COS_Z_OUT], row[SIN_Z_OUT], m)
    _translate(position, row[TX_OUT], row[TY_OUT], row[TZ_OUT], m)


@njit(inline="always", error_model="numpy", cache=True)
def _plane_distance(
    position: tuple, direction: tuple, distance: np.ndarray, m: int
) -> None:
    """``Plane.distance``: t = -z / N."""
    bz = position[2]
    bN = direction[2]
    for j in range(m):
        distance[j] = -bz[j] / bN[j]


@njit(inline="always", error_model="numpy", cache=True)
def _flat_distance(
    position: tuple, direction: tuple, distance: np.ndarray, m: int
) -> None:
    """``_conic_intersection_distance`` for an infinite radius: -z / N_safe."""
    bz = position[2]
    bN = direction[2]
    for j in range(m):
        Nj = bN[j]
        distance[j] = -bz[j] / (Nj if abs(Nj) > _N_MIN else _N_MIN)


@njit(inline="always", error_model="numpy", cache=True)
def _contains(aperture: int, row: np.ndarray, px: float, py: float) -> bool:
    """``RadialAperture.contains`` or ``OffsetRadialAperture.contains``."""
    if aperture == RADIAL:
        radius2 = px * px + py * py
    else:
        dx = px - row[OFFSET_X]
        dy = py - row[OFFSET_Y]
        radius2 = dx * dx + dy * dy
    return (radius2 <= row[R_MAX_SQUARED]) & (radius2 >= row[R_MIN_SQUARED])


@njit(inline="always", error_model="numpy", cache=True)
def _roots(
    x: float, y: float, z: float, L: float, M: float, N: float, row: np.ndarray
) -> tuple:
    """The roots of the conic of ``row`` (RADIUS, CONIC_K) for one ray.

    The conic arithmetic of the generic trace (``optiland.backend._conic``),
    with the scalar functions of its compiled NumPy loop, so there is no
    second solver. ``_sqrt`` gives the bits of ``math.sqrt`` of that loop:
    both are correctly rounded.
    """
    return _conic_candidates(
        x,
        y,
        z,
        L,
        M,
        N,
        row[RADIUS],
        row[CONIC_K],
        _scalar_where,
        _sqrt,
        math.copysign,
        _float64_eps,
    )


@njit(error_model="numpy", cache=True)
def _conic_distance(
    position: tuple, direction: tuple, distance: np.ndarray, row: np.ndarray, m: int
) -> None:
    """``be.conic_intersection`` without an aperture.

    A function that calls ``_roots`` is not inlined into ``trace_plan``.
    ``_conic_candidates`` and the scalar functions of ``conic.py`` are
    inline overloads. After Numba inlines an overload, it removes the
    statements that it finds dead, and in ``trace_plan`` (Numba 0.65) it
    removed stores into the block buffers. Here, the stores go to
    ``distance``, an argument.
    """
    bx, by, bz = position
    bL, bM, bN = direction
    for j in range(m):
        roots = _roots(bx[j], by[j], bz[j], bL[j], bM[j], bN[j], row)
        distance[j] = (
            (roots.second if roots.pick_second else roots.first)
            if roots.solvable
            else math.nan
        )


@njit(error_model="numpy", cache=True)
def _conic_distance_in_aperture(
    position: tuple,
    direction: tuple,
    distance: np.ndarray,
    row: np.ndarray,
    aperture: int,
    m: int,
) -> None:
    """``be.conic_intersection`` that prefers a root inside the aperture."""
    bx, by, bz = position
    bL, bM, bN = direction
    for j in range(m):
        xj, yj, Lj, Mj = bx[j], by[j], bL[j], bM[j]
        roots = _roots(xj, yj, bz[j], Lj, Mj, bN[j], row)
        # Repeats _select_distance for one ray. Change both together.
        t1, t2 = roots.first, roots.second
        in1 = _contains(aperture, row, xj + t1 * Lj, yj + t1 * Mj)
        in2 = _contains(aperture, row, xj + t2 * Lj, yj + t2 * Mj)
        prefer1 = roots.first_valid & in1
        prefer2 = roots.second_valid & in2
        pick2 = prefer2 | (False if prefer1 else roots.pick_second)
        distance[j] = (t2 if pick2 else t1) if roots.solvable else math.nan


@njit(inline="always", error_model="numpy", cache=True)
def _propagate(position: tuple, direction: tuple, distance: np.ndarray, m: int) -> None:
    """``HomogeneousPropagation.propagate``: move each ray by t along its path."""
    bx, by, bz = position
    bL, bM, bN = direction
    for j in range(m):
        t = distance[j]
        bx[j] = bx[j] + t * bL[j]
        by[j] = by[j] + t * bM[j]
        bz[j] = bz[j] + t * bN[j]


@njit(inline="always", error_model="numpy", cache=True)
def _absorption_exponents(
    exponents: np.ndarray, distance: np.ndarray, row: np.ndarray, m: int
) -> None:
    """The exponent of ``HomogeneousPropagation.propagate`` (ALPHA).

    alpha = 4*pi*k/lambda is in 1/um (lambda in um). t is in mm, so
    alpha * t * 1e3 has no unit. exp(-alpha*d) is applied after the kernel.
    """
    alpha = row[ALPHA]
    for j in range(m):
        exponents[j] = -alpha * distance[j] * 1e3


@njit(inline="always", error_model="numpy", cache=True)
def _add_optical_path(
    opd: np.ndarray, distance: np.ndarray, row: np.ndarray, m: int
) -> None:
    """``BasePropagationModel.propagate_to_surface``: opd + t * n (OPD_INDEX)."""
    index = row[OPD_INDEX]
    for j in range(m):
        opd[j] = opd[j] + distance[j] * index


@njit(inline="always", error_model="numpy", cache=True)
def _clip(
    position: tuple,
    intensity: np.ndarray,
    clipped: np.ndarray,
    row: np.ndarray,
    aperture: int,
    m: int,
) -> None:
    """``BaseAperture.clip``: zero intensity for the rays outside the aperture.

    ``clipped`` (a row of the clip table) receives the mask of these rays.
    """
    bx, by, _ = position
    for j in range(m):
        outside = not _contains(aperture, row, bx[j], by[j])
        clipped[j] = outside
        intensity[j] = 0.0 if outside else intensity[j]


@njit(inline="always", error_model="numpy", cache=True)
def _plane_normal(normal: tuple, m: int) -> None:
    """``Plane.surface_normal``: (0, 0, 1) at each point."""
    bnx, bny, bnz = normal
    for j in range(m):
        bnx[j] = 0.0
        bny[j] = 0.0
        bnz[j] = 1.0


@njit(inline="always", error_model="numpy", cache=True)
def _conic_normal(position: tuple, normal: tuple, row: np.ndarray, m: int) -> None:
    """``StandardGeometry.surface_normal``, also for an infinite radius.

    Reads RADIUS, ONE_PLUS_K and RADIUS_SQUARED.
    """
    bx, by, _ = position
    bnx, bny, bnz = normal
    radius, one_plus_k = row[RADIUS], row[ONE_PLUS_K]
    radius_squared = row[RADIUS_SQUARED]
    for j in range(m):
        xj, yj = bx[j], by[j]
        r2 = xj * xj + yj * yj
        denom = radius * _sqrt(1.0 - one_plus_k * r2 / radius_squared)
        dfdx = xj / denom
        dfdy = yj / denom
        mag = _sqrt(dfdx * dfdx + dfdy * dfdy + 1.0)
        bnx[j] = dfdx / mag
        bny[j] = dfdy / mag
        bnz[j] = -1.0 / mag


@njit(inline="always", error_model="numpy", cache=True)
def _refract(
    direction: tuple, incident: tuple, normal: tuple, row: np.ndarray, m: int
) -> None:
    """``RealRays.refract`` with one index ratio u = n1 / n2 for all rays.

    Reads INDEX_RATIO and INDEX_RATIO_SQUARED. The incident directions go to
    L0, M0 and N0, as in ``RealRays.refract``.
    """
    bL, bM, bN = direction
    bL0, bM0, bN0 = incident
    bnx, bny, bnz = normal
    u, u2 = row[INDEX_RATIO], row[INDEX_RATIO_SQUARED]
    for j in range(m):
        L0, M0, N0 = bL[j], bM[j], bN[j]
        bL0[j], bM0[j], bN0[j] = L0, M0, N0
        nx, ny, nz, dot = _aligned_normal(L0, M0, N0, bnx[j], bny[j], bnz[j])
        # Snell's law in vector form. Total internal reflection gives a
        # negative radicand and NaN directions, as in RealRays.refract.
        root = _sqrt(1.0 - u2 * (1.0 - dot * dot))
        bL[j] = u * L0 + nx * root - u * nx * dot
        bM[j] = u * M0 + ny * root - u * ny * dot
        bN[j] = u * N0 + nz * root - u * nz * dot


@njit(inline="always", error_model="numpy", cache=True)
def _reflect(direction: tuple, incident: tuple, normal: tuple, m: int) -> None:
    """``RealRays.reflect``.

    The incident directions go to L0, M0 and N0, as in ``RealRays.reflect``.
    """
    bL, bM, bN = direction
    bL0, bM0, bN0 = incident
    bnx, bny, bnz = normal
    for j in range(m):
        L0, M0, N0 = bL[j], bM[j], bN[j]
        bL0[j], bM0[j], bN0[j] = L0, M0, N0
        nx, ny, nz, dot = _aligned_normal(L0, M0, N0, bnx[j], bny[j], bnz[j])
        bL[j] = L0 - 2.0 * dot * nx
        bM[j] = M0 - 2.0 * dot * ny
        bN[j] = N0 - 2.0 * dot * nz


@njit(inline="always", error_model="numpy", cache=True)
def _store_records(state: tuple, records: tuple, s: int, start: int, m: int) -> None:
    """``Surface._record_real``: write row ``s`` of each record for a block."""
    stop = start + m
    for k in range(len(records)):
        records[k][s, start:stop] = state[k][:m]


@njit(inline="always", error_model="numpy", cache=True)
def _store_block(state: tuple, rays_out: tuple, start: int, m: int) -> None:
    """Copy the buffers of a block to rays ``start`` to ``start + m``."""
    stop = start + m
    for k in range(len(rays_out)):
        rays_out[k][start:stop] = state[k][:m]


@njit(error_model="numpy", cache=True)
def trace_plan(
    rays_in: tuple,
    ints: np.ndarray,
    floats: np.ndarray,
    exponents: np.ndarray,
    clipped: np.ndarray,
    rays_out: tuple,
    records: tuple,
    record: bool,
) -> None:
    """Trace the rays through all surfaces of a plan.

    The input arrays are not changed. The intensity in ``rays_out`` and in
    ``records`` includes the clips of the apertures, but not the absorption.

    Args:
        rays_in: x, y, z, L, M, N, i and opd of the rays: 1-D float64
            arrays of one length.
        ints: The integer table (see the module docstring).
        floats: The float table.
        exponents: One row per absorbing medium. Row ABSORB_ROW of a surface
            receives the exponent -alpha * t * 1e3 of each ray in the medium
            before that surface.
        clipped: One row per aperture. Row CLIP_ROW of a surface receives
            True for each ray that its aperture clips.
        rays_out: Receive x, y, z, L, M, N, i and opd after the last surface,
            then L0, M0 and N0: the direction of each ray before the last
            refraction or reflection, in the local frame of that surface.
        records: x, y, z, L, M, N, intensity and opd, each with one row per
            surface. Row s receives the ray state after surface s.
        record: Whether to write ``records``.
    """
    n_rays = rays_in[0].shape[0]
    n_surfaces = ints.shape[0]
    bx, by, bz = np.empty(_BLOCK), np.empty(_BLOCK), np.empty(_BLOCK)
    bL, bM, bN = np.empty(_BLOCK), np.empty(_BLOCK), np.empty(_BLOCK)
    bL0, bM0, bN0 = np.empty(_BLOCK), np.empty(_BLOCK), np.empty(_BLOCK)
    bi, bopd, bdist = np.empty(_BLOCK), np.empty(_BLOCK), np.empty(_BLOCK)
    normal = (np.empty(_BLOCK), np.empty(_BLOCK), np.empty(_BLOCK))
    position, direction, incident = (bx, by, bz), (bL, bM, bN), (bL0, bM0, bN0)
    # The buffers in the order of rays_out. The records are the first eight.
    state = (bx, by, bz, bL, bM, bN, bi, bopd, bL0, bM0, bN0)
    for start in range(0, n_rays, _BLOCK):
        m = min(_BLOCK, n_rays - start)
        _load_block(rays_in, state, start, m)
        for s in range(n_surfaces):
            kind = ints[s, KIND]
            if kind != OBJECT:
                row = floats[s]
                rotate, aperture = ints[s, ROTATE], ints[s, APERTURE]
                _localize(position, direction, row, rotate, m)
                # The distance along each ray to the surface.
                if kind == PLANE:
                    _plane_distance(position, direction, bdist, m)
                elif kind == FLAT:
                    _flat_distance(position, direction, bdist, m)
                elif aperture == NO_APERTURE:
                    _conic_distance(position, direction, bdist, row, m)
                else:
                    _conic_distance_in_aperture(
                        position, direction, bdist, row, aperture, m
                    )
                # HomogeneousPropagation: move, absorb, add the optical path.
                _propagate(position, direction, bdist, m)
                absorb_row = ints[s, ABSORB_ROW]
                if absorb_row != NO_ROW:
                    block_exponents = exponents[absorb_row, start : start + m]
                    _absorption_exponents(block_exponents, bdist, row, m)
                _add_optical_path(bopd, bdist, row, m)
                # BaseAperture.clip.
                if aperture != NO_APERTURE:
                    block_clipped = clipped[ints[s, CLIP_ROW], start : start + m]
                    _clip(position, bi, block_clipped, row, aperture, m)
                # The surface normal at each hit point.
                if kind == PLANE:
                    _plane_normal(normal, m)
                else:
                    _conic_normal(position, normal, row, m)
                # RefractiveReflectiveModel.interact_real_rays.
                if ints[s, REFLECT]:
                    _reflect(direction, incident, normal, m)
                else:
                    _refract(direction, incident, normal, row, m)
                _globalize(position, direction, row, rotate, m)
            if record:
                _store_records(state, records, s, start, m)
        _store_block(state, rays_out, start, m)
