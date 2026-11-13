"""NumPy-only fast path of ``SurfaceGroup.trace``. The generic trace
(``Surface.trace``) is the reference: it runs on every backend, and this
module must give its results bit for bit.

This module holds the compiled kernels of the trace plan. The builder,
``optiland.surfaces._trace_plan``, writes one row per surface into two
tables and calls a kernel. A kernel traces the rays in blocks of ``_BLOCK``
rays. For each block, each surface runs short loops over the rays of the
block, one function per step of the generic trace. These loops vectorize,
and the buffers of a block stay in the L1 or L2 cache while the block goes
through all surfaces. Numba inlines the step functions into the loop nest
``_trace_blocks`` and the loop nest into each kernel (``inline="always"``),
so each kernel compiles to one loop nest. Functions that call ``_roots`` are
the exception (see ``_conic_distance``).

Two kernels compile from the loop nest. ``trace_plan`` traces all plans.
``trace_basic_plan`` traces a basic plan, a plan without per-ray values and
without even aspheres (see below), for example of a trace with one
wavelength through spheres and conics. It has no code for per-ray values
and for the Newton-Raphson iteration of an even asphere. With that code in
the loop nest, a trace with one wavelength took about 7 percent more time
in the kernel, although it ran none of that code. On a basic plan, both
kernels run the same steps, so they give the same bits. The tests compare
them.

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
0, "aperture" that APERTURE is not NO_APERTURE, and "asphere" that NEWTON
is 1.

======================  ============================  ====  ===========  ===========
Column of ``ints``      Meaning                       Unit  Valid when   Default
======================  ============================  ====  ===========  ===========
KIND                    OBJECT, PLANE, CONIC or FLAT  -     always       OBJECT
ABSORB_ROW              row of ``exponents``          -     absorbs      NO_ROW
ROTATE                  ROTATE_X, _Y and _Z flags     -     traced       0
APERTURE                RADIAL or OFFSET_RADIAL       -     traced       NO_APERTURE
REFLECT                 1 for a mirror                -     traced       0
CLIP_ROW                row of ``clipped``            -     aperture     NO_ROW
N1_ROW                  row of ``ray_values``: n1     -     refracts     NO_ROW
N2_ROW                  row of ``ray_values``: n2     -     refracts     NO_ROW
OPD_INDEX_ROW           row of ``ray_values``: n      -     traced       NO_ROW
ALPHA_ROW               row of ``ray_values``: alpha  -     absorbs      NO_ROW
NEWTON                  1 for an EvenAsphere          -     traced       0
NCOEF                   used entries in row s of      -     asphere      0
                        ``coef`` and ``dcoef``
MAX_ITER                ``max_iter`` of the asphere   -     asphere      0
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
N1, N2                  n1, n2 next to a per-ray row  -     refracts     0.0
TOL                     ``tol`` of the asphere        mm    asphere      0.0
======================  ============================  ====  ===========  ===========

x, y, z, rx, ry and rz are the values of the ``CoordinateSystem`` of the
surface. n1 and n2 are the refractive indices before and after the surface.
``clipped`` has one row per aperture: the mask of the rays that the aperture
clips, for the intensity pass of a system that absorbs.

An EvenAsphere has the KIND of its base conic, CONIC or FLAT, with its
columns, and NEWTON = 1. The kernel starts at the distance to the base
conic and refines it with the Newton-Raphson iteration of
``NewtonRaphsonGeometry._solve_distance_primal`` (see ``_newton``).
``coef`` and ``dcoef`` have one row per surface: the coefficients C_i of
the asphere and the products i * C_i, the coefficients of d(sag)/d(r**2),
for Horner's method. The rows of other surfaces are zero. ``seed_stats``
has one row per surface, with the columns SEED_NAN and SEED_MAX (see
``_add_seeds``).

``ray_values`` has one row per material value that differs from ray to ray,
for example the refractive index of a glass in a bundle of several
wavelengths. A ``*_ROW`` column that names a row of ``ray_values`` replaces
a float column: OPD_INDEX_ROW replaces OPD_INDEX, ALPHA_ROW replaces ALPHA,
and N1_ROW or N2_ROW replace INDEX_RATIO and INDEX_RATIO_SQUARED. The
kernel then computes u = n1 / n2 and u * u for each ray, with N1 or N2 for
an index that is the same for all rays.

Manuel Fragata Mendes, 2026
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from numba import njit, types
from numba.extending import intrinsic

from optiland.backend._conic import _conic_candidates
from optiland.backend.numpy_backend.conic import (
    _FLOAT64_EPS,
    _float64_eps,
    _scalar_where,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numba.core.typing.templates import Signature

# Rays per block. A block has 15 buffers of 256 float64 values (2 KiB each,
# 30 KiB in all), so it stays in the L1 or L2 cache. Per-ray values add two
# (the index ratios of trace_plan), and the Newton-Raphson iteration of an
# even asphere adds 9 (``work``) and 3 boolean buffers (``flags``).
_BLOCK = 256

# The default of each *_ROW column: the surface has no row in that table.
NO_ROW = -1

# Step kinds (column KIND). CONIC and FLAT also hold the base conic of an
# EvenAsphere, with NEWTON set.
OBJECT = 0  # ObjectSurface: record only
PLANE = 1  # Plane
CONIC = 2  # StandardGeometry or EvenAsphere with a finite radius
FLAT = 3  # StandardGeometry or EvenAsphere with an infinite radius

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
N1_ROW = 6
N2_ROW = 7
OPD_INDEX_ROW = 8
ALPHA_ROW = 9
NEWTON = 10
NCOEF = 11  # used entries in row s of coef and dcoef
MAX_ITER = 12
INT_COLUMNS = 13

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
N1, N2 = 30, 31
TOL = 32
FLOAT_COLUMNS = 33

# Columns of ``seed_stats``.
SEED_NAN = 0
SEED_MAX = 1
SEED_COLUMNS = 2

# Rows of ``work``, the buffers of the Newton-Raphson iteration of a block.
_RESIDUAL, _T_NEXT, _SAG, _PX, _PY, _DFDX, _DFDY, _R2, _HORNER = range(9)
_WORK_ROWS = 9
# Rows of ``flags``, the state of each ray in the iteration.
_NO_SEED, _CONVERGED, _IS_ACTIVE = range(3)
_FLAG_ROWS = 3

# N_safe of _conic_intersection_distance for an infinite radius, and of the
# plane seed of NewtonRaphsonGeometry._solve_distance_primal.
_N_MIN = 1e-14

# _DENOM_EPS_MULTIPLIER of optiland.geometries.newton_raphson. Numba compiles
# a global as a constant, and its cache does not see a change in another
# module, so the value is repeated here. A test checks that both are equal.
_DENOM_EPS_MULTIPLIER = 32.0
# The threshold of _nz_threshold, and of _denominator_threshold before the
# scale, for float64.
_DENOM_TAU = _DENOM_EPS_MULTIPLIER * _FLOAT64_EPS


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
def _vertex_plane_distance(z: float, N: float) -> float:
    """The distance -z / N_safe of one ray to the plane z = 0.

    ``_conic_intersection_distance`` computes it for an infinite radius, and
    ``_solve_distance_primal`` for its plane seed. N_safe is N, or _N_MIN if
    abs(N) is not above _N_MIN.
    """
    return -z / (N if abs(N) > _N_MIN else _N_MIN)


@njit(inline="always", error_model="numpy", cache=True)
def _flat_distance(
    position: tuple, direction: tuple, distance: np.ndarray, m: int
) -> None:
    """``_conic_intersection_distance`` for an infinite radius: -z / N_safe."""
    bz = position[2]
    bN = direction[2]
    for j in range(m):
        distance[j] = _vertex_plane_distance(bz[j], bN[j])


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
def _absorption_exponents_per_ray(
    exponents: np.ndarray, distance: np.ndarray, alphas: np.ndarray, m: int
) -> None:
    """``_absorption_exponents`` with the alpha of each ray (ALPHA_ROW)."""
    for j in range(m):
        exponents[j] = -alphas[j] * distance[j] * 1e3


@njit(inline="always", error_model="numpy", cache=True)
def _add_optical_path(
    opd: np.ndarray, distance: np.ndarray, row: np.ndarray, m: int
) -> None:
    """``BasePropagationModel.propagate_to_surface``: opd + t * n (OPD_INDEX)."""
    index = row[OPD_INDEX]
    for j in range(m):
        opd[j] = opd[j] + distance[j] * index


@njit(inline="always", error_model="numpy", cache=True)
def _add_optical_path_per_ray(
    opd: np.ndarray, distance: np.ndarray, indices: np.ndarray, m: int
) -> None:
    """``_add_optical_path`` with the index n of each ray (OPD_INDEX_ROW)."""
    for j in range(m):
        opd[j] = opd[j] + distance[j] * indices[j]


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
def _unit_normal(dfdx: float, dfdy: float) -> tuple[float, float, float]:
    """The surface normal (dfdx, dfdy, -1) / mag, with the slopes of the sag.

    The expression of ``StandardGeometry._normal_components`` and
    ``EvenAsphere._surface_normal``: mag = sqrt(dfdx**2 + dfdy**2 + 1).
    """
    mag = _sqrt(dfdx * dfdx + dfdy * dfdy + 1.0)
    return dfdx / mag, dfdy / mag, -1.0 / mag


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
        bnx[j], bny[j], bnz[j] = _unit_normal(xj / denom, yj / denom)


# Even aspheres. Each function repeats an expression of EvenAsphere or of
# NewtonRaphsonGeometry, for one value or for the rays of a block. A loop
# over the coefficients or over the steps is outside the loops over the
# rays, so each loop over the rays is an innermost loop, which LLVM can
# vectorize.


@njit(inline="always", error_model="numpy", cache=True)
def _maximum(value: float, other: float) -> float:
    """``numpy.maximum`` of two float64 values: NaN if ``value`` is NaN."""
    return value if (value >= other or value != value) else other


@njit(inline="always", error_model="numpy", cache=True)
def _sign_preserving_floor(value: float, eps: float) -> float:
    """``_sign_preserving_floor`` of ``newton_raphson`` for one value.

    A value with abs(value) <= eps becomes eps or -eps, by its sign. NaN
    becomes -eps, because NaN >= 0 is False.
    """
    return value if abs(value) > eps else (eps if value >= 0 else -eps)


@njit(inline="always", error_model="numpy", cache=True)
def _regularize_signed(value: float, tau: float) -> float:
    """The floored value of ``_regularize_signed`` of ``newton_raphson``.

    A value with abs(value) <= tau becomes tau or -tau, by its sign. NaN
    stays NaN.
    """
    return (tau if value >= 0 else -tau) if abs(value) <= tau else value


@njit(inline="always", error_model="numpy", cache=True)
def _horner(
    r2: np.ndarray, out: np.ndarray, coefficients: np.ndarray, n_coef: int, m: int
) -> None:
    """Horner's method, as ``EvenAsphere`` uses it, for each ray of a block.

    Args:
        r2: r**2 = x * x + y * y of each ray.
        out: Receives c[0] + r2 * (c[1] + r2 * (... + r2 * c[n_coef - 1]))
            for each ray, with c = ``coefficients``.
        coefficients: A row of ``coef`` or of ``dcoef``.
        n_coef: The number of coefficients. At least 1.
        m: The number of rays in the block.
    """
    c = coefficients[n_coef - 1]
    for j in range(m):
        out[j] = c
    for i in range(n_coef - 2, -1, -1):
        c = coefficients[i]
        for j in range(m):
            out[j] = c + r2[j] * out[j]


@njit(inline="always", error_model="numpy", cache=True)
def _asphere_sag(
    point: tuple,
    sag: np.ndarray,
    scratch: tuple,
    row: np.ndarray,
    coef: np.ndarray,
    n_coef: int,
    m: int,
) -> None:
    """``EvenAsphere.sag`` at a point of each ray of a block.

    Args:
        point: The buffers of x and y of the points.
        sag: Receives the sag at each point.
        scratch: Two buffers that receive r**2 and the sum of Horner's method.
        row: The float row of the surface: RADIUS, ONE_PLUS_K and
            RADIUS_SQUARED of the base conic.
        coef: The row of ``coef`` of the surface: C_1 to C_n.
        n_coef: The number of coefficients.
        m: The number of rays in the block.
    """
    px, py = point
    r2, horner = scratch
    radius, one_plus_k = row[RADIUS], row[ONE_PLUS_K]
    radius_squared = row[RADIUS_SQUARED]
    for j in range(m):
        r2j = px[j] * px[j] + py[j] * py[j]
        r2[j] = r2j
        root = _sqrt(1.0 - one_plus_k * r2j / radius_squared)
        sag[j] = r2j / (radius * (1.0 + root))
    if n_coef > 0:
        # r2 * (C_1 + r2 * (C_2 + ... + r2 * C_n)).
        _horner(r2, horner, coef, n_coef, m)
        for j in range(m):
            sag[j] = sag[j] + r2[j] * horner[j]


@njit(inline="always", error_model="numpy", cache=True)
def _asphere_slopes(
    point: tuple,
    slopes: tuple,
    scratch: tuple,
    row: np.ndarray,
    dcoef: np.ndarray,
    n_coef: int,
    m: int,
) -> None:
    """The slopes dfdx and dfdy of ``EvenAsphere._surface_normal``.

    Args:
        point: The buffers of x and y of the points.
        slopes: Two buffers that receive d(sag)/dx and d(sag)/dy.
        scratch: Two buffers that receive r**2 and the sum of Horner's method.
        row: The float row of the surface: RADIUS, ONE_PLUS_K and
            RADIUS_SQUARED of the base conic.
        dcoef: The row of ``dcoef`` of the surface: i * C_i for i = 1 to n.
        n_coef: The number of coefficients.
        m: The number of rays in the block.
    """
    px, py = point
    dfdx, dfdy = slopes
    r2, horner = scratch
    radius, one_plus_k = row[RADIUS], row[ONE_PLUS_K]
    radius_squared = row[RADIUS_SQUARED]
    for j in range(m):
        xj, yj = px[j], py[j]
        r2j = xj * xj + yj * yj
        r2[j] = r2j
        denom = radius * _sqrt(1.0 - one_plus_k * r2j / radius_squared)
        dfdx[j] = xj / denom
        dfdy[j] = yj / denom
    if n_coef > 0:
        # d(sag)/dx = 2 x d(sag)/d(r2), with d(sag)/d(r2) of the polynomial
        # C_1 + 2 C_2 r2 + ... + n C_n r2^(n-1). The same for y.
        _horner(r2, horner, dcoef, n_coef, m)
        for j in range(m):
            dfdx[j] = dfdx[j] + 2.0 * px[j] * horner[j]
            dfdy[j] = dfdy[j] + 2.0 * py[j] * horner[j]


@njit(inline="always", error_model="numpy", cache=True)
def _asphere_normal(
    position: tuple,
    normal: tuple,
    work: np.ndarray,
    row: np.ndarray,
    dcoef: np.ndarray,
    n_coef: int,
    m: int,
) -> None:
    """``EvenAsphere._surface_normal`` at the hit points of a block.

    The slopes go into the buffers of nx and ny, which then receive the
    normal. The iteration of the surface is over, so this step only borrows
    the rows _R2 and _HORNER of ``work`` as scratch.
    """
    bx, by, _ = position
    bnx, bny, bnz = normal
    scratch = (work[_R2], work[_HORNER])
    _asphere_slopes((bx, by), (bnx, bny), scratch, row, dcoef, n_coef, m)
    for j in range(m):
        bnx[j], bny[j], bnz[j] = _unit_normal(bnx[j], bny[j])


@njit(inline="always", error_model="numpy", cache=True)
def _add_seeds(distance: np.ndarray, seed_stats: np.ndarray, m: int) -> None:
    """Add the seeds of a block to the seed summary of the surface.

    ``_effective_tolerance`` takes the largest abs(t) over the seeds of all
    rays, and NumPy's max is NaN if one seed is NaN. So the summary has a
    flag, SEED_NAN = 1.0 if a seed is NaN, and SEED_MAX, the largest abs(t)
    of the other seeds. The builder checks the tolerance after the kernel
    (``_floor_applies`` in ``optiland.surfaces._trace_plan``).

    Args:
        distance: The seed of each ray.
        seed_stats: The row of ``seed_stats`` of the surface. It accumulates
            over the blocks of a call, and over calls.
        m: The number of rays in the block.
    """
    seed_nan, seed_max = seed_stats[SEED_NAN], seed_stats[SEED_MAX]
    for j in range(m):
        t = distance[j]
        if t != t:
            seed_nan = 1.0
        elif abs(t) > seed_max:
            seed_max = abs(t)
    seed_stats[SEED_NAN], seed_stats[SEED_MAX] = seed_nan, seed_max


@njit(inline="always", error_model="numpy", cache=True)
def _seed(
    position: tuple,
    direction: tuple,
    distance: np.ndarray,
    work: np.ndarray,
    flags: np.ndarray,
    row: np.ndarray,
    coef: np.ndarray,
    n_coef: int,
    seed_stats: np.ndarray,
    m: int,
) -> None:
    """Start the iteration of a block: the seeds and the residual there.

    ``distance`` holds the distance to the base conic. A ray that misses the
    base conic (NaN) starts on the vertex plane z = 0 instead, as in
    ``_solve_distance_primal``, and gets the flag _NO_SEED. Then the
    residual at the seed decides which rays have converged.

    Args:
        position: The buffers of x, y and z, in the frame of the surface.
        direction: The buffers of L, M and N.
        distance: Holds the conic seeds and receives the seeds.
        work: The buffers of the iteration (rows _RESIDUAL to _HORNER).
        flags: The ray states of the iteration (rows _NO_SEED to _IS_ACTIVE).
        row: The float row of the surface.
        coef: The row of ``coef`` of the surface.
        n_coef: The number of coefficients.
        seed_stats: The row of ``seed_stats`` of the surface.
        m: The number of rays in the block.
    """
    bx, by, bz = position
    bL, bM, bN = direction
    px, py = work[_PX], work[_PY]
    no_seed = flags[_NO_SEED]
    for j in range(m):
        t = distance[j]
        missing = t != t
        t = _vertex_plane_distance(bz[j], bN[j]) if missing else t
        no_seed[j] = missing
        distance[j] = t
        px[j] = bx[j] + t * bL[j]
        py[j] = by[j] + t * bM[j]
    _add_seeds(distance, seed_stats, m)
    # _surface_residual: F(t) = sag(x + t L, y + t M) - (z + t N).
    sag, residual = work[_SAG], work[_RESIDUAL]
    _asphere_sag((px, py), sag, (work[_R2], work[_HORNER]), row, coef, n_coef, m)
    converged = flags[_CONVERGED]
    tol = row[TOL]
    for j in range(m):
        f = sag[j] - (bz[j] + distance[j] * bN[j])
        residual[j] = f
        converged[j] = abs(f) < tol


@njit(inline="always", error_model="numpy", cache=True)
def _count_active(work: np.ndarray, flags: np.ndarray, m: int) -> int:
    """Mark the rays that take the next step, and return their number.

    A ray is active until it converges or its residual is NaN. The generic
    solve stops early when no ray of the call is active.

    Args:
        work: The buffers of the iteration. Reads _RESIDUAL.
        flags: The ray states. Reads _CONVERGED and writes _IS_ACTIVE.
        m: The number of rays in the block.

    Returns:
        The number of active rays.
    """
    residual = work[_RESIDUAL]
    converged, is_active = flags[_CONVERGED], flags[_IS_ACTIVE]
    n_active = 0
    for j in range(m):
        f = residual[j]
        active = (not converged[j]) & (f == f)
        is_active[j] = active
        n_active += active
    return n_active


@njit(inline="always", error_model="numpy", cache=True)
def _step(
    position: tuple,
    direction: tuple,
    distance: np.ndarray,
    work: np.ndarray,
    flags: np.ndarray,
    row: np.ndarray,
    coef: np.ndarray,
    dcoef: np.ndarray,
    n_coef: int,
    m: int,
) -> None:
    """One Newton-Raphson step t - F(t) / (dF/dt) for the active rays.

    The expressions of ``_surface_residual_dt``, ``_regularize_signed`` and
    ``_surface_residual``. These invariants keep the bits of the generic
    solve:

    - fx and fy come from the normalized normal, as in
      ``_surface_residual_dt``. In exact arithmetic they are dfdx and dfdy,
      but dfdx and dfdy give other bits.
    - The step is computed for all rays, so that the loop vectorizes, but
      only the active rays keep it.
    - _PX and _PY hold the point at t only for the active rays. An inactive
      ray never becomes active again, so its point is not used.

    The arguments are those of ``_seed``, with ``dcoef``, the row of
    ``dcoef`` of the surface.
    """
    bx, by, bz = position
    bL, bM, bN = direction
    px, py = work[_PX], work[_PY]
    dfdx, dfdy = work[_DFDX], work[_DFDY]
    residual, t_next, sag = work[_RESIDUAL], work[_T_NEXT], work[_SAG]
    scratch = (work[_R2], work[_HORNER])
    _asphere_slopes((px, py), (dfdx, dfdy), scratch, row, dcoef, n_coef, m)
    for j in range(m):
        nx, ny, nz = _unit_normal(dfdx[j], dfdy[j])
        nz_safe = _sign_preserving_floor(nz, _DENOM_TAU)
        fx = -nx / nz_safe
        fy = -ny / nz_safe
        L, M, N = bL[j], bM[j], bN[j]
        fxL, fyM = fx * L, fy * M
        scale = abs(fxL) + abs(fyM) + abs(N)
        df_dt = _regularize_signed(fxL + fyM - N, _DENOM_TAU * _maximum(scale, 1.0))
        t = distance[j] - residual[j] / df_dt
        t_next[j] = t
        px[j] = bx[j] + t * L
        py[j] = by[j] + t * M
    _asphere_sag((px, py), sag, scratch, row, coef, n_coef, m)
    converged, is_active = flags[_CONVERGED], flags[_IS_ACTIVE]
    tol = row[TOL]
    for j in range(m):
        t = t_next[j]
        f = sag[j] - (bz[j] + t * bN[j])
        active = is_active[j]
        distance[j] = t if active else distance[j]
        residual[j] = f if active else residual[j]
        converged[j] = (abs(f) < tol) if active else converged[j]


@njit(inline="always", error_model="numpy", cache=True)
def _finish(distance: np.ndarray, work: np.ndarray, flags: np.ndarray, m: int) -> None:
    """The rule of ``_solve_distance_primal`` for the rays that missed.

    A ray with a NaN residual gets the distance NaN. So does a ray without a
    conic seed, unless it converged to a root in front of it (t > 0).

    Args:
        distance: The distance of each ray. Receives NaN for a ray that
            missed.
        work: The buffers of the iteration. Reads _RESIDUAL.
        flags: The ray states. Reads _NO_SEED and _CONVERGED.
        m: The number of rays in the block.
    """
    residual = work[_RESIDUAL]
    no_seed, converged = flags[_NO_SEED], flags[_CONVERGED]
    for j in range(m):
        t, f = distance[j], residual[j]
        dead = f != f
        if no_seed[j]:
            dead = dead or not (converged[j] and t > 0)
        distance[j] = math.nan if dead else t


@njit(inline="always", error_model="numpy", cache=True)
def _newton(
    position: tuple,
    direction: tuple,
    distance: np.ndarray,
    work: np.ndarray,
    flags: np.ndarray,
    row: np.ndarray,
    int_row: np.ndarray,
    coef: np.ndarray,
    dcoef: np.ndarray,
    seed_stats: np.ndarray,
    m: int,
) -> None:
    """``NewtonRaphsonGeometry._solve_distance_primal`` for the rays of a block.

    The rays of a block iterate together, until no ray is active or for
    MAX_ITER steps. In the generic solve, a converged ray takes a zero step,
    and the distance of a ray with a NaN residual becomes NaN at the next
    step or at the end. Here neither ray takes a step, and ``_finish`` sets
    NaN for a NaN residual. A ray that stops iterating never starts again,
    and other rays do not change its steps. So each ray takes the steps that
    it takes in the generic solve.

    The tolerance is TOL. The generic solve raises it to a round-off floor
    that depends on the seeds of all rays. The builder checks after the
    kernel, with ``seed_stats``, that the floor does not apply.

    Args:
        position: The buffers of x, y and z, in the frame of the surface.
        direction: The buffers of L, M and N.
        distance: Holds the distance to the base conic of each ray, and
            receives the distance to the asphere.
        work: The buffers of the iteration (rows _RESIDUAL to _HORNER).
        flags: The ray states of the iteration (rows _NO_SEED to _IS_ACTIVE).
        row: The float row of the surface.
        int_row: The integer row of the surface: NCOEF and MAX_ITER.
        coef: The row of ``coef`` of the surface.
        dcoef: The row of ``dcoef`` of the surface.
        seed_stats: The row of ``seed_stats`` of the surface.
        m: The number of rays in the block.
    """
    n_coef = int_row[NCOEF]
    _seed(position, direction, distance, work, flags, row, coef, n_coef, seed_stats, m)
    for _ in range(int_row[MAX_ITER]):
        if _count_active(work, flags, m) == 0:
            break
        _step(position, direction, distance, work, flags, row, coef, dcoef, n_coef, m)
    _finish(distance, work, flags, m)


@njit(inline="always", error_model="numpy", cache=True)
def _refracted(
    u: float,
    u2: float,
    L0: float,
    M0: float,
    N0: float,
    nx: float,
    ny: float,
    nz: float,
) -> tuple[float, float, float]:
    """The direction of one ray after ``RealRays.refract``.

    Args:
        u: The index ratio n1 / n2.
        u2: u**2, as ``RealRays.refract`` computes it.
        L0: The x direction cosine of the incident ray.
        M0: The y direction cosine of the incident ray.
        N0: The z direction cosine of the incident ray.
        nx: The x component of the surface normal.
        ny: The y component of the surface normal.
        nz: The z component of the surface normal.

    Returns:
        The direction cosines of the refracted ray.
    """
    nx, ny, nz, dot = _aligned_normal(L0, M0, N0, nx, ny, nz)
    # Snell's law in vector form. Total internal reflection gives a negative
    # radicand and NaN directions, as in RealRays.refract.
    root = _sqrt(1.0 - u2 * (1.0 - dot * dot))
    return (
        u * L0 + nx * root - u * nx * dot,
        u * M0 + ny * root - u * ny * dot,
        u * N0 + nz * root - u * nz * dot,
    )


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
        nxj, nyj, nzj = bnx[j], bny[j], bnz[j]
        bL[j], bM[j], bN[j] = _refracted(u, u2, L0, M0, N0, nxj, nyj, nzj)


@njit(inline="always", error_model="numpy", cache=True)
def _index_ratios(
    ratios: tuple,
    ray_values: np.ndarray,
    int_row: np.ndarray,
    row: np.ndarray,
    start: int,
    m: int,
) -> None:
    """u = n1 / n2 and u * u for each ray, from N1_ROW, N2_ROW, N1 and N2.

    One index per ray makes u an array in ``RealRays.refract``, and NumPy
    squares an array as u * u.
    """
    ratio, ratio_squared = ratios
    row1, row2 = int_row[N1_ROW], int_row[N2_ROW]
    for j in range(m):
        n1 = ray_values[row1, start + j] if row1 != NO_ROW else row[N1]
        n2 = ray_values[row2, start + j] if row2 != NO_ROW else row[N2]
        u = n1 / n2
        ratio[j] = u
        ratio_squared[j] = u * u


@njit(inline="always", error_model="numpy", cache=True)
def _refract_per_ray(
    direction: tuple, incident: tuple, normal: tuple, ratios: tuple, m: int
) -> None:
    """``RealRays.refract`` with the index ratios of ``_index_ratios``."""
    bL, bM, bN = direction
    bL0, bM0, bN0 = incident
    bnx, bny, bnz = normal
    ratio, ratio_squared = ratios
    for j in range(m):
        L0, M0, N0 = bL[j], bM[j], bN[j]
        bL0[j], bM0[j], bN0[j] = L0, M0, N0
        u, u2, nxj, nyj, nzj = ratio[j], ratio_squared[j], bnx[j], bny[j], bnz[j]
        bL[j], bM[j], bN[j] = _refracted(u, u2, L0, M0, N0, nxj, nyj, nzj)


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


@njit(inline="always", error_model="numpy", cache=True)
def _trace_blocks(
    rays_in: tuple,
    ints: np.ndarray,
    floats: np.ndarray,
    ray_values: np.ndarray | None,
    coef: np.ndarray | None,
    dcoef: np.ndarray | None,
    exponents: np.ndarray,
    clipped: np.ndarray,
    seed_stats: np.ndarray | None,
    rays_out: tuple,
    records: tuple,
    record: bool,
    per_ray: bool,
    newton: bool,
) -> None:
    """The loop nest of both kernels, ``trace_plan`` and ``trace_basic_plan``.

    ``per_ray`` and ``newton`` are constants of each kernel. Numba removes
    the branches that the constants do not take before it compiles the
    kernel, so ``trace_basic_plan`` has no code for per-ray values and even
    aspheres, and gets None for ``ray_values``, ``coef``, ``dcoef`` and
    ``seed_stats``. On a basic plan, both kernels run the same steps.

    The arguments are those of ``trace_plan``, and two flags: ``per_ray``,
    True if the kernel takes rows of ``ray_values``, and ``newton``, True if
    it takes even aspheres.
    """
    n_rays = rays_in[0].shape[0]
    n_surfaces = ints.shape[0]
    bx, by, bz = np.empty(_BLOCK), np.empty(_BLOCK), np.empty(_BLOCK)
    bL, bM, bN = np.empty(_BLOCK), np.empty(_BLOCK), np.empty(_BLOCK)
    bL0, bM0, bN0 = np.empty(_BLOCK), np.empty(_BLOCK), np.empty(_BLOCK)
    bi, bopd, bdist = np.empty(_BLOCK), np.empty(_BLOCK), np.empty(_BLOCK)
    normal = (np.empty(_BLOCK), np.empty(_BLOCK), np.empty(_BLOCK))
    if per_ray:
        ratios = (np.empty(_BLOCK), np.empty(_BLOCK))
    if newton:
        work = np.empty((_WORK_ROWS, _BLOCK))
        flags = np.empty((_FLAG_ROWS, _BLOCK), dtype=np.bool_)
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
                # EvenAsphere: the Newton-Raphson iteration from the conic seed.
                if newton and ints[s, NEWTON]:
                    _newton(
                        position,
                        direction,
                        bdist,
                        work,
                        flags,
                        row,
                        ints[s],
                        coef[s],
                        dcoef[s],
                        seed_stats[s],
                        m,
                    )
                # HomogeneousPropagation: move, absorb, add the optical path.
                # Each value is the same for all rays, or comes from a row of
                # ray_values.
                _propagate(position, direction, bdist, m)
                absorb_row = ints[s, ABSORB_ROW]
                if absorb_row != NO_ROW:
                    block_exponents = exponents[absorb_row, start : start + m]
                    alpha_row = ints[s, ALPHA_ROW]
                    if per_ray and alpha_row != NO_ROW:
                        alphas = ray_values[alpha_row, start : start + m]
                        _absorption_exponents_per_ray(block_exponents, bdist, alphas, m)
                    else:
                        _absorption_exponents(block_exponents, bdist, row, m)
                index_row = ints[s, OPD_INDEX_ROW]
                if per_ray and index_row != NO_ROW:
                    indices = ray_values[index_row, start : start + m]
                    _add_optical_path_per_ray(bopd, bdist, indices, m)
                else:
                    _add_optical_path(bopd, bdist, row, m)
                # BaseAperture.clip.
                if aperture != NO_APERTURE:
                    block_clipped = clipped[ints[s, CLIP_ROW], start : start + m]
                    _clip(position, bi, block_clipped, row, aperture, m)
                # The surface normal at each hit point.
                n_coef = ints[s, NCOEF]
                if kind == PLANE:
                    _plane_normal(normal, m)
                elif newton and ints[s, NEWTON] and n_coef > 0:
                    # Without coefficients, the asphere normal is the conic
                    # normal below.
                    _asphere_normal(position, normal, work, row, dcoef[s], n_coef, m)
                else:
                    _conic_normal(position, normal, row, m)
                # RefractiveReflectiveModel.interact_real_rays.
                if ints[s, REFLECT]:
                    _reflect(direction, incident, normal, m)
                elif per_ray and (
                    ints[s, N1_ROW] != NO_ROW or ints[s, N2_ROW] != NO_ROW
                ):
                    _index_ratios(ratios, ray_values, ints[s], row, start, m)
                    _refract_per_ray(direction, incident, normal, ratios, m)
                else:
                    _refract(direction, incident, normal, row, m)
                _globalize(position, direction, row, rotate, m)
            if record:
                _store_records(state, records, s, start, m)
        _store_block(state, rays_out, start, m)


@njit(error_model="numpy", cache=True)
def trace_plan(
    rays_in: tuple,
    ints: np.ndarray,
    floats: np.ndarray,
    ray_values: np.ndarray,
    coef: np.ndarray,
    dcoef: np.ndarray,
    exponents: np.ndarray,
    clipped: np.ndarray,
    seed_stats: np.ndarray,
    rays_out: tuple,
    records: tuple,
    record: bool,
) -> None:
    """Trace the rays through all surfaces of a plan.

    The input arrays are not changed. The intensity in ``rays_out`` and in
    ``records`` includes the clips of the apertures, but not the absorption.
    The builder calls ``trace_basic_plan`` for a basic plan.

    Args:
        rays_in: x, y, z, L, M, N, i and opd of the rays: 1-D float64
            arrays of one length.
        ints: The integer table (see the module docstring).
        floats: The float table.
        ray_values: The values that differ from ray to ray, one row each.
        coef: The coefficients C_i of each even asphere, one row per
            surface. Without an asphere, a table without rows.
        dcoef: The products i * C_i, with the shape of ``coef``.
        exponents: One row per absorbing medium. Row ABSORB_ROW of a surface
            receives the exponent -alpha * t * 1e3 of each ray in the medium
            before that surface.
        clipped: One row per aperture. Row CLIP_ROW of a surface receives
            True for each ray that its aperture clips.
        seed_stats: One row per surface, with the columns SEED_NAN and
            SEED_MAX. The kernel adds the seeds of each even asphere to its
            row and never resets a row, so a row accumulates over the blocks
            and over calls. The caller fills the table with zeros. Without
            an asphere, a table without rows.
        rays_out: Receive x, y, z, L, M, N, i and opd after the last surface,
            then L0, M0 and N0: the direction of each ray before the last
            refraction or reflection, in the local frame of that surface.
        records: x, y, z, L, M, N, intensity and opd, each with one row per
            surface. Row s receives the ray state after surface s.
        record: Whether to write ``records``.
    """
    _trace_blocks(
        rays_in,
        ints,
        floats,
        ray_values,
        coef,
        dcoef,
        exponents,
        clipped,
        seed_stats,
        rays_out,
        records,
        record,
        per_ray=True,
        newton=True,
    )


@njit(error_model="numpy", cache=True)
def trace_basic_plan(
    rays_in: tuple,
    ints: np.ndarray,
    floats: np.ndarray,
    exponents: np.ndarray,
    clipped: np.ndarray,
    rays_out: tuple,
    records: tuple,
    record: bool,
) -> None:
    """``trace_plan`` for a basic plan: no rows in ``ray_values``, no asphere.

    The arguments are those of ``trace_plan``, without ``ray_values``,
    ``coef``, ``dcoef`` and ``seed_stats``.
    """
    _trace_blocks(
        rays_in,
        ints,
        floats,
        None,
        None,
        None,
        exponents,
        clipped,
        None,
        rays_out,
        records,
        record,
        per_ray=False,
        newton=False,
    )
