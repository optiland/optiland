"""NumPy-only fast path of ``SurfaceGroup.trace``. The generic trace
(``Surface.trace``) is the reference: it runs on every backend, and this
module must give its results bit for bit.

``trace_with_plan`` reads the live objects at each trace and builds a plan:
two small tables with one row per surface (the kernel docstring documents
the columns). The compiled kernel
(``optiland.backend.numpy_backend._trace_kernel``) then traces blocks of
rays through all surfaces in one loop, so the Python cost is paid once per
trace. No plan is kept between traces.

The builder computes each per-surface scalar of the generic trace (for
example the cosines and sines of rotations, 1 + k, R**2, the squared
aperture radii, u = n1/n2 and u**2, the absorption coefficient, i * C_i of
an even asphere) with the same IEEE 754 operations in the same order, so
the kernel gets the same values. Python floats give the bits of NumPy
float64 for + - * /. NumPy computes ``x**2`` as ``x * x`` only for an
array, and calls the C function ``pow`` for a scalar (see ``_index_ratio``).

The developer page "Performance Internals" lists the domain of the plan,
with the reason for each limit. Outside the domain, ``trace_with_plan``
returns False and the generic trace runs.

``RealRayTracer`` traces the rays through all surfaces, then propagates them
past the image surface by its thickness (usually 0). It gives this
propagation to the plan as a ``FinalPropagation``. In its domain, the plan
does both in one pass: the kernel function ``_propagate_past_last_surface``
moves each part of ``_FINAL_PART`` rays right after the kernel has traced
it.

With ``be.set_trace_threads(n)`` and n > 1, a bundle of
``_MIN_THREADED_RAYS`` rays or more is traced in parts of whole blocks, on
the calling thread and the threads of a private pool (``_run_kernel`` and
``_run_parts``). Each kernel call writes only the rays of its part. No
kernel step combines values of several rays. The one exception, the Newton
seed summary, is a NaN flag and a maximum, which do not depend on the order
of the parts. So each number of threads gives the bits of one thread.

Manuel Fragata Mendes, 2026
"""

from __future__ import annotations

import concurrent.futures
import functools
import itertools
import math
import os
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, TypeVar

import numba
import numpy as np

import optiland.backend as be
from optiland.geometries.newton_raphson import _CONV_EPS_MULTIPLIER
from optiland.surfaces._chunked_trace import _RECORDED

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from types import ModuleType

    from optiland._types import ScalarOrArray
    from optiland.geometries import EvenAsphere
    from optiland.materials.base import BaseMaterial
    from optiland.propagation.base import BasePropagationModel
    from optiland.rays import BaseRays, RealRays
    from optiland.surfaces.standard_surface import Surface

_T = TypeVar("_T")

# Private switch for the equivalence tests.
_ENABLED = True

# The ray arrays that the kernel reads (x to opd) and writes again, and the
# arrays that it writes in addition: the direction of each ray before the
# last refraction. The order is the order of the kernel arguments.
_RAY_STATE = ("x", "y", "z", "L", "M", "N", "i", "opd")
_OUTPUTS = (*_RAY_STATE, "L0", "M0", "N0")

# The records that Surface.reset clears: those of Surface._record_real
# (_RECORDED, in the order of the kernel records), and two that a real trace
# leaves empty. A test compares this list with the arrays of a reset surface.
_CLEARED = (*_RECORDED, "u", "aoi")

# Read-only: a write into a released record raises an error, so it cannot
# change an array that all surfaces share.
_RELEASED = np.empty(0)
_RELEASED.flags.writeable = False

# Tables without rows, for the records of a trace without records, the
# exponents of a plan without absorption, the clip masks of a plan without
# apertures, and the coefficients of a plan without an even asphere. The
# kernel never writes into them.
_NO_ROWS = np.empty((0, 1))
_NO_CLIPS = np.empty((0, 1), dtype=np.bool_)

# Rays per part of the intensity pass. Each NumPy operation of the pass makes
# a new array. _chunked_trace uses parts of the same size (_CHUNK_SIZE) for
# the same reason: an array of 65536 float64 values (512 KiB) stays below the
# size from which the heap of 64-bit Windows gives memory back to the system.
_PART = 65536

# Rays per kernel call of a plan with a final step (_FinalStep). The final
# step reads x, y, z, L, M and N of the rays that the kernel has just
# written: 768 KiB for 16384 rays, which stay in the L2 or L3 cache between
# the two calls. A much larger part would leave the cache. A much smaller
# part would make more calls, and each call has a fixed cost to pass its
# arguments. A kernel takes only a first ray at a block boundary, so the
# size is a multiple of the block size (_BLOCK, 256).
_FINAL_PART = 16384

# Threads of the trace (be.set_trace_threads). The limits come from timings
# of the first version of this code on a CPU with 12 cores (see #1147).
# Measure them again when the kernel changes.

# The smallest bundle that the kernel traces on several threads. For a
# smaller bundle, the pool threads cost about as much time as they save.
_MIN_THREADED_RAYS = 8192
# The parts of a bundle for each thread. A thread takes the next part when
# it completes one, so a thread that the system slows down does not hold
# back the others.
_PARTS_PER_THREAD = 4
# The fewest blocks of rays in a part (8 blocks are 2048 rays). A smaller
# part costs about as much to hand to a thread as it saves.
_MIN_PART_BLOCKS = 8
# The smallest bundle whose intensity pass runs on several threads. Each
# NumPy operation of the pass takes the GIL to start, so the parts of a
# smaller bundle are too short to gain.
_MIN_THREADED_INTENSITY_RAYS = 65536
# The most threads of the intensity pass. With more threads, its NumPy
# operations wait for the GIL.
_MAX_INTENSITY_THREADS = 4

# machine_eps of a float64 array, as _effective_tolerance uses it:
# np.finfo(np.float64).eps, written as a number so that importing this module
# needs no NumPy value (the docs build replaces NumPy with a mock).
_FLOAT64_EPS = 2.0**-52
# The range of max_iter that fits the int64 table.
_INT64 = np.iinfo(np.int64)


class _Types(NamedTuple):
    """The exact types of the domain of the plan."""

    rays: type
    object_surface: type
    surfaces: tuple[type, ...]
    coordinate_system: type
    plane: type
    standard: type
    even_asphere: type
    radial: type
    offset_radial: type
    interaction: type
    propagation: type


@functools.cache
def _types() -> _Types:
    """Return the exact types of the domain of the plan.

    The imports are local because ``SurfaceGroup`` imports this module while
    ``optiland.surfaces`` is still being imported.
    """
    from optiland.coordinate_system import CoordinateSystem
    from optiland.geometries import EvenAsphere, Plane, StandardGeometry
    from optiland.interactions.refractive_reflective_model import (
        RefractiveReflectiveModel,
    )
    from optiland.physical_apertures import OffsetRadialAperture, RadialAperture
    from optiland.propagation.homogeneous import HomogeneousPropagation
    from optiland.rays.real_rays import RealRays
    from optiland.surfaces.image_surface import ImageSurface
    from optiland.surfaces.object_surface import ObjectSurface
    from optiland.surfaces.standard_surface import Surface

    return _Types(
        rays=RealRays,
        object_surface=ObjectSurface,
        surfaces=(Surface, ImageSurface),
        coordinate_system=CoordinateSystem,
        plane=Plane,
        standard=StandardGeometry,
        even_asphere=EvenAsphere,
        radial=RadialAperture,
        offset_radial=OffsetRadialAperture,
        interaction=RefractiveReflectiveModel,
        propagation=HomogeneousPropagation,
    )


@functools.cache
def _kernel() -> ModuleType:
    """Return the kernel module, imported at the first trace that can use it.

    The import makes the Numba dispatchers, which takes some milliseconds. A
    process that traces no real rays with NumPy does not pay for it.
    """
    from optiland.backend.numpy_backend import _trace_kernel

    return _trace_kernel


@dataclass
class FinalPropagation:
    """The propagation past the last surface of a trace by its thickness.

    ``RealRayTracer`` traces the rays through all surfaces, then calls
    ``model.propagate(rays, t)``: the rays propagate past the image surface
    by its thickness (usually 0) in the medium after it.
    ``trace_with_plan`` does this propagation in its pass if it can, and
    then sets ``applied``. ``finish`` propagates the rays if the plan did
    not.

    Attributes:
        model: The propagation model of the medium after the last surface.
        t: The thickness of the last surface.
        applied: True if the rays have been propagated.
    """

    model: BasePropagationModel
    t: ScalarOrArray
    applied: bool = False

    @classmethod
    def after(cls, surface: Surface) -> FinalPropagation:
        """Return the propagation past ``surface`` by its thickness.

        Args:
            surface: The last surface of the trace.

        Returns:
            The propagation, not applied yet.
        """
        return cls(surface.material_post.propagation_model, surface.thickness)

    def finish(self, rays: RealRays) -> None:
        """Propagate ``rays`` past the last surface, unless the plan did it.

        Args:
            rays: The rays after the last surface. Changed in place.
        """
        if not self.applied:
            self.model.propagate(rays, self.t)
            self.applied = True


class _FinalStep(NamedTuple):
    """The final step of a plan, in the form that the kernel takes.

    The values are the arguments of ``_propagate_past_last_surface`` after
    its range of rays.

    Attributes:
        t: The distance, the thickness of the last surface (mm).
        absorb_row: The row of the exponents of the medium after the last
            surface, or NO_ROW if the medium does not absorb.
        alpha_row: The row of ``ray_values`` with the alpha of each ray, or
            NO_ROW if all rays have the same alpha.
        alpha: The alpha of all rays (1/um), when ``alpha_row`` is NO_ROW.
    """

    t: float
    absorb_row: int
    alpha_row: int
    alpha: float


@dataclass(frozen=True)
class _Plan:
    """The tables of one trace, and the facts about them that the trace needs.

    Attributes:
        ints: The integer table, one row per surface (see the kernel).
        floats: The float table, one row per surface.
        ray_values: The material values that differ from ray to ray, one row
            each (see ``_PerRayValues``).
        coef: The coefficients C_i of each even asphere, one row per surface.
        dcoef: The products i * C_i, one row per surface.
        basic: True if ``ray_values`` has no rows and no surface is an even
            asphere. The kernel ``trace_basic_plan`` then traces the plan.
        moves_rays: False if all surfaces are object surfaces. The rays then
            keep their arrays.
        absorbing: The number of absorbing media, the medium of a final
            step included: the rows of the exponents.
        clips: The number of apertures: the rows of the clip masks.
        aspheres: The number of even aspheres. With one or more, the kernel
            writes a seed summary for ``_floor_applies``.
        final_step: The propagation past the last surface, or None if the
            plan has no final step.
    """

    ints: np.ndarray
    floats: np.ndarray
    ray_values: np.ndarray
    coef: np.ndarray
    dcoef: np.ndarray
    basic: bool
    moves_rays: bool
    absorbing: int
    clips: int
    aspheres: int
    final_step: _FinalStep | None


class _Form(NamedTuple):
    """A material value of one trace (n or k), in the form the kernel takes.

    Attributes:
        per_ray: True if the rays have different values.
        value: The value as a Python float, or with ``per_ray`` the 1-D array
            of one value per ray.
        returned: The object that the material returned. ``_index_ratio``
            needs its type.
    """

    per_ray: bool
    value: float | np.ndarray
    returned: object


def _form(value: object, n: int) -> _Form | None:
    """Return the form of the value of ``n(w)`` or ``k(w)`` for ``n`` rays.

    The domain has float64 values: a Python float, a NumPy float64, a 0-d
    array, or a 1-D array of one value or of one value per ray. A 1-D array
    is uniform if it has one element, if it is a broadcast view (stride 0) or
    if all its elements have the same bits. Other values give None.
    """
    if type(value) is float or type(value) is np.float64:
        return _Form(False, float(value), value)
    if type(value) is not np.ndarray or value.dtype != np.float64:
        return None
    if value.ndim == 0:
        return _Form(False, float(value), value)
    if value.ndim != 1 or value.shape[0] not in (1, n):
        return None
    bits = value.view(np.int64)
    if value.shape[0] == 1 or value.strides[0] == 0 or bool((bits == bits[0]).all()):
        return _Form(False, float(value[0]), value)
    return _Form(True, value, value)


def _float64_scalar(value: object) -> float | None:
    """Return a 0-d float64 array as a Python float, or None for other values."""
    if type(value) is np.ndarray and value.ndim == 0 and value.dtype == np.float64:
        return float(value)
    return None


def _number(value: object) -> float | None:
    """Return a Python, NumPy or 0-d float64 number as a float, or None."""
    if type(value) is int or type(value) is float or type(value) is np.float64:
        return float(value)
    return _float64_scalar(value)


def _rotation(angle: float) -> tuple[float, float]:
    """Return cos and sin of ``angle`` as ``RealRays.rotate_x`` computes them."""
    # be.* calls repeat those of the generic trace.
    value = be.array(angle)
    return float(be.cos(value)), float(be.sin(value))


def _index_ratio(n1: _Form, n2: _Form) -> tuple[float, float]:
    """Return u = n1 / n2 and u**2 with the bits of ``RealRays.refract``.

    If ``u`` is an ndarray, NumPy computes ``u**2`` as ``u * u``
    (``numpy.square``). If ``u`` is a scalar (Python or NumPy), Python and
    NumPy call the C function ``pow``, which libm does not promise to round
    as ``u * u``. ``u`` is an ndarray if n1 or n2 is an ndarray with ndim >= 1.

    Args:
        n1: The uniform refractive index before the surface.
        n2: The uniform refractive index after the surface.

    Returns:
        ``u`` and ``u**2`` as Python floats.
    """
    if _is_array(n1.returned) or _is_array(n2.returned):
        u = n1.value / n2.value
        return u, u * u
    # Two scalars: repeat the expressions of the generic trace on the objects
    # that the materials returned, so that u**2 calls the same pow.
    u = n1.returned / n2.returned
    return float(u), float(u**2)


def _is_array(value: object) -> bool:
    """Return True if ``value`` is an ndarray with ndim >= 1."""
    return type(value) is np.ndarray and value.ndim >= 1


def _absorbs(k: _Form) -> bool:
    """Return the decision of ``HomogeneousPropagation.propagate``: any k > 0."""
    if k.per_ray:
        # be.* calls repeat those of the generic trace.
        return bool(be.any(k.value > 0))
    return k.value > 0


def _newton_values(
    geometry: EvenAsphere,
) -> tuple[list[float], list[float], int, float] | None:
    """Return the values of the Newton-Raphson iteration of an even asphere.

    The coefficients are a list or a 1-D array of numbers (see ``_number``),
    ``tol`` is a number, and ``max_iter`` is an int that fits the int64
    table. The products i * C_i use the objects of the geometry, as
    ``EvenAsphere._surface_normal`` does, so they get its bits.

    Args:
        geometry: An EvenAsphere.

    Returns:
        The coefficients C_i, the products i * C_i, ``max_iter`` and ``tol``,
        or None outside the domain.
    """
    values = geometry.coefficients
    if type(values) is not list and not (
        type(values) is np.ndarray and values.ndim == 1
    ):
        return None
    coefficients = [_number(c) for c in values]
    tol = _number(geometry.tol)
    max_iter = geometry.max_iter
    if None in coefficients or tol is None or type(max_iter) is not int:
        return None
    if not _INT64.min <= max_iter <= _INT64.max:
        return None
    derivatives = [float(i * c) for i, c in enumerate(values, start=1)]
    return coefficients, derivatives, max_iter, tol


def _memo(
    memo: dict[int, tuple[object, _T]], key: object, compute: Callable[[object], _T]
) -> _T:
    """Return ``compute(key)``, computed once for each object of one build.

    The memo is keyed by ``id(key)``. Each entry keeps ``key`` alive, so no
    other object can get the same id while the memo exists.
    """
    entry = memo.get(id(key))
    if entry is None:
        entry = memo[id(key)] = (key, compute(key))
    return entry[1]


def _material_value(evaluate: Callable[[np.ndarray], object], w: np.ndarray) -> object:
    """Return ``evaluate(w)``, or None if it raises an exception.

    A material can raise an error, for example for a wavelength outside its
    data. The plan then returns False, and the generic trace raises the
    error itself.
    """
    try:
        return evaluate(w)
    except Exception:  # the generic trace raises it again
        return None


class _PerRayValues:
    """The rows of ``ray_values``: material values that differ from ray to ray.

    Examples are the refractive index of a glass in a bundle of several
    wavelengths, and its absorption coefficient. Each array gets one row.
    """

    def __init__(self) -> None:
        self.rows: list[np.ndarray] = []
        self._row_of: dict[int, tuple[object, int]] = {}

    def row(self, values: np.ndarray) -> int:
        """Return the row of ``values``, an array of one value per ray."""
        return _memo(self._row_of, values, self._add)

    def _add(self, values: np.ndarray) -> int:
        self.rows.append(values)
        return len(self.rows) - 1

    def table(self) -> np.ndarray:
        """Return the rows as one C-contiguous table."""
        return np.stack(self.rows) if self.rows else _NO_ROWS


class _Builder:
    """Builds the tables of the plan of one trace, one row per surface.

    Each ``_*_columns`` method fills the columns of one step of a surface and
    returns False if the surface is outside the domain of the plan. Each
    material is evaluated once per trace.
    """

    def __init__(
        self, rays: RealRays, n: int, types: _Types, kernel: ModuleType
    ) -> None:
        self.types = types
        self.kernel = kernel
        self.wavelengths = rays.w
        self.n = n
        self.int_rows: list[list[int]] = []
        self.float_rows: list[list[float]] = []
        self.moves_rays = False
        self.absorbing = 0
        self.clips = 0
        self.final_step: _FinalStep | None = None
        self.values = _PerRayValues()
        # The C_i and i * C_i of each even asphere, by the row of its surface.
        self.coefficients: dict[int, tuple[list[float], list[float]]] = {}
        self._indices: dict[int, tuple[object, _Form | None]] = {}
        self._extinctions: dict[int, tuple[object, _Form | None]] = {}
        self._alphas: dict[int, tuple[object, _Form | None]] = {}
        # Rows with the defaults of the layout table. Python lists are faster
        # to fill than arrays; plan() converts them once.
        self._empty_ints = [0] * kernel.INT_COLUMNS
        self._empty_ints[kernel.KIND] = kernel.OBJECT
        self._empty_ints[kernel.ABSORB_ROW] = kernel.NO_ROW
        self._empty_ints[kernel.APERTURE] = kernel.NO_APERTURE
        self._empty_ints[kernel.CLIP_ROW] = kernel.NO_ROW
        self._empty_ints[kernel.N1_ROW] = kernel.NO_ROW
        self._empty_ints[kernel.N2_ROW] = kernel.NO_ROW
        self._empty_ints[kernel.OPD_INDEX_ROW] = kernel.NO_ROW
        self._empty_ints[kernel.ALPHA_ROW] = kernel.NO_ROW
        self._empty_floats = [0.0] * kernel.FLOAT_COLUMNS

    def add(self, surface: Surface) -> bool:
        """Add the row of ``surface``. Return False outside the domain."""
        ints, floats = list(self._empty_ints), list(self._empty_floats)
        # An ObjectSurface keeps the default row: OBJECT, record only.
        if type(surface) is not self.types.object_surface:
            if not (
                type(surface) in self.types.surfaces
                and self._frame_columns(surface, ints, floats)
                and self._geometry_columns(surface, ints, floats)
                and self._aperture_columns(surface, ints, floats)
                and self._medium_columns(surface, ints, floats)
                and self._interaction_columns(surface, ints, floats)
            ):
                return False
            self.moves_rays = True
        self.int_rows.append(ints)
        self.float_rows.append(floats)
        return True

    def add_final_step(self, final: FinalPropagation) -> None:
        """Add the propagation past the last surface, if it is in the domain.

        The domain: exactly ``HomogeneousPropagation``, a thickness that is a
        number (see ``_number``), an absorption coefficient that the plan can
        take, and at least one surface step. A plan of object surfaces only
        gives the rays their own arrays back, so it has no new arrays to
        propagate. Outside the domain, the plan has no final step, and the
        caller propagates the rays after the plan.

        Args:
            final: The propagation past the last surface.
        """
        if not self.moves_rays or type(final.model) is not self.types.propagation:
            return
        t = _number(final.t)
        if t is None:
            return
        absorption = self._absorption(final.model.material)
        if absorption is not None:
            self.final_step = _FinalStep(t, *absorption)

    def plan(self) -> _Plan:
        """Return the plan of the rows and the final step added so far."""
        coef, dcoef = self._coefficient_tables()
        return _Plan(
            ints=np.array(self.int_rows, dtype=np.int64),
            floats=np.array(self.float_rows, dtype=np.float64),
            ray_values=self.values.table(),
            coef=coef,
            dcoef=dcoef,
            basic=not self.values.rows and not self.coefficients,
            moves_rays=self.moves_rays,
            absorbing=self.absorbing,
            clips=self.clips,
            aspheres=len(self.coefficients),
            final_step=self.final_step,
        )

    def _coefficient_tables(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the tables ``coef`` and ``dcoef``, one row per surface.

        Row s holds the C_i or the i * C_i of the even asphere of surface s,
        padded with zeros. The rows of other surfaces are zero.
        """
        if not self.coefficients:
            return _NO_ROWS, _NO_ROWS
        width = max(len(c) for c, _ in self.coefficients.values())
        coef = np.zeros((len(self.int_rows), width))
        dcoef = np.zeros_like(coef)
        for s, (c, d) in self.coefficients.items():
            coef[s, : len(c)] = c
            dcoef[s, : len(d)] = d
        return coef, dcoef

    def _frame_columns(
        self, surface: Surface, ints: list[int], floats: list[float]
    ) -> bool:
        """``CoordinateSystem.localize`` and ``globalize``.

        CoordinateSystem skips the rotation by an angle of zero, and rotates
        by any other angle, also by NaN.
        """
        kernel, cs = self.kernel, surface.geometry.cs
        if type(cs) is not self.types.coordinate_system or cs.reference_cs is not None:
            return False
        x, y, z = _float64_scalar(cs.x), _float64_scalar(cs.y), _float64_scalar(cs.z)
        rx, ry, rz = (
            _float64_scalar(cs.rx),
            _float64_scalar(cs.ry),
            _float64_scalar(cs.rz),
        )
        if None in (x, y, z, rx, ry, rz):
            return False
        floats[kernel.TX_IN], floats[kernel.TY_IN], floats[kernel.TZ_IN] = -x, -y, -z
        floats[kernel.TX_OUT], floats[kernel.TY_OUT], floats[kernel.TZ_OUT] = x, y, z
        rotate = 0
        if rx:
            rotate |= kernel.ROTATE_X
            floats[kernel.COS_X_IN], floats[kernel.SIN_X_IN] = _rotation(-rx)
            floats[kernel.COS_X_OUT], floats[kernel.SIN_X_OUT] = _rotation(rx)
        if ry:
            rotate |= kernel.ROTATE_Y
            floats[kernel.COS_Y_IN], floats[kernel.SIN_Y_IN] = _rotation(-ry)
            floats[kernel.COS_Y_OUT], floats[kernel.SIN_Y_OUT] = _rotation(ry)
        if rz:
            rotate |= kernel.ROTATE_Z
            floats[kernel.COS_Z_IN], floats[kernel.SIN_Z_IN] = _rotation(-rz)
            floats[kernel.COS_Z_OUT], floats[kernel.SIN_Z_OUT] = _rotation(rz)
        ints[kernel.ROTATE] = rotate
        return True

    def _geometry_columns(
        self, surface: Surface, ints: list[int], floats: list[float]
    ) -> bool:
        """The step kind, and the values of the distance and the normal.

        An EvenAsphere gets the columns of its base conic, because its
        Newton-Raphson iteration starts at the distance to the base conic.
        """
        kernel, geometry = self.kernel, surface.geometry
        if type(geometry) is self.types.plane:
            ints[kernel.KIND] = kernel.PLANE
            return True
        is_asphere = type(geometry) is self.types.even_asphere
        if type(geometry) is not self.types.standard and not is_asphere:
            return False
        radius, conic = _float64_scalar(geometry.radius), _float64_scalar(geometry.k)
        if radius is None or conic is None:
            return False
        ints[kernel.KIND] = kernel.FLAT if math.isinf(radius) else kernel.CONIC
        floats[kernel.RADIUS], floats[kernel.CONIC_K] = radius, conic
        # 1 + k and R**2 of StandardGeometry. R is a 0-d array, and NumPy
        # computes R**2 of an array as R * R.
        floats[kernel.ONE_PLUS_K] = 1.0 + conic
        floats[kernel.RADIUS_SQUARED] = radius * radius
        if is_asphere:
            return self._newton_columns(geometry, ints, floats)
        return True

    def _newton_columns(
        self, geometry: EvenAsphere, ints: list[int], floats: list[float]
    ) -> bool:
        """The Newton-Raphson iteration of an EvenAsphere.

        Fills NEWTON, NCOEF, MAX_ITER and TOL, and keeps the coefficients for
        the rows of ``coef`` and ``dcoef``.
        """
        values = _newton_values(geometry)
        if values is None:
            return False
        coefficients, derivatives, max_iter, tol = values
        kernel = self.kernel
        ints[kernel.NEWTON] = 1
        ints[kernel.NCOEF] = len(coefficients)
        ints[kernel.MAX_ITER] = max_iter
        floats[kernel.TOL] = tol
        # add() appends the row of this surface after its columns.
        self.coefficients[len(self.int_rows)] = (coefficients, derivatives)
        return True

    def _aperture_columns(
        self, surface: Surface, ints: list[int], floats: list[float]
    ) -> bool:
        """``BaseAperture.clip`` and the root preference of a radial aperture."""
        kernel, aperture = self.kernel, surface.aperture
        if aperture is None:
            return True
        if type(aperture) is self.types.radial:
            ints[kernel.APERTURE] = kernel.RADIAL
            offsets = (0.0, 0.0)
        elif type(aperture) is self.types.offset_radial:
            ints[kernel.APERTURE] = kernel.OFFSET_RADIAL
            offsets = (_number(aperture.offset_x), _number(aperture.offset_y))
        else:
            return False
        radii = (_number(aperture.r_max), _number(aperture.r_min))
        if None in offsets or None in radii:
            return False
        floats[kernel.OFFSET_X], floats[kernel.OFFSET_Y] = offsets
        # r_max**2 and r_min**2 of contains(), on the numbers of the aperture.
        floats[kernel.R_MAX_SQUARED] = float(aperture.r_max**2)
        floats[kernel.R_MIN_SQUARED] = float(aperture.r_min**2)
        ints[kernel.CLIP_ROW] = self.clips
        self.clips += 1
        return True

    def _medium_columns(
        self, surface: Surface, ints: list[int], floats: list[float]
    ) -> bool:
        """``HomogeneousPropagation`` through the medium before the surface.

        The refractive index of the medium gives the optical path, and its
        extinction coefficient k gives the absorption.
        """
        kernel, material = self.kernel, surface.material_pre
        propagation = material.propagation_model
        if type(propagation) is not self.types.propagation:
            return False
        index = self._index(material)
        if index is None:
            return False
        if index.per_ray:
            ints[kernel.OPD_INDEX_ROW] = self.values.row(index.value)
        else:
            floats[kernel.OPD_INDEX] = index.value
        absorption = self._absorption(propagation.material)
        if absorption is None:
            return False
        absorb_row, alpha_row, alpha = absorption
        ints[kernel.ABSORB_ROW], ints[kernel.ALPHA_ROW] = absorb_row, alpha_row
        floats[kernel.ALPHA] = alpha
        return True

    def _absorption(self, material: BaseMaterial) -> tuple[int, int, float] | None:
        """Return the absorption of a medium, as ABSORB_ROW, ALPHA_ROW, ALPHA.

        ``HomogeneousPropagation.propagate`` absorbs if any k > 0, with
        alpha = 4 * pi * k / w. An absorbing medium gets the next row of the
        exponents. alpha goes into ALPHA, or into a row of ``ray_values``
        (ALPHA_ROW) if it differs from ray to ray.

        Args:
            material: The material of the propagation model of the medium.

        Returns:
            The values of the columns ABSORB_ROW, ALPHA_ROW and ALPHA. For a
            medium that does not absorb, these are the defaults NO_ROW,
            NO_ROW and 0.0. None outside the domain.
        """
        kernel = self.kernel
        k = self._extinction(material)
        if k is None:
            return None
        if not _absorbs(k):
            return kernel.NO_ROW, kernel.NO_ROW, 0.0
        alpha = self._absorption_coefficient(material, k)
        if alpha is None:
            return None
        absorb_row = self.absorbing
        self.absorbing += 1
        if alpha.per_ray:
            return absorb_row, self.values.row(alpha.value), 0.0
        return absorb_row, kernel.NO_ROW, alpha.value

    def _interaction_columns(
        self, surface: Surface, ints: list[int], floats: list[float]
    ) -> bool:
        """``RefractiveReflectiveModel.interact_real_rays``.

        A refraction takes each index from ``boundary_index`` of the
        propagation model of its medium, which is ``material.n(w)`` for
        HomogeneousPropagation.
        """
        kernel, model = self.kernel, surface.interaction_model
        if (
            type(model) is not self.types.interaction
            or model.parent_surface is not surface
            or model.coating is not None
            or model.bsdf is not None
        ):
            return False
        if model.is_reflective:
            ints[kernel.REFLECT] = 1
            return True
        after = surface.material_post
        if type(after.propagation_model) is not self.types.propagation:
            return False
        n1, n2 = self._index(surface.material_pre), self._index(after)
        if n1 is None or n2 is None:
            return False
        if n1.per_ray or n2.per_ray:
            # u = n1 / n2 is an array. The kernel computes u and u * u for
            # each ray, as RealRays.refract computes u and u**2.
            if n1.per_ray:
                ints[kernel.N1_ROW] = self.values.row(n1.value)
            else:
                floats[kernel.N1] = n1.value
            if n2.per_ray:
                ints[kernel.N2_ROW] = self.values.row(n2.value)
            else:
                floats[kernel.N2] = n2.value
            return True
        ratio, ratio_squared = _index_ratio(n1, n2)
        floats[kernel.INDEX_RATIO] = ratio
        floats[kernel.INDEX_RATIO_SQUARED] = ratio_squared
        return True

    def _index(self, material: BaseMaterial) -> _Form | None:
        """Return the form of ``material.n(w)``, evaluated once per trace."""
        return _memo(self._indices, material, lambda m: self._form_of(m.n))

    def _extinction(self, material: BaseMaterial) -> _Form | None:
        """Return the form of ``material.k(w)``, evaluated once per trace."""
        return _memo(self._extinctions, material, lambda m: self._form_of(m.k))

    def _form_of(self, evaluate: Callable[[np.ndarray], object]) -> _Form | None:
        """Return the form of ``evaluate(w)``, or None outside the domain."""
        return _form(_material_value(evaluate, self.wavelengths), self.n)

    @functools.cached_property
    def _wavelength(self) -> _Form:
        """The form of ``rays.w``, computed at the first absorbing medium."""
        return _form(self.wavelengths, self.n)

    def _absorption_coefficient(self, material: BaseMaterial, k: _Form) -> _Form | None:
        """Return alpha = 4 * be.pi * k / rays.w of ``HomogeneousPropagation``.

        alpha is in 1/um. It is computed once per material and trace.
        """
        return _memo(self._alphas, material, lambda _: self._alpha(k))

    def _alpha(self, k: _Form) -> _Form | None:
        """Return alpha as a float, or as an array of one value per ray."""
        w = self._wavelength
        # be.* calls repeat those of the generic trace.
        if not (k.per_ray or w.per_ray):
            alpha = 4 * be.pi * k.value / w.value
            return _Form(False, alpha, alpha)
        alpha = 4 * be.pi * k.returned / self.wavelengths
        if type(alpha) is not np.ndarray or alpha.shape != (self.n,):
            return None
        return _Form(True, alpha, alpha)


def _ray_count(rays: RealRays) -> int | None:
    """Return the number of rays, or None outside the domain.

    The arrays of the ray state and ``rays.w`` must be 1-D float64
    ``np.ndarray`` (exact type) of one length of at least one ray.
    """
    count = None
    for name in (*_RAY_STATE, "w"):
        value = getattr(rays, name)
        if type(value) is not np.ndarray or value.ndim != 1:
            return None
        if value.dtype != np.float64:
            return None
        if count is None:
            count = value.shape[0]
        elif value.shape[0] != count:
            return None
    return count or None


def _build(
    surfaces: Sequence[Surface],
    rays: RealRays,
    n: int,
    kernel: ModuleType,
    final: FinalPropagation | None = None,
) -> _Plan | None:
    """Return the plan of a trace, or None outside the domain.

    Args:
        surfaces: The surfaces of the trace.
        rays: The rays of the trace.
        n: The number of rays.
        kernel: The kernel module.
        final: The propagation past the last surface, or None. The plan
            gets it as a final step if it is in the domain of that step
            (see ``_Builder.add_final_step``).

    Returns:
        The plan, or None if a surface is outside the domain.
    """
    builder = _Builder(rays, n, _types(), kernel)
    for surface in surfaces:
        if not builder.add(surface):
            return None
    if final is not None:
        builder.add_final_step(final)
    return builder.plan()


def _floor_applies(plan: _Plan, seed_stats: np.ndarray) -> bool:
    """Return True if the generic solve raises the tolerance of an asphere.

    The kernel iterates with the tolerance of the geometry, TOL.
    ``_effective_tolerance`` raises it to the round-off floor
    8 * eps * max(1, max|t|), with the largest abs(t) over the seeds of all
    rays of the call. The floor is above TOL only for a seed above
    TOL / (8 * eps): about 5.6e4 mm for TOL = 1e-10, and 563 mm for
    TOL = 1e-12. NumPy's max is NaN if one seed is NaN, and Python's
    ``max(1.0, nan)`` is 1.0, so the floor is then 8 * eps.

    Up to the first asphere where the floor applies, the kernel has the
    generic rays, so one True is enough.

    Args:
        plan: The plan of the trace.
        seed_stats: The seed summary that the kernel wrote, one row per
            surface (columns SEED_NAN and SEED_MAX).

    Returns:
        True if the floor is above TOL for an asphere of the plan. The
        generic trace must then run.
    """
    kernel = _kernel()
    for s in np.flatnonzero(plan.ints[:, kernel.NEWTON]):
        if seed_stats[s, kernel.SEED_NAN]:
            largest = math.nan
        else:
            largest = float(seed_stats[s, kernel.SEED_MAX])
        floor = _CONV_EPS_MULTIPLIER * _FLOAT64_EPS * max(1.0, largest)
        if floor > plan.floats[s, kernel.TOL]:
            return True
    return False


def _absorbed_intensity(
    plan: _Plan,
    intensity: np.ndarray,
    exponents: np.ndarray,
    clipped: np.ndarray,
    intensity_records: np.ndarray,
    record: bool,
    *,
    threads: int,
) -> np.ndarray:
    """Return the intensity after the last surface, with the absorption.

    The kernel writes the exponent of each ray in each absorbing medium and
    the clip mask of each aperture. This pass repeats ``RealRays.attenuate``
    and ``RealRays.clip`` in the order of the generic trace, from the
    intensity before the first surface, in parts of at most ``_PART`` rays.
    The absorption of a final step comes last. With more than one thread,
    the parts are smaller, so that each thread gets a part. ``numpy.exp``
    gives the same value for an element at any position in a contiguous
    float64 array, so a part of any size gets the values of the whole
    bundle.

    Args:
        plan: The plan of the trace.
        intensity: The intensity of the rays before the first surface.
        exponents: The exponents, one row per absorbing medium.
        clipped: The clip masks, one row per aperture.
        intensity_records: One row per surface. If ``record`` is True, row s
            receives the intensity after surface s.
        record: Whether to write ``intensity_records``.
        threads: The number of threads of the pass.

    Returns:
        The intensity after the last surface, and after the final step if
        the plan has one.
    """
    kernel = _kernel()
    rows = plan.ints[:, [kernel.ABSORB_ROW, kernel.CLIP_ROW]].tolist()
    # Only the surfaces have a row in intensity_records. The generic trace
    # records the rays at the last surface, before the final propagation.
    surface_count = len(rows)
    final_step = plan.final_step
    if final_step is not None and final_step.absorb_row != kernel.NO_ROW:
        rows.append([final_step.absorb_row, kernel.NO_ROW])
    n = intensity.shape[0]
    result = np.empty(n)
    part = min(_PART, _ceil_div(n, threads))
    starts = range(0, n, part)

    def run(k: int) -> None:
        start = starts[k]
        stop = min(start + part, n)
        i = intensity[start:stop]
        for s, (absorb_row, clip_row) in enumerate(rows):
            # be.* calls repeat those of the generic trace.
            if absorb_row != kernel.NO_ROW:
                # A ray without power stays dark, also when a later miss
                # gives a factor that is not finite.
                factor = be.exp(exponents[absorb_row, start:stop])
                i = i * be.where(i == 0, 0.0, factor)
            if clip_row != kernel.NO_ROW:
                i = be.where(clipped[clip_row, start:stop], be.zeros_like(i), i)
            if record and s < surface_count:
                intensity_records[s, start:stop] = i
        result[start:stop] = i

    _run_parts(len(starts), threads, run)
    return result


def _ceil_div(a: int, b: int) -> int:
    """Return the smallest integer that is not less than a / b."""
    return -(-a // b)


def _run_kernel(
    trace_range: Callable[[int, int, np.ndarray], None],
    n: int,
    seed_rows: int,
    *,
    threads: int,
) -> np.ndarray:
    """Trace the n rays of a plan with ``trace_range``, on ``threads`` threads.

    With one thread, one call traces all rays. With more threads, the rays
    are divided into parts of whole blocks: about ``_PARTS_PER_THREAD`` parts
    for each thread, of at least ``_MIN_PART_BLOCKS`` blocks. The kernel
    updates the seed summary in place, so each part has its own.

    Args:
        trace_range: ``trace_range(first, last, seed_stats)`` traces the
            rays first to last - 1 (see ``trace_plan``).
        n: The number of rays.
        seed_rows: The rows of the seed summary: one per surface for a plan
            with an even asphere, else 0.
        threads: The number of threads.

    Returns:
        The seed summary of all rays. With several parts, its NaN flag and
        its largest abs(t) are the maxima over the parts, which do not
        depend on the order of the parts.
    """
    kernel = _kernel()
    if threads == 1:
        seed_stats = np.zeros((seed_rows, kernel.SEED_COLUMNS))
        trace_range(0, n, seed_stats)
        return seed_stats
    blocks = _ceil_div(n, kernel._BLOCK)
    part_blocks = max(_MIN_PART_BLOCKS, _ceil_div(blocks, threads * _PARTS_PER_THREAD))
    # Whole blocks, so that each part starts at a block boundary.
    span = part_blocks * kernel._BLOCK
    parts = _ceil_div(n, span)
    part_seeds = np.zeros((parts, seed_rows, kernel.SEED_COLUMNS))

    def run(k: int) -> None:
        first = k * span
        trace_range(first, min(first + span, n), part_seeds[k])

    _run_parts(parts, threads, run)
    return part_seeds.max(axis=0)


class _Pool(NamedTuple):
    """The helper threads of the trace in one process.

    Attributes:
        pid: The process that made the threads. The threads of a parent
            process do not exist in a forked child, so the child makes its
            own pool.
        workers: The number of threads, ``be.get_trace_threads() - 1``: the
            calling thread also traces parts.
        executor: The thread pool.
    """

    pid: int
    workers: int
    executor: concurrent.futures.ThreadPoolExecutor


# The pool of this process, made at the first trace that needs it. The lock
# lets one thread make the pool when several threads need it at once.
_pool: _Pool | None = None
_pool_lock = threading.Lock()


def _executor() -> concurrent.futures.ThreadPoolExecutor:
    """Return the pool of helper threads of the trace.

    The pool has ``be.get_trace_threads() - 1`` threads. It is made at the
    first call, and again in a forked child or after a change of the number
    of threads. An old pool of this process finishes its queued parts and
    then stops.
    """
    global _pool
    workers = be.get_trace_threads() - 1
    with _pool_lock:
        pid = os.getpid()
        pool = _pool
        if pool is None or pool.pid != pid or pool.workers != workers:
            if pool is not None and pool.pid == pid:
                pool.executor.shutdown(wait=False)
            executor = concurrent.futures.ThreadPoolExecutor(
                workers, thread_name_prefix="optiland-trace"
            )
            pool = _pool = _Pool(pid, workers, executor)
        return pool.executor


def _run_parts(parts: int, threads: int, run: Callable[[int], None]) -> None:
    """Call ``run(k)`` once for each part k in ``range(parts)``, on threads.

    The calling thread and up to ``threads - 1`` helper threads of the pool
    take the next part from a shared counter until no part is left.
    ``next()`` on the counter, an ``itertools.count``, is atomic under the
    GIL, so each part goes to one thread. The pool has
    ``be.get_trace_threads() - 1`` threads, and traces in several Python
    threads share it. With one thread or one part, the calling thread runs
    all parts in order, and no pool is made.

    ``run`` runs in the pool threads without the context variables of the
    caller: there is no evaluation epoch and no host-value scope, and NumPy
    has its default floating-point error state (``numpy.errstate``). The
    kernel and the intensity pass need no epoch and no scope.

    Args:
        parts: The number of parts.
        threads: The most threads that run parts at the same time, with the
            calling thread. At most ``be.get_trace_threads()`` are used.
        run: Runs part k. It must write only the entries of part k.

    Raises:
        Exception: The exception of a part, raised when all threads have
            stopped. If a part in the calling thread raised, it is that
            exception. Otherwise, it is the exception of the first helper,
            in the order of submission, whose part raised.
    """
    helpers = min(threads, parts, be.get_trace_threads()) - 1
    if helpers < 1:
        for k in range(parts):
            run(k)
        return
    counter = itertools.count()

    def take_parts() -> None:
        for k in counter:
            if k >= parts:
                return
            run(k)

    executor = _executor()
    futures = [executor.submit(take_parts) for _ in range(helpers)]
    try:
        take_parts()
    finally:
        # Also after an exception here: no helper may write into the outputs
        # after this function ends.
        concurrent.futures.wait(futures)
    for future in futures:
        future.result()  # raises the exception of the helper, if any


def _release_records(surfaces: Sequence[Surface]) -> None:
    """Drop the records of the previous trace before the kernel runs.

    ``Surface.reset`` does this before the generic trace. Without it, a large
    trace would hold two sets of records at its peak.
    """
    for surface in surfaces:
        for name in _CLEARED:
            setattr(surface, name, _RELEASED)


def _store_records(
    surfaces: Sequence[Surface], records: dict[str, np.ndarray], record: bool
) -> None:
    """Give each surface the records of the generic trace.

    Args:
        surfaces: The surfaces of the trace.
        records: The records of the kernel by name, one row per surface.
        record: Whether the kernel wrote the records.
    """
    if not record:
        for surface in surfaces:
            surface.reset()
        return
    # be.* calls repeat those of the generic trace. Surface.reset gives u and
    # aoi an empty array, and a real trace does not set them.
    blank = be.empty(0)
    for s, surface in enumerate(surfaces):
        for name, rows in records.items():
            setattr(surface, name, rows[s])
        surface.u = surface.aoi = blank


def trace_with_plan(
    surfaces: Sequence[Surface],
    rays: BaseRays,
    record: bool,
    final: FinalPropagation | None = None,
) -> bool:
    """Trace ``rays`` through ``surfaces`` with the compiled kernel, if possible.

    With ``be.set_trace_threads(n)`` and n > 1, a bundle of
    ``_MIN_THREADED_RAYS`` rays or more is traced on n threads.

    Args:
        surfaces: The surfaces to trace, in order.
        rays: The rays to trace. Other ray types than ``RealRays`` get False.
        record: Whether each surface stores the ray state after it.
        final: The propagation past the last surface, or None. If it is in
            the domain of the final step, the plan does it after the last
            surface and sets ``final.applied``. Otherwise the caller
            propagates the rays after the plan (``FinalPropagation.finish``).

    Returns:
        True if ``rays`` and the records of ``surfaces`` now hold the result
        of the generic trace, and of the final propagation if
        ``final.applied`` is True. False if the trace is outside the domain
        of the plan: ``rays`` is then unchanged, and so is ``final``. The
        records are unchanged too, except when the tolerance floor of an
        even asphere applies (see ``_floor_applies``): then the kernel has
        run, and the records of the previous trace are released. After
        False, the caller resets the records.
    """
    if not _ENABLED or not surfaces or be.get_backend() != "numpy":
        return False
    # The generic trace converts the translations and the rotation angles
    # with be.array to the backend precision. Without the JIT compiler, the
    # kernel would run as Python code.
    if be.get_precision() != 64 or numba.config.DISABLE_JIT:
        return False
    if type(rays) is not _types().rays or not rays.is_normalized:
        return False
    n = _ray_count(rays)
    if n is None:
        return False
    kernel = _kernel()
    plan = _build(surfaces, rays, n, kernel, final)
    if plan is None:
        return False

    _release_records(surfaces)
    outputs = {name: np.empty(n) for name in _OUTPUTS}
    shape = (len(surfaces), n)
    records = {name: np.empty(shape) if record else _NO_ROWS for name in _RECORDED}
    exponents = np.empty((plan.absorbing, n)) if plan.absorbing else _NO_ROWS
    clipped = np.empty((plan.clips, n), dtype=np.bool_) if plan.clips else _NO_CLIPS
    rays_in = tuple(getattr(rays, name) for name in _RAY_STATE)
    rays_out, record_rows = tuple(outputs.values()), tuple(records.values())
    final_step = plan.final_step

    # The kernel call for the rays first to last - 1. Two kernels compile
    # from one loop nest. trace_basic_plan has no code for per-ray values and
    # even aspheres (see the kernel module).
    def trace_part(first: int, last: int, seed_stats: np.ndarray) -> None:
        if plan.basic:
            kernel.trace_basic_plan(
                first,
                last,
                rays_in,
                plan.ints,
                plan.floats,
                exponents,
                clipped,
                rays_out,
                record_rows,
                record,
            )
        else:
            kernel.trace_plan(
                first,
                last,
                rays_in,
                plan.ints,
                plan.floats,
                plan.ray_values,
                plan.coef,
                plan.dcoef,
                exponents,
                clipped,
                seed_stats,
                rays_out,
                record_rows,
                record,
            )

    # _run_kernel calls trace_range once for all rays, or once for each part
    # of a trace on threads. With a final step, trace_range traces its rays
    # in parts of _FINAL_PART rays. The final step moves each part past the
    # last surface right after the kernel, while its rays are in the cache.
    def trace_range(first: int, last: int, seed_stats: np.ndarray) -> None:
        if final_step is None:
            trace_part(first, last, seed_stats)
            return
        for start in range(first, last, _FINAL_PART):
            stop = min(start + _FINAL_PART, last)
            trace_part(start, stop, seed_stats)
            kernel._propagate_past_last_surface(
                start,
                stop,
                rays_out[:3],
                rays_out[3:6],
                final_step.t,
                final_step.absorb_row,
                final_step.alpha_row,
                final_step.alpha,
                plan.ray_values,
                exponents,
            )

    # The seed summary has one row per surface for a plan with an asphere.
    seed_rows = len(surfaces) if plan.aspheres else 0
    threads = be.get_trace_threads() if n >= _MIN_THREADED_RAYS else 1
    seed_stats = _run_kernel(trace_range, n, seed_rows, threads=threads)
    # The kernel iterated with the tolerance of each asphere. If the generic
    # solve raises a tolerance, the generic trace runs instead.
    if plan.aspheres and _floor_applies(plan, seed_stats):
        return False
    if plan.absorbing:
        intensity_threads = (
            min(threads, _MAX_INTENSITY_THREADS)
            if n >= _MIN_THREADED_INTENSITY_RAYS
            else 1
        )
        outputs["i"] = _absorbed_intensity(
            plan,
            rays.i,
            exponents,
            clipped,
            records["intensity"],
            record,
            threads=intensity_threads,
        )
    elif not plan.clips:
        # Nothing changed the intensity, so the generic trace keeps its array.
        del outputs["i"]
    if plan.moves_rays:
        for name, value in outputs.items():
            setattr(rays, name, value)
    _store_records(surfaces, records, record)
    if final_step is not None:
        final.applied = True
    return True
