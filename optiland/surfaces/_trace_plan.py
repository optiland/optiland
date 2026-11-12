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
aperture radii, u = n1/n2 and u**2, the absorption coefficient) with
the same IEEE 754 operations in the same order, so the kernel gets the same
values. Python floats give the bits of NumPy float64 for + - * /. NumPy
computes ``x**2`` as ``x * x`` only for an array, and calls the C function
``pow`` for a scalar (see ``_index_ratio``).

The developer page "Performance Internals" lists the domain of the plan,
with the reason for each limit. Outside the domain, ``trace_with_plan``
returns False and the generic trace runs.

Manuel Fragata Mendes, 2026
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, TypeVar

import numba
import numpy as np

import optiland.backend as be
from optiland.surfaces._chunked_trace import _RECORDED

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from types import ModuleType

    from optiland.materials.base import BaseMaterial
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
# exponents of a plan without absorption and the clip masks of a plan without
# apertures. The kernel never writes into them.
_NO_ROWS = np.empty((0, 1))
_NO_CLIPS = np.empty((0, 1), dtype=np.bool_)

# Rays per part of the intensity pass. Each NumPy operation of the pass makes
# a new array. _chunked_trace uses parts of the same size (_CHUNK_SIZE) for
# the same reason: an array of 65536 float64 values (512 KiB) stays below the
# size from which the heap of 64-bit Windows gives memory back to the system.
_PART = 65536


class _Types(NamedTuple):
    """The exact types of the domain of the plan."""

    rays: type
    object_surface: type
    surfaces: tuple[type, ...]
    coordinate_system: type
    plane: type
    standard: type
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
    from optiland.geometries import Plane, StandardGeometry
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


@dataclass(frozen=True)
class _Plan:
    """The tables of one trace, and the facts about them that the trace needs.

    Attributes:
        ints: The integer table, one row per surface (see the kernel).
        floats: The float table, one row per surface.
        moves_rays: False if all surfaces are object surfaces. The rays then
            keep their arrays.
        absorbing: The number of absorbing media: the rows of the exponents.
        clips: The number of apertures: the rows of the clip masks.
    """

    ints: np.ndarray
    floats: np.ndarray
    moves_rays: bool
    absorbing: int
    clips: int


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
        self._indices: dict[int, tuple[object, _Form | None]] = {}
        self._extinctions: dict[int, tuple[object, _Form | None]] = {}
        # Rows with the defaults of the layout table. Python lists are faster
        # to fill than arrays; plan() converts them once.
        self._empty_ints = [0] * kernel.INT_COLUMNS
        self._empty_ints[kernel.KIND] = kernel.OBJECT
        self._empty_ints[kernel.ABSORB_ROW] = kernel.NO_ROW
        self._empty_ints[kernel.APERTURE] = kernel.NO_APERTURE
        self._empty_ints[kernel.CLIP_ROW] = kernel.NO_ROW
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

    def plan(self) -> _Plan:
        """Return the plan of the rows added so far."""
        return _Plan(
            ints=np.array(self.int_rows, dtype=np.int64),
            floats=np.array(self.float_rows, dtype=np.float64),
            moves_rays=self.moves_rays,
            absorbing=self.absorbing,
            clips=self.clips,
        )

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
        """The step kind, and the values of the distance and the normal."""
        kernel, geometry = self.kernel, surface.geometry
        if type(geometry) is self.types.plane:
            ints[kernel.KIND] = kernel.PLANE
            return True
        if type(geometry) is not self.types.standard:
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
        k = self._extinction(propagation.material)
        if index is None or index.per_ray or k is None:
            return False
        floats[kernel.OPD_INDEX] = index.value
        if _absorbs(k):
            alpha = self._absorption_coefficient(k)
            if alpha is None:
                return False
            ints[kernel.ABSORB_ROW] = self.absorbing
            floats[kernel.ALPHA] = alpha
            self.absorbing += 1
        return True

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
        if n1 is None or n2 is None or n1.per_ray or n2.per_ray:
            return False
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

    def _absorption_coefficient(self, k: _Form) -> float | None:
        """Return alpha = 4 * be.pi * k / rays.w of ``HomogeneousPropagation``.

        alpha is in 1/um. Returns None if alpha differs from ray to ray.
        """
        w = self._wavelength
        if k.per_ray or w.per_ray:
            return None
        # be.* calls repeat those of the generic trace.
        return 4 * be.pi * k.value / w.value


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
    surfaces: Sequence[Surface], rays: RealRays, n: int, kernel: ModuleType
) -> _Plan | None:
    """Return the plan of a trace, or None outside the domain."""
    builder = _Builder(rays, n, _types(), kernel)
    for surface in surfaces:
        if not builder.add(surface):
            return None
    return builder.plan()


def _absorbed_intensity(
    plan: _Plan,
    intensity: np.ndarray,
    exponents: np.ndarray,
    clipped: np.ndarray,
    intensity_records: np.ndarray,
    record: bool,
) -> np.ndarray:
    """Return the intensity after the last surface, with the absorption.

    The kernel writes the exponent of each ray in each absorbing medium and
    the clip mask of each aperture. This pass repeats ``RealRays.attenuate``
    and ``RealRays.clip`` in the order of the generic trace, from the
    intensity before the first surface, in parts of ``_PART`` rays.
    ``numpy.exp`` gives the same value for an element at any position in a
    contiguous float64 array, so a part gets the values of the whole bundle.

    Args:
        plan: The plan of the trace.
        intensity: The intensity of the rays before the first surface.
        exponents: The exponents, one row per absorbing medium.
        clipped: The clip masks, one row per aperture.
        intensity_records: One row per surface. If ``record`` is True, row s
            receives the intensity after surface s.
        record: Whether to write ``intensity_records``.

    Returns:
        The intensity after the last surface.
    """
    kernel = _kernel()
    rows = plan.ints[:, [kernel.ABSORB_ROW, kernel.CLIP_ROW]].tolist()
    n = intensity.shape[0]
    result = np.empty(n)
    for start in range(0, n, _PART):
        stop = min(start + _PART, n)
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
            if record:
                intensity_records[s, start:stop] = i
        result[start:stop] = i
    return result


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


def trace_with_plan(surfaces: Sequence[Surface], rays: BaseRays, record: bool) -> bool:
    """Trace ``rays`` through ``surfaces`` with the compiled kernel, if possible.

    Args:
        surfaces: The surfaces to trace, in order.
        rays: The rays to trace. Other ray types than ``RealRays`` get False.
        record: Whether each surface stores the ray state after it.

    Returns:
        True if ``rays`` and the records of ``surfaces`` now hold the result
        of the generic trace. False if the trace is outside the domain of the
        plan: ``rays`` and the records are then unchanged.
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
    plan = _build(surfaces, rays, n, kernel)
    if plan is None:
        return False

    _release_records(surfaces)
    outputs = {name: np.empty(n) for name in _OUTPUTS}
    shape = (len(surfaces), n)
    records = {name: np.empty(shape) if record else _NO_ROWS for name in _RECORDED}
    exponents = np.empty((plan.absorbing, n)) if plan.absorbing else _NO_ROWS
    clipped = np.empty((plan.clips, n), dtype=np.bool_) if plan.clips else _NO_CLIPS
    kernel.trace_plan(
        tuple(getattr(rays, name) for name in _RAY_STATE),
        plan.ints,
        plan.floats,
        exponents,
        clipped,
        tuple(outputs.values()),
        tuple(records.values()),
        record,
    )
    if plan.absorbing:
        outputs["i"] = _absorbed_intensity(
            plan, rays.i, exponents, clipped, records["intensity"], record
        )
    elif not plan.clips:
        # Nothing changed the intensity, so the generic trace keeps its array.
        del outputs["i"]
    if plan.moves_rays:
        for name, value in outputs.items():
            setattr(rays, name, value)
    _store_records(surfaces, records, record)
    return True
