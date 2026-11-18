"""Trace a large NumPy bundle of real rays in parts.

Each step of a trace makes new arrays of the bundle size. On 64-bit Windows,
the heap gets blocks of about 1 MB or more from the operating system
(VirtualAlloc) and gives them back when they are freed. Each new array of
that size then pays a page fault for each page. 131072 rays (1 MiB per
float64 array) is the smallest power of two above that limit. Parts of
65536 rays (512 KiB) stay below it. So, on Windows, ``trace_in_parts``
traces a bundle of 131072 rays or more through all surfaces in parts of
65536 rays.

On Linux, the parts cost time instead. glibc gives the memory of each part
back to the system after the part, so each new part pays the page faults.
With the scenarios of the performance gate, a trace in parts took 1.4 to 1.6
times the time of the whole bundle, and with that memory kept, it was not
faster than the whole bundle. So the parts run only on Windows
(``_ENABLED``). macOS was not measured; it traces the whole bundle.

The reference is the loop over the surfaces in
``SurfaceGroup._trace_surfaces``. It runs for other backends and whenever
``trace_in_parts`` returns False. The parts must give the bits of that loop
on the whole bundle, so they are traced only in a domain where each ray is
traced independently of the other rays. ``_can_trace_in_parts`` checks it:

- the NumPy backend, and rays of exactly the type ``RealRays`` whose arrays
  are 1-D float64 arrays of one length;
- surfaces, geometries, interaction models, coatings, apertures and
  propagation models of the exact types of ``_allowed_types``, and no BSDF.
  A subclass could add a step that uses the whole bundle, so subclasses are
  not in the domain.

Each type in this list traces each ray independently, or reports each
decision that uses all rays through ``optiland._trace_probe``. A step can
take one decision for all rays of a call only if both outcomes give each ray
the same bits. The early exit and the plane seed of the Newton-Raphson
solver are examples. Two decisions can change the bits, and they report
through ``optiland._trace_probe``: the tolerance floor, which depends on the
largest ``|t|`` of the start values, and absorption, which applies only if
a ray has k > 0. A part that uses the floor asks for the whole bundle. If
the parts take different absorption decisions, the whole bundle could take
another one. In both cases the whole bundle is traced.

The types of the materials are not checked, only their propagation models.
Each material must compute the value of a ray from that ray alone, as the
built-in ones do. For example, a part can have one wavelength while the
whole bundle has several. The part is then evaluated at one element and
broadcast, and the whole bundle element by element. Coatings must be of a
built-in type, and each built-in coating computes the value of a ray from
that ray alone.

A part holds read-only views of the input arrays, so a step that writes into
an input array in place raises. Each exception in a part ends the trace in
parts, and so does a part whose arrays differ in name, length or dtype from
those of the first part. Nothing is written to the rays or to the surfaces
until all parts are traced. The bundle is then traced as a whole, so the
caller gets the result or the error of the generic trace.

Manuel Fragata Mendes, 2026
"""

from __future__ import annotations

import copy
import functools
import sys
from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland import _trace_probe

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterator, Sequence

    from optiland.rays import BaseRays, RealRays
    from optiland.surfaces.standard_surface import Surface

# Rays per part, and the smallest bundle that is traced in parts. A bundle of
# 2 * _CHUNK_SIZE = 131072 rays (1 MiB per float64 array) is the smallest
# power of two above the heap limit of the module docstring. A smaller bundle
# gets its arrays from the heap without page faults, so parts give no gain.
# The tests set small values to trace small bundles in many parts.
_CHUNK_SIZE = 65536
_MIN_RAYS = 2 * _CHUNK_SIZE

# Traces in parts run only on Windows (see the module docstring). The tests
# set this to True, so that they test the parts on every platform.
_ENABLED = sys.platform == "win32"

# The arrays that Surface._record_real stores on each surface. A name that is
# missing here would keep the record of the last part only, so a test checks
# that this list is complete.
_RECORDED = ("x", "y", "z", "L", "M", "N", "intensity", "opd")


@functools.cache
def _allowed_types() -> dict[str, tuple[type, ...]]:
    """Return the exact types for which each ray is traced independently.

    The imports are local because ``SurfaceGroup`` imports this module while
    ``optiland.surfaces`` is still being imported. Local imports keep this
    module free of constraints on that import order.

    Returns:
        The allowed types of the rays, surfaces, geometries, interaction
        models, coatings, apertures and propagation models, by kind.
    """
    from optiland.coatings import (
        CartesianJonesCoating,
        FresnelCoating,
        PolarizerCoating,
        RetarderCoating,
        SimpleCoating,
        ThinFilmCoating,
    )
    from optiland.geometries import (
        BiconicGeometry,
        EvenAsphere,
        OddAsphere,
        Plane,
        PolynomialGeometry,
        StandardGeometry,
        ToroidalGeometry,
    )
    from optiland.interactions.refractive_reflective_model import (
        RefractiveReflectiveModel,
    )
    from optiland.interactions.thin_lens_interaction_model import (
        ThinLensInteractionModel,
    )
    from optiland.physical_apertures import (
        EllipticalAperture,
        OffsetRadialAperture,
        RadialAperture,
        RectangularAperture,
    )
    from optiland.propagation.homogeneous import HomogeneousPropagation
    from optiland.rays.real_rays import RealRays
    from optiland.surfaces.image_surface import ImageSurface
    from optiland.surfaces.object_surface import ObjectSurface
    from optiland.surfaces.standard_surface import Surface

    return {
        "rays": (RealRays,),
        "surface": (Surface, ObjectSurface, ImageSurface),
        "geometry": (
            StandardGeometry,
            Plane,
            EvenAsphere,
            OddAsphere,
            PolynomialGeometry,
            BiconicGeometry,
            ToroidalGeometry,
        ),
        "interaction": (RefractiveReflectiveModel, ThinLensInteractionModel),
        "coating": (
            type(None),
            SimpleCoating,
            CartesianJonesCoating,
            FresnelCoating,
            PolarizerCoating,
            RetarderCoating,
            ThinFilmCoating,
        ),
        "aperture": (
            type(None),
            RadialAperture,
            OffsetRadialAperture,
            EllipticalAperture,
            RectangularAperture,
        ),
        "propagation": (HomogeneousPropagation,),
    }


def _ray_arrays(rays: BaseRays) -> dict[str, np.ndarray]:
    """Return the NumPy arrays of ``rays`` by attribute name.

    Args:
        rays: The rays.

    Returns:
        Each attribute whose type is exactly ``np.ndarray``.
    """
    return {
        name: value for name, value in vars(rays).items() if type(value) is np.ndarray
    }


def _can_trace_in_parts(surfaces: Sequence[Surface], rays: BaseRays) -> bool:
    """Return True if a trace in parts gives the bits of the whole bundle.

    The conditions, with their reasons:

    - The backend is NumPy. The heap limit concerns NumPy arrays, and with
      Torch the parts would split the autograd graph.
    - ``rays`` is exactly a ``RealRays``. Another ray type, for example
      ``PolarizedRays``, has other arrays and other steps.
    - Each array of ``rays`` is a 1-D float64 ``np.ndarray`` (exact type, no
      subclass), and all arrays have one length, the number of rays. Only
      then does one slice select the same rays from each array. The
      bit-identity argument is made for float64 (see the developer page on
      performance internals), and the part size is chosen for 8-byte
      values.
    - There are at least ``_MIN_RAYS`` rays.
    - Each surface, geometry, interaction model, coating and aperture, and
      the propagation model of each medium, has an exact type of
      ``_allowed_types``, and there is no BSDF.

    Args:
        surfaces: The surfaces to trace, in order.
        rays: The rays to trace.

    Returns:
        True if the bundle and the surfaces are in the domain.
    """
    if be.get_backend() != "numpy":
        return False
    allowed = _allowed_types()
    if type(rays) not in allowed["rays"]:
        return False
    # Most bundles are small, and this is the cheapest check. A short x
    # array also fails the checks below.
    if len(rays.x) < _MIN_RAYS:
        return False
    lengths = set()
    for value in vars(rays).values():
        if not isinstance(value, np.ndarray):
            continue
        # An array subclass would not be split into parts (see _ray_arrays).
        if type(value) is not np.ndarray or value.ndim != 1:
            return False
        if value.dtype != np.float64:
            return False
        lengths.add(value.shape[0])
    if len(lengths) != 1 or lengths.pop() < _MIN_RAYS:
        return False
    for surface in surfaces:
        model = surface.interaction_model
        types = (
            ("surface", surface),
            ("geometry", surface.geometry),
            ("interaction", model),
            ("coating", model.coating),
            ("aperture", surface.aperture),
            ("propagation", surface.material_pre.propagation_model),
            ("propagation", surface.material_post.propagation_model),
        )
        if any(type(value) not in allowed[kind] for kind, value in types):
            return False
        if model.bsdf is not None:
            return False
    return True


def _part(rays: RealRays, start: int, stop: int) -> RealRays:
    """Return the part of ``rays`` from ray ``start`` to ray ``stop``.

    The part is a shallow copy of ``rays`` whose arrays are read-only views
    of that range. A step that writes into an input array in place raises,
    and the whole bundle is then traced from the unchanged input.

    Args:
        rays: The whole bundle.
        start: The index of the first ray of the part.
        stop: The index after the last ray of the part.

    Returns:
        The part.
    """
    part = copy.copy(rays)
    for name, value in _ray_arrays(rays).items():
        view = value[start:stop]
        view.flags.writeable = False
        setattr(part, name, view)
    return part


def _trace_part(
    surfaces: Sequence[Surface], part: RealRays, record: bool
) -> list[tuple[object, Hashable]]:
    """Trace one part through all surfaces.

    Args:
        surfaces: The surfaces to trace, in order.
        part: The part. It is traced in place.
        record: Whether each surface stores the ray state of the part.

    Returns:
        The reports of the steps (see ``optiland._trace_probe``).
    """
    with _trace_probe.recording() as reports:
        for surface in surfaces:
            surface.trace(part, record=record)
    return reports


def _part_arrays(
    surfaces: Sequence[Surface], part: RealRays, record: bool
) -> Iterator[tuple[Hashable, object]]:
    """Yield the arrays of a traced part with their keys in the whole bundle.

    Args:
        surfaces: The surfaces that traced the part, in order.
        part: The traced part.
        record: Whether the surfaces stored the ray state of the part.

    Yields:
        ``(name, array)`` for each array of the part, then, with records,
        ``((surface index, name), array)`` for each recorded array.
    """
    yield from _ray_arrays(part).items()
    if record:
        for index, surface in enumerate(surfaces):
            for name in _RECORDED:
                yield (index, name), getattr(surface, name)


def _store(
    whole: dict[Hashable, np.ndarray],
    key: Hashable,
    value: object,
    start: int,
    stop: int,
    ray_count: int,
) -> None:
    """Copy the array of one part into its range of a whole-bundle array.

    The whole-bundle array is made at the first part, with the dtype of the
    array of that part.

    Args:
        whole: The whole-bundle arrays, by key. Changed in place.
        key: The key of the array.
        value: The array of the part.
        start: The index of the first ray of the part.
        stop: The index after the last ray of the part.
        ray_count: The number of rays of the whole bundle.

    Raises:
        ValueError: If ``value`` is not a 1-D ``np.ndarray`` with one value
            for each ray of the part, or if its dtype is not the dtype of
            the first part.
    """
    if type(value) is not np.ndarray or value.shape != (stop - start,):
        raise ValueError(f"{key}: not an array of the rays of the part")
    out = whole.get(key)
    if out is None:
        out = whole[key] = np.empty(ray_count, dtype=value.dtype)
    elif out.dtype != value.dtype:
        raise ValueError(f"{key}: the dtype is not the same in all parts")
    out[start:stop] = value


def _write_back(
    surfaces: Sequence[Surface],
    rays: RealRays,
    last_part: RealRays,
    whole: dict[Hashable, np.ndarray],
    record: bool,
) -> None:
    """Write the whole-bundle arrays to ``rays`` and to the surfaces.

    The attributes of ``rays`` that are not arrays, such as
    ``is_normalized``, get the values of the last part. In the domain, they
    are the same in each part.

    Args:
        surfaces: The surfaces that traced the parts, in order.
        rays: The whole bundle. Changed in place.
        last_part: The last traced part.
        whole: The whole-bundle arrays, by key.
        record: Whether the surfaces stored the ray state.
    """
    for name, value in vars(last_part).items():
        setattr(rays, name, whole[name] if type(value) is np.ndarray else value)
    if record:
        for index, surface in enumerate(surfaces):
            for name in _RECORDED:
                setattr(surface, name, whole[(index, name)])


def trace_in_parts(surfaces: Sequence[Surface], rays: BaseRays, record: bool) -> bool:
    """Trace ``rays`` through ``surfaces`` in parts, if the result is exact.

    Args:
        surfaces: The surfaces to trace, in order, with reset records.
        rays: The rays to trace. They are changed only if True is returned.
        record: Whether each surface stores the ray state.

    Returns:
        True if ``rays`` and the surface records now hold the traced result.
        False if the caller must trace the whole bundle: the platform is not
        Windows, the bundle is not in the domain, a decision for the whole
        bundle could differ from the decisions of the parts, or a part
        raised. ``rays`` is then unchanged.
    """
    if not (_ENABLED and _can_trace_in_parts(surfaces, rays)):
        return False
    ray_count = rays.x.shape[0]
    # Each part goes into the whole-bundle arrays at once, so only the arrays
    # of one part are alive at a time. If all parts stayed alive until a join
    # at the end, the heap would grow by the bundle size in blocks of one
    # part. On Windows, later traces of other bundles then get much of their
    # memory from that space, with a page fault for each page, and are up to
    # 25% slower.
    whole: dict[Hashable, np.ndarray] = {}
    first_part = None
    try:
        for start in range(0, ray_count, _CHUNK_SIZE):
            stop = min(start + _CHUNK_SIZE, ray_count)
            part = _part(rays, start, stop)
            reports = _trace_part(surfaces, part, record)
            if _trace_probe.asks_for_whole_bundle(reports):
                return False
            # Each part must take the decisions of the first part and have
            # the arrays of the first part.
            this_part = (reports, frozenset(_ray_arrays(part)))
            if first_part is None:
                first_part = this_part
            elif this_part != first_part:
                return False
            for key, value in _part_arrays(surfaces, part, record):
                _store(whole, key, value, start, stop, ray_count)
    except Exception:  # the whole-bundle trace gives the generic result
        return False
    # All parts are traced. Only now write to rays and the surfaces, so each
    # return False above leaves rays unchanged.
    _write_back(surfaces, rays, last_part=part, whole=whole, record=record)
    return True
