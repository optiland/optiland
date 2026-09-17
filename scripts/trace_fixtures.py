"""Shared system and bundle fixtures for the fused-trace tests and harnesses.

WP5 (plan 4/WP5) fills the WP0 stub: one builder per branch of the fused
kernel, ``REFUSAL_FIXTURES`` (one case per ``FusedTraceSkip`` value),
``KNOWN_INELIGIBLE`` (measured, see :func:`audit_catalog`), ``CATALOG`` and
``register``.

It lives under ``scripts/`` rather than ``tests/`` on purpose: ``tests`` is
excluded from ruff, and this module is imported by tests *and* by
``scripts/metal_oracle_e2e.py`` / ``metal_suite.py`` / ``metal_benchmark.py``,
so it has to stay lint-clean.

Import it with ``scripts/`` on ``sys.path``::

    import sys; sys.path.insert(0, "scripts")
    import trace_fixtures

Conventions (frozen here so every consumer reads the fixtures the same way):

* A **system fixture** is a zero-argument function returning either a fresh
  ``Optic`` or a ``(optic, rays_factory)`` pair.  Nothing is cached: every
  call builds new objects, so a test may mutate what it gets.
* A **rays factory** has the signature ``f(optic, num_rays=DEFAULT_RAYS)`` and
  returns a ``RealRays`` bundle in the *global* frame, ready to hand to
  ``optic.surfaces.trace(rays)``.  A fixture that returns a bare ``Optic``
  is meant to be launched with :func:`pupil_bundle`.
* Every bundle has ``num_rays > HOST_THRESHOLD`` (256) by default, because a
  smaller bundle is host-resident and is refused *structurally* (plan 1.2) —
  a fixture that silently dropped under the threshold would test nothing.
* Each fixture's docstring names the kernel branch it exercises and the
  status bits the trace must set, with the counts measured on the numpy
  reference path (``python scripts/trace_fixtures.py --audit``).  A count is
  a prediction, not an observation to be rationalised afterwards
  (plan 0.2.6).

Status-bit vocabulary (``trace_layout.ST_*``) and where the per-op path
produces each one, so that "the kernel must set bit X" is checkable against
the mirrored Python source rather than against the kernel:

===================== =========================================================
``ST_MISS``           no admissible conic root: ``_conic_intersection_distance``
                      returns NaN (``geometries/standard.py``), so x/y/z go NaN
``ST_TIR``            ``1 - u^2 (1 - dot^2) < 0`` in ``RealRays.refract``:
                      ``sqrt`` of a negative number, so L/M/N go NaN
``ST_CLIPPED``        ``aperture.clip`` zeroes the intensity of rays outside
                      ``aperture.contains`` (inclusive bounds)
``ST_NEWTON_NOT_CONVERGED``
                      ``|F(t)| >= tol`` after ``max_iter`` Newton steps
                      (``newton_raphson.py::_solve_distance_primal``)
``ST_TOL_CROSSOVER``  ``|t_seed|`` past the df64/sf64 crossover (3.5e3 mm /
                      1.1e5 mm): the driver's late fallback, not a Python site
``ST_NZ_FLOORED``     ``be.where(be.abs(rays.N) > 1e-14, rays.N, 1e-14)`` in the
                      infinite-radius branch of ``_conic_intersection_distance``,
                      and ``_sign_preserving_floor(nz, tau_nz)`` in
                      ``newton_raphson.py::_surface_residual_dt``
``ST_DF_FLOORED``     ``_regularize_signed(df_dt, scale)`` in
                      ``newton_raphson.py::_solve_distance_primal``
``ST_NONUNIFORM_W``   pre-launch refusal (``mixed_wavelength``), never a bit on
                      a completed trace
===================== =========================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

import optiland.backend as be
import optiland.samples as _samples
from optiland.backend.torch_backend.metal.trace_adapters import (
    APERTURE_ADAPTERS,
    GEOMETRY_ADAPTERS,
    INTERACTION_ADAPTERS,
    FusedTraceSkip,
)
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import EvenAsphere, OddAsphere, Plane, StandardGeometry
from optiland.interactions.refractive_reflective_model import RefractiveReflectiveModel
from optiland.optic import Optic
from optiland.physical_apertures.elliptical import EllipticalAperture
from optiland.physical_apertures.offset_radial import OffsetRadialAperture
from optiland.physical_apertures.radial import RadialAperture
from optiland.physical_apertures.rectangular import RectangularAperture
from optiland.rays import RealRays
from optiland.samples import objectives as _objectives

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

__all__ = [
    "CATALOG",
    "DEFAULT_RAYS",
    "FIXTURES",
    "FIXTURES_BY_CLASS",
    "KNOWN_INELIGIBLE",
    "MAX_FUSED_RAYS",
    "REFUSAL_FIXTURES",
    "TIER_B_RAYS",
    "TILTED_TRIPLET_POSES",
    "UNEXPORTED_OBJECTIVES",
    "Patch",
    "RefusalCase",
    "apodized_bundle",
    "aspheric_singlet",
    "audit_catalog",
    "backward_newton",
    "backward_plane",
    "build",
    "collimated_bundle",
    "cooke",
    "denormal_aperture_params",
    "ellipse_aperture",
    "even_asphere_5coeff",
    "even_asphere_inf_radius",
    "exact_grazing_bundle",
    "fold_mirror",
    "grazing_bundle",
    "hubble",
    "long_path_asphere",
    "long_path_batch_values",
    "make_rays",
    "miss_bundle",
    "mixed_wavelength_bundle",
    "nonconverging_asphere",
    "nonzero_image_thickness",
    "odd_asphere_singlet",
    "off_axis_parabola_far_root",
    "offset_radial_aperture",
    "planes_both_kinds",
    "pupil_bundle",
    "rect_aperture",
    "register",
    "reverse_bundle",
    "rim_bundle",
    "tilted_fold_mirror",
    "tilted_triplet",
    "tir_singlet",
    "uv_projection",
    "zero_rmax_aperture",
]

#: Default bundle size.  Above ``HOST_THRESHOLD`` (256) so every bundle is
#: GPU-resident, and above 1024 so every comparison is tier A (plan 7.1).
DEFAULT_RAYS = 4096

#: The tier-B sampling of plan 7.1 (site 1): small, still GPU-resident.
TIER_B_RAYS = 300

#: The five poses of the decentred/tilted triplet (design 8.2).
TILTED_TRIPLET_POSES: tuple[str, ...] = ("rx", "ry", "rz", "dxdy", "rxryrz")

#: Sample classes that ``optiland.samples.__all__`` omits but that the
#: conformance sweep must still cover (design 8.5).
UNEXPORTED_OBJECTIVES: tuple[str, ...] = (
    "WideAngle100FOV",
    "ProjectionLens120FOV",
    "ProjectionLens160FOV",
    "WideAngle170FOV",
)

#: The gate's ray ceiling (design 2.2 step 8): ``n > 2**30`` is refused with
#: ``too_many_rays``.  Mirrored here so the refusal fixture does not have to
#: monkeypatch a literal.
MAX_FUSED_RAYS = 2**30

_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))

# Sodium d, the wavelength every hand-built system below is defined at.
_W0 = 0.5876


# ---------------------------------------------------------------------------
# Bundles
# ---------------------------------------------------------------------------


def _spiral(num_rays: int, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """``num_rays`` golden-angle points filling a disc of ``radius``.

    Deterministic and exactly ``num_rays`` long (a hexapolar ring count only
    hits 1 + 3n(n+1)), with no point at the exact centre, so a test that
    predicts a count gets the count it predicts.
    """
    idx = np.arange(num_rays, dtype=np.float64)
    r = radius * np.sqrt((idx + 0.5) / num_rays)
    theta = idx * _GOLDEN_ANGLE
    return r * np.cos(theta), r * np.sin(theta)


def make_rays(x, y, z, L, M, N, intensity=1.0, wavelength=_W0) -> RealRays:
    """Build a ``RealRays`` bundle on the active backend.

    Scalars are broadcast against the arrays, every component is converted
    with ``be.array`` (so the bundle lands on the active backend and, under
    Metal, in the active mode), and the result is a plain ``RealRays``.
    """
    arrays = np.broadcast_arrays(
        *(np.asarray(v, dtype=np.float64) for v in (x, y, z, L, M, N, intensity))
    )
    w = np.broadcast_to(np.asarray(wavelength, dtype=np.float64), arrays[0].shape)
    return RealRays(*(be.array(np.ascontiguousarray(a)) for a in (*arrays, w)))


def collimated_bundle(
    num_rays: int = DEFAULT_RAYS,
    *,
    radius: float = 5.0,
    z: float = -10.0,
    x0: float = 0.0,
    y0: float = 0.0,
    L: float = 0.0,
    M: float = 0.0,
    intensity: float = 1.0,
    wavelength: float = _W0,
) -> RealRays:
    """A disc of ``num_rays`` parallel rays, direction ``(L, M, +sqrt(...))``."""
    px, py = _spiral(num_rays, radius)
    n = math.sqrt(1.0 - L * L - M * M)
    return make_rays(
        x0 + px,
        y0 + py,
        np.full(num_rays, z),
        np.full(num_rays, L),
        np.full(num_rays, M),
        np.full(num_rays, n),
        intensity,
        wavelength,
    )


def pupil_bundle(
    optic: Any,
    num_rays: int = DEFAULT_RAYS,
    *,
    Hx: float = 0.0,
    Hy: float = 0.0,
    wavelength: float | None = None,
) -> RealRays:
    """``num_rays`` rays through a real optic's pupil, via its ray generator.

    The pupil points are the golden-angle disc of :func:`_spiral`, so the
    bundle is exactly ``num_rays`` long for every system in ``CATALOG``.
    """
    if wavelength is None:
        wavelength = optic.primary_wavelength
    px, py = _spiral(num_rays, 1.0)
    generator = optic.ray_tracer.ray_generator
    return generator.generate_rays(
        be.array(np.full(num_rays, Hx)),
        be.array(np.full(num_rays, Hy)),
        be.array(px),
        be.array(py),
        wavelength,
    )


# ---------------------------------------------------------------------------
# System builders
# ---------------------------------------------------------------------------


def _build(
    surfaces: list[dict],
    *,
    epd: float = 10.0,
    fields: tuple[float, ...] = (0.0,),
    wavelengths: tuple[float, ...] = (_W0,),
) -> Optic:
    """Assemble an ``Optic`` from a list of ``surfaces.add`` keyword dicts."""
    lens = Optic()
    for index, kwargs in enumerate(surfaces):
        lens.surfaces.add(index=index, **kwargs)
    lens.set_aperture(aperture_type="EPD", value=epd)
    lens.fields.set_type(field_type="angle")
    for y in fields:
        lens.fields.add(y=y)
    for k, w in enumerate(wavelengths):
        lens.wavelengths.add(value=w, is_primary=(k == 0))
    return lens


#: The CookeTriplet prescription (``optiland.samples.objectives``), as data so
#: that a fixture can add an aperture or a pose to one surface without
#: re-deriving the design.
_COOKE_SURFACES: tuple[dict, ...] = (
    {"radius": be.inf, "thickness": be.inf},
    {"radius": 22.01359, "thickness": 3.25896, "material": ("SK16", "hikari")},
    {"radius": -435.76044, "thickness": 6.00755},
    {"radius": -22.21328, "thickness": 0.99997, "material": ("F2", "schott")},
    {"radius": 20.29192, "thickness": 4.75041, "is_stop": True},
    {"radius": 79.68360, "thickness": 2.95208, "material": ("SK16", "hikari")},
    {"radius": -18.39533, "thickness": 42.20778},
    {},
)


def _cooke(extra: dict[int, dict] | None = None) -> Optic:
    """A Cooke triplet, with ``extra[i]`` merged into surface ``i``'s kwargs."""
    extra = extra or {}
    surfaces = []
    for index, kwargs in enumerate(_COOKE_SURFACES):
        merged = dict(kwargs)
        merged.update(extra.get(index, {}))
        surfaces.append(merged)
    return _build(
        surfaces,
        epd=10.0,
        fields=(0.0, 14.0, 20.0),
        wavelengths=(0.55, 0.48, 0.65),
    )


def cooke() -> Optic:
    """CookeTriplet: 6 finite-radius conics, 2 planes, three absorbing glasses.

    Branches: ``GEOM_PLANE`` (object + image), ``GEOM_CONIC`` with
    ``k = 0``, refraction at every surface, ``FL_ABSORBING`` (SK16 and F2 have
    ``k > 0`` at 0.55 um, so the kernel takes the ``exp`` path).  No aperture,
    no tilt.  Status bits: none — every ray hits every surface.
    """
    return _cooke()


def hubble() -> Optic:
    """HubbleTelescope: two mirrors, an annular obscuration, two conics.

    Branches: ``FL_REFLECTIVE`` (both mirrors), ``FL_HAS_APERTURE`` +
    ``AP_RADIAL`` with ``r_max = inf`` and ``r_min = 177.80035`` (the
    obscuration; ``FL_AP_IN_ROOT`` because ``StandardGeometry.distance``
    accepts ``aperture=``), conics ``k = -1.001152`` and ``-1.483014``,
    negative thicknesses (the beam folds back on itself).
    Status bits: ``ST_CLIPPED`` on the obscured centre of the pupil.
    """
    return _samples.HubbleTelescope()


def aspheric_singlet() -> Optic:
    """AsphericSinglet: the only Newton geometry in the shipped samples.

    Branches: ``GEOM_EVEN`` with three coefficients (r^2, r^4, r^6),
    ``SI_MAXITER = 100``, ``tol = 1e-6``, one absorbing glass (N-SF11).
    Status bits: none (every ray converges well inside ``max_iter``).
    """
    return _samples.AsphericSinglet()


def odd_asphere_singlet() -> Optic:
    """A hand-built odd asphere: ``sum(C_i r^i)``, odd powers included.

    Branches: ``GEOM_ODD``, three coefficients (r^1, r^2, r^3), so the
    kernel's odd-power path (``O::pow`` with an odd exponent, and the
    ``r = sqrt(r2)`` the even asphere never needs) runs.
    Status bits: none.
    """
    return _build(
        [
            {"radius": be.inf, "thickness": be.inf},
            {
                "surface_type": "odd_asphere",
                "radius": 25.0,
                "thickness": 6.0,
                "material": "N-BK7",
                "is_stop": True,
                "conic": 0.0,
                "coefficients": [1.0e-4, -2.5e-5, 4.0e-7],
            },
            {"radius": -40.0, "thickness": 60.0},
            {},
        ],
        epd=12.0,
    )


def tilted_triplet(pose: str = "rxryrz") -> Optic:
    """A Cooke triplet with surfaces 3 and 4 in one of five poses.

    ``pose`` is one of :data:`TILTED_TRIPLET_POSES`.  Branches:
    ``FL_HAS_RX`` / ``FL_HAS_RY`` / ``FL_HAS_RZ`` and the decentre slots
    ``SR_TX`` / ``SR_TY`` — i.e. the full localize/globalize pair of
    ``SR_CNRZ..SR_SRX``.  ``rz`` alone is a rotation about the optical axis
    of a rotationally symmetric surface: the trace is unchanged but the
    kernel's rz branch runs, which is exactly what makes it a good test.
    Status bits: none.
    """
    poses = {
        "rx": {"rx": 0.03},
        "ry": {"ry": 0.03},
        "rz": {"rz": 0.4},
        "dxdy": {"dx": 0.35, "dy": -0.25},
        "rxryrz": {"rx": 0.02, "ry": -0.015, "rz": 0.3},
    }
    if pose not in poses:
        raise ValueError(
            f"unknown pose {pose!r}; expected one of {TILTED_TRIPLET_POSES}"
        )
    return _cooke({3: poses[pose], 4: poses[pose]})


def fold_mirror() -> tuple[Optic, Callable[..., RealRays]]:
    """A 45-degree plane fold mirror: ``rx = pi/4``, absolute coordinates.

    Branches: ``FL_REFLECTIVE`` on a ``Plane``, ``FL_HAS_RX`` with the
    largest tilt v1 supports, and an image plane at ``rx = pi/2`` (its
    normal along -y) so the folded beam meets it at normal incidence.  The
    beam leaves along +y, which the accumulated-thickness model cannot
    express, hence the absolute ``z``/``y`` placement.
    Status bits: none.
    """
    lens = _build(
        [
            {"radius": be.inf, "thickness": be.inf},
            {"z": 0.0, "material": "mirror", "rx": math.pi / 4, "is_stop": True},
            {"y": 60.0, "z": 0.0, "rx": math.pi / 2},
        ],
        epd=10.0,
    )
    return lens, _fold_rays


def _fold_rays(optic: Any, num_rays: int = DEFAULT_RAYS) -> RealRays:
    """Collimated +z bundle, 8 mm across, starting 30 mm before the mirror."""
    del optic
    return collimated_bundle(num_rays, radius=4.0, z=-30.0)


def tilted_fold_mirror() -> tuple[Optic, Callable[..., RealRays]]:
    """A fold mirror tilted in all three axes and decentred, behind glass.

    Branches: ``FL_REFLECTIVE`` together with ``FL_HAS_RX`` + ``FL_HAS_RY`` +
    ``FL_HAS_RZ`` and non-zero ``SR_TX``/``SR_TY``, reached through an
    absorbing pre-material (N-SF11 slab) so the reflective branch runs with
    ``FL_ABSORBING`` set and ``n_pre != 1`` (fix L2.18: a fold mirror in air
    leaves the ``n_pre`` slot at exactly 1.0 and cannot see a mis-mirrored
    OPL term).
    Status bits: none.
    """
    lens = _build(
        [
            {"radius": be.inf, "thickness": be.inf},
            {"z": 0.0, "material": "N-SF11", "is_stop": True},
            {"z": 12.0},
            {
                "x": 0.6,
                "y": -0.4,
                "z": 40.0,
                "rx": math.pi / 4,
                "ry": 0.02,
                "rz": 0.15,
                "material": "mirror",
            },
            {"y": 55.0, "z": 40.0, "rx": math.pi / 2},
        ],
        epd=8.0,
    )
    return lens, _fold_rays


def planes_both_kinds() -> tuple[Optic, Callable[..., RealRays]]:
    """Both plane codes in one system: ``Plane`` and ``StandardGeometry(inf)``.

    ``GeometryFactory`` collapses ``radius = inf`` to a ``Plane``
    (``geometry_factory.py::_create_standard``), so the second surface's
    geometry is replaced by hand with a live ``StandardGeometry`` of infinite
    radius: the only way to reach ``GEOM_STD_INF`` (code 2), which the gate
    distinguishes from ``GEOM_CONIC`` by the host ``_is_radius_infinite``
    test.  The bundle is tilted (L = 0.2, M = 0.1) so refraction at each
    plane is non-trivial.
    Status bits: none.
    """
    lens = _build(
        [
            {"radius": be.inf, "thickness": be.inf},
            {
                "surface_type": "plane",
                "thickness": 5.0,
                "material": "N-BK7",
                "is_stop": True,
            },
            {"surface_type": "plane", "thickness": 40.0},
            {},
        ],
        epd=8.0,
    )
    surface = lens.surfaces.surfaces[2]
    surface.geometry = StandardGeometry(
        coordinate_system=surface.geometry.cs, radius=be.inf, conic=0.0
    )
    return lens, _tilted_plane_rays


def _tilted_plane_rays(optic: Any, num_rays: int = DEFAULT_RAYS) -> RealRays:
    """Collimated bundle at 0.2/0.1 direction cosines, 4 mm across."""
    del optic
    return collimated_bundle(num_rays, radius=4.0, z=-5.0, L=0.2, M=0.1)


def backward_plane() -> tuple[Optic, Callable[..., RealRays]]:
    """Planes reached with ``t < 0``: fully behind, then partly behind.

    Surface 1 sits 5 mm *behind* the bundle's launch plane, so every ray
    propagates backwards (``t = -5``).  Surface 2 is a plane at the same
    global z tilted ``ry = 0.6`` rad, so half the bundle is already past it
    and half is not: ``t`` changes sign inside one launch, which is the case
    a kernel that clamps ``t >= 0`` would silently get wrong.
    Status bits: none (a negative ``t`` is a legal virtual propagation; the
    OPL contribution is negative, ``standard_surface.py:301-304``).
    """
    lens = _build(
        [
            {"radius": be.inf, "thickness": be.inf},
            {"z": -5.0, "is_stop": True},
            {"z": -5.0, "ry": 0.6},
            {"z": 30.0},
        ],
        epd=20.0,
    )
    return lens, _backward_plane_rays


def _backward_plane_rays(optic: Any, num_rays: int = DEFAULT_RAYS) -> RealRays:
    """Wide collimated bundle launched at z = 0, i.e. past surface 1."""
    del optic
    return collimated_bundle(num_rays, radius=10.0, z=0.0)


def backward_newton() -> tuple[Optic, Callable[..., RealRays]]:
    """An even asphere reached with ``t < 0``.

    The bundle starts 2 mm *past* the asphere's vertex, so the base-conic
    seed has no admissible forward root and falls back to the legacy
    vertex-nearest finite root (``standard.py::_conic_intersection_distance``,
    the ``t > 0`` admissibility rule); Newton then converges on a negative
    ``t``.  The kernel must mirror both the seed fallback and the sign.
    Status bits: none.
    """
    lens = _build(
        [
            {"radius": be.inf, "thickness": be.inf},
            {
                "surface_type": "even_asphere",
                "radius": 30.0,
                "thickness": 5.0,
                "material": "N-BK7",
                "is_stop": True,
                "conic": 0.0,
                "coefficients": [2.0e-4, -1.0e-6],
            },
            {"radius": -30.0, "thickness": 40.0},
            {},
        ],
        epd=6.0,
    )
    return lens, _backward_newton_rays


def _backward_newton_rays(optic: Any, num_rays: int = DEFAULT_RAYS) -> RealRays:
    """Narrow collimated bundle launched 2 mm past the asphere vertex."""
    del optic
    return collimated_bundle(num_rays, radius=2.0, z=2.0)


def _force_standard_infinite(optic: Optic, index: int) -> None:
    """Replace surface ``index``'s geometry with ``StandardGeometry(inf)``.

    ``GeometryFactory`` never builds one (``radius = inf`` collapses to
    ``Plane``), and the two are *not* interchangeable: ``Plane.distance`` is
    a bare ``-z / N`` while the infinite-radius branch of
    ``_conic_intersection_distance`` floors the divisor at ``1e-14``.  A
    grazing ray therefore returns NaN on a ``Plane`` and ``0`` on a
    ``StandardGeometry(inf)``.
    """
    surface = optic.surfaces.surfaces[index]
    surface.geometry = StandardGeometry(
        coordinate_system=surface.geometry.cs, radius=be.inf, conic=0.0
    )


def tir_singlet() -> tuple[Optic, Callable[..., RealRays]]:
    """A glass slab with an exit face tilted just past the critical angle.

    N-BK7 at 0.5876 um has ``n = 1.5167984``, so the critical angle is
    ``asin(1/n) = 0.71987`` rad; the exit plane is tilted ``ry = 0.7218``
    rad, i.e. 0.0019 rad past it.  The bundle is a point-source fan in the
    x-z plane (``L`` from -0.35 to +0.35), so the internal angle sweeps the
    critical angle and the rays split into a transmitted and a
    totally-internally-reflected set.
    Status bits: ``ST_TIR`` on the low-``L`` half of the fan — 258 of 512
    rays, a contiguous prefix (measured, numpy reference path); the fan
    crosses the critical angle at ``L = -0.0029``, the rest of the trace
    carries their NaN direction cosines forward.
    """
    lens = _build(
        [
            {"radius": be.inf, "thickness": be.inf},
            {"z": 0.0, "material": "N-BK7", "is_stop": True},
            {"z": 10.0, "ry": 0.7218},
            {"z": 60.0},
        ],
        epd=6.0,
    )
    return lens, _tir_rays


def _tir_rays(optic: Any, num_rays: int = DEFAULT_RAYS) -> RealRays:
    """Point-source fan at (0, 0, -5), ``L`` uniform on [-0.35, 0.35]."""
    del optic
    lx = -0.35 + 0.7 * (np.arange(num_rays) + 0.5) / num_rays
    zeros = np.zeros(num_rays)
    return make_rays(
        zeros, zeros, np.full(num_rays, -5.0), lx, zeros, np.sqrt(1 - lx * lx)
    )


def miss_bundle() -> tuple[Optic, Callable[..., RealRays]]:
    """A Cooke triplet with a stop aperture, hit 200 mm off axis.

    The bundle is launched at ``x = 200`` mm, where the discriminant of the
    first sphere (``R = 22.01359``) is negative, so no root is admissible:
    ``_conic_intersection_distance`` returns NaN and the ray positions go
    NaN at surface 1.  The NaN then reaches the absorbing SK16 glass (the
    intensity picks up ``exp`` of a NaN path length) and finally the
    ``RadialAperture(r_max = 12)`` added at the stop, which clips it
    (``contains`` on NaN is False) -- the clip-after-NaN ordering fix L2.17
    asks for.
    Status bits: ``ST_MISS`` on every ray at surface 1, ``ST_CLIPPED`` on
    every ray at surface 4.  Measured intensity sequence on the numpy
    reference path: 1 after surface 1, NaN after surfaces 2-3 (SK16 absorbs,
    ``exp`` of a NaN path), 0 after surfaces 4-5 (the clip *assigns* zero),
    NaN again after surfaces 6-7 (the second absorbing element multiplies
    zero by ``exp`` of a NaN).  A kernel that clipped before propagating,
    or that treated NaN as "inside", produces a different sequence.
    """
    lens = _cooke({4: {"aperture": RadialAperture(r_max=12.0)}})
    return lens, _far_off_axis_rays


def _far_off_axis_rays(optic: Any, num_rays: int = DEFAULT_RAYS) -> RealRays:
    """Collimated bundle 5 mm across, centred 200 mm off axis."""
    del optic
    return collimated_bundle(num_rays, radius=5.0, z=-10.0, x0=200.0)


def reverse_bundle() -> tuple[Optic, Callable[..., RealRays]]:
    """A Cooke triplet entered by rays travelling away from it (``N < 0``).

    Both conic roots are then non-positive, so neither is admissible
    (``t > 0`` is the admissibility rule) and the vertex-nearest fallback
    picks a negative root: the "both roots non-positive" branch of
    ``_conic_candidates``, which a kernel that only implements the
    two-forward-roots case gets wrong without ever producing a NaN.
    Status bits: implementation-defined per root; the observable is that the
    fused and per-op ``x/y/z`` agree component-for-component.
    """
    return _cooke(), _reverse_rays


def _reverse_rays(optic: Any, num_rays: int = DEFAULT_RAYS) -> RealRays:
    """Collimated bundle at z = -20 travelling in -z, away from surface 1."""
    del optic
    px, py = _spiral(num_rays, 4.0)
    return make_rays(
        px,
        py,
        np.full(num_rays, -20.0),
        np.zeros(num_rays),
        np.zeros(num_rays),
        np.full(num_rays, -1.0),
    )


def _flat_newton_system() -> Optic:
    """Three infinite-radius surfaces, the middle one a weak even asphere.

    Every surface is reached through the infinite-radius branch of
    ``_conic_intersection_distance`` (``-z / max(|N|, 1e-14)``), including
    the image surface, so a ray travelling *inside* the surface plane has a
    finite distance everywhere and the trace stays bounded.  The asphere
    carries one coefficient, ``C1 = 1e-5``, with ``tol = 1e-3``: a ray at
    ``r = 12`` has a seed residual of ``1.44e-3 > tol`` and takes exactly
    one Newton step, while a ray at ``r <= 6`` is converged at its seed.
    """
    lens = _build(
        [
            {"radius": be.inf, "thickness": be.inf},
            {"z": 0.0, "is_stop": True},
            {
                "surface_type": "even_asphere",
                "z": 0.0,
                "radius": be.inf,
                "coefficients": [1.0e-5],
                "tol": 1e-3,
                "max_iter": 100,
            },
            {"z": 0.0},
        ],
        epd=24.0,
    )
    _force_standard_infinite(lens, 1)
    _force_standard_infinite(lens, 3)
    return lens


def grazing_bundle() -> tuple[Optic, Callable[..., RealRays]]:
    """Near-grazing rays (``L = 0.999``) that floor *nothing*: the control.

    ``|N| = 0.0447`` is nine orders above the ``1e-14`` divisor floor and
    ``|dF/dt| = 0.0447`` is thirteen orders above ``32 * eps * max(1, scale)``,
    so neither floor engages while the arithmetic is as ill-conditioned as
    a non-degenerate ray gets.  A floor test that also passes here is
    measuring the floor, not the fixture.
    Status bits: none.
    """
    return _flat_newton_system(), _near_grazing_rays


def _near_grazing_rays(optic: Any, num_rays: int = DEFAULT_RAYS) -> RealRays:
    """Collimated bundle at direction (0.999, 0, 0.0447), launched at z = -1."""
    del optic
    px, py = _spiral(num_rays, 6.0)
    lx = 0.999
    return make_rays(
        px,
        py,
        np.full(num_rays, -1.0),
        np.full(num_rays, lx),
        np.zeros(num_rays),
        np.full(num_rays, math.sqrt(1.0 - lx * lx)),
    )


def exact_grazing_bundle() -> tuple[Optic, Callable[..., RealRays]]:
    """``sign(0)``: rays lying exactly in the surface plane, ``N == 0``.

    The first ``num_rays // 8`` rays start at ``z = 0`` with direction
    ``(1, 0, 0)``: exactly in the plane of all three surfaces.  Every
    surface is an infinite-radius conic, so the divisor floor
    (``be.where(be.abs(rays.N) > 1e-14, rays.N, 1e-14)``) engages and
    returns ``t = -0 / 1e-14 = 0`` instead of ``0 / 0 = NaN`` -- the whole
    point of using ``StandardGeometry(inf)`` rather than ``Plane``, whose
    bare ``-z / N`` does produce NaN here.  On the asphere their
    ``dF/dt = f_x L + f_y M - N`` is *exactly* zero (they sit on the y axis,
    so ``f_x = 0``, and ``M = N = 0``), so ``_regularize_signed`` floors it;
    the remaining rays are at ``r = 12``, unconverged at their seed, which
    is what makes the Newton loop run at all (a converged batch never
    evaluates ``dF/dt``).  Refraction is a no-op: ``sign(0) = 0`` zeroes the
    aligned normal and ``u = 1`` leaves the direction cosines untouched.
    Status bits (measured, numpy reference path, 512 rays): ``ST_NZ_FLOORED``
    on the 64 grazing rays at all three surfaces, ``ST_DF_FLOORED`` on the
    same 64 at the asphere, one Newton iteration, every output finite
    (``max |x| = 12``, ``max |z| = 0``).
    """
    return _flat_newton_system(), _exact_grazing_rays


def _exact_grazing_rays(optic: Any, num_rays: int = DEFAULT_RAYS) -> RealRays:
    """``num_rays // 8`` in-plane rays, the rest at r = 12, normal incidence."""
    del optic
    n_graze = max(1, num_rays // 8)
    n_rest = num_rays - n_graze
    y_graze = 4.0 + 2.0 * (np.arange(n_graze) + 0.5) / n_graze
    theta = np.arange(n_rest) * _GOLDEN_ANGLE
    return make_rays(
        np.r_[np.zeros(n_graze), 12.0 * np.cos(theta)],
        np.r_[y_graze, 12.0 * np.sin(theta)],
        np.r_[np.zeros(n_graze), np.full(n_rest, -10.0)],
        np.r_[np.ones(n_graze), np.zeros(n_rest)],
        np.zeros(num_rays),
        np.r_[np.zeros(n_graze), np.ones(n_rest)],
    )


#: The 16 exact probe points of :func:`rim_bundle`, as
#: ``(x, y, inside_rect, inside_annulus)``.  Every coordinate is a small
#: power-of-two-friendly float, so ``x*x + y*y`` is exact in both modes and
#: the inclusive ``<=`` / ``>=`` bounds are decided by arithmetic, not by
#: rounding.
_RIM_PROBES: tuple[tuple[float, float, bool, bool], ...] = (
    (10.0, 0.0, True, False),  # exactly x_max; r = 10 > r_max = 8
    (-10.0, 0.0, True, False),  # exactly x_min
    (0.0, 5.0, True, True),  # exactly y_max; r = 5 inside the annulus
    (0.0, -5.0, True, True),  # exactly y_min
    (10.0, 5.0, True, False),  # exactly the corner
    (-10.0, -5.0, True, False),
    (8.0, 0.0, True, True),  # exactly r_max
    (-8.0, 0.0, True, True),
    (0.0, 2.0, True, True),  # exactly r_min
    (0.0, -2.0, True, True),
    (2.0, 0.0, True, True),  # exactly r_min, other axis
    (0.0, 1.0, True, False),  # inside r_min: clipped by the annulus
    (10.5, 0.0, False, False),  # just outside x_max
    (0.0, 5.5, False, False),  # just outside y_max
    (4.0, 3.0, True, True),  # interior, r = 5 exactly
    (6.0, 2.5, True, True),  # interior
)


def rim_bundle() -> tuple[Optic, Callable[..., RealRays]]:
    """Rays exactly on aperture boundaries, at normal incidence.

    Surface 1 carries ``RectangularAperture(-10, 10, -5, 5)`` and surface 2
    ``RadialAperture(r_max = 8, r_min = 2)``; both ``contains`` predicates
    are inclusive (``<=`` / ``>=``), so a ray exactly on a boundary is
    *inside* and keeps its intensity.  The bundle is :data:`_RIM_PROBES`
    tiled to ``num_rays`` at normal incidence, so every x and y is carried
    to the aperture unchanged (propagation only moves z) and the expected
    intensity pattern is the tiled ``inside_rect and inside_annulus``
    column, with no ray within a rounding step of a boundary decision.
    Status bits (measured, numpy reference path): ``ST_CLIPPED`` on 7 of
    every 16 rays -- surface 1 clips the two "just outside" probes, surface
    2 clips five more (the two at ``x = +/-10``, the two at the corners and
    the one inside ``r_min``) -- i.e. 224 of a 512-ray bundle.
    """
    lens = _build(
        [
            {"radius": be.inf, "thickness": be.inf},
            {
                "z": 0.0,
                "is_stop": True,
                "aperture": RectangularAperture(
                    x_min=-10.0, x_max=10.0, y_min=-5.0, y_max=5.0
                ),
            },
            {"z": 20.0, "aperture": RadialAperture(r_max=8.0, r_min=2.0)},
            {"z": 40.0},
        ],
        epd=20.0,
    )
    return lens, _rim_rays


def _rim_rays(optic: Any, num_rays: int = DEFAULT_RAYS) -> RealRays:
    """:data:`_RIM_PROBES` tiled to ``num_rays``, direction (0, 0, 1)."""
    del optic
    probes = np.asarray([(p[0], p[1]) for p in _RIM_PROBES], dtype=np.float64)
    reps = -(-num_rays // len(_RIM_PROBES))
    tiled = np.tile(probes, (reps, 1))[:num_rays]
    return make_rays(
        tiled[:, 0],
        tiled[:, 1],
        np.full(num_rays, -10.0),
        np.zeros(num_rays),
        np.zeros(num_rays),
        np.ones(num_rays),
    )


def nonconverging_asphere() -> tuple[Optic, Callable[..., RealRays]]:
    """An even asphere capped at ``max_iter = 1``: the non-converged branch.

    ``tol`` is the factory default ``1e-6``.  The bundle is tilted
    (``L = 0.3``) on purpose: at normal incidence ``x`` and ``y`` do not
    depend on ``t``, the residual is linear in ``t`` and one Newton step is
    *exact*, so a max_iter cap would never bite.  With a tilted bundle and a
    0.07 mm aspheric departure, one step leaves ``|F| = 1.0e-3`` on the
    outer rays.  Capping the iteration count rather than tightening ``tol``
    keeps the effective tolerance at ``tol`` in both modes
    (``_effective_tolerance``'s round-off floor is ``8 * eps * max(1, |t|)
    = 1.4e-12`` at this path scale), so the kernel's per-thread tolerance
    equals Python's batch tolerance exactly (design 4.6).
    Status bits (measured, numpy reference path, 512 rays):
    ``ST_NEWTON_NOT_CONVERGED`` on 372 of 512 rays at surface 1, no NaN;
    with ``max_iter = 2`` the same bundle converges completely, which is
    what makes the cap the cause.  ``iters``: Python reports one *batch*
    iteration count, the kernel writes a per-thread one -- 1 for the 372
    unconverged rays and 0 for the 140 that were converged at their seed
    (the per-op loop evaluates them too but freezes their step, so the
    distances agree).
    """
    lens = _build(
        [
            {"radius": be.inf, "thickness": be.inf},
            {
                "surface_type": "even_asphere",
                "radius": 20.0,
                "thickness": 6.0,
                "material": "N-BK7",
                "is_stop": True,
                "conic": 0.0,
                "coefficients": [-1.0e-3, -1.0e-5],
                "max_iter": 1,
            },
            {"radius": -40.0, "thickness": 40.0},
            {},
        ],
        epd=16.0,
    )
    return lens, _tilted_collimated_rays


def _tilted_collimated_rays(optic: Any, num_rays: int = DEFAULT_RAYS) -> RealRays:
    """Collimated bundle 8 mm in radius, tilted to ``L = 0.3``.

    The tilt is what makes ``x`` and ``y`` functions of ``t``, so the Newton
    residual is non-linear and the iteration count is meaningful.
    """
    del optic
    return collimated_bundle(num_rays, radius=8.0, z=-10.0, L=0.3)


def _wide_collimated_rays(optic: Any, num_rays: int = DEFAULT_RAYS) -> RealRays:
    """Collimated bundle 8 mm in radius, launched 10 mm before surface 1."""
    del optic
    return collimated_bundle(num_rays, radius=8.0, z=-10.0)


def even_asphere_5coeff() -> tuple[Optic, Callable[..., RealRays]]:
    """An even asphere with five coefficients: r^2 through r^10.

    Branches: the coefficient loop past ``i = 1``, i.e. ``pow_scalar(r2, i+1)``
    for exponents 2..5, which is where a kernel that unrolls only the first
    two terms (or that uses ``exp(n log r)`` instead of repeated squaring)
    diverges from Python's ``r2 ** (i + 1)``.
    Status bits: none.
    """
    lens = _build(
        [
            {"radius": be.inf, "thickness": be.inf},
            {
                "surface_type": "even_asphere",
                "radius": 25.0,
                "thickness": 6.0,
                "material": "N-BK7",
                "is_stop": True,
                "conic": -0.5,
                "coefficients": [
                    -1.0e-4,
                    2.0e-6,
                    -5.0e-8,
                    1.0e-9,
                    -2.5e-12,
                ],
            },
            {"radius": -35.0, "thickness": 45.0},
            {},
        ],
        epd=14.0,
    )
    return lens, _wide_collimated_rays


def even_asphere_inf_radius() -> tuple[Optic, Callable[..., RealRays]]:
    """An even asphere whose base conic is a plane (``radius = inf``).

    The conic part of the sag is ``r2 / (inf * (1 + sqrt(1 - 0)))`` -- zero
    through an infinity the kernel must not turn into a NaN -- and the base
    seed comes from the infinite-radius branch of
    ``_conic_intersection_distance`` rather than from ``conic_intersection``.
    ``FL_RADIUS_INF`` is set while ``SI_GEOM`` stays ``GEOM_EVEN``.
    Status bits: none.
    """
    lens = _build(
        [
            {"radius": be.inf, "thickness": be.inf},
            {
                "surface_type": "even_asphere",
                "radius": be.inf,
                "thickness": 6.0,
                "material": "N-BK7",
                "is_stop": True,
                "coefficients": [-1.5e-4, 3.0e-7],
            },
            {"radius": -40.0, "thickness": 45.0},
            {},
        ],
        epd=14.0,
    )
    return lens, _wide_collimated_rays


def off_axis_parabola_far_root() -> tuple[Optic, Callable[..., RealRays]]:
    """An off-axis parabola where the aperture must pick the *far* root.

    A parabola has ``k1 = 1 + k = 0``, so the sag-sheet test
    ``1 - k1 z / R >= 0`` is vacuous and both quadratic roots are admissible
    whenever both are forward.  The bundle enters the bowl below the vertex
    and leaves it 130 mm off axis; ``_select_distance``'s default policy
    takes the nearer root (``t = 64.06``, next to the vertex), while the
    ``OffsetRadialAperture(r_max = 25, offset_y = 130)`` contains only the
    far hit and flips the selection to ``t = 204.86`` (measured, numpy
    reference path, all 512 rays).  This is the only fixture where a
    mis-encoded ``FL_AP_IN_ROOT`` changes a position rather than only an
    intensity.
    Status bits: none -- every ray hits inside the aperture *after* the
    selection, so nothing is clipped.
    """
    lens = _build(
        [
            {"radius": be.inf, "thickness": be.inf},
            {
                "z": 0.0,
                "radius": 200.0,
                "conic": -1.0,
                "material": "mirror",
                "is_stop": True,
                "aperture": OffsetRadialAperture(r_max=25.0, offset_y=130.0),
            },
            {"z": 100.0},
        ],
        epd=6.0,
    )
    return lens, _oap_rays


def _oap_rays(optic: Any, num_rays: int = DEFAULT_RAYS) -> RealRays:
    """A 4 mm tall fan at y = -60, z = -20, climbing at M = 0.95."""
    del optic
    m = 0.95
    y = -60.0 + np.linspace(-2.0, 2.0, num_rays)
    zeros = np.zeros(num_rays)
    return make_rays(
        zeros,
        y,
        np.full(num_rays, -20.0),
        zeros,
        np.full(num_rays, m),
        np.full(num_rays, math.sqrt(1.0 - m * m)),
    )


def _aperture_singlet(aperture: Any) -> Optic:
    """A plano-convex singlet carrying ``aperture`` on its stop surface."""
    return _build(
        [
            {"radius": be.inf, "thickness": be.inf},
            {
                "radius": 50.0,
                "thickness": 5.0,
                "material": "N-BK7",
                "is_stop": True,
                "aperture": aperture,
            },
            {"radius": -50.0, "thickness": 45.0},
            {},
        ],
        epd=20.0,
    )


def rect_aperture() -> tuple[Optic, Callable[..., RealRays]]:
    """``RectangularAperture(-6, 6, -3, 3)`` on the stop of a singlet.

    Branches: ``AP_RECT`` with all four bounds in ``SR_AP0..SR_AP3`` and the
    four-comparison ``contains``; the bundle is a 10 mm disc, so the corners
    of the pupil fall outside in x, in y, and in both.
    Status bits: ``ST_CLIPPED`` on the rays outside the rectangle.
    """
    lens = _aperture_singlet(
        RectangularAperture(x_min=-6.0, x_max=6.0, y_min=-3.0, y_max=3.0)
    )
    return lens, _disc_rays


def _disc_rays(optic: Any, num_rays: int = DEFAULT_RAYS) -> RealRays:
    """Collimated bundle, 10 mm radius, wider than every aperture fixture."""
    del optic
    return collimated_bundle(num_rays, radius=10.0, z=-10.0)


def ellipse_aperture() -> tuple[Optic, Callable[..., RealRays]]:
    """``EllipticalAperture(a = 7, b = 3, offset = (1, -0.5))`` on the stop.

    Branches: ``AP_ELLIPSE``, the only aperture whose ``contains`` divides
    (``x^2/a^2 + y^2/b^2 <= 1``) and the only one with both semi-axes *and*
    an offset in the four real slots.
    Status bits: ``ST_CLIPPED`` outside the ellipse.
    """
    lens = _aperture_singlet(
        EllipticalAperture(a=7.0, b=3.0, offset_x=1.0, offset_y=-0.5)
    )
    return lens, _disc_rays


def offset_radial_aperture() -> tuple[Optic, Callable[..., RealRays]]:
    """``OffsetRadialAperture(r_max = 6, r_min = 2, offset = (1.5, -1))``.

    Branches: ``AP_OFFSET_RADIAL`` -- an annulus that is *not* centred, so a
    kernel that folds the offset into the radius test at the wrong place
    clips the wrong rays; both bounds are active.
    Status bits: ``ST_CLIPPED`` outside the annulus and inside its hole.
    """
    lens = _aperture_singlet(
        OffsetRadialAperture(r_max=6.0, r_min=2.0, offset_x=1.5, offset_y=-1.0)
    )
    return lens, _disc_rays


def zero_rmax_aperture() -> tuple[Optic, Callable[..., RealRays]]:
    """``RadialAperture(r_max = 0)``: everything is clipped.

    ``contains`` is ``r2 <= 0 and r2 >= 0``, i.e. true only at the exact
    origin, and the golden-angle disc has no ray there (the first point sits
    at ``r = radius * sqrt(0.5 / num_rays)``).  The degenerate aperture is
    worth a fixture because ``r_max ** 2 = 0`` is the one squared parameter
    that is *not* in the df64 denormal band the gate refuses, so it must be
    accepted and must clip everything.
    Status bits: ``ST_CLIPPED`` on every ray at surface 1.
    """
    return _aperture_singlet(RadialAperture(r_max=0.0)), _disc_rays


def uv_projection() -> Optic:
    """UVProjectionLens: 44 surfaces, finite conjugates, the deepest sample.

    Branches: the chunking path (``weighted_steps`` is 43 per ray, so a 1e6
    ray launch exceeds ``MAX_STEPS`` and ``_slab_plan`` must split it) and
    the memory guard of plan 3.6 (all-row recording at 1e6 rays needs
    ~3.1 GB and is refused with ``memory`` on a 16 GB machine, by design).
    Status bits: none.
    """
    return _samples.UVProjectionLens()


def nonzero_image_thickness() -> Optic:
    """A Cooke triplet whose image surface has a non-zero thickness.

    ``RealRayTracer.trace`` propagates through ``last_surface.thickness``
    *after* ``SurfaceGroup.trace`` returns, so the hook must not absorb that
    step: the fused trace ends at the image surface exactly as the per-op
    loop does, and the trailing propagation is applied by the caller
    afterwards (WP3's ``test_trailing_propagate_unchanged``).
    Status bits: none.
    """
    return _cooke({7: {"thickness": 5.0}})


def apodized_bundle() -> tuple[Optic, Callable[..., RealRays]]:
    """A launch that already contains ``i = 0`` rays (apodization).

    ``RayGenerator.generate_rays`` multiplies the launch intensity by the
    optic's apodization profile, so a bundle can arrive at the kernel with
    zeros in ``Q_I``.  Those rays must stay at zero through the whole trace
    (the absorbing-glass ``exp`` multiplies, it does not assign) without
    being confused with rays *clipped* to zero by an aperture.
    Status bits: none; every ray hits every surface, and exactly
    ``num_rays // 4`` of them carry zero intensity from launch to image.
    """
    return _cooke(), _apodized_rays


def _apodized_rays(optic: Any, num_rays: int = DEFAULT_RAYS) -> RealRays:
    """Cooke pupil bundle with every fourth ray launched at zero intensity."""
    rays = pupil_bundle(optic, num_rays)
    intensity = np.ones(num_rays)
    intensity[::4] = 0.0
    rays.i = be.array(intensity)
    return rays


def long_path_asphere() -> tuple[Optic, Callable[..., RealRays]]:
    """A Newton surface 5 m downstream: past the tolerance crossover.

    The asphere's ``tol`` is the ``NewtonRaphsonGeometry`` default ``1e-10``,
    for which Python's round-off floor ``8 * eps * max(1, |t|)`` overtakes
    ``tol`` at ``|t| = 3.5e3`` mm in df64 and ``1.1e5`` mm in sf64 -- the
    crossover of plan 1.2.  The seed distance here is ~5000 mm, so a df64
    launch must flag ``ST_TOL_CROSSOVER``, count
    ``fused_trace:late_fallback`` and write nothing, while sf64 completes.
    Status bits: ``ST_TOL_CROSSOVER`` on every ray at surface 2 in df64;
    none in sf64.
    """
    lens = _build(
        [
            {"radius": be.inf, "thickness": be.inf},
            {"thickness": 5000.0, "is_stop": True},
            {
                "surface_type": "even_asphere",
                "radius": 200.0,
                "thickness": 50.0,
                "material": "N-BK7",
                "conic": 0.0,
                "coefficients": [1.0e-7],
                "tol": 1e-10,
            },
            {"radius": -200.0, "thickness": 100.0},
            {},
        ],
        epd=10.0,
    )
    return lens, _long_path_rays


def _long_path_rays(optic: Any, num_rays: int = DEFAULT_RAYS) -> RealRays:
    """Collimated bundle, 5 mm radius, launched 10 mm before surface 1."""
    del optic
    return collimated_bundle(num_rays, radius=5.0, z=-10.0)


def long_path_batch_values() -> tuple[Optic, list, np.ndarray]:
    """``(optic, variables, values)`` where only design 2 crosses over.

    The variable is surface 1's thickness, i.e. the distance to the Newton
    surface of :func:`long_path_asphere`.  ``values`` is ``[B, 1]`` in the
    **scaled** units ``Variable.update`` expects (plan 3.9), produced with
    ``var.variable.scale(physical)``; the physical distances are
    ``[1000, 2000, 5000, 1500]`` mm, so in df64 exactly design index 2 is
    past the 3.5e3 mm crossover and must fall back per design, while the
    other three complete on the kernel (design 6.3, WP7's
    ``test_batch_late_fallback_per_design``).
    """
    from optiland.optimization.variable import Variable

    lens, _ = long_path_asphere()
    variable = Variable(lens, "thickness", surface_number=1)
    physical = np.array([1000.0, 2000.0, 5000.0, 1500.0])
    values = np.array([[variable.variable.scale(v)] for v in physical])
    return lens, [variable], values


def mixed_wavelength_bundle() -> tuple[Optic, Callable[..., RealRays]]:
    """A bundle carrying two wavelengths: the pre-launch refusal.

    ``can_fuse_trace``'s one device readback compares every ``w`` against
    ``w[0]``; this bundle fails it and the trace falls back before any
    launch (``mixed_wavelength``, a *feature* reason, so ``require``
    raises).  Direct ``SurfaceGroup.trace`` callers really do this --
    ``analysis/irradiance.py`` and ``intensity.py`` build multi-wavelength
    bundles -- so the refusal is a supported path, not a defensive check.
    Status bits: none; no launch happens.
    """
    return _cooke(), _two_wavelength_rays


def _two_wavelength_rays(optic: Any, num_rays: int = DEFAULT_RAYS) -> RealRays:
    """Cooke pupil bundle, first half at 0.48 um and second half at 0.65 um."""
    rays = pupil_bundle(optic, num_rays, wavelength=0.55)
    w = np.full(num_rays, 0.48)
    w[num_rays // 2 :] = 0.65
    rays.w = be.array(w)
    return rays


# ---------------------------------------------------------------------------
# The shipped catalog and its measured eligibility
# ---------------------------------------------------------------------------


def _build_catalog() -> dict[str, type]:
    """Every shipped sample system, keyed by class name."""
    catalog: dict[str, type] = {}
    for name in _samples.__all__:
        catalog[name] = getattr(_samples, name)
    for name in UNEXPORTED_OBJECTIVES:
        catalog[name] = getattr(_objectives, name)
    return catalog


#: name -> zero-argument class that builds the optic.
CATALOG: dict[str, type] = _build_catalog()

#: name -> the ``FusedTraceSkip`` reason a catalog system is expected to hit.
#:
#: **Measured, and empty.**  :func:`audit_catalog` walks every surface of
#: every ``CATALOG`` entry against the v1 feature set of plan 1.1 -- without
#: importing the gate, so this is an independent census and not the gate
#: agreeing with itself -- and finds all 29 systems eligible: only ``Plane``,
#: ``StandardGeometry`` and ``EvenAsphere`` geometries (AsphericSinglet is
#: the only Newton surface), only ``RefractiveReflectiveModel`` interactions,
#: no coating, no BSDF, no GRIN, no ``reference_cs``, finite poses, finite
#: ``n``/``k`` at every primary wavelength, and only ``RadialAperture``
#: apertures (HubbleTelescope's obscuration and UVReflectingMicroscope's).
#: This matches note 09 and the plan's preface claim that v1 covers all 29
#: samples end to end.  A row added here is a decision on the record: WP2's
#: ``test_gate_accepts_catalog`` requires every unlisted system to fuse.
KNOWN_INELIGIBLE: dict[str, str] = {}


def build(name: str) -> Any:
    """Instantiate the catalog system called ``name``."""
    return CATALOG[name]()


def audit_catalog() -> dict[str, tuple[str, str]]:
    """Independent eligibility census over ``CATALOG`` (no gate import).

    Returns ``{name: (reason, evidence)}`` for every system that the v1
    feature set of plan 1.1 cannot fuse; an eligible system is absent.  The
    checks and their order mirror design 2.2's per-surface sequence, but the
    implementation is deliberately separate from ``can_fuse_trace`` so that
    ``KNOWN_INELIGIBLE`` is evidence rather than a restatement of the gate
    (plan 0.2.6).  Run it on the NumPy backend; it builds every sample.
    """
    from optiland.interactions.refractive_reflective_model import (
        RefractiveReflectiveModel as _RRM,
    )
    from optiland.propagation.homogeneous import HomogeneousPropagation
    from optiland.surfaces.image_surface import ImageSurface
    from optiland.surfaces.object_surface import ObjectSurface
    from optiland.surfaces.standard_surface import Surface

    geometries = (Plane, StandardGeometry, EvenAsphere, OddAsphere)
    apertures = (
        RadialAperture,
        OffsetRadialAperture,
        RectangularAperture,
        EllipticalAperture,
    )
    out: dict[str, tuple[str, str]] = {}
    for name in sorted(CATALOG):
        optic = build(name)
        surfaces = optic.surfaces.surfaces
        w0 = optic.primary_wavelength
        verdict: tuple[str, str] | None = None
        if len(surfaces) < 2 or type(surfaces[0]) is not ObjectSurface:
            verdict = ("surface0_not_object", type(surfaces[0]).__name__)
        for index, surface in enumerate(surfaces[1:], start=1):
            if verdict is not None:
                break
            verdict = _audit_surface(
                surface,
                index,
                w0,
                geometries,
                apertures,
                (Surface, ImageSurface),
                _RRM,
                HomogeneousPropagation,
            )
        if verdict is not None:
            out[name] = verdict
    return out


def _audit_surface(
    surface: Any,
    index: int,
    w0: float,
    geometries: tuple,
    apertures: tuple,
    surface_types: tuple,
    interaction_cls: type,
    homogeneous_cls: type,
) -> tuple[str, str] | None:
    """One surface's verdict for :func:`audit_catalog` (None when eligible)."""
    where = f"surface {index}"
    if type(surface) not in surface_types:
        return "surface_type", f"{where} is {type(surface).__name__}"
    geometry = surface.geometry
    if type(geometry) not in geometries:
        return "geometry_type", f"{where} geometry {type(geometry).__name__}"
    cs = geometry.cs
    if cs.reference_cs is not None:
        return "reference_cs", where
    for axis in ("x", "y", "z", "rx", "ry", "rz"):
        value = float(be.to_numpy(getattr(cs, axis)))
        if not math.isfinite(value):
            return "pose_nonfinite", f"{where}.{axis} = {value}"
    model = surface.interaction_model
    if type(model) is not interaction_cls:
        return "interaction_type", f"{where} {type(model).__name__}"
    if getattr(model, "coating", None) is not None:
        return "coating", f"{where} {type(model.coating).__name__}"
    if getattr(model, "bsdf", None) is not None:
        return "bsdf", f"{where} {type(model.bsdf).__name__}"
    for material in (surface.material_pre, surface.material_post):
        if material is None:
            continue
        propagation = getattr(material, "propagation_model", None)
        if type(propagation) is not homogeneous_cls:
            return "propagation_model", f"{where} {type(propagation).__name__}"
    aperture = surface.aperture
    if aperture is not None and type(aperture) not in apertures:
        return "aperture_type", f"{where} {type(aperture).__name__}"
    if isinstance(geometry, EvenAsphere):
        max_iter = geometry.max_iter
        if not isinstance(max_iter, int) or not 0 <= max_iter <= 254:
            return "newton_params", f"{where} max_iter = {max_iter!r}"
        if not isinstance(geometry.tol, float):
            return "newton_params", f"{where} tol = {geometry.tol!r}"
    for material in (surface.material_pre, surface.material_post):
        if material is None:
            continue
        n = float(np.asarray(be.to_numpy(material.n(w0))).reshape(-1)[0])
        k = float(np.asarray(be.to_numpy(material.k(w0))).reshape(-1)[0])
        if not (math.isfinite(n) and math.isfinite(k)):
            return "nonfinite_index", f"{where} n = {n}, k = {k}"
    return None


# ---------------------------------------------------------------------------
# Refusal fixtures: one per FusedTraceSkip value
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Patch:
    """A monkeypatch a refusal case needs: ``setattr(import(module), attr, value)``."""

    module: str
    attr: str
    value: Any


@dataclass(frozen=True)
class RefusalCase:
    """Everything needed to make the gate return exactly one reason.

    Attributes:
        reason: the ``FusedTraceSkip`` value this case must produce.
        how: one line naming the mechanism, for the test id and the report.
        optic: the fresh ``Optic`` to trace (``None`` for cases that carry
            their own group).
        rays: ``rays_factory(optic, num_rays)``; ``None`` means
            :func:`pupil_bundle`.
        group: the object to pass to the gate / to ``trace`` when it is not
            ``optic.surfaces``.
        skip: the ``skip`` argument to pass to ``SurfaceGroup.trace``.
        num_rays: bundle size for this case (256 is host-resident on
            purpose; everything else stays above the threshold).
        env: environment variables to set for the trace.
        patches: monkeypatches to apply for the trace.
        grad: whether the trace must run with ``be.grad_mode`` enabled.
        per_op_raises: the exception the *per-op* path raises on this
            system, or ``None`` when it completes.  Three cases have one:
            the refusal is still the contract (counted, no launch), but the
            fallback then reproduces the upstream exception rather than an
            array, so "the fused result equals the per-op result" is
            "both raise the same exception".
        deterministic: False when the per-op trace is not reproducible even
            against itself (the BSDF fixture scatters randomly), so no
            array comparison is meaningful.
    """

    reason: FusedTraceSkip
    how: str
    optic: Any = None
    rays: Any = None
    group: Any = None
    skip: int = 0
    num_rays: int = DEFAULT_RAYS
    env: dict[str, str] = field(default_factory=dict)
    patches: tuple[Patch, ...] = ()
    grad: bool = False
    per_op_raises: type[BaseException] | None = None
    deterministic: bool = True


_TRACE = "optiland.backend.torch_backend.metal.trace"
_TRACE_RECORD = "optiland.backend.torch_backend.metal.trace_record"
_TRACE_MIRROR = "optiland.backend.torch_backend.metal.trace_mirror"


def _refuse_group_type() -> RefusalCase:
    """A ``SequencedSurfaceGroup``: not a ``SurfaceGroup``, never hooked."""
    from optiland.sequences import SequencedOptic

    lens = _cooke()
    steps = list(range(len(lens.surfaces.surfaces)))
    sequence = SequencedOptic(lens, "all", steps)
    return RefusalCase(
        reason=FusedTraceSkip.GROUP_TYPE,
        how="SequencedSurfaceGroup over the Cooke triplet's surfaces",
        optic=lens,
        group=sequence.surfaces,
    )


def _refuse_rays_type() -> RefusalCase:
    """``PolarizedRays``: a ``RealRays`` subclass, so the check is exact-type."""
    return RefusalCase(
        reason=FusedTraceSkip.RAYS_TYPE,
        how="PolarizedRays bundle (exact-type check, not isinstance)",
        optic=_cooke(),
        rays=_polarized_rays,
    )


def _polarized_rays(optic: Any, num_rays: int = DEFAULT_RAYS) -> Any:
    """A ``PolarizedRays`` copy of the Cooke pupil bundle."""
    from optiland.rays.polarized_rays import PolarizedRays

    rays = pupil_bundle(optic, num_rays)
    return PolarizedRays(rays.x, rays.y, rays.z, rays.L, rays.M, rays.N, rays.i, rays.w)


def _refuse_rays_shape() -> RefusalCase:
    """A ragged bundle: ``y`` one element shorter than ``x``.

    The per-op path cannot trace it either -- it raises ``ValueError`` on
    the first broadcast -- so the contract here is that the gate refuses
    *structurally* (no ``require`` raise, counter + 1, no launch) and the
    upstream loop then raises exactly what it raises today.
    """
    return RefusalCase(
        reason=FusedTraceSkip.RAYS_SHAPE,
        how="ragged bundle: len(y) == len(x) - 1",
        optic=_cooke(),
        rays=_ragged_rays,
        per_op_raises=ValueError,
    )


def _ragged_rays(optic: Any, num_rays: int = DEFAULT_RAYS) -> RealRays:
    """Cooke pupil bundle with one component truncated by a single element."""
    rays = pupil_bundle(optic, num_rays)
    rays.y = rays.y[:-1]
    return rays


def _refuse_host_resident() -> RefusalCase:
    """256 rays: exactly ``HOST_THRESHOLD``, so the bundle lives on the host."""
    return RefusalCase(
        reason=FusedTraceSkip.HOST_RESIDENT,
        how="N = 256 = HOST_THRESHOLD (257 is eligible)",
        optic=_cooke(),
        num_rays=256,
    )


def _refuse_requires_grad() -> RefusalCase:
    """Autograd on: every ``be.array`` is a ``requires_grad`` leaf."""
    return RefusalCase(
        reason=FusedTraceSkip.REQUIRES_GRAD,
        how="be.grad_mode enabled, so the ray tensors require grad",
        optic=_cooke(),
        grad=True,
    )


def _refuse_skip() -> RefusalCase:
    """``skip = 1``: the trace starts past the object surface."""
    return RefusalCase(
        reason=FusedTraceSkip.SKIP,
        how="SurfaceGroup.trace(rays, skip=1)",
        optic=_cooke(),
        skip=1,
    )


def _refuse_surface_type() -> RefusalCase:
    """A surface whose exact type is neither ``Surface`` nor ``ImageSurface``.

    The plan names ``SurfaceView`` here, but a view cannot in fact sit in a
    ``SurfaceGroup``: ``_update_surface_links`` assigns ``previous_surface``
    (a read-only property on a view) and ``SurfaceGroup.trace`` calls
    ``surface.trace(rays, record=record)`` while ``SurfaceView.trace`` takes
    no ``record`` keyword -- so such a group raises before reaching the gate
    and could never be compared against a per-op result.  What the check
    really guards is a *subclass*: ``type(surface) in (Surface,
    ImageSurface)`` refuses one even though it traces identically, because
    a subclass may override ``_trace_real``.  The fixture re-classes one
    Cooke surface into a local subclass, so the per-op trace is unchanged
    and only the type differs.
    """
    from optiland.surfaces.standard_surface import Surface

    class _ThirdPartySurface(Surface):
        """A Surface subclass that overrides nothing (the gate cannot know)."""

    lens = _cooke()
    lens.surfaces.surfaces[2].__class__ = _ThirdPartySurface
    return RefusalCase(
        reason=FusedTraceSkip.SURFACE_TYPE,
        how="surface 2 re-classed to a Surface subclass",
        optic=lens,
    )


def _refuse_surface0_not_object() -> RefusalCase:
    """A group whose first surface is a ``Surface``, not an ``ObjectSurface``."""
    from optiland.surfaces.surface_group import SurfaceGroup

    lens = _cooke()
    return RefusalCase(
        reason=FusedTraceSkip.SURFACE0_NOT_OBJECT,
        how="SurfaceGroup built from surfaces[1:] (no ObjectSurface at 0)",
        optic=lens,
        group=SurfaceGroup(list(lens.surfaces.surfaces[1:])),
    )


def _refuse_too_many_rays() -> RefusalCase:
    """A bundle of ``MAX_FUSED_RAYS + 1`` rays that costs 37 KB.

    Materialising 2^30 + 1 rays would be 8 GiB per df64 plane, and the
    ceiling is a literal in the gate, not a constant a test could lower.
    Instead every plane is a stride-0 ``expand`` of a single element: the
    shape the gate reads is genuine, the storage is one element, and the
    refusal happens (design 2.2 step 8) before anything touches the data.
    Measured: ``can_fuse_trace`` returns ``too_many_rays`` with
    ``n = 1073741825`` and MPS allocation stays at 0.04 MB.
    """
    return RefusalCase(
        reason=FusedTraceSkip.TOO_MANY_RAYS,
        how=f"{MAX_FUSED_RAYS + 1} rays as stride-0 views of one element",
        optic=_cooke(),
        rays=_oversized_rays,
    )


def _oversized_rays(optic: Any, num_rays: int = DEFAULT_RAYS) -> RealRays:
    """A Cooke bundle whose nine planes are expanded past the ray ceiling.

    ``num_rays`` is ignored: the point of the bundle is its shape.  The
    planes are replaced in place rather than passed to ``RealRays``, so
    ``__init__``'s ``be.zeros_like(x)`` never sees the large shape.
    """
    del num_rays
    rays = pupil_bundle(optic, 512)
    for name in ("x", "y", "z", "L", "M", "N", "i", "w", "opd"):
        plane = getattr(rays, name)
        setattr(rays, name, plane[:1].expand(MAX_FUSED_RAYS + 1))
    return rays


def _refuse_min_rays() -> RefusalCase:
    """A bundle below ``OPTILAND_METAL_FUSED_TRACE_MIN_RAYS``."""
    return RefusalCase(
        reason=FusedTraceSkip.MIN_RAYS,
        how="OPTILAND_METAL_FUSED_TRACE_MIN_RAYS = 1000000 over a 4096-ray bundle",
        optic=_cooke(),
        env={"OPTILAND_METAL_FUSED_TRACE_MIN_RAYS": "1000000"},
    )


def _refuse_geometry_type() -> RefusalCase:
    """A Zernike surface: a ``NewtonRaphsonGeometry`` v1 does not mirror."""
    lens = _build(
        [
            {"radius": be.inf, "thickness": be.inf},
            {
                "surface_type": "zernike",
                "radius": 50.0,
                "thickness": 5.0,
                "material": "N-BK7",
                "is_stop": True,
                "norm_radius": 10.0,
                "coefficients": [0.0, 0.0, 0.0, 0.0, 2e-3, 0.0, 0.0, 1e-3, 5e-4],
            },
            {"radius": -50.0, "thickness": 45.0},
            {},
        ],
        epd=10.0,
        wavelengths=(0.55,),
    )
    return RefusalCase(
        reason=FusedTraceSkip.GEOMETRY_TYPE,
        how="ZernikePolynomialGeometry on surface 1",
        optic=lens,
    )


def _refuse_reference_cs() -> RefusalCase:
    """A surface whose coordinate system is relative to another one."""
    lens = _cooke()
    lens.surfaces.surfaces[3].geometry.cs.reference_cs = CoordinateSystem(z=1.0)
    return RefusalCase(
        reason=FusedTraceSkip.REFERENCE_CS,
        how="surface 3 given a reference_cs (nested coordinate systems)",
        optic=lens,
    )


def _refuse_pose_nonfinite() -> RefusalCase:
    """A NaN pose scalar."""
    lens = _cooke()
    lens.surfaces.surfaces[3].geometry.cs.rx = float("nan")
    return RefusalCase(
        reason=FusedTraceSkip.POSE_NONFINITE,
        how="surface 3 rx = NaN",
        optic=lens,
    )


def _refuse_interaction_type() -> RefusalCase:
    """A paraxial (thin-lens) surface: the factory forces that model."""
    lens = _build(
        [
            {"radius": be.inf, "thickness": be.inf},
            {
                "surface_type": "paraxial",
                "f": 100.0,
                "thickness": 100.0,
                "is_stop": True,
            },
            {},
        ],
        epd=10.0,
    )
    return RefusalCase(
        reason=FusedTraceSkip.INTERACTION_TYPE,
        how="ThinLensInteractionModel (surface_type='paraxial')",
        optic=lens,
    )


def _refuse_coating() -> RefusalCase:
    """An unpolarized coating on the stop."""
    from optiland.coatings import SimpleCoating

    return RefusalCase(
        reason=FusedTraceSkip.COATING,
        how="SimpleCoating(transmittance=0.98) on surface 1",
        optic=_cooke(
            {1: {"coating": SimpleCoating(transmittance=0.98, reflectance=0.02)}}
        ),
    )


def _refuse_bsdf() -> RefusalCase:
    """A scattering model on the stop."""
    from optiland.scatter import LambertianBSDF

    return RefusalCase(
        reason=FusedTraceSkip.BSDF,
        how="LambertianBSDF on surface 1",
        optic=_cooke({1: {"bsdf": LambertianBSDF()}}),
        deterministic=False,
    )


def _refuse_polarization() -> RefusalCase:
    """A *polarized* coating: ``group.uses_polarization`` is then True.

    Kept distinct from :func:`_refuse_coating` because ``FusedTraceSkip``
    carries both values.

    **Measured against the landed gate: this case returns ``coating``, not
    ``polarization``.**  ``POLARIZATION`` is defined in ``FusedTraceSkip``
    and returned nowhere in ``trace_record.py``, so the reason is currently
    unreachable and ``test_refusal_reason[polarization]`` cannot pass.  The
    fixture keeps the realization the split needs -- report ``polarization``
    for a ``BaseCoatingPolarized`` and ``coating`` for every other coating,
    which is the only place in the gate that can see polarization at all
    (a polarized *bundle* is already ``rays_type``) -- and the request to
    WP2 is recorded in ``status.md``.

    The per-op path cannot trace this system either: the ray generator
    raises "Polarization must be set..." (hence the hand-built bundle) and
    the coating then fails on a ``RealRays``, so, as with the ragged
    bundle, the contract is that both paths raise the same way after the
    refusal is counted.
    """
    from optiland.coatings import PolarizerCoating

    return RefusalCase(
        reason=FusedTraceSkip.POLARIZATION,
        how="PolarizerCoating (BaseCoatingPolarized) on surface 1",
        optic=_cooke({1: {"coating": PolarizerCoating()}}),
        rays=_collimated_cooke_rays,
        per_op_raises=AttributeError,
    )


def _collimated_cooke_rays(optic: Any, num_rays: int = DEFAULT_RAYS) -> RealRays:
    """A hand-built collimated bundle for the Cooke triplet's 10 mm pupil.

    Used where ``RayGenerator`` itself would refuse to build a bundle --
    with a polarized coating on a surface it raises "Polarization must be
    set..." before the gate is ever reached -- so that the refusal under
    test is the gate's, not the generator's.
    """
    del optic
    return collimated_bundle(num_rays, radius=5.0, z=-10.0, wavelength=0.55)


def _refuse_propagation_model() -> RefusalCase:
    """A material whose propagation model is not homogeneous."""
    from optiland.materials.ideal import IdealMaterial
    from optiland.propagation.grin import GRINPropagation

    material = IdealMaterial(n=1.5, k=0.0, propagation_model=GRINPropagation())
    return RefusalCase(
        reason=FusedTraceSkip.PROPAGATION_MODEL,
        how="IdealMaterial with GRINPropagation on surface 1",
        optic=_cooke({1: {"material": material}}),
        per_op_raises=NotImplementedError,
    )


def _refuse_aperture_type() -> RefusalCase:
    """A polygon aperture: no closed-form ``contains`` for the kernel."""
    from optiland.physical_apertures.polygon import PolygonAperture

    aperture = PolygonAperture(x=[-6.0, 6.0, 6.0, -6.0], y=[-4.0, -4.0, 4.0, 4.0])
    return RefusalCase(
        reason=FusedTraceSkip.APERTURE_TYPE,
        how="PolygonAperture on surface 4",
        optic=_cooke({4: {"aperture": aperture}}),
    )


def _refuse_aperture_params() -> RefusalCase:
    """A supported aperture type carrying a non-scalar parameter.

    A 0-d or 1-element tensor is *accepted* on purpose (``Variable.update``
    stores raw 0-d tensors, plan 3.9), so the refusal needs a genuinely
    non-scalar parameter.  The other half of this reason -- a df64 squared
    parameter in the float32 denormal band -- is
    :func:`denormal_aperture_params`, which is mode-dependent and therefore
    not the entry in ``REFUSAL_FIXTURES``.
    """
    return RefusalCase(
        reason=FusedTraceSkip.APERTURE_PARAMS,
        how="RadialAperture whose r_max is a two-element array",
        optic=_cooke({4: {"aperture": RadialAperture(r_max=be.array([12.0, 12.0]))}}),
    )


def denormal_aperture_params() -> RefusalCase:
    """The df64-only half of ``aperture_params``: a denormal squared radius.

    ``_fill_radial`` stores ``r_max ** 2``; at ``r_max = 1e-20`` that is
    ``1e-40``, inside the float32 denormal band, which the df64 encoder
    flushes to zero -- turning an annulus into a disc.  The gate refuses it
    in df64 and *accepts* it in sf64 (int64 words, no float32 exponent), so
    this case is deliberately kept out of ``REFUSAL_FIXTURES``, whose entries
    must refuse in both modes.
    """
    return RefusalCase(
        reason=FusedTraceSkip.APERTURE_PARAMS,
        how="RadialAperture(r_max=1e-20): r_max**2 is a float32 denormal (df64 only)",
        optic=_cooke({4: {"aperture": RadialAperture(r_max=1e-20)}}),
    )


def _refuse_newton_params() -> RefusalCase:
    """``max_iter = 255``: one past what ``iters`` (uint8) can encode.

    ``ITERS_UNWRITTEN`` is ``0xFF``, so 255 iterations would collide with
    the write-completion sentinel; 254 is accepted.
    """
    lens = _build(
        [
            {"radius": be.inf, "thickness": be.inf},
            {
                "surface_type": "even_asphere",
                "radius": 25.0,
                "thickness": 6.0,
                "material": "N-BK7",
                "is_stop": True,
                "coefficients": [-1.0e-4],
                "max_iter": 255,
            },
            {"radius": -40.0, "thickness": 45.0},
            {},
        ],
        epd=12.0,
    )
    return RefusalCase(
        reason=FusedTraceSkip.NEWTON_PARAMS,
        how="EvenAsphere with max_iter = 255 (254 is accepted)",
        optic=lens,
    )


def _refuse_nonfinite_index() -> RefusalCase:
    """A material whose refractive index is NaN at the bundle's wavelength."""
    from optiland.materials.ideal import IdealMaterial

    return RefusalCase(
        reason=FusedTraceSkip.NONFINITE_INDEX,
        how="IdealMaterial(n = NaN) on surface 1",
        optic=_cooke({1: {"material": IdealMaterial(n=float("nan"))}}),
    )


def _refuse_memory() -> RefusalCase:
    """A memory budget too small for any launch."""
    return RefusalCase(
        reason=FusedTraceSkip.MEMORY,
        how="OPTILAND_METAL_FUSED_TRACE_MEMORY_FRACTION = 1e-9",
        optic=_cooke(),
        env={"OPTILAND_METAL_FUSED_TRACE_MEMORY_FRACTION": "1e-9"},
    )


def _refuse_mixed_wavelength() -> RefusalCase:
    """Two wavelengths in one bundle: the only refusal that costs a readback."""
    lens, rays = mixed_wavelength_bundle()
    return RefusalCase(
        reason=FusedTraceSkip.MIXED_WAVELENGTH,
        how="bundle carrying 0.48 um and 0.65 um",
        optic=lens,
        rays=rays,
    )


def _drifted_check_all() -> list[str]:
    """Stand-in for ``trace_mirror.check_all`` reporting one drifted mirror."""
    return ["optiland.rays.real_rays.RealRays.refract"]


def _refuse_mirror_drift() -> RefusalCase:
    """A mirrored Python function reported as drifted.

    ``check_all`` is patched to name one qualname; the driver's cached
    result (``trace._DRIFT``, ``None`` = not yet checked) is reset in the
    same breath so the refusal is re-evaluated in this process.  Both are
    strict patches, so a rename in WP3 fails the test loudly instead of
    letting it pass for the wrong reason (``trace.reset_driver_state()``
    clears the same cache).
    """
    return RefusalCase(
        reason=FusedTraceSkip.MIRROR_DRIFT,
        how="trace_mirror.check_all reports RealRays.refract as drifted",
        optic=_cooke(),
        patches=(
            Patch(_TRACE_MIRROR, "check_all", _drifted_check_all),
            Patch(_TRACE, "_DRIFT", None),
        ),
    )


def _unavailable_library(*args: Any, **kwargs: Any) -> Any:
    """Stand-in for ``trace._kernel_library`` on a machine without Metal."""
    del args, kwargs
    raise RuntimeError("fused-trace kernel unavailable (fixture)")


def _refuse_unavailable() -> RefusalCase:
    """The kernel fails to build: Metal missing, or a compile error."""
    return RefusalCase(
        reason=FusedTraceSkip.UNAVAILABLE,
        how="trace._kernel_library raises RuntimeError",
        optic=_cooke(),
        patches=(Patch(_TRACE, "_kernel_library", _unavailable_library),),
    )


#: reason -> zero-argument builder returning a :class:`RefusalCase`.
#: Exactly one entry per ``FusedTraceSkip`` value (WP5 acceptance).
REFUSAL_FIXTURES: dict[FusedTraceSkip, Callable[[], RefusalCase]] = {
    FusedTraceSkip.GROUP_TYPE: _refuse_group_type,
    FusedTraceSkip.RAYS_TYPE: _refuse_rays_type,
    FusedTraceSkip.RAYS_SHAPE: _refuse_rays_shape,
    FusedTraceSkip.HOST_RESIDENT: _refuse_host_resident,
    FusedTraceSkip.REQUIRES_GRAD: _refuse_requires_grad,
    FusedTraceSkip.SKIP: _refuse_skip,
    FusedTraceSkip.SURFACE_TYPE: _refuse_surface_type,
    FusedTraceSkip.SURFACE0_NOT_OBJECT: _refuse_surface0_not_object,
    FusedTraceSkip.TOO_MANY_RAYS: _refuse_too_many_rays,
    FusedTraceSkip.MIN_RAYS: _refuse_min_rays,
    FusedTraceSkip.GEOMETRY_TYPE: _refuse_geometry_type,
    FusedTraceSkip.REFERENCE_CS: _refuse_reference_cs,
    FusedTraceSkip.POSE_NONFINITE: _refuse_pose_nonfinite,
    FusedTraceSkip.INTERACTION_TYPE: _refuse_interaction_type,
    FusedTraceSkip.COATING: _refuse_coating,
    FusedTraceSkip.BSDF: _refuse_bsdf,
    FusedTraceSkip.POLARIZATION: _refuse_polarization,
    FusedTraceSkip.PROPAGATION_MODEL: _refuse_propagation_model,
    FusedTraceSkip.APERTURE_TYPE: _refuse_aperture_type,
    FusedTraceSkip.APERTURE_PARAMS: _refuse_aperture_params,
    FusedTraceSkip.NEWTON_PARAMS: _refuse_newton_params,
    FusedTraceSkip.NONFINITE_INDEX: _refuse_nonfinite_index,
    FusedTraceSkip.MEMORY: _refuse_memory,
    FusedTraceSkip.MIXED_WAVELENGTH: _refuse_mixed_wavelength,
    FusedTraceSkip.MIRROR_DRIFT: _refuse_mirror_drift,
    FusedTraceSkip.UNAVAILABLE: _refuse_unavailable,
}


# ---------------------------------------------------------------------------
# Registration on the adapters
# ---------------------------------------------------------------------------

#: The system fixtures, by name, in the order the report lists them.  Every
#: entry is a zero-argument callable returning ``Optic`` or
#: ``(Optic, rays_factory)``.
FIXTURES: dict[str, Callable[[], Any]] = {
    "cooke": cooke,
    "hubble": hubble,
    "aspheric_singlet": aspheric_singlet,
    "odd_asphere_singlet": odd_asphere_singlet,
    "fold_mirror": fold_mirror,
    "tilted_fold_mirror": tilted_fold_mirror,
    "planes_both_kinds": planes_both_kinds,
    "backward_plane": backward_plane,
    "backward_newton": backward_newton,
    "tir_singlet": tir_singlet,
    "miss_bundle": miss_bundle,
    "reverse_bundle": reverse_bundle,
    "grazing_bundle": grazing_bundle,
    "exact_grazing_bundle": exact_grazing_bundle,
    "rim_bundle": rim_bundle,
    "nonconverging_asphere": nonconverging_asphere,
    "even_asphere_5coeff": even_asphere_5coeff,
    "even_asphere_inf_radius": even_asphere_inf_radius,
    "off_axis_parabola_far_root": off_axis_parabola_far_root,
    "rect_aperture": rect_aperture,
    "ellipse_aperture": ellipse_aperture,
    "offset_radial_aperture": offset_radial_aperture,
    "zero_rmax_aperture": zero_rmax_aperture,
    "uv_projection": uv_projection,
    "nonzero_image_thickness": nonzero_image_thickness,
    "apodized_bundle": apodized_bundle,
    "long_path_asphere": long_path_asphere,
    "mixed_wavelength_bundle": mixed_wavelength_bundle,
    **{
        f"tilted_triplet_{pose}": (lambda p=pose: tilted_triplet(p))
        for pose in TILTED_TRIPLET_POSES
    },
}


def _optic_only(fixture: Callable[[], Any]) -> Callable[[], Optic]:
    """Wrap a fixture so it returns the ``Optic`` alone.

    ``GeometryAdapter.fixtures`` is typed ``tuple[Callable[[], Optic], ...]``
    (plan 3.3), so the ``(optic, rays_factory)`` fixtures are adapted rather
    than excluded -- an adapter whose only coverage came from a bundle
    fixture would otherwise look untested.
    """

    def build_optic() -> Optic:
        result = fixture()
        return result[0] if isinstance(result, tuple) else result

    build_optic.__name__ = getattr(fixture, "__name__", "fixture")
    build_optic.__doc__ = fixture.__doc__
    return build_optic


#: Exact adapter class -> the fixtures that exercise it.  Keyed by class, so
#: an adapter registered for a class that is not here gets no fixtures and
#: fails WP2's ``test_adapters_have_fixtures`` meta-test (design 8.5).
FIXTURES_BY_CLASS: dict[type, tuple[Callable[[], Any], ...]] = {
    Plane: (cooke, planes_both_kinds, backward_plane, rim_bundle),
    StandardGeometry: (
        cooke,
        hubble,
        planes_both_kinds,
        off_axis_parabola_far_root,
        exact_grazing_bundle,
    ),
    EvenAsphere: (
        aspheric_singlet,
        even_asphere_5coeff,
        even_asphere_inf_radius,
        nonconverging_asphere,
        backward_newton,
        long_path_asphere,
    ),
    OddAsphere: (odd_asphere_singlet,),
    RadialAperture: (hubble, zero_rmax_aperture, rim_bundle, miss_bundle),
    OffsetRadialAperture: (offset_radial_aperture, off_axis_parabola_far_root),
    RectangularAperture: (rect_aperture, rim_bundle),
    EllipticalAperture: (ellipse_aperture,),
    RefractiveReflectiveModel: (
        cooke,
        hubble,
        fold_mirror,
        tilted_fold_mirror,
        tir_singlet,
    ),
}


def register(strict: bool = False) -> dict[str, int]:
    """Attach the fixtures of :data:`FIXTURES_BY_CLASS` to the adapters.

    The adapter dataclasses are frozen, so each registry entry is replaced
    by a copy carrying its ``fixtures`` tuple.  Idempotent: calling it twice
    installs the same tuples.  Returns ``{registry name: adapters filled}``.

    Args:
        strict: when True, raise if a registered adapter has no fixtures.
            WP2's meta-test turns this on once the registries are filled; at
            WP5 time they may still be empty, and filling nothing is not an
            error.

    Raises:
        LookupError: with ``strict`` and an adapter class that has no
            fixtures here.
    """
    filled: dict[str, int] = {}
    registries = (
        ("geometry", GEOMETRY_ADAPTERS),
        ("aperture", APERTURE_ADAPTERS),
        ("interaction", INTERACTION_ADAPTERS),
    )
    missing: list[str] = []
    for name, registry in registries:
        count = 0
        for cls, adapter in list(registry.items()):
            fixtures = FIXTURES_BY_CLASS.get(cls)
            if not fixtures:
                missing.append(f"{name}:{cls.__name__}")
                continue
            registry[cls] = replace(adapter, fixtures=tuple(fixtures))
            count += 1
        filled[name] = count
    if strict and missing:
        raise LookupError("no trace_fixtures entry for " + ", ".join(sorted(missing)))
    return filled


def _main() -> int:
    """``python scripts/trace_fixtures.py``: build and audit everything."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        action="store_true",
        help="run the independent catalog eligibility census",
    )
    parser.add_argument("--num-rays", type=int, default=512)
    args = parser.parse_args()

    if args.audit:
        ineligible = audit_catalog()
        print(f"catalog: {len(CATALOG)} systems, {len(ineligible)} ineligible")
        for name, (reason, evidence) in sorted(ineligible.items()):
            print(f"  {name}: {reason} ({evidence})")
        print(f"KNOWN_INELIGIBLE: {KNOWN_INELIGIBLE}")
        if set(ineligible) != set(KNOWN_INELIGIBLE):
            print("MISMATCH: KNOWN_INELIGIBLE disagrees with the census")
            return 1
        return 0

    for name, fixture in FIXTURES.items():
        result = fixture()
        optic, rays_factory = result if isinstance(result, tuple) else (result, None)
        rays = (
            rays_factory(optic, args.num_rays)
            if rays_factory is not None
            else pupil_bundle(optic, args.num_rays)
        )
        optic.surfaces.trace(rays)
        x = np.asarray(be.to_numpy(rays.x))
        intensity = np.asarray(be.to_numpy(rays.i))
        print(
            f"{name:28s} S={len(optic.surfaces.surfaces):3d} "
            f"nan={int(np.isnan(x).sum()):5d} zero_i={int((intensity == 0).sum()):5d}"
        )
    for reason, factory in REFUSAL_FIXTURES.items():
        case = factory()
        print(f"{str(reason):24s} {case.how}")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual use
    raise SystemExit(_main())
