"""Tests for the compiled trace plan of real rays.

The plan is a fast path of ``SurfaceGroup.trace``, so it must give the bits
of the generic trace. Each test traces once with the plan and once with the
private switch ``_ENABLED`` off, in one process, and compares all ray arrays
and records with ``assert_bits_equal``. NumPy only: no ``set_test_backend``;
the Torch test checks the fallback.

Manuel Fragata Mendes, 2026
"""

from __future__ import annotations

import functools

import numba
import numpy as np
import pytest

import optiland.backend as be
import optiland.geometries.newton_raphson as newton_raphson
import optiland.surfaces._chunked_trace as chunked
import optiland.surfaces._trace_plan as plan
from optiland.geometries import EvenAsphere, OddAsphere, StandardGeometry
from optiland.materials import IdealMaterial
from optiland.materials.base import BaseMaterial
from optiland.optic import Optic
from optiland.physical_apertures import (
    EllipticalAperture,
    OffsetRadialAperture,
    RadialAperture,
)
from optiland.propagation.homogeneous import HomogeneousPropagation
from optiland.rays import PolarizedRays, RealRays
from optiland.samples.objectives import CookeTriplet, DoubleGauss
from optiland.samples.simple import AsphericSinglet, CementedAchromat, TelescopeDoublet
from optiland.samples.telescopes import HubbleTelescope
from optiland.surfaces import ImageSurface, SurfaceGroup
from optiland.surfaces.standard_surface import Surface

from .utils import assert_bits_equal

# The ray arrays and the records that a trace sets, in the order of the plan.
# test_cleared_records_are_the_arrays_of_a_reset_surface checks that the
# records are complete.
RAY_ATTRS = (*plan._OUTPUTS, "w")
RECORD_ATTRS = plan._CLEARED


def _fan(n, wavelengths=(0.55,), angle=0.05, height=2.0):
    """n rays in a fan that starts at z = 0, with skew rays.

    The heights in y go from -height to height and the angles from -angle to
    angle (radians). L = 0.01 gives skew rays, which leave the y-z plane.
    np.resize repeats the wavelengths from ray to ray.
    """
    ang = np.linspace(-angle, angle, n)
    return RealRays(
        np.zeros(n),
        np.linspace(-height, height, n),
        np.zeros(n),
        np.full(n, 0.01),
        np.sin(ang),
        np.sqrt(1 - 0.01**2 - np.sin(ang) ** 2),
        np.ones(n),
        np.resize(np.asarray(wavelengths, dtype=float), n),
    )


def _image_surface():
    """The Cooke triplet with an ImageSurface as its last surface."""
    optic = CookeTriplet()
    surfaces = optic.surfaces._surfaces
    last = surfaces[-1]
    surfaces[-1] = ImageSurface(surfaces[-2], last.geometry, last.material_post)
    return optic


def _flat_and_conic():
    """The Cooke triplet with a flat StandardGeometry and a conic surface."""
    optic = CookeTriplet()
    surfaces = optic.surfaces.surfaces
    cs = surfaces[3].geometry.cs
    surfaces[3].geometry = StandardGeometry(cs, be.inf)
    surfaces[5].geometry.k = be.array(-0.6)
    return optic


def _decentered():
    """The Cooke triplet with surfaces moved in x and in y."""
    optic = CookeTriplet()
    surfaces = optic.surfaces.surfaces
    surfaces[2].geometry.cs.x = be.array(0.1)
    surfaces[4].geometry.cs.y = be.array(-0.2)
    return optic


def _absorbing_singlet():
    """A singlet of an IdealMaterial with k > 0: one absorption coefficient."""
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1,
        radius=20.0,
        thickness=5.0,
        material=IdealMaterial(n=1.5, k=1e-5),
        is_stop=True,
    )
    optic.surfaces.add(index=2, radius=-40.0, thickness=30.0)
    optic.surfaces.add(index=3)
    optic.set_aperture(aperture_type="EPD", value=8.0)
    optic.fields.set_type("angle")
    optic.fields.add(y=0)
    optic.fields.add(y=3)
    optic.wavelengths.add(0.55, is_primary=True)
    return optic


def _oap():
    """An off-axis parabolic mirror: decenter, tilt, conic, offset aperture.

    Both roots of the parabola can be hits in front of the ray, and the
    aperture selects the root on the used part of the mirror.
    """
    rfl = 25.4
    optic = Optic(name="oap")
    optic.surfaces.add(index=0, z=0, radius=be.inf)
    optic.surfaces.add(index=1, z=10, is_stop=True)
    optic.surfaces.add(
        index=2,
        z=0,
        y=-rfl / 2,
        radius=-rfl,
        conic=-1,
        aperture=OffsetRadialAperture(r_max=12.7, offset_y=rfl),
        material="mirror",
        rx=np.pi / 2,
    )
    optic.surfaces.add(
        index=3,
        z=rfl,
        y=40,
        radius=be.inf,
        aperture=OffsetRadialAperture(r_max=12.7, offset_y=0),
        rx=np.pi / 2,
    )
    optic.set_aperture(aperture_type="EPD", value=5.0)
    optic.fields.set_type("angle")
    optic.fields.add(y=0)
    optic.fields.add(y=1)
    optic.wavelengths.add(0.55, is_primary=True)
    return optic


def _clipped():
    """The Cooke triplet with radial apertures that clip part of the pupil."""
    optic = CookeTriplet()
    surfaces = optic.surfaces.surfaces
    surfaces[1].aperture = RadialAperture(r_max=9.0)
    surfaces[3].aperture = RadialAperture(r_max=3.0, r_min=0.5)
    surfaces[5].aperture = OffsetRadialAperture(r_max=8.0, offset_x=0.4)
    return optic


def _tilted():
    """The telescope doublet with a surface tilted about x, y and z."""
    optic = TelescopeDoublet()
    cs = optic.surfaces.surfaces[2].geometry.cs
    cs.rx = be.array(0.01)
    cs.ry = be.array(-0.02)
    cs.rz = be.array(0.3)
    cs.x = be.array(0.1)
    return optic


def _asphere_triplet():
    """The Cooke triplet with two even aspheres (two and six coefficients)."""
    optic = CookeTriplet()
    surfaces = optic.surfaces.surfaces
    for index, coefficients in (
        (1, [2e-4, -3e-6]),
        (4, [-1e-4, 2e-6, -3e-8, 4e-10, -5e-12, 6e-14]),
    ):
        geometry = surfaces[index].geometry
        surfaces[index].geometry = EvenAsphere(
            geometry.cs, geometry.radius, -0.3, coefficients=coefficients
        )
    return optic


def _flat_asphere():
    """The Cooke triplet with a flat even asphere: R = inf, with coefficients.

    The base conic is the plane z = 0, so the seed is the distance of FLAT.
    """
    optic = CookeTriplet()
    surface = optic.surfaces.surfaces[3]
    surface.geometry = EvenAsphere(
        surface.geometry.cs, be.inf, coefficients=[1e-3, -2e-5]
    )
    return optic


def _clipped_aspheres():
    """The asphere triplet with an aperture on each asphere.

    Each aperture takes part in the conic seed and clips rays.
    """
    optic = _asphere_triplet()
    surfaces = optic.surfaces.surfaces
    surfaces[1].aperture = RadialAperture(r_max=6.0)
    surfaces[4].aperture = OffsetRadialAperture(r_max=4.0, offset_y=0.3)
    return optic


def _oap_asphere():
    """The off-axis parabolic mirror of ``_oap``, with even asphere terms.

    A decentered and tilted asphere mirror with an offset aperture.
    """
    optic = _oap()
    surface = optic.surfaces.surfaces[2]
    geometry = surface.geometry
    surface.geometry = EvenAsphere(
        geometry.cs, geometry.radius, -1.0, coefficients=[2e-6, -1e-9]
    )
    return optic


# Systems in the domain of the plan. CookeTriplet has a glass with k > 0
# (one absorption coefficient), so its traces run the intensity pass.
SYSTEMS = {
    "cooke": CookeTriplet,
    "double_gauss": DoubleGauss,
    "achromat": CementedAchromat,
    "doublet": TelescopeDoublet,
    "image_surface": _image_surface,
    "flat_and_conic": _flat_and_conic,
    "decentered": _decentered,
    "absorbing": _absorbing_singlet,
    "oap": _oap,
    "clipped": _clipped,
    "tilted": _tilted,
    "hubble": HubbleTelescope,
    "aspheric_singlet": AsphericSinglet,
    "asphere_triplet": _asphere_triplet,
    "flat_asphere": _flat_asphere,
    "clipped_aspheres": _clipped_aspheres,
    "oap_asphere": _oap_asphere,
}


@pytest.fixture
def calls(monkeypatch):
    """The results of the calls of trace_with_plan, in order.

    SurfaceGroup looks up trace_with_plan in its module at each call, so the
    spy replaces the module attribute.
    """
    results = []
    original = plan.trace_with_plan

    def spy(*args, **kwargs):
        results.append(original(*args, **kwargs))
        return results[-1]

    monkeypatch.setattr(plan, "trace_with_plan", spy)
    return results


@pytest.fixture
def floor_checks(monkeypatch):
    """The results of the calls of _floor_applies, in order.

    trace_with_plan calls it after the kernel, for a plan with an even
    asphere. So it tells a False of the build (no call) from a False after
    the kernel (True).
    """
    results = []
    original = plan._floor_applies

    def spy(*args):
        results.append(original(*args))
        return results[-1]

    monkeypatch.setattr(plan, "_floor_applies", spy)
    return results


def _state(optic, rays):
    """The ray arrays and the records of all surfaces, by name."""
    out = {f"ray.{name}": getattr(rays, name) for name in RAY_ATTRS}
    for k, surface in enumerate(optic.surfaces.surfaces):
        for name in RECORD_ATTRS:
            out[f"s{k}.{name}"] = getattr(surface, name)
    return out


def _trace(build, rays, record, monkeypatch, enabled):
    """Trace the rays of ``rays()`` through ``build()``, with or without the plan.

    ``_MIN_RAYS = 10**12`` turns off the traces in parts, so the generic
    trace is the loop over the surfaces.
    """
    monkeypatch.setattr(plan, "_ENABLED", enabled)
    monkeypatch.setattr(chunked, "_MIN_RAYS", 10**12)
    optic = build()
    traced = rays()
    optic.surfaces.trace(traced, record=record)
    return _state(optic, traced)


def _assert_same(a, b):
    """Assert that two trace results hold the same arrays, bit for bit.

    Compiled code can give a NaN another sign bit (see the developer page
    on performance internals), so the sign of NaN is not compared.
    """
    assert a.keys() == b.keys()
    for key in a:
        if a[key] is None or b[key] is None:
            assert a[key] is None and b[key] is None, key
        else:
            assert_bits_equal(a[key], b[key], ignore_nan_sign=True)


# The paraxial ray aiming warns that it ignores the decenters of "decentered".
@pytest.mark.filterwarnings("ignore:paraxial ray tracing")
@pytest.mark.parametrize("record", [True, False])
@pytest.mark.parametrize("hy", [0.0, 1.0])
@pytest.mark.parametrize("name", sorted(SYSTEMS))
def test_optic_trace_is_identical(calls, monkeypatch, name, hy, record):
    """optic.trace: ray aiming, the plan, and the propagation to the image."""

    def run(enabled):
        monkeypatch.setattr(plan, "_ENABLED", enabled)
        optic = SYSTEMS[name]()
        rays = optic.trace(
            Hx=0.0,
            Hy=hy,
            wavelength=optic.primary_wavelength,
            num_rays=9,
            distribution="hexapolar",
            record=record,
        )
        return _state(optic, rays)

    with_plan = run(True)
    assert calls == [True]
    _assert_same(with_plan, run(False))


@pytest.mark.parametrize("record", [True, False])
@pytest.mark.parametrize("n", [1, 7, 255, 256, 257, 1000])
@pytest.mark.parametrize(
    "name", ["cooke", "double_gauss", "clipped", "tilted", "asphere_triplet"]
)
def test_ray_counts_and_wavelengths_are_identical(calls, monkeypatch, name, n, record):
    """Blocks of any length, with one wavelength and with all of the system.

    255, 256 and 257 rays are at the _BLOCK boundary. The second fan has the
    wavelengths of the system, so each glass has one index per ray, which
    gives rows of per-ray values.
    """
    build = SYSTEMS[name]
    wavelengths = [w.value for w in build().wavelengths.wavelengths]
    for factory in (lambda: _fan(n), lambda: _fan(n, wavelengths)):
        calls.clear()
        with_plan = _trace(build, factory, record, monkeypatch, True)
        assert calls == [True]
        _assert_same(with_plan, _trace(build, factory, record, monkeypatch, False))


@pytest.mark.parametrize("name", ["cooke", "absorbing", "clipped", "asphere_triplet"])
def test_rays_that_miss_are_identical(calls, monkeypatch, name):
    """Rays far outside the lens give NaN at the same positions.

    In an absorbing glass, the exponent of a ray that missed is NaN, so its
    intensity is NaN, unless an aperture clipped the ray before.
    """

    def rays():
        return _fan(300, height=40.0, angle=0.3)

    with_plan = _trace(SYSTEMS[name], rays, True, monkeypatch, True)
    assert calls == [True]
    assert np.isnan(with_plan["ray.x"]).any()
    if name != "clipped":
        assert np.isnan(with_plan["ray.i"]).any()
    _assert_same(with_plan, _trace(SYSTEMS[name], rays, True, monkeypatch, False))


@pytest.mark.parametrize("record", [True, False])
@pytest.mark.parametrize("name", ["cooke", "clipped"])
def test_intensity_pass_in_parts_is_identical(calls, monkeypatch, name, record):
    """Three parts of 64 rays and a part of 8 rays give the bits of the bundle.

    The apertures of "clipped" clip rays in the absorbing Cooke triplet.
    """
    monkeypatch.setattr(plan, "_PART", 64)

    def rays():
        return _fan(200, height=6.0)

    with_plan = _trace(SYSTEMS[name], rays, record, monkeypatch, True)
    assert calls == [True]
    _assert_same(with_plan, _trace(SYSTEMS[name], rays, record, monkeypatch, False))


def _clip_then_miss():
    """An aperture clips a ray, which then misses in an absorbing glass."""
    optic = Optic(name="clip-then-miss")
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1,
        radius=20.0,
        thickness=5.0,
        material=IdealMaterial(n=1.5, k=1e-6),
        aperture=RadialAperture(r_max=3.0),
        is_stop=True,
    )
    optic.surfaces.add(index=2, radius=-4.0, thickness=10.0)
    optic.surfaces.add(index=3)
    return optic


def test_a_clipped_ray_that_misses_in_an_absorbing_glass_stays_dark(calls, monkeypatch):
    """RealRays.attenuate keeps i = 0 when a later factor is not finite."""

    def rays():
        n = 41
        return RealRays(
            np.zeros(n),
            np.linspace(-6.0, 6.0, n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.ones(n),
            np.ones(n),
            np.full(n, 0.55),
        )

    with_plan = _trace(_clip_then_miss, rays, True, monkeypatch, True)
    assert calls == [True]
    _assert_same(with_plan, _trace(_clip_then_miss, rays, True, monkeypatch, False))
    missed = np.abs(np.linspace(-6.0, 6.0, 41)) > 4.0
    assert np.isnan(with_plan["ray.x"][missed]).all()
    assert (with_plan["ray.i"][missed] == 0.0).all()


def test_object_surface_alone_records_the_rays(calls, monkeypatch):
    """A plan of an ObjectSurface only records the rays and keeps their arrays."""

    def build():
        optic = CookeTriplet()
        optic.surfaces._surfaces = optic.surfaces._surfaces[:1]
        return optic

    rays = _fan(30)
    arrays = {name: getattr(rays, name) for name in RAY_ATTRS}
    build().surfaces.trace(rays)
    assert calls == [True]
    for name, value in arrays.items():
        assert getattr(rays, name) is value, name
    with_plan = _trace(build, lambda: _fan(30), True, monkeypatch, True)
    _assert_same(with_plan, _trace(build, lambda: _fan(30), True, monkeypatch, False))


def test_input_arrays_are_not_written(calls):
    """The kernel writes new arrays and leaves the input arrays unchanged.

    When trace_with_plan returns False, the trace in parts (_chunked_trace)
    or the generic trace starts from these arrays. ``rays.x is not x`` shows
    that the plan traced the rays and gave them new arrays.
    """
    rays = _fan(500)
    names = ("x", "y", "z", "L", "M", "N", "i", "w", "opd")
    arrays = {name: getattr(rays, name) for name in names}
    before = {name: value.copy() for name, value in arrays.items()}
    CookeTriplet().surfaces.trace(rays)
    assert calls == [True]
    assert rays.x is not arrays["x"]
    for name, value in arrays.items():
        assert_bits_equal(value, before[name])


def test_a_change_between_traces_is_used(calls, monkeypatch):
    """The plan is built from the live objects at each trace."""

    def change(optic):
        surface = optic.surfaces.surfaces[2]
        surface.geometry.radius = be.array(-30.0)
        surface.geometry.cs.x = be.array(0.05)
        return optic

    optic = CookeTriplet()
    optic.surfaces.trace(_fan(50))
    first = optic.surfaces.surfaces[-1].y.copy()
    rays = _fan(50)
    change(optic).surfaces.trace(rays)
    assert calls == [True, True]
    assert not np.array_equal(first, optic.surfaces.surfaces[-1].y)
    generic = _trace(
        lambda: change(CookeTriplet()), lambda: _fan(50), True, monkeypatch, False
    )
    _assert_same(_state(optic, rays), generic)


def test_previous_records_are_released_before_the_kernel(calls, monkeypatch):
    """A trace does not hold two sets of records at its peak."""
    optic = CookeTriplet()
    optic.surfaces.trace(_fan(40))
    kernel = plan._kernel()
    sizes = []

    class Watched:
        """The kernel module, with a look at the records before each trace."""

        def __getattr__(self, name):
            return getattr(kernel, name)

        def trace_plan(self, *args):
            look()
            return kernel.trace_plan(*args)

        def trace_basic_plan(self, *args):
            look()
            return kernel.trace_basic_plan(*args)

    def look():
        for surface in optic.surfaces.surfaces:
            sizes.extend(getattr(surface, name).size for name in RECORD_ATTRS)

    monkeypatch.setattr(plan, "_kernel", Watched)
    optic.surfaces.trace(_fan(40))
    assert calls == [True, True]
    assert sizes and not any(sizes)


def test_skip_resets_the_records_of_the_skipped_surfaces(calls, monkeypatch):
    """The records of a previous trace do not stay on the skipped surfaces."""

    def run(enabled):
        monkeypatch.setattr(plan, "_ENABLED", enabled)
        optic = CookeTriplet()
        optic.surfaces.trace(_fan(40))
        rays = _fan(40)
        optic.surfaces.trace(rays, skip=2)
        return _state(optic, rays)

    with_plan = run(True)
    assert calls == [True, True]
    _assert_same(with_plan, run(False))


def test_no_surfaces_use_the_generic_trace(calls):
    """An empty group, or a skip of all surfaces, gives no tables."""
    rays = _fan(20)
    x = rays.x
    SurfaceGroup().trace(rays)
    optic = CookeTriplet()
    optic.surfaces.trace(rays, skip=len(optic.surfaces.surfaces))
    assert calls == [False, False]
    assert rays.x is x


class _Raising(BaseMaterial):
    """A material without an index for any wavelength."""

    def _cache_state(self):
        return ()

    def _calculate_n(self, wavelength, **kwargs):
        raise ValueError("no index for this wavelength")

    def _calculate_k(self, wavelength, **kwargs):
        return be.zeros_like(be.asarray(wavelength))


def test_a_material_error_comes_from_the_generic_trace(calls):
    optic = CookeTriplet()
    optic.surfaces.surfaces[2].material_post = _Raising()
    with pytest.raises(ValueError, match="no index"):
        optic.surfaces.trace(_fan(10))
    assert calls == [False]


class _ScalarIndex(BaseMaterial):
    """A material whose n(w) is a Python float.

    Without a cache state, a bundle of up to 128 rays (no evaluation epoch)
    gets the float itself, not an array.
    """

    def __init__(self, index):
        super().__init__()
        self.index = index

    def _calculate_n(self, wavelength, **kwargs):
        return self.index

    def _calculate_k(self, wavelength, **kwargs):
        return 0.0


def _index_where_pow_differs():
    """Return an index n for which (1 / n)**2 and (1 / n) * (1 / n) differ.

    The C function pow does not promise the rounding of u * u. If pow gives
    u * u for each candidate, the test still compares the bits, but cannot
    tell the two rules apart.
    """
    for k in range(100_000):
        index = 1.5 + k * 1e-7
        u = 1.0 / index
        if u**2 != u * u:
            return index
    return 1.5


def _scalar_index_singlet(index):
    """A singlet whose glass and object space give their index as a float."""
    optic = Optic()
    optic.surfaces.add(
        index=0, radius=be.inf, thickness=be.inf, material=_ScalarIndex(1.0)
    )
    optic.surfaces.add(
        index=1, radius=20.0, thickness=4.0, material=_ScalarIndex(index)
    )
    optic.surfaces.add(
        index=2, radius=-30.0, thickness=40.0, material=_ScalarIndex(1.0)
    )
    optic.surfaces.add(index=3)
    return optic


def test_scalar_indices_square_the_ratio_with_pow(calls, monkeypatch):
    """u**2 of two scalar indices is C pow, as in RealRays.refract.

    NumPy computes u**2 as u * u only for an array. At surfaces 1 and 2,
    both media return a Python float, so the generic trace calls pow.
    """
    index = _index_where_pow_differs()
    build = functools.partial(_scalar_index_singlet, index)
    with_plan = _trace(build, lambda: _fan(50), True, monkeypatch, True)
    assert calls == [True]
    _assert_same(with_plan, _trace(build, lambda: _fan(50), True, monkeypatch, False))
    kernel = plan._kernel()
    table = plan._build(build().surfaces.surfaces, _fan(50), 50, kernel).floats
    u = 1.0 / index
    assert table[1, kernel.INDEX_RATIO_SQUARED] == u**2


# Even aspheres: the Newton-Raphson iteration of NewtonRaphsonGeometry.


def _strong_asphere(coefficients=(-1e-2, 2e-5), max_iter=100):
    """A singlet whose first surface (at z = 3) is a strong even asphere.

    The polynomial bends the surface away from its base conic (R = 6,
    k = -0.5), so some steep rays hit the asphere but miss the base conic.
    """
    optic = Optic()
    optic.surfaces.add(index=0, z=0.0)
    optic.surfaces.add(index=1, z=3.0, material="N-BK7")
    optic.surfaces.add(index=2, z=8.0)
    surface = optic.surfaces.surfaces[1]
    surface.geometry = EvenAsphere(
        surface.geometry.cs,
        6.0,
        -0.5,
        max_iter=max_iter,
        coefficients=list(coefficients),
    )
    return optic


def _steep_fan(n, height, angle):
    """n rays from z = 0 in the y-z plane, with heights and angles up to these."""
    ang = np.linspace(-angle, angle, n)
    return RealRays(
        np.zeros(n),
        np.linspace(-height, height, n),
        np.zeros(n),
        np.zeros(n),
        np.sin(ang),
        np.cos(ang),
        np.ones(n),
        np.full(n, 0.55),
    )


def _at_strong_asphere(rays):
    """The rays in the frame of the asphere of ``_strong_asphere``."""
    rays.z = rays.z - 3.0
    return rays


def test_plane_seeded_rays_are_identical(calls, monkeypatch):
    """A ray that misses the base conic starts on the vertex plane z = 0.

    It is kept only if the iteration converges to a root in front of it.
    The fan has rays of both kinds.
    """

    def rays():
        return _steep_fan(400, 9.0, 1.2)

    geometry = _strong_asphere().surfaces.surfaces[1].geometry
    seed = StandardGeometry.distance(geometry, _at_strong_asphere(rays()))
    hit = geometry.distance(_at_strong_asphere(rays()))
    assert (np.isnan(seed) & np.isfinite(hit)).any()
    assert (np.isnan(seed) & np.isnan(hit)).any()

    with_plan = _trace(_strong_asphere, rays, True, monkeypatch, True)
    assert calls == [True]
    _assert_same(with_plan, _trace(_strong_asphere, rays, True, monkeypatch, False))


def _mirror_section():
    """A concave asphere mirror with an aperture off the axis.

    The base conic is a parabola (R = 20, k = -1). A ray along y at z = 5
    crosses it twice, near y = -14 and y = 14. The aperture holds only the
    second crossing, so it selects the seed of the iteration.
    """
    optic = Optic()
    optic.surfaces.add(index=0, z=0.0)
    optic.surfaces.add(
        index=1,
        z=0.0,
        surface_type="even_asphere",
        radius=20.0,
        conic=-1.0,
        coefficients=[1e-5],
        material="mirror",
        aperture=OffsetRadialAperture(r_max=4.0, offset_y=14.0),
    )
    optic.surfaces.add(index=2, z=30.0)
    return optic


def _side_fan():
    """60 rays from y = -20 at heights z = 3 to 6, nearly along the y axis."""
    n = 60
    ang = np.linspace(-0.05, 0.05, n)
    return RealRays(
        np.linspace(-1.0, 1.0, n),
        np.full(n, -20.0),
        np.linspace(3.0, 6.0, n),
        np.zeros(n),
        np.cos(ang),
        np.sin(ang),
        np.ones(n),
        np.full(n, 0.55),
    )


def test_the_aperture_selects_the_seed_of_an_asphere(calls, monkeypatch):
    """The seed is the root of the base conic that the aperture prefers.

    The precondition checks that the aperture changes the seed of some
    rays: without it, the seed would be the nearer crossing.
    """
    surface = _mirror_section().surfaces.surfaces[1]
    preferred = StandardGeometry.distance(
        surface.geometry, _side_fan(), aperture=surface.aperture
    )
    nearest = StandardGeometry.distance(surface.geometry, _side_fan())
    assert (preferred > nearest).any()

    with_plan = _trace(_mirror_section, _side_fan, True, monkeypatch, True)
    assert calls == [True]
    generic = _trace(_mirror_section, _side_fan, True, monkeypatch, False)
    _assert_same(with_plan, generic)


def _newton_fan():
    """701 rays; ray 350, in the second block, runs along the axis."""
    return _steep_fan(701, 5.0, 0.9)


def _solve(max_iter):
    """``_solve_distance_primal`` of the strong asphere for ``_newton_fan``."""
    geometry = _strong_asphere(max_iter=max_iter).surfaces.surfaces[1].geometry
    return geometry._solve_distance_primal(_at_strong_asphere(_newton_fan()))


def test_the_rays_of_a_block_take_different_numbers_of_steps():
    """The precondition of test_newton_steps_are_identical.

    In the second block of _newton_fan (rays 256 to 511), some rays converge
    at the seed, and one step and two steps give different distances.
    """
    block = slice(256, 512)
    at_seed = _solve(0).converged[block]
    assert at_seed.any()
    assert not at_seed.all()
    one, two = _solve(1).t[block], _solve(2).t[block]
    assert not np.array_equal(one, two, equal_nan=True)


@pytest.mark.parametrize(
    ("coefficients", "max_iter"),
    [
        pytest.param((), 100, id="no_coefficients"),
        pytest.param((-1e-2, 2e-5), 100, id="converged"),
        pytest.param((-1e-2, 2e-5), 2, id="two_steps"),
        pytest.param((-1e-2, 2e-5), 1, id="one_step"),
        pytest.param((-1e-2, 2e-5), 0, id="no_steps"),
    ],
)
def test_newton_steps_are_identical(calls, monkeypatch, coefficients, max_iter):
    """The rays of a block take different numbers of steps, up to max_iter.

    Without coefficients, the asphere is its base conic, and the kernel
    uses the conic normal.
    """
    build = functools.partial(_strong_asphere, coefficients, max_iter)
    with_plan = _trace(build, _newton_fan, True, monkeypatch, True)
    assert calls == [True]
    generic = _trace(build, _newton_fan, True, monkeypatch, False)
    _assert_same(with_plan, generic)
    if coefficients and max_iter < 100:
        # max_iter stops rays before they converge.
        converged = _trace(_strong_asphere, _newton_fan, True, monkeypatch, False)
        assert not np.array_equal(generic["s1.z"], converged["s1.z"], equal_nan=True)


@pytest.mark.parametrize(
    "coefficients",
    [
        pytest.param([2e-4, -3e-6, 4e-8], id="floats"),
        pytest.param([0, 0, 0], id="ints"),
        pytest.param([np.float64(2e-4), np.float64(-3e-6)], id="numpy_floats"),
        pytest.param([np.array(2e-4), np.array(-3e-6)], id="arrays_0d"),
        pytest.param(np.array([2e-4, -3e-6, 4e-8]), id="array"),
    ],
)
def test_coefficient_types_are_identical(calls, monkeypatch, coefficients):
    """Each form of the coefficients gives the bits of the generic trace.

    The builder computes i * C_i with the objects of the geometry, as
    EvenAsphere._surface_normal does. The optimizer stores the coefficients
    in an array when it pads them.
    """

    def build():
        optic = CookeTriplet()
        surface = optic.surfaces.surfaces[1]
        geometry = surface.geometry
        surface.geometry = EvenAsphere(geometry.cs, geometry.radius)
        surface.geometry.coefficients = coefficients
        return optic

    with_plan = _trace(build, lambda: _fan(300), True, monkeypatch, True)
    assert calls == [True]
    _assert_same(with_plan, _trace(build, lambda: _fan(300), True, monkeypatch, False))


def test_tolerance_floor_uses_the_generic_trace(calls, floor_checks, monkeypatch):
    """Below the round-off floor, the tolerance depends on the seeds of all rays.

    The kernel runs, then _floor_applies gives True, and the generic trace
    traces the rays.
    """

    def build():
        optic = _asphere_triplet()
        optic.surfaces.surfaces[4].geometry.tol = 1e-30
        return optic

    with_plan = _trace(build, lambda: _fan(100), True, monkeypatch, True)
    assert calls == [False]
    assert floor_checks == [True]
    _assert_same(with_plan, _trace(build, lambda: _fan(100), True, monkeypatch, False))


def _far_asphere():
    """A sphere in air (R = 5), then an even asphere with tol = 1e-13 at 100 mm.

    A ray more than 5 mm from the axis misses the sphere, so its seed at the
    asphere is NaN.
    """
    optic = Optic()
    optic.surfaces.add(index=0, z=0.0)
    optic.surfaces.add(index=1, z=1.0, radius=5.0)
    optic.surfaces.add(
        index=2,
        z=101.0,
        surface_type="even_asphere",
        radius=50.0,
        coefficients=[1e-6],
        tol=1e-13,
    )
    optic.surfaces.add(index=3, z=110.0)
    return optic


@pytest.mark.parametrize(
    ("first_height", "floor_applies"),
    [
        pytest.param(4.0, True, id="finite_seeds"),
        pytest.param(6.0, False, id="nan_seed"),
    ],
)
def test_a_nan_seed_gives_the_lowest_floor(
    calls, floor_checks, monkeypatch, first_height, floor_applies
):
    """The floor 8 * eps * max(1, max|t|) when one seed is NaN.

    The seeds at the asphere are near 100 mm, so the floor is about
    1.8e-13, above tol = 1e-13. NumPy's max of the seeds is NaN if one seed
    is NaN, and Python's max(1.0, nan) is 1.0, so the floor is then 8 * eps,
    and the plan traces. The first ray, in the first of three blocks, has
    the NaN seed, so the seed summary must keep it over the blocks. A spy
    on _effective_tolerance shows the decision of the generic solve.
    """

    def rays():
        n = 600
        heights = np.linspace(-4.0, 4.0, n)
        heights[0] = first_height
        return RealRays(
            np.zeros(n),
            heights,
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.ones(n),
            np.ones(n),
            np.full(n, 0.55),
        )

    tolerances = []
    effective_tolerance = newton_raphson._effective_tolerance

    def spy(tol, t):
        tolerances.append(effective_tolerance(tol, t))
        return tolerances[-1]

    monkeypatch.setattr(newton_raphson, "_effective_tolerance", spy)
    generic = _trace(_far_asphere, rays, True, monkeypatch, False)
    assert (tolerances[0] > 1e-13) == floor_applies
    calls.clear()
    with_plan = _trace(_far_asphere, rays, True, monkeypatch, True)
    assert floor_checks == [floor_applies]
    assert calls == [not floor_applies]
    _assert_same(with_plan, generic)


def test_kernel_thresholds_match_newton_raphson():
    """The kernel repeats _DENOM_EPS_MULTIPLIER of newton_raphson.py.

    Numba compiles a global as a constant, and its cache does not see a
    change in another module, so the kernel cannot import the value.
    """
    kernel = plan._kernel()
    assert kernel._DENOM_EPS_MULTIPLIER == newton_raphson._DENOM_EPS_MULTIPLIER
    nz = np.zeros(1)
    assert newton_raphson._nz_threshold(nz) == kernel._DENOM_TAU
    assert newton_raphson._denominator_threshold(nz) == kernel._DENOM_TAU


def _odd_asphere(optic):
    """An OddAsphere: a subclass of EvenAsphere with another sag."""
    surface = optic.surfaces.surfaces[1]
    geometry = surface.geometry
    surface.geometry = OddAsphere(
        geometry.cs, geometry.radius, coefficients=[1e-6, 1e-7]
    )


def _asphere_with(**attributes):
    """Return a change that makes surface 1 an EvenAsphere with ``attributes``.

    The attributes are set after the constructor, which makes a list of the
    coefficients.
    """

    def change(optic):
        surface = optic.surfaces.surfaces[1]
        geometry = surface.geometry
        surface.geometry = EvenAsphere(
            geometry.cs, geometry.radius, coefficients=[1e-6]
        )
        for name, value in attributes.items():
            setattr(surface.geometry, name, value)

    return change


def _coating(optic):
    optic.surfaces.surfaces[1].set_fresnel_coating()


def _reference_frame(optic):
    from optiland.coordinate_system import CoordinateSystem

    optic.surfaces.surfaces[2].geometry.cs.reference_cs = CoordinateSystem()


def _elliptical_aperture(optic):
    optic.surfaces.surfaces[2].aperture = EllipticalAperture(a=5.0, b=4.0)


class _MyAperture(RadialAperture):
    """A subclass of RadialAperture: it could change contains()."""


def _aperture_subclass(optic):
    optic.surfaces.surfaces[2].aperture = _MyAperture(r_max=5.0)


def _array_aperture_radius(optic):
    optic.surfaces.surfaces[2].aperture = RadialAperture(r_max=np.array([5.0]))


def _bsdf(optic):
    from optiland.scatter import LambertianBSDF

    optic.surfaces.surfaces[2].interaction_model.bsdf = LambertianBSDF()


class _MySurface(Surface):
    """A subclass of Surface: it could change a step of the trace."""


def _surface_subclass(optic):
    optic.surfaces.surfaces[2].__class__ = _MySurface


def _shared_interaction_model(optic):
    surfaces = optic.surfaces.surfaces
    surfaces[3].interaction_model = surfaces[2].interaction_model


class _MyPropagation(HomogeneousPropagation):
    """A subclass of HomogeneousPropagation: it could change the path."""


def _other_propagation(optic):
    material = optic.surfaces.surfaces[2].material_post
    material.propagation_model = _MyPropagation(material)


def _grin_after(optic):
    from optiland.materials import GradientMaterial

    optic.surfaces.surfaces[2].material_post = GradientMaterial(n0=1.6, nr2=-1e-3)


def _float32_radius(optic):
    geometry = optic.surfaces.surfaces[2].geometry
    geometry.radius = geometry.radius.astype(np.float32)


def _scalar_radius(optic):
    geometry = optic.surfaces.surfaces[2].geometry
    geometry.radius = np.float64(geometry.radius)


@pytest.mark.parametrize(
    "change",
    [
        _odd_asphere,
        pytest.param(_asphere_with(coefficients=(1e-6,)), id="coefficient_tuple"),
        pytest.param(
            _asphere_with(coefficients=[np.float32(1e-6)]), id="float32_coefficient"
        ),
        pytest.param(_asphere_with(tol=np.float32(1e-10)), id="float32_tol"),
        pytest.param(_asphere_with(max_iter=np.int64(100)), id="numpy_max_iter"),
        pytest.param(_asphere_with(max_iter=2**63), id="max_iter_above_int64"),
        _coating,
        _reference_frame,
        _elliptical_aperture,
        _aperture_subclass,
        _array_aperture_radius,
        _bsdf,
        _surface_subclass,
        _shared_interaction_model,
        _other_propagation,
        _grin_after,
        _float32_radius,
        _scalar_radius,
    ],
)
def test_systems_outside_the_domain_use_the_generic_trace(calls, floor_checks, change):
    """Each limit of the domain on the developer page gives the generic trace.

    The build stops, so the kernel does not run, and the tolerance floor of
    an even asphere is not checked.
    """
    optic = CookeTriplet()
    change(optic)
    rays = _fan(20)
    x = rays.x
    optic.surfaces.trace(rays)
    assert calls == [False]
    assert floor_checks == []
    assert rays.x is not x


def _polarized(n):
    return PolarizedRays(*(np.zeros(n),) * 5, np.ones(n), np.ones(n), np.full(n, 0.55))


def _with(rays, **arrays):
    for name, value in arrays.items():
        setattr(rays, name, value)
    return rays


@pytest.mark.parametrize(
    "rays",
    [
        lambda: _with(_fan(20), is_normalized=False),
        lambda: _with(_fan(20), x=np.zeros(20, dtype=np.float32)),
        lambda: _with(_fan(20), i=np.ones(19)),
        lambda: _with(_fan(20), x=np.zeros((20, 1))),
        lambda: _fan(0),
        lambda: _polarized(20),
    ],
    ids=["unnormalized", "float32", "other_length", "2d", "no_rays", "polarized"],
)
def test_rays_outside_the_domain_use_the_generic_trace(rays):
    """The domain has normalized RealRays with 1-D float64 arrays of one length."""
    surfaces = CookeTriplet().surfaces.surfaces
    assert plan.trace_with_plan(surfaces, rays(), True) is False


@pytest.mark.parametrize("record", [True, False])
def test_absorption_per_ray_is_identical(calls, monkeypatch, record):
    """With three wavelengths, alpha = 4 pi k / lambda differs from ray to ray.

    The IdealMaterial has one index for all rays, so only alpha has a row of
    per-ray values.
    """

    def rays():
        return _fan(300, wavelengths=(0.4861, 0.5876, 0.6563))

    with_plan = _trace(_absorbing_singlet, rays, record, monkeypatch, True)
    assert calls == [True]
    generic = _trace(_absorbing_singlet, rays, record, monkeypatch, False)
    _assert_same(with_plan, generic)


def test_one_row_per_glass():
    """Each glass is the medium after one surface and before the next.

    Its indices get one row of per-ray values, not one row per use, and so
    does its absorption coefficient.
    """
    surfaces = CementedAchromat().surfaces.surfaces
    rays = _fan(50, wavelengths=(0.4861, 0.5876, 0.6563))
    kernel = plan._kernel()
    built = plan._build(surfaces, rays, 50, kernel)
    columns = [kernel.N1_ROW, kernel.N2_ROW, kernel.OPD_INDEX_ROW]
    index_rows = {int(row) for row in built.ints[:, columns].ravel()} - {-1}
    alpha_rows = set(built.ints[:, kernel.ALPHA_ROW].tolist()) - {-1}
    # The achromat has two glasses with k > 0. Air has one index for all
    # rays and does not absorb.
    assert len(index_rows) == 2
    assert len(alpha_rows) == 2
    assert index_rows | alpha_rows == set(range(built.ray_values.shape[0]))


@pytest.fixture
def kernel_calls(monkeypatch):
    """The names of the kernels that the traces call, in order."""
    names = []
    kernel = plan._kernel()

    def spy(name):
        original = getattr(kernel, name)

        def counted(*args):
            names.append(name)
            return original(*args)

        return counted

    for name in ("trace_plan", "trace_basic_plan"):
        monkeypatch.setattr(kernel, name, spy(name))
    return names


def _has_asphere(optic):
    """Return True if a surface of ``optic`` is an even asphere."""
    return any(type(s.geometry) is EvenAsphere for s in optic.surfaces.surfaces)


def test_each_plan_gets_its_kernel(calls, kernel_calls, monkeypatch):
    """trace_basic_plan for a basic plan, trace_plan for the other plans.

    A basic plan has no per-ray values and no even asphere: here the Cooke
    triplet with one wavelength. Three wavelengths give per-ray values.
    """
    for wavelengths in ((0.55,), (0.48, 0.55, 0.65)):
        _trace(CookeTriplet, lambda w=wavelengths: _fan(50, w), True, monkeypatch, True)
    _trace(_asphere_triplet, lambda: _fan(50), True, monkeypatch, True)
    assert calls == [True, True, True]
    assert kernel_calls == ["trace_basic_plan", "trace_plan", "trace_plan"]


# The paraxial ray aiming warns that it ignores the decenters of "decentered".
@pytest.mark.filterwarnings("ignore:paraxial ray tracing")
@pytest.mark.parametrize("name", sorted(SYSTEMS))
def test_both_kernels_give_the_bits_of_a_basic_plan(
    calls, kernel_calls, monkeypatch, name
):
    """trace_plan and trace_basic_plan compile from one loop nest.

    A trace with one wavelength through a system without an even asphere has
    a basic plan, which trace_basic_plan traces. Here trace_plan traces it
    too, with empty tables of per-ray values, coefficients and seeds, and
    must give the same bits. A system with an even asphere has no basic
    plan, so trace_plan traces it both times.
    """
    kernel = plan._kernel()

    def run():
        optic = SYSTEMS[name]()
        rays = optic.trace(
            Hx=0.0,
            Hy=1.0,
            wavelength=optic.primary_wavelength,
            num_rays=9,
            distribution="hexapolar",
            record=True,
        )
        return _state(optic, rays)

    def full_kernel(rays_in, ints, floats, exponents, clipped, *rest):
        no_rows = plan._NO_ROWS
        tables = (no_rows, no_rows, no_rows, exponents, clipped, no_rows)
        return kernel.trace_plan(rays_in, ints, floats, *tables, *rest)

    basic = run()
    monkeypatch.setattr(kernel, "trace_basic_plan", full_kernel)
    full = run()
    first = "trace_plan" if _has_asphere(SYSTEMS[name]()) else "trace_basic_plan"
    assert calls == [True, True]
    assert kernel_calls == [first, "trace_plan"]
    _assert_same(basic, full)


def test_torch_backend_uses_the_generic_trace(calls):
    """NumPy only: with Torch, trace_with_plan returns False."""
    pytest.importorskip("torch")
    be.set_backend("torch")
    try:
        optic = CookeTriplet()
        optic.trace(Hx=0.0, Hy=1.0, wavelength=0.55, num_rays=3)
    finally:
        be.set_backend("numpy")
    assert calls == [False]


def test_float32_precision_uses_the_generic_trace(calls):
    """With float32 precision, be.array rounds the translations of the trace."""
    optic = _decentered()
    be.set_precision("float32")
    try:
        optic.surfaces.trace(_fan(20))
    finally:
        be.set_precision("float64")
    assert calls == [False]


def test_numba_without_jit_uses_the_generic_trace(calls, monkeypatch):
    """With NUMBA_DISABLE_JIT=1, the kernel would run as Python code."""
    monkeypatch.setattr(numba.config, "DISABLE_JIT", True)
    CookeTriplet().surfaces.trace(_fan(20))
    assert calls == [False]


def test_cleared_records_are_the_arrays_of_a_reset_surface():
    """_CLEARED names each array of a surface after Surface.reset.

    The plan releases and sets these records. A missing name would keep a
    record of the previous trace.
    """
    for surface in CookeTriplet().surfaces.surfaces:
        surface.reset()
        arrays = {
            name
            for name, value in vars(surface).items()
            if isinstance(value, np.ndarray)
        }
        assert arrays == set(plan._CLEARED), type(surface).__name__
