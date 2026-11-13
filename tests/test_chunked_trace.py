"""Tests for the trace of large NumPy bundles in parts.

A trace in parts is a fast path, so it must give the bits of the trace of
the whole bundle. The tests compare bits with ``assert_bits_equal``. The
trace in parts exists only on the NumPy backend, so the tests run on NumPy.
They set ``_CHUNK_SIZE`` and ``_MIN_RAYS`` to small values, so that small
bundles are traced in many parts.

Manuel Fragata Mendes, 2026
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
import optiland.surfaces._chunked_trace as chunked
from optiland.coatings import (
    CartesianJonesCoating,
    PolarizerCoating,
    RetarderCoating,
    SimpleCoating,
    ThinFilmCoating,
)
from optiland.geometries import StandardGeometry
from optiland.materials import IdealMaterial
from optiland.materials.base import BaseMaterial
from optiland.optic import Optic
from optiland.physical_apertures import (
    EllipticalAperture,
    OffsetRadialAperture,
    RadialAperture,
    RectangularAperture,
)
from optiland.rays import PolarizedRays, RealRays
from optiland.samples.objectives import CookeTriplet, DoubleGauss
from optiland.samples.telescopes import HubbleTelescope
from optiland.surfaces import ImageSurface

from .utils import assert_bits_equal

RAY_ATTRS = ("x", "y", "z", "L", "M", "N", "i", "w", "opd", "L0", "M0", "N0")
# The records of a surface. The list is the test's own, not _RECORDED, so
# that a name missing from _RECORDED makes the tests fail.
RECORD_ATTRS = ("x", "y", "z", "L", "M", "N", "intensity", "aoi", "opd")
PART = 64  # rays per part in these tests


@pytest.fixture
def small_parts(monkeypatch):
    """Trace bundles of more than 64 rays in parts of 64 rays."""
    monkeypatch.setattr(chunked, "_CHUNK_SIZE", PART)
    monkeypatch.setattr(chunked, "_MIN_RAYS", PART + 1)


@pytest.fixture
def calls(monkeypatch):
    """The results of the calls of trace_in_parts, in order.

    SurfaceGroup looks up trace_in_parts in its module at each trace, so the
    spy replaces the module attribute.
    """
    results = []
    original = chunked.trace_in_parts

    def spy(*args, **kwargs):
        results.append(original(*args, **kwargs))
        return results[-1]

    monkeypatch.setattr(chunked, "trace_in_parts", spy)
    return results


@pytest.fixture(autouse=True)
def parts_on(monkeypatch):
    """Trace in parts on each platform; outside Windows they are off."""
    monkeypatch.setattr(chunked, "_ENABLED", True)


def _fan(n, wavelengths=(0.55,), angle=0.05, height=2.0):
    """A fan of n rays in the y-z plane, starting at z = 0.

    The rays have heights from -height to height and angles from -angle to
    angle (radians). np.resize repeats the wavelengths from ray to ray, so
    each part has all of them.
    """
    ang = np.linspace(-angle, angle, n)
    return RealRays(
        np.zeros(n),
        np.linspace(-height, height, n),
        np.zeros(n),
        np.zeros(n),
        np.sin(ang),
        np.cos(ang),
        np.ones(n),
        np.resize(np.asarray(wavelengths, dtype=float), n),
    )


def _spot(n, height=2.0, angle=0.05, wavelength=0.55):
    """n rays from random points of a disk at z = 0, in random directions.

    The disk has the radius ``height``. The direction angles in x and in y
    are up to ``angle`` (radians). Surfaces without rotational symmetry need
    rays in x and in y.
    """
    rng = np.random.default_rng(1)
    r = height * np.sqrt(rng.uniform(0.0, 1.0, n))
    phi = rng.uniform(0.0, 2 * np.pi, n)
    L = np.sin(rng.uniform(-angle, angle, n))
    M = np.sin(rng.uniform(-angle, angle, n))
    return RealRays(
        r * np.cos(phi),
        r * np.sin(phi),
        np.zeros(n),
        L,
        M,
        np.sqrt(1.0 - L**2 - M**2),
        np.ones(n),
        np.full(n, wavelength),
    )


def _singlet(**front):
    """An N-BK7 singlet with the stop on its front surface."""
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(index=1, thickness=5.0, material="N-BK7", is_stop=True, **front)
    optic.surfaces.add(index=2, radius=-60.0, thickness=40.0)
    optic.surfaces.add(index=3)
    return optic


def _thin_lens():
    """A thin lens (a paraxial surface) and a plano-convex lens behind it."""
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1, surface_type="paraxial", f=50.0, thickness=20.0, is_stop=True
    )
    optic.surfaces.add(index=2, radius=30.0, thickness=4.0, material="N-BK7")
    optic.surfaces.add(index=3, thickness=20.0)
    optic.surfaces.add(index=4)
    return optic


def _image_surface():
    """The Cooke triplet with an ImageSurface as its last surface.

    The surface factory never makes an ImageSurface, so the test puts one in
    the list of surfaces.
    """
    optic = CookeTriplet()
    surfaces = optic.surfaces._surfaces
    last = surfaces[-1]
    surfaces[-1] = ImageSurface(surfaces[-2], last.geometry, last.material_post)
    return optic


def _apertures():
    """The Cooke triplet with one aperture of each allowed type."""
    optic = CookeTriplet()
    surfaces = optic.surfaces.surfaces
    surfaces[1].aperture = RadialAperture(r_max=5.0, r_min=0.5)
    surfaces[2].aperture = OffsetRadialAperture(r_max=5.0, offset_x=0.5)
    surfaces[3].aperture = EllipticalAperture(a=5.0, b=4.0, offset_y=-0.3)
    surfaces[4].aperture = RectangularAperture(-4.0, 3.5, -3.0, 4.5)
    return optic


def _coatings():
    """The Cooke triplet with one coating of each allowed type."""
    optic = CookeTriplet()
    surfaces = optic.surfaces.surfaces
    surfaces[1].interaction_model.coating = SimpleCoating(0.9, 0.05)
    surfaces[2].set_fresnel_coating()
    surfaces[3].interaction_model.coating = ThinFilmCoating(
        surfaces[3].material_pre,
        surfaces[3].material_post,
        layers=[(IdealMaterial(n=1.38), 100.0, "MgF2")],
    )
    surfaces[4].interaction_model.coating = PolarizerCoating()
    surfaces[5].interaction_model.coating = RetarderCoating(retardance=0.5)
    surfaces[6].interaction_model.coating = CartesianJonesCoating()
    return optic


# One system for each allowed type, with rays that reach its surfaces. Plane,
# StandardGeometry, Surface, ObjectSurface, RefractiveReflectiveModel and
# HomogeneousPropagation are in each system.
SYSTEMS = {
    "cooke": (CookeTriplet, lambda n: _spot(n, height=6.0)),
    "double_gauss": (DoubleGauss, lambda n: _spot(n, height=10.0)),
    "hubble_mirrors": (
        HubbleTelescope,
        lambda n: _spot(n, height=1000.0, angle=0.001),
    ),
    "even_asphere": (
        lambda: _singlet(
            surface_type="even_asphere",
            radius=40.0,
            conic=-0.5,
            coefficients=[1e-5, -2e-8],
        ),
        lambda n: _spot(n, height=4.0),
    ),
    "odd_asphere": (
        lambda: _singlet(
            surface_type="odd_asphere", radius=40.0, coefficients=[1e-4, 1e-5]
        ),
        lambda n: _spot(n, height=4.0),
    ),
    "polynomial": (
        lambda: _singlet(
            surface_type="polynomial",
            radius=40.0,
            coefficients=[[0.0, 0.0, 1e-4], [0.0, 2e-5, 0.0], [3e-4, 0.0, 0.0]],
        ),
        lambda n: _spot(n, height=4.0),
    ),
    "biconic": (
        lambda: _singlet(
            surface_type="biconic",
            radius_x=40.0,
            radius_y=60.0,
            conic_x=-0.3,
            conic_y=0.2,
        ),
        lambda n: _spot(n, height=4.0),
    ),
    "toroidal": (
        lambda: _singlet(
            surface_type="toroidal",
            radius_x=100.0,
            radius_y=50.0,
            conic=-0.5,
            toroidal_coeffs_poly_y=[1e-3, 2e-6],
        ),
        lambda n: _spot(n, height=4.0),
    ),
    "thin_lens": (_thin_lens, lambda n: _spot(n, height=4.0)),
    "image_surface": (_image_surface, lambda n: _spot(n, height=6.0)),
    "apertures": (_apertures, lambda n: _spot(n, height=6.0)),
    "coatings": (_coatings, lambda n: _spot(n, height=6.0)),
}


def _trace(optic, rays, record, monkeypatch, parts):
    """Trace ``rays`` through ``optic``, in parts or as a whole bundle.

    Returns:
        The arrays of the traced rays and the records of the surfaces.
    """
    if not parts:
        # No bundle has 10**12 rays, so each bundle is traced as a whole.
        monkeypatch.setattr(chunked, "_MIN_RAYS", 10**12)
    optic.surfaces.trace(rays, record=record)
    out = {f"ray.{a}": getattr(rays, a) for a in RAY_ATTRS}
    for k, surface in enumerate(optic.surfaces.surfaces):
        for a in RECORD_ATTRS:
            out[f"s{k}.{a}"] = getattr(surface, a)
    return out


def _assert_same(a, b):
    """Assert that two trace results hold the same arrays, bit for bit.

    The sign bit of a NaN is not compared. The parts pass read-only views to
    the compiled conic kernel, so they run another compiled version of it
    than the whole bundle, and compiled code can give a NaN another sign bit.
    On the Linux CI, a ray that misses got a NaN with another sign bit in a
    part. The positions of the NaNs and all other bits must be the same.
    """
    assert a.keys() == b.keys()
    for key in a:
        if a[key] is None or b[key] is None:
            assert a[key] is None and b[key] is None, key
        else:
            assert_bits_equal(a[key], b[key], ignore_nan_sign=True)


def test_off_traces_the_whole_bundle(small_parts, calls, monkeypatch):
    """With _ENABLED False, as on Linux and macOS, the whole bundle is traced."""
    monkeypatch.setattr(chunked, "_ENABLED", False)
    build, rays = SYSTEMS["cooke"]
    n = 3 * PART + 7
    off = _trace(build(), rays(n), True, monkeypatch, parts=True)
    assert calls == [False]
    _assert_same(off, _trace(build(), rays(n), True, monkeypatch, parts=False))


@pytest.mark.parametrize("record", [True, False])
@pytest.mark.parametrize("name", sorted(SYSTEMS))
def test_each_allowed_type_is_bit_identical(
    small_parts, calls, monkeypatch, name, record
):
    """Each system is traced in parts and gives the bits of the whole bundle.

    Three parts of 64 rays and one part of 7 rays.
    """
    build, rays = SYSTEMS[name]
    n = 3 * PART + 7
    in_parts = _trace(build(), rays(n), record, monkeypatch, parts=True)
    assert calls == [True]
    whole = _trace(build(), rays(n), record, monkeypatch, parts=False)
    _assert_same(in_parts, whole)


@pytest.mark.parametrize("record", [True, False])
@pytest.mark.parametrize("n", [65, 128, 129, 1000])
@pytest.mark.parametrize("build", [CookeTriplet, DoubleGauss])
def test_parts_of_any_length_are_bit_identical(
    small_parts, calls, monkeypatch, build, n, record
):
    """A last part of one ray, of a full part or of a few rays is exact.

    The rays repeat the wavelengths of the system, so each part has several.
    """
    wavelengths = [w.value for w in build().wavelengths.wavelengths]
    in_parts = _trace(build(), _fan(n, wavelengths), record, monkeypatch, True)
    assert calls == [True]
    whole = _trace(build(), _fan(n, wavelengths), record, monkeypatch, False)
    _assert_same(in_parts, whole)


@pytest.mark.parametrize("build", [CookeTriplet, DoubleGauss])
def test_parts_with_one_wavelength_are_bit_identical(
    small_parts, calls, monkeypatch, build
):
    """A part with one wavelength gives the bits of a polychromatic bundle.

    Each part of 64 rays has one wavelength, so a material evaluates it at
    one element and broadcasts the result. The whole bundle has three
    wavelengths, so a material evaluates it element by element.
    """

    def rays():
        r = _fan(3 * PART)
        r.w = np.repeat([0.4861, 0.5876, 0.6563], PART)
        return r

    in_parts = _trace(build(), rays(), True, monkeypatch, parts=True)
    assert calls == [True]
    whole = _trace(build(), rays(), True, monkeypatch, parts=False)
    _assert_same(in_parts, whole)


def test_rays_that_miss_are_bit_identical(small_parts, calls, monkeypatch):
    """Rays far outside the lenses give NaN and zero intensity."""
    in_parts = _trace(
        CookeTriplet(), _fan(300, height=40.0), True, monkeypatch, parts=True
    )
    assert calls == [True]
    assert np.isnan(in_parts["ray.x"]).any()
    whole = _trace(
        CookeTriplet(), _fan(300, height=40.0), True, monkeypatch, parts=False
    )
    _assert_same(in_parts, whole)


def test_input_arrays_are_not_written(small_parts, calls):
    """A trace in parts does not write into the arrays of its input rays.

    When a part cannot be used, the whole bundle is traced again from the
    input, so the input must be unchanged.
    """
    rays = _fan(500)
    names = ("x", "y", "z", "L", "M", "N", "i", "w", "opd")
    arrays = {a: getattr(rays, a) for a in names}
    before = {a: value.copy() for a, value in arrays.items()}
    CookeTriplet().surfaces.trace(rays)
    assert calls == [True]
    for a, value in arrays.items():
        assert_bits_equal(value, before[a])


def test_an_in_place_write_traces_the_whole_bundle(small_parts, calls, monkeypatch):
    """A part holds read-only views, so an in-place write ends the parts.

    The whole bundle is then traced from the unchanged input, with the
    in-place write of the generic trace.
    """
    optic = CookeTriplet()
    surface = optic.surfaces.surfaces[1]
    original = type(surface).trace

    def writing(self, rays, record=True):
        if self is surface:
            rays.i *= 1.0  # an in-place write into an input array
        return original(self, rays, record=record)

    monkeypatch.setattr(type(surface), "trace", writing)
    result = _trace(optic, _fan(200), True, monkeypatch, parts=True)
    assert calls == [False]
    whole = _trace(CookeTriplet(), _fan(200), True, monkeypatch, parts=False)
    _assert_same(result, whole)


@pytest.mark.filterwarnings("ignore:invalid value encountered in sqrt")
def test_newton_raphson_decisions_of_the_parts_are_exact(
    small_parts, calls, monkeypatch
):
    """The parts take other plane-seed and early-exit decisions than the
    whole bundle, and give its bits.

    The outer rays of the fan miss the base sphere, so they have no conic
    seed and are seeded on the vertex plane. The part in the middle has no
    such ray. The short last part stops the iteration earlier than the
    others. Both outcomes of each decision give each ray the same bits.
    """
    from optiland.geometries import EvenAsphere
    from optiland.geometries.newton_raphson import NewtonRaphsonGeometry

    seeded, steps = [], []
    distance = StandardGeometry.distance
    solve = NewtonRaphsonGeometry._solve_distance_primal

    def seed_spy(self, rays, aperture=None):
        t = distance(self, rays, aperture=aperture)
        if type(self) is EvenAsphere:
            seeded.append(bool(np.isnan(t).any()))
        return t

    def solve_spy(self, rays, aperture=None):
        result = solve(self, rays, aperture=aperture)
        steps.append(result.iterations)
        return result

    monkeypatch.setattr(StandardGeometry, "distance", seed_spy)
    monkeypatch.setattr(NewtonRaphsonGeometry, "_solve_distance_primal", solve_spy)

    def build():
        return _singlet(
            surface_type="even_asphere", radius=10.0, coefficients=[1e-3, -2e-5]
        )

    def rays():
        return _fan(3 * PART + 7, angle=0.0, height=12.0)

    in_parts = _trace(build(), rays(), True, monkeypatch, parts=True)
    assert calls == [True]
    assert True in seeded and False in seeded
    assert len(set(steps)) > 1
    whole = _trace(build(), rays(), True, monkeypatch, parts=False)
    _assert_same(in_parts, whole)


def test_newton_tolerance_floor_traces_the_whole_bundle(small_parts):
    """A tolerance below the round-off floor makes the floor depend on max|t|."""
    from optiland.geometries import EvenAsphere

    optic = CookeTriplet()
    surface = optic.surfaces.surfaces[1]
    geometry = surface.geometry
    surface.geometry = EvenAsphere(
        geometry.cs, geometry.radius, conic=0.0, coefficients=[1e-6], tol=1e-30
    )
    surfaces = optic.surfaces.surfaces
    rays = _fan(500)
    optic.surfaces.reset()
    assert chunked.trace_in_parts(surfaces, rays, True) is False
    assert_bits_equal(rays.x, _fan(500).x)


class _BandAbsorber(BaseMaterial):
    """n = 1.5; k > 0 only above 0.6 um."""

    def _cache_state(self):
        return ()

    def _calculate_n(self, wavelength, **kwargs):
        return be.ones_like(be.asarray(wavelength)) * 1.5

    def _calculate_k(self, wavelength, **kwargs):
        w = be.asarray(wavelength)
        return be.where(w > 0.6, 1e-6, 0.0)


def _absorbing_optic():
    """A singlet of _BandAbsorber."""
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1, radius=50.0, thickness=5.0, material=_BandAbsorber(), is_stop=True
    )
    optic.surfaces.add(index=2, radius=-50.0, thickness=45.0)
    optic.surfaces.add(index=3)
    return optic


def test_all_parts_absorb(small_parts, calls, monkeypatch):
    """All parts take the absorption decision, so the parts are used.

    Some rays miss the lens, so their path length is NaN in the absorbing
    medium.
    """

    def rays():
        return _fan(200, wavelengths=(0.65,), height=60.0)

    in_parts = _trace(_absorbing_optic(), rays(), True, monkeypatch, parts=True)
    assert calls == [True]
    assert np.isnan(in_parts["ray.i"]).any()
    whole = _trace(_absorbing_optic(), rays(), True, monkeypatch, parts=False)
    _assert_same(in_parts, whole)


def test_different_absorption_decisions_trace_the_whole_bundle(
    small_parts, calls, monkeypatch
):
    """The first parts do not absorb, the last part does."""

    def rays():
        r = _fan(200, height=8.0)
        r.w = np.where(np.arange(200) < 150, 0.55, 0.65)
        return r

    result = _trace(_absorbing_optic(), rays(), True, monkeypatch, parts=True)
    assert calls == [False]
    whole = _trace(_absorbing_optic(), rays(), True, monkeypatch, parts=False)
    _assert_same(result, whole)


def test_an_error_in_a_part_is_raised_by_the_whole_trace(small_parts, monkeypatch):
    """After an error in a part, the generic trace of the whole bundle runs.

    The caller then gets the error of the generic trace.
    """
    optic = CookeTriplet()
    surface = optic.surfaces.surfaces[2]
    original = type(surface).trace
    seen = []

    def failing(self, rays, record=True):
        seen.append(rays.x.shape[0])
        if self is surface:
            raise ValueError("bad ray")
        return original(self, rays, record=record)

    monkeypatch.setattr(type(surface), "trace", failing)
    with pytest.raises(ValueError, match="bad ray"):
        optic.surfaces.trace(_fan(500))
    # One part, then the whole bundle.
    assert PART in seen
    assert 500 in seen


def _record_as_float32(surface, rays):
    surface.opd = surface.opd.astype(np.float32)


def _record_one_short(surface, rays):
    surface.opd = surface.opd[:-1]


def _one_more_ray_array(surface, rays):
    rays.extra = np.zeros(rays.x.shape[0])


@pytest.mark.parametrize(
    "change", [_record_as_float32, _record_one_short, _one_more_ray_array]
)
def test_a_part_with_other_arrays_traces_the_whole_bundle(
    small_parts, calls, monkeypatch, change
):
    """The whole-bundle arrays are filled only from parts of the same arrays."""
    optic = CookeTriplet()
    surface = optic.surfaces.surfaces[2]
    original = type(surface).trace
    seen = []

    def changing(self, rays, record=True):
        out = original(self, rays, record=record)
        if self is surface:
            seen.append(rays.x.shape[0])
            if len(seen) == 2:  # the second part
                change(self, rays)
        return out

    monkeypatch.setattr(type(surface), "trace", changing)
    result = _trace(optic, _fan(200), True, monkeypatch, parts=True)
    assert calls == [False]
    assert seen == [PART, PART, 200]
    monkeypatch.setattr(type(surface), "trace", original)
    whole = _trace(CookeTriplet(), _fan(200), True, monkeypatch, parts=False)
    _assert_same(result, whole)


class _Subarray(np.ndarray):
    """An ndarray subclass: not split into parts."""


def _polarized(n):
    return PolarizedRays(
        np.zeros(n),
        np.linspace(-2, 2, n),
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
        np.ones(n),
        np.ones(n),
        np.full(n, 0.55),
    )


def _with(rays, **arrays):
    for name, value in arrays.items():
        setattr(rays, name, value)
    return rays


@pytest.mark.parametrize(
    "rays",
    [
        lambda: _polarized(200),
        lambda: _fan(PART),  # one part only
        lambda: _with(_fan(200), w=np.full(200, 0.55, dtype=np.float32)),
        lambda: _with(_fan(200), i=np.ones(199)),
        lambda: _with(_fan(200), x=np.zeros((200, 1))),
        lambda: _with(_fan(200), x=np.zeros(200).view(_Subarray)),
    ],
    ids=["polarized", "too_few", "float32", "other_length", "2d", "subclass"],
)
def test_rays_outside_the_domain_are_traced_whole(small_parts, rays):
    """Only RealRays with 1-D float64 arrays of one length are split."""
    surfaces = CookeTriplet().surfaces.surfaces
    assert chunked.trace_in_parts(surfaces, rays(), True) is False


def test_other_backends_are_traced_whole(small_parts):
    """The Torch backend never traces in parts."""
    pytest.importorskip("torch")
    surfaces = CookeTriplet().surfaces.surfaces
    be.set_backend("torch")
    try:
        assert chunked.trace_in_parts(surfaces, _fan(200), True) is False
    finally:
        be.set_backend("numpy")


class _MyGeometry(StandardGeometry):
    """A subclass of an allowed geometry."""


class _MyCoating(SimpleCoating):
    """A subclass of an allowed coating."""


class _MyAperture(RadialAperture):
    """A subclass of an allowed aperture."""


def _change_geometry(surface):
    geometry = surface.geometry
    surface.geometry = _MyGeometry(geometry.cs, geometry.radius)


def _change_coating(surface):
    surface.interaction_model.coating = _MyCoating(0.9)


def _change_aperture(surface):
    surface.aperture = _MyAperture(r_max=10.0)


def _add_bsdf(surface):
    from optiland.scatter import LambertianBSDF

    surface.interaction_model.bsdf = LambertianBSDF()


@pytest.mark.parametrize(
    "change", [_change_geometry, _change_coating, _change_aperture, _add_bsdf]
)
def test_other_types_are_traced_whole(small_parts, change):
    """A subclass of an allowed type, or a BSDF, gives no trace in parts."""
    optic = CookeTriplet()
    change(optic.surfaces.surfaces[1])
    assert chunked.trace_in_parts(optic.surfaces.surfaces, _fan(200), True) is False


def test_recorded_arrays_are_the_arrays_of_record_real():
    """_RECORDED names each array that _record_real stores on a surface.

    A missing name would keep the record of the last part only.
    """
    for surface in CookeTriplet().surfaces.surfaces:
        surface.reset()
        before = dict(vars(surface))
        surface._record_real(_fan(8))
        stored = {
            name
            for name, value in vars(surface).items()
            if value is not before.get(name)
        }
        assert stored == set(chunked._RECORDED), type(surface).__name__


def test_default_parts_are_large():
    """The default sizes come from a measurement on Windows.

    The other tests use small parts, so this test keeps a change of the
    default sizes from passing unseen.
    """
    assert chunked._CHUNK_SIZE == 65536
    assert chunked._MIN_RAYS == 2 * chunked._CHUNK_SIZE
