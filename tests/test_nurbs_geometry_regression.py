"""Regression pins for six fixed NurbsGeometry defects.

Each test names the defect it pins and fails if it comes back. They are kept apart
from test_nurbs_geometry.py, which covers ordinary behaviour, because these exist to
stop a refactor silently reintroducing something specific:

    1. weights, degrees and knot vectors were computed in __init__ and discarded
    2. the Newton was seeded at a corner of the patch and restarted at random
    3. sag() opened on x.shape, so a scalar raised
    4. distance() returned |S - P0|, unsigned
    5. a root off the patch iterated to max_iter every time
    6. _standard_surface fitted n control points to n + 1 data points

A sphere is the reference throughout: its sag is known in closed form, so nothing
here is checked against the code that produces it.
"""

from __future__ import annotations

import numpy as np
import pytest

from optiland import backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.nurbs.nurbs_geometry import NurbsGeometry
from tests.utils import assert_allclose

RADIUS = 50.0
HALF = 13.0
COUNT = 16


def conic_sag(x, y, radius=RADIUS, conic=0.0):
    """Closed-form sag of the reference conic, in plain numpy."""
    r2 = np.asarray(x, dtype=float) ** 2 + np.asarray(y, dtype=float) ** 2
    return r2 / (radius * (1 + np.sqrt(1 - (1 + conic) * r2 / radius**2)))


def fitted_sphere(count=COUNT, radius=RADIUS, half=HALF):
    """The reference sphere as optiland fits it for itself."""
    geometry = NurbsGeometry(
        CoordinateSystem(),
        radius=radius,
        conic=0.0,
        nurbs_norm_x=half,
        nurbs_norm_y=half,
        n_points_u=count,
        n_points_v=count,
    )
    geometry.fit_surface()
    return geometry


def probe(n=13, reach=12.0):
    """Sample heights across the patch, out to its rim where a lost root shows."""
    x = np.linspace(-reach, reach, n)
    return (
        be.asarray(x, dtype=be.float64),
        be.asarray(np.zeros_like(x), dtype=be.float64),
        x,
    )


def off_patch(n=7):
    """Heights well beyond the patch, whose root is not on it at all.

    This is the case that used to hand every ray a fresh random (u, v), and it does
    so on ANY net -- which is why nothing here needs an ill conditioned one. A net
    whose control points run to 1e10 does expose the same bugs, but its arithmetic
    is all cancellation, so whether it converges depends on the BLAS underneath and
    the test passes on one machine and fails on the next.
    """
    x = np.linspace(HALF + 2.0, HALF + 20.0, n)
    return (
        be.asarray(x, dtype=be.float64),
        be.asarray(np.zeros_like(x), dtype=be.float64),
        x,
    )


class MockRays:
    """The attributes distance() and surface_normal() read off a ray bundle."""

    def __init__(self, x, y, z, L, M, N):
        self.x = x
        self.y = y
        self.z = z
        self.L = L
        self.M = M
        self.N = N


# ── 1. weights, degrees and knots were computed and discarded ────────────────


def test_weights_are_supplied_when_omitted(set_test_backend):
    """A B-spline's weights are all 1, so a caller should not have to say so.

    They used to be worked out into a local and dropped, leaving self.W as None and
    the first evaluation dying on W.ndim.
    """
    fitted = fitted_sphere()
    geometry = NurbsGeometry(
        CoordinateSystem(),
        control_points=fitted.P,
        u_knots=fitted.U,
        v_knots=fitted.V,
        u_degree=fitted.p,
        v_degree=fitted.q,
        nurbs_norm_x=HALF,
        nurbs_norm_y=HALF,
    )
    assert geometry.W is not None
    assert be.shape(geometry.W) == be.shape(fitted.P)[1:]

    # and it evaluates rather than raising
    x, y, _ = probe(3)
    assert np.all(np.isfinite(be.to_numpy(geometry.sag(x, y))))


def test_knot_vectors_are_built_when_omitted(set_test_backend):
    """The branch that builds a clamped knot vector for you also kept it to itself."""
    fitted = fitted_sphere()
    geometry = NurbsGeometry(
        CoordinateSystem(),
        control_points=fitted.P,
        u_degree=fitted.p,
        v_degree=fitted.q,
        nurbs_norm_x=HALF,
        nurbs_norm_y=HALF,
    )
    assert geometry.U is not None
    assert geometry.V is not None
    assert geometry.W is not None


# ── 2. the Newton was seeded at a corner and restarted at random ─────────────


def test_seed_places_a_point_across_the_patch(set_test_backend):
    """The seed is the fix itself: a point's place across the patch, rather than the
    corner every ray used to start from however far away its root was.

    A clamped net interpolates its corner control points, so the box those span is
    the patch's own extent and the fraction across it is exact.
    """
    geometry = fitted_sphere()
    points = np.array([-HALF, -HALF / 2, 0.0, HALF / 2, HALF])
    u, v = geometry._seed(
        be.asarray(points, dtype=be.float64), be.asarray(points, dtype=be.float64)
    )
    expected = (points + HALF) / (2 * HALF)
    assert_allclose(be.to_numpy(u), expected, atol=1e-9)
    assert_allclose(be.to_numpy(v), expected, atol=1e-9)


def test_solver_lands_on_the_point_it_was_asked_about(set_test_backend):
    """Landing error -- how far the point the solver settled on is from the point it
    was asked about -- is the solver's own error, and says nothing about whether the
    net is a good sphere. Probed out to the rim, where a lost root shows.
    """
    geometry = fitted_sphere()
    x, y, _ = probe(reach=HALF)
    u, v = geometry._newton(
        lambda u, v: geometry._corr(u, v, -be.ravel(y), -be.ravel(x)),
        *geometry._seed(x, y),
    )
    surface = be.to_numpy(geometry.get_value(u, v))
    assert_allclose(surface[0], be.to_numpy(x), atol=1e-9)
    assert_allclose(surface[1], be.to_numpy(y), atol=1e-9)


def test_sag_is_repeatable_off_the_patch(set_test_backend):
    """The old recovery threw a random (u, v) at every ray that left the patch, so
    the same call on the same surface did not have to give the same answer twice.
    Four consecutive calls on master return four different sags.
    """
    geometry = fitted_sphere()
    x, y, _ = off_patch()
    first = be.to_numpy(geometry.sag(x, y))
    for _ in range(4):
        assert_allclose(be.to_numpy(geometry.sag(x, y)), first, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("sampler", [probe, off_patch], ids=["on the patch", "off it"])
def test_sag_is_symmetric_about_a_symmetric_surface(set_test_backend, sampler):
    """A sphere is even in x, so any left-right difference is the solver rather than
    the surface. This needs no reference to compare against, which is why it is here.
    """
    geometry = fitted_sphere()
    x, y, _ = sampler()
    left = be.to_numpy(geometry.sag(-x, y))
    right = be.to_numpy(geometry.sag(x, y))
    assert_allclose(left, right[::-1], atol=1e-9)


# ── 3. sag() opened on x.shape, so a scalar raised ───────────────────────────


def test_sag_accepts_scalars(set_test_backend):
    """Every other geometry takes a float. A caller sampling one height at a time --
    sizing an aperture, say -- had every probe raise AttributeError.
    """
    geometry = fitted_sphere()
    value = float(be.to_numpy(geometry.sag(0.0, 5.0)))
    assert value == pytest.approx(float(conic_sag(0.0, 5.0)), abs=1e-5)


def test_sag_returns_the_callers_own_shape(set_test_backend):
    """A scalar in, a scalar out; a grid in, that grid out."""
    geometry = fitted_sphere()
    assert be.to_numpy(geometry.sag(0.0, 0.0)).shape == ()

    grid = np.linspace(-5.0, 5.0, 6).reshape(2, 3)
    shaped = geometry.sag(
        be.asarray(grid, dtype=be.float64),
        be.asarray(np.zeros_like(grid), dtype=be.float64),
    )
    assert be.to_numpy(shaped).shape == (2, 3)


# ── 4. distance() returned |S - P0|, unsigned ────────────────────────────────


@pytest.mark.parametrize(
    ("start", "sign"),
    [(-5.0, 1.0), (5.0, -1.0)],
    ids=["ray in front of the surface", "ray behind it"],
)
def test_distance_is_signed_along_the_ray(set_test_backend, start, sign):
    """An intersection BEHIND a ray used to come back as an equal step in front of
    it, putting the ray on the wrong side of itself. A plain trace rarely looks
    back; an iterative aimer probes both ways and cannot recover from the sign.
    """
    geometry = fitted_sphere()
    ones = be.asarray(np.ones(3), dtype=be.float64)
    zeros = be.asarray(np.zeros(3), dtype=be.float64)
    rays = MockRays(
        x=be.asarray(np.array([0.0, 2.0, -2.0]), dtype=be.float64),
        y=be.asarray(np.array([0.0, 1.0, 1.0]), dtype=be.float64),
        z=be.asarray(np.full(3, start), dtype=be.float64),
        L=zeros,
        M=zeros,
        N=ones,
    )
    distance = be.to_numpy(geometry.distance(rays))
    assert np.all(np.sign(distance) == sign)

    # and its magnitude is still the gap to the surface
    sag = conic_sag(np.array([0.0, 2.0, -2.0]), np.array([0.0, 1.0, 1.0]))
    assert_allclose(distance, sag - start, atol=1e-4)


# ── 5. a root off the patch iterated to max_iter every time ──────────────────


def test_a_pinned_ray_stops_once_it_stops_improving(set_test_backend):
    """Clamping to the patch leaves a ray whose root is outside it pinned on the
    rim, where the residual never reaches tol. Without an early out that is the
    full max_iter every time, for an answer reached in a handful.
    """
    geometry = fitted_sphere()
    assert geometry.max_stall < geometry.max_iter

    outside = np.linspace(20.0, 40.0, 9)  # well beyond the +-13 mm patch
    x = be.asarray(outside, dtype=be.float64)
    y = be.asarray(np.zeros_like(outside), dtype=be.float64)

    calls = []
    original = geometry._corr

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    geometry._corr = counted
    geometry.sag(x, y)

    assert len(calls) <= geometry.max_stall + 1
    assert len(calls) < geometry.max_iter


def test_max_stall_reaches_a_surface_built_through_the_factory(set_test_backend):
    """NurbsConfig has to carry it, or the factory cannot pass one and every surface
    on an optic is stuck on the constructor's default whatever the caller asked.
    """
    from optiland.surfaces.factories.geometry_factory import GeometryFactory

    geometry = GeometryFactory.create(
        "nurbs",
        CoordinateSystem(),
        radius=RADIUS,
        conic=0.0,
        nurbs_norm_x=HALF,
        nurbs_norm_y=HALF,
        n_points_u=COUNT,
        n_points_v=COUNT,
        max_stall=7,
    )
    assert geometry.max_stall == 7


# ── 6. the fit asked n control points of n + 1 data points ───────────────────


def test_fit_is_idempotent(set_test_backend):
    """update_normalization refits on every paraxial pass, so a fit that shrinks the
    net degrades the surface as the optic updates. It used to lose one per fit.
    """
    geometry = fitted_sphere()
    sizes = []
    for _ in range(4):
        geometry.fit_surface()
        sizes.append(tuple(be.shape(geometry.P)[1:]))
    assert sizes == [(COUNT, COUNT)] * 4


def test_the_fitted_net_describes_its_own_surface(set_test_backend):
    """A square least squares leaves control points decades from the surface: 1e10 mm
    of control z for a surface a few mm deep, cancelling to something nearly right.
    """
    geometry = fitted_sphere(count=32)
    control_z = be.to_numpy(geometry.P)[2]
    depth = float(conic_sag(HALF, HALF))
    assert control_z.min() > -depth
    assert control_z.max() < 2 * depth


def test_the_fit_reproduces_the_sphere(set_test_backend):
    """Accuracy follows from the conditioning above -- four orders of it."""
    geometry = fitted_sphere(count=32)
    x, y, raw = probe(21)
    error = np.abs(be.to_numpy(geometry.sag(x, y)) - conic_sag(raw, 0.0)).max()
    assert error < 1e-6


def test_a_box_reaching_past_the_conic_edge_still_fits(set_test_backend):
    """The sag has no real root past the conic's own edge, and the corners of a
    square box reach there whenever 2.half^2 > R^2. A bare square root gives NaN,
    which the fitter rejects outright rather than fitting anything at all.
    """
    geometry = fitted_sphere(count=COUNT, radius=15.0, half=13.0)
    assert not np.any(np.isnan(be.to_numpy(geometry.P)))

    # and it is still the sphere well inside the edge
    x, y, raw = probe(9, reach=10.0)
    error = np.abs(
        be.to_numpy(geometry.sag(x, y)) - conic_sag(raw, 0.0, radius=15.0)
    ).max()
    assert error < 1e-2


@pytest.mark.parametrize("count", [4, 8, 16])
def test_n_points_is_the_control_count_for_either_branch(set_test_backend, count):
    """It is documented as the grid size of control points, but was read as a DATA
    count by the conic branch and a control count by the plane one, so the same
    n_points_u gave nets a size apart depending on which surface it was.
    """
    conic = fitted_sphere(count=count)

    plane = NurbsGeometry(
        CoordinateSystem(),
        radius=be.inf,
        nurbs_norm_x=HALF,
        nurbs_norm_y=HALF,
        n_points_u=count,
        n_points_v=count,
    )
    plane.fit_surface()

    assert tuple(be.shape(conic.P)[1:]) == (count, count)
    assert tuple(be.shape(plane.P)[1:]) == (count, count)
