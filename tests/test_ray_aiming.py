from __future__ import annotations

import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.optic import Optic
from optiland.rays import RealRays
from optiland.rays.ray_aiming.iterative import IterativeRayAimer
from optiland.rays.ray_aiming.paraxial import ParaxialRayAimer
from optiland.rays.ray_aiming.robust import RobustRayAimer
from optiland.samples.objectives import (
    CookeTriplet,
    ProjectionLens120FOV,
    ProjectionLens160FOV,
    ReverseTelephoto,
    WideAngle100FOV,
    WideAngle170FOV,
)
from optiland.samples.simple import Edmund_49_847, SingletStopSurf2

from .utils import assert_allclose


def test_iterative_aimer_infinite(set_test_backend):
    """Test IterativeRayAimer on an infinite conjugate system."""
    optic = ReverseTelephoto()
    aimer = IterativeRayAimer(optic, tol=1e-8)

    Hx, Hy = 0.0, 1.0
    Px, Py = 0.0, 1.0
    wavelength = 0.55

    x, y, z, L, M, N = aimer.aim_rays((Hx, Hy), wavelength, (Px, Py))

    stop_idx = optic.surfaces.stop_index

    from optiland.rays import RealRays

    rays = RealRays(x, y, z, L, M, N, intensity=1.0, wavelength=wavelength)
    for i in range(1, stop_idx + 1):
        optic.surfaces[i].trace(rays)

    stop_surf = optic.surfaces[stop_idx]
    assert not be.any(be.isnan(x))


def test_robust_aimer_infinite(set_test_backend):
    """Test RobustRayAimer on an infinite conjugate system."""
    optic = ReverseTelephoto()
    aimer = RobustRayAimer(optic, tol=1e-8)

    Hx, Hy = 0.0, 1.0
    Px, Py = 0.0, 1.0
    wavelength = 0.55

    x, y, z, L, M, N = aimer.aim_rays((Hx, Hy), wavelength, (Px, Py))
    assert not be.any(be.isnan(x))


def test_aimer_consistency(set_test_backend):
    """Ensure Iterative and Robust aimers yield similar results for easy rays."""
    optic = ReverseTelephoto()
    iter_aimer = IterativeRayAimer(optic, tol=1e-10)
    robust_aimer = RobustRayAimer(optic, tol=1e-10)

    Hx, Hy = 0.0, 0.5
    Px, Py = 0.0, 0.5
    wavelength = 0.55

    res_iter = iter_aimer.aim_rays((Hx, Hy), wavelength, (Px, Py))
    res_robust = robust_aimer.aim_rays((Hx, Hy), wavelength, (Px, Py))

    for r_i, r_r in zip(res_iter, res_robust, strict=False):
        assert be.allclose(r_i, r_r, atol=1e-6)


def test_large_batch(set_test_backend):
    """Test aiming with a large batch of rays."""
    optic = ReverseTelephoto()
    aimer = IterativeRayAimer(optic)

    n = 100
    Hx = np.zeros(n)
    Hy = np.linspace(0, 1, n)
    Px = np.linspace(-1, 1, n)
    Py = np.zeros(n)
    wavelength = 0.55

    x, y, z, L, M, N = aimer.aim_rays((Hx, Hy), wavelength, (Px, Py))

    assert len(x) == n
    assert not be.any(be.isnan(x))


def test_robust_aimer_initialization(set_test_backend):
    """Test the initialization of the RobustRayAimer."""
    optic = ReverseTelephoto()
    aimer = RobustRayAimer(optic, max_iter=30, tol=1e-7, scale_fields=False)

    assert aimer.optic == optic
    assert aimer._iterative.max_iter == 30
    assert aimer._iterative.tol == 1e-7
    assert aimer.scale_fields is False
    assert isinstance(aimer._paraxial, type(aimer._iterative._paraxial_aimer))


def test_integration_via_optic(set_test_backend):
    """Test setting the ray aimer via the Optic class."""
    optic = ReverseTelephoto()

    optic.ray_tracer.set_aiming("iterative", max_iter=25, tol=1e-5)
    ray_gen = optic.ray_tracer.ray_generator
    optic.trace(0, 0, 0.55, num_rays=1)

    aimer = ray_gen.aimer
    assert isinstance(aimer, IterativeRayAimer)
    assert aimer.max_iter == 25
    assert aimer.tol == 1e-5

    optic.ray_tracer.set_aiming("robust", max_iter=15, tol=1e-6, scale_fields=True)
    optic.trace(0, 0, 0.55, num_rays=1)

    aimer = ray_gen.aimer
    assert isinstance(aimer, RobustRayAimer)
    assert aimer._iterative.max_iter == 15
    assert aimer._iterative.tol == 1e-6
    assert aimer.scale_fields is True


def test_robust_caching_regression(set_test_backend):
    """Regression test for cached RobustRayAimer accepting initial_guess."""
    optic = ReverseTelephoto()
    optic.ray_tracer.set_aiming("robust", cache=True)

    # 1. First trace (populates cache)
    optic.trace(0, 0, 0.55, num_rays=1)

    # 2. Perturb system to force reuse of result as initial_guess
    # Modify a radius slightly. We can just set a new value directly.
    # This ensures the system hash changes.
    optic.updater.set_radius(100.0, 1)

    # 3. Second trace (should call robust aimer with initial_guess)
    optic.trace(0, 0, 0.55, num_rays=1)


def test_robust_aimer_infinite_object_90_degree_field(set_test_backend):
    """Regression test: verify RobustRayAimer aims correctly for 90 deg field @ infinity.

    See bug fix where IterativeRayAimer inherited bad L,M,N from initial_guess.
    """
    optic = Optic()
    # Construct a minimal wide angle lens setup that reproduces the infinite + 90 deg scenario
    # We'll use a simplified version of the user's lens to avoid clutter,
    # but ensure it has infinite object and large field.

    optic.surfaces.add(index=0, radius=float("inf"), thickness=float("inf"))
    # A dummy surface to aim at
    optic.surfaces.add(
        index=1, radius=100.0, thickness=10.0, material="air", is_stop=True
    )
    optic.surfaces.add(index=2)

    optic.set_aperture("EPD", 1.0)
    optic.fields.set_type("angle")
    optic.fields.add(y=0)
    optic.fields.add(y=90)
    optic.wavelengths.add(0.55, is_primary=True)

    optic.ray_tracer.set_aiming("robust")

    from optiland.rays.ray_generator import RayGenerator

    rg = RayGenerator(optic)

    # Generate rays for 90 degree field (Hy=1.0)
    # 90 degrees means rays come from +Y relative to Z.
    # Direction vector should be approx (0, 1, 0).
    # N (z-dir cosine) should be near 0.
    rays = rg.generate_rays(Hx=0, Hy=1, Px=0, Py=0, wavelength=0.55)

    # Check N is close to 0 (allow small tolerance due to numerical precision/mapping)
    # Using the fixed code, we saw N ~ 0.02 which is small enough compared to N ~ 1.
    assert abs(rays.N[0]) < 0.1
    assert rays.M[0] > 0.9  # Should be largely in Y direction


def test_instantiate_wide_angle_lenses(set_test_backend):
    """This tests only if we can instantiate wide angle lenses with error"""
    assert WideAngle100FOV() is not None
    assert ProjectionLens120FOV() is not None
    assert ProjectionLens160FOV() is not None
    assert WideAngle170FOV() is not None


# ---------------------------------------------------------------------------
# Regression tests for issue #613: ray aiming must be invariant under a rigid
# translation of every surface along z.

def _shift_optic(optic: Optic, dz: float) -> None:
    """Translate every finite-z surface of *optic* along z by *dz*.

    Reassigns rather than using ``+=`` so torch's autograd doesn't reject the
    update on leaf tensors that require grad.
    """
    for surf in optic.surfaces.surfaces:
        z = float(be.to_numpy(surf.geometry.cs.z))
        if math.isfinite(z):
            surf.geometry.cs.z = surf.geometry.cs.z + dz


def _issue_613_reproducer() -> Optic:
    """The exact optic from issue #613: finite-conjugate object-height fields
    with a float-by-stop-size aperture and flat air-to-air surfaces, so a
    paraxially aimed ray reaches the stop without refraction."""
    system = Optic()
    system.surfaces.add(index=0, thickness=10, comment="start")
    system.surfaces.add(index=1, thickness=10, comment="flat")
    system.surfaces.add(index=2, thickness=10, comment="aperture")
    system.surfaces.add(index=3, thickness=0, comment="end")
    system.wavelengths.add(value=0.55)
    system.surfaces[2].is_stop = True
    system.fields.set_type(field_type="object_height")
    system.fields.add(y=0)
    system.fields.add(y=3)
    system.set_aperture(aperture_type="float_by_stop_size", value=4.0)
    return system


def _issue_636_reproducer() -> Optic:
    """Build the infinite-object angle-field system from issue #636."""
    system = Optic()
    system.surfaces.add(index=0, thickness=np.inf, comment="start")
    system.surfaces.add(index=1, thickness=10, comment="flat")
    system.surfaces.add(index=2, thickness=10, comment="aperture")
    system.surfaces.add(index=3, thickness=0, comment="end")
    system.wavelengths.add(value=0.55)
    system.surfaces[2].is_stop = True
    system.fields.set_type(field_type="angle")
    system.fields.add(y=0)
    system.fields.add(y=10)
    system.set_aperture(aperture_type="float_by_stop_size", value=4.0)
    return system


def _trace_aimed_chief_ray_to_stop(optic: Optic, Hy: float) -> float:
    """Aim a paraxial chief ray for field ``(0, Hy)``, trace it through every
    surface up to the stop, and return its y in the stop's local frame."""
    wavelength = optic.primary_wavelength
    aimer = ParaxialRayAimer(optic)
    x0, y0, z0, L, M, N = aimer.aim_rays(
        (be.array([0.0]), be.array([Hy])),
        be.array([wavelength]),
        (be.array([0.0]), be.array([0.0])),
    )
    rays = RealRays(
        x=be.copy(x0), y=be.copy(y0), z=be.copy(z0),
        L=be.copy(L), M=be.copy(M), N=be.copy(N),
        intensity=be.array([1.0]),
        wavelength=be.array([wavelength]),
    )
    stop_index = optic.surfaces.stop_index
    for i in range(1, stop_index + 1):
        optic.surfaces[i].trace(rays)
    optic.surfaces[stop_index].geometry.cs.localize(rays)
    return float(be.to_numpy(rays.y)[0])


@pytest.mark.parametrize(
    "build",
    [CookeTriplet, ReverseTelephoto, SingletStopSurf2, _issue_613_reproducer],
    ids=["CookeTriplet", "ReverseTelephoto", "SingletStopSurf2", "issue613"],
)
@pytest.mark.parametrize("dz", [25.0, -15.0])
def test_epl_is_invariant_under_translation(set_test_backend, build, dz):
    """``EPL()`` returns a value relative to the first physical surface, so it
    must be unchanged by a rigid translation of every surface along z.

    This guards the *convention*. The authoritative regression test for the
    end-to-end bug in issue #613 is
    ``test_paraxial_chief_ray_invariant_under_translation`` below — a future
    contributor mis-routing the relative value into a global-coordinate
    expression will be caught there, not here.
    """
    epl_ref = float(build().paraxial.EPL())
    shifted = build()
    _shift_optic(shifted, dz)
    assert_allclose(float(shifted.paraxial.EPL()), epl_ref)


@pytest.mark.parametrize(
    "build",
    [CookeTriplet, ReverseTelephoto, SingletStopSurf2, _issue_613_reproducer],
    ids=["CookeTriplet", "ReverseTelephoto", "SingletStopSurf2", "issue613"],
)
@pytest.mark.parametrize("dz", [25.0, -15.0])
def test_entrance_pupil_global_z_shifts_by_translation(set_test_backend, build, dz):
    """``entrance_pupil_z()`` returns a global z and must shift by exactly
    ``dz`` under rigid translation. This is the helper internal consumers
    use; if it ever stops tracking translation, issue #613 returns."""
    z_ref = float(build().paraxial.entrance_pupil_z())
    shifted = build()
    _shift_optic(shifted, dz)
    assert_allclose(float(shifted.paraxial.entrance_pupil_z()), z_ref + dz)


def test_epl_is_zero_when_stop_is_surface_1_with_shifted_optic(set_test_backend):
    """When the stop coincides with surface 1, the entrance pupil sits *on*
    surface 1, so its location in surface 1's local frame is 0 — independent
    of where surface 1 is in global z. This test shifts the whole optic so
    ``positions[1, 0] != 0``; the previous code returned ``positions[1, 0]``
    from the ``stop_index == 1`` branch, which silently mixed conventions
    and only happened to agree with the docstring when surface 1 was at the
    origin."""
    optic = Edmund_49_847()
    _shift_optic(optic, dz=25.0)
    assert optic.surfaces.stop_index == 1
    pos1 = float(be.to_numpy(optic.surfaces.positions[1, 0]))
    assert_allclose(pos1, 25.0)
    assert_allclose(float(optic.paraxial.EPL()), 0.0)
    assert_allclose(float(optic.paraxial.entrance_pupil_z()), 25.0)


@pytest.mark.parametrize(
    "build",
    [CookeTriplet, ReverseTelephoto, SingletStopSurf2, _issue_613_reproducer],
    ids=["CookeTriplet", "ReverseTelephoto", "SingletStopSurf2", "issue613"],
)
@pytest.mark.parametrize("dz", [25.0, -15.0])
def test_epd_invariant_under_translation(set_test_backend, build, dz):
    """``EPD()`` is a length, unchanged by rigid translation."""
    epd_ref = float(build().paraxial.EPD())
    shifted = build()
    _shift_optic(shifted, dz)
    assert_allclose(float(shifted.paraxial.EPD()), epd_ref)


@pytest.mark.parametrize(
    "build",
    [CookeTriplet, ReverseTelephoto, SingletStopSurf2],
    ids=["CookeTriplet", "ReverseTelephoto", "SingletStopSurf2"],
)
@pytest.mark.parametrize("dz", [25.0, -15.0])
def test_paraxial_chief_ray_invariant_under_translation(set_test_backend, build, dz):
    """Chief-ray heights and slopes at every surface are translation-invariant.
    The aimer consumes these via ``Paraxial.chief_ray``."""
    y_ref, u_ref = build().paraxial.chief_ray()
    shifted = build()
    _shift_optic(shifted, dz)
    y_shifted, u_shifted = shifted.paraxial.chief_ray()
    assert_allclose(y_shifted, y_ref)
    assert_allclose(u_shifted, u_ref)


@pytest.mark.parametrize(
    "build",
    [CookeTriplet, ReverseTelephoto, SingletStopSurf2],
    ids=["CookeTriplet", "ReverseTelephoto", "SingletStopSurf2"],
)
@pytest.mark.parametrize("dz", [25.0, -15.0])
def test_paraxial_marginal_ray_invariant_under_translation(
    set_test_backend, build, dz
):
    """Marginal-ray heights and slopes at every surface are translation-invariant."""
    y_ref, u_ref = build().paraxial.marginal_ray()
    shifted = build()
    _shift_optic(shifted, dz)
    y_shifted, u_shifted = shifted.paraxial.marginal_ray()
    assert_allclose(y_shifted, y_ref)
    assert_allclose(u_shifted, u_ref)


@pytest.mark.parametrize("dz", [0.0, 30.0, -30.0, 1000.0])
@pytest.mark.parametrize("Hy", [0.0, 0.5, 1.0])
def test_paraxial_aimer_hits_stop_center_after_translation(set_test_backend, dz, Hy):
    """Issue #613 verbatim: paraxially aimed chief rays must intersect the
    stop center regardless of where the system sits along z."""
    optic = _issue_613_reproducer()
    if dz:
        _shift_optic(optic, dz)
    assert_allclose(_trace_aimed_chief_ray_to_stop(optic, Hy), 0.0, atol=1e-9)


@pytest.mark.parametrize("dz", [0.0, 50.0, -25.0])
def test_paraxial_aimer_infinite_object_chief_ray_matches_reference(
    set_test_backend, dz
):
    """For an infinite-conjugate system (AngleField), where the aimed chief
    ray actually lands at the stop must not depend on axial translation."""
    optic_ref = CookeTriplet()
    shifted = CookeTriplet()
    if dz:
        _shift_optic(shifted, dz)
    for Hy in (0.0, 0.5, 1.0):
        y_ref = _trace_aimed_chief_ray_to_stop(optic_ref, Hy)
        y_shifted = _trace_aimed_chief_ray_to_stop(shifted, Hy)
        assert_allclose(y_shifted, y_ref, atol=1e-9)


@pytest.mark.parametrize("dz", [100.0, -100.0])
def test_infinite_object_ray_path_invariant_under_translation(
    set_test_backend, dz
):
    """Issue #636: plotted ray paths must follow a rigid system translation."""
    optic_ref = _issue_636_reproducer()
    optic_ref.trace(0.0, 1.0, 0.55, 3, "line_y")

    shifted = _issue_636_reproducer()
    _shift_optic(shifted, dz)
    shifted.trace(0.0, 1.0, 0.55, 3, "line_y")

    assert_allclose(shifted.surfaces.y, optic_ref.surfaces.y)
    assert_allclose(shifted.surfaces.z - dz, optic_ref.surfaces.z)


@pytest.mark.parametrize("dz", [30.0, -10.0, 1000.0])
def test_float_by_stop_epd_invariant_under_translation(set_test_backend, dz):
    """``FloatByStopAperture.compute_epd`` returned a negative number after
    ``move_z`` in #613 because EPL and obj_z were in different coordinate
    systems. After the fix the computed EPD must match the reference and stay
    positive."""
    optic_ref = _issue_613_reproducer()
    epd_ref = float(optic_ref.aperture.compute_epd(optic_ref.paraxial))
    shifted = _issue_613_reproducer()
    _shift_optic(shifted, dz)
    epd_shifted = float(shifted.aperture.compute_epd(shifted.paraxial))
    assert_allclose(epd_shifted, epd_ref)
    assert epd_shifted > 0


@pytest.mark.parametrize("AimerCls", [IterativeRayAimer, RobustRayAimer])
@pytest.mark.parametrize("dz", [0.0, 40.0])
def test_real_aimer_converges_after_translation(set_test_backend, AimerCls, dz):
    """Iterative/Robust aimers seed from the paraxial aimer, so a broken EPL
    used to feed a poor initial guess. Both must still converge to non-NaN
    rays after the system is translated."""
    optic = ReverseTelephoto()
    if dz:
        _shift_optic(optic, dz)
    aimer = AimerCls(optic, tol=1e-8)
    x, y, z, _, _, _ = aimer.aim_rays(
        (0.0, 1.0), optic.primary_wavelength, (0.0, 1.0)
    )
    assert not be.any(be.isnan(x))
    assert not be.any(be.isnan(y))
    assert not be.any(be.isnan(z))


# ---------------------------------------------------------------------------
# Tests for the chief-ray calibrated RobustRayAimer.

from optiland.distribution import create_distribution  # noqa: E402
from optiland.rays.ray_aiming.initialization import (  # noqa: E402
    get_stop_radius_strategy,
)
from optiland.rays.ray_aiming.pupil_map import PupilMap  # noqa: E402
from optiland.samples.lithography import UVProjectionLens  # noqa: E402

_WIDE_ANGLE_SAMPLES = [
    CookeTriplet,
    WideAngle100FOV,
    ProjectionLens120FOV,
    ProjectionLens160FOV,
    WideAngle170FOV,
]


def test_pupil_map_seed_affine_arithmetic(set_test_backend):
    """Unit test for the affine fit itself: launch(Px, Py) = c + A @ [Px, Py],
    with the non-solved DOF held fixed at their chief-ray values."""
    pmap = PupilMap(
        c=(1.0, 2.0),
        A=((0.5, 0.1), (0.2, 0.3)),
        is_infinite=True,
        fixed=(10.0, 0.0, 0.0, 1.0),
    )
    x, y, z, L, M, N = pmap.seed(be.array([0.0, 1.0]), be.array([0.0, -1.0]))
    assert_allclose(x, [1.0, 1.4])
    assert_allclose(y, [2.0, 1.9])
    assert_allclose(z, [10.0, 10.0])
    assert_allclose(L, [0.0, 0.0])
    assert_allclose(M, [0.0, 0.0])
    assert_allclose(N, [1.0, 1.0])


def test_robust_aimer_chief_ray_matches_field_angle_per_field(set_test_backend):
    """Regression test: for an infinite-conjugate (angle field) system, the
    chief-ray direction cosines are fixed purely by the field angle (no
    refraction occurs before the first surface), so they must equal
    sin/cos of each field's *own* raw angle -- not collapse toward the
    on-axis (or any other field's) direction. This is the bug class where
    a field-marching warm start reused another field's fixed launch
    components (direction, for infinite conjugates) instead of recomputing
    them for the actual target field."""
    optic = WideAngle170FOV()
    aimer = RobustRayAimer(optic)
    stop_idx = optic.surfaces.stop_index
    is_inf = True

    raw_degrees = [f.y for f in optic.fields.fields]
    coords = optic.fields.get_field_coords()
    assert raw_degrees == [0.0, 60.0, 85.0]

    seen_M = []
    for (Hx, Hy), deg in zip(coords, raw_degrees, strict=False):
        # Fresh aimer per field: no cross-field warm start available, so
        # this also exercises the cold chief-ray marching fallback.
        aimer_fresh = RobustRayAimer(WideAngle170FOV())
        chief = aimer_fresh._solve_chief(
            Hx, Hy, optic.primary_wavelength, stop_idx, is_inf, None
        )
        _, _, _, L, M, N = chief
        assert_allclose(M, math.sin(math.radians(deg)), atol=1e-6)
        assert_allclose(N, math.cos(math.radians(deg)), atol=1e-6)
        seen_M.append(M)

    # The three fields must be genuinely distinct, not all collapsed to
    # the same (e.g. on-axis) direction.
    assert len({round(m, 3) for m in seen_M}) == 3


def test_robust_aimer_multi_field_batch_matches_per_field_reference(
    set_test_backend,
):
    """Regression test: aiming several distinct fields in a single batched
    call (as ``optic.trace(Hx_array, Hy_array, ...)`` does when plotting
    every field at once) must give the same result as aiming each field
    individually. This reproduces the exact call pattern used by system
    plotting, where a bug in cross-field warm-start reuse previously
    caused every field to be aimed as if on-axis."""
    optic_multi = CookeTriplet()
    optic_multi.ray_tracer.set_aiming("robust")
    coords = optic_multi.fields.get_field_coords()

    Hx = be.array([c[0] for c in coords])
    Hy = be.array([c[1] for c in coords])
    rays = optic_multi.trace(
        Hx, Hy, optic_multi.primary_wavelength, num_rays=3, distribution="hexapolar"
    )
    n_fields = len(coords)
    per_field = len(be.to_numpy(rays.y)) // n_fields

    for i, (Hxi, Hyi) in enumerate(coords):
        optic_ref = CookeTriplet()
        optic_ref.ray_tracer.set_aiming("robust")
        rays_ref = optic_ref.trace(
            Hxi, Hyi, optic_ref.primary_wavelength, num_rays=3,
            distribution="hexapolar",
        )
        batch_slice = slice(i * per_field, (i + 1) * per_field)
        assert_allclose(
            be.to_numpy(rays.y)[batch_slice],
            be.to_numpy(rays_ref.y),
            atol=1e-6,
        )

    # Distinct (non-axial) fields must not produce identical image heights.
    ys = [be.to_numpy(rays.y)[i * per_field] for i in range(n_fields)]
    assert len({round(float(v), 3) for v in ys}) == n_fields


@pytest.mark.parametrize(
    "build", _WIDE_ANGLE_SAMPLES, ids=[c.__name__ for c in _WIDE_ANGLE_SAMPLES]
)
def test_robust_aimer_converges_cold_all_fields(set_test_backend, build):
    """G1/G4: every target system, including WideAngle170FOV (+-85 deg),
    converges cold across all its fields with a full hexapolar pupil -- no
    NaN blow-up and no exception, even though each system gets a brand-new
    aimer instance with an empty warm-start cache."""
    optic = build()
    optic.ray_tracer.set_aiming("robust")
    for Hx, Hy in optic.fields.get_field_coords():
        rays = optic.trace(
            Hx, Hy, optic.primary_wavelength, num_rays=15, distribution="hexapolar"
        )
        valid = ~be.to_numpy(be.isnan(rays.x))
        assert valid.any(), f"all rays vignetted for field ({Hx}, {Hy})"


def test_robust_aimer_isolated_extreme_field_converges_cold(set_test_backend):
    """G1: the same cold-convergence guarantee holds even when the extreme
    field is aimed in complete isolation -- i.e. with no smaller field ever
    solved first to warm-start the cache via field marching."""
    optic = WideAngle170FOV()
    aimer = RobustRayAimer(optic)
    dist = create_distribution("hexapolar")
    dist.generate_points(15)
    Hx, Hy = optic.fields.get_field_coords()[-1]

    x, y, z, L, M, N = aimer.aim_rays(
        (Hx, Hy), optic.primary_wavelength, (dist.x, dist.y)
    )
    valid = ~be.to_numpy(be.isnan(x))
    assert valid.any()


def test_robust_aimer_direction_cosines_unit_norm(set_test_backend):
    """G3: L^2 + M^2 + N^2 == 1 for every non-vignetted aimed ray."""
    optic = WideAngle170FOV()
    aimer = RobustRayAimer(optic)
    dist = create_distribution("hexapolar")
    dist.generate_points(10)

    for Hx, Hy in optic.fields.get_field_coords():
        x, y, z, L, M, N = aimer.aim_rays(
            (Hx, Hy), optic.primary_wavelength, (dist.x, dist.y)
        )
        valid = ~be.to_numpy(be.isnan(L))
        assert valid.any()
        norm = (
            be.to_numpy(L)[valid] ** 2
            + be.to_numpy(M)[valid] ** 2
            + be.to_numpy(N)[valid] ** 2
        )
        assert_allclose(norm, np.ones_like(norm), atol=1e-10)


def test_robust_aimer_hits_target_to_tolerance(set_test_backend):
    """G4: converged rays hit their (Px, Py) target on the stop to `tol`."""
    optic = CookeTriplet()
    tol = 1e-9
    aimer = RobustRayAimer(optic, tol=tol)
    Px = be.array([0.0, 0.5, -0.5, 1.0, -1.0])
    Py = be.array([0.0, 0.5, -0.5, 0.0, 0.0])
    Hx, Hy = 0.0, 0.7

    x, y, z, L, M, N = aimer.aim_rays((Hx, Hy), optic.primary_wavelength, (Px, Py))

    stop_idx = optic.surfaces.stop_index
    is_inf = getattr(optic.object_surface, "is_infinite", False)
    rays = RealRays(
        be.copy(x), be.copy(y), be.copy(z), be.copy(L), be.copy(M), be.copy(N),
        intensity=be.ones_like(x), wavelength=optic.primary_wavelength,
    )
    start = 1 if is_inf else 0
    for i in range(start, stop_idx + 1):
        optic.surfaces[i].trace(rays)
    optic.surfaces[stop_idx].geometry.cs.localize(rays)

    r_stop = get_stop_radius_strategy(optic, "robust").calculate_stop_radius()
    assert_allclose(rays.x, Px * r_stop, atol=1e-6)
    assert_allclose(rays.y, Py * r_stop, atol=1e-6)


def test_robust_aimer_partial_vignetting_does_not_raise(set_test_backend):
    """D6: a batch mixing reachable and physically unreachable pupil targets
    must not raise -- the unreachable rays come back as NaN, the rest are
    solved normally."""
    optic = CookeTriplet()
    aimer = RobustRayAimer(optic)
    Px = be.array([0.0, 0.5, -0.5, 8.0, 1.0])
    Py = be.array([0.0, 0.5, -0.5, 8.0, 0.0])

    x, y, z, L, M, N = aimer.aim_rays((0.0, 0.5), optic.primary_wavelength, (Px, Py))

    nan_mask = be.to_numpy(be.isnan(x))
    assert nan_mask[3]
    assert not nan_mask[[0, 1, 2, 4]].any()


def test_robust_aimer_raises_when_every_ray_fails(set_test_backend):
    """D6: a batch where every ray targets a physically unreachable pupil
    point (a misconfiguration, not ordinary vignetting) must raise."""
    optic = CookeTriplet()
    aimer = RobustRayAimer(optic)
    Px = be.array([5.0, 6.0, 7.0])
    Py = be.array([5.0, 6.0, 7.0])

    with pytest.raises(ValueError):
        aimer.aim_rays((0.0, 0.5), optic.primary_wavelength, (Px, Py))


def test_robust_aimer_warm_start_after_perturbation(set_test_backend):
    """G6: after the first (cold) call, a small system perturbation must
    recalibrate the chief ray in very few Newton iterations, since the
    warm-started seed is already close to the new solution."""
    optic = CookeTriplet()
    aimer = RobustRayAimer(optic)
    Hx, Hy = 0.0, 0.7
    wl = optic.primary_wavelength
    stop_idx = optic.surfaces.stop_index
    is_inf = getattr(optic.object_surface, "is_infinite", False)

    aimer._solve_chief(Hx, Hy, wl, stop_idx, is_inf, None)
    cold_iters = aimer._iterative.last_iterations

    aimer.aim_rays((Hx, Hy), wl, (be.array([0.0]), be.array([0.0])))
    seed_map = aimer._cache.get_stale(Hx, Hy, wl)

    # Small perturbation -- warm-started calibration should now need only a
    # couple of Newton iterations for the chief ray, not a cold re-solve.
    optic.updater.set_radius(optic.surfaces.radii[1] * 1.001, 1)
    aimer._solve_chief(Hx, Hy, wl, stop_idx, is_inf, seed_map)

    assert aimer._iterative.last_iterations <= 2
    assert aimer._iterative.last_iterations <= cold_iters


def test_robust_aimer_reuses_pupil_map_across_pupil_distributions(set_test_backend):
    """D3: the same field/wavelength with two different pupil distributions
    (and no intervening system change) must reuse the cached PupilMap
    instance rather than recalibrating."""
    optic = CookeTriplet()
    aimer = RobustRayAimer(optic)
    Hx, Hy = 0.0, 0.7
    wl = optic.primary_wavelength

    aimer.aim_rays((Hx, Hy), wl, (be.array([0.0]), be.array([0.0])))
    pmap_first = aimer._cache.get_fresh(Hx, Hy, wl)
    assert isinstance(pmap_first, PupilMap)

    aimer.aim_rays((Hx, Hy), wl, (be.array([0.3, -0.3]), be.array([0.2, -0.2])))
    pmap_second = aimer._cache.get_fresh(Hx, Hy, wl)

    assert pmap_second is pmap_first


@pytest.mark.parametrize(
    "build",
    [ReverseTelephoto, _issue_613_reproducer, UVProjectionLens],
    ids=["infinite", "finite", "telecentric"],
)
def test_robust_aimer_all_conjugates(set_test_backend, build):
    """D5: the aimer solves launch position for infinite conjugates, launch
    direction for finite conjugates, and stays consistent for a telecentric
    object space (which keeps the paraxial seed path, but is still
    Newton-polished the same as any other finite conjugate)."""
    optic = build()
    # The "float by stop size" aperture on the finite-conjugate fixture only
    # populates surface.semi_aperture (needed by the robust stop-radius
    # strategy) after a paraxial update.
    optic.updater.update_paraxial()
    aimer = RobustRayAimer(optic)
    _, Hy = optic.fields.get_field_coords()[1]
    x, y, z, L, M, N = aimer.aim_rays(
        (0.0, Hy), optic.primary_wavelength, (0.0, 0.5)
    )
    assert not be.any(be.isnan(x))
    assert not be.any(be.isnan(L))


def test_robust_aimer_autograd_matches_finite_difference(set_test_backend):
    """G7: gradients flow through the polish. A finite-difference check on
    a simple ray-based quantity (stop-surface ray height) must match the
    backend gradient; this only makes sense on the torch backend."""
    if be.get_backend() != "torch":
        pytest.skip("Autograd check only applies to the torch backend.")

    import torch

    def build(radius_value):
        optic = CookeTriplet()
        optic.ray_tracer.set_aiming("robust")
        r = be.array(radius_value)
        r.requires_grad_(True)
        optic.surfaces[1].geometry.radius = r
        return optic, r

    optic, r = build(22.0)
    be.grad_mode.enable()
    rays = optic.trace(0.0, 1.0, optic.primary_wavelength, num_rays=5,
                        distribution="line_y")
    loss = rays.y.sum()
    loss.backward()
    grad_autograd = r.grad.item()
    be.grad_mode.disable()

    eps = 1e-4
    optic_p, _ = build(22.0 + eps)
    loss_p = float(
        optic_p.trace(0.0, 1.0, optic_p.primary_wavelength, num_rays=5,
                       distribution="line_y").y.sum()
    )
    optic_m, _ = build(22.0 - eps)
    loss_m = float(
        optic_m.trace(0.0, 1.0, optic_m.primary_wavelength, num_rays=5,
                       distribution="line_y").y.sum()
    )
    grad_fd = (loss_p - loss_m) / (2 * eps)

    assert abs(grad_autograd - grad_fd) / abs(grad_fd) < 1e-2
