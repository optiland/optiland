"""Tests for conic intersection root selection (issue #329).

The line-conic intersection is a quadratic whose two roots may both, one, or
neither lie on the physical surface. Issue #329 reported two distinct
failure mechanisms of the legacy "vertex-nearest root" selection when tracing
off-axis parabolic (OAP) mirrors:

1. For an off-axis section of a conic, *both* roots can be genuine forward
   hits on the infinite parent surface, and only the physical aperture
   identifies the used region. The vertex-nearest rule then reflects rays off
   the wrong (unused) part of the parent conic.
2. For rays crossing the focal region nearly perpendicular to the local
   optical axis, the vertex-nearest root can lie *behind* the ray, silently
   propagating rays backwards until they are lost.

The fix (``_conic_intersection_distance`` in
``optiland/geometries/standard.py``) admits a root only when it is solvable,
finite, in front of the ray (above a scale-aware floor), and on the sheet
described by the sag function (``1 - (1 + k) z / R >= 0``); roots inside the
surface's physical aperture are preferred, the nearest admissible root wins,
and rays without any admissible root fall back to the legacy vertex-nearest
root (NaN for a negative discriminant). ``StandardGratingGeometry`` and the
Newton-Raphson seed delegate to the same helper.
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland import geometries
from optiland.coordinate_system import CoordinateSystem
from optiland.optic import Optic
from optiland.physical_apertures import OffsetRadialAperture, RadialAperture
from optiland.rays import RealRays

from .utils import assert_allclose, assert_array_equal


class TestConicRootSelection:
    """Unit-level pins of the root-selection rule on ``StandardGeometry``."""

    def test_backward_root_trap_selects_forward_root(self, set_test_backend):
        """A vertex-nearest root behind the ray must never be selected (#329).

        Local-frame reduction of the 90-deg OAP collimator: a parabola with
        R = -25.4 (k = -1) and rays in its focal plane z = R/2 = -12.7
        traveling perpendicular to the local axis (N = 0). The quadratic
        roots are t = y0 -/+ 25.4; both crossings sit at z = -12.7, so the
        legacy vertex-nearest rule (|z| tie broken toward t1) chose the
        *backward* root t = y0 - 25.4 < 0 and teleported the rays behind
        the mirror. The forward rule must return t = y0 + 25.4 > 0. The y0
        values reproduce the misrouted rays diagnosed in #329, whose buggy
        distances were in [-34.78, -30.75].
        """
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=-25.4, conic=-1.0)

        y0_values = [-9.38, -8.0, -6.5, -5.35]
        n = len(y0_values)
        rays = RealRays(
            [0.0] * n,
            y0_values,
            [-12.7] * n,
            [0.0] * n,
            [-1.0] * n,
            [0.0] * n,
            [1.0] * n,
            [0.55] * n,
        )
        distance = geometry.distance(rays)
        t = be.to_numpy(distance)

        # the trap exists: for every ray the other quadratic root is behind
        backward_roots = np.asarray(y0_values) - 25.4
        assert np.all(backward_roots < 0)

        # the selected root is the forward crossing t = y0 + 25.4
        assert np.all(t > 0)
        assert_allclose(distance, np.asarray(y0_values) + 25.4, rtol=0, atol=1e-11)

        # every intersection lies on the parabola sheet z = sag(x, y)
        z_hit = rays.z + distance * rays.N
        sag = geometry.sag(rays.x + distance * rays.L, rays.y + distance * rays.M)
        assert_allclose(z_hit, sag, rtol=0, atol=1e-10)

    def test_backward_root_trap_analytic_value(self, set_test_backend):
        """Pin one hand-computable focal-plane crossing exactly (#329).

        For z = y^2 / (2 R) with R = -25.4, a ray at (0, -8, -12.7) traveling
        along -y solves a = 1, b = 16, c = 64 - 25.4^2, giving roots
        t = -8 -/+ 25.4. The forward crossing is t = 17.4, landing at
        (0, -25.4, -12.7), which satisfies sag(0, -25.4) = -12.7.
        """
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=-25.4, conic=-1.0)

        rays = RealRays(0.0, -8.0, -12.7, 0.0, -1.0, 0.0, 1.0, 0.55)
        distance = geometry.distance(rays)
        assert_allclose(distance, 17.4, rtol=0, atol=1e-11)

        y_hit = rays.y + distance * rays.M
        assert_allclose(y_hit, -25.4, rtol=0, atol=1e-11)

    def test_aperture_prefers_in_aperture_root(self, set_test_backend):
        """The aperture tier resolves the two-forward-hits ambiguity (#329).

        Mechanism 1 of #329: on the R = -25.4 parabola, a ray at
        (0, 30, -12.7) traveling along -y crosses the parent surface twice
        *in front* of the ray, at t = 4.6 (landing y = +25.4) and t = 55.4
        (landing y = -25.4). Both are genuine on-sheet hits, so without an
        aperture the nearest (4.6) is correct. When the surface's physical
        aperture covers only the y = -25.4 off-axis section, the in-aperture
        root t = 55.4 must be selected even though it is farther.
        """
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=-25.4, conic=-1.0)
        aperture = OffsetRadialAperture(r_max=12.7, offset_y=-25.4)

        rays = RealRays(0.0, 30.0, -12.7, 0.0, -1.0, 0.0, 1.0, 0.55)
        assert_allclose(geometry.distance(rays), 4.6, rtol=0, atol=1e-11)

        rays = RealRays(0.0, 30.0, -12.7, 0.0, -1.0, 0.0, 1.0, 0.55)
        assert_allclose(
            geometry.distance(rays, aperture=aperture), 55.4, rtol=0, atol=1e-11
        )

    def test_ray_starting_on_surface_gets_second_crossing(self, set_test_backend):
        """A ray starting exactly on the surface must not return t = 0.

        Starting at (0, 25.4, -12.7) -- exactly on the R = -25.4 parabola --
        and traveling along -y, the quadratic has roots t = 0 (the starting
        point itself, c = 0 exactly) and t = 50.8 (the genuine second
        crossing at y = -25.4). The scale-aware forward floor must reject the
        t = 0 self-intersection; the legacy code could return 0 or swap to
        it, freezing the ray in place.
        """
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=-25.4, conic=-1.0)

        rays = RealRays(0.0, 25.4, -12.7, 0.0, -1.0, 0.0, 1.0, 0.55)
        distance = geometry.distance(rays)

        assert be.to_numpy(distance)[0] > 1.0  # never the t = 0 root
        assert_allclose(distance, 50.8, rtol=0, atol=1e-11)

    def test_sphere_far_hemisphere_rejected(self, set_test_backend):
        """The sag-sheet test rejects the far hemisphere of a sphere.

        A concave sphere with R = -100 (center at z = -100) is crossed by an
        axial ray from z = -250 at z = -200 (t = 50, far hemisphere) and
        z = 0 (t = 250, the vertex the sag function describes). The sheet
        test ``1 - z / R >= 0`` rejects z = -200 (gives -1), so the distance
        must be exactly 250. A naive smallest-positive-t rule would return
        50, i.e. hit the *inside* of the far hemisphere.
        """
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=-100.0, conic=0.0)

        rays = RealRays(0.0, 0.0, -250.0, 0.0, 0.0, 1.0, 1.0, 0.55)
        distance = geometry.distance(rays)

        # all quadratic coefficients are exact in float64, so the root is too
        assert be.to_numpy(distance)[0] == 250.0

    def test_hyperboloid_phantom_sheet_rejected(self, set_test_backend):
        """A k < -1 hyperboloid's detached second sheet is never selected.

        Hubble-primary parameters (R = -11040, k = -1.0012) put the phantom
        second sheet of the quadric near z = 2 R / (1 + k) = +1.84e7. A
        slightly tilted ray from z = -1000 reaches both quadric roots
        *forward*: the genuine near root (t ~ 1000, landing r ~ 101) and the
        phantom (t ~ 1.84e7, landing r ~ 1.85e4). The sheet test must reject
        the phantom even when an aperture "contains" its landing point -- in
        the annular case the near root lands inside r_min (out of aperture)
        while the phantom lands in the annulus, so a selection that let the
        aperture tier hop sheets would pick the phantom.
        """
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=-11040.0, conic=-1.0012)
        N = float(np.sqrt(1 - 0.001**2))

        def make_rays():
            return RealRays(0.0, 100.0, -1000.0, 0.0, 0.001, N, 1.0, 0.55)

        huge_disk = RadialAperture(r_max=1e9)
        annulus = OffsetRadialAperture(r_max=1e9, r_min=177.8)

        t_none = geometry.distance(make_rays())
        t_disk = geometry.distance(make_rays(), aperture=huge_disk)
        t_annulus = geometry.distance(make_rays(), aperture=annulus)

        # near root in every aperture configuration, never the phantom
        assert be.to_numpy(t_none)[0] < 2000.0
        assert_array_equal(t_none, t_disk)
        assert_array_equal(t_none, t_annulus)

        # the chosen intersection lies on the sag sheet ...
        rays = make_rays()
        x_hit = rays.x + t_none * rays.L
        y_hit = rays.y + t_none * rays.M
        z_hit = rays.z + t_none * rays.N
        assert_allclose(z_hit, geometry.sag(x_hit, y_hit), rtol=0, atol=1e-8)

        # ... even though the annulus does NOT contain it (setup soundness:
        # the aperture tier was empty, and selection still refused to hop
        # to the in-annulus phantom root)
        inside = be.to_numpy(annulus.contains(x_hit, y_hit))
        assert not np.any(inside)

    def test_true_miss_returns_nan(self, set_test_backend):
        """A ray missing the surface entirely yields NaN, not a fake hit.

        The line x = 0, y = 25 passes 25 units from the axis of a sphere of
        radius 10 (R = +10, centered at z = +10): the discriminant is
        negative and no intersection exists. The legacy NaN contract for a
        genuine miss must survive the new root selection.
        """
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=10.0, conic=0.0)

        rays = RealRays(0.0, 25.0, -5.0, 0.0, 0.0, 1.0, 1.0, 0.55)
        distance = geometry.distance(rays)
        assert np.all(np.isnan(be.to_numpy(distance)))

    def test_degenerate_a_axial_parabola_with_aperture(self, set_test_backend):
        """The a = 0 linear branch survives the aperture-preference tier.

        Mirrors ``test_distance_parabola_axial_ray``: for an axial ray on a
        parabola the quadratic's leading coefficient vanishes exactly and
        only the linear root t = c / q = 49.5 exists. Passing an aperture
        that contains the hit at (10, 0) must leave the result unchanged --
        the single admissible root is simply promoted to the aperture tier.
        """
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=-100.0, conic=-1.0)
        aperture = RadialAperture(r_max=15.0)

        rays = RealRays(10.0, 0.0, -50.0, 0.0, 0.0, 1.0, 1.0, 0.55)
        distance = geometry.distance(rays, aperture=aperture)
        assert_allclose(distance, 49.5, rtol=0, atol=1e-12)

    def test_planar_branch_keeps_signed_distance(self, set_test_backend):
        """The planar early-return keeps signed (negative) distances.

        For an infinite radius the intersection is the single plane crossing
        t = -z / N, which is legitimately *negative* for a ray behind the
        plane traveling away (mirroring the Plane test's -1.5 case). The
        forward-only admissibility rule applies to conic roots only; the
        planar branch must not gain it, and must ignore the aperture (a
        plane has exactly one crossing to choose from).
        """
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=be.inf, conic=0.0)

        rays = RealRays(1.0, 2.0, -1.5, 0.0, 0.0, -1.0, 1.0, 0.0)
        assert_allclose(geometry.distance(rays), -1.5, rtol=0, atol=1e-12)

        # even an aperture excluding the hit point leaves the plane result
        rays = RealRays(1.0, 2.0, -1.5, 0.0, 0.0, -1.0, 1.0, 0.0)
        tiny_aperture = RadialAperture(r_max=0.1)
        assert_allclose(
            geometry.distance(rays, aperture=tiny_aperture), -1.5, rtol=0, atol=1e-12
        )

    def test_batch_composition_independence(self, set_test_backend):
        """A ray's distance must not depend on its batch companions.

        The legacy implementation gated the root swap on batch-level
        ``bool()`` reductions, so adding a steep or missing ray to a batch
        could change the root selected for an unrelated well-behaved ray.
        All masking is now per ray: the probe ray must get the bitwise
        identical distance alone and inside a batch that also contains a
        grazing ray and a true miss (NaN lane).
        """
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=-12.0, conic=0.5)

        probe = RealRays(1.0, 2.0, -3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        t_alone = be.to_numpy(geometry.distance(probe))

        M_grazing = 0.9999
        N_grazing = float(np.sqrt(1 - M_grazing**2))
        batch = RealRays(
            [1.0, 0.0, 0.0],
            [2.0, -20.0, 40.0],
            [-3.0, -3.0, -5.0],
            [0.0, 0.0, 0.0],
            [0.0, M_grazing, 0.0],
            [1.0, N_grazing, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        )
        t_batch = be.to_numpy(geometry.distance(batch))

        # companions behaved as designed: one finite grazing hit, one miss
        assert np.isfinite(t_batch[1])
        assert np.isnan(t_batch[2])

        # exact equality: same lane-wise arithmetic regardless of the batch
        assert t_alone[0] == t_batch[0]


class TestGratingDelegation:
    """``StandardGratingGeometry.distance`` delegates to the shared helper."""

    def test_grating_matches_standard_distance(self, set_test_backend):
        """Grating and standard conics agree bitwise for identical rays.

        The grating geometry previously carried its own pre-#648 copy of the
        textbook quadratic formula (numerically unstable near a = 0) and
        missed the #329 root-selection fix entirely. Both classes now call
        ``_conic_intersection_distance``, so a batch spanning the trap ray,
        the aperture-swap ray, an ordinary oblique hit, and a near-degenerate
        axial ray must give bitwise identical distances -- with and without
        an aperture.
        """
        cs = CoordinateSystem()
        standard = geometries.StandardGeometry(cs, radius=-25.4, conic=-1.0)
        grating = geometries.StandardGratingGeometry(
            cs,
            radius=-25.4,
            grating_order=1,
            grating_period=10.0,
            groove_orientation_angle=0.0,
            conic=-1.0,
        )
        aperture = OffsetRadialAperture(r_max=12.7, offset_y=-25.4)

        x = [0.0, 0.0, 1.0, 0.0]
        y = [-8.0, 30.0, 2.0, -3.0]
        z = [-12.7, -12.7, -3.0, -10.0]
        L = [0.0, 0.0, 0.0, 0.0]
        M = [-1.0, -1.0, 0.0, 2e-6]
        N = [0.0, 0.0, 1.0, float(np.sqrt(1 - 4e-12))]

        def make_rays():
            return RealRays(x, y, z, L, M, N, [1.0] * 4, [0.55] * 4)

        assert_array_equal(
            standard.distance(make_rays()), grating.distance(make_rays())
        )
        assert_array_equal(
            standard.distance(make_rays(), aperture=aperture),
            grating.distance(make_rays(), aperture=aperture),
        )

        # sanity: the batch exercises the #329 mechanisms (trap ray forward,
        # aperture swap to the far root)
        assert_allclose(
            grating.distance(make_rays())[:2], [17.4, 4.6], rtol=0, atol=1e-11
        )
        assert_allclose(
            grating.distance(make_rays(), aperture=aperture)[:2],
            [17.4, 55.4],
            rtol=0,
            atol=1e-11,
        )

    def test_grating_distance_parabola_near_degenerate_a(self, set_test_backend):
        """Near-degenerate-a stability now also holds for gratings.

        Port of ``test_distance_parabola_near_degenerate_a`` to
        ``StandardGratingGeometry``, which previously used the naive
        ``(-b +/- sqrt(d)) / (2a)`` formula: for near-axial rays on a
        parabola, a is a tiny nonzero float and the cancellation in the
        numerator is amplified by orders of magnitude. Every intersection
        must lie exactly on the parabola (y^2 = 2 R z) and the sag must vary
        monotonically as M sweeps through zero.
        """
        cs = CoordinateSystem()
        geometry = geometries.StandardGratingGeometry(
            cs,
            radius=-250.0,
            grating_order=1,
            grating_period=10.0,
            groove_orientation_angle=0.0,
            conic=-1.0,
        )

        y0, z0 = -33.49, -110.0
        M_values = [-4e-6, -2e-6, -5e-7, 0.0, 5e-7, 2e-6, 4e-6]
        N_values = [np.sqrt(1 - m**2) for m in M_values]

        rays = RealRays(
            [0.0] * len(M_values),
            [y0] * len(M_values),
            [z0] * len(M_values),
            [0.0] * len(M_values),
            M_values,
            N_values,
            [1.0] * len(M_values),
            [0.55] * len(M_values),
        )
        t = geometry.distance(rays)

        M_arr = be.array(M_values)
        N_arr = be.array(N_values)
        y_hit = y0 + M_arr * t
        z_hit = z0 + N_arr * t

        # every intersection must lie exactly on the parabola: y**2 = 2*R*z
        assert_allclose(y_hit**2, 2 * geometry.radius * z_hit)

        # the sag must vary smoothly (monotonically) as M sweeps through zero
        z_np = be.to_numpy(z_hit)
        diffs = np.diff(z_np)
        assert np.all(diffs > 0) or np.all(diffs < 0)


class TestNewtonRaphsonSeed:
    """The Newton-Raphson conic seed receives the surface aperture."""

    def test_newton_raphson_seed_receives_aperture(self, set_test_backend):
        """The NR iteration starts in the basin of the in-aperture root.

        An even asphere with all-zero coefficients is exactly its base conic,
        so the converged NR distance equals the conic-seed root. On the #329
        mechanism-1 setup (two forward on-sheet roots at 4.6 and 55.4, the
        far one inside the aperture), forwarding the aperture to the seed
        must land the iteration on 55.4; without the aperture the nearest
        root 4.6 is kept. Newton-Raphson cannot recover from a seed in the
        wrong basin, which is why the seed itself must already select the
        physical branch.
        """
        cs = CoordinateSystem()
        geometry = geometries.EvenAsphere(
            cs, radius=-25.4, conic=-1.0, coefficients=[0.0]
        )
        aperture = OffsetRadialAperture(r_max=12.7, offset_y=-25.4)

        rays = RealRays(0.0, 30.0, -12.7, 0.0, -1.0, 0.0, 1.0, 0.55)
        assert_allclose(geometry.distance(rays), 4.6, rtol=0, atol=1e-9)

        rays = RealRays(0.0, 30.0, -12.7, 0.0, -1.0, 0.0, 1.0, 0.55)
        assert_allclose(
            geometry.distance(rays, aperture=aperture), 55.4, rtol=0, atol=1e-9
        )


class TestTorchAutogradSafety:
    """Gradients through the guarded root selection stay finite (torch only)."""

    @staticmethod
    def _grad_rays(x, y, z, L, M, N):
        """Build single- or multi-lane RealRays whose pose tensors are
        autograd leaves, returning (rays, leaves)."""
        leaves = [
            be.array(np.atleast_1d(np.asarray(v, dtype=float)))
            for v in (x, y, z, L, M, N)
        ]
        for leaf in leaves:
            leaf.requires_grad_(True)
        n = leaves[0].shape[0]
        rays = RealRays(*leaves, be.ones(n), be.full((n,), 0.55))
        return rays, leaves

    def test_gradients_finite_normal_case(self, set_test_backend):
        """An ordinary conic hit backpropagates finite pose gradients.

        The guards (``maximum(d, eps)`` before the sqrt, floored
        denominators before every division) must keep the backward pass
        clean even where the unguarded expressions would be singular.
        """
        if be.get_backend() != "torch":
            pytest.skip("autograd safety is a torch-backend property")
        import torch

        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=-12.0, conic=0.5)
        rays, leaves = self._grad_rays(1.0, 2.0, -3.0, 0.0, 0.0, 1.0)

        t = geometry.distance(rays)
        assert_allclose(t, 2.7888809636986154)
        t.sum().backward()

        for leaf in leaves:
            assert leaf.grad is not None
            assert torch.all(torch.isfinite(leaf.grad))

    def test_gradients_finite_aperture_swap(self, set_test_backend):
        """The aperture-preferred far root carries exact pose gradients.

        Mechanism-1 setup (#329): with the aperture, the selected root is
        analytically t = y0 + sqrt(2 R z0), so dt/dy0 = 1 and
        dt/dz0 = R / sqrt(2 R z0) = -1 at (y0, z0) = (30, -12.7). The
        boolean aperture tier must not detach or pollute the gradient of
        the selected root.
        """
        if be.get_backend() != "torch":
            pytest.skip("autograd safety is a torch-backend property")
        import torch

        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=-25.4, conic=-1.0)
        aperture = OffsetRadialAperture(r_max=12.7, offset_y=-25.4)
        rays, leaves = self._grad_rays(0.0, 30.0, -12.7, 0.0, -1.0, 0.0)

        t = geometry.distance(rays, aperture=aperture)
        assert_allclose(t, 55.4, rtol=0, atol=1e-11)
        t.sum().backward()

        for leaf in leaves:
            assert leaf.grad is not None
            assert torch.all(torch.isfinite(leaf.grad))

        _, y, z, _, _, _ = leaves
        assert_allclose(y.grad, 1.0, rtol=0, atol=1e-9)
        assert_allclose(z.grad, -1.0, rtol=0, atol=1e-9)

    def test_nan_miss_lane_does_not_pollute_gradients(self, set_test_backend):
        """A NaN miss lane must not leak NaN into valid lanes' gradients.

        A batch mixes a genuine hit (t = 5 on the R = +10 sphere, the far
        hemisphere root t = 25 sheet-rejected) with a true miss (negative
        discriminant, NaN forward value). Summing only the finite lanes and
        backpropagating must produce finite gradients everywhere: the
        guarded radicand and denominators keep the masked-out miss lane from
        contaminating the batch through the backward pass.
        """
        if be.get_backend() != "torch":
            pytest.skip("autograd safety is a torch-backend property")
        import torch

        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=10.0, conic=0.0)
        rays, leaves = self._grad_rays(
            [0.0, 0.0],
            [0.0, 25.0],
            [-5.0, -5.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
        )

        t = geometry.distance(rays)
        finite = torch.isfinite(t)
        assert finite.tolist() == [True, False]
        assert_allclose(t[0], 5.0, rtol=0, atol=1e-12)

        t[finite].sum().backward()

        for leaf in leaves:
            assert leaf.grad is not None
            assert torch.all(torch.isfinite(leaf.grad))


def _build_double_oap_relay(rx_angle_deg):
    """4F double-OAP relay from #329 (narcuak's reproducer).

    Two identical parabolic mirrors (R = 76.2, conic = -1) used off axis
    through offset radial apertures, re-imaging a collimated beam 1:1. The
    second mirror can be tilted by ``rx_angle_deg`` -- on the buggy code
    rx = 0 lost about half the fan while rx = -10 traced fully, because the
    tilt happened to move the vertex-nearest root onto the used section.
    """
    curvature = 1.312335958005249499e-02
    radius = 1 / curvature  # 76.2
    ap1 = OffsetRadialAperture(r_min=0, r_max=25.4, offset_y=76.2)
    ap2 = OffsetRadialAperture(r_min=0, r_max=25.4, offset_y=-76.2)

    relay = Optic()
    relay.surfaces.add(index=0, thickness=be.inf)
    relay.surfaces.add(
        index=1, radius=be.inf, thickness=152.4, material="air", semi_aperture=22.86
    )
    relay.surfaces.add(
        index=2,
        radius=-radius,
        thickness=-38.1 * 2,
        material="mirror",
        is_stop=True,
        conic=-1,
        dy=-76.2,
        aperture=ap1,
    )
    relay.surfaces.add(
        index=3,
        radius=radius,
        thickness=38.1 * 3,
        material="mirror",
        is_stop=False,
        conic=-1,
        dy=-76.2,
        aperture=ap2,
        rx=np.deg2rad(rx_angle_deg),
    )
    relay.surfaces.add(
        index=4, thickness=50, material="air", aperture=100, dy=-76.2 * 2
    )
    relay.set_aperture(aperture_type="EPD", value=15.0)
    relay.fields.set_type(field_type="angle")
    relay.fields.add(y=0)
    relay.wavelengths.add(value=0.4861, is_primary=True)
    return relay


def _build_oap_collimator():
    """90-deg OAP collimator from #329 (hellerkopf's reproducer).

    A point source at the focus of a 90-deg off-axis parabola
    (RFL = 25.4); correctly reflected rays leave collimated toward +y, to a
    screen at y = 40. On the buggy code about half the rays reflected off
    the wrong intersection branch and left the mirror away from the screen.
    """
    lens = Optic()
    rfl = 25.4
    aperture1 = OffsetRadialAperture(r_max=12.7, offset_y=rfl)
    aperture2 = OffsetRadialAperture(r_max=12.7, offset_y=0)

    lens.surfaces.add(index=0, z=0, radius=be.inf)
    lens.surfaces.add(index=1, z=10, is_stop=True)
    lens.surfaces.add(
        index=2,
        z=0,
        y=-rfl / 2,
        radius=-rfl,
        conic=-1,
        aperture=aperture1,
        material="mirror",
        rx=np.pi / 2,
    )
    lens.surfaces.add(
        index=3, z=rfl, y=40, radius=be.inf, aperture=aperture2, rx=np.pi / 2
    )
    lens.set_aperture(aperture_type="EPD", value=5.0)
    lens.fields.set_type(field_type="object_height")
    lens.fields.add(y=-0.1)
    lens.fields.add(y=0)
    lens.fields.add(y=0.1)
    lens.wavelengths.add(value=0.633, is_primary=True)
    return lens


def _alive_mask(rays):
    """Per-ray survival mask: positive intensity and finite state."""
    i = be.to_numpy(rays.i)
    finite = (
        np.isfinite(i)
        & np.isfinite(be.to_numpy(rays.y))
        & np.isfinite(be.to_numpy(rays.z))
        & np.isfinite(be.to_numpy(rays.M))
    )
    return (i > 0) & finite


class TestOAPSystemTraces:
    """End-to-end traces of the two #329 reproducer systems."""

    @pytest.mark.parametrize("rx_deg", [0.0, -2.0, -5.0, -10.0])
    def test_double_oap_relay_traces_all_rays(self, set_test_backend, rx_deg):
        """Every fan ray survives the 4F double-OAP relay at every tilt.

        The #329 bug signature: at rx = 0 about half of a 41-ray line_y fan
        vanished (backward/wrong-branch intersections clipped by the
        apertures), while rx = -10 deg traced fully. With per-ray forward
        on-sheet root selection plus the aperture tier, all rays must
        survive at every tilt.
        """
        relay = _build_double_oap_relay(rx_deg)
        rays = relay.trace(
            Hx=0, Hy=0, wavelength=0.4861, num_rays=41, distribution="line_y"
        )
        alive = _alive_mask(rays)
        assert int(alive.sum()) == 41

    def test_double_oap_relay_recollimates(self, set_test_backend):
        """The aligned 4F relay re-collimates a collimated input.

        Two identical parabolas separated by 2f re-image a collimated beam
        into a collimated beam. For the aligned system (rx = 0) the output
        direction cosines must be uniform across the fan; the observed
        spread is ~2e-15 (pinned one order looser at 2e-14). A wrong-branch
        intersection on either mirror would produce degree-scale spread.
        """
        relay = _build_double_oap_relay(0.0)
        rays = relay.trace(
            Hx=0, Hy=0, wavelength=0.4861, num_rays=41, distribution="line_y"
        )
        alive = _alive_mask(rays)
        assert int(alive.sum()) == 41

        for component in (rays.L, rays.M, rays.N):
            c = be.to_numpy(component)[alive]
            assert np.all(np.isfinite(c))
            assert c.max() - c.min() < 2e-14

        # after an even number of mirrors the beam travels toward +z again
        assert np.all(be.to_numpy(rays.N)[alive] > 0.999999)

    @pytest.mark.parametrize("Hy", [-1.0, 0.0, 1.0])
    def test_oap_collimator_all_rays_toward_screen(self, set_test_backend, Hy):
        """Every collimator ray survives and leaves toward the screen.

        The #329 bug signature: about half the rays reflected off the wrong
        branch of the parent parabola and left the mirror with M < 0, away
        from the screen at y = +40. For every field, all 21 rays must stay
        alive and travel toward +y.
        """
        lens = _build_oap_collimator()
        rays = lens.trace(
            Hx=0, Hy=Hy, wavelength=0.633, num_rays=21, distribution="line_y"
        )
        alive = _alive_mask(rays)
        assert int(alive.sum()) == 21
        assert np.all(be.to_numpy(rays.M)[alive] > 0)

    def test_oap_collimator_on_axis_collimation(self, set_test_backend):
        """A point source at the parabola focus collimates exactly.

        For the on-axis field the source sits at the exact focal point of
        the parabola, so the reflected beam is perfectly collimated: the
        angular spread max |direction - mean direction| across the fan is
        pure round-off, observed at ~4e-16 (pinned one order looser at
        5e-15). Any wrong-root reflection breaks this by many orders of
        magnitude.
        """
        lens = _build_oap_collimator()
        rays = lens.trace(
            Hx=0, Hy=0.0, wavelength=0.633, num_rays=21, distribution="line_y"
        )
        alive = _alive_mask(rays)
        assert int(alive.sum()) == 21

        directions = np.stack(
            [
                be.to_numpy(rays.L)[alive],
                be.to_numpy(rays.M)[alive],
                be.to_numpy(rays.N)[alive],
            ],
            axis=1,
        )
        deviation = np.abs(directions - directions.mean(axis=0)).max()
        assert deviation < 5e-15

    @pytest.mark.parametrize("Hy", [-1.0, 0.0, 1.0])
    def test_oap_collimator_beam_lands_inside_screen(self, set_test_backend, Hy):
        """The collimated beam lands compactly inside the screen aperture.

        The screen is the y = 40 plane (a plane rotated by rx = pi/2) with a
        12.7-radius aperture centered on the OAP output axis; in global
        coordinates its local radial coordinate is
        sqrt(x^2 + (z - 25.4)^2). The 5 mm collimated beam must land well
        inside it (observed max ~7.4). A ray taking the phantom or backward
        branch on the mirror lands tens of millimetres away or never
        reaches the screen at all.
        """
        lens = _build_oap_collimator()
        rays = lens.trace(
            Hx=0, Hy=Hy, wavelength=0.633, num_rays=21, distribution="line_y"
        )
        alive = _alive_mask(rays)
        assert int(alive.sum()) == 21

        x = be.to_numpy(rays.x)[alive]
        y = be.to_numpy(rays.y)[alive]
        z = be.to_numpy(rays.z)[alive]

        # all rays landed on the screen plane ...
        assert_allclose(y, np.full_like(y, 40.0), rtol=0, atol=1e-9)

        # ... within the screen's physical aperture
        r_local = np.sqrt(x**2 + (z - 25.4) ** 2)
        assert np.all(r_local <= 12.7)
