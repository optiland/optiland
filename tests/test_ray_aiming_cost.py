"""Cost of failing solves in ray aiming and in the Newton-Raphson intersection.

A solve that cannot succeed (an unreachable aiming target, a dead ray in an
intersection batch) must give up as soon as further work provably cannot
change its outcome, and a ray that crosses an aspheric surface but misses its
base conic must still be traced.
"""

from __future__ import annotations

import numpy as np

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.even_asphere import EvenAsphere
from optiland.geometries.standard import StandardGeometry
from optiland.rays import RealRays
from optiland.rays.ray_aiming.iterative import IterativeRayAimer
from optiland.rays.ray_aiming.parameterization import LaunchParameterization
from optiland.rays.ray_aiming.robust import _EDGE_PROBES, RobustRayAimer
from optiland.samples.objectives import CookeTriplet

# Front surface of the fifth lens of US 10,001,627 B2, Example 1 (R, k = 0,
# A4 ... A20), and a ray of the 55.3 deg field arriving at it in the surface's
# local frame. The ray misses the base sphere but crosses the asphere.
_RIM_RADIUS = 2.089
_RIM_COEFFICIENTS = [
    0.0,
    -1.729039e-01,
    -3.671858e-01,
    7.689520e-01,
    -7.722536e-01,
    4.703523e-01,
    -1.808460e-01,
    4.281231e-02,
    -5.675622e-03,
    3.207934e-04,
]
_RIM_RAY = (0.0, 1.2993453585021189, -0.8607990730690558, 0.0, 0.512061242413161)


def _rim_asphere():
    return EvenAsphere(
        CoordinateSystem(), radius=_RIM_RADIUS, coefficients=_RIM_COEFFICIENTS
    )


def _rays(x, y, z, L, M):
    x, y, z, L, M = (be.array(np.atleast_1d(v).astype(float)) for v in (x, y, z, L, M))
    N = be.sqrt(1.0 - L**2 - M**2)
    n = len(x)
    return RealRays(x, y, z, L, M, N, be.ones(n), be.full((n,), 0.55))


class TestNewtonRaphsonSeed:
    def test_ray_missing_base_sphere_hits_asphere(self, set_test_backend):
        geometry = _rim_asphere()
        rays = _rays(*_RIM_RAY)

        # The conic seed does not exist for this ray ...
        seed = StandardGeometry.distance(geometry, rays)
        assert bool(be.all(be.isnan(seed)))

        # ... but the surface is crossed, and it is found.
        t = float(be.to_numpy(geometry.distance(rays)).reshape(-1)[0])
        assert np.isfinite(t) and t > 0.0
        x0, y0, z0, L, M = _RIM_RAY
        N = float(np.sqrt(1.0 - L**2 - M**2))
        hit_y, hit_z = y0 + t * M, z0 + t * N
        sag = float(be.to_numpy(geometry.sag(be.array(0.0), be.array(hit_y))))
        assert abs(sag - hit_z) < 1e-9

        # It is the first crossing along the ray.
        s = np.linspace(0.0, t, 2001)[:-1]
        ys = be.array(y0 + s * M)
        f = be.to_numpy(geometry.sag(be.zeros_like(ys), ys)) - (z0 + s * N)
        assert np.all(np.sign(f) == np.sign(f[0]))

    def test_conic_seeded_rays_unchanged_next_to_plane_seeded(self, set_test_backend):
        geometry = _rim_asphere()
        # A paraxial ray alone and next to the rim ray.
        near = (0.0, 0.2, -0.5, 0.0, 0.05)
        alone = geometry.distance(_rays(*near))
        both = geometry.distance(
            _rays(*(np.array([a, b]) for a, b in zip(near, _RIM_RAY, strict=True)))
        )
        assert float(be.to_numpy(alone)[0]) == float(be.to_numpy(both)[0])
        assert np.isfinite(float(be.to_numpy(both)[1]))

    def test_nan_ray_does_not_hold_the_batch(self, set_test_backend):
        geometry = _rim_asphere()
        rays = _rays([0.0, np.nan], [0.2, 0.2], [-0.5, -0.5], [0.0, 0.0], [0.05, 0.05])
        result = geometry._solve_distance_primal(rays)
        t = be.to_numpy(result.t)
        assert np.isnan(t[1])
        assert result.iterations < geometry.max_iter
        alone = geometry.distance(_rays(0.0, 0.2, -0.5, 0.0, 0.05))
        assert t[0] == float(be.to_numpy(alone)[0])


def _count_traces(monkeypatch):
    count = [0]
    original = IterativeRayAimer._trace_subset

    def counting(self, *args, **kwargs):
        count[0] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(IterativeRayAimer, "_trace_subset", counting)
    return count


class TestStalledAimingSolve:
    def test_unreachable_target_stops_after_a_rejected_line_search(
        self, set_test_backend, monkeypatch
    ):
        optic = CookeTriplet()
        aimer = IterativeRayAimer(optic, max_iter=50, tol=1e-9)
        seed = aimer._paraxial_aimer.aim_rays(
            (be.array([0.0]), be.array([0.0])),
            be.array([0.55]),
            (be.array([0.0]), be.array([0.0])),
        )
        count = _count_traces(monkeypatch)
        # A stop height of 100 mm is out of reach: the solve cannot converge.
        out = aimer._solve_core(
            *seed,
            be.array([0.55]),
            optic.surfaces.stop_index,
            True,
            be.array([0.0]),
            be.array([100.0]),
        )
        assert not bool(be.all(out[6]))
        # Before the stall exit this ran all 50 iterations: about 400 traces.
        assert count[0] < 60
        assert aimer.last_iterations < aimer.max_iter


class TestBatchedEdgeProbes:
    def test_batch_matches_one_probe_at_a_time(self, set_test_backend):
        optic = CookeTriplet()
        aimer = RobustRayAimer(optic, max_iter=50, tol=1e-9)
        stop = optic.surfaces.stop_index
        param = LaunchParameterization.for_optic(optic, True)
        seed = aimer._chief_seed(0.0, 1.0, be.array([0.55]))
        chief, _, _ = aimer._solve_chief(
            0.0, 1.0, 0.55, stop, True, None, param, seed=seed
        )
        r_stop = 1.0
        batched = aimer._solve_probes(0.55, stop, True, r_stop, chief, param)
        single = [
            aimer._solve_probe(0.55, stop, True, px, py, r_stop, chief, param)
            for px, py in _EDGE_PROBES
        ]
        assert [fb for _, fb in batched] == [fb for _, fb in single]
        np.testing.assert_allclose(
            np.array([p for p, _ in batched]),
            np.array([p for p, _ in single]),
            rtol=0.0,
            atol=1e-12,
        )
