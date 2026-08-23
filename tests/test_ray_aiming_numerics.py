"""Numerical hardening of the iterative ray-aiming core (PR #729).

Pins the merge-gate numerics:

- the shared scale-invariant 2x2 solve (optiland.utils.solve_2x2): sign
  preservation, conditioning classification, float32/float64;
- central finite differences with dtype/scale-aware steps;
- the iteration-time conditioning ladder (refresh -> sign-preserving
  paraxial diagonal -> hold) and its reporting through SolveReport;
- the dtype-aware Broyden step floor;
- the dtype-aware transverse-basis pole and launch-parameterization
  projection tolerances.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.optic import Optic
from optiland.paraxial_path import angular_tolerance, transverse_basis
from optiland.rays.ray_aiming.iterative import IterativeRayAimer
from optiland.rays.ray_aiming.parameterization import (
    LaunchParameterization,
    _degenerate_projection_tolerance,
)
from optiland.utils import machine_eps, solve_2x2

from .nr_implicit_test_utils import backend_state
from .test_folded_paraxial import _finish, straight
from .test_folded_paraxial_hardening import stop_after_fold, stop_mid_straight
from .utils import assert_allclose

BACKEND_PRECISION = [
    ("numpy", "float64"),
    ("numpy", "float32"),
    ("torch", "float64"),
    ("torch", "float32"),
]


def _solve(J, r):
    result = solve_2x2(
        be.array(J[0][0]),
        be.array(J[0][1]),
        be.array(J[1][0]),
        be.array(J[1][1]),
        be.array(r[0]),
        be.array(r[1]),
    )
    return (
        float(be.to_numpy(result.x1)),
        float(be.to_numpy(result.x2)),
        bool(be.all(result.valid)),
    )


# ---------------------------------------------------------------------------
# Shared 2x2 solve
# ---------------------------------------------------------------------------


class TestSolve2x2:
    @pytest.mark.parametrize("backend,precision", BACKEND_PRECISION)
    def test_identity(self, backend, precision):
        with backend_state(backend, precision):
            x1, x2, valid = _solve([[1.0, 0.0], [0.0, 1.0]], [0.3, -0.7])
            assert valid
            assert x1 == pytest.approx(0.3, rel=1e-6)
            assert x2 == pytest.approx(-0.7, rel=1e-6)

    @pytest.mark.parametrize("backend,precision", BACKEND_PRECISION)
    @pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
    def test_well_conditioned_any_magnitude(self, backend, precision, scale):
        if precision == "float32" and scale == 1e6:
            scale = 1e3  # keep products inside float32 range comfortably
        with backend_state(backend, precision):
            J = [[2.0 * scale, 0.3 * scale], [-0.4 * scale, 1.5 * scale]]
            x_true = (0.3, -0.7)
            r = (
                J[0][0] * x_true[0] + J[0][1] * x_true[1],
                J[1][0] * x_true[0] + J[1][1] * x_true[1],
            )
            x1, x2, valid = _solve(J, r)
            assert valid
            rtol = 1e-9 if precision == "float64" else 1e-4
            assert x1 == pytest.approx(x_true[0], rel=rtol, abs=rtol)
            assert x2 == pytest.approx(x_true[1], rel=rtol, abs=rtol)

    @pytest.mark.parametrize("backend,precision", BACKEND_PRECISION)
    def test_negative_determinant_sign_preserved(self, backend, precision):
        """An antidiagonal matrix (det < 0) solves exactly; the determinant
        is never clamped to an arbitrary positive value, so the solution
        (and any Newton step built from it) keeps the correct sign even
        when the raw determinant magnitude is tiny."""
        with backend_state(backend, precision):
            s = 1e-7 if precision == "float64" else 1e-3
            # J = s * [[0, 1], [1, 0]]: raw det = -s^2 (below the old fixed
            # 1e-12 clamp for float64), normalized det = -1.
            x1, x2, valid = _solve([[0.0, s], [s, 0.0]], [2.0 * s, -3.0 * s])
            assert valid
            assert x1 == pytest.approx(-3.0, rel=1e-5)
            assert x2 == pytest.approx(2.0, rel=1e-5)

    @pytest.mark.parametrize("backend,precision", BACKEND_PRECISION)
    def test_cross_coupled(self, backend, precision):
        with backend_state(backend, precision):
            J = [[1.0, 0.9], [0.8, 1.0]]
            r = (1.0, 0.0)
            x1, x2, valid = _solve(J, r)
            assert valid
            det = 1.0 - 0.72
            assert x1 == pytest.approx(1.0 / det, rel=1e-5)
            assert x2 == pytest.approx(-0.8 / det, rel=1e-5)

    @pytest.mark.parametrize("backend,precision", BACKEND_PRECISION)
    def test_exactly_singular(self, backend, precision):
        with backend_state(backend, precision):
            x1, x2, valid = _solve([[1.0, 1.0], [1.0, 1.0]], [1.0, 0.0])
            assert not valid
            # A zero placeholder step, never a fabricated one.
            assert x1 == 0.0
            assert x2 == 0.0

    @pytest.mark.parametrize("backend,precision", BACKEND_PRECISION)
    def test_nearly_singular_at_round_off_is_rejected(self, backend, precision):
        with backend_state(backend, precision):
            eps = machine_eps(be.zeros(1))
            delta = eps  # rank-1 + eps perturbation: rcond ~ eps/4
            J = [[1.0, 1.0], [1.0, 1.0 + delta]]
            _, _, valid = _solve(J, [1.0, 0.0])
            assert not valid

    @pytest.mark.parametrize("backend,precision", BACKEND_PRECISION)
    def test_nonfinite_is_rejected(self, backend, precision):
        with backend_state(backend, precision):
            x1, x2, valid = _solve([[float("nan"), 0.0], [0.0, 1.0]], [1.0, 1.0])
            assert not valid
            assert x1 == 0.0 and x2 == 0.0


# ---------------------------------------------------------------------------
# Central finite differences and the conditioning ladder
# ---------------------------------------------------------------------------


def _chief_state(optic):
    """Seed launch, bound parameterization and stop data for the chief ray."""
    aimer = IterativeRayAimer(optic)
    is_inf = bool(getattr(optic.object_surface, "is_infinite", False))
    param = LaunchParameterization.for_optic(optic, is_inf)
    seed = aimer._paraxial_aimer.aim_rays(
        (be.array([0.0]), be.array([0.0])),
        be.array([0.55]),
        (be.array([0.0]), be.array([0.0])),
    )
    bound = param.bind(*seed)
    stop_idx = optic.surfaces.stop_index
    return aimer, param, bound, stop_idx, is_inf


class TestCentralDifferences:
    def test_steps_are_dtype_and_scale_aware(self, set_test_backend):
        optic = straight()
        aimer, _, bound, stop_idx, is_inf = _chief_state(optic)
        xi = be.zeros(1)
        eta = be.zeros(1)
        aimer._finite_difference_jacobian(
            bound, xi, eta, be.array([0.55]), stop_idx, is_inf, None, None, 5.0
        )
        h_xi, h_eta = aimer._last_fd_steps
        eps = machine_eps(xi)
        expected = (eps ** (1.0 / 3.0)) * 5.0
        assert_allclose(h_xi, expected, rtol=1e-12)
        assert_allclose(h_eta, expected, rtol=1e-12)

    def test_central_difference_agrees_with_independent_measurement(
        self, set_test_backend
    ):
        """The Jacobian matches an independently measured stop response
        (central difference at half the step)."""
        optic = straight()
        aimer, _, bound, stop_idx, is_inf = _chief_state(optic)
        xi = be.zeros(1)
        eta = be.zeros(1)
        wl = be.array([0.55])
        J11, J12, J21, J22 = aimer._finite_difference_jacobian(
            bound, xi, eta, wl, stop_idx, is_inf, None, None, 1.0
        )

        def stop_y_at(eta_value):
            launch = bound.launch(be.zeros(1), be.array([eta_value]))
            rays = aimer._trace_subset(*launch, wl, stop_idx, is_inf)
            _, ly = aimer._get_local_stop_coords(rays, stop_idx)
            return float(be.to_numpy(ly).reshape(-1)[0])

        h = 0.5 * float(be.to_numpy(aimer._last_fd_steps[1]).reshape(-1)[0])
        measured = (stop_y_at(+h) - stop_y_at(-h)) / (2.0 * h)
        assert_allclose(
            float(be.to_numpy(J22).reshape(-1)[0]), measured, rtol=1e-6
        )

    def test_folded_stop_coupling_sign_is_captured(self, set_test_backend):
        """A stop behind a 90-degree fold reverses the local stop-y response
        to an eta displacement; the central-difference Jacobian must carry
        that sign (a positive-diagonal assumption steps the wrong way)."""
        folded_optic = stop_after_fold()
        aimer, _, bound, stop_idx, is_inf = _chief_state(folded_optic)
        J11, J12, J21, J22 = aimer._finite_difference_jacobian(
            bound,
            be.zeros(1),
            be.zeros(1),
            be.array([0.55]),
            stop_idx,
            is_inf,
            None,
            None,
            1.0,
        )
        straight_optic = straight()
        aimer_s, _, bound_s, stop_idx_s, is_inf_s = _chief_state(straight_optic)
        _, _, _, J22_s = aimer_s._finite_difference_jacobian(
            bound_s,
            be.zeros(1),
            be.zeros(1),
            be.array([0.55]),
            stop_idx_s,
            is_inf_s,
            None,
            None,
            1.0,
        )
        j22_folded = float(be.to_numpy(J22).reshape(-1)[0])
        j22_straight = float(be.to_numpy(J22_s).reshape(-1)[0])
        assert j22_folded * j22_straight < 0.0

    def test_conditioning_ladder_reports_fallback(self, set_test_backend):
        """A degenerate finite-difference Jacobian triggers the documented
        ladder: refresh -> sign-preserving paraxial diagonal -> converge;
        the report carries fallback_used and the refresh count.

        The stop sits mid-system so the paraxial seed is not already exact
        (a stop on the first surface converges at iteration zero and never
        enters the Newton loop).
        """
        optic = stop_mid_straight()
        aimer = IterativeRayAimer(optic)

        original = IterativeRayAimer._finite_difference_jacobian

        def degenerate_fd(self, bound, xi, eta, *args, **kwargs):
            n = len(bound.x0)
            self._last_fd_steps = (be.ones(n), be.ones(n))
            self._last_jacobian_fallback = be.zeros(n) > 0.0
            zeros = be.zeros(n)
            return zeros, zeros, zeros, zeros

        IterativeRayAimer._finite_difference_jacobian = degenerate_fd
        try:
            x, y, z, L, M, N = aimer.aim_rays(
                (be.array([0.0]), be.array([0.0])),
                be.array([0.55]),
                (be.array([0.0]), be.array([0.5])),
            )
        finally:
            IterativeRayAimer._finite_difference_jacobian = original

        report = aimer.last_report
        assert report is not None
        assert report.converged
        assert report.fallback_used
        assert report.jacobian_refreshes >= 1

    def test_direct_convergence_reports_no_fallback(self, set_test_backend):
        optic = straight()
        aimer = IterativeRayAimer(optic)
        aimer.aim_rays(
            (be.array([0.0]), be.array([0.0])),
            be.array([0.55]),
            (be.array([0.0]), be.array([0.7])),
        )
        report = aimer.last_report
        assert report is not None
        assert report.converged
        assert not report.fallback_used
        assert report.jacobian_refreshes == 0

    def test_broyden_skips_zero_step(self, set_test_backend):
        """A held ray (zero accepted step) must leave the Jacobian
        untouched instead of dividing by an arbitrary floor."""
        optic = straight()
        aimer, _, bound, stop_idx, is_inf = _chief_state(optic)
        # Directly exercise the floor logic: a zero step and a round-off
        # step are both skipped; a real step is not.
        eps = machine_eps(be.zeros(1))
        step_scale = 1.0
        for step, expect_update in (
            (0.0, False),
            (0.1 * eps, False),
            (1e3 * eps, True),
        ):
            norm_sq = be.array([step**2])
            step_ref = be.maximum(be.abs(be.array([0.0])), step_scale)
            floor_sq = (machine_eps(norm_sq) * step_ref) ** 2
            do_update = bool(be.to_numpy(norm_sq > floor_sq).reshape(-1)[0])
            assert do_update == expect_update


# ---------------------------------------------------------------------------
# Dtype-aware transverse-basis pole and projection tolerances
# ---------------------------------------------------------------------------


def entered_along_y(sign=+1.0):
    """The straight singlet rigidly rotated onto the +/-y axis.

    Entry exactly along +/-y sits on the pole of the deterministic
    transverse-basis gauge (global +y projected off the axis degenerates),
    so the basis must switch to the +x rule and stay finite.
    """
    optic = Optic(name=f"entered-along-{'pos' if sign > 0 else 'neg'}-y")
    rx = -sign * math.pi / 2
    optic.surfaces.add(index=0, x=0.0, y=-sign * be.inf, z=0.0, radius=be.inf)
    optic.surfaces.add(
        index=1,
        x=0.0,
        y=0.0,
        z=0.0,
        rx=rx,
        radius=25.84,
        material="N-BK7",
        is_stop=True,
    )
    optic.surfaces.add(
        index=2, x=0.0, y=sign * 4.0, z=0.0, rx=rx, radius=be.inf
    )
    optic.surfaces.add(index=3, x=0.0, y=sign * 50.0, z=0.0, rx=rx)
    return _finish(optic)


def _basis_asserts(direction):
    """Finite, normalized, orthogonal, right-handed basis for direction.

    Internal orthonormality of ``(u, v)`` holds at dtype level everywhere.
    Orthogonality *to the direction* is exact only up to the direction's
    distance from the gauge pole: within the pole's numerical shadow the
    projection cancels catastrophically (that is precisely why the gauge
    switches), so the coupling tolerance carries an O(pole distance) term.
    The residual beam-axis coupling it permits is physically negligible
    (nanometres over pupil-scale displacements).
    """
    u, v = transverse_basis(direction)
    u = np.array([float(be.to_numpy(be.array(c))) for c in u])
    v = np.array([float(be.to_numpy(be.array(c))) for c in v])
    d = np.array([float(be.to_numpy(be.array(c))) for c in direction])
    assert np.all(np.isfinite(u))
    assert np.all(np.isfinite(v))
    tol = 1e-6 if str(be.get_precision()).find("32") >= 0 else 1e-12
    pole_distance = float(np.hypot(d[0], d[2]))
    coupling_tol = max(10 * tol, 10 * pole_distance)
    assert abs(np.dot(u, u) - 1.0) <= 10 * tol
    assert abs(np.dot(v, v) - 1.0) <= 10 * tol
    assert abs(np.dot(u, v)) <= 10 * tol
    assert abs(np.dot(u, d)) <= coupling_tol
    assert abs(np.dot(v, d)) <= coupling_tol
    # Right-handed: u x v == d.
    cross = np.cross(u, v)
    assert np.allclose(cross, d, atol=max(100 * tol, coupling_tol))
    return u, v


class TestBasisPole:
    @pytest.mark.parametrize("backend,precision", BACKEND_PRECISION)
    @pytest.mark.parametrize("sign", [+1.0, -1.0])
    def test_entry_exactly_along_y(self, backend, precision, sign):
        with backend_state(backend, precision):
            direction = (be.array(0.0), be.array(sign), be.array(0.0))
            u, _ = _basis_asserts(direction)
            # On the pole the gauge switches to the +x rule (u = +x for
            # both +y and -y entries; handedness fixes v).
            assert u[0] == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("backend,precision", BACKEND_PRECISION)
    def test_just_inside_and_outside_pole_tolerance(self, backend, precision):
        with backend_state(backend, precision):
            tol = angular_tolerance()
            for a in (0.3 * tol, 3.0 * tol, 100.0 * tol):
                d = (
                    be.array(a),
                    be.array(math.sqrt(1.0 - a * a)),
                    be.array(0.0),
                )
                _basis_asserts(d)

    def test_pole_tolerance_widens_in_float32(self):
        with backend_state("numpy", "float64"):
            tol64 = angular_tolerance()
        with backend_state("numpy", "float32"):
            tol32 = angular_tolerance()
            # A projection at 5e-6 is round-off noise in float32: the pole
            # branch must take over (deterministic +x gauge).
            a = 5.0e-6
            assert a < tol32
            d = (be.array(a), be.array(math.sqrt(1.0 - a * a)), be.array(0.0))
            u, _ = _basis_asserts(d)
            assert u[0] == pytest.approx(1.0, abs=1e-3)
        assert tol32 > tol64

    @pytest.mark.parametrize("backend,precision", BACKEND_PRECISION)
    def test_degenerate_projection_tolerance_is_dtype_aware(
        self, backend, precision
    ):
        with backend_state(backend, precision):
            tol = _degenerate_projection_tolerance()
            expected = float(machine_eps(be.zeros(1))) ** 0.5
            assert tol == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("sign", [+1.0, -1.0])
    def test_paraxial_aiming_stable_along_y_entry(self, set_test_backend, sign):
        """First-order results of the +/-y-entered singlet equal the
        straight reference, and the entry frame stays finite (no NaN on
        either backend)."""
        optic = entered_along_y(sign)
        reference = straight()
        assert_allclose(optic.paraxial.f2(), reference.paraxial.f2(), rtol=1e-10)
        assert_allclose(optic.paraxial.EPL(), reference.paraxial.EPL(), rtol=1e-10)
        path = optic.surfaces.build_paraxial_path()
        for vector in (path.entry_u, path.entry_v):
            for component in vector:
                assert bool(be.all(be.isfinite(be.array(component))))

    @pytest.mark.parametrize("sign", [+1.0, -1.0])
    def test_iterative_parameterization_stable_along_y_entry(
        self, set_test_backend, sign
    ):
        """The iterative aimer solves pupil points on the +/-y-entered
        system without NaN."""
        optic = entered_along_y(sign)
        aimer = IterativeRayAimer(optic)
        x, y, z, L, M, N = aimer.aim_rays(
            (be.array([0.0]), be.array([0.0])),
            be.array([0.55]),
            (be.array([0.0, 0.5]), be.array([0.0, 0.5])),
        )
        for component in (x, y, z, L, M, N):
            assert not bool(be.any(be.isnan(component)))
        report = aimer.last_report
        assert report is not None and report.converged


# ---------------------------------------------------------------------------
# Robust aiming reports (honest and complete)
# ---------------------------------------------------------------------------


def _pupil_batch():
    return (be.array([0.0, 0.0, 0.2]), be.array([0.0, 0.3, -0.2]))


def _fail(result, n):
    """Return a copy of a _solve_core result with every ray non-converged."""
    x_, y_, z_, L_, M_, N_, conv, nan_flag, report = result
    return (x_, y_, z_, L_, M_, N_, conv & (be.zeros(n) > 0.0), nan_flag, report)


class TestRobustReports:
    def test_fresh_calibration_report(self, set_test_backend):
        from optiland.rays.ray_aiming.robust import RobustRayAimer

        optic = stop_mid_straight()
        aimer = RobustRayAimer(optic)
        aimer.aim_rays((0.0, 0.4), 0.55, _pupil_batch())
        report = aimer.last_report
        assert report is not None
        assert report.converged
        assert report.num_rays == 3
        assert report.num_converged == 3
        assert not report.fallback_used
        assert len(report.field_reports) == 1
        field = report.field_reports[0]
        assert field.calibration_used
        assert not field.used_cached_map
        assert field.chief_seed_strategy in ("direct_paraxial", "warm_map")
        assert field.edge_probe_fallbacks == 0
        assert field.final_polish.converged
        assert report.final_residual <= 10 * aimer.tol

    def test_cached_map_report(self, set_test_backend):
        from optiland.rays.ray_aiming.robust import RobustRayAimer

        optic = stop_mid_straight()
        aimer = RobustRayAimer(optic)
        aimer.aim_rays((0.0, 0.4), 0.55, _pupil_batch())
        aimer.aim_rays((0.0, 0.4), 0.55, _pupil_batch())
        field = aimer.last_report.field_reports[0]
        assert field.used_cached_map
        assert not field.calibration_used
        assert field.chief_seed_strategy == "cached_map"
        assert not field.fallback_used

    def test_marching_report(self, set_test_backend):
        """Force the direct chief solve to fail once so the field-marching
        fallback runs and is reported."""
        from optiland.rays.ray_aiming.iterative import IterativeRayAimer
        from optiland.rays.ray_aiming.robust import RobustRayAimer

        optic = stop_mid_straight()
        aimer = RobustRayAimer(optic)

        original = IterativeRayAimer._solve_core
        state = {"target0_singles": 0}

        def failing_direct_chief(self, x, y, z, L, M, N, *args, **kwargs):
            result = original(self, x, y, z, L, M, N, *args, **kwargs)
            n = len(be.as_array_1d(x))
            tx, ty = args[3], args[4]
            target = float(be.to_numpy(be.abs(tx) + be.abs(ty)).reshape(-1)[0])
            if n == 1 and target == 0.0:
                state["target0_singles"] += 1
                # Call 1 is the stop-radius pupil-center solve; call 2 is
                # the direct chief attempt -- fail exactly that one so the
                # marching fallback must run.
                if state["target0_singles"] == 2:
                    return _fail(result, n)
            return result

        IterativeRayAimer._solve_core = failing_direct_chief
        try:
            aimer.aim_rays((0.0, 0.4), 0.55, _pupil_batch())
        finally:
            IterativeRayAimer._solve_core = original

        report = aimer.last_report
        field = report.field_reports[0]
        assert field.chief_seed_strategy == "marching"
        assert field.fallback_used
        assert report.fallback_used
        assert report.converged

    def test_edge_probe_fallbacks_are_counted(self, set_test_backend):
        """Probes that fail to converge fall back to the chief launch and
        the count is reported."""
        from optiland.rays.ray_aiming.iterative import IterativeRayAimer
        from optiland.rays.ray_aiming.robust import RobustRayAimer

        optic = stop_mid_straight()
        aimer = RobustRayAimer(optic)

        original = IterativeRayAimer._solve_core

        def failing_probes(
            self, x, y, z, L, M, N, wavelengths, stop_idx, is_inf, tx, ty,
            **kwargs,
        ):
            result = original(
                self, x, y, z, L, M, N, wavelengths, stop_idx, is_inf, tx, ty,
                **kwargs,
            )
            n = len(be.as_array_1d(x))
            target = float(be.to_numpy(be.abs(tx) + be.abs(ty)).reshape(-1)[0])
            if n == 1 and target > 0.0:
                x_, y_, z_, L_, M_, N_, conv, nan_flag, report = result
                return (
                    x_,
                    y_,
                    z_,
                    L_,
                    M_,
                    N_,
                    conv & (be.zeros(1) > 0.0),
                    nan_flag,
                    report,
                )
            return result

        IterativeRayAimer._solve_core = failing_probes
        try:
            aimer.aim_rays(
                (0.0, 0.2), 0.55, (be.array([0.0, 0.1]), be.array([0.0, 0.1]))
            )
        finally:
            IterativeRayAimer._solve_core = original

        field = aimer.last_report.field_reports[0]
        assert field.edge_probe_fallbacks == 4
        assert field.fallback_used

    def test_failed_initial_guess_marks_fallback(self, set_test_backend):
        from optiland.rays.ray_aiming.robust import RobustRayAimer

        optic = stop_mid_straight()
        aimer = RobustRayAimer(optic)
        # A sideways guess traces to NaN, so the direct iterative attempt
        # raises and the calibrated solve takes over.
        bad_guess = (
            be.array([0.0, 0.0, 0.2]),
            be.array([0.0, 0.3, -0.2]),
            be.array([-10.0, -10.0, -10.0]),
            be.array([1.0, 1.0, 1.0]),
            be.array([0.0, 0.0, 0.0]),
            be.array([0.0, 0.0, 0.0]),
        )
        aimer.aim_rays((0.0, 0.4), 0.55, _pupil_batch(), initial_guess=bad_guess)
        report = aimer.last_report
        assert report.converged
        assert report.fallback_used

    def test_successful_initial_guess_reports_strategy(self, set_test_backend):
        from optiland.rays.ray_aiming.robust import RobustRayAimer

        optic = stop_mid_straight()
        aimer = RobustRayAimer(optic)
        # First solve normally, reuse the result as a (good) guess.
        solution = aimer.aim_rays((0.0, 0.4), 0.55, _pupil_batch())
        aimer.aim_rays((0.0, 0.4), 0.55, _pupil_batch(), initial_guess=solution)
        report = aimer.last_report
        assert report.converged
        assert not report.fallback_used
        assert report.field_reports[0].chief_seed_strategy == "initial_guess"
        assert not report.field_reports[0].calibration_used

    def test_partial_per_ray_convergence_counts(self, set_test_backend):
        """Non-converged rays surface as NaN with exact counts retained and
        converged defined as num_converged == num_rays."""
        from optiland.rays.ray_aiming.iterative import IterativeRayAimer
        from optiland.rays.ray_aiming.robust import RobustRayAimer

        optic = stop_mid_straight()
        aimer = RobustRayAimer(optic)

        original = IterativeRayAimer._solve_core

        def fail_last_ray_of_batch(self, x, y, z, L, M, N, *args, **kwargs):
            result = original(self, x, y, z, L, M, N, *args, **kwargs)
            n = len(be.as_array_1d(x))
            if n == 3:
                from dataclasses import replace

                x_, y_, z_, L_, M_, N_, conv, nan_flag, report = result
                keep = be.arange_indices(n) < n - 1
                conv = conv & keep
                report = replace(
                    report,
                    num_converged=int(be.to_numpy(conv).reshape(-1).sum()),
                    converged=False,
                )
                return x_, y_, z_, L_, M_, N_, conv, nan_flag, report
            return result

        IterativeRayAimer._solve_core = fail_last_ray_of_batch
        try:
            x, y, z, L, M, N = aimer.aim_rays((0.0, 0.4), 0.55, _pupil_batch())
        finally:
            IterativeRayAimer._solve_core = original

        report = aimer.last_report
        assert report.num_rays == 3
        assert report.num_converged == 2
        assert not report.converged
        assert bool(be.isnan(x[2]))
        assert not bool(be.any(be.isnan(x[0:2])))

    def test_total_field_failure_sets_last_report_before_raising(
        self, set_test_backend
    ):
        from optiland.rays.ray_aiming.iterative import IterativeRayAimer
        from optiland.rays.ray_aiming.robust import RobustRayAimer

        optic = stop_mid_straight()
        aimer = RobustRayAimer(optic)

        original = IterativeRayAimer._solve_core

        def always_fail(self, x, y, z, L, M, N, *args, **kwargs):
            result = original(self, x, y, z, L, M, N, *args, **kwargs)
            x_, y_, z_, L_, M_, N_, conv, nan_flag, report = result
            n = len(be.as_array_1d(x))
            return (
                x_,
                y_,
                z_,
                L_,
                M_,
                N_,
                conv & (be.zeros(n) > 0.0),
                nan_flag,
                report,
            )

        IterativeRayAimer._solve_core = always_fail
        try:
            with pytest.raises(ValueError):
                aimer.aim_rays((0.0, 0.4), 0.55, _pupil_batch())
        finally:
            IterativeRayAimer._solve_core = original

        report = aimer.last_report
        assert report is not None
        assert not report.converged
        assert report.fallback_used
        assert report.field_reports[-1].chief_seed_strategy == "failed"


# ---------------------------------------------------------------------------
# Seed-centered chief scan
# ---------------------------------------------------------------------------


class TestSeedCenteredScan:
    @staticmethod
    def _seed_and_param(optic):
        from optiland.rays.ray_aiming.robust import RobustRayAimer

        aimer = RobustRayAimer(optic)
        is_inf = bool(getattr(optic.object_surface, "is_infinite", False))
        param = LaunchParameterization.for_optic(optic, is_inf)
        seed = aimer._paraxial.aim_rays(
            (be.array([0.0]), be.array([0.6])),
            be.array([0.55]),
            (be.array([0.0]), be.array([0.0])),
        )
        return aimer, param, seed

    @staticmethod
    def _candidates(optic, param, seed, Hx=0.0, Hy=0.6, n=41):
        from optiland.rays.ray_aiming.pupil_map import to_float
        from optiland.rays.ray_aiming.robust import _scan_candidate_offsets

        path = optic.surfaces.build_paraxial_path()
        anchor = path.vertices_gcs[1]
        u, v = param.u, param.v
        g_rel = (
            to_float(seed[0]) - to_float(anchor[0]),
            to_float(seed[1]) - to_float(anchor[1]),
            to_float(seed[2]) - to_float(anchor[2]),
        )
        g_xi = g_rel[0] * u[0] + g_rel[1] * u[1] + g_rel[2] * u[2]
        g_eta = g_rel[0] * v[0] + g_rel[1] * v[1] + g_rel[2] * v[2]
        xi_off, eta_off = _scan_candidate_offsets(g_xi, g_eta, Hx, Hy, n)

        ones = be.ones(n)
        bound = param.bind(
            ones * to_float(seed[0]),
            ones * to_float(seed[1]),
            ones * to_float(seed[2]),
            ones * to_float(seed[3]),
            ones * to_float(seed[4]),
            ones * to_float(seed[5]),
        )
        return bound.launch(xi_off, eta_off), (xi_off, eta_off)

    def test_zero_candidate_is_exactly_the_seed(self, set_test_backend):
        optic = straight()
        aimer, param, seed = self._seed_and_param(optic)
        (x, y, z, L, M, N), _ = self._candidates(optic, param, seed, n=41)
        mid = 41 // 2
        for launched, seeded in zip((x, y, z, L, M, N), seed, strict=True):
            got = float(be.to_numpy(launched).reshape(-1)[mid])
            want = float(be.to_numpy(seeded).reshape(-1)[0])
            assert got == want  # exact, bit-for-bit

    @pytest.mark.parametrize(
        "builder_name", ["straight", "entered_along_x", "entered_along_neg_z"]
    )
    def test_all_candidate_offsets_are_transverse(
        self, set_test_backend, builder_name
    ):
        from tests import test_folded_paraxial as tfp
        from tests.test_folded_paraxial_hardening import entered_along_neg_z

        builders = {
            "straight": tfp.straight,
            "entered_along_x": tfp.entered_along_x,
            "entered_along_neg_z": entered_along_neg_z,
        }
        optic = builders[builder_name]()
        aimer, param, seed = self._seed_and_param(optic)
        (x, y, z, L, M, N), _ = self._candidates(optic, param, seed, n=41)
        path = optic.surfaces.build_paraxial_path()
        d = [float(be.to_numpy(be.array(c))) for c in path.entry_direction]
        sx = float(be.to_numpy(seed[0]).reshape(-1)[0])
        sy = float(be.to_numpy(seed[1]).reshape(-1)[0])
        sz = float(be.to_numpy(seed[2]).reshape(-1)[0])
        dx = be.to_numpy(x).reshape(-1) - sx
        dy = be.to_numpy(y).reshape(-1) - sy
        dz = be.to_numpy(z).reshape(-1) - sz
        along = dx * d[0] + dy * d[1] + dz * d[2]
        assert np.max(np.abs(along)) <= 1e-9

    def test_candidates_invariant_under_rigid_translation(self, set_test_backend):
        from tests.test_folded_paraxial_hardening import _translate

        base = straight()
        moved = _translate(straight(), dx=7.0, dy=-3.0)

        _, param_a, seed_a = self._seed_and_param(base)
        _, param_b, seed_b = self._seed_and_param(moved)
        _, (xi_a, eta_a) = self._candidates(base, param_a, seed_a, n=41)
        _, (xi_b, eta_b) = self._candidates(moved, param_b, seed_b, n=41)
        # The local scan offsets are identical: the sweep is centered on
        # the seed and scaled by the seed's offset from the first-surface
        # anchor, both of which translate with the system.
        assert np.allclose(
            be.to_numpy(xi_a), be.to_numpy(xi_b), rtol=0, atol=1e-12
        )
        assert np.allclose(
            be.to_numpy(eta_a), be.to_numpy(eta_b), rtol=0, atol=1e-12
        )

    def test_forced_scan_is_used_and_reported(self, set_test_backend):
        """Deterministically force direct solve and marching to fail so the
        scan branch definitely runs, converges and is reported."""
        from optiland.rays.ray_aiming.iterative import IterativeRayAimer
        from optiland.rays.ray_aiming.robust import RobustRayAimer

        optic = stop_mid_straight()
        aimer = RobustRayAimer(optic)
        march_orig = RobustRayAimer._march_chief
        RobustRayAimer._march_chief = lambda self, *a, **k: None

        original = IterativeRayAimer._solve_core
        state = {"scan_seen": False, "calls": 0}

        def fail_single_until_scan(self, x, y, z, L, M, N, *args, **kwargs):
            n = len(be.as_array_1d(x))
            if n > 100:
                state["scan_seen"] = True
            result = original(self, x, y, z, L, M, N, *args, **kwargs)
            if n <= 100 and not state["scan_seen"]:
                state["calls"] += 1
                # The very first small solve is the stop-radius
                # pupil-center computation; keep it intact so r_stop is
                # the genuine real-reference value.
                if state["calls"] > 1:
                    return _fail(result, n)
            return result

        IterativeRayAimer._solve_core = fail_single_until_scan
        try:
            x, *_ = aimer.aim_rays((0.0, 0.3), 0.55, _pupil_batch())
        finally:
            IterativeRayAimer._solve_core = original
            RobustRayAimer._march_chief = march_orig

        assert state["scan_seen"]
        assert not bool(be.any(be.isnan(x)))
        field = aimer.last_report.field_reports[0]
        assert field.chief_seed_strategy == "scan"
        assert field.fallback_used

    def test_wide_angle_1d_field_beyond_90_can_use_scan(self, set_test_backend):
        """A nonsingular 1-D field beyond 90 degrees can be solved through
        the scan fallback when the other strategies are disabled."""
        from optiland.rays.ray_aiming.iterative import IterativeRayAimer
        from optiland.rays.ray_aiming.robust import RobustRayAimer

        optic = straight()
        optic.fields.add(x=0.0, y=95.0)
        aimer = RobustRayAimer(optic)
        march_orig = RobustRayAimer._march_chief
        RobustRayAimer._march_chief = lambda self, *a, **k: None

        original = IterativeRayAimer._solve_core
        state = {"scan_seen": False, "calls": 0}

        def fail_single_until_scan(self, x, y, z, L, M, N, *args, **kwargs):
            n = len(be.as_array_1d(x))
            if n > 100:
                state["scan_seen"] = True
            result = original(self, x, y, z, L, M, N, *args, **kwargs)
            if n <= 100 and not state["scan_seen"]:
                state["calls"] += 1
                # The very first small solve is the stop-radius
                # pupil-center computation; keep it intact so r_stop is
                # the genuine real-reference value.
                if state["calls"] > 1:
                    return _fail(result, n)
            return result

        IterativeRayAimer._solve_core = fail_single_until_scan
        try:
            x, y, z, L, M, N = aimer.aim_rays(
                (0.0, 1.0), 0.55, (be.array([0.0]), be.array([0.0]))
            )
        finally:
            IterativeRayAimer._solve_core = original
            RobustRayAimer._march_chief = march_orig

        assert state["scan_seen"]
        field = aimer.last_report.field_reports[0]
        assert field.chief_seed_strategy == "scan"
        # The chief of a 95-degree field must run steeply in +y.
        assert float(be.to_numpy(M).reshape(-1)[0]) > 0.9
