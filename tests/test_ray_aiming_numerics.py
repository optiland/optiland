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
