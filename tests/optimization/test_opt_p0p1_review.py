"""Regression tests for the P0/P1 optimization review fixes.

Covers:
  - P0: the Levenberg gain-ratio ρ is in the correct units (ρ ≈ 1 on a linear
    least-squares problem, where the quadratic model is exact).  Previously a
    factor-of-2 unit mismatch made ρ ≈ 2.
  - P0: the KKT controller's adaptive λ schedule still converges and matches the
    plain LM solution on an unconstrained problem.
  - P1.2: the ``linear_solver="qr"`` augmented-least-squares path agrees with
    the normal-equation path on a well-conditioned problem and converges on an
    ill-conditioned one.
  - P1.2: the KKT range-space QR solve agrees with the dense normal solve.
  - P1.3: the torch forward-mode (``jacfwd``) Jacobian matches the verified
    stateful reverse-mode Jacobian, and ``"auto"`` resolves sensibly.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import pytest

from optiland.optimization.control.base import StepInfo
from optiland.optimization.control.levenberg import LevenbergController
from optiland.optimization.native.least_squares import LevenbergMarquardt
from optiland.optimization.state import OptimizationState
from optiland.optimization.stopping.criteria import CostTolerance, MaxIter

# ---------------------------------------------------------------------------
# Minimal pure-NLLS evaluator (no Optiland problem needed)
# ---------------------------------------------------------------------------


class _MockEvaluator:
    def __init__(self, fun, x0):
        self._fun = fun
        self._x = np.asarray(x0, dtype=float)
        self.n_vars = len(self._x)
        self.backend = "numpy"

    def read_x(self):
        return self._x.copy()

    def write_x(self, x):
        self._x = np.asarray(x, dtype=float)

    def residuals(self, x):
        self.write_x(x)
        return np.asarray(self._fun(self._x), dtype=float)

    def value(self, x):
        r = self.residuals(x)
        return float(np.dot(r, r))

    def jacobian(self, x, r0=None):
        x = np.asarray(x, dtype=float)
        r0 = self.residuals(x) if r0 is None else np.asarray(r0, dtype=float)
        m = r0.shape[0]
        J = np.empty((m, self.n_vars))
        for i in range(self.n_vars):
            h = max(1e-6 * abs(x[i]), 1e-8)
            xp = x.copy()
            xp[i] += h
            J[:, i] = (self.residuals(xp) - r0) / h
        self.write_x(x)
        return J


def _linear(A, b):
    def _fun(x):
        return A @ x - b

    return _fun


# ---------------------------------------------------------------------------
# P0: gain-ratio units
# ---------------------------------------------------------------------------


class TestGainRatioUnits:
    def test_rho_is_one_on_linear_lsq(self):
        """On a linear residual the model is exact ⇒ ρ == 1 (not 2)."""
        rng = np.random.default_rng(0)
        A = rng.standard_normal((10, 3))
        b = rng.standard_normal(10)
        x0 = np.zeros(3)

        def val(x):
            r = A @ x - b
            return float(r @ r)

        r0 = A @ x0 - b
        state = OptimizationState(x=x0, value=val(x0), residuals=r0, iteration=0)
        # Tiny λ so the damping term is negligible, but the identity
        # actual == predicted holds for *any* λ on a linear problem.
        ctrl = LevenbergController(lam_init=1e-8)
        ctrl.reset(state)
        info = StepInfo(direction=None, residuals=r0, jacobian=A, value_fn=val)

        outcome = ctrl.transform(None, info, state)

        assert outcome.accepted
        assert outcome.info["rho"] == pytest.approx(1.0, abs=1e-6)

    def test_rho_one_holds_with_damping(self):
        """actual == predicted for a linear problem at non-trivial λ ⇒ ρ ≈ 1."""
        rng = np.random.default_rng(1)
        A = rng.standard_normal((6, 2))
        b = rng.standard_normal(6)
        x0 = np.zeros(2)

        def val(x):
            r = A @ x - b
            return float(r @ r)

        r0 = A @ x0 - b
        state = OptimizationState(x=x0, value=val(x0), residuals=r0, iteration=0)
        ctrl = LevenbergController(lam_init=1.0)  # heavy damping on purpose
        ctrl.reset(state)
        info = StepInfo(direction=None, residuals=r0, jacobian=A, value_fn=val)

        outcome = ctrl.transform(None, info, state)

        assert outcome.accepted
        assert outcome.info["rho"] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# P1.2: QR vs normal linear solver (unconstrained Levenberg)
# ---------------------------------------------------------------------------


def _run_lm(fun, x0, *, linear_solver, max_iter=300):
    ev = _MockEvaluator(fun, x0)
    lm = LevenbergMarquardt()
    ctrl = LevenbergController(linear_solver=linear_solver)
    state = lm.initialize(ev, x0, controller=ctrl, constraints=None)
    stop = MaxIter(max_iter) | CostTolerance(1e-12)
    while not lm.converged(state):
        should, _ = stop.should_stop(state)
        if should:
            break
        state = lm.step(state)
    return state


class TestQRLinearSolver:
    def test_qr_matches_normal_well_conditioned(self):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((8, 3))
        b = rng.standard_normal(8)
        x_ref = np.linalg.lstsq(A, b, rcond=None)[0]

        s_norm = _run_lm(_linear(A, b), np.zeros(3), linear_solver="normal")
        s_qr = _run_lm(_linear(A, b), np.zeros(3), linear_solver="qr")

        assert np.allclose(s_norm.x, x_ref, atol=1e-6)
        assert np.allclose(s_qr.x, x_ref, atol=1e-6)
        assert np.allclose(s_norm.x, s_qr.x, atol=1e-6)

    def test_qr_solves_ill_conditioned(self):
        """A Jacobian with cond ~1e6: the QR path reaches the LS solution."""
        rng = np.random.default_rng(7)
        # Construct A = U diag(s) Vᵀ with a wide spread of singular values.
        n = 4
        U, _ = np.linalg.qr(rng.standard_normal((12, n)))
        V, _ = np.linalg.qr(rng.standard_normal((n, n)))
        s = np.array([1.0, 1e-2, 1e-4, 1e-6])
        A = U @ np.diag(s) @ V.T
        b = rng.standard_normal(12)
        x_ref = np.linalg.lstsq(A, b, rcond=None)[0]

        s_qr = _run_lm(_linear(A, b), np.zeros(n), linear_solver="qr", max_iter=500)
        assert np.allclose(s_qr.x, x_ref, atol=1e-4)

    def test_invalid_linear_solver_rejected(self):
        with pytest.raises(ValueError, match="linear_solver"):
            LevenbergController(linear_solver="bogus")


# ---------------------------------------------------------------------------
# P1.2: KKT range-space QR vs dense normal solve
# ---------------------------------------------------------------------------


class TestKKTRangeSpaceSolve:
    def test_make_hsolve_applies_h_inverse(self):
        """hsolve(v) == (JᵀJ + λD)⁻¹ v without forming JᵀJ."""
        from optiland.optimization.control.kkt import KKTController

        rng = np.random.default_rng(3)
        J = rng.standard_normal((9, 4))
        lam = 0.37
        d = np.abs(rng.standard_normal(4)) + 0.1
        H = J.T @ J + lam * np.diag(d)
        v = rng.standard_normal(4)

        hsolve = KKTController._make_hsolve(J, lam, d)
        assert np.allclose(hsolve(v), np.linalg.solve(H, v), atol=1e-9)
        # 2-D RHS (used for H⁻¹Aᵀ)
        M = rng.standard_normal((4, 2))
        assert np.allclose(hsolve(M), np.linalg.solve(H, M), atol=1e-9)

    def test_range_space_matches_dense_kkt(self):
        """Range-space (hsolve) KKT solve == dense assembled solve."""
        from optiland.optimization.control.kkt import KKTController

        rng = np.random.default_rng(5)
        J = rng.standard_normal((10, 4))
        lam = 0.2
        d = np.abs(rng.standard_normal(4)) + 0.1
        H = J.T @ J + lam * np.diag(d)
        A = rng.standard_normal((2, 4))  # 2 active constraint rows
        rhs_top = rng.standard_normal(4)
        rhs_bot = rng.standard_normal(2)

        dx_dense, mu_dense = KKTController._solve_kkt(H, A, rhs_top, rhs_bot)
        hsolve = KKTController._make_hsolve(J, lam, d)
        dx_qr, mu_qr = KKTController._solve_kkt(H, A, rhs_top, rhs_bot, hsolve=hsolve)

        assert np.allclose(dx_dense, dx_qr, atol=1e-7)
        assert np.allclose(mu_dense, mu_qr, atol=1e-7)

    def test_range_space_unconstrained_block(self):
        """k == 0 (no active rows): range-space reduces to H⁻¹·rhs_top."""
        from optiland.optimization.control.kkt import KKTController

        rng = np.random.default_rng(11)
        J = rng.standard_normal((7, 3))
        lam = 0.5
        d = np.abs(rng.standard_normal(3)) + 0.1
        H = J.T @ J + lam * np.diag(d)
        A = np.zeros((0, 3))
        rhs_top = rng.standard_normal(3)

        hsolve = KKTController._make_hsolve(J, lam, d)
        dx, mu = KKTController._solve_kkt(H, A, rhs_top, np.zeros(0), hsolve=hsolve)
        assert np.allclose(dx, np.linalg.solve(H, rhs_top), atol=1e-9)
        assert mu.shape == (0,)


# ---------------------------------------------------------------------------
# P1.3: forward-mode Jacobian (torch only)
# ---------------------------------------------------------------------------


def _make_torch_problem(n_vars):
    """First-order problem (no per-field ray config) with m=2 operands.

    Returns a problem with ``n_vars`` radius variables and two scalar operands
    (``f2``, ``total_track``), so ``m == 2``.
    """
    from optiland.optimization import optimization
    from optiland.samples.objectives import CookeTriplet

    lens = CookeTriplet()
    p = optimization.OptimizationProblem(batching=False)
    p.add_operand(
        operand_type="f2", target=50.0, weight=1.0, input_data={"optic": lens}
    )
    p.add_operand(
        operand_type="total_track", target=100.0, weight=1.0, input_data={"optic": lens}
    )
    surfaces = (1, 3, 2)[:n_vars]
    for s in surfaces:
        p.add_variable(lens, "radius", surface_number=s)
    return p


class TestForwardJacobian:
    def test_forward_mode_matches_stateful(self):
        pytest.importorskip("torch")
        import optiland.backend as be

        be.set_backend("torch")
        try:
            be.grad_mode.enable()
            from optiland.optimization.evaluators.autograd import AutogradEvaluator

            problem = _make_torch_problem(n_vars=2)
            ev_fwd = AutogradEvaluator(problem, jacobian_mode="forward")
            ev_state = AutogradEvaluator(problem, jacobian_mode="stateful")
            x = ev_state.read_x()

            J_fwd = np.asarray(be.to_numpy(ev_fwd.jacobian(x)), dtype=float)
            J_state = np.asarray(be.to_numpy(ev_state.jacobian(x)), dtype=float)

            assert J_fwd.shape == J_state.shape
            assert np.allclose(J_fwd, J_state, atol=1e-5, rtol=1e-4)
        finally:
            be.set_backend("numpy")

    def test_auto_mode_matches_stateful_when_m_gt_n(self):
        pytest.importorskip("torch")
        import optiland.backend as be

        be.set_backend("torch")
        try:
            be.grad_mode.enable()
            from optiland.optimization.evaluators.autograd import AutogradEvaluator

            problem = _make_torch_problem(n_vars=1)  # n=1 < m=2 → auto→forward
            ev_auto = AutogradEvaluator(problem, jacobian_mode="auto")
            ev_state = AutogradEvaluator(problem, jacobian_mode="stateful")
            x = ev_state.read_x()

            J_auto = np.asarray(be.to_numpy(ev_auto.jacobian(x)), dtype=float)
            J_state = np.asarray(be.to_numpy(ev_state.jacobian(x)), dtype=float)
            assert np.allclose(J_auto, J_state, atol=1e-5, rtol=1e-4)
        finally:
            be.set_backend("numpy")
