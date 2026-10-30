"""Evaluator layer tests (opt v2 PR4.1).

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from benchmarks.optimization.problems import PROBLEMS
from optiland import _epoch
from optiland.optimization import OptimizationProblem
from optiland.optimization.evaluators import (
    Evaluator,
    ExactAim,
    FiniteDiffEvaluator,
    FrozenAim,
    SupportsJacobian,
    SupportsResiduals,
    resolve_aim_policy,
)
from optiland.optimization.failure import (
    ConfigurationError,
    OptimizationFailure,
    check_initial_point,
    normalize_failure_mode,
)
from optiland.samples.objectives import CookeTriplet

pytestmark = pytest.mark.filterwarnings("ignore")


def _eligible(name: str):
    """Benchmark problem with an object-side aperture (partial-trace eligible)."""
    optic, problem = PROBLEMS[name]()
    if optic.aperture.ap_type != "EPD":
        optic.set_aperture("EPD", float(be.to_numpy(optic.paraxial.EPD())))
    return optic, problem


def _simple_problem(min_val=None, max_val=None):
    optic = CookeTriplet()
    problem = OptimizationProblem()
    problem.add_operand("f2", target=50.0, weight=1.0, input_data={"optic": optic})
    for wl in range(3):
        problem.add_operand(
            "rms_spot_size",
            target=0.0,
            weight=1.0,
            input_data={
                "optic": optic,
                "surface_number": -1,
                "Hx": 0.0,
                "Hy": 0.7,
                "num_rays": 4,
                "wavelength": optic.wavelengths.get_wavelengths()[wl],
                "distribution": "hexapolar",
            },
        )
    problem.add_variable(
        optic, "radius", surface_number=1, min_val=min_val, max_val=max_val
    )
    problem.add_variable(optic, "radius", surface_number=5)
    problem.add_variable(optic, "thickness", surface_number=4)
    return optic, problem


# ---------------------------------------------------------------------------
# Protocols and residual convention
# ---------------------------------------------------------------------------


def test_protocols(set_test_backend):
    _, problem = _simple_problem()
    ev = FiniteDiffEvaluator(problem)
    assert isinstance(ev, Evaluator)
    assert isinstance(ev, SupportsResiduals)
    assert isinstance(ev, SupportsJacobian)
    assert ev.n_vars == 3
    assert ev.backend == be.get_backend()


def test_residuals_are_effective_weight_times_delta(set_test_backend):
    optic, problem = _simple_problem()
    optic.wavelengths.wavelengths[0].weight = 0.25  # V7: field/wl weights count
    ev = FiniteDiffEvaluator(problem)
    x = ev.read_x()
    r = ev.residuals(x)
    expected = np.array(
        [
            float(be.to_numpy(op.effective_weight() * op.delta()))
            for op in problem.operands
        ]
    )
    np.testing.assert_allclose(r, expected, rtol=1e-12)
    assert ev.value(x) == pytest.approx(
        float(be.to_numpy(problem.sum_squared())), rel=1e-12
    )


def test_write_read_roundtrip(set_test_backend):
    _, problem = _simple_problem()
    ev = FiniteDiffEvaluator(problem)
    x = ev.read_x()
    ev.write_x(x * 1.01)
    np.testing.assert_allclose(ev.read_x(), x * 1.01, rtol=1e-12)


# ---------------------------------------------------------------------------
# Jacobian: r0 reuse, schemes, bounds
# ---------------------------------------------------------------------------


def test_r0_reuse_counts(set_test_backend):
    _, problem = _simple_problem()
    ev = FiniteDiffEvaluator(problem, partial_trace=False)
    x = ev.read_x()
    r = ev.residuals(x)
    assert ev.nfev == 1
    ev.jacobian(x, r0=r)
    assert ev.nfev == 1 + ev.n_vars  # baseline not re-evaluated
    ev.jacobian(x)  # baseline remembered: still n columns only
    assert ev.nfev == 1 + 2 * ev.n_vars
    assert ev.njev == 2


def test_forward_matches_central(set_test_backend):
    _, problem = _simple_problem()
    fwd = FiniteDiffEvaluator(problem)
    cen = FiniteDiffEvaluator(problem, scheme="central")
    x = fwd.read_x()
    Jf = fwd.jacobian(x)
    Jc = cen.jacobian(x)
    assert cen.nfev == 1 + 2 * cen.n_vars
    np.testing.assert_allclose(Jf, Jc, rtol=1e-4, atol=1e-6 * np.abs(Jc).max())


def test_jacobian_restores_state(set_test_backend):
    _, problem = _simple_problem()
    ev = FiniteDiffEvaluator(problem)
    x = ev.read_x()
    ev.jacobian(x)
    np.testing.assert_array_equal(ev.read_x(), x)


def test_step_flips_toward_interior_at_bound(set_test_backend):
    _, problem = _simple_problem()
    x0 = FiniteDiffEvaluator(problem).read_x()
    # Upper bound exactly at x0 for the first variable (scaled space)
    var = problem.variables[0]
    var.max_val = var.variable.inverse_scale(x0[0])
    ev = FiniteDiffEvaluator(problem)
    lo, hi = ev.bounds()
    assert hi[0] == pytest.approx(x0[0])
    seen = []
    original = ev._shifted

    def spy(x, j, step, baseline, start):
        seen.append((j, step))
        return original(x, j, step, baseline, start)

    ev._shifted = spy
    J_bounded = ev.jacobian(x0)
    assert all(step < 0 for j, step in seen if j == 0)
    J_free = FiniteDiffEvaluator(_simple_problem()[1]).jacobian(x0)
    np.testing.assert_allclose(
        J_bounded, J_free, rtol=1e-3, atol=1e-6 * np.abs(J_free).max()
    )


def test_interior_step_rules():
    step = FiniteDiffEvaluator._interior_step
    assert step(0.0, 0.1, -1, 1, True, True) == 0.1
    assert step(1.0, 0.1, -1, 1, False, True) == -0.1
    assert step(0.0, 0.1, -0.02, 0.05, False, False) == 0.05
    assert step(0.0, 0.1, -0.05, 0.02, False, False) == -0.05
    assert step(0.0, 0.1, 0.0, 0.0, False, False) == 0.0


def test_bad_scheme_and_policy():
    _, problem = _simple_problem()
    with pytest.raises(ConfigurationError):
        FiniteDiffEvaluator(problem, scheme="backward")
    with pytest.raises(ConfigurationError):
        FiniteDiffEvaluator(problem, aim_policy="sloppy")
    with pytest.raises(ConfigurationError):
        FiniteDiffEvaluator(problem, on_failure="penalty")


# ---------------------------------------------------------------------------
# Partial trace
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["cooke", "double_gauss", "scaling_10"])
def test_partial_trace_jacobian_matches_full(set_test_backend, name):
    _, problem = _eligible(name)
    full = FiniteDiffEvaluator(problem, partial_trace=False)
    part = FiniteDiffEvaluator(problem)
    x = full.read_x()
    J_full = full.jacobian(x)
    J_part = part.jacobian(x)
    # Resumed traces are bitwise equal to full ones (PR3.1), so the Jacobians
    # agree far below finite-difference noise.
    np.testing.assert_allclose(J_part, J_full, rtol=1e-10, atol=1e-12)
    if be.get_backend() == "numpy":
        starts = part.partial_trace_starts()
        assert any(s is not None for s in starts)
        assert part.resumed_traces > 0
    else:
        assert part.resumed_traces == 0  # numpy-only optimization


def test_partial_trace_after_moving(set_test_backend):
    # Snapshots belong to the baseline x: a Jacobian at a new point re-records.
    _, problem = _eligible("cooke")
    full = FiniteDiffEvaluator(problem, partial_trace=False)
    part = FiniteDiffEvaluator(problem)
    x = full.read_x() * 1.002
    r = part.residuals(x)
    np.testing.assert_allclose(
        part.jacobian(x, r0=r), full.jacobian(x), rtol=1e-10, atol=1e-12
    )


def test_partial_trace_ineligible_falls_back():
    _, problem = PROBLEMS["cooke"]()  # image-space F/#: ineligible
    ev = FiniteDiffEvaluator(problem)
    if all(s is None for s in ev.partial_trace_starts()):
        assert not ev._partial
        ev.jacobian(ev.read_x())
        assert ev.resumed_traces == 0


def test_tracer_unchanged_outside_hook():
    from optiland.raytrace import _resume

    assert _resume.active() is None
    optic, _ = _eligible("cooke")
    a = optic.trace_generic(0.0, 0.5, 0.1, 0.2, optic.primary_wavelength)
    b = optic.trace_generic(0.0, 0.5, 0.1, 0.2, optic.primary_wavelength)
    np.testing.assert_array_equal(be.to_numpy(a.y), be.to_numpy(b.y))


# ---------------------------------------------------------------------------
# Aim policy
# ---------------------------------------------------------------------------


def test_resolve_aim_policy():
    assert isinstance(resolve_aim_policy(None), ExactAim)
    assert isinstance(resolve_aim_policy("exact"), ExactAim)
    assert isinstance(resolve_aim_policy("FROZEN"), FrozenAim)
    policy = FrozenAim()
    assert resolve_aim_policy(policy) is policy


def test_frozen_aim_seeds_column_epochs(set_test_backend):
    optic, problem = _simple_problem()
    optic.ray_tracer.set_aiming("iterative")
    ev = FiniteDiffEvaluator(problem, aim_policy="frozen", partial_trace=False)
    x = ev.read_x()
    ev.residuals(x)
    baseline = ev._baselines[x.tobytes()].epoch
    assert len(baseline) > 0
    epoch = _epoch.EvaluationEpoch()
    ev.aim_policy.prepare_column(epoch, baseline)
    assert 0 < len(epoch) <= len(baseline)


def test_frozen_aim_jacobian_close_to_exact(set_test_backend):
    optic, problem = _simple_problem()
    optic.ray_tracer.set_aiming("iterative")
    exact = FiniteDiffEvaluator(problem, partial_trace=False)
    frozen = FiniteDiffEvaluator(problem, aim_policy="frozen", partial_trace=False)
    x = exact.read_x()
    # Merit evaluations stay exact under the frozen policy.
    np.testing.assert_allclose(frozen.residuals(x), exact.residuals(x), rtol=1e-12)
    J_exact = exact.jacobian(x)
    J_frozen = frozen.jacobian(x)
    assert np.all(np.isfinite(J_frozen))
    # Variables after the stop barely move the aim: columns agree closely.
    np.testing.assert_allclose(
        J_frozen[:, 1:], J_exact[:, 1:], rtol=1e-2, atol=1e-3 * np.abs(J_exact).max()
    )


# ---------------------------------------------------------------------------
# Failure policy (Section 8.4)
# ---------------------------------------------------------------------------


def test_normalize_failure_mode():
    assert normalize_failure_mode(None) == "reject"
    assert normalize_failure_mode("RAISE") == "raise"


def test_exception_becomes_nan_rows(set_test_backend, monkeypatch):
    _, problem = _simple_problem()
    ev = FiniteDiffEvaluator(problem)
    x = ev.read_x()
    r = ev.residuals(x)

    def boom():
        raise RuntimeError("trace failed")

    monkeypatch.setattr(problem, "residual_vector", boom)
    bad = ev.residuals(x)
    assert bad.shape == r.shape
    assert np.all(np.isnan(bad))
    assert np.isnan(ev.value(x))


def test_raise_mode(set_test_backend, monkeypatch):
    _, problem = _simple_problem()
    ev = FiniteDiffEvaluator(problem, on_failure="raise")
    x = ev.read_x()
    ev.residuals(x)
    monkeypatch.setattr(
        problem, "residual_vector", lambda: be.array([1.0, float("nan"), 0.0, 0.0])
    )
    with pytest.raises(OptimizationFailure, match="rms_spot_size"):
        ev.residuals(x)


def test_failed_column_rows_are_zero(set_test_backend, monkeypatch):
    _, problem = _simple_problem()
    ev = FiniteDiffEvaluator(problem, partial_trace=False)
    x = ev.read_x()
    r0 = ev.residuals(x)
    real = problem.residual_vector

    def flaky():
        out = be.to_numpy(real()).copy()
        if ev.read_x()[0] != x[0]:  # every evaluation that moves variable 0
            out[1] = np.nan
        return be.array(out)

    monkeypatch.setattr(problem, "residual_vector", flaky)
    J = ev.jacobian(x, r0=r0)
    assert np.all(np.isfinite(J))
    assert J[1, 0] == 0.0
    assert J[0, 0] != 0.0


def test_check_initial_point_names_operands():
    _, problem = _simple_problem()
    with pytest.raises(ConfigurationError, match="#2 rms_spot_size"):
        check_initial_point(problem, np.array([0.0, 1.0, np.nan, 0.0]))
    check_initial_point(problem, np.zeros(4))


def test_material_variables_rejected():
    optic = CookeTriplet()
    problem = OptimizationProblem()
    problem.add_operand("f2", target=50.0, input_data={"optic": optic})
    problem.add_variable(
        optic, "material", surface_number=1, glass_selection=["N-BK7", "N-SF11"]
    )
    with pytest.raises(ConfigurationError):
        FiniteDiffEvaluator(problem)


# ---------------------------------------------------------------------------
# Autograd evaluator (torch)
# ---------------------------------------------------------------------------


def test_autograd_matches_finite_differences():
    pytest.importorskip("torch")
    from optiland.optimization.evaluators import AutogradEvaluator

    be.set_backend("torch")
    be.set_device("cpu")
    be.set_precision("float64")
    be.grad_mode.enable()
    try:
        _, problem = _simple_problem()
        fd = FiniteDiffEvaluator(problem, scheme="central")
        x = fd.read_x()
        J_fd = fd.jacobian(x)
        r_fd = fd.residuals(x)
        ag = AutogradEvaluator(problem)
        r_ag = be.to_numpy(ag.residuals(x))
        J_ag = be.to_numpy(ag.jacobian(x))
        np.testing.assert_allclose(r_ag, r_fd, rtol=1e-10)
        np.testing.assert_allclose(
            J_ag, J_fd, rtol=1e-4, atol=1e-6 * np.abs(J_fd).max()
        )
        g = be.to_numpy(ag.gradient(x))
        np.testing.assert_allclose(g, 2 * J_ag.T @ r_ag, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(be.to_numpy(ag.read_x()), x, rtol=1e-12)
        assert ag.nfev >= 2 and ag.njev == 1
    finally:
        be.set_backend("numpy")


def test_autograd_requires_torch_backend():
    pytest.importorskip("torch")
    from optiland.optimization.evaluators import AutogradEvaluator

    be.set_backend("numpy")
    _, problem = _simple_problem()
    with pytest.raises(ConfigurationError):
        AutogradEvaluator(problem)
