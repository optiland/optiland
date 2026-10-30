"""Hard-constraint data model (opt v2 PR7.2).

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland import _epoch, _instrument
from optiland.optimization import (
    Constraint,
    ConstraintManager,
    OptimizationProblem,
)
from optiland.optimization.constraint import EQUALITY, INEQUALITY
from optiland.optimization.evaluators import FiniteDiffEvaluator
from optiland.optimization.failure import ConfigurationError
from optiland.samples.objectives import CookeTriplet

from ..utils import assert_allclose

pytestmark = pytest.mark.filterwarnings("ignore")


def _ray(lens, surface=-1, py=1.0):
    return {
        "optic": lens,
        "surface_number": surface,
        "Hx": 0.0,
        "Hy": 0.7,
        "Px": 0.0,
        "Py": py,
        "wavelength": 0.55,
    }


def _problem(batching=True):
    lens = CookeTriplet()
    problem = OptimizationProblem(batching=batching)
    problem.add_operand("real_y_intercept", target=0.0, input_data=_ray(lens))
    problem.add_operand("f2", target=50.0, weight=2.0, input_data={"optic": lens})
    problem.add_variable(lens, "radius", surface_number=1)
    problem.add_variable(lens, "radius", surface_number=3)
    return lens, problem


# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------


def test_rows_follow_the_spec():
    manager = ConstraintManager()
    manager.add("f2", target=50.0)
    manager.add("f2", min_val=10.0)
    manager.add("f2", max_val=90.0)
    manager.add("f2", min_val=10.0, max_val=90.0, weight=3.0, tol=1e-6)

    assert len(manager) == 4
    assert manager.n_rows == 5
    assert manager.n_equality == 1
    assert manager.n_inequality == 4
    assert manager.signs == [1.0, -1.0, 1.0, -1.0, 1.0]
    assert manager.equality_mask.tolist() == [True, False, False, False, False]
    np.testing.assert_allclose(manager.tols, [1e-8, 1e-8, 1e-8, 1e-6, 1e-6])
    assert [r.kind for r in manager[3].rows] == [INEQUALITY, INEQUALITY]
    assert manager[0].rows[0].kind == EQUALITY
    assert [r.operand.target for r in manager[3].rows] == [10.0, 90.0]
    assert manager.labels[1] == "#1 f2 >= 10"
    assert len(set(manager.labels)) == 5


@pytest.mark.parametrize(
    "spec",
    [
        {"operand_type": "not_an_operand", "target": 1.0},
        {"operand_type": "f2"},
        {"operand_type": "f2", "target": 1.0, "min_val": 0.0},
        {"operand_type": "f2", "min_val": 2.0, "max_val": 1.0},
        {"operand_type": "f2", "target": float("nan")},
        {"operand_type": "f2", "max_val": float("inf")},
        {"operand_type": "f2", "target": 1.0, "weight": 0.0},
        {"operand_type": "f2", "target": 1.0, "weight": -1.0},
        {"operand_type": "f2", "target": 1.0, "tol": 0.0},
    ],
)
def test_invalid_specs_raise(spec):
    with pytest.raises(ConfigurationError):
        Constraint(**spec)


def test_add_and_clear_on_the_problem():
    lens, problem = _problem()
    problem.add_constraint("f2", target=50.0, input_data={"optic": lens})
    assert len(problem.constraints) == 1
    assert problem.batching_enabled
    problem.clear_constraints()
    assert len(problem.constraints) == 0
    assert be.to_numpy(problem.constraint_vector()).size == 0


# ---------------------------------------------------------------------------
# Residual convention
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batching", [True, False])
def test_rows_are_signed_effective_weight_delta(set_test_backend, batching):
    lens, problem = _problem(batching)
    f2 = float(be.to_numpy(lens.paraxial.f2()))
    data = {"optic": lens}
    problem.add_constraint("f2", target=f2 + 1.0, weight=2.0, input_data=data)
    problem.add_constraint(
        "f2", min_val=f2 + 2.0, max_val=f2 - 3.0 + 10.0, input_data={"optic": lens}
    )
    assert_allclose(problem.constraint_vector(), [-2.0, 2.0, -7.0], atol=1e-9)


def test_field_and_wavelength_weights_scale_the_row(set_test_backend):
    lens, problem = _problem()
    lens.fields.fields[1].weight = 0.5
    data = {"optic": lens, "field": 1}
    constraint = Constraint("f2", target=1.0, weight=4.0, input_data=data)
    assert constraint.rows[0].operand.effective_weight() == pytest.approx(2.0)

    constraint = problem.add_constraint(
        "real_y_intercept", target=1.0, weight=4.0, input_data=_ray(lens)
    )
    operand = constraint.rows[0].operand
    operand.effective_weight = lambda optic=None: 2.0  # as a field weight of 1/2
    value = float(be.to_numpy(constraint.value))
    assert_allclose(problem.constraint_vector(), [2.0 * (value - 1.0)], atol=1e-9)


def test_zero_effective_weight_row_is_exactly_zero(set_test_backend):
    lens, problem = _problem()
    lens.fields.fields[1].weight = 0.0
    problem.add_constraint("f2", target=1.0, input_data={"optic": lens, "field": 1})
    assert_allclose(problem.constraint_vector(), [0.0])
    assert problem.constraints.max_violation(problem.constraint_vector()) == 0.0


def test_merit_ignores_constraints(set_test_backend):
    for batching in (True, False):
        lens, problem = _problem(batching)
        before = be.to_numpy(problem.residual_vector()).copy()
        merit = float(be.to_numpy(problem.sum_squared()))
        terms = be.to_numpy(problem.fun_array()).copy()
        problem.add_constraint("f2", target=1.0, input_data={"optic": lens})
        problem.add_constraint("real_y_intercept", max_val=0.0, input_data=_ray(lens))
        assert_allclose(problem.residual_vector(), before, atol=1e-12)
        assert_allclose(problem.fun_array(), terms, atol=1e-12)
        assert float(be.to_numpy(problem.sum_squared())) == pytest.approx(merit)


def test_batched_and_direct_rows_agree(set_test_backend):
    rows = []
    for batching in (True, False):
        lens, problem = _problem(batching)
        problem.add_constraint("f2", min_val=1.0, input_data={"optic": lens})
        problem.add_constraint(
            "real_y_intercept", min_val=-1.0, max_val=1.0, input_data=_ray(lens, 3)
        )
        problem.add_constraint(
            "rms_spot_size",
            max_val=0.01,
            input_data={
                "optic": lens,
                "surface_number": -1,
                "Hx": 0,
                "Hy": 0,
                "num_rays": 3,
                "wavelength": 0.55,
                "distribution": "hexapolar",
            },
        )
        rows.append(be.to_numpy(problem.residuals_and_constraints()).copy())
    assert_allclose(rows[0], rows[1], atol=1e-10)
    assert rows[0].size == 2 + 4


# ---------------------------------------------------------------------------
# Same epoch, same bundles
# ---------------------------------------------------------------------------


def _counts(problem, read):
    read()  # warm caches
    problem.update_optics()
    with _instrument.counting() as counts:
        read()
    return dict(counts)


def test_constraints_share_the_merit_bundle(set_test_backend):
    lens, problem = _problem()
    plain = _counts(problem, problem.residual_vector)
    problem.add_constraint("real_y_intercept", max_val=9.0, input_data=_ray(lens, 3))
    problem.add_constraint("real_L", min_val=-1.0, input_data=_ray(lens, 3))
    both = _counts(problem, problem.residuals_and_constraints)
    assert both["trace_generic"] == plain["trace_generic"] == 1
    assert both.get("trace", 0) == 0


def test_constraints_read_in_one_epoch(set_test_backend):
    lens, problem = _problem()
    problem.add_constraint("f2", min_val=1.0, input_data={"optic": lens})
    problem.add_constraint("f2", max_val=1e4, input_data={"optic": lens})
    alone = _counts(problem, problem.residual_vector)
    both = _counts(problem, problem.residuals_and_constraints)
    assert both["paraxial_trace"] == alone["paraxial_trace"]


def test_epoch_debug_guard_covers_constraint_optics(monkeypatch):
    lens, problem = _problem()
    other = CookeTriplet()
    problem.add_constraint("f2", min_val=1.0, input_data={"optic": other})
    assert other in problem._batched_evaluator._optics
    with _epoch.evaluation_epoch():
        problem.residuals_and_constraints()


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------


def test_fd_rows_and_jacobian_match_differences():
    lens, problem = _problem()
    problem.add_constraint("f2", target=48.0, input_data={"optic": lens})
    problem.add_constraint("real_y_intercept", max_val=2.0, input_data=_ray(lens, 3))
    ev = FiniteDiffEvaluator(problem)
    x = ev.read_x()
    r = ev.residuals(x)
    assert r.size == 2
    s = ev.constraints(x)
    assert s.size == 2
    J = ev.jacobian(x, r0=r)
    assert J.shape == (2, 2)
    A = ev.constraint_jacobian(x)
    assert ev.njev == 1

    h = 1e-6
    for j in range(2):
        xp = x.copy()
        xp[j] += h
        ev.residuals(xp)
        np.testing.assert_allclose(
            A[:, j], (ev.constraints(xp) - s) / h, rtol=1e-3, atol=1e-6
        )


def test_partial_trace_keeps_constraint_columns():
    def jacobian(partial):
        lens, problem = _problem()
        lens.set_aperture("EPD", float(be.to_numpy(lens.paraxial.EPD())))
        lens.ray_tracer.set_aiming("paraxial")
        problem.add_variable(lens, "radius", surface_number=5)
        problem.add_constraint(
            "real_y_intercept", max_val=2.0, input_data=_ray(lens, -1)
        )
        problem.add_constraint("f2", min_val=1.0, input_data={"optic": lens})
        ev = FiniteDiffEvaluator(problem, partial_trace=partial)
        x = ev.read_x()
        ev.jacobian(x, r0=ev.residuals(x))
        return ev.constraint_jacobian(x), ev

    full, _ = jacobian(False)
    resumed, ev = jacobian(True)
    assert ev.resumed_traces > 0
    np.testing.assert_allclose(resumed, full, rtol=1e-6, atol=1e-7)


def test_fd_failed_evaluation_keeps_row_count():
    lens, problem = _problem()
    problem.add_constraint("f2", target=48.0, input_data={"optic": lens})
    ev = FiniteDiffEvaluator(problem)
    x = ev.read_x()
    ev.residuals(x)
    problem.residuals_and_constraints = lambda: (_ for _ in ()).throw(RuntimeError())
    bad = x + 1.0
    assert np.isnan(ev.residuals(bad)).all() and ev.residuals(bad).size == 2
    assert np.isnan(ev.constraints(bad)).all() and ev.constraints(bad).size == 1


def test_autograd_rows_match_finite_differences():
    pytest.importorskip("torch")
    from optiland.optimization.evaluators import AutogradEvaluator

    be.set_backend("torch")
    be.set_device("cpu")
    be.grad_mode.enable()
    be.set_precision("float64")
    try:
        lens, problem = _problem()
        problem.add_constraint("f2", target=48.0, input_data={"optic": lens})
        problem.add_constraint(
            "real_y_intercept", max_val=2.0, input_data=_ray(lens, 3)
        )
        ad = AutogradEvaluator(problem)
        x = ad.read_x()
        assert ad.residuals(x).shape == (2,)
        s = be.to_numpy(ad.constraints(x)).copy()
        assert ad.nfev == 1
        J = ad.jacobian(x)
        assert J.shape == (2, 2)
        A = be.to_numpy(ad.constraint_jacobian(x)).copy()
        assert ad.njev == 1
        assert ad.gradient(x).shape == (2,)

        fd = FiniteDiffEvaluator(problem)
        x_np = fd.read_x()
        np.testing.assert_allclose(fd.constraints(x_np), s, rtol=1e-9)
        fd.jacobian(x_np)
        np.testing.assert_allclose(
            fd.constraint_jacobian(x_np), A, rtol=1e-4, atol=1e-5
        )
    finally:
        be.set_precision("float32")
        be.grad_mode.disable()
        be.set_backend("numpy")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_report_and_max_violation(set_test_backend):
    lens, problem = _problem()
    f2 = float(be.to_numpy(lens.paraxial.f2()))
    problem.add_constraint("f2", target=f2, input_data={"optic": lens})
    problem.add_constraint("f2", max_val=f2 - 2.0, input_data={"optic": lens})
    report = problem.constraints.report(problem.constraint_vector())
    assert [row["feasible"] for row in report] == [True, False]
    assert report[1]["violation"] == pytest.approx(2.0)
    assert report[1]["kind"] == INEQUALITY
    assert problem.constraints.max_violation(problem.constraint_vector()) == (
        pytest.approx(2.0)
    )
    assert ConstraintManager().max_violation() == 0.0


def test_constraint_info_prints(capsys, set_test_backend):
    lens, problem = _problem()
    problem.info()
    assert "Constraint" not in capsys.readouterr().out
    problem.add_constraint("f2", min_val=1.0, input_data={"optic": lens})
    problem.constraint_info()
    assert "f2 >= 1" in capsys.readouterr().out
    problem.info()
    assert "f2 >= 1" in capsys.readouterr().out
