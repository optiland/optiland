"""Residual blocks and ray-set re-validation.

Kramer Harrison, 2026
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import optiland.backend as be
from optiland import _instrument
from optiland.optimization import (
    BundleRequest,
    LeastSquares,
    OptimizationProblem,
    ResidualBlock,
    minimize,
)
from optiland.optimization.evaluators import FiniteDiffEvaluator
from optiland.optimization.failure import ConfigurationError, describe_rows
from optiland.optimization.native import dls
from optiland.samples.objectives import CookeTriplet

from ..utils import assert_allclose

pytestmark = pytest.mark.filterwarnings("ignore")


class HeightBlock(ResidualBlock):
    """Ray heights on the image surface, as ``weight * y``."""

    def __init__(self, optic, weight=2.0, wavelength=0.55):
        self.optic = optic
        self.weight = weight
        self.wavelength = wavelength

    def _request(self):
        Py = be.array([0.25, 0.5, 0.75])
        return BundleRequest(
            self.optic, self.wavelength, 0.0, 0.7, be.zeros_like(Py), Py
        )

    def size(self):
        return 3

    def trace_requests(self):
        return [self._request()]

    def residuals(self, context):
        return self.weight * context.read(self._request(), "y")

    def describe(self):
        return {"Note": "heights"}


class FailingBlock(HeightBlock):
    """Heights with rows forced to ``NaN`` while ``fail(row)`` holds."""

    def __init__(self, optic, fail):
        super().__init__(optic)
        self.fail = fail

    def residuals(self, context):
        rows = super().residuals(context)
        failed = be.array([1.0 if self.fail(i) else 0.0 for i in range(3)])
        return be.where(failed > 0.5, be.full((3,), be.nan), rows)


class RaisingBlock(HeightBlock):
    def residuals(self, context):
        raise RuntimeError("boom")


def _ray(lens, py):
    return {
        "optic": lens,
        "surface_number": -1,
        "Hx": 0.0,
        "Hy": 0.7,
        "Px": 0.0,
        "Py": py,
        "wavelength": 0.55,
    }


def _heights(lens):
    Py = be.array([0.25, 0.5, 0.75])
    lens.trace_generic(0.0, 0.7, be.zeros_like(Py), Py, 0.55)
    return lens.surfaces.y[-1, :]


def _problem(batching=True):
    lens = CookeTriplet()
    problem = OptimizationProblem(batching=batching)
    problem.add_operand("real_y_intercept", target=1.0, input_data=_ray(lens, 1.0))
    problem.add_operand("f2", target=50.0, input_data={"optic": lens})
    problem.add_variable(lens, "radius", surface_number=1)
    return lens, problem


@pytest.mark.parametrize("batching", [True, False])
def test_rows_follow_the_operands(set_test_backend, batching):
    lens, problem = _problem(batching)
    plain = be.to_numpy(problem.residual_vector()).copy()
    problem.add_residual_block(HeightBlock(lens))

    expected = np.concatenate([plain, 2.0 * be.to_numpy(_heights(lens))])
    assert problem.num_residuals == 5
    r = be.to_numpy(problem.residual_vector())
    assert_allclose(r, expected)
    assert_allclose(problem.fun_array(), expected**2)
    assert_allclose(problem.sum_squared(), np.sum(expected**2))


@pytest.mark.parametrize("batching", [True, False])
def test_block_without_operands(set_test_backend, batching):
    lens = CookeTriplet()
    problem = OptimizationProblem(batching=batching)
    problem.add_residual_block(HeightBlock(lens))
    expected = 2.0 * be.to_numpy(_heights(lens))
    assert_allclose(problem.residual_vector(), expected)
    assert_allclose(problem.sum_squared(), np.sum(expected**2))


def test_clear_residual_blocks(set_test_backend):
    lens, problem = _problem()
    plain = be.to_numpy(problem.residual_vector()).copy()
    problem.add_residual_block(HeightBlock(lens))
    problem.clear_residual_blocks()
    assert problem.num_residuals == 2
    assert_allclose(problem.residual_vector(), plain)


def test_add_rejects_a_non_block(set_test_backend):
    _, problem = _problem()
    with pytest.raises(ConfigurationError):
        problem.add_residual_block(object())


def test_block_rays_join_the_bundle(set_test_backend):
    lens, problem = _problem()
    problem.add_residual_block(HeightBlock(lens))
    problem.sum_squared()
    with _instrument.counting() as counts:
        problem.sum_squared()
    assert counts["trace_generic"] == 1


def test_identical_requests_share_a_segment(set_test_backend):
    lens = CookeTriplet()
    problem = OptimizationProblem()
    problem.add_residual_block(HeightBlock(lens))
    problem.add_residual_block(HeightBlock(lens))
    jobs = problem._batched_evaluator._bundles
    assert len(jobs) == 1
    assert len(jobs[0].segments) == 1
    assert problem.num_residuals == 6


def test_blocks_precede_constraint_rows(set_test_backend):
    lens, problem = _problem()
    problem.add_residual_block(HeightBlock(lens))
    problem.add_constraint("f2", min_val=10.0, input_data={"optic": lens})
    full = be.to_numpy(problem.residuals_and_constraints())
    row = be.to_numpy(problem.constraint_vector())
    assert full.size == 6
    assert_allclose(full[:5], problem.residual_vector())
    assert_allclose(full[5:], row)
    assert_allclose(row, 10.0 - be.to_numpy(lens.paraxial.f2()))


def test_exception_inside_a_block_gives_nan_rows(set_test_backend):
    lens, problem = _problem()
    problem.add_residual_block(RaisingBlock(lens))
    r = be.to_numpy(problem.residual_vector())
    assert np.all(np.isfinite(r[:2]))
    assert np.all(np.isnan(r[2:]))


def test_wrong_row_count_is_an_error(set_test_backend):
    class Wrong(HeightBlock):
        def size(self):
            return 2

    lens, problem = _problem()
    problem.add_residual_block(Wrong(lens))
    with pytest.raises(ValueError, match="rows"):
        problem.residual_vector()


def test_failed_trace_marks_only_the_block(set_test_backend):
    lens, problem = _problem()
    problem.clear_operands()
    problem.add_operand("f2", target=50.0, input_data={"optic": lens})
    block = HeightBlock(lens)

    def fail(*args, **kwargs):
        raise RuntimeError("trace failed")

    problem.add_residual_block(block)
    lens.trace_generic = fail
    r = be.to_numpy(problem.residual_vector())
    assert np.isfinite(r[0])
    assert np.all(np.isnan(r[1:]))


def test_info_prints_one_row_per_block(set_test_backend, capsys):
    lens, problem = _problem()
    problem.add_residual_block(HeightBlock(lens))
    problem.info()
    out = capsys.readouterr().out
    assert "HeightBlock" in out
    assert "heights" in out


def test_every_optimizer_family_sees_blocks(set_test_backend):
    lens, problem = _problem()
    problem.clear_operands()
    problem.add_residual_block(HeightBlock(lens))
    before = float(be.to_numpy(problem.sum_squared()))
    LeastSquares(problem).optimize(maxiter=8, method_choice="trf")
    middle = float(be.to_numpy(problem.sum_squared()))
    assert middle < before
    result = minimize(problem, "l-bfgs-b", maxiter=3)
    assert result.fun <= middle


def test_legacy_lm_counts_block_rows(set_test_backend):
    lens = CookeTriplet()
    problem = OptimizationProblem()
    problem.add_residual_block(HeightBlock(lens))
    for surface in (1, 2, 3):
        problem.add_variable(lens, "radius", surface_number=surface)
    # Three block rows and three variables: no fallback to "trf" is needed.
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*less than number of variables")
        LeastSquares(problem).optimize(maxiter=8, method_choice="lm")


def test_jacobian_includes_blocks(set_test_backend):
    lens, problem = _problem()
    problem.add_residual_block(HeightBlock(lens))
    evaluator = FiniteDiffEvaluator(problem)
    x = evaluator.read_x()
    assert evaluator.jacobian(x).shape == (5, 1)


def test_dls_converges_with_a_block(set_test_backend):
    lens = CookeTriplet()
    problem = OptimizationProblem()
    problem.add_residual_block(HeightBlock(lens))
    problem.add_variable(lens, "radius", surface_number=1)
    problem.add_variable(lens, "conic", surface_number=1)
    start = float(be.to_numpy(problem.sum_squared()))
    result = minimize(problem, "dls", maxiter=40)
    assert result.fun < start
    assert_allclose(problem.sum_squared(), result.fun)


# ---------------------------------------------------------------------------
# Ray-set re-validation
# ---------------------------------------------------------------------------


def _variable_problem(block_factory):
    lens = CookeTriplet()
    problem = OptimizationProblem()
    problem.add_operand("f2", target=45.0, input_data={"optic": lens})
    block = problem.add_residual_block(block_factory(lens))
    problem.add_variable(lens, "radius", surface_number=1)
    return lens, problem, block


def test_rows_failing_at_the_start_are_dropped_with_one_warning(set_test_backend):
    lens, problem, block = _variable_problem(
        lambda optic: FailingBlock(optic, lambda row: row == 1)
    )
    evaluator = FiniteDiffEvaluator(problem)
    with pytest.warns(UserWarning, match="1 of 3 rows of FailingBlock") as record:
        result = dls(evaluator, max_nfev=30)
    assert len([w for w in record if "Rays failed" in str(w.message)]) == 1
    assert result.fun[2] == 0.0
    assert np.all(np.isfinite(result.fun))
    assert block.dropped_rows.size == 0  # only for the rest of that run


def test_operand_rows_are_never_dropped(set_test_backend):
    lens, problem, _ = _variable_problem(HeightBlock)
    problem.clear_operands()
    problem.add_operand("f2", target=be.nan, input_data={"optic": lens})
    evaluator = FiniteDiffEvaluator(problem)
    with pytest.raises(ConfigurationError, match="f2"):
        dls(evaluator, max_nfev=30)


def test_dead_block_at_the_start_raises_naming_it(set_test_backend):
    _, problem, block = _variable_problem(
        lambda optic: FailingBlock(optic, lambda row: True)
    )
    evaluator = FiniteDiffEvaluator(problem)
    with pytest.raises(ConfigurationError, match="FailingBlock"):
        dls(evaluator, max_nfev=30)
    assert block.dropped_rows.size == 0


def test_row_failing_in_every_trial_is_dropped(set_test_backend):
    start = []

    def fail(row):
        # Fails everywhere but at the start point, so no step can pass.
        return row == 2 and start and start[0] != _radius(lens_box[0])

    lens_box = []

    def _radius(lens):
        return float(np.ravel(be.to_numpy(lens.surfaces[1].geometry.radius))[0])

    def build(optic):
        lens_box.append(optic)
        return FailingBlock(optic, fail)

    lens, problem, block = _variable_problem(build)
    start.append(_radius(lens))
    evaluator = FiniteDiffEvaluator(problem)
    with pytest.warns(UserWarning, match="1 of 3 rows"):
        result = dls(evaluator, max_nfev=400)
    assert result.nit > 0
    assert result.fun[3] == 0.0


def test_a_rejected_trial_keeps_the_ray_set(set_test_backend):
    """A ray failing only far from the start rejects the trial, nothing else."""

    def radius(lens):
        return float(np.ravel(be.to_numpy(lens.surfaces[1].geometry.radius))[0])

    box = []

    def build(optic):
        box.append(optic)
        limit = radius(optic) + 0.5
        return FailingBlock(optic, lambda row: abs(radius(box[0])) > abs(limit))

    _, problem, _ = _variable_problem(build)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Rays failed")
        result = dls(FiniteDiffEvaluator(problem), max_nfev=60)
    assert np.all(result.fun != 0.0)


def test_describe_rows_names_blocks_and_constraints(set_test_backend):
    lens, problem = _problem()
    problem.add_residual_block(HeightBlock(lens))
    problem.add_constraint("f2", min_val=10.0, input_data={"optic": lens})
    text = describe_rows(problem, [1, 3, 5, 9])
    assert text == "#1 f2, #3 HeightBlock[1], #5 f2, #9"
