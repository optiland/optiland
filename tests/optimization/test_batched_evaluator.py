"""Tests for the BatchedRayEvaluator.

These tests verify that:
1. Batched evaluation produces the same results as standard evaluation.
2. It works with both trace_generic and trace (distribution) operands.
3. Mixed operand types (ray + non-ray) are handled correctly.
4. Inequality operands (min/max bounds) work correctly.
5. The evaluator can be refreshed after operand changes.
"""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.optimization import optimization
from optiland.optimization.batched_evaluator import BatchedRayEvaluator
from optiland.optimization.operand.operand import operand_registry
from optiland.samples.microscopes import Microscope20x, Objective60x
from optiland.samples.objectives import CookeTriplet

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_problem_with_ray_operands(lens=None):
    """Create a problem with several trace_generic ray operands."""
    if lens is None:
        lens = CookeTriplet()
    problem = optimization.OptimizationProblem()

    # Add some trace_generic operands (different fields, same wavelength)
    for op_type in ("real_x_intercept", "real_y_intercept", "real_L"):
        problem.add_operand(
            operand_type=op_type,
            target=0.0,
            weight=1.0,
            input_data={
                "optic": lens,
                "surface_number": -1,
                "Hx": 0.0,
                "Hy": 0.0,
                "Px": 0.0,
                "Py": 1.0,
                "wavelength": 0.55,
            },
        )
    return problem, lens


def _make_problem_with_rms_spot(lens=None):
    """Create a problem with rms_spot_size operands."""
    if lens is None:
        lens = CookeTriplet()
    problem = optimization.OptimizationProblem()

    # Two rms_spot_size operands — same trace params, different surface
    # (they share one trace)
    problem.add_operand(
        operand_type="rms_spot_size",
        target=0.0,
        weight=1.0,
        input_data={
            "optic": lens,
            "surface_number": -1,
            "Hx": 0.0,
            "Hy": 0.0,
            "wavelength": 0.55,
            "num_rays": 50,
            "distribution": "hexapolar",
        },
    )
    return problem, lens


def _make_mixed_problem(lens=None):
    """Create a problem with ray, rms_spot, paraxial, and lens operands."""
    if lens is None:
        lens = CookeTriplet()
    problem = optimization.OptimizationProblem()

    # Paraxial operand (non-ray)
    problem.add_operand(
        operand_type="f2",
        target=50.0,
        weight=1.0,
        input_data={"optic": lens},
    )

    # Ray operands (trace_generic)
    problem.add_operand(
        operand_type="real_y_intercept",
        target=0.0,
        weight=1.0,
        input_data={
            "optic": lens,
            "surface_number": -1,
            "Hx": 0.0,
            "Hy": 0.0,
            "Px": 0.0,
            "Py": 1.0,
            "wavelength": 0.55,
        },
    )

    # rms_spot_size operand (trace)
    problem.add_operand(
        operand_type="rms_spot_size",
        target=0.0,
        weight=1.0,
        input_data={
            "optic": lens,
            "surface_number": -1,
            "Hx": 0.0,
            "Hy": 0.0,
            "wavelength": 0.55,
            "num_rays": 50,
            "distribution": "hexapolar",
        },
    )

    return problem, lens


# ---------------------------------------------------------------------------
# Tests: Batched vs Standard evaluation — numerical agreement
# ---------------------------------------------------------------------------


class TestBatchedVsStandard:
    """Verify that batched evaluation gives the same result as standard."""

    def test_trace_generic_operands_match(self):
        """Batched trace_generic operands produce same sum_squared."""
        problem, _ = _make_problem_with_ray_operands()
        standard = float(be.to_numpy(problem.sum_squared()))

        evaluator = BatchedRayEvaluator(problem)
        batched = float(be.to_numpy(evaluator.sum_squared()))

        assert batched == pytest.approx(standard, rel=1e-6)

    def test_rms_spot_operands_match(self):
        """Batched rms_spot_size operands produce same sum_squared."""
        problem, _ = _make_problem_with_rms_spot()
        standard = float(be.to_numpy(problem.sum_squared()))

        evaluator = BatchedRayEvaluator(problem)
        batched = float(be.to_numpy(evaluator.sum_squared()))

        assert batched == pytest.approx(standard, rel=1e-6)

    def test_mixed_operands_match(self):
        """Mixed ray + non-ray operands produce same sum_squared."""
        problem, _ = _make_mixed_problem()
        standard = float(be.to_numpy(problem.sum_squared()))

        evaluator = BatchedRayEvaluator(problem)
        batched = float(be.to_numpy(evaluator.sum_squared()))

        assert batched == pytest.approx(standard, rel=1e-6)

    def test_fun_array_match(self):
        """fun_array gives the same per-operand values."""
        problem, _ = _make_mixed_problem()
        standard = be.to_numpy(problem.fun_array())

        evaluator = BatchedRayEvaluator(problem)
        batched = be.to_numpy(evaluator.fun_array())

        assert len(batched) == len(standard)
        for s, b in zip(standard, batched, strict=True):
            assert float(b) == pytest.approx(float(s), rel=1e-6)


# ---------------------------------------------------------------------------
# Tests: Specific operand types via batching
# ---------------------------------------------------------------------------


class TestTraceGenericOperands:
    """Tests for all individual trace_generic operand types."""

    @pytest.mark.parametrize(
        "op_type",
        [
            "real_x_intercept",
            "real_y_intercept",
            "real_z_intercept",
            "real_x_intercept_lcs",
            "real_y_intercept_lcs",
            "real_z_intercept_lcs",
            "real_L",
            "real_M",
            "real_N",
        ],
    )
    def test_single_operand_matches(self, op_type):
        lens = CookeTriplet()
        problem = optimization.OptimizationProblem()

        problem.add_operand(
            operand_type=op_type,
            target=0.0,
            weight=1.0,
            input_data={
                "optic": lens,
                "surface_number": 3,
                "Hx": 0.0,
                "Hy": 0.0,
                "Px": 0.0,
                "Py": 0.5,
                "wavelength": 0.55,
            },
        )

        standard = float(be.to_numpy(problem.sum_squared()))

        evaluator = BatchedRayEvaluator(problem)
        batched = float(be.to_numpy(evaluator.sum_squared()))

        assert batched == pytest.approx(standard, rel=1e-6)

    def test_aoi_operand_matches(self):
        lens = CookeTriplet()
        problem = optimization.OptimizationProblem()

        problem.add_operand(
            operand_type="AOI",
            target=0.0,
            weight=1.0,
            input_data={
                "optic": lens,
                "surface_number": 3,
                "Hx": 0.0,
                "Hy": 0.0,
                "Px": 0.0,
                "Py": 0.5,
                "wavelength": 0.55,
            },
        )

        standard = float(be.to_numpy(problem.sum_squared()))

        evaluator = BatchedRayEvaluator(problem)
        batched = float(be.to_numpy(evaluator.sum_squared()))

        assert batched == pytest.approx(standard, rel=1e-6)


# ---------------------------------------------------------------------------
# Tests: Multiple operands sharing a trace
# ---------------------------------------------------------------------------


class TestBatching:
    """Verify that multiple operands sharing a trace produce correct results."""

    def test_multiple_operands_same_wavelength(self):
        """Multiple trace_generic operands on same optic+wavelength batch."""
        lens = CookeTriplet()
        problem = optimization.OptimizationProblem()

        # Add several operands — same optic, same wavelength, different Px/Py
        for py in [0.0, 0.5, 1.0]:
            problem.add_operand(
                operand_type="real_y_intercept",
                target=0.0,
                weight=1.0,
                input_data={
                    "optic": lens,
                    "surface_number": -1,
                    "Hx": 0.0,
                    "Hy": 0.0,
                    "Px": 0.0,
                    "Py": py,
                    "wavelength": 0.55,
                },
            )

        standard = float(be.to_numpy(problem.sum_squared()))
        evaluator = BatchedRayEvaluator(problem)
        batched = float(be.to_numpy(evaluator.sum_squared()))

        assert batched == pytest.approx(standard, rel=1e-6)

    def test_operands_different_wavelengths(self):
        """Operands with different wavelengths go to different trace jobs."""
        lens = CookeTriplet()
        problem = optimization.OptimizationProblem()

        for wl in [0.48, 0.55, 0.65]:
            problem.add_operand(
                operand_type="real_y_intercept",
                target=0.0,
                weight=1.0,
                input_data={
                    "optic": lens,
                    "surface_number": -1,
                    "Hx": 0.0,
                    "Hy": 0.0,
                    "Px": 0.0,
                    "Py": 1.0,
                    "wavelength": wl,
                },
            )

        standard = float(be.to_numpy(problem.sum_squared()))
        evaluator = BatchedRayEvaluator(problem)
        batched = float(be.to_numpy(evaluator.sum_squared()))

        # Should create 3 separate trace_generic jobs (one per wavelength)
        assert len(evaluator._generic_jobs) == 3
        assert batched == pytest.approx(standard, rel=1e-6)

    def test_rms_spot_shared_trace(self):
        """Two rms_spot_size with same params share one trace."""
        lens = CookeTriplet()
        problem = optimization.OptimizationProblem()

        # Two operands with same trace params but different weights
        for weight in [1.0, 2.0]:
            problem.add_operand(
                operand_type="rms_spot_size",
                target=0.0,
                weight=weight,
                input_data={
                    "optic": lens,
                    "surface_number": -1,
                    "Hx": 0.0,
                    "Hy": 0.0,
                    "wavelength": 0.55,
                    "num_rays": 50,
                    "distribution": "hexapolar",
                },
            )

        standard = float(be.to_numpy(problem.sum_squared()))
        evaluator = BatchedRayEvaluator(problem)
        batched = float(be.to_numpy(evaluator.sum_squared()))

        # Should share one distribution trace job
        assert len(evaluator._distribution_jobs) == 1
        assert batched == pytest.approx(standard, rel=1e-6)


# ---------------------------------------------------------------------------
# Tests: Inequality operands
# ---------------------------------------------------------------------------


class TestInequalityOperands:
    """Test that inequality (min/max) operands work with batching."""

    def test_min_max_ray_operand(self):
        lens = CookeTriplet()
        problem = optimization.OptimizationProblem()

        problem.add_operand(
            operand_type="real_y_intercept",
            min_val=-5.0,
            max_val=5.0,
            weight=1.0,
            input_data={
                "optic": lens,
                "surface_number": -1,
                "Hx": 0.0,
                "Hy": 0.0,
                "Px": 0.0,
                "Py": 1.0,
                "wavelength": 0.55,
            },
        )

        standard = float(be.to_numpy(problem.sum_squared()))
        evaluator = BatchedRayEvaluator(problem)
        batched = float(be.to_numpy(evaluator.sum_squared()))

        assert batched == pytest.approx(standard, rel=1e-6)

    def test_min_only_ray_operand(self):
        lens = CookeTriplet()
        problem = optimization.OptimizationProblem()

        problem.add_operand(
            operand_type="real_y_intercept",
            min_val=-10.0,
            weight=1.0,
            input_data={
                "optic": lens,
                "surface_number": -1,
                "Hx": 0.0,
                "Hy": 0.0,
                "Px": 0.0,
                "Py": 1.0,
                "wavelength": 0.55,
            },
        )

        standard = float(be.to_numpy(problem.sum_squared()))
        evaluator = BatchedRayEvaluator(problem)
        batched = float(be.to_numpy(evaluator.sum_squared()))

        assert batched == pytest.approx(standard, rel=1e-6)

    def test_mixed_scalar_and_singleton_direct_inequality(self, monkeypatch):
        lens = Objective60x()

        # Mimic a direct operand that returns a singleton backend array.
        monkeypatch.setitem(
            operand_registry._registry,
            "_singleton_metric_for_batch_test",
            lambda optic: be.array([1.0]),
        )

        standard_problem = optimization.OptimizationProblem(batching=False)
        standard_problem.add_operand(
            operand_type="f2",
            target=90.0,
            weight=1.0,
            input_data={"optic": lens},
        )
        standard_problem.add_operand(
            operand_type="_singleton_metric_for_batch_test",
            min_val=0.5,
            weight=1.0,
            input_data={"optic": lens},
        )

        batched_problem = optimization.OptimizationProblem(batching=True)
        batched_problem.add_operand(
            operand_type="f2",
            target=90.0,
            weight=1.0,
            input_data={"optic": lens},
        )
        batched_problem.add_operand(
            operand_type="_singleton_metric_for_batch_test",
            min_val=0.5,
            weight=1.0,
            input_data={"optic": lens},
        )

        standard = float(be.to_numpy(standard_problem.sum_squared()))
        batched = float(be.to_numpy(batched_problem.sum_squared()))
        assert batched == pytest.approx(standard, rel=1e-6)


# ---------------------------------------------------------------------------
# Tests: Problem integration (enable_batching / disable_batching)
# ---------------------------------------------------------------------------


class TestProblemIntegration:
    """Test that enable_batching/disable_batching on OptimizationProblem works."""

    def test_default_batching_and_enable_disable(self):
        problem, _ = _make_mixed_problem()

        # Batching is on by default
        assert problem.batching_enabled
        batched_default = float(be.to_numpy(problem.sum_squared()))

        # Disable batching for legacy per-operand behavior
        problem.disable_batching()
        assert not problem.batching_enabled
        standard = float(be.to_numpy(problem.sum_squared()))
        assert batched_default == pytest.approx(standard, rel=1e-6)

        # Re-enable batching
        problem.enable_batching()
        assert problem.batching_enabled
        batched_again = float(be.to_numpy(problem.sum_squared()))
        assert batched_again == pytest.approx(standard, rel=1e-6)

    def test_opt_out_constructor_starts_without_batching(self):
        lens = CookeTriplet()
        problem = optimization.OptimizationProblem(batching=False)
        assert not problem.batching_enabled

        problem.add_operand(
            operand_type="f2",
            target=90.0,
            weight=1.0,
            input_data={"optic": lens},
        )

        standard = float(be.to_numpy(problem.sum_squared()))

        problem.enable_batching()
        batched = float(be.to_numpy(problem.sum_squared()))
        assert batched == pytest.approx(standard, rel=1e-6)

        problem.disable_batching()
        assert not problem.batching_enabled

    def test_batching_refresh_on_add_operand(self):
        """Adding an operand refreshes the evaluator."""
        lens = CookeTriplet()
        problem = optimization.OptimizationProblem()

        problem.add_operand(
            operand_type="f2",
            target=50.0,
            weight=1.0,
            input_data={"optic": lens},
        )
        problem.enable_batching()

        # Add another operand — evaluator should auto-refresh
        problem.add_operand(
            operand_type="real_y_intercept",
            target=0.0,
            weight=1.0,
            input_data={
                "optic": lens,
                "surface_number": -1,
                "Hx": 0.0,
                "Hy": 0.0,
                "Px": 0.0,
                "Py": 1.0,
                "wavelength": 0.55,
            },
        )

        # Should still produce correct results
        expected = float(be.to_numpy(problem.fun_array().sum()))
        problem.disable_batching()
        problem.enable_batching()
        actual = float(be.to_numpy(problem.sum_squared()))
        assert actual == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Tests: Non-ray operands
# ---------------------------------------------------------------------------


class TestNonRayOperands:
    """Test paraxial, aberration, and lens operands work through batching."""

    def test_paraxial_only(self):
        lens = Objective60x()
        problem = optimization.OptimizationProblem()

        problem.add_operand(
            operand_type="f2",
            target=90.0,
            weight=1.0,
            input_data={"optic": lens},
        )

        standard = float(be.to_numpy(problem.sum_squared()))
        evaluator = BatchedRayEvaluator(problem)
        batched = float(be.to_numpy(evaluator.sum_squared()))

        assert batched == pytest.approx(standard, rel=1e-6)

    def test_edge_thickness(self):
        lens = CookeTriplet()

        problem = optimization.OptimizationProblem()

        problem.add_operand(
            operand_type="edge_thickness",
            target=2.0,
            weight=1.0,
            input_data={"optic": lens, "surface_number": 1},
        )

        standard = float(be.to_numpy(problem.sum_squared()))
        evaluator = BatchedRayEvaluator(problem)
        batched = float(be.to_numpy(evaluator.sum_squared()))

        assert batched == pytest.approx(standard, rel=1e-6)

    def test_aberration_operand(self):
        lens = CookeTriplet()
        problem = optimization.OptimizationProblem()

        problem.add_operand(
            operand_type="TSC",
            target=0.0,
            weight=1.0,
            input_data={"optic": lens, "surface_number": 1},
        )

        standard = float(be.to_numpy(problem.sum_squared()))
        evaluator = BatchedRayEvaluator(problem)
        batched = float(be.to_numpy(evaluator.sum_squared()))

        assert batched == pytest.approx(standard, rel=1e-6)


# ---------------------------------------------------------------------------
# Tests: Multi-wavelength rms_spot_size falls back correctly
# ---------------------------------------------------------------------------


class TestMultiWavelength:
    """Test that rms_spot_size with wavelength='all' falls back to direct."""

    def test_rms_spot_all_wavelengths(self):
        lens = CookeTriplet()
        problem = optimization.OptimizationProblem()

        problem.add_operand(
            operand_type="rms_spot_size",
            target=0.0,
            weight=1.0,
            input_data={
                "optic": lens,
                "surface_number": -1,
                "Hx": 0.0,
                "Hy": 0.0,
                "wavelength": "all",
                "num_rays": 50,
                "distribution": "hexapolar",
            },
        )

        standard = float(be.to_numpy(problem.sum_squared()))
        evaluator = BatchedRayEvaluator(problem)
        batched = float(be.to_numpy(evaluator.sum_squared()))

        assert batched == pytest.approx(standard, rel=1e-6)


# ---------------------------------------------------------------------------
# Tests: Evaluator structure
# ---------------------------------------------------------------------------


class TestEvaluatorStructure:
    """Test that the evaluator creates the expected job structure."""

    def test_generic_job_count(self):
        """One job per (optic, wavelength) group."""
        lens = CookeTriplet()
        problem = optimization.OptimizationProblem()

        # 3 operands at wavelength 0.55, 2 at 0.48
        for wl, count in [(0.55, 3), (0.48, 2)]:
            for _ in range(count):
                problem.add_operand(
                    operand_type="real_y_intercept",
                    target=0.0,
                    weight=1.0,
                    input_data={
                        "optic": lens,
                        "surface_number": -1,
                        "Hx": 0.0,
                        "Hy": 0.0,
                        "Px": 0.0,
                        "Py": 1.0,
                        "wavelength": wl,
                    },
                )

        evaluator = BatchedRayEvaluator(problem)
        assert len(evaluator._generic_jobs) == 2
        assert len(evaluator._generic_jobs[0].ray_params) == 3
        assert len(evaluator._generic_jobs[1].ray_params) == 2

    def test_distribution_job_count(self):
        """One job per unique (optic, Hx, Hy, wavelength, num_rays, dist)."""
        lens = CookeTriplet()
        problem = optimization.OptimizationProblem()

        # Two identical trace params -> 1 job
        for _ in range(2):
            problem.add_operand(
                operand_type="rms_spot_size",
                target=0.0,
                weight=1.0,
                input_data={
                    "optic": lens,
                    "surface_number": -1,
                    "Hx": 0.0,
                    "Hy": 0.0,
                    "wavelength": 0.55,
                    "num_rays": 50,
                },
            )

        # Different Hy -> new job
        problem.add_operand(
            operand_type="rms_spot_size",
            target=0.0,
            weight=1.0,
            input_data={
                "optic": lens,
                "surface_number": -1,
                "Hx": 0.0,
                "Hy": 0.5,
                "wavelength": 0.55,
                "num_rays": 50,
            },
        )

        evaluator = BatchedRayEvaluator(problem)
        assert len(evaluator._distribution_jobs) == 2

    def test_refresh(self):
        """Calling refresh re-creates the plan."""
        problem, _ = _make_problem_with_ray_operands()
        evaluator = BatchedRayEvaluator(problem)
        initial_plan_len = len(evaluator._operand_plan)

        # Add an operand without going through the problem
        problem.operands.add(
            "f2",
            target=50.0,
            weight=1.0,
            input_data={"optic": CookeTriplet()},
        )

        evaluator.refresh()
        assert len(evaluator._operand_plan) == initial_plan_len + 1


class TestEvaluatorHelpers:
    """Coverage-oriented tests for internal helper methods."""

    def test_ensure_plan_current_rebuilds_when_mismatched(self):
        problem, _ = _make_problem_with_ray_operands()
        evaluator = BatchedRayEvaluator(problem)

        # Simulate stale internal state and ensure helper re-analyzes.
        evaluator._operand_plan = []
        evaluator._ensure_plan_current()

        assert len(evaluator._operand_plan) == len(problem.operands)

    def test_safe_execute_returns_none_on_exception(self):
        class _BadJob:
            def execute(self):
                raise RuntimeError("boom")

        assert BatchedRayEvaluator._safe_execute(_BadJob()) is None

    def test_evaluate_direct_operands_skips_zero_effective_weight(self):
        lens = CookeTriplet()
        problem = optimization.OptimizationProblem()
        problem.add_operand(
            operand_type="f2",
            target=50.0,
            weight=1.0,
            input_data={"optic": lens},
        )
        evaluator = BatchedRayEvaluator(problem)

        # Force the zero-effective-weight path.
        op = problem.operands[0]
        op.effective_weight = lambda optic=None: 0.0

        raw_values = [None]
        evaluator._evaluate_direct_operands(raw_values)
        assert raw_values[0] is None

        # No contribution terms should be produced for zero-effective-weight ops.
        assert evaluator._build_contribution_terms(raw_values) == []

    def test_evaluate_direct_operands_unknown_operand_raises(self):
        lens = CookeTriplet()
        problem = optimization.OptimizationProblem()
        problem.add_operand(
            operand_type="f2",
            target=50.0,
            weight=1.0,
            input_data={"optic": lens},
        )
        evaluator = BatchedRayEvaluator(problem)

        # Force unknown direct metric lookup branch.
        problem.operands[0].operand_type = "not_registered_operand"

        with pytest.raises(ValueError, match="Unknown operand type"):
            evaluator._evaluate_direct_operands([None])

    def test_build_contribution_terms_raises_for_missing_value(self):
        lens = CookeTriplet()
        problem = optimization.OptimizationProblem()
        problem.add_operand(
            operand_type="f2",
            target=50.0,
            weight=1.0,
            input_data={"optic": lens},
        )
        evaluator = BatchedRayEvaluator(problem)

        with pytest.raises(RuntimeError, match="was not evaluated"):
            evaluator._build_contribution_terms([None])


# ---------------------------------------------------------------------------
# Tests: Scipy optimization with batching enabled
# ---------------------------------------------------------------------------


class TestScipyWithBatching:
    """Test that scipy optimizers work when batching is enabled."""

    def test_optimizer_generic_with_batching(self):
        lens = Microscope20x()
        problem = optimization.OptimizationProblem()
        problem.add_variable(
            lens,
            "radius",
            surface_number=1,
            min_val=10,
            max_val=100,
        )
        problem.add_operand(
            operand_type="f2",
            target=90,
            weight=1.0,
            input_data={"optic": lens},
        )

        # Enable batching before optimization
        problem.enable_batching()

        optimizer = optimization.OptimizerGeneric(problem)
        result = optimizer.optimize(maxiter=10, disp=False, tol=1e-3)
        assert result.success

    def test_optimizer_with_ray_operands_and_batching(self):
        lens = CookeTriplet()
        problem = optimization.OptimizationProblem()
        problem.add_variable(
            lens,
            "radius",
            surface_number=1,
            min_val=10,
            max_val=100,
        )

        problem.add_operand(
            operand_type="real_y_intercept",
            target=0.0,
            weight=1.0,
            input_data={
                "optic": lens,
                "surface_number": -1,
                "Hx": 0.0,
                "Hy": 0.0,
                "Px": 0.0,
                "Py": 1.0,
                "wavelength": 0.55,
            },
        )

        problem.add_operand(
            operand_type="rms_spot_size",
            target=0.0,
            weight=1.0,
            input_data={
                "optic": lens,
                "surface_number": -1,
                "Hx": 0.0,
                "Hy": 0.0,
                "wavelength": 0.55,
                "num_rays": 50,
            },
        )

        problem.enable_batching()

        optimizer = optimization.OptimizerGeneric(problem)
        result = optimizer.optimize(maxiter=5, disp=False, tol=1e-3)
        # Just verify it ran without error
        assert result is not None


# ---------------------------------------------------------------------------
# Torch autograd compatibility
# ---------------------------------------------------------------------------


def _torch_available():
    """Check whether the torch backend is available."""
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
class TestTorchAutograd:
    """Verify that the batched evaluator preserves PyTorch autograd graphs."""

    @pytest.fixture(autouse=True)
    def _set_torch_backend(self):
        """Switch to torch backend for the duration of each test."""
        original = be.get_backend()
        be.set_backend("torch")
        be.grad_mode.enable()
        yield
        be.grad_mode.disable()
        be.set_backend(original)

    def test_sum_squared_supports_backward(self):
        """sum_squared() returns a tensor with a grad_fn when batching is
        enabled under the torch backend + grad_mode."""
        import torch

        lens = CookeTriplet()
        problem = optimization.OptimizationProblem()

        # Variable: radius of surface 1
        problem.add_variable(
            lens,
            "radius",
            surface_number=1,
            min_val=10,
            max_val=100,
        )

        # Operand: y-intercept on image surface
        problem.add_operand(
            operand_type="real_y_intercept",
            target=0.0,
            weight=1.0,
            input_data={
                "optic": lens,
                "surface_number": -1,
                "Hx": 0.0,
                "Hy": 0.0,
                "Px": 0.0,
                "Py": 1.0,
                "wavelength": 0.55,
            },
        )

        problem.enable_batching()

        # Make the variable a torch.nn.Parameter so we can track grads
        param = torch.nn.Parameter(be.array(problem.variables[0].variable.get_value()))
        problem.variables[0].variable.update_value(param)
        problem.update_optics()

        loss = problem.sum_squared()

        # The loss should be a torch tensor with an autograd graph
        assert isinstance(loss, torch.Tensor)
        assert loss.grad_fn is not None, (
            "sum_squared() should produce a tensor with grad_fn"
        )

        # backward() should not raise
        loss.backward()

        # The parameter should have a non-None gradient
        assert param.grad is not None, (
            "Parameter gradient should be populated after backward()"
        )

    def test_torch_adam_optimizer_with_batching(self):
        """TorchAdamOptimizer works with batching enabled and reduces loss.

        Uses a paraxial operand (``f2``) because multi-step backward
        through ``trace_generic`` is a known pre-existing limitation of
        the ray-tracing graph (also fails without batching).
        """
        from optiland.optimization import TorchAdamOptimizer

        lens = CookeTriplet()
        # Match the setup used in test_torch_optimization.py
        lens.surface_group.surfaces[1].radius = 5.0
        problem = optimization.OptimizationProblem()

        problem.add_variable(
            lens,
            "radius",
            surface_number=1,
            min_val=1.0,
            max_val=10.0,
        )

        # Paraxial operand — the existing torch tests also use f2
        problem.add_operand(
            operand_type="f2",
            target=12.0,
            weight=1.0,
            input_data={"optic": lens},
        )

        problem.enable_batching()
        problem.update_optics()
        initial_loss = problem.sum_squared().item()

        optimizer = TorchAdamOptimizer(problem)
        result = optimizer.optimize(n_steps=50, lr=1e-2, disp=False)

        assert result.fun < initial_loss, (
            f"Loss should decrease: {result.fun} >= {initial_loss}"
        )

    def test_batched_fun_array_gradient_flows(self):
        """fun_array() elements retain grad_fn under torch backend."""
        import torch

        lens = CookeTriplet()
        problem = optimization.OptimizationProblem()

        problem.add_variable(
            lens,
            "radius",
            surface_number=1,
            min_val=10,
            max_val=100,
        )

        # Two operands to make batching meaningful
        for py_val in (0.5, 1.0):
            problem.add_operand(
                operand_type="real_y_intercept",
                target=0.0,
                weight=1.0,
                input_data={
                    "optic": lens,
                    "surface_number": -1,
                    "Hx": 0.0,
                    "Hy": 0.0,
                    "Px": 0.0,
                    "Py": py_val,
                    "wavelength": 0.55,
                },
            )

        problem.enable_batching()

        param = torch.nn.Parameter(be.array(problem.variables[0].variable.get_value()))
        problem.variables[0].variable.update_value(param)
        problem.update_optics()

        arr = problem.fun_array()
        assert isinstance(arr, torch.Tensor)
        assert arr.grad_fn is not None, (
            "fun_array() should produce tensors with grad_fn"
        )

    def test_batched_matches_standard_under_torch(self):
        """Batched and standard evaluation agree numerically under torch."""
        lens = CookeTriplet()

        # Standard (no batching)
        p_std = optimization.OptimizationProblem()
        p_std.add_operand(
            operand_type="real_y_intercept",
            target=0.0,
            weight=1.0,
            input_data={
                "optic": lens,
                "surface_number": -1,
                "Hx": 0.0,
                "Hy": 0.0,
                "Px": 0.0,
                "Py": 1.0,
                "wavelength": 0.55,
            },
        )
        standard_val = float(be.to_numpy(p_std.sum_squared()))

        # Batched
        p_bat = optimization.OptimizationProblem()
        p_bat.add_operand(
            operand_type="real_y_intercept",
            target=0.0,
            weight=1.0,
            input_data={
                "optic": lens,
                "surface_number": -1,
                "Hx": 0.0,
                "Hy": 0.0,
                "Px": 0.0,
                "Py": 1.0,
                "wavelength": 0.55,
            },
        )
        p_bat.enable_batching()
        batched_val = float(be.to_numpy(p_bat.sum_squared()))

        assert abs(standard_val - batched_val) < 1e-10, (
            f"standard={standard_val}, batched={batched_val}"
        )
