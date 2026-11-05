"""Opt-in physical thickness limits.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.optimization import LeastSquares, OptimizationProblem, minimize
from optiland.optimization.failure import ConfigurationError
from optiland.samples.objectives import CookeTriplet

from ..utils import assert_allclose

pytestmark = pytest.mark.filterwarnings("ignore")

# Cooke triplet: surfaces 1, 3, 5 start glass; 2 and 4 start air.
GLASS_GAPS = {1, 3, 5}
AIR_GAPS = {2, 4}


def _surfaces(items):
    specs = [item.input_data for item in items]
    return [spec["surface_number"] for spec in specs]


def test_nothing_is_applied_by_default(set_test_backend):
    problem = OptimizationProblem()
    assert len(problem.constraints) == 0
    assert len(problem.operands) == 0


def test_constraints_split_glass_and_air(set_test_backend):
    lens = CookeTriplet()
    problem = OptimizationProblem()
    made = problem.add_physical_constraints(
        lens, min_center_glass=1.0, min_center_air=0.5
    )
    assert len(made) == len(problem.constraints) == 5
    limits = {c.input_data["surface_number"]: c.min_val for c in made}
    assert {s for s, v in limits.items() if v == 1.0} == GLASS_GAPS
    assert {s for s, v in limits.items() if v == 0.5} == AIR_GAPS
    assert all(c.operand_type == "center_thickness" for c in made)


def test_image_and_object_gaps_are_left_out(set_test_backend):
    lens = CookeTriplet()
    problem = OptimizationProblem()
    made = problem.add_physical_constraints(lens, min_center_air=0.1)
    assert 0 not in _surfaces(made)
    assert len(lens.surfaces.surfaces) - 2 not in _surfaces(made)


def test_max_center_covers_both_media_with_overrides(set_test_backend):
    lens = CookeTriplet()
    problem = OptimizationProblem()
    made = problem.add_physical_constraints(lens, max_center=9.0, max_center_air=7.0)
    limits = {c.input_data["surface_number"]: c.max_val for c in made}
    assert {s for s, v in limits.items() if v == 9.0} == GLASS_GAPS
    assert {s for s, v in limits.items() if v == 7.0} == AIR_GAPS


def test_all_limits_emit_center_and_edge(set_test_backend):
    lens = CookeTriplet()
    problem = OptimizationProblem()
    made = problem.add_physical_constraints(
        lens,
        min_center_glass=1.0,
        min_edge_glass=0.5,
        min_center_air=0.1,
        min_edge_air=0.05,
        max_center=20.0,
        max_edge_glass=15.0,
        max_edge_air=16.0,
    )
    assert len(made) == 10
    assert sum(c.operand_type == "edge_thickness" for c in made) == 5


def test_penalty_mode_adds_operands(set_test_backend):
    lens = CookeTriplet()
    problem = OptimizationProblem()
    made = problem.add_physical_constraints(
        lens, min_edge_glass=1e6, min_center_air=0.1, mode="penalty", weight=2.0
    )
    assert len(problem.constraints) == 0
    assert len(problem.operands) == 5
    assert made[0] is problem.operands.operands[0]
    assert all(op.min_val is not None and op.target is None for op in made)
    assert float(be.to_numpy(problem.sum_squared())) > 0.0


def test_penalty_is_zero_when_satisfied(set_test_backend):
    lens = CookeTriplet()
    problem = OptimizationProblem()
    problem.add_physical_constraints(lens, min_center_air=1e-6, mode="penalty")
    assert_allclose(problem.sum_squared(), 0.0, atol=1e-20)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "at least one limit"),
        ({"min_center_air": 1.0, "mode": "soft"}, "Unknown mode"),
        ({"min_center_air": 5.0, "max_center": 1.0}, "min_val exceeds max_val"),
    ],
)
def test_bad_arguments_raise(set_test_backend, kwargs, match):
    problem = OptimizationProblem()
    with pytest.raises(ConfigurationError, match=match):
        problem.add_physical_constraints(CookeTriplet(), **kwargs)


def test_semi_apertures_are_initialized_for_edge_limits(set_test_backend):
    lens = CookeTriplet()
    for surface in lens.surfaces.surfaces:
        surface.semi_aperture = None
    problem = OptimizationProblem()
    problem.add_physical_constraints(lens, min_edge_glass=0.1)
    assert lens.surfaces[1].semi_aperture is not None
    problem.constraint_vector()


def test_limits_are_magnitudes_after_a_mirror(set_test_backend):
    lens = Optic()
    lens.surfaces.add(index=0, thickness=be.inf)
    lens.surfaces.add(index=1, thickness=5.0, is_stop=True)
    lens.surfaces.add(index=2, radius=-50.0, thickness=-4.0, material="mirror")
    lens.surfaces.add(index=3, thickness=-3.0, material=IdealMaterial(1.5))
    lens.surfaces.add(index=4, thickness=-10.0)
    lens.surfaces.add(index=5)
    lens.set_aperture("EPD", 2.0)
    lens.set_field_type("angle")
    lens.add_field(0.0)
    lens.add_wavelength(0.55, is_primary=True)

    problem = OptimizationProblem()
    made = problem.add_physical_constraints(
        lens, min_center_glass=1.0, min_center_air=0.5
    )
    by_surface = {c.input_data["surface_number"]: c for c in made}
    assert by_surface[1].min_val == 0.5
    assert by_surface[2].max_val == -0.5 and by_surface[2].min_val is None
    assert by_surface[3].max_val == -1.0
    assert_allclose(problem.constraint_vector()[1], 0.5 - 4.0)


def test_dls_enforces_the_limit(set_test_backend):
    lens = CookeTriplet()
    problem = OptimizationProblem()
    problem.add_default_merit(lens, fields=[0, 1], wavelengths=[1])
    problem.add_variable(lens, "thickness", surface_number=2)
    problem.add_variable(lens, "thickness", surface_number=4)
    problem.add_variable(lens, "radius", surface_number=1)
    gaps = [
        float(np.ravel(be.to_numpy(lens.surfaces.get_thickness(s)))[0]) for s in (2, 4)
    ]
    limit = min(gaps) + 0.05  # infeasible at the start: DLS must restore it
    problem.add_physical_constraints(lens, min_center_air=limit, tol=1e-7)
    result = minimize(problem, "dls", maxiter=120)
    assert result.max_constraint_violation <= 1e-7
    assert result.success


def test_penalty_works_with_scipy(set_test_backend):
    lens = CookeTriplet()
    problem = OptimizationProblem()
    problem.add_default_merit(lens, fields=[0], wavelengths=[1])
    problem.add_variable(lens, "thickness", surface_number=2)
    problem.add_physical_constraints(lens, min_center_air=0.1, mode="penalty")
    LeastSquares(problem).optimize(maxiter=6, method_choice="trf")


def test_constraint_mode_with_scipy_raises(set_test_backend):
    lens = CookeTriplet()
    problem = OptimizationProblem()
    problem.add_default_merit(lens, fields=[0], wavelengths=[1])
    problem.add_variable(lens, "thickness", surface_number=2)
    problem.add_physical_constraints(lens, min_center_air=0.1)
    with pytest.raises(ConfigurationError, match="hard constraints"):
        LeastSquares(problem).optimize(maxiter=6, method_choice="trf")
