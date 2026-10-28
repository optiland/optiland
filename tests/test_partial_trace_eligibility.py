"""Eligibility rules of ``partial_trace_start`` (opt v2 PR3.1).

One group of tests per rule. The base case is the Cooke triplet benchmark:
EPD aperture, angle field, paraxial aiming, stop at surface 4, so a radius on
surface 5 resumes at 5. Each test breaks one rule and expects a full trace
(None).

Kramer Harrison, 2026
"""

from __future__ import annotations

import pytest

import optiland.backend as be
from benchmarks.optimization.problems import PROBLEMS
from optiland.optimization.variable.variable import Variable
from optiland.raytrace.partial_trace import partial_trace_start

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture
def cooke(set_test_backend):
    optic, _ = PROBLEMS["cooke"]()
    assert optic.surfaces.stop_index == 4
    return optic


def _start(optic, type_name: str, s: int, **kwargs) -> int | None:
    return optic.ray_tracer.partial_trace_start(
        Variable(optic, type_name, surface_number=s, **kwargs)
    )


def test_base_case_is_eligible(cooke):
    assert _start(cooke, "radius", 5) == 5
    assert _start(cooke, "thickness", 6) == 6


def test_accepts_variable_behavior_and_module_function(cooke):
    variable = Variable(cooke, "radius", surface_number=5)
    assert partial_trace_start(cooke, variable.variable) == 5
    assert partial_trace_start(cooke, variable) == 5


def test_unknown_object_falls_back_to_full_trace(cooke):
    assert partial_trace_start(cooke, object()) is None


# Rule 1: one mutated surface, no pickup or solve writing upstream.


@pytest.mark.parametrize(
    ("type_name", "kwargs"),
    [("tilt", {"axis": "x"}), ("decenter", {"axis": "y"})],
)
def test_rule1_coordinate_variables_are_ineligible(cooke, type_name, kwargs):
    assert _start(cooke, type_name, 5, **kwargs) is None


def test_rule1_variable_of_another_optic_is_ineligible(cooke):
    other, _ = PROBLEMS["cooke"]()
    variable = Variable(other, "radius", surface_number=5)
    assert partial_trace_start(cooke, variable) is None


def test_rule1_pickup_from_s_to_upstream_is_ineligible(cooke):
    cooke.pickups.add(5, "radius", 2, scale=-1)
    assert _start(cooke, "radius", 5) is None


def test_rule1_pickup_untouched_by_variable_is_eligible(cooke):
    # Upstream source and target: not triggered by surface 5.
    cooke.pickups.add(1, "radius", 2, scale=-1)
    # Downstream target: written at or after s.
    cooke.pickups.add(5, "radius", 6, scale=-1)
    assert _start(cooke, "radius", 5) == 5


def test_rule1_upstream_solve_is_ineligible(cooke):
    cooke.solves.add("marginal_ray_angle", 2, angle=-0.1)
    assert _start(cooke, "radius", 5) is None


def test_rule1_downstream_solve_is_eligible(cooke):
    cooke.solves.add("quick_focus")
    assert _start(cooke, "radius", 5) == 5


# Rule 2: s after the stop, aperture set upstream of the stop.


@pytest.mark.parametrize("s", [1, 3, 4])
def test_rule2_surface_at_or_before_stop_is_ineligible(cooke, s):
    assert _start(cooke, "radius", s) is None


def test_rule2_image_space_fno_is_ineligible(cooke):
    cooke.set_aperture("imageFNO", 5.0)
    assert _start(cooke, "radius", 5) is None


@pytest.mark.parametrize(
    ("aperture_type", "value"), [("float_by_stop_size", 5.0), ("objectNA", 0.1)]
)
def test_rule2_upstream_aperture_types_are_eligible(cooke, aperture_type, value):
    if aperture_type == "objectNA":
        cooke.surfaces[0].geometry.cs.z = be.array(-100.0)
        cooke.surfaces[0].thickness = 100.0
        cooke.set_field_type("object_height")
    cooke.set_aperture(aperture_type, value)
    assert _start(cooke, "radius", 5) == 5


# Rule 3: field type defined in object space.


@pytest.mark.parametrize("field_type", ["paraxial_image_height", "real_image_height"])
def test_rule3_image_side_field_is_ineligible(cooke, field_type):
    cooke.fields.set_type(field_type)
    assert _start(cooke, "radius", 5) is None


# Rule 4: ray aiming upstream of the stop only.


def test_rule4_iterative_aiming_is_eligible(cooke):
    cooke.ray_tracer.set_aiming("iterative")
    assert _start(cooke, "radius", 5) == 5


@pytest.mark.parametrize(
    ("mode", "kwargs"),
    [("robust", {}), ("iterative", {"cache": True}), ("paraxial", {"cache": True})],
)
def test_rule4_other_aiming_is_ineligible(cooke, mode, kwargs):
    cooke.ray_tracer.set_aiming(mode, **kwargs)
    assert _start(cooke, "radius", 5) is None


def test_rule4_built_aimer_must_match_config(cooke):
    from optiland.rays.ray_aiming.robust import RobustRayAimer

    tracer = cooke.ray_tracer
    tracer.trace_generic(0.0, 0.0, 0.0, 0.0, cooke.primary_wavelength)  # build
    assert _start(cooke, "radius", 5) == 5
    tracer.ray_generator.aimer = RobustRayAimer(cooke)
    assert _start(cooke, "radius", 5) is None


# Rule 5: nothing upstream moves or changes shape.


def test_rule5_upstream_cs_referencing_downstream_is_ineligible(cooke):
    cooke.surfaces[2].geometry.cs.reference_cs = cooke.surfaces[6].geometry.cs
    assert _start(cooke, "radius", 5) is None


def test_rule5_shared_geometry_is_ineligible(cooke):
    cooke.surfaces[2].geometry = cooke.surfaces[5].geometry
    assert _start(cooke, "radius", 5) is None


def test_rule5_thickness_with_inconsistent_upstream_positions(cooke):
    # set_thickness would rebuild surface 2's position from the thicknesses
    # and move it; a radius variable rebuilds nothing.
    surface = cooke.surfaces[2]
    surface.geometry.cs.z = surface.geometry.cs.z + 1e-9
    assert _start(cooke, "thickness", 5) is None
    assert _start(cooke, "radius", 5) == 5
