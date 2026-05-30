from __future__ import annotations

import pytest

from optiland.samples.simple import Edmund_49_847
from optiland.tolerancing.compensator import CompensatorOptimizer
from optiland.tolerancing.core import Tolerancing
from optiland.tolerancing.perturbation import Perturbation, ScalarSampler
from tests.utils import assert_allclose


@pytest.fixture
def setup_tolerancing():
    optic = Edmund_49_847()
    tolerancing = Tolerancing(optic)
    return tolerancing, optic


def test_init(setup_tolerancing, set_test_backend):
    tolerancing, optic = setup_tolerancing
    assert tolerancing.optic == optic
    assert tolerancing.method == "generic"
    assert_allclose(tolerancing.tol, 1e-5)
    assert tolerancing.operands == []
    assert tolerancing.perturbations == []
    assert isinstance(tolerancing.compensator, CompensatorOptimizer)


def test_add_operand(setup_tolerancing, set_test_backend):
    tolerancing, optic = setup_tolerancing
    operand_type = "f2"
    input_data = {"optic": optic}
    target = 20.0
    weight = 2.0

    # Ensure original optic data is not modified before adding operand
    original_f2 = optic.paraxial.f2()

    tolerancing.add_operand(operand_type, input_data, target, weight)
    assert len(tolerancing.operands) == 1
    operand = tolerancing.operands[0]
    assert operand.operand_type == operand_type
    assert operand.input_data == input_data
    assert_allclose(operand.target, target)
    assert_allclose(operand.weight, weight)

    # Verify original optic data remains unchanged
    assert_allclose(optic.paraxial.f2(), original_f2)


def test_add_operand_no_target(setup_tolerancing, set_test_backend):
    tolerancing, optic = setup_tolerancing
    operand_type = "f2"
    input_data = {"optic": optic}
    weight = 2.0
    target = optic.paraxial.f2()  # Calculate target based on current state

    # Ensure original optic data is not modified before adding operand
    original_f2 = optic.paraxial.f2()

    tolerancing.add_operand(
        operand_type=operand_type,
        input_data=input_data,
        weight=weight,
    )
    assert len(tolerancing.operands) == 1
    operand = tolerancing.operands[0]
    assert operand.operand_type == operand_type
    assert operand.input_data == input_data
    assert_allclose(operand.target, target)
    assert_allclose(operand.weight, weight)

    # Verify original optic data remains unchanged
    assert_allclose(optic.paraxial.f2(), original_f2)


def test_add_perturbation(setup_tolerancing, set_test_backend):
    tolerancing, optic = setup_tolerancing
    variable_type = "radius"
    perturbation = ScalarSampler(value=100.0)
    surface_number = 1

    # Ensure original optic data is not modified before adding perturbation
    original_radius = optic.surfaces[surface_number].geometry.radius

    tolerancing.add_perturbation(
        variable_type, perturbation, surface_number=surface_number
    )
    assert len(tolerancing.perturbations) == 1
    added_perturbation = tolerancing.perturbations[0]
    assert isinstance(added_perturbation, Perturbation)
    assert added_perturbation.optic == optic
    assert added_perturbation.type == variable_type
    # Check that adding the perturbation definition doesn't apply it yet
    assert_allclose(optic.surfaces[surface_number].geometry.radius, original_radius)


def test_add_compensator(setup_tolerancing, set_test_backend):
    tolerancing, optic = setup_tolerancing
    variable_type = "thickness"

    tolerancing.add_compensator(variable_type, surface_number=2)
    assert len(tolerancing.compensator.variables) == 1
    compensator_variable = tolerancing.compensator.variables[0]
    assert compensator_variable.optic == optic
    assert compensator_variable.type == variable_type


def test_apply_compensators(setup_tolerancing, set_test_backend):
    tolerancing, optic = setup_tolerancing
    tolerancing.add_compensator("radius", surface_number=1)
    tolerancing.add_operand(operand_type="f2", input_data={"optic": optic})

    result = tolerancing.apply_compensators()
    first_key = list(result.keys())[0]
    assert first_key == "C0: Radius of Curvature, Surface 1"


def test_evaluate(setup_tolerancing, set_test_backend):
    tolerancing, optic = setup_tolerancing
    tolerancing.add_operand(operand_type="f1", input_data={"optic": optic})
    tolerancing.add_operand(operand_type="f2", input_data={"optic": optic})

    result = tolerancing.evaluate()
    assert_allclose(result, [optic.paraxial.f1(), optic.paraxial.f2()])


def test_reset(setup_tolerancing, set_test_backend):
    tolerancing, optic = setup_tolerancing
    tolerancing.add_perturbation("radius", ScalarSampler(value=100.0), surface_number=1)
    tolerancing.perturbations[0].apply()
    assert_allclose(tolerancing.perturbations[0].value, 100.0)
    tolerancing.reset()
    assert_allclose(tolerancing.perturbations[0].value, 19.93)  # original value
