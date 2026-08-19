from __future__ import annotations

import pytest

from optiland.optimization import LeastSquares, OptimizerGeneric
from optiland.samples.simple import Edmund_49_847
from optiland.tolerancing.compensator import CompensatorOptimizer


def test_initialization_default():
    optimizer = CompensatorOptimizer()
    assert optimizer.method == "generic"
    assert optimizer.tol == 1e-5
    assert isinstance(optimizer._optimizer_map["generic"], type(OptimizerGeneric))
    assert isinstance(optimizer._optimizer_map["least_squares"], type(LeastSquares))


def test_initialization_custom():
    optimizer = CompensatorOptimizer(method="least_squares", tol=1e-4)
    assert optimizer.method == "least_squares"
    assert optimizer.tol == 1e-4


def test_has_variables():
    optimizer = CompensatorOptimizer()
    optimizer.variables = []
    assert not optimizer.has_variables
    optimizer.variables = [1, 2, 3]
    assert optimizer.has_variables


def test_run_optimizer_generic():
    optic = Edmund_49_847()
    optimizer = CompensatorOptimizer(method="generic")
    optimizer.add_variable(optic, "radius", surface_number=1)
    optimizer.add_operand(
        operand_type="f2",
        target=25,
        weight=1,
        input_data={"optic": optic},
    )
    result = optimizer.run()
    assert result is not None


def test_run_optimizer_least_squares():
    optic = Edmund_49_847()
    optimizer = CompensatorOptimizer(method="least_squares")
    optimizer.add_variable(optic, "radius", surface_number=1)
    optimizer.add_operand(
        operand_type="f2",
        target=2525,
        weight=1,
        input_data={"optic": optic},
    )
    result = optimizer.run()
    assert result is not None


def test_invalid_method(set_test_backend):
    with pytest.raises(ValueError):
        optimizer = CompensatorOptimizer(method="invalid_method")
        optimizer.run()


def test_empty_variables_run():
    optimizer = CompensatorOptimizer(method="generic")
    optimizer.variables = []
    with pytest.raises(ValueError):
        optimizer.run()


def test_tolerance_setting():
    optimizer = CompensatorOptimizer(tol=1e-6)
    assert optimizer.tol == 1e-6


def test_optimizer_map_content():
    optimizer = CompensatorOptimizer()
    assert "generic" in optimizer._optimizer_map
    assert "least_squares" in optimizer._optimizer_map
    assert isinstance(optimizer._optimizer_map["generic"], type(OptimizerGeneric))
    assert isinstance(optimizer._optimizer_map["least_squares"], type(LeastSquares))
