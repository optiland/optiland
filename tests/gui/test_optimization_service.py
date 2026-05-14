"""Tests for OptimizationService."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from optiland_gui.services.optimization_service import OptimizationService


@pytest.fixture()
def mock_connector(minimal_optic):
    conn = MagicMock()
    conn._optic = minimal_optic
    conn._capture_optic_state.return_value = {}
    conn._undo_redo_manager = MagicMock()
    return conn


@pytest.fixture()
def service(mock_connector):
    return OptimizationService(mock_connector)


class TestOptimizationService:
    def test_add_variable(self, service):
        service.add_variable({"surface_number": 1, "type": "radius"})
        assert len(service.get_variables()) == 1
        assert service.get_variables()[0]["type"] == "radius"

    def test_remove_variable(self, service):
        service.add_variable({"surface_number": 1, "type": "radius"})
        service.add_variable({"surface_number": 2, "type": "thickness"})
        service.remove_variable(0)
        variables = service.get_variables()
        assert len(variables) == 1
        assert variables[0]["type"] == "thickness"

    def test_add_operand(self, service):
        service.add_operand(
            {
                "type": "total_track",
                "category": "Paraxial",
                "target": 50.0,
                "weight": 1.0,
                "input_data_str": "{}",
            }
        )
        assert len(service.get_operands()) == 1
        assert service.get_operands()[0]["type"] == "total_track"

    def test_build_problem_counts(self, service, minimal_optic):
        service.add_variable({"surface_number": 1, "type": "radius"})
        service.add_operand(
            {
                "type": "total_track",
                "category": "Paraxial",
                "target": 50.0,
                "weight": 1.0,
                "input_data_str": "{}",
            }
        )
        problem = service.build_problem(minimal_optic)
        assert len(problem.variables) == 1
        assert len(problem.operands) == 1

    def test_least_squares_run_does_not_raise(self, service, minimal_optic, qapp):
        """Synchronous LeastSquares run should complete without error."""
        from optiland.optimization.optimizer.scipy import LeastSquares

        service.add_variable({"surface_number": 1, "type": "radius"})
        service.add_operand(
            {
                "type": "total_track",
                "category": "Paraxial",
                "target": 50.0,
                "weight": 1.0,
                "input_data_str": "{}",
            }
        )
        problem = service.build_problem(minimal_optic)
        optimizer = LeastSquares(problem)
        # Run synchronously with a very low iteration limit
        optimizer.optimize(maxiter=2)


class TestValidateOperandInputData:
    def test_validate_missing_surface_number(self, service):
        err = service.validate_operand_input_data("clearance", "{}")
        assert err is not None
        assert "surface_number" in err

    def test_validate_valid_input_data(self, service):
        err = service.validate_operand_input_data("clearance", '{"surface_number": 1}')
        assert err is None

    def test_validate_invalid_json(self, service):
        err = service.validate_operand_input_data("clearance", "not valid json {")
        assert err is not None
        assert "Invalid JSON" in err

    def test_validate_seidel_requires_two_keys(self, service):
        err = service.validate_operand_input_data(
            "seidel", '{"surface_number": 1}'
        )
        assert err is not None
        assert "seidel_number" in err

    def test_validate_seidel_valid(self, service):
        err = service.validate_operand_input_data(
            "seidel", '{"seidel_number": 0, "surface_number": 1}'
        )
        assert err is None

    def test_validate_paraxial_operand_no_required_keys(self, service):
        """Operands like total_track need no extra keys."""
        err = service.validate_operand_input_data("total_track", "{}")
        assert err is None

    def test_build_problem_skipped_operand_shows_toast(
        self, service, minimal_optic, mock_connector
    ):
        """When add_operand raises during build_problem, a toast must be shown."""
        from unittest.mock import patch

        service.add_variable({"surface_number": 1, "type": "radius"})
        service.add_operand(
            {
                "type": "total_track",
                "category": "Paraxial",
                "target": 0.0,
                "weight": 1.0,
                "input_data_str": "{}",
            }
        )
        # Patch OptimizationProblem.add_operand to force an exception
        with patch(
            "optiland.optimization.OptimizationProblem.add_operand",
            side_effect=TypeError("forced error"),
        ):
            # Ensure the connector has a toast_manager mock
            mock_connector.toast_manager = MagicMock()
            service.build_problem(minimal_optic)

        mock_connector.toast_manager.notify.assert_called()
        args = mock_connector.toast_manager.notify.call_args
        assert args[0][1] == "warning"


class TestOptimizerGroupsAndBounds:
    def test_get_optimizer_groups_has_local_and_global(self, service):
        groups = service.get_optimizer_groups()
        assert "Local" in groups
        assert "Global" in groups

    def test_local_group_is_non_empty(self, service):
        groups = service.get_optimizer_groups()
        assert len(groups["Local"]) > 0

    def test_global_group_contains_dual_annealing(self, service):
        from optiland.optimization.optimizer.scipy import DualAnnealing

        groups = service.get_optimizer_groups()
        cls_list = [cls for _, cls, _ in groups["Global"]]
        assert DualAnnealing in cls_list

    def test_validate_bounds_dual_annealing_no_bounds(self, service):
        from optiland.optimization.optimizer.scipy import DualAnnealing

        service.add_variable(
            {"surface_number": 1, "type": "radius", "min_val": None, "max_val": None}
        )
        err = service.validate_bounds_for_optimizer(DualAnnealing)
        assert err is not None
        assert "bounds" in err.lower()

    def test_validate_bounds_dual_annealing_with_bounds(self, service):
        from optiland.optimization.optimizer.scipy import DualAnnealing

        service.add_variable(
            {
                "surface_number": 1,
                "type": "radius",
                "min_val": 10.0,
                "max_val": 200.0,
            }
        )
        err = service.validate_bounds_for_optimizer(DualAnnealing)
        assert err is None

    def test_validate_bounds_basin_hopping_rejects_bounds(self, service):
        from optiland.optimization.optimizer.scipy import BasinHopping

        service.add_variable(
            {
                "surface_number": 1,
                "type": "radius",
                "min_val": 10.0,
                "max_val": 200.0,
            }
        )
        err = service.validate_bounds_for_optimizer(BasinHopping)
        assert err is not None

    def test_validate_bounds_least_squares_no_bounds(self, service):
        from optiland.optimization.optimizer.scipy import LeastSquares

        service.add_variable(
            {"surface_number": 1, "type": "radius", "min_val": None, "max_val": None}
        )
        err = service.validate_bounds_for_optimizer(LeastSquares)
        assert err is None
