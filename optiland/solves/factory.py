"""Solve Factory Module

Factory for creating solve objects.

Kramer Harrison, 2025
"""

from __future__ import annotations

from optiland.solves.curvature import (
    ChiefRayAngleCurvatureSolve,
    MarginalRayAngleCurvatureSolve,
)
from optiland.solves.quick_focus import QuickFocusSolve
from optiland.solves.thickness import (
    ChiefRayHeightThicknessSolve,
    MarginalRayHeightThicknessSolve,
)


class SolveFactory:
    """Factory class for creating solves.

    This factory uses a predefined map to find and instantiate
    the appropriate solve class based on the provided `solve_type`.
    """

    _solve_map = {
        "marginal_ray_height": MarginalRayHeightThicknessSolve,
        "marginal_ray_height_thickness": MarginalRayHeightThicknessSolve,
        "quick_focus": QuickFocusSolve,
        "chief_ray_height": ChiefRayHeightThicknessSolve,
        "chief_ray_height_thickness": ChiefRayHeightThicknessSolve,
        "marginal_ray_angle": MarginalRayAngleCurvatureSolve,
        "marginal_ray_angle_curvature": MarginalRayAngleCurvatureSolve,
        "chief_ray_angle": ChiefRayAngleCurvatureSolve,
        "chief_ray_angle_curvature": ChiefRayAngleCurvatureSolve,
    }

    @staticmethod
    def create_solve(optic, solve_type: str, surface_idx, *args, **kwargs):
        """Creates a solve instance based on the given solve type.

        Args:
            optic (Optic): The optic object to which the solve will be applied.
            solve_type (str): The type of solve to create (e.g.,
                'marginal_ray_height', 'quick_focus', 'chief_ray_height').
            surface_idx (int): The index of the surface relevant to the solve.
                May not be used by all solve types (e.g., 'quick_focus').
            *args: Variable length argument list. For height-based solves,
                   the first argument (args[0]) is expected to be the height.
            **kwargs: Arbitrary keyword arguments. (Currently unused in favor
                      of positional arguments for height).

        Returns:
            An instance of the solve class corresponding to the given solve type.

        Raises:
            ValueError: If the solve type is invalid or if required arguments
                for a specific solve type are missing or invalid.
        """
        solve_class = SolveFactory._solve_map.get(solve_type)
        if solve_class is None:
            raise ValueError(f"Invalid solve type: {solve_type}")
        return solve_class(optic, surface_idx, *args, **kwargs)
