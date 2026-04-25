"""
Provides a phase profile defined on a grid.
"""

from __future__ import annotations

from optiland import backend as be
from optiland.phase.base import BasePhaseProfile
from optiland.phase.interpolators import GridInterpolator


class GridPhaseProfile(BasePhaseProfile):
    """A phase profile defined by a grid of phase values.

    This class interpolates the phase and its gradient at arbitrary points
    using the active backend.

    Args:
        x_coords (be.Array): The x-coordinates of the grid points.
        y_coords (be.Array): The y-coordinates of the grid points.
        phase_grid (be.Array): The phase values at the grid points.
            The shape must be (len(y_coords), len(x_coords)).
    """

    phase_type = "grid"

    def __init__(self, x_coords: be.Array, y_coords: be.Array, phase_grid: be.Array):
        super().__init__()
        self.backend = be.get_backend()
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.phase_grid = phase_grid

        self._interp = GridInterpolator(x_coords, y_coords, phase_grid)

    def get_phase(
        self, x: be.Array, y: be.Array, wavelength: be.Array = None
    ) -> be.Array:
        return self._interp.height(x, y)

    def get_gradient(
        self, x: be.Array, y: be.Array, wavelength: be.Array = None
    ) -> tuple[be.Array, be.Array, be.Array]:
        d_phi_dx, d_phi_dy = self._interp.gradient(x, y)
        return d_phi_dx, d_phi_dy, be.zeros_like(x)

    def get_paraxial_gradient(
        self, y: be.Array, wavelength: be.Array = None
    ) -> be.Array:
        x0 = be.zeros_like(y)
        _, d_phi_dy = self._interp.gradient(x0, y)
        return d_phi_dy

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["x_coords"] = be.to_numpy(self.x_coords).tolist()
        data["y_coords"] = be.to_numpy(self.y_coords).tolist()
        data["phase_grid"] = be.to_numpy(self.phase_grid).tolist()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> GridPhaseProfile:
        return cls(
            x_coords=be.array(data["x_coords"]),
            y_coords=be.array(data["y_coords"]),
            phase_grid=be.array(data["phase_grid"]),
        )
