"""Grid Sag Geometry

This module defines a geometry based on a grid of sag values, which are
interpolated to find the surface shape.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import Any

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.base import BaseGeometry


class GridSagGeometry(BaseGeometry):
    """Represents a geometry defined by a grid of sag values.

    The sag of the surface at any (x, y) point is determined by bilinearly
    interpolating the surrounding points in the sag grid.

    Args:
        coordinate_system (CoordinateSystem): The coordinate system of the geometry.
        x_coordinates (list[float]): 1D list of x-coordinates for the grid.
        y_coordinates (list[float]): 1D list of y-coordinates for the grid.
        sag_values (ArrayLike): 2D array-like (list of lists, numpy array, or
            torch tensor) of sag values, with shape (len(y_coordinates),
            len(x_coordinates)).
        tol (float, optional): Tolerance for the iterative ray intersection solver.
            Defaults to 1e-6.
        max_iter (int, optional): Maximum number of iterations for the solver.
            Defaults to 100.
    """

    def __init__(
        self,
        coordinate_system: CoordinateSystem,
        x_coordinates: list[float],
        y_coordinates: list[float],
        sag_values: Any,
        tol: float = 1e-6,
        max_iter: int = 100,
    ):
        super().__init__(coordinate_system)
        self.x_grid = be.asarray(x_coordinates)
        self.y_grid = be.asarray(y_coordinates)
        self.sag_grid = be.asarray(sag_values)
        self.tol = tol
        self.max_iter = max_iter
        self.is_symmetric = False

        if self.sag_grid.shape != (len(self.y_grid), len(self.x_grid)):
            raise ValueError(
                f"Shape of sag_values {self.sag_grid.shape} must match "
                f"(len(y_coordinates), len(x_coordinates)) = "
                f"({len(self.y_grid)}, {len(self.x_grid)})."
            )

    def _interpolate(self, x, y):
        """Performs bilinear interpolation and calculates derivatives."""
        # Find indices for lower-left corner of the interpolation cell
        # be.searchsorted with side="right" returns the index after elements equal to
        # x/y, so we subtract 1 to get the lower bound index for interpolation.
        i = be.searchsorted(self.x_grid, x, side="right") - 1
        j = be.searchsorted(self.y_grid, y, side="right") - 1

        # Handle out-of-bounds cases
        nan_mask = (
            (x < self.x_grid[0])
            | (x > self.x_grid[-1])
            | (y < self.y_grid[0])
            | (y > self.y_grid[-1])
        )
        i = be.where(i < 0, 0, i)
        j = be.where(j < 0, 0, j)
        i = be.where(i >= len(self.x_grid) - 1, len(self.x_grid) - 2, i)
        j = be.where(j >= len(self.y_grid) - 1, len(self.y_grid) - 2, j)

        # Get corner coordinates and sag values
        x1, x2 = self.x_grid[i], self.x_grid[i + 1]
        y1, y2 = self.y_grid[j], self.y_grid[j + 1]
        z11, z12 = self.sag_grid[j, i], self.sag_grid[j, i + 1]
        z21, z22 = self.sag_grid[j + 1, i], self.sag_grid[j + 1, i + 1]

        # Calculate interpolation weights
        tx = (x - x1) / (x2 - x1)
        ty = (y - y1) / (y2 - y1)

        # Interpolate
        z_y1 = z11 * (1 - tx) + z12 * tx
        z_y2 = z21 * (1 - tx) + z22 * tx
        sag = z_y1 * (1 - ty) + z_y2 * ty

        # Calculate derivatives
        ds_dx = ((z12 - z11) * (1 - ty) + (z22 - z21) * ty) / (x2 - x1)
        ds_dy = ((z21 - z11) * (1 - tx) + (z22 - z12) * tx) / (y2 - y1)

        sag = be.where(nan_mask, be.nan, sag)
        return sag, ds_dx, ds_dy

    def sag(self, x=0, y=0):
        """Calculate the surface sag of the geometry."""
        sag_val, _, _ = self._interpolate(x, y)
        return sag_val

    def distance(self, rays):
        """Find the propagation distance to the geometry."""
        t = be.zeros_like(rays.x)  # Initial guess for distance
        for _ in range(self.max_iter):
            x_intersect = rays.x + t * rays.L
            y_intersect = rays.y + t * rays.M
            z_intersect = rays.z + t * rays.N

            sag_val, ds_dx, ds_dy = self._interpolate(x_intersect, y_intersect)

            # Function f(t) = sag(x(t), y(t)) - z(t)
            f = sag_val - z_intersect

            # Derivative f'(t) = (ds/dx * dx/dt + ds/dy * dy/dt) - dz/dt
            f_prime = ds_dx * rays.L + ds_dy * rays.M - rays.N

            # Newton-Raphson step
            dt = -f / f_prime
            t = t + dt

            if be.max(be.abs(dt)) < self.tol:
                break

        # Clip rays that miss the grid
        x_final = rays.x + t * rays.L
        y_final = rays.y + t * rays.M
        out_of_bounds = (
            (x_final < self.x_grid[0])
            | (x_final > self.x_grid[-1])
            | (y_final < self.y_grid[0])
            | (y_final > self.y_grid[-1])
        )
        return be.where(out_of_bounds, be.nan, t)

    def surface_normal(self, rays):
        """Find the surface normal of the geometry at the given ray positions."""
        _, ds_dx, ds_dy = self._interpolate(rays.x, rays.y)

        # Normal vector is (-ds/dx, -ds/dy, 1), normalized
        nx, ny, nz = -ds_dx, -ds_dy, be.ones_like(rays.x)
        mag = be.sqrt(nx**2 + ny**2 + nz**2)
        return nx / mag, ny / mag, nz / mag

    def flip(self):
        """Flip the geometry by negating the sag values."""
        self.sag_grid = -self.sag_grid

    def scale(self, scale_factor: float):
        """Scale the geometry parameters.

        Args:
            scale_factor (float): The factor by which to scale the geometry.
        """
        self.x_grid = self.x_grid * scale_factor
        self.y_grid = self.y_grid * scale_factor
        self.sag_grid = self.sag_grid * scale_factor

    def to_dict(self):
        """Convert the geometry to a dictionary."""
        geometry_dict = super().to_dict()
        geometry_dict.update(
            {
                "x_coordinates": self.x_grid.tolist(),
                "y_coordinates": self.y_grid.tolist(),
                "sag_values": self.sag_grid.tolist(),
                "tol": self.tol,
                "max_iter": self.max_iter,
            }
        )
        return geometry_dict

    @classmethod
    def from_dict(cls, data):
        """Create a geometry from a dictionary."""
        cs = CoordinateSystem.from_dict(data["cs"])
        return cls(
            cs,
            data["x_coordinates"],
            data["y_coordinates"],
            data["sag_values"],
            data.get("tol", 1e-6),
            data.get("max_iter", 100),
        )
