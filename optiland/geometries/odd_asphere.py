"""Odd Asphere Geometry

The Odd Asphere geometry represents a surface defined by an odd asphere
polynomial superimposed on a base conic surface. The surface is defined as:

z = r^2 / (R * (1 + sqrt(1 - (1 + k) * r^2 / R^2))) + sum(Ci * r^i)

where
- r^2 = x^2 + y^2
- R is the radius of curvature
- k is the conic constant
- Ci are the aspheric coefficients

Kramer Harrison, 2025
"""

from __future__ import annotations

import warnings

import optiland.backend as be
from optiland.geometries.even_asphere import EvenAsphere


class OddAsphere(EvenAsphere):
    """Represents an odd asphere geometry defined as:

    z = r^2 / (R * (1 + sqrt(1 - (1 + k) * r^2 / R^2))) + sum(Ci * r^i)

    where
    - r^2 = x^2 + y^2
    - R is the radius of curvature
    - k is the conic constant
    - Ci are the aspheric coefficients
    - i is an integer from 1 to n

    Args:
        coordinate_system (CoordinateSystem): The coordinate system of the geometry.
        radius (float): The radius of curvature of the base sphere.
        conic (float, optional): The conic constant of the base sphere.
            Defaults to 0.0.
        tol (float, optional): Tolerance for Newton-Raphson iteration.
            Defaults to 1e-10.
        max_iter (int, optional): Maximum iterations for Newton-Raphson.
            Defaults to 100.
        coefficients (list[float], optional): A list of odd aspheric
            coefficients C_i, where the term is C_i * r^i.
            The list index corresponds to i-1 (e.g., coefficients[0] is C_1 for r^1).
            Defaults to an empty list (no aspheric contribution).

    Attributes:
        c (list[float]): List of aspheric coefficients.

    """

    def __init__(
        self,
        coordinate_system,
        radius,
        conic=0.0,
        tol=1e-10,
        max_iter=100,
        coefficients=None,
    ):
        if coefficients is None:
            coefficients = []
        super().__init__(coordinate_system, radius, conic, tol, max_iter, coefficients)
        self.order = 1  # used for optimization scaling

    def __str__(self):
        return "Odd Asphere"

    def scale(self, scale_factor: float):
        """Scale the geometry parameters.

        Args:
            scale_factor (float): The factor by which to scale the geometry.
        """
        # Skip EvenAsphere.scale as it incorrectly scales coefficients for OddAsphere
        # Call the next class in MRO after EvenAsphere (NewtonRaphson -> Standard)
        super(EvenAsphere, self).scale(scale_factor)
        for i in range(len(self.coefficients)):
            # C_i' = C_i * s^(1 - (i+1))
            self.coefficients[i] *= scale_factor ** (1 - (i + 1))

    def sag(self, x=0, y=0):
        """Calculates the sag of the asphere at the given coordinates.

        Args:
            x (float or be.ndarray, optional): The x-coordinate(s). Defaults to 0.
            y (float or be.ndarray, optional): The y-coordinate(s). Defaults to 0.

        Returns:
            be.ndarray or float: The sag value(s) at the given coordinates.

        """
        r2 = be.array(x**2 + y**2)
        r = be.sqrt(r2)
        z = r2 / (self.radius * (1 + be.sqrt(1 - (1 + self.k) * r2 / self.radius**2)))
        for i, Ci in enumerate(self.coefficients):
            z = z + Ci * r ** (i + 1)

        return z

    def _surface_normal(self, x, y):
        """Calculates the surface normal of the asphere at the given x and y
        position.

        Args:
            x (be.ndarray): The x-coordinate(s) at which to calculate the normal.
            y (be.ndarray): The y-coordinate(s) at which to calculate the normal.

        Returns:
            tuple[be.ndarray, be.ndarray, be.ndarray]: The surface normal
            components (nx, ny, nz).

        """
        r2 = x**2 + y**2
        r = be.sqrt(r2)

        denom = self.radius * be.sqrt(1 - (1 + self.k) * r2 / self.radius**2)
        dfdx = x / denom
        dfdy = y / denom

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, Ci in enumerate(self.coefficients):
                x_term = (i + 1) * x * Ci * r ** (i - 1)
                y_term = (i + 1) * y * Ci * r ** (i - 1)

                x_term[~be.isfinite(x_term)] = 0
                y_term[~be.isfinite(y_term)] = 0

                dfdx = dfdx + x_term
                dfdy = dfdy + y_term

        mag = be.sqrt(dfdx**2 + dfdy**2 + 1)

        nx = dfdx / mag
        ny = dfdy / mag
        nz = -1 / mag

        return nx, ny, nz

    def flip(self):
        """Flip the geometry.

        Changes the sign of the radius of curvature and the signs of coefficients.
        The conic constant remains unchanged.
        """
        self.radius = -self.radius
        self.coefficients = [-c for c in self.coefficients]
