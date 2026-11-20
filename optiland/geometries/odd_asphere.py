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

    _r2_index = 1  # coefficients[0] is the r^1 term

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
        return self._sag_and_terms(x, y)[0]

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
        dfdx, dfdy = self._slopes_from_terms(x, y, self._odd_terms(x, y))

        mag = be.sqrt(dfdx**2 + dfdy**2 + 1)

        nx = dfdx / mag
        ny = dfdy / mag
        nz = -1 / mag

        return nx, ny, nz

    def _odd_terms(self, x, y):
        """Return r**2, r, the mask r > 0 and the root of the conic.

        The root of the conic is sqrt(1 - (1 + k) r**2 / R**2). At the vertex,
        r is 0 with a finite derivative: sqrt has an infinite derivative at
        0, so the code takes the root of 1 there and multiplies it by 0.
        Elsewhere, r has the bits of sqrt(r**2). A NaN stays NaN.
        """
        r2 = be.array(x**2 + y**2)
        positive = r2 > 0
        r = be.sqrt(r2 + ~positive) * positive
        root = be.sqrt(1 - (1 + self.k) * r2 / self.radius**2)
        return r2, r, positive, root

    def _sag_and_terms(self, x, y):
        """Return the sag at (x, y), and the terms that its slopes share.

        See :meth:`NewtonRaphsonGeometry._sag_and_terms`. The terms are r**2,
        r, the mask r > 0 and the conic root.
        """
        terms = self._odd_terms(x, y)
        r2, r, _, root = terms
        z = r2 / (self.radius * (1 + root))
        # Use the length: the coefficients can also be an array or a tensor.
        n = len(self.coefficients)
        if n > 0:
            # Horner's method: r * (C1 + r * (C2 + ... + r * Cn)).
            polynomial = self.coefficients[n - 1]
            for i in range(n - 2, -1, -1):
                polynomial = self.coefficients[i] + r * polynomial
            z = z + r * polynomial
        return z, terms

    def _slopes_from_terms(self, x, y, terms):
        """Return d(sag)/dx and d(sag)/dy from the terms of ``_sag_and_terms``.

        See :meth:`NewtonRaphsonGeometry._slopes_from_terms`. The polynomial
        gives d(sag)/dx = x g and d(sag)/dy = y g, with
        g = (dz/dr) / r = C1 / r + 2 C2 + 3 C3 r + ... + n Cn r^(n-2).
        """
        r2, r, positive, root = terms
        denom = self.radius * root
        dfdx = x / denom
        dfdy = y / denom
        n = len(self.coefficients)
        if n > 0:
            # 1 / r, and 0 at the vertex: there the slope of the C1 term is 0,
            # as the former code set its value 0 / 0 to 0. The divisor is 1 at
            # the vertex, so the gradient stays finite.
            inverse_r = positive / (r + ~positive)
            g = self.coefficients[0] * inverse_r
            if n > 1:
                # Horner's method: 2 C2 + r * (3 C3 + ... + r * n Cn).
                rest = n * self.coefficients[n - 1]
                for degree in range(n - 1, 1, -1):
                    rest = degree * self.coefficients[degree - 1] + r * rest
                g = g + rest
            dfdx = dfdx + x * g
            dfdy = dfdy + y * g
        return dfdx, dfdy

    def flip(self):
        """Flip the geometry.

        Changes the sign of the radius of curvature and the signs of coefficients.
        The conic constant remains unchanged.
        """
        self.radius = -self.radius
        self.coefficients = [-c for c in self.coefficients]
