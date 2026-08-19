"""Chebyshev Geometry

The Chebyshev polynomial geometry represents a surface defined by a Chebyshev
polynomial in two dimensions. The surface is defined as:

z = r^2 / (R * (1 + sqrt(1 - (1 + k) * r^2 / R^2))) +
    sum(Cij * T_i(x / norm_x) * T_j(y / norm_y))

where
- r^2 = x^2 + y^2
- R is the radius of curvature
- k is the conic constant
- Cij are the Chebyshev polynomial coefficients
- T_i(x) is the Chebyshev polynomial of the first kind of degree i
- norm_x and norm_y are normalization factors for the x and y coordinates

Chebyshev polynomials are derived in Cartesian coordinates,
which - unlike many other polynomial freeform surfaces used
to describe rotationally-symmetric systems - allows for
straightforward definition of anamorphic or non-rotationally
symmetric systems and non-elliptical apertures.

Kramer Harrison, 2024
"""

from __future__ import annotations

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.newton_raphson import NewtonRaphsonGeometry


class ChebyshevPolynomialGeometry(NewtonRaphsonGeometry):
    """Represents a Chebyshev polynomial geometry defined as:

    z = r^2 / (R * (1 + sqrt(1 - (1 + k) * r^2 / R^2))) +
        sum(Cij * T_i(x / norm_x) * T_j(y / norm_y))

    where
    - r^2 = x^2 + y^2
    - R is the radius of curvature
    - k is the conic constant
    - Cij are the Chebyshev polynomial coefficients
    - T_i(x) is the Chebyshev polynomial of the first kind of degree i
    - norm_x and norm_y are normalization factors for the x and y coordinates

    The coefficients are defined in a 2D array where coefficients[i][j] is the
    coefficient for T_i(x) * T_j(y).

    Chebyshev polynomials are derived in Cartesian coordinates,
    which - unlike many other polynomial freeform surfaces used
    to describe rotationally-symmetric systems - allows for
    straightforward definition of anamorphic or non-rotationally
    symmetric systems and non-elliptical apertures.

    Args:
        coordinate_system (CoordinateSystem): The coordinate system of the geometry.
        radius (float): The radius of curvature of the base sphere.
        conic (float, optional): The conic constant of the base sphere.
            Defaults to 0.0.
        tol (float, optional): Tolerance for Newton-Raphson iteration.
            Defaults to 1e-10.
        max_iter (int, optional): Maximum iterations for Newton-Raphson.
            Defaults to 100.
        coefficients (list[list[float]] or be.ndarray, optional): A 2D array or
            list of lists representing the Chebyshev coefficients Cij.
            `coefficients[i][j]` is the coefficient for T_i(x) * T_j(y).
            Defaults to an empty list (no polynomial contribution).
        norm_x (float, optional): Normalization radius for the x-coordinate.
            If None, the radius scales automatically during paraxial updates.
            Defaults to None.
        norm_y (float, optional): Normalization radius for the y-coordinate.
            If None, the radius scales automatically during paraxial updates.
            Defaults to None.

    Attributes:
        c (be.ndarray): 2D array of Chebyshev coefficients.
        norm_x (be.ndarray): Normalization factor for x.
        norm_y (be.ndarray): Normalization factor for y.

    """

    def __init__(
        self,
        coordinate_system,
        radius,
        conic=0.0,
        tol=1e-10,
        max_iter=100,
        coefficients=None,
        norm_x=None,
        norm_y=None,
    ):
        if coefficients is None:
            coefficients = []
        super().__init__(coordinate_system, radius, conic, tol, max_iter)
        self.coefficients = be.atleast_2d(coefficients)

        self.normalization_mode = (
            "manual" if (norm_x is not None or norm_y is not None) else "auto"
        )
        self.norm_x = be.array(norm_x if norm_x is not None else 1.0)
        self.norm_y = be.array(norm_y if norm_y is not None else 1.0)

        self.is_symmetric = False

    def __str__(self):
        return "Chebyshev Polynomial"

    def scale(self, scale_factor: float):
        """Scale the geometry parameters.

        Args:
            scale_factor (float): The factor by which to scale the geometry.
        """
        super().scale(scale_factor)
        self.norm_x = self.norm_x * scale_factor
        self.norm_y = self.norm_y * scale_factor
        self.coefficients = self.coefficients * scale_factor

    def update_normalization(self, semi_aperture: float) -> None:
        if self.normalization_mode == "auto":
            self.norm_x = semi_aperture * 1.25
            self.norm_y = semi_aperture * 1.25

    def sag(self, x=0, y=0):
        """Calculates the sag of the Chebyshev polynomial surface at the given
        coordinates.

        Args:
            x (float or be.ndarray, optional): The x-coordinate(s). Defaults to 0.
            y (float or be.ndarray, optional): The y-coordinate(s). Defaults to 0.

        Returns:
            be.ndarray or float: The sag value(s) at the given coordinates.

        """
        x_norm = x / self.norm_x
        y_norm = y / self.norm_y

        self._validate_inputs(x_norm, y_norm)

        r2 = x**2 + y**2
        z = r2 / (self.radius * (1 + be.sqrt(1 - (1 + self.k) * r2 / self.radius**2)))

        non_zero_indices = be.argwhere(self.coefficients != 0)
        for i, j in non_zero_indices:
            z = z + self.coefficients[i, j] * self._chebyshev(
                i, x_norm
            ) * self._chebyshev(j, y_norm)

        return z

    def _surface_normal(self, x, y):
        """Calculates the surface normal of the Chebyshev polynomial surface at
        the given x and y position.

        Args:
            x (be.ndarray): The x-coordinate(s) at which to calculate the normal.
            y (be.ndarray): The y-coordinate(s) at which to calculate the normal.

        Returns:
            tuple[be.ndarray, be.ndarray, be.ndarray]: The surface normal
            components (nx, ny, nz).

        """
        x_norm = x / self.norm_x
        y_norm = y / self.norm_y

        self._validate_inputs(x_norm, y_norm)

        r2 = x**2 + y**2
        denom = self.radius * be.sqrt(1 - (1 + self.k) * r2 / self.radius**2)
        dzdx = x / denom
        dzdy = y / denom

        non_zero_indices = be.argwhere(self.coefficients != 0)
        for i, j in non_zero_indices:
            dzdx = dzdx + (
                self._chebyshev_derivative(i, x_norm)
                * self.coefficients[i, j]
                * self._chebyshev(j, y_norm)
            )
            dzdy = dzdy + (
                self._chebyshev_derivative(j, y_norm)
                * self.coefficients[i, j]
                * self._chebyshev(i, x_norm)
            )

        norm = be.sqrt(dzdx**2 + dzdy**2 + 1)
        nx = dzdx / norm
        ny = dzdy / norm
        nz = -1 / norm

        return nx, ny, nz

    def _chebyshev(self, n, x):
        """Calculates the Chebyshev polynomial of the first kind of degree n at
        the given x value.

        Args:
            n (int): The degree of the Chebyshev polynomial.
            x (be.ndarray or float): The coordinate value(s) (normalized).

        Returns:
            be.ndarray or float: The Chebyshev polynomial T_n(x).

        """
        return be.cos(n * be.arccos(x))

    def _chebyshev_derivative(self, n, x):
        """Calculates the derivative of the Chebyshev polynomial of the first kind
        of degree n at the given x value.

        Args:
            n (int): The degree of the Chebyshev polynomial.
            x (be.ndarray or float): The coordinate value(s) (normalized).

        Returns:
            be.ndarray or float: The derivative of the Chebyshev polynomial T_n(x)
            with respect to x, scaled by 1/norm_factor if applicable
            (handled by caller). Returns 0 for n=0.

        """
        return n * be.sin(n * be.arccos(x)) / be.sqrt(1 - x**2)

    def _validate_inputs(self, x_norm, y_norm):
        """Validates the input coordinates for the Chebyshev polynomial surface.

        Args:
            x_norm (be.ndarray or float): The normalized x-coordinate(s).
            y_norm (be.ndarray or float): The normalized y-coordinate(s).

        """
        if be.any(be.abs(x_norm) > 1) or be.any(be.abs(y_norm) > 1):
            raise ValueError(
                "Chebyshev input coordinates must be normalized "
                "to [-1, 1]. Consider updating the normalization "
                "factors.",
            )

    def to_dict(self):
        """Converts the Chebyshev polynomial geometry to a dictionary.

        Returns:
            dict: A dictionary representation of the Chebyshev polynomial geometry.

        """
        geometry_dict = super().to_dict()
        geometry_dict.update(
            {
                "coefficients": self.coefficients.tolist(),
                "norm_x": self.norm_x,
                "norm_y": self.norm_y,
            },
        )

        return geometry_dict

    @classmethod
    def from_dict(cls, data):
        """Creates a Chebyshev polynomial geometry from a dictionary.

        Args:
            data (dict): The dictionary representation of the Chebyshev
                polynomial geometry.

        Returns:
            ChebyshevPolynomialGeometry: An instance of
            ChebyshevPolynomialGeometry.

        """
        required_keys = {"cs", "radius"}
        if not required_keys.issubset(data):
            missing = required_keys - data.keys()
            raise ValueError(f"Missing required keys: {missing}")

        cs = CoordinateSystem.from_dict(data["cs"])

        return cls(
            cs,
            data["radius"],
            data.get("conic", 0.0),
            data.get("tol", 1e-10),
            data.get("max_iter", 100),
            data.get("coefficients", []),
            data.get("norm_x", 1),
            data.get("norm_y", 1),
        )
