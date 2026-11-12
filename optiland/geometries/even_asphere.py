"""Even Asphere Geometry

The Even Asphere geometry represents a surface defined by an even asphere in
two dimensions. The surface is defined as:

z = r^2 / (R * (1 + sqrt(1 - (1 + k) * r^2 / R^2))) + sum(Ci * r^(2i))

where
- r^2 = x^2 + y^2
- R is the radius of curvature
- k is the conic constant
- Ci are the aspheric coefficients

Even-order aspheric surfaces are commonly used to correct specific aberrations
while maintaining rotational symmetry. These surfaces are defined by polynomial
terms with even exponents, ensuring symmetry about the optical axis.

Kramer Harrison, 2024
"""

from __future__ import annotations

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.newton_raphson import NewtonRaphsonGeometry


class EvenAsphere(NewtonRaphsonGeometry):
    """Represents an even asphere geometry defined as:

    z = r^2 / (R * (1 + sqrt(1 - (1 + k) * r^2 / R^2))) + sum(Ci * r^(2i))

    where
    - r^2 = x^2 + y^2
    - R is the radius of curvature
    - k is the conic constant
    - Ci are the aspheric coefficients

    Even-order aspheric surfaces are commonly used to correct specific
    aberrations while maintaining rotational symmetry. These surfaces are
    defined by polynomial terms with even exponents, ensuring symmetry about
    the optical axis.

    Args:
        coordinate_system (CoordinateSystem): The coordinate system of the geometry.
        radius (float): The radius of curvature of the base sphere.
        conic (float, optional): The conic constant of the base sphere.
            Defaults to 0.0.
        tol (float, optional): Tolerance for Newton-Raphson iteration.
            Defaults to 1e-10.
        max_iter (int, optional): Maximum iterations for Newton-Raphson.
            Defaults to 100.
        coefficients (list[float], optional): A list of even aspheric
            coefficients C_i, where the term is C_i * r^(2i).
            The list index corresponds to i-1 (e.g., coefficients[0] is C_1 for r^2).
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
        super().__init__(coordinate_system, radius, conic, tol, max_iter)
        self.coefficients = list(coefficients)
        self.is_symmetric = True
        self.order = 2  # used for optimization scaling

    def __str__(self):
        return "Even Asphere"

    _r2_index = 0  # index of the r^2 coefficient

    @property
    def paraxial_radius(self):
        """Vertex radius including the curvature 2*C of the r^2 term."""
        if len(self.coefficients) <= self._r2_index:
            return self.radius
        c2 = self.coefficients[self._r2_index]
        if c2 == 0:
            return self.radius
        return 1 / (1 / self.radius + 2 * c2)

    def scale(self, scale_factor: float):
        """Scale the geometry parameters.

        Args:
            scale_factor (float): The factor by which to scale the geometry.
        """
        super().scale(scale_factor)
        for i in range(len(self.coefficients)):
            # C_i' = C_i * s^(1 - 2(i+1))
            self.coefficients[i] *= scale_factor ** (1 - 2 * (i + 1))

    def flip(self):
        """Flip the geometry.

        Changes the sign of the radius of curvature and of every aspheric
        coefficient, so that the sag changes sign. The conic constant remains
        unchanged.
        """
        super().flip()
        self.coefficients = [-c for c in self.coefficients]

    def sag(self, x=0, y=0):
        """Calculates the sag of the asphere at the given coordinates.

        Args:
            x (float or be.ndarray, optional): The x-coordinate(s). Defaults to 0.
            y (float or be.ndarray, optional): The y-coordinate(s). Defaults to 0.

        Returns:
            be.ndarray or float: The sag value(s) at the given coordinates.

        """
        # The compiled trace plan repeats these expressions (_asphere_sag in
        # optiland/backend/numpy_backend/_trace_kernel.py). Change both.
        r2 = x**2 + y**2
        z = r2 / (self.radius * (1 + be.sqrt(1 - (1 + self.k) * r2 / self.radius**2)))
        # Use the length: the coefficients can also be an array or a tensor.
        n = len(self.coefficients)
        if n > 0:
            # Horner's method: r2 * (C1 + r2 * (C2 + ... + r2 * Cn)).
            polynomial = self.coefficients[n - 1]
            for i in range(n - 2, -1, -1):
                polynomial = self.coefficients[i] + r2 * polynomial
            z = z + r2 * polynomial

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
        # The compiled trace plan repeats these expressions (_asphere_slopes
        # and _unit_normal in optiland/backend/numpy_backend/_trace_kernel.py).
        # Change both.
        r2 = x**2 + y**2

        denom = self.radius * be.sqrt(1 - (1 + self.k) * r2 / self.radius**2)
        dfdx = x / denom
        dfdy = y / denom

        n = len(self.coefficients)
        if n > 0:
            # d(sag)/d(r2) of the polynomial, C1 + 2 C2 r2 + ... + n Cn r2^(n-1),
            # with Horner's method. It serves both slopes, because
            # d(sag)/dx = 2 x d(sag)/d(r2) and d(sag)/dy = 2 y d(sag)/d(r2).
            derivative = n * self.coefficients[n - 1]
            for i in range(n - 1, 0, -1):
                derivative = i * self.coefficients[i - 1] + r2 * derivative
            dfdx = dfdx + 2 * x * derivative
            dfdy = dfdy + 2 * y * derivative

        mag = be.sqrt(dfdx**2 + dfdy**2 + 1)

        nx = dfdx / mag
        ny = dfdy / mag
        nz = -1 / mag

        return nx, ny, nz

    def to_dict(self):
        """Converts the geometry to a dictionary.

        Returns:
            dict: The dictionary representation of the geometry.

        """
        data = super().to_dict()
        data["coefficients"] = self.coefficients

        return data

    @classmethod
    def from_dict(cls, data):
        """Creates an asphere from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the asphere.

        Returns:
            EvenAsphere: An instance of EvenAsphere.

        """
        required_keys = {"cs", "radius"}
        if not required_keys.issubset(data):
            missing = required_keys - data.keys()
            raise ValueError(f"Missing required keys: {missing}")

        cs = CoordinateSystem.from_dict(data["cs"])
        conic = data.get("conic", 0.0)
        tol = data.get("tol", 1e-10)
        max_iter = data.get("max_iter", 100)
        coefficients = data.get("coefficients", [])

        return cls(cs, data["radius"], conic, tol, max_iter, coefficients)
