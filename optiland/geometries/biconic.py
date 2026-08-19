"""Biconic Geometry

The biconic geometry represents a surface defined by:

zx = (cx * x^2) / (1 + sqrt(1 - (1 + kx) * cx^2 * x^2))
zy = (cy * y^2) / (1 + sqrt(1 - (1 + ky) * cy^2 * y^2))
z = zx + zy

where:
- cx = 1 / Rx (curvature in x)
- cy = 1 / Ry (curvature in y)
- kx is the conic constant for the x-profile
- ky is the conic constant for the y-profile

A biconic surface is a generalization of a conic surface that allows for
independent curvature in the x and y directions. It is commonly used in
optical systems where different curvatures are required in orthogonal planes.

Kramer Harrison, 2025
"""

from __future__ import annotations

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.newton_raphson import NewtonRaphsonGeometry


class BiconicGeometry(NewtonRaphsonGeometry):
    """
    Represents a biconic geometry.

    The sag is defined by the standard sag equation applied independently
    to the x and y axes.
    zx = (cx * x^2) / (1 + sqrt(1 - (1 + kx) * cx^2 * x^2))
    zy = (cy * y^2) / (1 + sqrt(1 - (1 + ky) * cy^2 * y^2))
    z = zx + zy

    where:
    - cx = 1 / Rx (curvature in x)
    - cy = 1 / Ry (curvature in y)
    - kx is the conic constant for the x-profile
    - ky is the conic constant for the y-profile
    """

    def __init__(
        self,
        coordinate_system: CoordinateSystem,
        radius_x: float,
        radius_y: float,
        conic_x: float = 0.0,
        conic_y: float = 0.0,
        tol: float = 1e-10,
        max_iter: int = 100,
    ):
        # Pass radius_x as the primary radius for NewtonRaphsonGeometry
        super().__init__(coordinate_system, radius_x, conic_x, tol, max_iter)

        self.Rx = be.array(radius_x)
        self.Ry = be.array(radius_y)
        self.kx = be.array(conic_x)
        self.ky = be.array(conic_y)

        self.cx = be.where(be.isinf(self.Rx) | (self.Rx == 0), 0.0, 1.0 / self.Rx)
        self.cy = be.where(be.isinf(self.Ry) | (self.Ry == 0), 0.0, 1.0 / self.Ry)

        self.is_symmetric = False  # Generally not symmetric

    def sag(self, x=0, y=0):
        """Calculate the surface sag of the geometry.

        Args:
            x (float or be.ndarray, optional): The x-coordinate(s). Defaults to 0.
            y (float or be.ndarray, optional): The y-coordinate(s). Defaults to 0.

        Returns:
            be.ndarray or float: The sag value(s) at the given coordinates.
        """
        x = be.asarray(x)
        y = be.asarray(y)

        zx = be.zeros_like(x)
        zy = be.zeros_like(y)

        # Calculate sag contribution from x-profile
        if not be.all(self.cx == 0):
            sqrt_term_x_val = 1.0 - (1.0 + self.kx) * self.cx**2 * x**2
            # Ensure root term is non-negative, avoid issues at exact boundary
            sqrt_term_x = be.where(sqrt_term_x_val < 1e-14, 0.0, sqrt_term_x_val)
            denom_x = 1.0 + be.sqrt(sqrt_term_x)
            # Avoid division by zero if denom_x is zero (e.g. x is too large)
            safe_denom_x = be.where(be.abs(denom_x) < 1e-14, 1e-14, denom_x)
            zx = (self.cx * x**2) / safe_denom_x

        # Calculate sag contribution from y-profile
        if not be.all(self.cy == 0):
            sqrt_term_y_val = 1.0 - (1.0 + self.ky) * self.cy**2 * y**2
            sqrt_term_y = be.where(sqrt_term_y_val < 1e-14, 0.0, sqrt_term_y_val)
            denom_y = 1.0 + be.sqrt(sqrt_term_y)
            safe_denom_y = be.where(be.abs(denom_y) < 1e-14, 1e-14, denom_y)
            zy = (self.cy * y**2) / safe_denom_y

        return zx + zy

    def _surface_normal(self, x, y):
        """Calculate the surface normal of the geometry at the given x and y position.

        Args:
            x (be.ndarray): The x-coordinate(s) at which to calculate the normal.
            y (be.ndarray): The y-coordinate(s) at which to calculate the normal.

        Returns:
            tuple[be.ndarray, be.ndarray, be.ndarray]: The surface normal
            components (nx, ny, nz).

        """
        x = be.asarray(x)
        y = be.asarray(y)

        # Partial derivative dz/dx
        if be.all(self.cx == 0):
            dfdx = be.zeros_like(x)
        else:
            sqrt_term_x_val = 1.0 - (1.0 + self.kx) * self.cx**2 * x**2
            # Clamp to small positive to avoid nan/inf for derivative at boundary
            sqrt_term_x = be.where(sqrt_term_x_val < 1e-14, 1e-14, sqrt_term_x_val)
            denom_sqrt_x = be.sqrt(sqrt_term_x)
            # Avoid division by zero if denom_sqrt_x is zero
            safe_denom_sqrt_x = be.where(
                be.abs(denom_sqrt_x) < 1e-14, 1e-14, denom_sqrt_x
            )
            dfdx = (self.cx * x) / safe_denom_sqrt_x

        # Partial derivative dz/dy
        if be.all(self.cy == 0):
            dfdy = be.zeros_like(y)
        else:
            sqrt_term_y_val = 1.0 - (1.0 + self.ky) * self.cy**2 * y**2
            sqrt_term_y = be.where(sqrt_term_y_val < 1e-14, 1e-14, sqrt_term_y_val)
            denom_sqrt_y = be.sqrt(sqrt_term_y)
            safe_denom_sqrt_y = be.where(
                be.abs(denom_sqrt_y) < 1e-14, 1e-14, denom_sqrt_y
            )
            dfdy = (self.cy * y) / safe_denom_sqrt_y

        # Normal vector components (Optiland convention: (fx, fy, -1) / mag)
        mag_sq = dfdx**2 + dfdy**2 + 1.0
        mag = be.sqrt(mag_sq)
        # Avoid division by zero if mag is zero
        safe_mag = be.where(
            mag < 1e-14, 1.0, mag
        )  # if mag is ~0, normal is (0,0,-1) approx

        nx = dfdx / safe_mag
        ny = dfdy / safe_mag
        nz = -1.0 / safe_mag

        return nx, ny, nz

    def flip(self):
        """Flip the geometry.

        Changes the sign of the radii of curvature Rx and Ry.
        Updates the curvature attributes cx and cy accordingly.
        The conic constants kx and ky remain unchanged.
        """
        self.Rx = -self.Rx
        self.Ry = -self.Ry

        # Update curvatures, handling potential division by zero if radius is zero
        self.cx = be.where(be.isinf(self.Rx) | (self.Rx == 0), 0.0, 1.0 / self.Rx)
        self.cy = be.where(be.isinf(self.Ry) | (self.Ry == 0), 0.0, 1.0 / self.Ry)

    def __str__(self) -> str:
        return "Biconic"

    def scale(self, scale_factor: float):
        """Scale the geometry parameters.

        Args:
            scale_factor (float): The factor by which to scale the geometry.
        """
        # Scale radii
        self.Rx = self.Rx * scale_factor
        self.Ry = self.Ry * scale_factor

        # Update curvatures
        self.cx = be.where(be.isinf(self.Rx) | (self.Rx == 0), 0.0, 1.0 / self.Rx)
        self.cy = be.where(be.isinf(self.Ry) | (self.Ry == 0), 0.0, 1.0 / self.Ry)

    def to_dict(self) -> dict:
        """Converts the geometry to a dictionary.

        Returns:
            dict: The dictionary representation of the geometry.

        """
        geometry_dict = super().to_dict()
        # Remove base class radius and conic as they are ambiguous for biconic
        if "radius" in geometry_dict:
            del geometry_dict["radius"]
        if "conic" in geometry_dict:
            del geometry_dict["conic"]

        geometry_dict.update(
            {
                "type": self.__class__.__name__,  # Ensure correct type
                "radius_x": float(self.Rx) if hasattr(self.Rx, "item") else self.Rx,
                "radius_y": float(self.Ry) if hasattr(self.Ry, "item") else self.Ry,
                "conic_x": float(self.kx) if hasattr(self.kx, "item") else self.kx,
                "conic_y": float(self.ky) if hasattr(self.ky, "item") else self.ky,
            }
        )
        return geometry_dict

    @classmethod
    def from_dict(cls, data: dict) -> BiconicGeometry:
        """Creates a BiconicGeometry from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the biconic surface,
                containing keys like 'cs', 'radius_x', 'radius_y', 'conic_x',
                'conic_y'.

        Returns:
            BiconicGeometry: An instance of BiconicGeometry.
        """
        required_keys = {"cs", "radius_x", "radius_y"}
        if not required_keys.issubset(data):
            missing = required_keys - set(data.keys())
            raise ValueError(f"Missing required BiconicGeometry keys: {missing}")

        cs = CoordinateSystem.from_dict(data["cs"])

        return cls(
            coordinate_system=cs,
            radius_x=data["radius_x"],
            radius_y=data["radius_y"],
            conic_x=data.get("conic_x", 0.0),
            conic_y=data.get("conic_y", 0.0),
            tol=data.get("tol", 1e-10),
            max_iter=data.get("max_iter", 100),
        )
