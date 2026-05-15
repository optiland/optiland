"""Zernike Geometry

The Zernike polynomial geometry represents a surface defined by a Zernike
polynomial in two dimensions. The surface is defined as:

z(x,y) = r^2 / (R * (1 + sqrt(1 - (1 + k) * r^2 / R^2))) +
    sum_i [c[i] * Z_i(rho, phi)]

where:
- r^2 = x^2 + y^2
- R is the radius of curvature
- k is the conic constant
- c[i] is the coefficient for the i-th Zernike polynomial
- Z_i(...) is the i-th Zernike polynomial in polar coordinates
- rho = sqrt(x^2 + y^2) / normalization, phi = atan2(y, x)

Zernike polynomials are a set of orthogonal functions defined over the unit
disk, widely used in freeform optical surface design. They efficiently
describe wavefront aberrations and complex surface deformations by decomposing
them into radial and azimuthal components. Their orthogonality ensures minimal
cross-coupling between terms, making them ideal for optimizing optical systems.
In freeform optics, they enable precise control of surface shape,
improving performance beyond traditional spherical and aspheric designs.

drpaprika, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.newton_raphson import NewtonRaphsonGeometry
from optiland.zernike import ZernikeFringe, ZernikeNoll, ZernikeStandard

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from optiland._types import ZernikeType
    from optiland.zernike.base import BaseZernike


__all__ = [
    "ZernikePolynomialGeometry",
]

_ZERNIKE_TYPES: dict[ZernikeType, type[BaseZernike]] = {
    "standard": ZernikeStandard,
    "noll": ZernikeNoll,
    "fringe": ZernikeFringe,
}


class ZernikePolynomialGeometry(NewtonRaphsonGeometry):
    """Represents a Zernike polynomial geometry defined as:

    z(x,y) = r^2 / (R * (1 + sqrt(1 - (1 + k) * r^2 / R^2))) +
        sum_i [c[i] * Z_i(rho, phi)]

    where:
    - r^2 = x^2 + y^2
    - R is the radius of curvature
    - k is the conic constant
    - c[i] is the coefficient for the i-th Zernike polynomial
    - Z_i(...) is the i-th Zernike polynomial in polar coordinates
    - rho = sqrt(x^2 + y^2) / normalization, phi = atan2(y, x)

    The coefficients are defined in a 1D array where coefficients[i] is the
    coefficient for Z_i.

    Args:
        coordinate_system (str): The coordinate system used for the geometry.
        radius (float): The radius of curvature of the geometry.
        conic (float, optional): The conic constant of the geometry.
            Defaults to 0.0.
        tol (float, optional): The tolerance value used in calculations.
            Defaults to 1e-10.
        max_iter (int, optional): The maximum number of iterations used in
            calculations. Defaults to 100.
        coefficients (list or be.ndarray, optional): The coefficients of the
            Zernike polynomial surface. Defaults to an empty list, indicating
            no Zernike polynomial coefficients are used.
        zernike_type (str, optional): The type of Zernike polynomial to use.
            Defaults to "standard". Options are "standard", "noll", or "fringe".
        norm_radius (float, optional): The normalization radius for the
            Zernike polynomial coordinates. If None, the radius scales
            automatically during paraxial updates. Defaults to None.
    """

    def __init__(
        self,
        coordinate_system: str,
        radius: float,
        conic: float = 0.0,
        tol: float = 1e-10,
        max_iter: int = 100,
        coefficients: NDArray | None = None,
        zernike_type: ZernikeType = "standard",
        norm_radius: float | None = None,
    ):
        super().__init__(coordinate_system, radius, conic, tol, max_iter)

        if zernike_type not in _ZERNIKE_TYPES:
            raise ValueError(
                "Zernike type must be one of 'standard', 'noll', or 'fringe', got "
                f"{zernike_type}",
            )
        if norm_radius is not None and norm_radius <= 0:
            raise ValueError(
                f"Normalization radius must be positive, got {norm_radius}"
            )

        coefficients = be.atleast_1d(coefficients if coefficients is not None else [])

        self.zernike = _ZERNIKE_TYPES[zernike_type](coeffs=coefficients)
        self.zernike_type: ZernikeType = zernike_type
        if norm_radius is not None:
            self.norm_radius = norm_radius
            self.normalization_mode = "manual"
        else:
            self.norm_radius = 1.0
            self.normalization_mode = "auto"
        self.is_symmetric = False

    @property
    def coefficients(self) -> NDArray:
        """Get the coefficients of the Zernike polynomial surface."""
        return self.zernike.coeffs

    @coefficients.setter
    def coefficients(self, value: NDArray) -> None:
        """Set the coefficients of the Zernike polynomial surface."""
        self.zernike = _ZERNIKE_TYPES[self.zernike_type](coeffs=be.atleast_1d(value))

    def __str__(self) -> str:
        return "Zernike Polynomial"

    def scale(self, scale_factor: float):
        """Scale the geometry parameters.

        Args:
            scale_factor (float): The factor by which to scale the geometry.
        """
        super().scale(scale_factor)
        self.norm_radius = self.norm_radius * scale_factor
        self.coefficients = self.coefficients * scale_factor

    def update_normalization(self, semi_aperture: float) -> None:
        if self.normalization_mode == "auto":
            self.norm_radius = semi_aperture * 1.25

    def sag(self, x: NDArray, y: NDArray) -> NDArray:  # type: ignore
        """Calculate the sag of the Zernike polynomial surface at the given
        coordinates.

        Args:
            x (float, be.ndarray): The Cartesian x-coordinate(s).
            y (float, be.ndarray): The Cartesian y-coordinate(s).

        Returns:
            be.ndarray: The sag value at the given Cartesian coordinates.
        """
        x_norm = x / self.norm_radius
        y_norm = y / self.norm_radius

        self._validate_inputs(x_norm, y_norm)

        # Convert to local polar
        rho2 = x_norm**2 + y_norm**2
        center = rho2 == 0

        safe_rho2 = be.where(center, be.ones_like(rho2), rho2)
        safe_rho = be.sqrt(safe_rho2)
        rho = be.where(center, be.zeros_like(safe_rho), safe_rho)

        x_phi = be.where(center, be.ones_like(x_norm), x_norm)
        y_phi = be.where(center, be.zeros_like(y_norm), y_norm)
        phi = be.arctan2(y_phi, x_phi)

        # Base conic
        r2 = x**2 + y**2

        if bool(be.all(be.isinf(self.radius))):
            z = be.zeros_like(x)
        else:
            z = r2 / (
                self.radius * (1 + be.sqrt(1 - (1 + self.k) * r2 / self.radius**2))
            )

        # Add Zernike polynomial contributions
        z = z + self.zernike.poly(rho, phi)

        return z

    def _surface_normal(
        self,
        x: NDArray,
        y: NDArray,
    ) -> tuple[float, float, float]:
        """Calculate the surface normal of the full surface (conic + Zernike)
        in Cartesian coordinates at (x, y).

        Args:
            x (float or be.ndarray): x-coordinate(s).
            y (float or be.ndarray): y-coordinate(s).

        Returns:
            (nx, ny, nz): Normal vector components in Cartesian coords.

        """
        # Conic partial derivatives:
        r2 = x**2 + y**2

        if bool(be.all(be.isinf(self.radius))):
            dzdx = be.zeros_like(x)
            dzdy = be.zeros_like(y)
        else:
            denominator = self.radius * be.sqrt(1 - (1 + self.k) * r2 / self.radius**2)

            valid_denominator = be.abs(denominator) > 0
            safe_denominator = be.where(
                valid_denominator,
                denominator,
                be.ones_like(denominator),
            )

            dzdx = be.where(
                valid_denominator,
                x / safe_denominator,
                be.zeros_like(x),
            )
            dzdy = be.where(
                valid_denominator,
                y / safe_denominator,
                be.zeros_like(y),
            )

        # Now add partial derivatives from the Zernike expansions
        x_norm = x / self.norm_radius
        y_norm = y / self.norm_radius

        rho2 = x_norm**2 + y_norm**2
        center = rho2 == 0

        safe_rho2 = be.where(center, be.ones_like(rho2), rho2)
        safe_rho = be.sqrt(safe_rho2)
        rho = be.where(center, be.zeros_like(safe_rho), safe_rho)

        x_phi = be.where(center, be.ones_like(x_norm), x_norm)
        y_phi = be.where(center, be.zeros_like(y_norm), y_norm)
        phi = be.arctan2(y_phi, x_phi)

        # Chain rule:
        # dZ/dx = dZ/drho * d(rho)/dx + dZ/dphi * d(phi)/dx
        # We'll define the partials of (rho,phi) wrt x:
        #   drho/dx    = x / (norm_x^2 * rho)
        #   dphi/dx  = - y / (rho^2 * norm_y * norm_x)
        drho_dx = be.where(
            center,
            be.zeros_like(x),
            (x / (self.norm_radius**2)) / safe_rho,
        )
        drho_dy = be.where(
            center,
            be.zeros_like(y),
            (y / (self.norm_radius**2)) / safe_rho,
        )
        dphi_dx = be.where(
            center,
            be.zeros_like(x),
            -(y_norm) / safe_rho2 * (1.0 / self.norm_radius),
        )
        dphi_dy = be.where(
            center,
            be.zeros_like(y),
            +(x_norm) / safe_rho2 * (1.0 / self.norm_radius),
        )

        for (n, m), c in zip(self.zernike.indices, self.zernike.coeffs, strict=True):
            dZdrho, dZdphi = self.zernike.get_derivative(n, m, rho, phi)
            # Partial derivatives w.r.t. x and y
            dzdx = dzdx + c * (dZdrho * drho_dx + dZdphi * dphi_dx)
            dzdy = dzdy + c * (dZdrho * drho_dy + dZdphi * dphi_dy)

        # Surface normal vector in cartesian coords: (-dzdx, -dzdy, 1)
        # normalized. Check sign conventions!
        nx = +dzdx
        ny = +dzdy
        norm = be.sqrt(nx**2 + ny**2 + 1)
        nx = nx / norm
        ny = ny / norm
        nz = -be.ones_like(x) / norm

        return (nx, ny, nz)

    def _validate_inputs(self, x_norm: float, y_norm: float) -> None:
        """Validate the input coordinates for the Zernike polynomial surface.

        Args:
            x_norm (be.ndarray): The normalized x values.

        """
        if be.any(be.abs(x_norm) > 1) or be.any(be.abs(y_norm) > 1):
            raise ValueError(
                "Zernike coordinates must be normalized "
                "to [-1, 1]. Consider updating the normalization "
                "radius to 1.1x the surface aperture.",
            )

    def to_dict(self) -> dict:
        """Convert the Zernike polynomial geometry to a dictionary.

        Returns:
            dict: The Zernike polynomial geometry as a dictionary.

        """
        geometry_dict = super().to_dict()
        geometry_dict.update(
            {
                "coefficients": list(self.zernike.coeffs),
                "zernike_type": self.zernike_type,
                "norm_radius": self.norm_radius,
            },
        )

        return geometry_dict

    @classmethod
    def from_dict(cls, data: dict) -> ZernikePolynomialGeometry:
        """Create a Zernike polynomial geometry from a dictionary.

        Args:
            data (dict): The dictionary representation of the Zernike
                polynomial geometry.

        Returns:
            ZernikePolynomialGeometry: The Zernike polynomial geometry.

        """
        required_keys = {"cs", "radius"}
        if not required_keys.issubset(data):
            missing = required_keys - data.keys()
            raise ValueError(f"Missing required keys: {missing}")

        cs = CoordinateSystem.from_dict(data["cs"])

        return cls(
            coordinate_system=cs,
            radius=data["radius"],
            conic=data.get("conic", 0.0),
            tol=data.get("tol", 1e-10),
            max_iter=data.get("max_iter", 100),
            coefficients=be.atleast_1d(data.get("coefficients", [])),
            zernike_type=data.get("zernike_type", "standard"),
            norm_radius=data.get("norm_radius", 1),
        )
