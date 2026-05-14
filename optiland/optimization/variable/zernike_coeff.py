"""Zernike Coefficients Variable Module

This module contains the class for a Zernike coefficient variable in an optic
system. The ZernikeCoeffVariable class is a subclass of the
PolynomialCoeffVariable class that represents a variable for a Zernike
coefficient of a ZernikeGeometry. It is used in the optimization process for
Zernike geometries.

drpaprika, 2025
"""

from __future__ import annotations

import optiland.backend as be
from optiland.optimization.scaling.identity import IdentityScaler
from optiland.optimization.variable.polynomial_coeff import PolynomialCoeffVariable


class ZernikeCoeffVariable(PolynomialCoeffVariable):
    """Represents a variable for a Zernike coefficient of a ZernikeGeometry.

    Args:
        optic (Optic): The optic object associated with the variable.
        surface_number (int): The index of the surface in the optical system.
        coeff_index (int): The i index of the Zernike coefficient.
        scaler (Scaler): The scaler to use for the variable. Defaults to
            IdentityScaler().
        **kwargs: Additional keyword arguments.

    Attributes:
        coeff_number (int): The index of the Zernike coefficient.

    """

    def __init__(
        self,
        optic,
        surface_number,
        coeff_index,
        scaler=None,
        **kwargs,
    ):
        if scaler is None:
            scaler = IdentityScaler()
        super().__init__(optic, surface_number, coeff_index, scaler=scaler, **kwargs)

    def get_value(self):
        """Get the current value of the Zernike coefficient.

        Returns:
            float: The current value of the Zernike coefficient.
        """
        surf = self._surfaces[self.surface_number]
        i = self.coeff_index
        coeffs = surf.geometry.coefficients

        if i < len(coeffs):
            return coeffs[i]

        pad_width = i + 1 - len(coeffs)
        padding = be.array([0.0] * pad_width)
        surf.geometry.coefficients = be.concatenate([coeffs, padding])

        return surf.geometry.coefficients[i]

    def update_value(self, new_value):
        """Update the value of the Zernike coefficient.

        Args:
            new_value (float): The new value of the Zernike coefficient.
        """
        surf = self.optic.surfaces[self.surface_number]
        i = self.coeff_index
        coeffs = surf.geometry.coefficients

        if i >= len(coeffs):
            pad_width = i + 1 - len(coeffs)
            padding = be.array([0.0] * pad_width)
            coeffs = be.concatenate([coeffs, padding])

        surf.geometry.coefficients = be.stack(
            [new_value if j == i else coeff for j, coeff in enumerate(coeffs)]
        )

    def __str__(self):
        """Return a string representation of the variable.

        Returns:
            str: A string representation of the variable.

        """
        return f"Zernike Coeff. {self.coeff_index}, Surface {self.surface_number}"
