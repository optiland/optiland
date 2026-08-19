"""Polynomial XY Variable Module

This module contains the PolynomialCoeffVariable class, which represents a
variable for a polynomial coefficient of a PolynomialGeometry. The class
inherits from the VariableBehavior class.

Kramer Harrison, 2024
"""

from __future__ import annotations

import numpy as np

from optiland.optimization.scaling.identity import IdentityScaler
from optiland.optimization.variable.base import VariableBehavior


class PolynomialCoeffVariable(VariableBehavior):
    """Represents a variable for a polynomial coefficient of a PolynomialGeometry.

    Args:
        optic (Optic): The optic object associated with the variable.
        surface_number (int): The index of the surface in the optical system.
        coeff_index (tuple(int, int)): The (x, y) indices of the polynomial
            coefficient.
        scaler (Scaler): The scaler to use for the variable. Defaults to
            IdentityScaler().
        **kwargs: Additional keyword arguments.

    Attributes:
        coeff_number (int): The index of the polynomial coefficient.

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
        super().__init__(optic, surface_number, scaler=scaler, **kwargs)
        self.coeff_index = coeff_index

    def get_value(self):
        """Get the current value of the polynomial coefficient.

        Returns:
            float: The current value of the polynomial coefficient.

        """
        surf = self._surfaces[self.surface_number]
        i, j = self.coeff_index
        try:
            value = surf.geometry.coefficients[i][j]
        except IndexError:
            pad_width_i = max(0, i + 1 - surf.geometry.coefficients.shape[0])
            pad_width_j = max(0, j + 1 - surf.geometry.coefficients.shape[1])
            c_new = np.pad(
                surf.geometry.coefficients,
                pad_width=((0, pad_width_i), (0, pad_width_j)),
                mode="constant",
                constant_values=0,
            )
            surf.geometry.coefficients = c_new
            value = 0
        return value

    def update_value(self, new_value):
        """Update the value of the polynomial coefficient.

        Args:
            new_value (float): The new value of the polynomial coefficient.

        """
        surf = self.optic.surfaces[self.surface_number]
        i, j = self.coeff_index
        try:
            surf.geometry.coefficients[i][j] = new_value
        except IndexError:
            pad_width_i = max(0, i + 1 - surf.geometry.coefficients.shape[0])
            pad_width_j = max(0, j + 1 - surf.geometry.coefficients.shape[1])
            c_new = np.pad(
                surf.geometry.coefficients,
                pad_width=((0, pad_width_i), (0, pad_width_j)),
                mode="constant",
                constant_values=0,
            )
            c_new[i][j] = new_value
            surf.geometry.coefficients = c_new

    def __str__(self):
        """Return a string representation of the variable.

        Returns:
            str: A string representation of the variable.

        """
        return f"Poly. Coeff. {self.coeff_index}, Surface {self.surface_number}"
