"""Asphere Coefficients Variable Module

This module contains the AsphereCoeffVariable class, which represents a
variable for an aspheric coefficient in an optical system. The variable can be
used in optimization problems to optimize the aspheric coefficients of an
optical system.

Kramer Harrison, 2024
"""

from __future__ import annotations

import numpy as np

from optiland.optimization.scaling.linear import LinearScaler
from optiland.optimization.variable.base import VariableBehavior


class AsphereCoeffVariable(VariableBehavior):
    """Represents a variable for an aspheric coefficient in an optical system.

    Args:
        optic (Optic): The optic object associated with the variable.
        surface_number (int): The index of the surface in the optical system.
        coeff_number (int): The index of the aspheric coefficient.
        scaler (Scaler): The scaler to use for the variable. Defaults to
            a linear scaler with a factor that depends on the coefficient
            number and order.
        **kwargs: Additional keyword arguments.

    Attributes:
        coeff_number (int): The index of the aspheric coefficient.

    """

    def __init__(
        self,
        optic,
        surface_number,
        coeff_number,
        scaler=None,
        **kwargs,
    ):
        self.coeff_number = coeff_number

        # Scaling changes with the order of the asphere per coefficient
        surf = optic.surfaces[surface_number]
        self.order = surf.geometry.order
        if scaler is None:
            factor = 10 ** (4 + self.order * self.coeff_number)
            scaler = LinearScaler(factor=factor)

        super().__init__(optic, surface_number, scaler=scaler, **kwargs)

    def get_value(self):
        """Get the current value of the aspheric coefficient.

        Returns:
            float: The current value of the aspheric coefficient.

        """
        surf = self._surfaces[self.surface_number]
        try:
            value = surf.geometry.coefficients[self.coeff_number]
        except IndexError:
            pad_width_i = max(0, self.coeff_number + 1)
            c_new = np.pad(
                surf.geometry.coefficients,
                pad_width=(0, pad_width_i),
                mode="constant",
                constant_values=0,
            )
            surf.geometry.coefficients = c_new
            value = 0
        return value

    def update_value(self, new_value):
        """Update the value of the aspheric coefficient.

        Args:
            new_value (float): The new value of the aspheric coefficient.

        """
        self.optic.updater.set_asphere_coeff(
            new_value, self.surface_number, self.coeff_number
        )

    def __str__(self):
        """Return a string representation of the variable.

        Returns:
            str: A string representation of the variable.

        """
        return f"Asphere Coeff. {self.coeff_number}, Surface {self.surface_number}"
