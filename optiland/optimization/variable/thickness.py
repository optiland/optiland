"""Thickness Variable Module

This module contains the ThicknessVariable class, which represents a variable
for the thickness of an optic surface. The class inherits from the
VariableBehavior class.

Kramer Harrison, 2024
"""

from __future__ import annotations

from optiland.optimization.scaling.linear import LinearScaler
from optiland.optimization.variable.base import VariableBehavior


class ThicknessVariable(VariableBehavior):
    """Represents a variable for the thickness of an optic surface.

    Args:
        optic (Optic): The optic object to which the surface belongs.
        surface_number (int): The number of the surface.
        scaler (Scaler): The scaler to use for the variable. Defaults to
            a linear scaler with factor=1/10.0 and offset=-1.0.
        **kwargs: Additional keyword arguments.

    Attributes:
        optic (Optic): The optic object to which the surface belongs.
        surface_number (int): The number of the surface.

    Methods:
        get_value(): Returns the current thickness value of the surface.
        update_value(new_value): Updates the thickness value of the surface.

    """

    def __init__(self, optic, surface_number, scaler=None, **kwargs):
        if scaler is None:
            scaler = LinearScaler(factor=1 / 10.0, offset=-1.0)
        super().__init__(optic, surface_number, scaler=scaler, **kwargs)

    def get_value(self):
        """Returns the current thickness value of the surface.

        Returns:
            float: The current thickness value.

        """
        return self._surfaces.get_thickness(self.surface_number)[0]

    def update_value(self, new_value):
        """Updates the thickness value of the surface.

        Args:
            new_value (float): The new thickness value.

        """
        self.optic.updater.set_thickness(new_value, self.surface_number)

    def __str__(self):
        """Return a string representation of the variable.

        Returns:
            str: A string representation of the variable.

        """
        return f"Thickness, Surface {self.surface_number}"
