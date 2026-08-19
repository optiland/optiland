"""Decenter Variable Module

This module contains the DecenterVariable class, which represents a variable
for the decenter of an optic surface. The decenter variable can be used to
optimize the decenter of a surface in the optical system.

Kramer Harrison, 2024
"""

from __future__ import annotations

from optiland.optimization.scaling.identity import IdentityScaler
from optiland.optimization.variable.base import VariableBehavior


class DecenterVariable(VariableBehavior):
    """Represents a variable for the decenter of an optic surface.

    Args:
        optic (Optic): The optic object to which the surface belongs.
        surface_number (int): The number of the surface.
        axis (str): The axis of the decenter. Valid values are 'x', 'y', and 'z'.
        scaler (Scaler): The scaler to use for the variable. Defaults to
            IdentityScaler().
        **kwargs: Additional keyword arguments.

    Attributes:
        optic (Optic): The optic object to which the surface belongs.
        surface_number (int): The number of the surface.

    Methods:
        get_value(): Returns the current decenter value of the surface.
        update_value(new_value): Updates the decenter value of the surface.

    """

    def __init__(self, optic, surface_number, axis, scaler=None, **kwargs):
        if scaler is None:
            scaler = IdentityScaler()
        super().__init__(optic, surface_number, scaler=scaler, **kwargs)
        self.axis = axis

        if self.axis not in ["x", "y", "z"]:
            raise ValueError(f'Invalid axis "{self.axis}" for decenter variable.')

    def get_value(self):
        """Returns the current decenter value of the surface.

        Returns:
            float: The current decenter value.

        """
        surf = self._surfaces[self.surface_number]
        if self.axis == "x":
            value = surf.geometry.cs.x
        elif self.axis == "y":
            value = surf.geometry.cs.y
        elif self.axis == "z":
            value = surf.geometry.cs.z
        else:
            raise ValueError(f'Invalid axis "{self.axis}" for decenter variable.')
        return self.scaler.scale(value)

    def update_value(self, new_value):
        """Updates the decenter value of the surface.

        Args:
            new_value (float): The new decenter value.

        """
        surf = self._surfaces[self.surface_number]
        unscaled_value = self.scaler.inverse_scale(new_value)
        if self.axis == "x":
            surf.geometry.cs.x = unscaled_value
        elif self.axis == "y":
            surf.geometry.cs.y = unscaled_value
        elif self.axis == "z":
            surf.geometry.cs.z = unscaled_value
        else:
            raise ValueError(f'Invalid axis "{self.axis}" for decenter variable.')

    def __str__(self):
        """Return a string representation of the variable.

        Returns:
            str: A string representation of the variable.

        """
        return f"Decenter {self.axis.upper()}, Surface {self.surface_number}"
