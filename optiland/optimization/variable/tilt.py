"""Tilt Variable Module

This module contains the TiltVariable class, which represents a variable for
the tilt of an optic surface. The tilt variable can be used to optimize the
tilt of a surface in an optical system.

Kramer Harrison, 2024
"""

from __future__ import annotations

from optiland.optimization.scaling.identity import IdentityScaler
from optiland.optimization.variable.base import VariableBehavior


class TiltVariable(VariableBehavior):
    """Represents a variable for the tilt of an optic surface.

    Args:
        optic (Optic): The optic object to which the surface belongs.
        surface_number (int): The number of the surface.
        axis (str): The axis of the tilt. Valid values are 'x' and 'y'.
        scaler (Scaler): The scaler to use for the variable. Defaults to
            IdentityScaler().
        **kwargs: Additional keyword arguments.

    Attributes:
        optic (Optic): The optic object to which the surface belongs.
        surface_number (int): The number of the surface.

    Methods:
        get_value(): Returns the current tilt value of the surface.
        update_value(new_value): Updates the tilt value of the surface.

    """

    def __init__(self, optic, surface_number, axis, scaler=None, **kwargs):
        if scaler is None:
            scaler = IdentityScaler()
        super().__init__(optic, surface_number, scaler=scaler, **kwargs)
        self.axis = axis

        if self.axis not in ["x", "y"]:
            raise ValueError(f'Invalid axis "{self.axis}" for tilt variable.')

    def get_value(self):
        """Returns the current tilt value of the surface.

        Returns:
            float: The current tilt value.

        """
        surf = self._surfaces[self.surface_number]
        if self.axis == "x":
            return surf.geometry.cs.rx
        elif self.axis == "y":
            return surf.geometry.cs.ry

    def update_value(self, new_value):
        """Updates the tilt value of the surface.

        Args:
            new_value (float): The new tilt value.

        """
        surf = self._surfaces[self.surface_number]
        if self.axis == "x":
            surf.geometry.cs.rx = new_value
        elif self.axis == "y":
            surf.geometry.cs.ry = new_value

    def __str__(self):
        """Return a string representation of the variable.

        Returns:
            str: A string representation of the variable.

        """
        return f"Tilt {self.axis.upper()}, Surface {self.surface_number}"
