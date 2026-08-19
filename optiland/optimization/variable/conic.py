"""Conic Constant Variable Module

This module contains the class for a conic constant variable in an optic
system. The ConicVariable class is a subclass of the VariableBehavior class
that represents a variable for the conic constant of a surface in an optic
system. It is used in the optimization process for conic surfaces.

Kramer Harrison, 2024
"""

from __future__ import annotations

from optiland.optimization.scaling.identity import IdentityScaler
from optiland.optimization.variable.base import VariableBehavior


class ConicVariable(VariableBehavior):
    """Represents a variable for the conic constant of a surface in an optic.

    Args:
        optic (Optic): The optic object to which the surface belongs.
        surface_number (int): The index of the surface in the optic.
        scaler (Scaler): The scaler to use for the variable. Defaults to
            IdentityScaler().
        **kwargs: Additional keyword arguments.

    Attributes:
        optic (Optic): The optic object to which the surface belongs.
        surface_number (int): The index of the surface in the optic.

    Methods:
        get_value: Returns the current conic constant of the surface.
        update_value: Updates the conic value of the surface.

    """

    def __init__(self, optic, surface_number, scaler=None, **kwargs):
        if scaler is None:
            scaler = IdentityScaler()
        super().__init__(optic, surface_number, scaler=scaler, **kwargs)

    def get_value(self):
        """Returns the current conic constant of the surface.

        Returns:
            float: The conic constant of the surface.

        """
        return self._surfaces.conic[self.surface_number]

    def update_value(self, new_value):
        """Updates the conic value of the surface.

        Args:
            new_value (float): The new conic constant to set.

        """
        self.optic.updater.set_conic(new_value, self.surface_number)

    def __str__(self):
        """Return a string representation of the variable.

        Returns:
            str: A string representation of the variable.

        """
        return f"Conic Constant, Surface {self.surface_number}"
