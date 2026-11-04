"""Abbe Number Variable Module

This module contains the AbbeVariable class, which represents a variable for
the Abbe number at a specific surface. The variable can
be used in optimization problems to optimize the Abbe number at a
specific surface.

Ziyi Xiong, 2026/11
"""

from __future__ import annotations

from optiland.optimization.scaling.linear import LinearScaler
from optiland.optimization.variable.base import VariableBehavior


class AbbeVariable(VariableBehavior):
    """Represents a variable for the Abbe number at a specific surface.

    Args:
        optic (Optic): The optic object associated with the variable.
        surface_number (int): The surface number where the variable is applied.
        scaler (Scaler): The scaler to use for the variable. Defaults to
            a linear scaler with offset=-1.5.
        **kwargs: Additional keyword arguments.

    Attributes:
        optic (Optic): The optic object to which the surface belongs.
        surface_number (int): The number of the surface.

    Methods:
        get_value(): Returns the value of the Abbe number at the
            specified surface and wavelength.
        update_value(new_value): Updates the value of the Abbe number
            at the specified surface.

    """

    def __init__(self, optic, surface_number, scaler=None, **kwargs):
        if scaler is None:
            scaler = LinearScaler(factor=1.0, offset=-1.5)
        super().__init__(optic, surface_number, scaler=scaler, **kwargs)

    def get_value(self):
        """Returns the value of the Abbe number at the specified surface
        and wavelength.

        Returns:
            float: The value of the Abbe number.

        """
        return self._surfaces.abbe()[self.surface_number]

    def update_value(self, new_value):
        """Updates the value of the Abbe number at the specified surface.

        Args:
            new_value (float): The new value of the Abbe number.

        """
        self.optic.updater.set_abbe(new_value, self.surface_number)

    def __str__(self):
        """Return a string representation of the variable.

        Returns:
            str: A string representation of the variable.

        """
        return f"Abbe Number, Surface {self.surface_number}"
