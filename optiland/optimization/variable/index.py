"""Refractive Index Variable Module

This module contains the IndexVariable class, which represents a variable for
the index of refraction at a specific surface and wavelength. The variable can
be used in optimization problems to optimize the index of refraction at a
specific surface.

Kramer Harrison, 2024
"""

from __future__ import annotations

from optiland.optimization.scaling.linear import LinearScaler
from optiland.optimization.variable.base import VariableBehavior


class IndexVariable(VariableBehavior):
    """Represents a variable for the index of refraction at a specific surface
    and wavelength.

    Args:
        optic (Optic): The optic object associated with the variable.
        surface_number (int): The surface number where the variable is applied.
        wavelength (float): The wavelength at which the index of refraction is
            calculated.
        scaler (Scaler): The scaler to use for the variable. Defaults to
            a linear scaler with offset=-1.5.
        **kwargs: Additional keyword arguments.

    Attributes:
        optic (Optic): The optic object to which the surface belongs.
        surface_number (int): The number of the surface.
        wavelength (float): The wavelength at which the index of refraction is
            calculated.

    Methods:
        get_value(): Returns the value of the index of refraction at the
            specified surface and wavelength.
        update_value(new_value): Updates the value of the index of refraction
            at the specified surface.

    """

    def __init__(self, optic, surface_number, wavelength, scaler=None, **kwargs):
        if scaler is None:
            scaler = LinearScaler(factor=1.0, offset=-1.5)
        super().__init__(optic, surface_number, scaler=scaler, **kwargs)
        self.wavelength = wavelength

    def get_value(self):
        """Returns the value of the index of refraction at the specified surface
        and wavelength.

        Returns:
            float: The value of the index of refraction.

        """
        n = self.optic.surfaces.n(self.wavelength)
        return n[self.surface_number]

    def update_value(self, new_value):
        """Updates the value of the index of refraction at the specified surface.

        Args:
            new_value (float): The new value of the index of refraction.

        """
        self.optic.updater.set_index(new_value, self.surface_number)

    def __str__(self):
        """Return a string representation of the variable.

        Returns:
            str: A string representation of the variable.

        """
        return f"Refractive Index, Surface {self.surface_number}"
