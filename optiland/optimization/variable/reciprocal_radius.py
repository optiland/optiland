"""Reciprocal Radius Variable Module

This module contains the ReciprocalRadiusVariable class, which represents a variable for
the reciprocal of the radius of a surface in an optic. The transformation is useful
because a nearly flat surface (with a very large positive or negative radius) has a
reciprocal near zero, which makes the optimization around a nearly flat surface
continuous. The reciprocal radius provides a continuous transition when surface changes
from convex to concave (when the radius changes sign).


Daniel Miranda, 2025
ALL rights ceded to Kramer Harrison
"""

from __future__ import annotations

import optiland.backend as be
from optiland.optimization.scaling.linear import LinearScaler
from optiland.optimization.variable.base import VariableBehavior


class ReciprocalRadiusVariable(VariableBehavior):
    """Represents a variable for the reciprocal of the radius of a surface in an optic.

    Args:
        optic (Optic): The optic object that contains the surface.
        surface_number (int): The index of the surface in the optic.
        scaler (Scaler): The scaler to use for the variable. Defaults to
            a linear scaler with factor=10.0.
        **kwargs: Additional keyword arguments.

    Attributes:
        optic (Optic): The optic object that contains the surface.
        surface_number (int): The index of the surface in the optic.

    Methods:
        get_value(): Returns the current value of the reciprocal radius.
        update_value(new_value): Updates the value via the reciprocal, converting back
         to radius.
    """

    def __init__(self, optic, surface_number, scaler=None, **kwargs):
        if scaler is None:
            scaler = LinearScaler(factor=10.0)
        super().__init__(optic, surface_number, scaler=scaler, **kwargs)

    def get_value(self):
        """Returns the current value of the reciprocal of the radius.

        Returns:
            float: The current reciprocal radius.
        """
        radius = self._surfaces.radii[self.surface_number]
        # Avoid division by zero
        if radius != 0:
            reciprocal = 1.0 / radius if be.isfinite(radius) else 0.0
        else:
            reciprocal = be.inf
        return reciprocal

    def update_value(self, new_value):
        """Updates the surface radius based on the new reciprocal variable value.

        Args:
            new_value (float): The new reciprocal radius value.
        """
        # Allow zero but handle appropriately
        new_radius = be.inf if new_value == 0 else 1.0 / new_value
        self.optic.updater.set_radius(new_radius, self.surface_number)

    def __str__(self):
        """Return a string representation of the variable.

        Returns:
            str: A string representation of the reciprocal radius variable.
        """
        n = self.surface_number
        v = self.get_value()

        return f"Reciprocal Radius of Curvature - Surface {n} - value: {v}"
