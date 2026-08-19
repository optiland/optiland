"""Radius of Curvature Variable Module

This module contains the RadiusVariable class, which represents a variable for
the radius of a surface in an optic. The class inherits from the
VariableBehavior class.

Kramer Harrison, 2024
"""

from __future__ import annotations

from optiland.geometries import ToroidalGeometry
from optiland.optimization.scaling.linear import LinearScaler
from optiland.optimization.variable.base import VariableBehavior


class RadiusVariable(VariableBehavior):
    """Represents a variable for the radius of a surface in an optic.

    Args:
        optic (Optic): The optic object that contains the surface.
        surface_number (int): The index of the surface in the optic.
        axis (str, optional): Radius axis for a toroidal surface. Must be
            ``"x"`` or ``"y"`` for toroidal geometry and omitted otherwise.
        scaler (Scaler): The scaler to use for the variable. Defaults to
            a linear scaler with factor=1/100 and offset=-1.0.
        **kwargs: Additional keyword arguments.

    Attributes:
        optic (Optic): The optic object that contains the surface.
        surface_number (int): The index of the surface in the optic.

    Methods:
        get_value(): Returns the current value of the radius.
        update_value(new_value): Updates the value of the radius.

    """

    def __init__(self, optic, surface_number, axis=None, scaler=None, **kwargs):
        if scaler is None:
            scaler = LinearScaler(factor=1 / 100.0, offset=-1.0)
        super().__init__(optic, surface_number, scaler=scaler, **kwargs)
        self.axis = axis

        geometry = self._surfaces[self.surface_number].geometry
        if isinstance(geometry, ToroidalGeometry):
            if self.axis is None:
                raise ValueError(
                    "An axis is required for radius variables on ToroidalGeometry."
                )
            if self.axis not in ("x", "y"):
                raise ValueError(
                    f'Invalid axis "{self.axis}" for toroidal radius variable.'
                )
        elif self.axis is not None:
            raise ValueError(
                "The axis argument is only supported for toroidal radius variables."
            )

    def get_value(self):
        """Returns the current value of the radius.

        Returns:
            float: The current value of the radius.

        """
        geometry = self._surfaces[self.surface_number].geometry
        if isinstance(geometry, ToroidalGeometry):
            if self.axis == "x":
                return geometry.R_rot
            return geometry.R_yz
        return self._surfaces.radii[self.surface_number]

    def update_value(self, new_value):
        """Updates the value of the radius.

        Args:
            new_value (float): The new value of the radius.

        """
        geometry = self._surfaces[self.surface_number].geometry
        if isinstance(geometry, ToroidalGeometry):
            if self.axis == "x":
                geometry.set_radius_x(new_value)
            else:
                geometry.set_radius_y(new_value)
            return
        self.optic.updater.set_radius(new_value, self.surface_number)

    def __str__(self):
        """Return a string representation of the variable.

        Returns:
            str: A string representation of the variable.

        """
        geometry = self._surfaces[self.surface_number].geometry
        if isinstance(geometry, ToroidalGeometry):
            return f"Radius {self.axis.upper()}, Surface {self.surface_number}"
        return f"Radius of Curvature, Surface {self.surface_number}"
