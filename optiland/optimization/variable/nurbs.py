"""This module contains variable classes for NURBS geometries.

This module contains the `NurbsPointsVariable` and `NurbsWeightsVariable`
classes, which represent variables for the control points and weights of a
NurbsGeometry, respectively. These classes inherit from the `VariableBehavior`
class.

Kramer Harrison, 2024
"""

from __future__ import annotations

import contextlib

from optiland import backend as be
from optiland.optimization.variable.base import VariableBehavior


class NurbsPointsVariable(VariableBehavior):
    """Represents a variable for a NURBS control point.

    Args:
        optic: The optic object associated with the variable.
        surface_number: The index of the surface in the optical system.
        coeff_index: The indices of the control point.
        apply_scaling: Whether to apply scaling to the variable.
    """

    def __init__(
        self,
        optic,
        surface_number,
        coeff_index,
        apply_scaling=True,
        **kwargs,
    ):
        super().__init__(optic, surface_number, **kwargs)
        self.apply_scaling = apply_scaling
        self.coeff_index = coeff_index

    def get_value(self):
        """Gets the current value of the control point.

        Returns:
            The current value of the control point.
        """
        surf = self._surfaces[self.surface_number]
        i, j, k = self.coeff_index
        with contextlib.suppress(IndexError):
            value = surf.geometry.P[i, j, k]
        if self.apply_scaling:
            return self.scale(value)
        return value

    def update_value(self, new_value):
        """Updates the value of the control point.

        Args:
            new_value: The new value of the control point.
        """
        if self.apply_scaling:
            new_value = self.inverse_scale(new_value)
        surf = self.optic.surfaces[self.surface_number]
        i, j, k = self.coeff_index
        with contextlib.suppress(IndexError):
            if be.get_backend() == "torch":
                p_copy = be.copy(surf.geometry.P)
                p_copy[i, j, k] = new_value
                surf.geometry.P = p_copy
            else:
                surf.geometry.P[i, j, k] = new_value

    def scale(self, value):
        """Scales the value of the variable.

        This can be used for improved optimization performance.

        Args:
            value: The value to scale.

        Returns:
            The scaled value.
        """
        return value

    def inverse_scale(self, scaled_value):
        """Inverse scales the value of the variable.

        Args:
            scaled_value: The scaled value to inverse scale.

        Returns:
            The inverse-scaled value.
        """
        return scaled_value

    def __str__(self):
        """Returns a string representation of the variable.

        Returns:
            A string representation of the variable.
        """
        return f"Control Point {self.coeff_index}, Surface {self.surface_number}"


class NurbsWeightsVariable(VariableBehavior):
    """Represents a variable for a NURBS weight.

    Args:
        optic: The optic object associated with the variable.
        surface_number: The index of the surface in the optical system.
        coeff_index: The indices of the weight.
        apply_scaling: Whether to apply scaling to the variable.
    """

    def __init__(
        self,
        optic,
        surface_number,
        coeff_index,
        apply_scaling=True,
        **kwargs,
    ):
        super().__init__(optic, surface_number, **kwargs)
        self.apply_scaling = apply_scaling
        self.coeff_index = coeff_index

    def get_value(self):
        """Gets the current value of the weight.

        Returns:
            The current value of the weight.
        """
        surf = self._surfaces[self.surface_number]
        j, k = self.coeff_index
        with contextlib.suppress(IndexError):
            value = surf.geometry.W[j, k]
        if self.apply_scaling:
            return self.scale(value)
        return value

    def update_value(self, new_value):
        """Updates the value of the weight.

        Args:
            new_value: The new value of the weight.
        """
        if self.apply_scaling:
            new_value = self.inverse_scale(new_value)
        surf = self.optic.surfaces[self.surface_number]
        j, k = self.coeff_index
        with contextlib.suppress(IndexError):
            if be.get_backend() == "torch":
                w_copy = be.copy(surf.geometry.W)
                w_copy[j, k] = new_value
                surf.geometry.W = w_copy
            else:
                surf.geometry.W[j, k] = new_value

    def scale(self, value):
        """Scales the value of the variable.

        This can be used for improved optimization performance.

        Args:
            value: The value to scale.

        Returns:
            The scaled value.
        """
        return value

    def inverse_scale(self, scaled_value):
        """Inverse scales the value of the variable.

        Args:
            scaled_value: The scaled value to inverse scale.

        Returns:
            The inverse-scaled value.
        """
        return scaled_value

    def __str__(self):
        """Returns a string representation of the variable.

        Returns:
            A string representation of the variable.
        """
        return f"Weight {self.coeff_index}, Surface {self.surface_number}"
