"""Grid Sag Variable Module

This module contains the GridSagVariable class, which represents a variable for
the sag grid of a GridSagGeometry surface in an optic.

Kramer Harrison, 2025
"""

from __future__ import annotations

from optiland.optimization.variable.base import VariableBehavior


class GridSagVariable(VariableBehavior):
    """Represents a variable for the sag grid of a GridSagGeometry surface.

    Args:
        optic (Optic): The optic object that contains the surface.
        surface_number (int): The index of the surface in the optic.
        **kwargs: Additional keyword arguments.

    """

    def get_value(self):
        """Returns the current value of the sag grid.

        Returns:
            be.ndarray: The current value of the sag grid.

        """
        return self._surfaces[self.surface_number].geometry.sag_grid

    def update_value(self, new_value):
        """Updates the value of the sag grid.

        Args:
            new_value (be.ndarray): The new value of the sag grid.

        """
        self._surfaces[self.surface_number].geometry.sag_grid = new_value

    def __str__(self):
        """Return a string representation of the variable.

        Returns:
            str: A string representation of the variable.

        """
        return f"Grid Sag, Surface {self.surface_number}"
