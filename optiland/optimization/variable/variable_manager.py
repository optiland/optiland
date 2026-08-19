"""Manages the collection of variables used during optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .variable import Variable

if TYPE_CHECKING:
    from ..scaling.base import Scaler


class VariableManager:
    """Manages a list of variables in a merit function.

    Attributes:
        variables (list): A list of Variable objects.

    Methods:
        add(optic, variable_type, **kwargs): Add a variable to the merit
            function.
        clear(): Clear all variables from the merit function.

    """

    def __init__(self):
        self.variables = []

    def add(self, optic, variable_type, scaler: Scaler = None, **kwargs):
        """Add a variable to the merit function

        Args:
            optic (OpticalSystem): The optical system to which the variable
                belongs.
            variable_type (str): The type of the variable.
            scaler (Scaler, optional): The scaler to use for the variable.
                Defaults to None, which will use the default scaler for the
                variable type.
            **kwargs: Additional keyword arguments to be passed to the Variable
                constructor.

        """
        self.variables.append(Variable(optic, variable_type, scaler=scaler, **kwargs))

        # Check for and remove any pickups that control this variable
        surface_number = kwargs.get("surface_number")
        if surface_number is not None:
            pickups_to_remove = []
            for pickup in optic.pickups.pickups:
                if (
                    pickup.target_surface_idx == surface_number
                    and pickup.attr_type == variable_type
                ):
                    pickups_to_remove.append(pickup)

            for pickup in pickups_to_remove:
                optic.pickups.pickups.remove(pickup)

    def clear(self):
        """Clear all variables from the merit function"""
        self.variables = []

    def __iter__(self):
        """Return the iterator object itself"""
        self._index = 0
        return self

    def __next__(self):
        """Return the next variable in the list"""
        if self._index < len(self.variables):
            result = self.variables[self._index]
            self._index += 1
            return result
        raise StopIteration

    def __len__(self):
        """Return the number of variables in the list"""
        return len(self.variables)

    def __getitem__(self, index):
        """Return the variable at the specified index"""
        return self.variables[index]

    def __setitem__(self, index, value):
        """Set the variable at the specified index"""
        if not isinstance(value, Variable):
            raise ValueError("Value must be an instance of Variable")
        self.variables[index] = value

    def __delitem__(self, index):
        """Delete the variable at the specified index"""
        del self.variables[index]
