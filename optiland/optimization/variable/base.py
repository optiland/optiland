"""Base Variable Module

This module contains the base class for a variable in an optic system. The
VariableBehavior class is an abstract class that represents the behavior of a
variable in an optic system. It is used as a base class for all variables in
the optimization process.

Kramer Harrison, 2024
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from optiland.optimization.scaling.identity import IdentityScaler

if TYPE_CHECKING:
    from optiland.optimization.scaling.base import Scaler


class VariableBehavior(ABC):
    """Represents the behavior of a variable in an optic system.

    Args:
        optic (Optic): The optic system to which the variable belongs.
        surface_number (int): The surface number of the variable.
        scaler (Scaler): The scaler to use for the variable. Defaults to
            IdentityScaler().
        **kwargs: Additional keyword arguments.

    Attributes:
        optic (Optic): The optic system to which the variable belongs.
        _surfaces (SurfaceGroup): The group of surfaces in the optic system.
        surface_number (int): The surface number of the variable.
        scaler (Scaler): The scaler to use for the variable.

    """

    def __init__(self, optic, surface_number, scaler: Scaler = None, **kwargs):
        self.optic = optic
        self._surfaces = self.optic.surfaces
        self.surface_number = surface_number
        if scaler is None:
            self.scaler = IdentityScaler()
        else:
            self.scaler = scaler

    @abstractmethod
    def get_value(self):
        """Get the value of the variable.

        Returns:
            The value of the variable.

        """
        # pragma: no cover

    @abstractmethod
    def update_value(self, new_value):
        """Update the value of the variable.

        Args:
            new_value: The new value of the variable.

        """
        # pragma: no cover

    def scale(self, value):
        """Scale the value of the variable for improved optimization performance.

        Args:
            value: The value to scale

        """
        return self.scaler.scale(value)

    def inverse_scale(self, scaled_value):
        """Inverse scale the value of the variable.

        Args:
            scaled_value: The scaled value to inverse scale

        """
        return self.scaler.inverse_scale(scaled_value)
