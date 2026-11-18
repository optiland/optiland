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

    def release_graph(self):
        """Drop the autograd graph that the last update left in the optic.

        A Torch optimizer calls this for all variables before it writes the
        values of a new step (``VariableManager.release_graphs``). A
        variable that writes one entry of an array, which other variables
        can share (a coefficient array), must start each step from plain
        values. Otherwise the array still holds the graph of the last step:
        the next backward pass then runs into that freed graph and fails, or
        the graph grows with each step.

        The default does nothing: a variable that replaces a whole attribute
        leaves no old graph behind.
        """
        return None


def without_graph(value):
    """Return ``value`` without the autograd graph of the last step.

    A tensor that requires grad gives a detached copy with its own memory,
    so that a later in-place write cannot change a tensor that the graph of
    the last step saved. Other values hold no graph and are returned as they
    are: NumPy arrays, numbers, and tensors that do not require grad.

    Args:
        value: A tensor, an array or a number.

    Returns:
        The value without an autograd graph.
    """
    if getattr(value, "requires_grad", False):
        return value.detach().clone()
    return value


def release_attribute(owner, name):
    """Replace ``owner.<name>`` by a copy without an autograd graph.

    Only a value that holds a graph is replaced (see ``without_graph``). So
    in one step only the first variable of a shared array copies it, and a
    setter of the attribute (``ZernikePolynomialGeometry.coefficients``
    builds a new Zernike object) runs only then.

    Args:
        owner: The object that holds the array, for example a geometry.
        name (str): The name of the attribute.
    """
    value = getattr(owner, name)
    released = without_graph(value)
    if released is not value:
        setattr(owner, name, released)
