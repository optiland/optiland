"""Layer Thickness Variable Module

This module contains the LayerThicknessVariable class, which represents a variable
for the thickness of a layer in a thin film stack. The class provides methods
to get and update the layer thickness for optimization purposes.

Corentin Nannini, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.optimization.variable.base import VariableBehavior

if TYPE_CHECKING:
    from optiland.thin_film import ThinFilmStack


class LayerThicknessVariable(VariableBehavior):
    """Represents a variable for a thin film layer thickness.

    This class inherits from VariableBehavior and provides specific functionality
    for optimizing the thickness of individual layers in a ThinFilmStack.

    Args:
        stack (ThinFilmStack): The thin film stack containing the layer.
        layer_index (int): The index of the layer in the stack (0-based).
        apply_scaling (bool): Whether to apply scaling for optimization.
            Defaults to True.
        **kwargs: Additional keyword arguments passed to parent class.

    Attributes:
        stack (ThinFilmStack): The thin film stack.
        layer_index (int): The index of the target layer.
    """

    def __init__(
        self, stack: ThinFilmStack, layer_index: int, apply_scaling=True, **kwargs
    ):
        # Initialize attributes without calling parent __init__
        # since we don't have an optic object
        self.optic = None  # Not used for thin film variables
        self._surfaces = None  # Not used for thin film variables
        self.surface_number = layer_index  # For compatibility
        self.apply_scaling = apply_scaling

        # Thin film specific attributes
        self.stack = stack
        self.layer_index = layer_index

        if layer_index < 0 or layer_index >= len(stack.layers):
            raise ValueError(
                f"layer_index {layer_index} is out of range for "
                + f"stack with {len(stack.layers)} layers"
            )

    def get_value(self) -> float:
        """Returns the current thickness value of the layer in micrometers.

        Returns:
            float: The current thickness value in μm.
        """
        value = self.stack.layers[self.layer_index].thickness_um
        if self.apply_scaling:
            return self.scale(value)
        return value

    def update_value(self, new_value: float) -> None:
        """Updates the thickness value of the layer.

        Args:
            new_value (float): The new thickness value in μm (or scaled units).
        """
        if self.apply_scaling:
            new_value = self.inverse_scale(new_value)

        # Ensure positive thickness - clamp to minimum instead of raising exception
        if new_value <= 0:
            new_value = 0.000001  # Force minimum thickness of 0.01 nm

        self.stack.layers[self.layer_index].update_thickness(new_value)

    def scale(self, value: float) -> float:
        """Scale the thickness value for improved optimization performance.

        Converts from micrometers to a scaled representation that's better
        conditioned for numerical optimization.

        Args:
            value (float): The thickness value in μm to scale.

        Returns:
            float: The scaled value.
        """
        # Scale: μm -> approximately unit scale
        # Typical thin film thicknesses are 0.01-1 μm, so divide by 0.1
        return value / 0.1 - 1.0

    def inverse_scale(self, scaled_value: float) -> float:
        """Inverse scale the thickness value.

        Converts from scaled representation back to micrometers.

        Args:
            scaled_value (float): The scaled value to inverse scale.

        Returns:
            float: The thickness value in μm.
        """
        return (scaled_value + 1.0) * 0.1

    @property
    def thickness_nm(self) -> float:
        """Get the current thickness in nanometers.

        Returns:
            float: The current thickness in nm.
        """
        return self.stack.layers[self.layer_index].thickness_um * 1000.0

    def __repr__(self) -> str:
        """String representation of the variable."""
        thickness_nm = self.thickness_nm
        return (
            f"LayerThicknessVariable(layer_index={self.layer_index}"
            + f", thickness={thickness_nm:.1f} nm)"
        )
