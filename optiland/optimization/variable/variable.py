"""Variable Module

This module contains the Variable class for defining variables in an optical
system for optimization. The Variable class serves as a wrapper
around specific variable behaviors defined in separate modules.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.optimization.variable.asphere_coeff import AsphereCoeffVariable
from optiland.optimization.variable.chebyshev_coeff import ChebyshevCoeffVariable
from optiland.optimization.variable.conic import ConicVariable
from optiland.optimization.variable.decenter import DecenterVariable
from optiland.optimization.variable.forbes_coeff import (
    ForbesQ2dCoeffVariable,
    ForbesQNormalSlopeCoeffVariable,
)
from optiland.optimization.variable.index import IndexVariable
from optiland.optimization.variable.material import MaterialVariable
from optiland.optimization.variable.norm_radius import NormalizationRadiusVariable
from optiland.optimization.variable.nurbs import (
    NurbsPointsVariable,
    NurbsWeightsVariable,
)
from optiland.optimization.variable.polynomial_coeff import PolynomialCoeffVariable
from optiland.optimization.variable.radius import RadiusVariable
from optiland.optimization.variable.reciprocal_radius import ReciprocalRadiusVariable
from optiland.optimization.variable.thickness import ThicknessVariable
from optiland.optimization.variable.tilt import TiltVariable
from optiland.optimization.variable.zernike_coeff import ZernikeCoeffVariable

if TYPE_CHECKING:
    from optiland.optimization.scaling.base import Scaler


class Variable:
    """Represents a general variable in an optical system for optimization.

    This class serves as a backend-agnostic abstraction for variables used in
    optical system optimization. It acts as a wrapper around specific variable
    behaviors defined in separate modules, and can be used with multiple optimization
    backends.
    Args:
        optic (OpticalSystem): The optical system to which the variable
            belongs.
        type (str): The type of the variable.
        min_val (float or None): The minimum value allowed for the variable.
            Defaults to None.
        max_val (float or None): The maximum value allowed for the variable.
            Defaults to None.
        scaler (Scaler): The scaler to use for the variable. Defaults to
            None, which will use the default scaler for the variable type.
        **kwargs: Additional keyword arguments to be stored as attributes of
            the variable.

    Properties:
        value: The current value of the variable.
        bounds: The bounds of the variable.

    Methods:
        update(new_value): Updates the variable to a new value.

    Raises:
        ValueError: If an invalid variable type is provided.

    """

    def __init__(
        self,
        optic,
        type_name,
        min_val=None,
        max_val=None,
        scaler: Scaler = None,
        **kwargs,
    ):
        self.optic = optic
        self.type = type_name
        self.min_val = min_val
        self.max_val = max_val
        self.scaler = scaler
        self.kwargs = kwargs

        for key, value in kwargs.items():
            if key in self.allowed_attributes():
                setattr(self, key, value)
            else:
                print(f"Warning: {key} is not a recognized attribute")

        self.variable = self._get_variable()
        self.initial_value = self.value

    @staticmethod
    def allowed_attributes():
        """This method returns a set of strings that are the names of allowed
        attributes.
        """
        return {
            "surface_number",
            "coeff_number",
            "wavelength",
            "coeff_index",
            "axis",
            "glass_selection",
        }

    def _get_variable(self):
        """Get the variable.

        Returns:
            The behavior of the variable, or None if an error occurs.

        """
        behavior_kwargs = {
            "type_name": self.type,
            "optic": self.optic,
            "scaler": self.scaler,
            **self.kwargs,
        }

        variable_types = {
            "radius": RadiusVariable,
            "conic": ConicVariable,
            "thickness": ThicknessVariable,
            "tilt": TiltVariable,
            "decenter": DecenterVariable,
            "index": IndexVariable,
            "material": MaterialVariable,
            "asphere_coeff": AsphereCoeffVariable,
            "polynomial_coeff": PolynomialCoeffVariable,
            "chebyshev_coeff": ChebyshevCoeffVariable,
            "zernike_coeff": ZernikeCoeffVariable,
            "reciprocal_radius": ReciprocalRadiusVariable,
            "forbes_qbfs_coeff": ForbesQNormalSlopeCoeffVariable,  # canonical
            "forbes_qnormalslope_coeff": ForbesQNormalSlopeCoeffVariable,
            "forbes_q2d_coeff": ForbesQ2dCoeffVariable,
            "norm_radius": NormalizationRadiusVariable,
            "nurbs_control_point": NurbsPointsVariable,
            "nurbs_weight": NurbsWeightsVariable,
        }

        variable_class = variable_types.get(self.type)

        # Instantiate the class if it exists
        if variable_class:
            return variable_class(**behavior_kwargs)
        raise ValueError(f'Invalid variable type "{self.type}"')

    @property
    def value(self):
        """Return the value of the variable.

        Returns:
            float: The value of the variable.

        Raises:
            ValueError: If the variable type is invalid.

        """
        raw_value = self.variable.get_value()
        return self.variable.scale(raw_value)

    @property
    def bounds(self):
        """Returns the bounds of the variable as a tuple.

        Returns:
            tuple: the bounds of the variable

        """
        return self.variable.scaler.transform_bounds(self.min_val, self.max_val)

    def update(self, new_value):
        """Update variable to a new value.

        Args:
            new_value (float): The new value with which to update the variable.

        Raises:
            ValueError: If the variable type is invalid.

        """
        import optiland.backend as be

        if isinstance(new_value, list | tuple):
            new_value = new_value[0]

        unscaled_value = self.variable.inverse_scale(new_value)

        if hasattr(unscaled_value, "ndim") and unscaled_value.ndim > 0:
            unscaled_value = be.ravel(unscaled_value)[0]

        self.variable.update_value(unscaled_value)

    def reset(self):
        """Reset the variable to its initial value."""
        self.update(self.initial_value)

    def __str__(self):
        """Return a string representation of the variable.

        Returns:
            str: A string representation of the variable.

        """
        return str(self.variable)
