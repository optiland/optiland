"""This module provides variable handling for normalization radius parameters.

The normalization radius is used in certain optical surface representations
(like Forbes polynomials) to control the scaling of the surface coordinates.
It contains the `NormalizationRadiusVariable` class for use in optimization problems.


Manuel Fragata Mendes, August 2025
"""

from __future__ import annotations

from optiland.optimization.scaling.identity import IdentityScaler
from optiland.optimization.variable.base import VariableBehavior


class NormalizationRadiusVariable(VariableBehavior):
    """
    Represents a variable for the normalization radius of a surface.
    """

    def __init__(self, optic, surface_number, scaler=None, **kwargs):
        if scaler is None:
            scaler = IdentityScaler()
        super().__init__(optic, surface_number, scaler=scaler, **kwargs)
        self.optic.surfaces[self.surface_number].is_norm_radius_variable = True

    def get_value(self):
        """Returns the current value of the normalization radius."""
        surf = self._surfaces[self.surface_number]
        if hasattr(surf.geometry, "norm_radius"):
            return surf.geometry.norm_radius
        else:
            raise AttributeError(
                f"Geometry for surface {self.surface_number} "
                "does not have a 'norm_radius' attribute."
            )

    def update_value(self, new_value):
        """Updates the value of the normalization radius."""
        self.optic.updater.set_norm_radius(new_value, self.surface_number)

    def __str__(self):
        return f"Normalization Radius, Surface {self.surface_number}"
