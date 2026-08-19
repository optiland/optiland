"""Ideal Material

This module contains the IdealMaterial class, which represents an ideal
material with a fixed refractive index and extinction coefficient for all
wavelengths.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.materials.base import BaseMaterial
from optiland.propagation.base import BasePropagationModel

if TYPE_CHECKING:
    from optiland.propagation.base import BasePropagationModel


class IdealMaterial(BaseMaterial):
    """Represents an ideal material with a fixed refractive index and extinction
    coefficient for all wavelengths.

    Attributes:
        index (float): The refractive index of the material.
        absorp (float): The extinction coefficient of the material.

    """

    def __init__(
        self,
        n: float,
        k: float = 0,
        propagation_model: BasePropagationModel | None = None,
    ):
        super().__init__(propagation_model)
        self.index = be.array([n])
        self.absorp = be.array([k])

    def _calculate_n(self, wavelength, **kwargs):
        """Returns the refractive index of the material.

        Args:
            wavelength (float or be.ndarray): The wavelength(s) of light in microns.
                This argument is not used by this material model as the index is
                constant.

        Returns:
            float or be.ndarray: The refractive index of the material. Returns a
            scalar if wavelength is scalar, otherwise an array of the same shape
            as wavelength, filled with the constant refractive index.
        """
        if be.is_array_like(wavelength) and be.size(wavelength) > 1:
            return be.full_like(wavelength, self.index[0])
        return self.index[0]

    def _calculate_k(self, wavelength, **kwargs):
        """Returns the extinction coefficient of the material.

        Args:
            wavelength (float or be.ndarray): The wavelength(s) of light in microns.
                This argument is not used by this material model as the value is
                constant.

        Returns:
            float or be.ndarray: The extinction coefficient of the material. Returns a
            scalar if wavelength is scalar, otherwise an array of the same shape
            as wavelength, filled with the constant extinction coefficient.
        """
        if be.is_array_like(wavelength) and be.size(wavelength) > 1:
            return be.full_like(wavelength, self.absorp[0])
        return self.absorp[0]

    def to_dict(self):
        """Returns a dictionary representation of the material.

        Returns:
            dict: A dictionary representation of the material.

        """
        material_dict = super().to_dict()
        material_dict.update({"index": self.index.item(), "absorp": self.absorp.item()})
        return material_dict

    @classmethod
    def from_dict(cls, data):
        """Creates a material from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the material.

        Returns:
            IdealMaterial: The material.

        """
        return cls(data["index"], data.get("absorp", 0))
