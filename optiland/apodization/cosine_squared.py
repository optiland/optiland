"""Cosine-Squared Apodization Module."""

from __future__ import annotations

import optiland.backend as be

from .base import BaseApodization


class CosineSquaredApodization(BaseApodization):
    """Cosine-squared apodization profile.

    The amplitude profile follows:
        A(r) = cos^2(pi * r / (2 * R)) for r < R
        A(r) = 0                         for r >= R

    This profile provides a smooth taper to zero at the edge of the pupil,
    which can help minimize diffraction effects.

    Args:
        R (float): The radius of the pupil (>0).

    Example:
        >>> import numpy as np
        >>> apod = CosineSquaredApodization(R=1.0)
        >>> Px = np.linspace(0, 1.5, 5)
        >>> Py = np.zeros_like(Px)
        >>> apod.get_intensity(Px, Py)
        array([1.        , 0.9045085 , 0.5       , 0.0954915 , 0.        ])
    """

    def __init__(self, R: float = 1.0):
        if R <= 0:
            raise ValueError("R must be positive for CosineSquaredApodization.")
        self.R = R

    def get_intensity(self, Px, Py):
        """Applies cosine-squared apodization."""
        r = (Px**2 + Py**2) ** 0.5
        cos_arg = (be.pi * r) / (2 * self.R)
        intensity = be.cos(cos_arg) ** 2
        return be.where(r < self.R, intensity, 0.0)

    def to_dict(self):
        """Converts the Cosine-squared apodization to a dictionary."""
        data = super().to_dict()
        data["R"] = self.R
        return data

    @classmethod
    def from_dict(cls, data):
        """Creates an instance of CosineSquaredApodization from a dictionary."""
        return cls(R=data.get("R", 1.0))
