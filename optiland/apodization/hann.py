"""Hann Apodization Module."""

from __future__ import annotations

import optiland.backend as be

from .base import BaseApodization


class HannApodization(BaseApodization):
    """Hann apodization profile.

    The amplitude profile follows:
        A(r) = 0.5 * (1 - cos(2 * pi * r / D)) for r < D/2
        A(r) = 0                              for r >= D/2

    This function is useful for FFT-based propagation and ensuring smooth
    transitions at the edges of the pupil.

    Args:
        D (float): The diameter of the pupil (>0).

    Example:
        >>> import numpy as np
        >>> apod = HannApodization(D=2.0)
        >>> # With D=2.0, the radius R is 1.0.
        >>> r = np.linspace(0, 1.0, 5) # r = [0.0, 0.25, 0.5, 0.75, 1.0]
        >>> apod.get_intensity(r, np.zeros_like(r))
        array([0.        , 0.14644661, 0.5       , 0.85355339, 0.        ])
    """

    def __init__(self, D: float = 2.0):
        if D <= 0:
            raise ValueError("D must be positive for HannApodization.")
        self.D = D

    def get_intensity(self, Px, Py):
        """Applies Hann apodization."""
        r = (Px**2 + Py**2) ** 0.5
        R = self.D / 2
        cos_arg = (2 * be.pi * r) / self.D
        intensity = 0.5 * (1 - be.cos(cos_arg))
        return be.where(r < R, intensity, 0.0)

    def to_dict(self):
        """Converts the Hann apodization to a dictionary."""
        data = super().to_dict()
        data["D"] = self.D
        return data

    @classmethod
    def from_dict(cls, data):
        """Creates an instance of HannApodization from a dictionary."""
        return cls(D=data.get("D", 2.0))
