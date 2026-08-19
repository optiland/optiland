"""Polynomial Apodization Module."""

from __future__ import annotations

import optiland.backend as be

from .base import BaseApodization


class PolynomialApodization(BaseApodization):
    """Polynomial apodization profile.

    The amplitude profile follows:
        A(r) = (1 - (r/R)^2)^p for r < R
        A(r) = 0               for r >= R

    This profile is commonly used in telescope pupil apodization.

    Args:
        R (float): The radius of the pupil (>0).
        p (float): The power of the polynomial (>=0).

    Example:
        >>> import numpy as np
        >>> apod = PolynomialApodization(R=1.0, p=2)
        >>> r = np.linspace(0, 1.5, 5)
        >>> apod.get_intensity(r, np.zeros_like(r))
        array([1.        , 0.9375    , 0.75      , 0.4375    , 0.        ])
    """

    def __init__(self, R: float = 1.0, p: float = 1.0):
        if R <= 0:
            raise ValueError("R must be positive for PolynomialApodization.")
        if p < 0:
            raise ValueError("p must be non-negative for PolynomialApodization.")
        self.R = R
        self.p = p

    def get_intensity(self, Px, Py):
        """Applies polynomial apodization."""
        r = (Px**2 + Py**2) ** 0.5
        r_norm_sq = (r / self.R) ** 2
        intensity = (1 - r_norm_sq) ** self.p
        return be.where(r < self.R, intensity, 0.0)

    def to_dict(self):
        """Converts the Polynomial apodization to a dictionary."""
        data = super().to_dict()
        data["R"] = self.R
        data["p"] = self.p
        return data

    @classmethod
    def from_dict(cls, data):
        """Creates an instance of PolynomialApodization from a dictionary."""
        return cls(R=data.get("R", 1.0), p=data.get("p", 1.0))
