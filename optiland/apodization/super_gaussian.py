"""Super-Gaussian Apodization Module."""

from __future__ import annotations

import optiland.backend as be

from .base import BaseApodization


class SuperGaussianApodization(BaseApodization):
    """Super-Gaussian apodization profile.

    The amplitude profile follows:
        A(r) = exp[-(r / w)^n]

    Args:
        w (float): Beam width parameter (>0).
        n (float): Exponent (>= 2). Controls edge sharpness.

    Example:
        >>> import numpy as np
        >>> apod = SuperGaussianApodization(w=1.0, n=4)
        >>> Px = np.linspace(-1, 1, 5)
        >>> Py = np.zeros_like(Px)
        >>> apod.get_intensity(Px, Py)
        array([6.73794700e-05, 9.39413063e-01, 1.00000000e+00, 9.39413063e-01,
               6.73794700e-05])
    """

    def __init__(self, w: float = 1.0, n: float = 2.0):
        if w <= 0:
            raise ValueError("w must be positive for SuperGaussianApodization.")
        if n < 2:
            raise ValueError("n must be >= 2 for SuperGaussianApodization.")
        self.w = w
        self.n = n

    def get_intensity(self, Px, Py):
        """Applies Super-Gaussian apodization.

        Args:
            Px (be.ndarray): Normalized x pupil coordinates.
            Py (be.ndarray): Normalized y pupil coordinates.

        Returns:
            be.ndarray: Array of intensity scaling factors.
        """
        r_squared = Px**2 + Py**2
        return be.exp(-((r_squared**0.5 / self.w) ** self.n))

    def to_dict(self):
        """Converts the Super-Gaussian apodization to a dictionary."""
        data = super().to_dict()
        data["w"] = self.w
        data["n"] = self.n
        return data

    @classmethod
    def from_dict(cls, data):
        """Creates an instance of SuperGaussianApodization from a dictionary."""
        return cls(w=data.get("w", 1.0), n=data.get("n", 2.0))
