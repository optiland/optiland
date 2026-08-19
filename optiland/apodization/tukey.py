"""Tukey Apodization Module."""

from __future__ import annotations

import optiland.backend as be

from .base import BaseApodization


class TukeyApodization(BaseApodization):
    """Tukey (tapered cosine) apodization profile.

    The Tukey window is flat in the center and tapers to zero at the edges
    using a cosine function.

    The amplitude profile is defined as:
        A(r) = 1.0                                for 0 <= r <= R(1 - alpha/2)
        A(r) = 0.5 * [1 + cos(pi * (r - R(1 - alpha/2)) / (R * alpha / 2))]
                                                  for R(1 - alpha/2) < r < R
        A(r) = 0.0                                for r >= R

    Args:
        R (float): The outer radius of the apodization (>0).
        alpha (float): The taper fraction, between 0 and 1.
                       alpha=0 is a uniform (top-hat) profile.
                       alpha=1 is a Hann-like profile.

    Example:
        >>> import numpy as np
        >>> apod = TukeyApodization(R=1.0, alpha=0.5)
        >>> r = np.linspace(0, 1, 5)
        >>> apod.get_intensity(r, np.zeros_like(r))
        array([1.        , 1.        , 1.        , 0.9045085 , 0.5       ])
    """

    def __init__(self, R: float = 1.0, alpha: float = 0.5):
        if R <= 0:
            raise ValueError("R must be positive for TukeyApodization.")
        if not (0 <= alpha <= 1):
            raise ValueError("alpha must be between 0 and 1 for TukeyApodization.")
        self.R = R
        self.alpha = alpha

    def get_intensity(self, Px, Py):
        """Applies Tukey apodization."""
        r = (Px**2 + Py**2) ** 0.5

        # Define the boundaries of the taper region
        flat_region_end = self.R * (1 - self.alpha / 2)

        # Calculate intensity in the taper region
        cos_arg = be.pi * (r - flat_region_end) / (self.R * self.alpha / 2)
        taper_intensity = 0.5 * (1 + be.cos(cos_arg))

        # Apply conditions
        # Condition 1: Inside flat region
        cond1 = r <= flat_region_end
        # Condition 2: Inside taper region
        cond2 = (r > flat_region_end) & (r < self.R)

        # Build the result using be.where
        intensity = be.where(cond1, 1.0, 0.0)
        intensity = be.where(cond2, taper_intensity, intensity)

        return intensity

    def to_dict(self):
        """Converts the Tukey apodization to a dictionary."""
        data = super().to_dict()
        data["R"] = self.R
        data["alpha"] = self.alpha
        return data

    @classmethod
    def from_dict(cls, data):
        """Creates an instance of TukeyApodization from a dictionary."""
        return cls(R=data.get("R", 1.0), alpha=data.get("alpha", 0.5))
