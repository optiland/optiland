"""Base source and Spectrum for Non-Sequential Raytracing.

Kramer Harrison, 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.ray_bundle import NSQRayBundle


@dataclass
class Spectrum:
    """Wavelength distribution for Monte Carlo sampling.

    Attributes:
        wavelengths: Wavelength values [µm].
        weights: Relative spectral power weights (unnormalized).
    """

    wavelengths: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        """Normalize weights and build inverse-CDF for sampling."""
        self.wavelengths = np.asarray(self.wavelengths, dtype=np.float64)
        self.weights = np.asarray(self.weights, dtype=np.float64)
        if self.wavelengths.shape != self.weights.shape:
            raise ValueError("wavelengths and weights must have the same shape.")
        if np.any(self.wavelengths > 20.0):
            raise ValueError("Wavelengths must be in µm (expected range 0.1-20 µm).")
        # Build cumulative distribution for inverse-CDF sampling
        self._cdf = np.cumsum(self.weights)
        self._cdf /= self._cdf[-1]

    @classmethod
    def monochromatic(cls, wavelength: float) -> Spectrum:
        """Create a monochromatic spectrum at a single wavelength.

        Args:
            wavelength: Wavelength [µm].

        Returns:
            A Spectrum with a single wavelength.
        """
        return cls(
            wavelengths=np.array([wavelength]),
            weights=np.array([1.0]),
        )

    def sample(self, num: int, rng: np.random.Generator) -> np.ndarray:
        """Sample n wavelengths from the spectrum.

        Uses inverse-CDF (quantile) sampling.

        Args:
            num: Number of wavelengths to sample.
            rng: NumPy random generator.

        Returns:
            Sampled wavelengths [µm], shape (n,).
        """
        if len(self.wavelengths) == 1:
            return np.full(num, self.wavelengths[0])
        u = rng.random(num)
        indices = np.searchsorted(self._cdf, u)
        indices = np.clip(indices, 0, len(self.wavelengths) - 1)
        return self.wavelengths[indices]


class BaseNSQSource(ABC):
    """Abstract base class for non-sequential ray sources.

    Attributes:
        cs: Coordinate system defining source position/orientation.
        spectrum: Wavelength distribution for Monte Carlo sampling.
        total_flux: Source total flux [W] (or photons/sec).
    """

    def __init__(
        self,
        cs: CoordinateSystem,
        spectrum: Spectrum,
        total_flux: float = 1.0,
    ) -> None:
        """Initialize BaseNSQSource.

        Args:
            cs: Coordinate system for source position/orientation.
            spectrum: Wavelength distribution.
            total_flux: Total emitted flux [W].
        """
        self.cs = cs
        self.spectrum = spectrum
        # Keep as-is (float or torch.Tensor) so autograd graph is not severed
        self.total_flux = total_flux

    @abstractmethod
    def generate(self, num_rays: int, rng: np.random.Generator) -> NSQRayBundle:
        """Generate num_rays rays in global coordinates.

        Each ray carries:
          - position sampled from source geometry
          - direction sampled from source emission pattern
          - wavelength sampled from spectrum (Monte Carlo)
          - initial flux = total_flux / num_rays

        Args:
            num_rays: Number of rays to generate.
            rng: NumPy random generator.

        Returns:
            NSQRayBundle with all rays alive and flux = total_flux / num_rays.
        """
