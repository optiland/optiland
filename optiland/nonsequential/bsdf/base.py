"""Base BSDF for Non-Sequential Raytracing.

Kramer Harrison, 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class BaseBSDF(ABC):
    """Abstract base class for bidirectional scattering distribution functions.

    All BSDF implementations must support vectorized operation over N rays.
    Array operations must be compatible with both NumPy and CuPy arrays.
    """

    @abstractmethod
    def sample(
        self,
        num_rays: int,
        incident_dirs: np.ndarray,
        normals: np.ndarray,
        wavelengths: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample scattered ray directions and relative flux weights.

        Args:
            num_rays: Number of rays to scatter.
            incident_dirs: Incident ray directions, shape (N, 3), unit vectors.
            normals: Surface normals at hit points, shape (N, 3), unit vectors
                pointing toward the incoming ray side.
            wavelengths: Per-ray wavelengths [nm], shape (N,).
            rng: Optional NumPy random generator for reproducibility.

        Returns:
            A tuple (scattered_dirs, flux_weights) where:
                - scattered_dirs: Scattered unit direction vectors, shape (N, 3).
                - flux_weights: Relative flux weights in [0, 1], shape (N,).
        """

    @abstractmethod
    def reflectance(
        self,
        incident_dirs: np.ndarray,
        normals: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """Total hemispherical reflectance for Russian-roulette decisions.

        Args:
            incident_dirs: Incident ray directions, shape (N, 3), unit vectors.
            normals: Surface normals at hit points, shape (N, 3), unit vectors.
            wavelengths: Per-ray wavelengths [nm], shape (N,).

        Returns:
            Total reflectance values in [0, 1], shape (N,).
        """
