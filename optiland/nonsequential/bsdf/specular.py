"""Specular BRDF for Non-Sequential Raytracing.

Perfect mirror reflection (delta function BSDF). Also used for total
internal reflection.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.nonsequential.bsdf.base import BaseBSDF

if TYPE_CHECKING:
    import numpy as np


class SpecularBRDF(BaseBSDF):
    """Perfect specular reflector (mirror).

    The scattered direction is the specular reflection of the incident ray.
    Flux weight is always 1.0 (no energy loss at the surface itself).
    """

    def sample(
        self,
        num_rays: int,
        incident_dirs: np.ndarray,
        normals: np.ndarray,
        wavelengths: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample specular reflection directions.

        Args:
            num_rays: Number of rays.
            incident_dirs: Incident directions, shape (N, 3).
            normals: Surface normals, shape (N, 3).
            wavelengths: Wavelengths [µm], shape (N,).
            rng: Unused for specular BRDF.

        Returns:
            (reflected_dirs, ones) -- reflected unit vectors and unit weights.
        """
        cos_theta = (incident_dirs * normals).sum(axis=1, keepdims=True)
        reflected = incident_dirs - 2.0 * cos_theta * normals
        norms = (reflected * reflected).sum(axis=1, keepdims=True) ** 0.5
        reflected = reflected / norms
        return reflected, be.ones(num_rays, dtype=incident_dirs.dtype)

    def reflectance(
        self,
        incident_dirs: np.ndarray,
        normals: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """Return reflectance = 1.0 (perfect mirror).

        Args:
            incident_dirs: Incident directions, shape (N, 3).
            normals: Surface normals, shape (N, 3).
            wavelengths: Wavelengths [µm], shape (N,).

        Returns:
            Array of ones, shape (N,).
        """
        return be.ones(incident_dirs.shape[0], dtype=incident_dirs.dtype)
