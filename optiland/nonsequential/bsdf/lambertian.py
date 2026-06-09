"""Lambertian BSDF for Non-Sequential Raytracing.

Cosine-weighted hemispherical scatter (diffuse surface).

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np

import optiland.backend as be
from optiland.backend.utils import to_numpy
from optiland.nonsequential.bsdf.base import BaseBSDF


class LambertianBSDF(BaseBSDF):
    """Cosine-weighted Lambertian diffuse scatter.

    Attributes:
        reflectance_value: Hemispherical diffuse reflectance in [0, 1].
    """

    def __init__(self, reflectance_value: float = 1.0) -> None:
        """Initialize LambertianBSDF.

        Args:
            reflectance_value: Total hemispherical reflectance in [0, 1].
                May be a torch Tensor with requires_grad=True for autograd.
        """
        # Intentionally not float()-cast so torch tensors remain attached.
        self.reflectance_value = reflectance_value

    def sample(
        self,
        num_rays: int,
        incident_dirs: np.ndarray,
        normals: np.ndarray,
        wavelengths: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample cosine-weighted hemisphere directions around normals.

        Uses Malley's method: sample uniform disk, project to hemisphere.
        Sampling is detached (numpy RNG); weights are plain scalars.

        Args:
            num_rays: Number of rays.
            incident_dirs: Incident directions, shape (N, 3).
            normals: Surface normals, shape (N, 3), pointing toward ray side.
            wavelengths: Wavelengths [µm], shape (N,).
            rng: NumPy random generator.

        Returns:
            (scattered_dirs, flux_weights) with flux_weights = reflectance.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Sampling is inherently stochastic/detached -- use numpy throughout
        normals_np = np.asarray(to_numpy(normals), dtype=np.float64)

        r1 = rng.random(num_rays)
        r2 = rng.random(num_rays)

        # Malley's method
        phi = 2.0 * np.pi * r1
        cos_theta = np.sqrt(r2)
        sin_theta = np.sqrt(1.0 - r2)

        lx = sin_theta * np.cos(phi)
        ly = sin_theta * np.sin(phi)
        lz = cos_theta

        t_vec, b_vec = _orthonormal_basis(normals_np)

        scattered = lx[:, None] * t_vec + ly[:, None] * b_vec + lz[:, None] * normals_np
        norms = (scattered * scattered).sum(axis=1, keepdims=True) ** 0.5
        scattered = scattered / norms

        # Convert back to backend array type (detached; no grad required)
        scattered_be = be.array(scattered.astype(np.float64))
        # be.ones * reflectance_value preserves the autograd graph when
        # reflectance_value is a torch Tensor with requires_grad=True.
        weights_be = be.ones(num_rays) * self.reflectance_value

        return scattered_be, weights_be

    def reflectance(
        self,
        incident_dirs: np.ndarray,
        normals: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """Return total hemispherical reflectance.

        Args:
            incident_dirs: Incident directions, shape (N, 3).
            normals: Surface normals, shape (N, 3).
            wavelengths: Wavelengths [µm], shape (N,).

        Returns:
            Array of reflectance_value, shape (N,).
        """
        return be.ones(incident_dirs.shape[0]) * self.reflectance_value


def _orthonormal_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build two tangent vectors perpendicular to n (numpy, for detached sampling).

    Args:
        n: Normal vectors, shape (N, 3), already normalised.

    Returns:
        Pair of tangent vectors (t, b), each shape (N, 3).
    """
    N = n.shape[0]

    abs_nx = np.abs(n[:, 0])
    use_y = abs_nx > 0.9

    ref = np.zeros((N, 3), dtype=n.dtype)
    ref[~use_y, 1] = 1.0
    ref[use_y, 2] = 1.0

    t_vec = np.cross(n, ref)
    t_vec /= (t_vec * t_vec).sum(axis=1, keepdims=True) ** 0.5
    b_vec = np.cross(n, t_vec)
    return t_vec, b_vec
