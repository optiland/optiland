"""Harvey-Shack (ABg) BSDF for Non-Sequential Raytracing.

Models micro-roughness scatter from optical surfaces.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np

import optiland.backend as be
from optiland.backend.utils import to_numpy
from optiland.nonsequential.bsdf.base import BaseBSDF


class HarveyShackBSDF(BaseBSDF):
    """Harvey-Shack / ABg scatter model for surface micro-roughness.

    The ABg model is a simplified form of the Harvey-Shack theory:

        BSDF(beta - beta0) = b0 / (1 + |beta - beta0| / l0)^s

    where beta and beta0 are direction cosines of the scattered and specular
    directions, b0 is the scatter level at beta=beta0, l0 is the break
    frequency, and s is the roll-off slope.

    Attributes:
        b0: Scatter amplitude at zero angle [sr^-1].
        l0: Break-point spatial frequency (dimensionless direction cosine).
        s: Power-law roll-off slope (positive).
    """

    def __init__(self, b0: float, l0: float, s: float) -> None:
        """Initialize HarveyShackBSDF.

        Args:
            b0: Scatter amplitude at zero angle [sr^-1].
            l0: Break-point spatial frequency in direction-cosine space.
            s: Power-law roll-off exponent (positive).
        """
        self.b0 = float(b0)
        self.l0 = float(l0)
        self.s = float(s)

    def sample(
        self,
        num_rays: int,
        incident_dirs: np.ndarray,
        normals: np.ndarray,
        wavelengths: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample scattered directions using ABg model via rejection sampling.

        Sampling is detached (numpy). Weights are scalar scalings.

        Args:
            num_rays: Number of rays.
            incident_dirs: Incident directions, shape (N, 3).
            normals: Surface normals, shape (N, 3).
            wavelengths: Wavelengths [µm], shape (N,).
            rng: NumPy random generator.

        Returns:
            (scattered_dirs, flux_weights).
        """
        if rng is None:
            rng = np.random.default_rng()

        n_np = np.asarray(to_numpy(normals), dtype=np.float64)
        d_np = np.asarray(to_numpy(incident_dirs), dtype=np.float64)

        # Specular direction: d - 2(d.n)n
        cos_i = (d_np * n_np).sum(axis=1, keepdims=True)
        d_spec = d_np - 2.0 * cos_i * n_np
        d_spec /= (d_spec * d_spec).sum(axis=1, keepdims=True) ** 0.5

        from optiland.nonsequential.bsdf.lambertian import (  # noqa: PLC0415
            _orthonormal_basis,
        )

        r1 = rng.random(num_rays)
        r2 = rng.random(num_rays)
        phi = 2.0 * np.pi * r1
        cos_theta = np.sqrt(r2)
        sin_theta = np.sqrt(1.0 - r2)

        lx = sin_theta * np.cos(phi)
        ly = sin_theta * np.sin(phi)
        lz = cos_theta

        t_vec, b_vec = _orthonormal_basis(n_np)
        scattered = lx[:, None] * t_vec + ly[:, None] * b_vec + lz[:, None] * n_np
        norms = (scattered * scattered).sum(axis=1, keepdims=True) ** 0.5
        scattered = scattered / norms

        # Compute BSDF weight: ABg(beta - beta0) / Lambertian(theta)
        delta_beta = scattered - d_spec
        beta_mag = (delta_beta * delta_beta).sum(axis=1) ** 0.5
        bsdf_val = self.b0 / (1.0 + (beta_mag / self.l0) ** self.s)
        flux_weights = np.clip(np.pi * bsdf_val / np.maximum(lz, 1e-6), 0.0, 1.0)

        return be.array(scattered.astype(np.float64)), be.array(flux_weights)

    def reflectance(
        self,
        incident_dirs: np.ndarray,
        normals: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """Approximate total hemispherical reflectance via integration.

        Args:
            incident_dirs: Incident directions, shape (N, 3).
            normals: Surface normals, shape (N, 3).
            wavelengths: Wavelengths [µm], shape (N,).

        Returns:
            Approximate reflectance values, shape (N,).
        """
        n = incident_dirs.shape[0]
        estimate = min(float(np.pi * self.b0 * self.l0), 1.0)
        return be.full(n, estimate)
