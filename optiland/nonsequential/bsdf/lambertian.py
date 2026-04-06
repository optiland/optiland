"""Lambertian BSDF for Non-Sequential Raytracing.

Cosine-weighted hemispherical scatter (diffuse surface).

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np

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
        """
        self.reflectance_value = float(reflectance_value)

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

        Args:
            num_rays: Number of rays.
            incident_dirs: Incident directions, shape (N, 3). Unused for
                Lambertian scatter but kept for API consistency.
            normals: Surface normals, shape (N, 3), pointing toward ray side.
            wavelengths: Wavelengths [nm], shape (N,).
            rng: NumPy random generator.

        Returns:
            (scattered_dirs, flux_weights) with flux_weights = reflectance.
        """
        if rng is None:
            rng = np.random.default_rng()
        xp = _get_xp(incident_dirs)

        # Sample cosine-weighted hemisphere in local frame
        # Using rejection sampling of unit sphere + normalisation
        r1 = np.asarray(rng.random(num_rays), dtype=np.float64)
        r2 = np.asarray(rng.random(num_rays), dtype=np.float64)

        # Malley's method
        phi = 2.0 * np.pi * r1
        cos_theta = np.sqrt(r2)
        sin_theta = np.sqrt(1.0 - r2)

        # Local frame: z = normal
        lx = sin_theta * np.cos(phi)
        ly = sin_theta * np.sin(phi)
        lz = cos_theta

        # Build orthonormal basis (n, t, b) around the normal
        n = np.asarray(_to_numpy(normals), dtype=np.float64)
        t, b = _orthonormal_basis(n)

        # Transform to global frame
        scattered = lx[:, None] * t + ly[:, None] * b + lz[:, None] * n

        # Normalize
        norms = (scattered * scattered).sum(axis=1, keepdims=True) ** 0.5
        scattered = scattered / norms

        if xp is not np:
            scattered = xp.array(scattered)

        weights = xp.full(num_rays, self.reflectance_value, dtype=incident_dirs.dtype)
        return scattered, weights

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
            wavelengths: Wavelengths [nm], shape (N,).

        Returns:
            Array of reflectance_value, shape (N,).
        """
        xp = _get_xp(incident_dirs)
        return xp.full(
            incident_dirs.shape[0], self.reflectance_value, dtype=incident_dirs.dtype
        )


def _orthonormal_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build two tangent vectors perpendicular to n.

    Args:
        n: Normal vectors, shape (N, 3), already normalised.

    Returns:
        Pair of tangent vectors (t, b), each shape (N, 3).
    """
    # Frisvad / Duff et al. method
    N = n.shape[0]
    t = np.empty_like(n)
    b = np.empty_like(n)

    # For each normal, pick a non-parallel reference vector
    abs_nx = np.abs(n[:, 0])
    use_y = abs_nx > 0.9

    ref = np.zeros((N, 3), dtype=n.dtype)
    ref[~use_y, 1] = 1.0  # use Y axis
    ref[use_y, 2] = 1.0  # use Z axis when n is close to X

    t = np.cross(n, ref)
    t /= (t * t).sum(axis=1, keepdims=True) ** 0.5
    b = np.cross(n, t)
    return t, b


def _to_numpy(arr: np.ndarray) -> np.ndarray:
    """Convert cupy array to numpy if needed."""
    try:
        import cupy  # type: ignore[import]

        if isinstance(arr, cupy.ndarray):
            return cupy.asnumpy(arr)
    except ImportError:
        pass
    return arr


def _get_xp(arr: np.ndarray):
    """Return numpy or cupy depending on the array type."""
    try:
        import cupy  # type: ignore[import]

        if isinstance(arr, cupy.ndarray):
            return cupy
    except ImportError:
        pass
    return np
