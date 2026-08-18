"""Tabulated BSDF for Non-Sequential Raytracing.

Loads scatter data from a CSV or Zemax scatter file and interpolates.

Kramer Harrison, 2026
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import optiland.backend as be
from optiland.backend.utils import to_numpy
from optiland.nonsequential.bsdf.base import BaseBSDF


class TabulatedBSDF(BaseBSDF):
    """BSDF loaded from tabulated data (CSV or Zemax scatter file).

    The file must contain columns: theta_i [deg], theta_s [deg], bsdf_value.
    The BSDF is assumed azimuthally symmetric (phi-independent).

    Attributes:
        path: Path to the scatter data file.
    """

    def __init__(self, path: str | Path) -> None:
        """Load tabulated BSDF from file.

        Args:
            path: Path to CSV file with columns [theta_i, theta_s, bsdf].
        """
        self.path = Path(path)
        self._load(self.path)

    def _load(self, path: Path) -> None:
        """Parse and build an interpolator from the data file.

        Args:
            path: Path to the CSV data file.
        """
        data = np.loadtxt(path, delimiter=",", comments="#")
        theta_i = np.unique(data[:, 0])
        theta_s = np.unique(data[:, 1])
        bsdf_grid = data[:, 2].reshape(len(theta_i), len(theta_s))
        self._interp = RegularGridInterpolator(
            (theta_i, theta_s),
            bsdf_grid,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        self._theta_i_vals = theta_i
        self._theta_s_vals = theta_s

    def sample(
        self,
        num_rays: int,
        incident_dirs: np.ndarray,
        normals: np.ndarray,
        wavelengths: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample scattered directions from the tabulated BSDF.

        Uses importance sampling via Lambertian hemisphere + BSDF weighting.
        Sampling is detached (numpy).

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

        # Compute angle of incidence
        cos_i = np.clip((-d_np * n_np).sum(axis=1), 0.0, 1.0)
        theta_i = np.degrees(np.arccos(cos_i))

        from optiland.nonsequential.bsdf.lambertian import (  # noqa: PLC0415
            _orthonormal_basis,
        )

        r1 = rng.random(num_rays)
        r2 = rng.random(num_rays)
        phi = 2.0 * np.pi * r1
        cos_theta = np.sqrt(r2)
        sin_theta = np.sqrt(1.0 - r2)
        theta_s = np.degrees(np.arccos(cos_theta))

        lx = sin_theta * np.cos(phi)
        ly = sin_theta * np.sin(phi)
        lz = cos_theta

        t_vec, b_vec = _orthonormal_basis(n_np)
        scattered = lx[:, None] * t_vec + ly[:, None] * b_vec + lz[:, None] * n_np
        norms = (scattered * scattered).sum(axis=1, keepdims=True) ** 0.5
        scattered = scattered / norms

        # Evaluate BSDF at (theta_i, theta_s) pairs
        query = np.column_stack([theta_i, theta_s])
        bsdf_vals = self._interp(query)
        flux_weights = np.clip(np.pi * bsdf_vals, 0.0, 1.0)

        return be.array(scattered.astype(np.float64)), be.array(flux_weights)

    def reflectance(
        self,
        incident_dirs: np.ndarray,
        normals: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """Approximate total hemispherical reflectance from tabulated data.

        Args:
            incident_dirs: Incident directions, shape (N, 3).
            normals: Surface normals, shape (N, 3).
            wavelengths: Wavelengths [µm], shape (N,).

        Returns:
            Reflectance values, shape (N,).
        """
        d_np = np.asarray(to_numpy(incident_dirs), dtype=np.float64)
        n_np = np.asarray(to_numpy(normals), dtype=np.float64)

        cos_i = np.clip((-d_np * n_np).sum(axis=1), 0.0, 1.0)
        theta_i = np.degrees(np.arccos(cos_i))

        theta_s_mid = np.mean(self._theta_s_vals)
        query = np.column_stack([theta_i, np.full_like(theta_i, theta_s_mid)])
        bsdf_vals = self._interp(query)

        refl = np.clip(np.pi * bsdf_vals, 0.0, 1.0)
        return be.array(refl.astype(np.float64))
