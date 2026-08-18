"""Tabulated BSDF for Non-Sequential Raytracing.

Loads scatter data from a CSV or Zemax scatter file and interpolates.

Kramer Harrison, 2026
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import optiland.backend as be
from optiland.backend.utils import to_numpy
from optiland.nonsequential.bsdf.base import BaseBSDF
from optiland.nonsequential.rng import EventSlot

if TYPE_CHECKING:
    from optiland.nonsequential.rng import NSQRng


class TabulatedBSDF(BaseBSDF):
    """BSDF loaded from tabulated data (CSV or Zemax scatter file).

    The file must contain columns: theta_i [deg], theta_s [deg], bsdf_value.
    The BSDF is assumed azimuthally symmetric (phi-independent).

    Attributes:
        path: Path to the scatter data file.
        transmissive_fraction: Probability in [0, 1] that a given scatter
            event samples the transmissive hemisphere (the far side of the
            surface) instead of the reflective one. Defaults to 0.0:
            a purely reflective scatter, identical to this class's
            behaviour before D-5. The tabulated data itself is treated as
            hemisphere-relative (``theta_s`` measured from whichever normal
            the draw lands on), not as a combined BRDF+BTDF table.
    """

    def __init__(self, path: str | Path, transmissive_fraction: float = 0.0) -> None:
        """Load tabulated BSDF from file.

        Args:
            path: Path to CSV file with columns [theta_i, theta_s, bsdf].
            transmissive_fraction: Probability in [0, 1] that a scatter
                event lands in the transmissive hemisphere rather than the
                reflective one.
        """
        self.path = Path(path)
        self.transmissive_fraction = float(transmissive_fraction)
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
        rng: NSQRng,
        ray_id: np.ndarray,
        bounce: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample scattered directions from the tabulated BSDF.

        Uses importance sampling via Lambertian hemisphere + BSDF weighting.
        A per-ray draw against :attr:`transmissive_fraction` picks
        whether that hemisphere is centred on ``normals`` (reflective) or
        ``-normals`` (transmissive); ``theta_i``/``theta_s`` are measured
        from whichever normal the ray's draw landed on. Sampling is detached
        (keyed PCG32).

        Args:
            num_rays: Number of rays.
            incident_dirs: Incident directions, shape (N, 3).
            normals: Surface normals, shape (N, 3).
            wavelengths: Wavelengths [µm], shape (N,).
            rng: Keyed PCG32 RNG.
            ray_id: Per-ray identifiers, shape (N,).
            bounce: Per-ray bounce/step index, shape (N,).

        Returns:
            (scattered_dirs, flux_weights, transmitted).
        """
        n_np = np.asarray(to_numpy(normals), dtype=np.float64)
        d_np = np.asarray(to_numpy(incident_dirs), dtype=np.float64)

        if self.transmissive_fraction > 0.0:
            u_lobe = rng.uniform(ray_id, bounce, EventSlot.BSDF_LOBE_BRANCH)
            transmitted_np = u_lobe < self.transmissive_fraction
            hemisphere_np = np.where(transmitted_np[:, None], -n_np, n_np)
        else:
            transmitted_np = np.zeros(n_np.shape[0], dtype=bool)
            hemisphere_np = n_np

        # Compute angle of incidence relative to the chosen hemisphere.
        cos_i = np.clip((-d_np * hemisphere_np).sum(axis=1), 0.0, 1.0)
        theta_i = np.degrees(np.arccos(cos_i))

        from optiland.nonsequential.bsdf.lambertian import (  # noqa: PLC0415
            _orthonormal_basis,
        )

        r1 = rng.uniform(ray_id, bounce, EventSlot.BSDF_U1)
        r2 = rng.uniform(ray_id, bounce, EventSlot.BSDF_U2)
        phi = 2.0 * np.pi * r1
        cos_theta = np.sqrt(r2)
        sin_theta = np.sqrt(1.0 - r2)
        theta_s = np.degrees(np.arccos(cos_theta))

        lx = sin_theta * np.cos(phi)
        ly = sin_theta * np.sin(phi)
        lz = cos_theta

        t_vec, b_vec = _orthonormal_basis(hemisphere_np)
        scattered = (
            lx[:, None] * t_vec + ly[:, None] * b_vec + lz[:, None] * hemisphere_np
        )
        norms = (scattered * scattered).sum(axis=1, keepdims=True) ** 0.5
        scattered = scattered / norms

        # Evaluate BSDF at (theta_i, theta_s) pairs
        query = np.column_stack([theta_i, theta_s])
        bsdf_vals = self._interp(query)
        flux_weights = np.clip(np.pi * bsdf_vals, 0.0, 1.0)

        return (
            be.array(scattered.astype(np.float64)),
            be.array(flux_weights),
            be.array(transmitted_np),
        )

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
