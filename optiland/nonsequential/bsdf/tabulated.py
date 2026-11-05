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
        self._build_sampling_tables(bsdf_grid)

    def _build_sampling_tables(self, bsdf_grid: np.ndarray) -> None:
        """Per-theta_i-row tables for importance sampling theta_s.

        For each table row k the scattered polar angle is drawn with density
        proportional to ``b_k(theta_s) cos(theta_s) sin(theta_s)``, using a
        piecewise-constant density on the ``theta_s`` grid. Stores the
        cumulative mass ``self._cdf`` (unnormalised, shape (K, M)) and the
        per-bin density ``self._bin_pdf`` (normalised per row, shape
        (K, M-1)) needed to weight each sample exactly.

        Args:
            bsdf_grid: BSDF values [1/sr], shape (len(theta_i), len(theta_s)).
        """
        ts = np.radians(self._theta_s_vals)
        integrand = np.clip(bsdf_grid, 0.0, None) * np.cos(ts) * np.sin(ts)
        integrand = np.clip(integrand, 0.0, None)  # guard theta_s > 90 deg entries
        mass = 0.5 * (integrand[:, 1:] + integrand[:, :-1]) * np.diff(ts)
        cdf = np.concatenate([np.zeros((mass.shape[0], 1)), np.cumsum(mass, 1)], 1)
        total = cdf[:, -1:]
        with np.errstate(invalid="ignore", divide="ignore"):
            self._bin_pdf = np.where(total > 0, mass / np.diff(ts) / total, 0.0)
        self._cdf = cdf

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

        A per-ray draw against :attr:`transmissive_fraction` picks whether
        the scatter hemisphere is centred on ``normals`` (reflective) or
        ``-normals`` (transmissive); ``theta_s`` is measured from the normal
        of that hemisphere. ``theta_i`` is always the angle of incidence,
        measured from the normal on the incident side (``normals`` faces the
        incoming ray), whichever hemisphere is drawn.

        ``theta_s`` is importance-sampled from the table itself: the two
        ``theta_i`` rows bracketing the actual incidence are mixed with the
        linear-interpolation weights, and within a row ``theta_s`` follows
        ``b cos(theta_s) sin(theta_s)``. Each ray carries the exact weight
        ``2 pi b(theta_i, theta_s) cos sin / pdf``, so the scattered flux
        equals the table's integrated scatter (no clipping of ``pi * b``).
        Rays whose ``theta_i`` lies outside the tabulated range get weight 0,
        as with the interpolator's ``fill_value``. Sampling is detached
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

        # Angle of incidence, from the incident-side normal (``normals`` faces
        # the incoming ray). Measuring it from the drawn hemisphere would give
        # 90 deg for every transmitted draw.
        cos_i = np.clip((-d_np * n_np).sum(axis=1), 0.0, 1.0)
        theta_i = np.degrees(np.arccos(cos_i))

        from optiland.nonsequential.bsdf.lambertian import (  # noqa: PLC0415
            _orthonormal_basis,
        )

        r1 = np.asarray(rng.uniform(ray_id, bounce, EventSlot.BSDF_U1))
        r2 = np.asarray(rng.uniform(ray_id, bounce, EventSlot.BSDF_U2))
        phi = 2.0 * np.pi * r1
        theta_s_rad, flux_weights = self._sample_theta_s(theta_i, r2)
        cos_theta = np.cos(theta_s_rad)
        sin_theta = np.sin(theta_s_rad)

        lx = sin_theta * np.cos(phi)
        ly = sin_theta * np.sin(phi)
        lz = cos_theta

        t_vec, b_vec = _orthonormal_basis(hemisphere_np)
        scattered = (
            lx[:, None] * t_vec + ly[:, None] * b_vec + lz[:, None] * hemisphere_np
        )
        norms = (scattered * scattered).sum(axis=1, keepdims=True) ** 0.5
        scattered = scattered / norms

        return (
            be.array(scattered.astype(np.float64)),
            be.array(flux_weights),
            be.array(transmitted_np),
        )

    def _sample_theta_s(
        self, theta_i: np.ndarray, u: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Importance-sample theta_s and return it with its flux weight.

        Args:
            theta_i: Angles of incidence [deg], shape (N,).
            u: Uniform deviates in [0, 1), shape (N,). Split to choose the
                table row and then to invert that row's CDF.

        Returns:
            (theta_s [rad], flux_weights), each of shape (N,).
        """
        ti_vals = self._theta_i_vals
        ts = np.radians(self._theta_s_vals)
        n_rows = len(ti_vals)
        u = np.clip(np.asarray(u, dtype=np.float64), 0.0, np.nextafter(1.0, 0.0))

        # Bracketing rows k, k+1 and linear-interpolation fraction f.
        if n_rows == 1:
            k = np.zeros(theta_i.shape, dtype=np.int64)
            f = np.zeros(theta_i.shape)
            k_hi = k
        else:
            k = np.clip(
                np.searchsorted(ti_vals, theta_i, side="right") - 1, 0, n_rows - 2
            )
            f = np.clip(
                (theta_i - ti_vals[k]) / (ti_vals[k + 1] - ti_vals[k]), 0.0, 1.0
            )
            k_hi = k + 1
        tol = 1e-9
        in_range = (theta_i >= ti_vals[0] - tol) & (theta_i <= ti_vals[-1] + tol)

        # Choose row k_hi with probability f, re-using u (stays uniform).
        take_hi = u < f
        u_row = np.where(
            take_hi,
            u / np.where(f > 0, f, 1.0),
            (u - f) / np.where(f < 1, 1.0 - f, 1.0),
        )
        row = np.where(take_hi, k_hi, k)

        # Invert the chosen row's piecewise-linear CDF.
        target = u_row * self._cdf[row, -1]
        j = np.empty(theta_i.shape, dtype=np.int64)
        for r in np.unique(row):
            sel = row == r
            j[sel] = np.searchsorted(self._cdf[r], target[sel], side="right") - 1
        j = np.clip(j, 0, len(ts) - 2)
        bin_mass = self._cdf[row, j + 1] - self._cdf[row, j]
        frac = np.where(bin_mass > 0, (target - self._cdf[row, j]) / bin_mass, 0.5)
        theta_s = ts[j] + np.clip(frac, 0.0, 1.0) * (ts[j + 1] - ts[j])

        # Exact weight: 2 pi b cos sin / (mixture density of theta_s).
        pdf = (1.0 - f) * self._bin_pdf[k, j] + f * self._bin_pdf[k_hi, j]
        b = self._interp(np.column_stack([theta_i, np.degrees(theta_s)]))
        integrand = np.clip(b, 0.0, None) * np.cos(theta_s) * np.sin(theta_s)
        with np.errstate(invalid="ignore", divide="ignore"):
            weights = np.where((pdf > 0) & in_range, 2.0 * np.pi * integrand / pdf, 0.0)
        return theta_s, np.clip(weights, 0.0, None)

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
