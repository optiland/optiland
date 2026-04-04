"""Spectral detector for Non-Sequential Raytracing.

Accumulates per-wavelength irradiance on a planar surface.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optiland.nonsequential.components.base import _get_transform, _get_xp
from optiland.nonsequential.components.geometry.analytic.plane import (
    FinitePlaneGeometry,
)
from optiland.nonsequential.detectors.base import BaseDetector
from optiland.nonsequential.results.spectral_result import SpectralResult

if TYPE_CHECKING:
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.ray_bundle import NSQRayBundle


class SpectralDetector(BaseDetector):
    """Per-wavelength irradiance detector on a planar rectangular surface.

    Records flux in a 3D (x, y, λ) grid.

    Attributes:
        cs: Coordinate system.
        width: Detector width [mm].
        height: Detector height [mm].
        n_pixels_x: Number of pixels along x.
        n_pixels_y: Number of pixels along y.
        wavelength_bins: Wavelength bin edges [nm].
    """

    def __init__(
        self,
        cs: CoordinateSystem,
        width: float,
        height: float,
        n_pixels_x: int,
        n_pixels_y: int,
        wavelength_bins: np.ndarray,
        name: str = "",
    ) -> None:
        """Initialize SpectralDetector.

        Args:
            cs: Coordinate system for detector position/orientation.
            width: Detector width [mm].
            height: Detector height [mm].
            n_pixels_x: Number of pixels along x.
            n_pixels_y: Number of pixels along y.
            wavelength_bins: Wavelength bin edges [nm], shape (n_lambda + 1,).
            name: Optional label.
        """
        geometry = FinitePlaneGeometry(width=width, height=height)
        super().__init__(cs, geometry, name=name)
        self.width = float(width)
        self.height = float(height)
        self.n_pixels_x = int(n_pixels_x)
        self.n_pixels_y = int(n_pixels_y)
        self.wavelength_bins = np.asarray(wavelength_bins, dtype=np.float64)
        n_lambda = len(wavelength_bins) - 1

        self._flux_map = np.zeros((n_pixels_y, n_pixels_x, n_lambda), dtype=np.float64)
        self._n_rays_hit = 0

        self._x_edges = np.linspace(-width / 2.0, width / 2.0, n_pixels_x + 1)
        self._y_edges = np.linspace(-height / 2.0, height / 2.0, n_pixels_y + 1)

    def record(self, rays: NSQRayBundle, t: np.ndarray, hit_mask: np.ndarray) -> None:
        """Accumulate per-wavelength flux from hit rays.

        Args:
            rays: Current ray bundle.
            t: Hit distances [mm], shape (N,).
            hit_mask: Boolean mask of hitting rays, shape (N,).
        """
        xp = _get_xp(rays.x)
        hit_mask_np = _to_numpy(xp, hit_mask).astype(bool)
        if not hit_mask_np.any():
            return

        translation, rot = _get_transform(self.cs)
        t_vec = np.array(translation, dtype=float)
        R = np.array(rot, dtype=float)

        x_g = _to_numpy(xp, rays.x)
        y_g = _to_numpy(xp, rays.y)
        z_g = _to_numpy(xp, rays.z)
        dx_g = _to_numpy(xp, rays.dx)
        dy_g = _to_numpy(xp, rays.dy)
        dz_g = _to_numpy(xp, rays.dz)
        t_np = _to_numpy(xp, t)
        flux_np = _to_numpy(xp, rays.flux)
        wl_np = _to_numpy(xp, rays.wavelength)

        idx = np.where(hit_mask_np)[0]
        t_hit = t_np[idx]
        hx_g = x_g[idx] + t_hit * dx_g[idx]
        hy_g = y_g[idx] + t_hit * dy_g[idx]
        hz_g = z_g[idx] + t_hit * dz_g[idx]

        pos_g = np.stack([hx_g, hy_g, hz_g], axis=1)
        pos_l = (pos_g - t_vec) @ R

        hx_l = pos_l[:, 0]
        hy_l = pos_l[:, 1]
        flux_hit = flux_np[idx]
        wl_hit = wl_np[idx]

        ix = np.clip(
            np.searchsorted(self._x_edges, hx_l, side="right") - 1,
            0,
            self.n_pixels_x - 1,
        )
        iy = np.clip(
            np.searchsorted(self._y_edges, hy_l, side="right") - 1,
            0,
            self.n_pixels_y - 1,
        )
        iwl = np.clip(
            np.searchsorted(self.wavelength_bins, wl_hit, side="right") - 1,
            0,
            self._flux_map.shape[2] - 1,
        )

        np.add.at(self._flux_map, (iy, ix, iwl), flux_hit)
        self._n_rays_hit += hit_mask_np.sum()

    def get_result(self) -> SpectralResult:
        """Return accumulated spectral result.

        Returns:
            SpectralResult with irradiance [W/mm²] per pixel per wavelength bin.
        """
        pixel_area = (self.width / self.n_pixels_x) * (self.height / self.n_pixels_y)
        irradiance = self._flux_map / pixel_area

        x_centres = 0.5 * (self._x_edges[:-1] + self._x_edges[1:])
        y_centres = 0.5 * (self._y_edges[:-1] + self._y_edges[1:])
        wl_centres = 0.5 * (self.wavelength_bins[:-1] + self.wavelength_bins[1:])

        return SpectralResult(
            irradiance=irradiance.copy(),
            x_coords=x_centres,
            y_coords=y_centres,
            wavelengths=wl_centres,
            total_flux=float(self._flux_map.sum()),
            n_rays_hit=self._n_rays_hit,
        )

    def reset(self) -> None:
        """Clear accumulated data."""
        self._flux_map[:] = 0.0
        self._n_rays_hit = 0


def _to_numpy(xp, arr):
    if xp is not np:
        return xp.asnumpy(arr)
    return np.asarray(arr)
