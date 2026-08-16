"""Spectral detector for Non-Sequential Raytracing.

Accumulates per-wavelength irradiance on a planar surface.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from optiland.nonsequential._utils import as_detached_param, to_numpy
from optiland.nonsequential.components.base import _get_transform
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

    Records flux in a 3D (x, y, wl) grid.

    Attributes:
        cs: Coordinate system.
        width: Detector width [mm].
        height: Detector height [mm].
        num_pixels_x: Number of pixels along x.
        num_pixels_y: Number of pixels along y.
        wavelength_bins: Wavelength bin edges [µm].
        splat: Splatting mode. Only 'hard' binning is currently implemented;
            'bilinear' and 'gaussian' are accepted but fall back to hard.
        splat_sigma: Reserved for future Gaussian splat support.
    """

    def __init__(
        self,
        cs: CoordinateSystem,
        width: float,
        height: float,
        num_pixels_x: int,
        num_pixels_y: int,
        wavelength_bins: np.ndarray,
        splat: Literal["bilinear", "gaussian", "hard"] = "bilinear",
        splat_sigma: float = 0.5,
        name: str = "",
    ) -> None:
        """Initialize SpectralDetector.

        Args:
            cs: Coordinate system for detector position/orientation.
            width: Detector width [mm].
            height: Detector height [mm].
            num_pixels_x: Number of pixels along x.
            num_pixels_y: Number of pixels along y.
            wavelength_bins: Wavelength bin edges [µm], shape (n_lambda + 1,).
            splat: Splatting mode. Accepts 'bilinear', 'gaussian', or 'hard'.
                Currently all modes use hard binning; bilinear splat for
                SpectralDetector is a TODO.
            splat_sigma: Gaussian splat sigma in pixels (reserved for future
                use).
            name: Optional label.
        """
        geometry = FinitePlaneGeometry(width=width, height=height)
        super().__init__(cs, geometry, name=name)
        _reason = "it accumulates into a NumPy histogram"
        self.width = as_detached_param(width, "width", "SpectralDetector", _reason)
        self.height = as_detached_param(height, "height", "SpectralDetector", _reason)
        self.num_pixels_x = int(num_pixels_x)
        self.num_pixels_y = int(num_pixels_y)
        self.splat = splat
        self.splat_sigma = float(splat_sigma)
        self.wavelength_bins = np.asarray(wavelength_bins, dtype=np.float64)
        n_lambda = len(wavelength_bins) - 1

        self._flux_map = np.zeros(
            (num_pixels_y, num_pixels_x, n_lambda), dtype=np.float64
        )
        self._num_rays_hit = 0

        self._x_edges = np.linspace(
            -self.width / 2.0, self.width / 2.0, num_pixels_x + 1
        )
        self._y_edges = np.linspace(
            -self.height / 2.0, self.height / 2.0, num_pixels_y + 1
        )

    def record(self, rays: NSQRayBundle, t: np.ndarray, hit_mask: np.ndarray) -> None:
        """Accumulate per-wavelength flux from hit rays.

        Args:
            rays: Current ray bundle.
            t: Hit distances [mm], shape (N,).
            hit_mask: Boolean mask of hitting rays, shape (N,).
        """
        hit_mask_np = to_numpy(hit_mask).astype(bool)
        if not hit_mask_np.any():
            return

        translation, rot = _get_transform(self.cs)
        t_vec = np.array(translation, dtype=float)
        R = np.array(rot, dtype=float)

        x_g = to_numpy(rays.x)
        y_g = to_numpy(rays.y)
        z_g = to_numpy(rays.z)
        L_g = to_numpy(rays.L)
        M_g = to_numpy(rays.M)
        N_g = to_numpy(rays.N)
        t_np = to_numpy(t)
        flux_np = to_numpy(rays.flux)
        wl_np = to_numpy(rays.wavelength)

        idx = np.where(hit_mask_np)[0]
        t_hit = t_np[idx]
        hx_g = x_g[idx] + t_hit * L_g[idx]
        hy_g = y_g[idx] + t_hit * M_g[idx]
        hz_g = z_g[idx] + t_hit * N_g[idx]

        pos_g = np.stack([hx_g, hy_g, hz_g], axis=1)
        pos_l = (pos_g - t_vec) @ R

        hx_l = pos_l[:, 0]
        hy_l = pos_l[:, 1]
        flux_hit = flux_np[idx]
        wl_hit = wl_np[idx]

        ix = np.clip(
            np.searchsorted(self._x_edges, hx_l, side="right") - 1,
            0,
            self.num_pixels_x - 1,
        )
        iy = np.clip(
            np.searchsorted(self._y_edges, hy_l, side="right") - 1,
            0,
            self.num_pixels_y - 1,
        )
        iwl = np.clip(
            np.searchsorted(self.wavelength_bins, wl_hit, side="right") - 1,
            0,
            self._flux_map.shape[2] - 1,
        )

        np.add.at(self._flux_map, (iy, ix, iwl), flux_hit)
        self._num_rays_hit += hit_mask_np.sum()

    def get_result(self) -> SpectralResult:
        """Return accumulated spectral result.

        Returns:
            SpectralResult with irradiance [W/mm^2] per pixel per wavelength bin.
        """
        pixel_area = (self.width / self.num_pixels_x) * (
            self.height / self.num_pixels_y
        )
        irradiance = self._flux_map / pixel_area

        x_centres = 0.5 * (self._x_edges[:-1] + self._x_edges[1:])
        y_centres = 0.5 * (self._y_edges[:-1] + self._y_edges[1:])
        # wl_centres in µm
        wl_centres = 0.5 * (self.wavelength_bins[:-1] + self.wavelength_bins[1:])

        return SpectralResult(
            irradiance=irradiance.copy(),
            x_coords=x_centres,
            y_coords=y_centres,
            wavelengths=wl_centres,
            total_flux=float(self._flux_map.sum()),
            num_rays_hit=self._num_rays_hit,
        )

    def reset(self) -> None:
        """Clear accumulated data."""
        self._flux_map[:] = 0.0
        self._num_rays_hit = 0
