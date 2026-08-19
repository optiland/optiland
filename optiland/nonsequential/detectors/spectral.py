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

# Longest wavelength Optiland's material catalogs cover is well under this, so
# any bin edge above it is a nanometre value that slipped through as µm.
_MAX_PLAUSIBLE_WAVELENGTH_UM = 100.0


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
        splat: Spatial splatting mode -- 'bilinear', 'gaussian', or 'hard'.
            Splatting is spatial (x, y) only; the wavelength bin is always
            hard-assigned.
        splat_sigma: Gaussian splat sigma in pixels (used when
            ``splat='gaussian'``).
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
        absorb: bool = True,
    ) -> None:
        """Initialize SpectralDetector.

        Args:
            cs: Coordinate system for detector position/orientation.
            width: Detector width [mm].
            height: Detector height [mm].
            num_pixels_x: Number of pixels along x.
            num_pixels_y: Number of pixels along y.
            wavelength_bins: Wavelength bin edges [µm], shape (n_lambda + 1,).
            splat: Spatial splatting mode. Accepts 'bilinear', 'gaussian',
                or 'hard'.
            splat_sigma: Gaussian splat sigma in pixels. Only used when
                ``splat='gaussian'``.
            name: Optional label.
            absorb: Whether a hit terminates the ray (default True).

        Raises:
            ValueError: If ``wavelength_bins`` are not plausibly in µm. Bin
                edges are compared against ``rays.wavelength``, which is in
                µm; nanometre edges would silently clip every ray into the
                first bin.
        """
        geometry = FinitePlaneGeometry(width=width, height=height)
        super().__init__(cs, geometry, name=name, absorb=absorb)
        _reason = "it accumulates into a NumPy histogram"
        self.width = as_detached_param(width, "width", "SpectralDetector", _reason)
        self.height = as_detached_param(height, "height", "SpectralDetector", _reason)
        self.num_pixels_x = int(num_pixels_x)
        self.num_pixels_y = int(num_pixels_y)
        self.splat = splat
        self.splat_sigma = float(splat_sigma)
        self.wavelength_bins = np.asarray(wavelength_bins, dtype=np.float64)
        if self.wavelength_bins.min() > _MAX_PLAUSIBLE_WAVELENGTH_UM:
            raise ValueError(
                f"wavelength_bins must be in µm, but the smallest bin edge is "
                f"{self.wavelength_bins.min():g}. Values this large look like "
                f"nanometres - divide by 1000 (e.g. 550 nm -> 0.55)."
            )
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

        iwl = np.clip(
            np.searchsorted(self.wavelength_bins, wl_hit, side="right") - 1,
            0,
            self._flux_map.shape[2] - 1,
        )

        nx, ny = self.num_pixels_x, self.num_pixels_y
        dx = self.width / nx
        dy = self.height / ny
        if self.splat == "hard":
            self._record_hard(hx_l, hy_l, flux_hit, iwl, nx, ny)
        elif self.splat == "gaussian":
            self._record_gaussian(hx_l, hy_l, flux_hit, iwl, nx, ny, dx, dy)
        else:
            self._record_bilinear(hx_l, hy_l, flux_hit, iwl, nx, ny, dx, dy)
        self._num_rays_hit += hit_mask_np.sum()

    def _record_hard(self, hx_l, hy_l, flux_hit, iwl, nx, ny) -> None:
        """Hard-bin spatial accumulation (see ``IrradianceDetector._record_hard``)."""
        ix = np.clip(np.searchsorted(self._x_edges, hx_l, side="right") - 1, 0, nx - 1)
        iy = np.clip(np.searchsorted(self._y_edges, hy_l, side="right") - 1, 0, ny - 1)
        np.add.at(self._flux_map, (iy, ix, iwl), flux_hit)

    def _record_bilinear(self, hx_l, hy_l, flux_hit, iwl, nx, ny, dx, dy) -> None:
        """Bilinear spatial splat (see ``IrradianceDetector._record_bilinear``)."""
        px = (hx_l + self.width / 2.0) / dx - 0.5
        py = (hy_l + self.height / 2.0) / dy - 0.5
        ix0 = np.floor(px).astype(np.int64)
        iy0 = np.floor(py).astype(np.int64)
        wx1 = px - ix0
        wy1 = py - iy0
        wx0 = 1.0 - wx1
        wy0 = 1.0 - wy1

        for dix, diy, wx, wy in (
            (0, 0, wx0, wy0),
            (1, 0, wx1, wy0),
            (0, 1, wx0, wy1),
            (1, 1, wx1, wy1),
        ):
            ix = np.clip(ix0 + dix, 0, nx - 1)
            iy = np.clip(iy0 + diy, 0, ny - 1)
            np.add.at(self._flux_map, (iy, ix, iwl), flux_hit * wx * wy)

    def _record_gaussian(self, hx_l, hy_l, flux_hit, iwl, nx, ny, dx, dy) -> None:
        """Gaussian spatial splat, truncated and renormalised per ray so
        truncation never loses energy (see
        ``IrradianceDetector._record_gaussian``)."""
        sigma = self.splat_sigma
        if sigma <= 0.0:
            self._record_hard(hx_l, hy_l, flux_hit, iwl, nx, ny)
            return

        radius = max(1, int(np.ceil(3.0 * sigma)))
        px = (hx_l + self.width / 2.0) / dx - 0.5
        py = (hy_l + self.height / 2.0) / dy - 0.5
        ix0 = np.floor(px).astype(np.int64)
        iy0 = np.floor(py).astype(np.int64)

        offsets = range(-radius, radius + 1)
        gx = {d: np.exp(-0.5 * ((ix0 + d - px) / sigma) ** 2) for d in offsets}
        gy = {d: np.exp(-0.5 * ((iy0 + d - py) / sigma) ** 2) for d in offsets}
        sx = sum(gx.values())
        sy = sum(gy.values())
        norm = sx * sy

        for dix in offsets:
            ix = np.clip(ix0 + dix, 0, nx - 1)
            for diy in offsets:
                iy = np.clip(iy0 + diy, 0, ny - 1)
                weight = (gx[dix] * gy[diy]) / norm
                np.add.at(self._flux_map, (iy, ix, iwl), flux_hit * weight)

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
