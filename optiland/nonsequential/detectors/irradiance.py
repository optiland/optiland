"""Irradiance detector for Non-Sequential Raytracing.

Accumulates a 2D flux map on a planar rectangular surface.

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
from optiland.nonsequential.results.irradiance_map import IrradianceMap

if TYPE_CHECKING:
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.ray_bundle import NSQRayBundle


class IrradianceDetector(BaseDetector):
    """2D irradiance map detector on a planar rectangular surface.

    Records flux in a pixel grid. Rays are absorbed on contact.

    Attributes:
        cs: Coordinate system.
        width: Detector width [mm].
        height: Detector height [mm].
        num_pixels_x: Number of pixels along x.
        num_pixels_y: Number of pixels along y.
    """

    def __init__(
        self,
        cs: CoordinateSystem,
        width: float,
        height: float,
        num_pixels_x: int,
        num_pixels_y: int,
        name: str = "",
    ) -> None:
        """Initialize IrradianceDetector.

        Args:
            cs: Coordinate system for detector position/orientation.
            width: Detector width [mm].
            height: Detector height [mm].
            num_pixels_x: Number of pixels along x.
            num_pixels_y: Number of pixels along y.
            name: Optional label.
        """
        geometry = FinitePlaneGeometry(width=width, height=height)
        super().__init__(cs, geometry, name=name)
        self.width = float(width)
        self.height = float(height)
        self.num_pixels_x = int(num_pixels_x)
        self.num_pixels_y = int(num_pixels_y)

        # Accumulation buffer: shape (ny, nx)
        self._flux_map = np.zeros((num_pixels_y, num_pixels_x), dtype=np.float64)
        self._num_rays_hit = 0

        # Pixel bin edges
        self._x_edges = np.linspace(-width / 2.0, width / 2.0, num_pixels_x + 1)
        self._y_edges = np.linspace(-height / 2.0, height / 2.0, num_pixels_y + 1)

    def record(self, rays: NSQRayBundle, t: np.ndarray, hit_mask: np.ndarray) -> None:
        """Accumulate flux from hit rays into the pixel grid.

        Computes hit positions in local detector frame, bins them, and
        scatter-adds flux into the 2D histogram.

        Args:
            rays: Current ray bundle (positions not yet advanced to hit point).
            t: Hit distances [mm], shape (N,).
            hit_mask: Boolean mask of hitting rays, shape (N,).
        """
        xp = _get_xp(rays.x)
        translation, rot = _get_transform(self.cs)

        # Get hit positions in global frame
        hit_mask_np = _to_numpy(xp, hit_mask).astype(bool)
        if not hit_mask_np.any():
            return

        x_g = _to_numpy(xp, rays.x)
        y_g = _to_numpy(xp, rays.y)
        z_g = _to_numpy(xp, rays.z)
        dx_g = _to_numpy(xp, rays.dx)
        dy_g = _to_numpy(xp, rays.dy)
        dz_g = _to_numpy(xp, rays.dz)
        t_np = _to_numpy(xp, t)
        flux_np = _to_numpy(xp, rays.flux)

        # Use only hit rays to avoid inf*0 NaN from non-hit rays
        idx = np.where(hit_mask_np)[0]
        t_hit = t_np[idx]
        hx_g = x_g[idx] + t_hit * dx_g[idx]
        hy_g = y_g[idx] + t_hit * dy_g[idx]
        hz_g = z_g[idx] + t_hit * dz_g[idx]

        # Transform hit positions to local detector frame
        pos_g = np.stack([hx_g, hy_g, hz_g], axis=1)
        t_vec = np.array(translation, dtype=float)
        R = np.array(rot, dtype=float)
        pos_l = (pos_g - t_vec) @ R  # global -> local

        hx_l = pos_l[:, 0]
        hy_l = pos_l[:, 1]
        flux_hit = flux_np[idx]

        # Bin into 2D histogram (scatter-add)
        ix = np.searchsorted(self._x_edges, hx_l, side="right") - 1
        iy = np.searchsorted(self._y_edges, hy_l, side="right") - 1

        # Clamp to valid pixel indices
        ix = np.clip(ix, 0, self.num_pixels_x - 1)
        iy = np.clip(iy, 0, self.num_pixels_y - 1)

        # Scatter-add (np.add.at is unbuffered)
        np.add.at(self._flux_map, (iy, ix), flux_hit)
        self._num_rays_hit += hit_mask_np.sum()

    def get_result(self) -> IrradianceMap:
        """Return the accumulated irradiance map.

        Returns:
            IrradianceMap with irradiance [W/mm^2] computed from stored flux.
        """
        pixel_area = (self.width / self.num_pixels_x) * (
            self.height / self.num_pixels_y
        )
        irradiance = self._flux_map / pixel_area

        x_centres = 0.5 * (self._x_edges[:-1] + self._x_edges[1:])
        y_centres = 0.5 * (self._y_edges[:-1] + self._y_edges[1:])

        return IrradianceMap(
            irradiance=irradiance.copy(),
            x_coords=x_centres,
            y_coords=y_centres,
            total_flux=float(self._flux_map.sum()),
            num_rays_hit=self._num_rays_hit,
        )

    def reset(self) -> None:
        """Clear accumulated data."""
        self._flux_map[:] = 0.0
        self._num_rays_hit = 0


def _to_numpy(xp, arr):
    if xp is not np:
        return xp.asnumpy(arr)
    return np.asarray(arr)
