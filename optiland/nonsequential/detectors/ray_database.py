"""Ray database detector for Non-Sequential Raytracing.

Stores individual ray phase-space data at the detector surface.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optiland.nonsequential.components.base import _get_xp
from optiland.nonsequential.detectors.base import BaseDetector
from optiland.nonsequential.results.ray_database import RayDatabase

if TYPE_CHECKING:
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.components.geometry.base import ComponentGeometry
    from optiland.nonsequential.ray_bundle import NSQRayBundle


class RayDatabaseDetector(BaseDetector):
    """Stores individual ray phase-space data at the detector surface.

    When store_rays=False, only aggregated flux is stored for maximum
    throughput. When store_rays=True, full per-ray data is accumulated.

    Attributes:
        cs: Coordinate system.
        geometry: Surface geometry defining the detector extent.
        store_rays: If True, store individual ray data. If False, only
            aggregate flux (faster for GPU runs).
        max_rays: Maximum number of rays to store. If set, uses a
            circular buffer. None means unlimited.
    """

    def __init__(
        self,
        cs: CoordinateSystem,
        geometry: ComponentGeometry,
        store_rays: bool = True,
        max_rays: int | None = None,
        name: str = "",
    ) -> None:
        """Initialize RayDatabaseDetector.

        Args:
            cs: Coordinate system.
            geometry: Surface geometry.
            store_rays: If True, store per-ray phase-space data.
            max_rays: Maximum rays to store (circular buffer). None = unlimited.
            name: Optional label.
        """
        super().__init__(cs, geometry, name=name)
        self.store_rays = store_rays
        self.max_rays = max_rays

        # Storage lists (appended per batch, concatenated on get_result)
        self._x: list[np.ndarray] = []
        self._y: list[np.ndarray] = []
        self._z: list[np.ndarray] = []
        self._dx: list[np.ndarray] = []
        self._dy: list[np.ndarray] = []
        self._dz: list[np.ndarray] = []
        self._flux: list[np.ndarray] = []
        self._wavelength: list[np.ndarray] = []
        self._total_flux: float = 0.0
        self._n_rays_hit: int = 0

    def record(self, rays: NSQRayBundle, t: np.ndarray, hit_mask: np.ndarray) -> None:
        """Record hit ray data.

        Args:
            rays: Current ray bundle.
            t: Hit distances [mm], shape (N,).
            hit_mask: Boolean mask of hitting rays, shape (N,).
        """
        xp = _get_xp(rays.x)
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
        wl_np = _to_numpy(xp, rays.wavelength)

        # Advance hit positions
        hx = (x_g + t_np * dx_g)[hit_mask_np]
        hy = (y_g + t_np * dy_g)[hit_mask_np]
        hz = (z_g + t_np * dz_g)[hit_mask_np]
        hdx = dx_g[hit_mask_np]
        hdy = dy_g[hit_mask_np]
        hdz = dz_g[hit_mask_np]
        hflux = flux_np[hit_mask_np]
        hwl = wl_np[hit_mask_np]

        self._total_flux += float(hflux.sum())
        self._n_rays_hit += len(hx)

        if self.store_rays:
            self._x.append(hx)
            self._y.append(hy)
            self._z.append(hz)
            self._dx.append(hdx)
            self._dy.append(hdy)
            self._dz.append(hdz)
            self._flux.append(hflux)
            self._wavelength.append(hwl)

            # Apply circular buffer limit
            if self.max_rays is not None:
                total = sum(len(a) for a in self._x)
                if total > self.max_rays:
                    # Rebuild keeping only the last max_rays entries
                    all_x = np.concatenate(self._x)[-self.max_rays :]
                    all_y = np.concatenate(self._y)[-self.max_rays :]
                    all_z = np.concatenate(self._z)[-self.max_rays :]
                    all_dx = np.concatenate(self._dx)[-self.max_rays :]
                    all_dy = np.concatenate(self._dy)[-self.max_rays :]
                    all_dz = np.concatenate(self._dz)[-self.max_rays :]
                    all_f = np.concatenate(self._flux)[-self.max_rays :]
                    all_wl = np.concatenate(self._wavelength)[-self.max_rays :]
                    self._x = [all_x]
                    self._y = [all_y]
                    self._z = [all_z]
                    self._dx = [all_dx]
                    self._dy = [all_dy]
                    self._dz = [all_dz]
                    self._flux = [all_f]
                    self._wavelength = [all_wl]

    def get_result(self) -> RayDatabase:
        """Return accumulated ray database.

        Returns:
            RayDatabase with stored ray phase-space data.
        """
        if self.store_rays and self._x:
            return RayDatabase(
                x=np.concatenate(self._x),
                y=np.concatenate(self._y),
                z=np.concatenate(self._z),
                dx=np.concatenate(self._dx),
                dy=np.concatenate(self._dy),
                dz=np.concatenate(self._dz),
                flux=np.concatenate(self._flux),
                wavelength=np.concatenate(self._wavelength),
            )
        # Aggregated-only mode: return empty database with flux summary
        return RayDatabase(
            x=np.array([]),
            y=np.array([]),
            z=np.array([]),
            dx=np.array([]),
            dy=np.array([]),
            dz=np.array([]),
            flux=np.array([self._total_flux]),
            wavelength=np.array([0.0]),
        )

    def reset(self) -> None:
        """Clear accumulated data."""
        self._x.clear()
        self._y.clear()
        self._z.clear()
        self._dx.clear()
        self._dy.clear()
        self._dz.clear()
        self._flux.clear()
        self._wavelength.clear()
        self._total_flux = 0.0
        self._n_rays_hit = 0


def _to_numpy(xp, arr):
    if xp is not np:
        return xp.asnumpy(arr)
    return np.asarray(arr)
