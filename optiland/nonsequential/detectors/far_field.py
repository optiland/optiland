"""Far-field detector for Non-Sequential Raytracing.

Accumulates angular flux distribution in the far field.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optiland.nonsequential._utils import to_numpy
from optiland.nonsequential.components.base import _get_transform
from optiland.nonsequential.components.geometry.analytic.plane import (
    FinitePlaneGeometry,
)
from optiland.nonsequential.detectors.base import BaseDetector
from optiland.nonsequential.results.far_field_pattern import FarFieldPattern

if TYPE_CHECKING:
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.ray_bundle import NSQRayBundle


class FarFieldDetector(BaseDetector):
    """Accumulates angular flux distribution in the far field.

    Records ray directions at the detector surface and bins them into a
    polar (theta, phi) histogram.

    Attributes:
        cs: Coordinate system.
        theta_max_deg: Maximum polar angle to record [deg].
        num_bins_theta: Number of polar angle bins.
        num_bins_phi: Number of azimuthal angle bins.
    """

    def __init__(
        self,
        cs: CoordinateSystem,
        theta_max_deg: float,
        num_bins_theta: int,
        num_bins_phi: int,
        aperture_radius: float = 1e6,
        name: str = "",
        absorb: bool = True,
    ) -> None:
        """Initialize FarFieldDetector.

        Args:
            cs: Coordinate system for detector position/orientation.
            theta_max_deg: Maximum polar half-angle to record [deg].
            num_bins_theta: Number of polar angle bins.
            num_bins_phi: Number of azimuthal angle bins.
            aperture_radius: Detector aperture radius [mm] (default: very large).
            name: Optional label.
            absorb: Whether a hit terminates the ray (default True).
        """
        geometry = FinitePlaneGeometry(aperture_radius=aperture_radius)
        super().__init__(cs, geometry, name=name, absorb=absorb)
        self.num_bins_theta = int(num_bins_theta)
        self.num_bins_phi = int(num_bins_phi)

        self._intensity = np.zeros((num_bins_theta, num_bins_phi), dtype=np.float64)
        self._num_rays_hit = 0
        self._total_flux = 0.0

        self._theta_edges = np.linspace(0.0, theta_max_deg, num_bins_theta + 1)
        self._phi_edges = np.linspace(-180.0, 180.0, num_bins_phi + 1)

    def record(self, rays: NSQRayBundle, t: np.ndarray, hit_mask: np.ndarray) -> None:
        """Accumulate angular flux from hit rays.

        Converts ray directions to (theta, phi) in local detector frame and bins.

        Args:
            rays: Current ray bundle.
            t: Hit distances [mm], shape (N,).
            hit_mask: Boolean mask of hitting rays, shape (N,).
        """
        hit_mask_np = to_numpy(hit_mask).astype(bool)
        if not hit_mask_np.any():
            return

        _, rot = _get_transform(self.cs)
        R = np.array(rot, dtype=float)

        L_g = to_numpy(rays.L)
        M_g = to_numpy(rays.M)
        N_g = to_numpy(rays.N)
        flux_np = to_numpy(rays.flux)

        dirs_g = np.stack([L_g, M_g, N_g], axis=1)
        dirs_l = dirs_g @ R  # global -> local

        dirs_hit = dirs_l[hit_mask_np]
        flux_hit = flux_np[hit_mask_np]

        # Compute polar angles in local frame
        # theta is angle from local +z axis
        cos_theta = np.clip(np.abs(dirs_hit[:, 2]), 0.0, 1.0)
        theta_deg = np.degrees(np.arccos(cos_theta))
        phi_deg = np.degrees(np.arctan2(dirs_hit[:, 1], dirs_hit[:, 0]))

        # Bin into 2D histogram
        i_theta = np.searchsorted(self._theta_edges, theta_deg, side="right") - 1
        i_phi = np.searchsorted(self._phi_edges, phi_deg, side="right") - 1
        i_theta = np.clip(i_theta, 0, self.num_bins_theta - 1)
        i_phi = np.clip(i_phi, 0, self.num_bins_phi - 1)

        # Solid-angle normalisation per bin (W/sr)
        d_theta = np.radians(self._theta_edges[1] - self._theta_edges[0])
        d_phi = np.radians(self._phi_edges[1] - self._phi_edges[0])
        theta_centres = np.radians(
            0.5 * (self._theta_edges[:-1] + self._theta_edges[1:])
        )
        solid_angle = np.sin(theta_centres[i_theta]) * d_theta * d_phi
        solid_angle = np.where(solid_angle > 0, solid_angle, 1.0)

        np.add.at(self._intensity, (i_theta, i_phi), flux_hit / solid_angle)
        self._num_rays_hit += hit_mask_np.sum()
        # Track the radiometric flux separately: _intensity is divided by the
        # per-bin solid angle, so summing it gives W/sr, not W.
        self._total_flux += float(flux_hit.sum())

    def get_result(self) -> FarFieldPattern:
        """Return the accumulated far-field pattern.

        Returns:
            FarFieldPattern with intensity [W/sr].
        """
        theta_centres = 0.5 * (self._theta_edges[:-1] + self._theta_edges[1:])
        phi_centres = 0.5 * (self._phi_edges[:-1] + self._phi_edges[1:])
        return FarFieldPattern(
            intensity=self._intensity.copy(),
            theta=theta_centres,
            phi=phi_centres,
            total_flux=self._total_flux,
            num_rays_hit=self._num_rays_hit,
        )

    def reset(self) -> None:
        """Clear accumulated data."""
        self._intensity[:] = 0.0
        self._num_rays_hit = 0
        self._total_flux = 0.0
