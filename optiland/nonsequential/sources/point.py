"""Point source for Non-Sequential Raytracing.

Isotropic emitter or emission into a defined solid angle cone.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optiland.nonsequential._utils import as_detached_param
from optiland.nonsequential.components.base import _get_transform
from optiland.nonsequential.ray_bundle import NSQRayBundle
from optiland.nonsequential.rng import EventSlot
from optiland.nonsequential.sources.base import BaseNSQSource, Spectrum

if TYPE_CHECKING:
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.rng import NSQRng


class PointSource(BaseNSQSource):
    """Point source emitting rays from a single position.

    Emission can be isotropic (full sphere) or confined to a cone around
    the local +z axis.

    Attributes:
        cs: Coordinate system (origin = source position).
        spectrum: Wavelength distribution.
        total_flux: Total emitted flux [W].
        half_angle_deg: Half-angle of the emission cone [deg]. 180 = isotropic.
        medium: Medium the source is embedded in.
    """

    def __init__(
        self,
        cs: CoordinateSystem,
        spectrum: Spectrum,
        total_flux: float = 1.0,
        half_angle_deg: float = 90.0,
        medium=None,
    ) -> None:
        """Initialize PointSource.

        Args:
            cs: Coordinate system for source position.
            spectrum: Wavelength distribution.
            total_flux: Total emitted flux [W].
            half_angle_deg: Half-angle of emission cone [deg].
                90 = hemisphere, 180 = full sphere (isotropic).
            medium: Medium the source is embedded in (default: vacuum).
        """
        super().__init__(cs, spectrum, total_flux)
        self.half_angle_deg = as_detached_param(
            half_angle_deg, "half_angle_deg", "PointSource"
        )
        self.medium = medium

    def generate(self, ray_id: np.ndarray, rng: NSQRng) -> NSQRayBundle:
        """Generate rays from a point source in global coordinates.

        Directions are sampled uniformly within the emission cone using
        spherical coordinates. Wavelengths are Monte Carlo sampled from
        the spectrum.

        Args:
            ray_id: Unique identifiers for the rays to generate, shape (N,).
            rng: Keyed PCG32 RNG.

        Returns:
            NSQRayBundle with all rays alive.
        """
        num_rays = len(ray_id)
        bounce0 = np.zeros(num_rays, dtype=np.int32)
        translation, rot = _get_transform(self.cs)

        # Sample directions in local frame (cone around +z)
        cos_max = np.cos(np.radians(self.half_angle_deg))
        # Uniform sampling on spherical cap
        u1 = rng.uniform(ray_id, bounce0, EventSlot.SOURCE_U1)
        u2 = rng.uniform(ray_id, bounce0, EventSlot.SOURCE_U2)
        cos_theta = 1.0 - u1 * (1.0 - cos_max)
        sin_theta = np.sqrt(np.maximum(1.0 - cos_theta**2, 0.0))
        phi = 2.0 * np.pi * u2

        # Local directions
        dl_x = sin_theta * np.cos(phi)
        dl_y = sin_theta * np.sin(phi)
        dl_z = cos_theta

        dirs_local = np.stack([dl_x, dl_y, dl_z], axis=1)

        # Transform directions to global frame: d_global = d_local @ R^T
        dirs_global = dirs_local @ rot.T

        # Source position in global frame
        pos = translation  # shape (3,)

        # Sample wavelengths [µm]
        wavelengths = self.spectrum.sample(ray_id, bounce0, rng)

        flux_per_ray = self.total_flux / num_rays

        # Initialize n_current/k_current from medium if provided
        medium = getattr(self, "medium", None)
        if medium is not None:
            n_init = np.asarray(medium.n(wavelengths), dtype=float)
            if np.ndim(n_init) == 0:
                n_init = np.full(num_rays, float(n_init))
            k_init = np.asarray(medium.k(wavelengths), dtype=float)
            if np.ndim(k_init) == 0:
                k_init = np.full(num_rays, float(k_init))
        else:
            n_init = np.ones(num_rays)
            k_init = np.zeros(num_rays)

        return NSQRayBundle(
            x=np.full(num_rays, pos[0]),
            y=np.full(num_rays, pos[1]),
            z=np.full(num_rays, pos[2]),
            L=dirs_global[:, 0].copy(),
            M=dirs_global[:, 1].copy(),
            N=dirs_global[:, 2].copy(),
            flux=np.full(num_rays, flux_per_ray),
            wavelength=wavelengths,
            n_current=n_init,
            bounce=bounce0,
            alive=np.ones(num_rays, dtype=bool),
            ray_id=ray_id,
            k_current=k_init,
        )
