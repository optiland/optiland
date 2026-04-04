"""Point source for Non-Sequential Raytracing.

Isotropic emitter or emission into a defined solid angle cone.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optiland.nonsequential.components.base import _get_transform
from optiland.nonsequential.ray_bundle import NSQRayBundle
from optiland.nonsequential.sources.base import BaseNSQSource, Spectrum

if TYPE_CHECKING:
    from optiland.coordinate_system import CoordinateSystem


class PointSource(BaseNSQSource):
    """Point source emitting rays from a single position.

    Emission can be isotropic (full sphere) or confined to a cone around
    the local +z axis.

    Attributes:
        cs: Coordinate system (origin = source position).
        spectrum: Wavelength distribution.
        total_flux: Total emitted flux [W].
        half_angle_deg: Half-angle of the emission cone [deg]. 180 = isotropic.
    """

    def __init__(
        self,
        cs: CoordinateSystem,
        spectrum: Spectrum,
        total_flux: float = 1.0,
        half_angle_deg: float = 90.0,
    ) -> None:
        """Initialize PointSource.

        Args:
            cs: Coordinate system for source position.
            spectrum: Wavelength distribution.
            total_flux: Total emitted flux [W].
            half_angle_deg: Half-angle of emission cone [deg].
                90 = hemisphere, 180 = full sphere (isotropic).
        """
        super().__init__(cs, spectrum, total_flux)
        self.half_angle_deg = float(half_angle_deg)

    def generate(self, n_rays: int, rng: np.random.Generator) -> NSQRayBundle:
        """Generate rays from a point source in global coordinates.

        Directions are sampled uniformly within the emission cone using
        spherical coordinates. Wavelengths are Monte Carlo sampled from
        the spectrum.

        Args:
            n_rays: Number of rays to generate.
            rng: NumPy random generator.

        Returns:
            NSQRayBundle with all rays alive.
        """
        translation, rot = _get_transform(self.cs)

        # Sample directions in local frame (cone around +z)
        cos_max = np.cos(np.radians(self.half_angle_deg))
        # Uniform sampling on spherical cap
        u1 = rng.random(n_rays)
        u2 = rng.random(n_rays)
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

        # Sample wavelengths
        wavelengths = self.spectrum.sample(n_rays, rng)

        flux_per_ray = self.total_flux / n_rays

        return NSQRayBundle(
            x=np.full(n_rays, pos[0]),
            y=np.full(n_rays, pos[1]),
            z=np.full(n_rays, pos[2]),
            dx=dirs_global[:, 0].copy(),
            dy=dirs_global[:, 1].copy(),
            dz=dirs_global[:, 2].copy(),
            flux=np.full(n_rays, flux_per_ray),
            wavelength=wavelengths,
            n_current=np.ones(n_rays),
            bounce=np.zeros(n_rays, dtype=np.int32),
            alive=np.ones(n_rays, dtype=bool),
        )
