"""Extended source for Non-Sequential Raytracing.

Uniform area source (rectangular or circular) with Lambertian or cone emission.

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


class ExtendedSource(BaseNSQSource):
    """Uniform area emitter on a rectangular or circular surface.

    The source surface lies in the local x-y plane (z=0). Emission
    direction is either Lambertian (cosine-weighted hemisphere) or
    confined to a cone around the local +z axis.

    Attributes:
        cs: Coordinate system (local z = emission axis).
        spectrum: Wavelength distribution.
        total_flux: Total emitted flux [W].
        width: Source width [mm] (used for rectangular aperture).
        height: Source height [mm] (used for rectangular aperture).
        aperture_radius: Circular aperture radius [mm]. If set, overrides
            width/height for a circular source.
        half_angle_deg: Half-angle of emission cone [deg].
            90 = Lambertian hemisphere.
        medium: Medium the source is embedded in.
    """

    def __init__(
        self,
        cs: CoordinateSystem,
        spectrum: Spectrum,
        total_flux: float = 1.0,
        width: float = 1.0,
        height: float = 1.0,
        aperture_radius: float | None = None,
        half_angle_deg: float = 90.0,
        medium=None,
    ) -> None:
        """Initialize ExtendedSource.

        Args:
            cs: Coordinate system.
            spectrum: Wavelength distribution.
            total_flux: Total emitted flux [W].
            width: Source width [mm] (rectangular aperture).
            height: Source height [mm] (rectangular aperture).
            aperture_radius: Circular aperture radius [mm]. Overrides
                width/height if set.
            half_angle_deg: Half-angle of emission cone [deg].
            medium: Medium the source is embedded in (default: vacuum).
        """
        super().__init__(cs, spectrum, total_flux)
        self.width = as_detached_param(width, "width", "ExtendedSource")
        self.height = as_detached_param(height, "height", "ExtendedSource")
        self.aperture_radius = (
            as_detached_param(aperture_radius, "aperture_radius", "ExtendedSource")
            if aperture_radius is not None
            else None
        )
        self.half_angle_deg = as_detached_param(
            half_angle_deg, "half_angle_deg", "ExtendedSource"
        )
        self.medium = medium

    def generate(self, ray_id: np.ndarray, rng: NSQRng) -> NSQRayBundle:
        """Generate rays from the extended source in global coordinates.

        Args:
            ray_id: Unique identifiers for the rays to generate, shape (N,).
            rng: Keyed PCG32 RNG.

        Returns:
            NSQRayBundle with all rays alive.
        """
        num_rays = len(ray_id)
        bounce0 = np.zeros(num_rays, dtype=np.int32)
        translation, rot = _get_transform(self.cs)

        # Sample positions on source surface (local x-y plane)
        if self.aperture_radius is not None:
            # Circular aperture: uniform disk sampling
            u1 = rng.uniform(ray_id, bounce0, EventSlot.SOURCE_U1)
            u2 = rng.uniform(ray_id, bounce0, EventSlot.SOURCE_U2)
            r = self.aperture_radius * np.sqrt(u1)
            phi_pos = 2.0 * np.pi * u2
            lx = r * np.cos(phi_pos)
            ly = r * np.sin(phi_pos)
        else:
            # Rectangular aperture
            u1 = rng.uniform(ray_id, bounce0, EventSlot.SOURCE_U1)
            u2 = rng.uniform(ray_id, bounce0, EventSlot.SOURCE_U2)
            lx = (u1 - 0.5) * self.width
            ly = (u2 - 0.5) * self.height

        lz_pos = np.zeros(num_rays)

        # Sample emission directions (Lambertian or cone)
        cos_max = np.cos(np.radians(self.half_angle_deg))
        u1d = rng.uniform(ray_id, bounce0, EventSlot.SOURCE_U3)
        u2d = rng.uniform(ray_id, bounce0, EventSlot.SOURCE_U4)

        if self.half_angle_deg >= 90.0:
            # Cosine-weighted hemisphere (Lambertian)
            cos_theta = np.sqrt(u1d)
        else:
            cos_theta = 1.0 - u1d * (1.0 - cos_max)

        sin_theta = np.sqrt(np.maximum(1.0 - cos_theta**2, 0.0))
        phi_dir = 2.0 * np.pi * u2d

        dl_x = sin_theta * np.cos(phi_dir)
        dl_y = sin_theta * np.sin(phi_dir)
        dl_z = cos_theta

        dirs_local = np.stack([dl_x, dl_y, dl_z], axis=1)
        pos_local = np.stack([lx, ly, lz_pos], axis=1)

        # Transform to global frame
        # pos_global = pos_local @ R^T + t
        pos_global = pos_local @ rot.T + translation
        dirs_global = dirs_local @ rot.T

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
            x=pos_global[:, 0].copy(),
            y=pos_global[:, 1].copy(),
            z=pos_global[:, 2].copy(),
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
