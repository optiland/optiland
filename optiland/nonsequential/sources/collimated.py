"""Collimated source for Non-Sequential Raytracing.

Parallel beam with circular aperture. Propagates along the local +z axis.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from optiland.nonsequential.components.base import _get_transform
from optiland.nonsequential.ray_bundle import NSQRayBundle
from optiland.nonsequential.sources.base import BaseNSQSource, Spectrum

if TYPE_CHECKING:
    from optiland.coordinate_system import CoordinateSystem


class CollimatedSource(BaseNSQSource):
    """Parallel collimated beam with circular aperture.

    All rays propagate along the local +z axis (after coordinate
    transformation to global frame). Intensity profile is either
    top-hat (uniform) or truncated Gaussian.

    Attributes:
        cs: Coordinate system (local z = beam propagation axis).
        spectrum: Wavelength distribution.
        total_flux: Total beam flux [W].
        aperture_radius: Beam aperture radius [mm].
        profile: Intensity profile ('tophat' or 'gaussian').
        gaussian_sigma: Gaussian sigma [mm] (used when profile='gaussian').
    """

    def __init__(
        self,
        cs: CoordinateSystem,
        spectrum: Spectrum,
        total_flux: float = 1.0,
        aperture_radius: float = 5.0,
        profile: Literal["tophat", "gaussian"] = "tophat",
        gaussian_sigma: float | None = None,
    ) -> None:
        """Initialize CollimatedSource.

        Args:
            cs: Coordinate system.
            spectrum: Wavelength distribution.
            total_flux: Total beam flux [W].
            aperture_radius: Beam aperture radius [mm].
            profile: Intensity profile ('tophat' or 'gaussian').
            gaussian_sigma: Gaussian standard deviation [mm].
                Defaults to aperture_radius / 2 if None.
        """
        super().__init__(cs, spectrum, total_flux)
        self.aperture_radius = float(aperture_radius)
        self.profile = profile
        self.gaussian_sigma = (
            float(gaussian_sigma)
            if gaussian_sigma is not None
            else aperture_radius / 2.0
        )

    def generate(self, n_rays: int, rng: np.random.Generator) -> NSQRayBundle:
        """Generate collimated rays in global coordinates.

        Positions are sampled within the circular aperture. All directions
        are along the local +z axis.

        Args:
            n_rays: Number of rays to generate.
            rng: NumPy random generator.

        Returns:
            NSQRayBundle with all rays alive and parallel directions.
        """
        translation, rot = _get_transform(self.cs)

        if self.profile == "gaussian":
            # Sample truncated Gaussian on disk
            lx, ly = self._sample_gaussian_disk(n_rays, rng)
        else:
            # Uniform disk sampling
            u1 = rng.random(n_rays)
            u2 = rng.random(n_rays)
            r = self.aperture_radius * np.sqrt(u1)
            phi = 2.0 * np.pi * u2
            lx = r * np.cos(phi)
            ly = r * np.sin(phi)

        lz_pos = np.zeros(n_rays)

        # All rays point in local +z direction
        dirs_local = np.zeros((n_rays, 3))
        dirs_local[:, 2] = 1.0

        pos_local = np.stack([lx, ly, lz_pos], axis=1)

        # Transform to global frame
        pos_global = pos_local @ rot.T + translation
        dirs_global = dirs_local @ rot.T

        wavelengths = self.spectrum.sample(n_rays, rng)
        flux_per_ray = self.total_flux / n_rays

        return NSQRayBundle(
            x=pos_global[:, 0].copy(),
            y=pos_global[:, 1].copy(),
            z=pos_global[:, 2].copy(),
            dx=dirs_global[:, 0].copy(),
            dy=dirs_global[:, 1].copy(),
            dz=dirs_global[:, 2].copy(),
            flux=np.full(n_rays, flux_per_ray),
            wavelength=wavelengths,
            n_current=np.ones(n_rays),
            bounce=np.zeros(n_rays, dtype=np.int32),
            alive=np.ones(n_rays, dtype=bool),
        )

    def _sample_gaussian_disk(
        self, n_rays: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample positions from a truncated Gaussian disk (rejection sampling).

        Args:
            n_rays: Number of samples required.
            rng: NumPy random generator.

        Returns:
            Tuple (x, y) of position arrays, each shape (n_rays,).
        """
        x_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []
        n_collected = 0
        max_r2 = self.aperture_radius**2

        while n_collected < n_rays:
            n_sample = max(n_rays * 2, 1000)
            x = rng.normal(0.0, self.gaussian_sigma, n_sample)
            y = rng.normal(0.0, self.gaussian_sigma, n_sample)
            mask = x**2 + y**2 <= max_r2
            x_list.append(x[mask])
            y_list.append(y[mask])
            n_collected += mask.sum()

        lx = np.concatenate(x_list)[:n_rays]
        ly = np.concatenate(y_list)[:n_rays]
        return lx, ly
