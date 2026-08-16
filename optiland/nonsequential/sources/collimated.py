"""Collimated source for Non-Sequential Raytracing.

Parallel beam with circular aperture. Propagates along the local +z axis.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

import optiland.backend as be
from optiland.nonsequential._utils import as_detached_param
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
        medium: Medium the source is embedded in.
    """

    def __init__(
        self,
        cs: CoordinateSystem,
        spectrum: Spectrum,
        total_flux: float = 1.0,
        aperture_radius: float = 5.0,
        profile: Literal["tophat", "gaussian"] = "tophat",
        gaussian_sigma: float | None = None,
        medium=None,
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
            medium: Medium the source is embedded in (default: vacuum).
        """
        super().__init__(cs, spectrum, total_flux)
        self.aperture_radius = as_detached_param(
            aperture_radius, "aperture_radius", "CollimatedSource"
        )
        self.profile = profile
        self.gaussian_sigma = (
            as_detached_param(gaussian_sigma, "gaussian_sigma", "CollimatedSource")
            if gaussian_sigma is not None
            else self.aperture_radius / 2.0
        )
        self.medium = medium

    def generate(self, num_rays: int, rng: np.random.Generator) -> NSQRayBundle:
        """Generate collimated rays in global coordinates.

        Positions are sampled within the circular aperture. All directions
        are along the local +z axis.

        Args:
            num_rays: Number of rays to generate.
            rng: NumPy random generator.

        Returns:
            NSQRayBundle with all rays alive and parallel directions.
        """
        translation, rot = _get_transform(self.cs)

        if self.profile == "gaussian":
            # Sample truncated Gaussian on disk
            lx, ly = self._sample_gaussian_disk(num_rays, rng)
        else:
            # Uniform disk sampling
            u1 = rng.random(num_rays)
            u2 = rng.random(num_rays)
            r = self.aperture_radius * np.sqrt(u1)
            phi = 2.0 * np.pi * u2
            lx = r * np.cos(phi)
            ly = r * np.sin(phi)

        lz_pos = np.zeros(num_rays)

        # All rays point in local +z direction
        dirs_local = np.zeros((num_rays, 3))
        dirs_local[:, 2] = 1.0

        pos_local = np.stack([lx, ly, lz_pos], axis=1)

        # Transform to global frame
        pos_global = pos_local @ rot.T + translation
        dirs_global = dirs_local @ rot.T

        # Sample wavelengths [µm]
        wavelengths = self.spectrum.sample(num_rays, rng)
        # Divide preserves torch tensor when total_flux is a Tensor (for autograd)
        flux_per_ray = self.total_flux / num_rays

        # Initialize n_current from medium if provided
        medium = getattr(self, "medium", None)
        if medium is not None:
            n_init = np.asarray(medium.n(wavelengths), dtype=float)
            if np.ndim(n_init) == 0:
                n_init = np.full(num_rays, float(n_init))
        else:
            n_init = np.ones(num_rays)

        return NSQRayBundle(
            x=pos_global[:, 0].copy(),
            y=pos_global[:, 1].copy(),
            z=pos_global[:, 2].copy(),
            L=dirs_global[:, 0].copy(),
            M=dirs_global[:, 1].copy(),
            N=dirs_global[:, 2].copy(),
            # be.ones * flux_per_ray keeps the torch autograd graph when
            # total_flux is a Tensor; falls back to numpy when it's a float.
            flux=be.ones(num_rays) * flux_per_ray,
            wavelength=wavelengths,
            n_current=n_init,
            bounce=np.zeros(num_rays, dtype=np.int32),
            alive=np.ones(num_rays, dtype=bool),
        )

    def _sample_gaussian_disk(
        self, num_rays: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample positions from a truncated Gaussian disk (rejection sampling).

        Args:
            num_rays: Number of samples required.
            rng: NumPy random generator.

        Returns:
            Tuple (x, y) of position arrays, each shape (num_rays,).
        """
        x_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []
        num_collected = 0
        max_r2 = self.aperture_radius**2

        while num_collected < num_rays:
            num_sample = max(num_rays * 2, 1000)
            x = rng.normal(0.0, self.gaussian_sigma, num_sample)
            y = rng.normal(0.0, self.gaussian_sigma, num_sample)
            mask = x**2 + y**2 <= max_r2
            x_list.append(x[mask])
            y_list.append(y[mask])
            num_collected += mask.sum()

        lx = np.concatenate(x_list)[:num_rays]
        ly = np.concatenate(y_list)[:num_rays]
        return lx, ly
