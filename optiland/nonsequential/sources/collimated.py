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
from optiland.nonsequential.rng import EventSlot
from optiland.nonsequential.sources.base import BaseNSQSource, Spectrum

if TYPE_CHECKING:
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.rng import NSQRng

# Bounded rejection-sampling attempts for the truncated Gaussian disk. Each
# round accepts ~1 - exp(-2) ~ 86% of samples for the default sigma =
# radius / 2, so 32 rounds leaves an astronomically small failure
# probability; any ray still rejected after that is clamped to the
# boundary (a deterministic, keyed fallback -- never an unbounded loop).
_MAX_GAUSSIAN_ATTEMPTS = 32


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

    def generate(self, ray_id: np.ndarray, rng: NSQRng) -> NSQRayBundle:
        """Generate collimated rays in global coordinates.

        Positions are sampled within the circular aperture. All directions
        are along the local +z axis.

        Args:
            ray_id: Unique identifiers for the rays to generate, shape (N,).
            rng: Keyed PCG32 RNG.

        Returns:
            NSQRayBundle with all rays alive and parallel directions.
        """
        num_rays = len(ray_id)
        bounce0 = np.zeros(num_rays, dtype=np.int32)
        translation, rot = _get_transform(self.cs)

        if self.profile == "gaussian":
            # Sample truncated Gaussian on disk
            lx, ly = self._sample_gaussian_disk(ray_id, bounce0, rng)
        else:
            # Uniform disk sampling
            u1 = rng.uniform(ray_id, bounce0, EventSlot.SOURCE_U1)
            u2 = rng.uniform(ray_id, bounce0, EventSlot.SOURCE_U2)
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
        wavelengths = self.spectrum.sample(ray_id, bounce0, rng)
        # Divide preserves torch tensor when total_flux is a Tensor (for autograd)
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
            # be.ones * flux_per_ray keeps the torch autograd graph when
            # total_flux is a Tensor; falls back to numpy when it's a float.
            flux=be.ones(num_rays) * flux_per_ray,
            wavelength=wavelengths,
            n_current=n_init,
            bounce=bounce0,
            alive=np.ones(num_rays, dtype=bool),
            ray_id=ray_id,
            k_current=k_init,
        )

    def _sample_gaussian_disk(
        self, ray_id: np.ndarray, bounce0: np.ndarray, rng: NSQRng
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample positions from a truncated Gaussian disk (rejection sampling).

        Each ray gets its own bounded rejection sequence keyed by its own
        id: attempt ``k`` draws a fresh (u1, u2) pair via
        ``offset=k`` on the same event slots, so the sequence is a pure
        function of the ray id and never depends on how many other rays
        are still rejecting in the same batch.

        Args:
            ray_id: Unique identifiers for the rays to generate, shape (N,).
            bounce0: Zero bounce array, shape (N,).
            rng: Keyed PCG32 RNG.

        Returns:
            Tuple (x, y) of position arrays, each shape (N,).
        """
        num_rays = len(ray_id)
        max_r2 = self.aperture_radius**2
        lx = np.zeros(num_rays)
        ly = np.zeros(num_rays)
        pending = np.ones(num_rays, dtype=bool)

        for attempt in range(_MAX_GAUSSIAN_ATTEMPTS):
            if not pending.any():
                break
            u1 = rng.uniform(ray_id, bounce0, EventSlot.SOURCE_U1, offset=attempt)
            u2 = rng.uniform(ray_id, bounce0, EventSlot.SOURCE_U2, offset=attempt)
            # Box-Muller transform: (u1, u2) -> independent standard normals.
            r_bm = np.sqrt(-2.0 * np.log(np.maximum(u1, 1e-300)))
            theta_bm = 2.0 * np.pi * u2
            x = self.gaussian_sigma * r_bm * np.cos(theta_bm)
            y = self.gaussian_sigma * r_bm * np.sin(theta_bm)
            accept = pending & (x**2 + y**2 <= max_r2)
            lx = np.where(accept, x, lx)
            ly = np.where(accept, y, ly)
            pending = pending & ~accept

        if pending.any():
            # Exhausted the attempt budget (astronomically unlikely): clamp
            # to the boundary along the last-drawn direction rather than
            # looping unboundedly or silently keeping an out-of-aperture
            # sample.
            theta_fallback = (
                2.0
                * np.pi
                * rng.uniform(
                    ray_id, bounce0, EventSlot.SOURCE_U2, offset=_MAX_GAUSSIAN_ATTEMPTS
                )
            )
            r_fallback = self.aperture_radius
            lx = np.where(pending, r_fallback * np.cos(theta_fallback), lx)
            ly = np.where(pending, r_fallback * np.sin(theta_fallback), ly)

        return lx, ly
