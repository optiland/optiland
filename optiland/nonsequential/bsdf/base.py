"""Base BSDF for Non-Sequential Raytracing.

Kramer Harrison, 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from optiland.nonsequential.rng import NSQRng


class BaseBSDF(ABC):
    """Abstract base class for bidirectional scattering distribution functions.

    All BSDF implementations must support vectorized operation over N rays.
    Array operations must be compatible with both NumPy and CuPy arrays.

    Lobes are explicitly REFLECT or TRANSMIT: ``sample()`` returns,
    alongside each scattered direction, whether that particular ray's draw
    landed in the reflective hemisphere (same side as the incident ray) or
    the transmissive one (far side). A surface's own optical topology
    decides what that means physically -- a mirror has no far side to
    transmit into, so ``ReflectiveComponent`` ignores the flag, while
    ``RefractiveComponent`` uses it to pick which of its two adjacent media
    (``material_front``/``material_back``) a scattered ray is now in:
    the medium a scattered ray ends up in is decided by its own lobe choice,
    never by the independent Fresnel branch draw that only applies to
    unscattered rays.
    """

    @abstractmethod
    def sample(
        self,
        num_rays: int,
        incident_dirs: np.ndarray,
        normals: np.ndarray,
        wavelengths: np.ndarray,
        rng: NSQRng,
        ray_id: np.ndarray,
        bounce: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample scattered ray directions, flux weights, and lobe side.

        Args:
            num_rays: Number of rays to scatter.
            incident_dirs: Incident ray directions, shape (N, 3), unit vectors.
            normals: Surface normals at hit points, shape (N, 3), unit vectors
                pointing toward the incoming ray side.
            wavelengths: Per-ray wavelengths [nm], shape (N,).
            rng: Keyed PCG32 RNG.
            ray_id: Per-ray identifiers, shape (N,), for keying the draw.
            bounce: Per-ray bounce/step index, shape (N,), for keying the
                draw.

        Returns:
            A tuple (scattered_dirs, flux_weights, transmitted) where:
                - scattered_dirs: Scattered unit direction vectors, shape (N, 3).
                - flux_weights: Relative flux weights in [0, 1], shape (N,).
                - transmitted: Boolean mask, shape (N,). True where the
                  returned direction is on the transmissive (far) side of
                  the surface; False for the reflective (incident) side.
                  A purely reflective BSDF (e.g. ``SpecularBRDF``) returns
                  all-False.
        """

    @abstractmethod
    def reflectance(
        self,
        incident_dirs: np.ndarray,
        normals: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """Total hemispherical reflectance for Russian-roulette decisions.

        Args:
            incident_dirs: Incident ray directions, shape (N, 3), unit vectors.
            normals: Surface normals at hit points, shape (N, 3), unit vectors.
            wavelengths: Per-ray wavelengths [nm], shape (N,).

        Returns:
            Total reflectance values in [0, 1], shape (N,).
        """
