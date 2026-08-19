"""Lambertian BSDF for Non-Sequential Raytracing.

Cosine-weighted hemispherical scatter (diffuse surface).

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.backend.utils import to_numpy
from optiland.nonsequential.bsdf.base import BaseBSDF
from optiland.nonsequential.rng import EventSlot

if TYPE_CHECKING:
    from optiland.nonsequential.rng import NSQRng


class LambertianBSDF(BaseBSDF):
    """Cosine-weighted Lambertian diffuse scatter.

    Attributes:
        reflectance_value: Hemispherical diffuse reflectance in [0, 1].
        transmissive_fraction: Probability in [0, 1] that a given scatter
            event samples the *transmissive* hemisphere (the far side of the
            surface, e.g. a ground-glass diffuser) instead of the
            reflective one. Defaults to 0.0: a pure diffuse reflector,
            identical to this class's behaviour before D-5.
    """

    def __init__(
        self, reflectance_value: float = 1.0, transmissive_fraction: float = 0.0
    ) -> None:
        """Initialize LambertianBSDF.

        Args:
            reflectance_value: Total hemispherical reflectance in [0, 1].
                May be a torch Tensor with requires_grad=True for autograd.
            transmissive_fraction: Probability in [0, 1] that a scatter
                event lands in the transmissive hemisphere rather than the
                reflective one.
        """
        # Intentionally not float()-cast so torch tensors remain attached.
        self.reflectance_value = reflectance_value
        self.transmissive_fraction = float(transmissive_fraction)

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
        """Sample cosine-weighted hemisphere directions around +/- normals.

        Uses Malley's method: sample uniform disk, project to hemisphere.
        A per-ray draw against :attr:`transmissive_fraction` picks
        whether that hemisphere is centred on ``normals`` (reflective) or
        ``-normals`` (transmissive). Sampling is detached (keyed PCG32);
        weights are plain scalars.

        Args:
            num_rays: Number of rays.
            incident_dirs: Incident directions, shape (N, 3).
            normals: Surface normals, shape (N, 3), pointing toward ray side.
            wavelengths: Wavelengths [µm], shape (N,).
            rng: Keyed PCG32 RNG.
            ray_id: Per-ray identifiers, shape (N,).
            bounce: Per-ray bounce/step index, shape (N,).

        Returns:
            (scattered_dirs, flux_weights, transmitted); flux_weights =
            reflectance_value for every ray (the lobe redistributes energy
            within whichever hemisphere it lands in, it does not remove it).
        """
        # Sampling is inherently stochastic/detached -- use numpy throughout
        normals_np = np.asarray(to_numpy(normals), dtype=np.float64)

        if self.transmissive_fraction > 0.0:
            u_lobe = rng.uniform(ray_id, bounce, EventSlot.BSDF_LOBE_BRANCH)
            transmitted_np = u_lobe < self.transmissive_fraction
            hemisphere_np = np.where(transmitted_np[:, None], -normals_np, normals_np)
        else:
            transmitted_np = np.zeros(normals_np.shape[0], dtype=bool)
            hemisphere_np = normals_np

        r1 = rng.uniform(ray_id, bounce, EventSlot.BSDF_U1)
        r2 = rng.uniform(ray_id, bounce, EventSlot.BSDF_U2)

        # Malley's method
        phi = 2.0 * np.pi * r1
        cos_theta = np.sqrt(r2)
        sin_theta = np.sqrt(1.0 - r2)

        lx = sin_theta * np.cos(phi)
        ly = sin_theta * np.sin(phi)
        lz = cos_theta

        t_vec, b_vec = _orthonormal_basis(hemisphere_np)

        scattered = (
            lx[:, None] * t_vec + ly[:, None] * b_vec + lz[:, None] * hemisphere_np
        )
        norms = (scattered * scattered).sum(axis=1, keepdims=True) ** 0.5
        scattered = scattered / norms

        # Convert back to backend array type (detached; no grad required)
        scattered_be = be.array(scattered.astype(np.float64))
        # be.ones * reflectance_value preserves the autograd graph when
        # reflectance_value is a torch Tensor with requires_grad=True.
        weights_be = be.ones(num_rays) * self.reflectance_value

        return scattered_be, weights_be, be.array(transmitted_np)

    def reflectance(
        self,
        incident_dirs: np.ndarray,
        normals: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """Return total hemispherical reflectance.

        Args:
            incident_dirs: Incident directions, shape (N, 3).
            normals: Surface normals, shape (N, 3).
            wavelengths: Wavelengths [µm], shape (N,).

        Returns:
            Array of reflectance_value, shape (N,).
        """
        return be.ones(incident_dirs.shape[0]) * self.reflectance_value


def _orthonormal_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build two tangent vectors perpendicular to n (numpy, for detached sampling).

    Uses the branchless construction of Duff et al., *Building an Orthonormal
    Basis, Revisited* (JCGT 2017). The denominator ``sign + n_z`` has
    magnitude >= 1 for any unit ``n``, because ``sign`` carries the sign of
    ``n_z``, so no normalisation and no division by zero is involved.

    The previous construction took ``cross(n, ref)`` against a fixed reference
    axis and normalised the result. That divides by zero whenever ``n`` is
    parallel to ``ref`` and, more importantly, whenever ``n`` is the zero
    vector: BSDF sampling is evaluated for every ray in the bundle, and rays
    that hit nothing carry a zero normal. The resulting NaN directions were
    masked out of the forward result but still emitted a RuntimeWarning, and
    would poison gradients on the torch backend.

    Args:
        n: Normal vectors, shape (N, 3), already normalised. A zero vector
            yields an arbitrary but finite basis.

    Returns:
        Pair of tangent vectors (t, b), each shape (N, 3).
    """
    nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]

    sign = np.copysign(1.0, nz)
    a = -1.0 / (sign + nz)
    b = nx * ny * a

    t_vec = np.stack([1.0 + sign * nx * nx * a, sign * b, -sign * nx], axis=1)
    b_vec = np.stack([b, sign + ny * ny * a, -ny], axis=1)
    return t_vec, b_vec
