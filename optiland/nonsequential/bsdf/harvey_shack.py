"""Harvey-Shack (ABg) BSDF for Non-Sequential Raytracing.

Models micro-roughness scatter from optical surfaces.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np

import optiland.backend as be
from optiland.backend.utils import to_numpy
from optiland.nonsequential.bsdf.base import BaseBSDF

# Largest reachable direction-cosine offset: both the specular and the
# scattered direction lie in the unit disk, so |beta - beta0| <= 2.
_BETA_MAX = 2.0
# Resolution of the tabulated radial inverse CDF.
_TABLE_SIZE = 4096


class HarveyShackBSDF(BaseBSDF):
    """Harvey-Shack / ABg scatter model for surface micro-roughness.

    The ABg model is a simplified form of the Harvey-Shack theory:

        BSDF(beta - beta0) = b0 / (1 + |beta - beta0| / l0)^s

    where beta and beta0 are direction cosines of the scattered and specular
    directions, b0 is the scatter level at beta=beta0, l0 is the break
    frequency, and s is the roll-off slope.

    Attributes:
        b0: Scatter amplitude at zero angle [sr^-1].
        l0: Break-point spatial frequency (dimensionless direction cosine).
        s: Power-law roll-off slope (positive).
    """

    def __init__(self, b0: float, l0: float, s: float) -> None:
        """Initialize HarveyShackBSDF.

        Args:
            b0: Scatter amplitude at zero angle [sr^-1].
            l0: Break-point spatial frequency in direction-cosine space.
            s: Power-law roll-off exponent (positive).
        """
        self.b0 = float(b0)
        self.l0 = float(l0)
        self.s = float(s)
        self._beta_grid: np.ndarray | None = None
        self._cdf_grid: np.ndarray | None = None
        self._tis: float | None = None

    def _abg(self, beta: np.ndarray) -> np.ndarray:
        """Evaluate the ABg BSDF at a direction-cosine offset.

        Args:
            beta: Magnitude of the direction-cosine offset from specular.

        Returns:
            BSDF value [sr^-1].
        """
        return self.b0 / (1.0 + (beta / self.l0) ** self.s)

    def _build_tables(self) -> None:
        """Build the radial inverse-CDF table and the total integrated scatter.

        In direction-cosine space the projected solid angle is
        ``cos(theta) dOmega = d(beta_x) d(beta_y)``, so the radial measure is
        ``2 * pi * beta d(beta)`` and

            TIS = integral of BSDF(beta) * 2 * pi * beta d(beta)

        over the reachable range ``beta <= _BETA_MAX``. TIS is the fraction of
        incident power the surface scatters, and it is what each scattered ray
        carries as its weight.
        """
        beta = np.linspace(0.0, _BETA_MAX, _TABLE_SIZE)
        radial = self._abg(beta) * 2.0 * np.pi * beta

        # Cumulative trapezoidal integral.
        cdf = np.concatenate(
            [[0.0], np.cumsum(0.5 * (radial[1:] + radial[:-1]) * np.diff(beta))]
        )
        self._tis = float(cdf[-1])
        # Normalise to a proper CDF. A degenerate (all-zero) integrand would
        # leave the table flat, so fall back to a uniform CDF in that case.
        self._cdf_grid = cdf / cdf[-1] if cdf[-1] > 0.0 else np.linspace(0, 1, cdf.size)
        self._beta_grid = beta

    @property
    def total_integrated_scatter(self) -> float:
        """Fraction of incident power scattered by this surface, in [0, 1].

        Returns:
            TIS, clipped to 1.0.
        """
        if self._tis is None:
            self._build_tables()
        return min(float(self._tis), 1.0)

    def sample(
        self,
        num_rays: int,
        incident_dirs: np.ndarray,
        normals: np.ndarray,
        wavelengths: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample scattered directions from the ABg lobe about the specular ray.

        The scatter offset is drawn directly from the ABg distribution in
        direction-cosine space: the radial magnitude ``|beta - beta0|`` comes
        from a tabulated inverse CDF of ``BSDF(beta) * 2 * pi * beta`` and the
        azimuth is uniform. Rays keep their full flux, so the surface acts as
        a mirror whose reflection is blurred by the ABg lobe: a polished
        surface (small ``l0``) stays near-specular, a rough one spreads.

        To model the physically scaled picture instead, a bright specular beam
        plus a faint scatter halo, set the surface's ``scatter_fraction`` to
        :attr:`total_integrated_scatter`. Then a TIS fraction of the rays enter
        the halo and the rest reflect specularly.

        Sampling the lobe directly matters: drawing from a cosine-weighted
        hemisphere and correcting with a clipped ``BSDF / cos`` weight (the
        previous approach) puts essentially every sample where the ABg lobe is
        negligible, which drove surface throughput to ~1e-7 of the incident
        flux and made the model behave as a black absorber.

        Sampling is detached (numpy); weights are plain scalars.

        Args:
            num_rays: Number of rays.
            incident_dirs: Incident directions, shape (N, 3).
            normals: Surface normals, shape (N, 3).
            wavelengths: Wavelengths [µm], shape (N,).
            rng: NumPy random generator.

        Returns:
            (scattered_dirs, flux_weights).
        """
        if rng is None:
            rng = np.random.default_rng()
        if self._beta_grid is None:
            self._build_tables()

        n_np = np.asarray(to_numpy(normals), dtype=np.float64)
        d_np = np.asarray(to_numpy(incident_dirs), dtype=np.float64)

        # Specular direction: d - 2(d.n)n
        cos_i = (d_np * n_np).sum(axis=1, keepdims=True)
        d_spec = d_np - 2.0 * cos_i * n_np
        # Rays that hit nothing carry zero direction and normal, so the
        # specular vector is zero. Guard the normalisation: a NaN here
        # propagates into the returned weights for every ray.
        d_spec_norm = (d_spec * d_spec).sum(axis=1, keepdims=True) ** 0.5
        valid = d_spec_norm[:, 0] > 1e-12
        d_spec = np.divide(
            d_spec, d_spec_norm, out=np.zeros_like(d_spec), where=d_spec_norm > 1e-12
        )

        from optiland.nonsequential.bsdf.lambertian import (  # noqa: PLC0415
            _orthonormal_basis,
        )

        t_vec, b_vec = _orthonormal_basis(n_np)

        # Specular direction expressed in the local tangent frame.
        beta0_x = (d_spec * t_vec).sum(axis=1)
        beta0_y = (d_spec * b_vec).sum(axis=1)
        spec_normal_sign = np.sign((d_spec * n_np).sum(axis=1))
        spec_normal_sign[spec_normal_sign == 0.0] = 1.0

        # Radial offset from the tabulated inverse CDF; azimuth uniform.
        u_radial = rng.random(num_rays)
        u_azimuth = rng.random(num_rays)
        delta = np.interp(u_radial, self._cdf_grid, self._beta_grid)
        psi = 2.0 * np.pi * u_azimuth

        beta_x = beta0_x + delta * np.cos(psi)
        beta_y = beta0_y + delta * np.sin(psi)

        # An offset can land outside the unit disk, i.e. below the surface.
        # Those samples are not physically reachable: keep the specular
        # direction and give them zero weight rather than folding them back,
        # which would distort the lobe.
        beta_sq = beta_x**2 + beta_y**2
        reachable = (beta_sq < 1.0) & valid
        normal_comp = np.sqrt(np.clip(1.0 - beta_sq, 0.0, None))

        scattered = (
            beta_x[:, None] * t_vec
            + beta_y[:, None] * b_vec
            + (spec_normal_sign * normal_comp)[:, None] * n_np
        )
        scattered = np.where(reachable[:, None], scattered, d_spec)

        norms = (scattered * scattered).sum(axis=1, keepdims=True) ** 0.5
        scattered = np.divide(
            scattered, norms, out=np.zeros_like(scattered), where=norms > 1e-12
        )

        # Full flux: the lobe redistributes energy rather than removing it.
        # The physical scatter level is applied via ``scatter_fraction``,
        # for which :attr:`total_integrated_scatter` is the natural value.
        flux_weights = np.where(reachable, 1.0, 0.0)

        return be.array(scattered.astype(np.float64)), be.array(flux_weights)

    def reflectance(
        self,
        incident_dirs: np.ndarray,
        normals: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """Return the fraction of incident power redistributed by the lobe.

        The sampler conserves energy (rays keep full flux and are only
        redirected), so this is 1.0. The ABg scatter level itself is
        :attr:`total_integrated_scatter`, which is what a
        ``scatter_fraction`` should be set to for a physically scaled halo.

        Args:
            incident_dirs: Incident directions, shape (N, 3).
            normals: Surface normals, shape (N, 3).
            wavelengths: Wavelengths [µm], shape (N,).

        Returns:
            Approximate reflectance values, shape (N,).
        """
        return be.ones(incident_dirs.shape[0])
