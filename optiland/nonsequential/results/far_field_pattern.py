"""Far-field angular pattern result for Non-Sequential Raytracing.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


class FarFieldPattern:
    """Angular flux distribution in the far field.

    Attributes:
        intensity: Intensity [W/sr], shape (n_theta, n_phi).
        theta: Polar angle bin centres [deg], shape (n_theta,).
        phi: Azimuthal angle bin centres [deg], shape (n_phi,).
        total_flux: Total flux recorded [W].
        num_rays_hit: Number of rays recorded.
    """

    def __init__(
        self,
        intensity: np.ndarray,
        theta: np.ndarray,
        phi: np.ndarray,
        total_flux: float,
        num_rays_hit: int,
    ) -> None:
        """Initialize FarFieldPattern.

        Args:
            intensity: Intensity [W/sr], shape (n_theta, n_phi).
            theta: Polar angle centres [deg].
            phi: Azimuthal angle centres [deg].
            total_flux: Total detected flux [W].
            num_rays_hit: Number of rays that contributed.
        """
        self.intensity = intensity
        self.theta = theta
        self.phi = phi
        self.total_flux = float(total_flux)
        self.num_rays_hit = int(num_rays_hit)

    def plot(self, ax=None, projection: str = "polar", **kwargs):
        """Plot the far-field pattern.

        Args:
            ax: Optional Matplotlib Axes. If None, a new figure is created.
            projection: Plot projection type ('polar' or 'cartesian').
            **kwargs: Additional arguments passed to the plot function.

        Returns:
            The Matplotlib Figure object.
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        # Sum over phi for 1D polar plot
        intensity_1d = self.intensity.sum(axis=1)
        ax.plot(self.theta, intensity_1d, **kwargs)
        ax.set_xlabel("theta [deg]")
        ax.set_ylabel("Intensity [W/sr] (integrated over phi)")
        ax.set_title(f"Far-Field Pattern -- {self.num_rays_hit} rays")
        return fig

    def save(self, path: str | Path) -> None:
        """Save far-field pattern to a .npz file.

        Args:
            path: Output file path.
        """
        np.savez(
            path,
            intensity=self.intensity,
            theta=self.theta,
            phi=self.phi,
            total_flux=self.total_flux,
            num_rays_hit=self.num_rays_hit,
        )
