"""Spectral irradiance result for Non-Sequential Raytracing.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


class SpectralResult:
    """Per-wavelength irradiance on a planar detector.

    Attributes:
        irradiance: Irradiance per wavelength bin [W/mm^2], shape
            (ny, nx, n_lambda). Flux is binned, not divided by bin width, so
            summing over the last axis gives the broadband irradiance.
        x_coords: Bin centre x-coordinates [mm].
        y_coords: Bin centre y-coordinates [mm].
        wavelengths: Wavelength bin centres [µm].
        total_flux: Total flux recorded [W].
        num_rays_hit: Number of rays recorded.
    """

    def __init__(
        self,
        irradiance: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        wavelengths: np.ndarray,
        total_flux: float,
        num_rays_hit: int,
    ) -> None:
        """Initialize SpectralResult.

        Args:
            irradiance: 3D irradiance array [W/mm^2], shape (ny, nx, n_lambda).
            x_coords: Bin centre x-coordinates [mm].
            y_coords: Bin centre y-coordinates [mm].
            wavelengths: Wavelength bin centres [µm].
            total_flux: Total detected flux [W].
            num_rays_hit: Number of rays that contributed.
        """
        self.irradiance = irradiance
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.wavelengths = wavelengths
        self.total_flux = float(total_flux)
        self.num_rays_hit = int(num_rays_hit)

    def plot_at_wavelength(self, wl: float, ax=None, **kwargs):
        """Plot the irradiance map at the wavelength closest to wl.

        Args:
            wl: Target wavelength [µm].
            ax: Optional Matplotlib Axes.
            **kwargs: Additional arguments passed to imshow.

        Returns:
            The Matplotlib Figure object.
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415

        idx = int(np.argmin(np.abs(self.wavelengths - wl)))
        actual_wl = float(self.wavelengths[idx])

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        extent = [
            self.x_coords[0],
            self.x_coords[-1],
            self.y_coords[0],
            self.y_coords[-1],
        ]
        im = ax.imshow(
            self.irradiance[:, :, idx],
            origin="lower",
            extent=extent,
            aspect="equal",
            **kwargs,
        )
        plt.colorbar(im, ax=ax, label="Irradiance [W/mm^2]")
        ax.set_title(f"Irradiance at wl={actual_wl:.3f} µm")
        return fig

    def save(self, path: str | Path) -> None:
        """Save spectral result to a .npz file.

        Args:
            path: Output file path.
        """
        np.savez(
            path,
            irradiance=self.irradiance,
            x_coords=self.x_coords,
            y_coords=self.y_coords,
            wavelengths=self.wavelengths,
            total_flux=self.total_flux,
            num_rays_hit=self.num_rays_hit,
        )
