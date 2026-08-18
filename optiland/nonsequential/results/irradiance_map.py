"""Irradiance map result for Non-Sequential Raytracing.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


class IrradianceMap:
    """2D irradiance distribution on a planar detector.

    Attributes:
        data: Flat accumulated flux buffer (be-array, shape ny*nx).
            This is the differentiable handle; call ``.data.backward()``
            to propagate gradients through the detector image.
            ``None`` when constructed from legacy hard-splatted data.
        irradiance: Irradiance [W/mm^2], shape (ny, nx), as a NumPy array.
        x_coords: Bin centre x-coordinates [mm], shape (nx,).
        y_coords: Bin centre y-coordinates [mm], shape (ny,).
        total_flux: Total flux recorded [W]. Attached to the active
            backend's autograd graph -- a torch.Tensor when the
            underlying data is, so ``result.total_flux.backward()``
            propagates a gradient. Use :attr:`total_flux_float` for
            printing or any consumer that expects a plain Python float.
        num_rays_hit: Number of rays recorded on this detector.
    """

    def __init__(
        self,
        irradiance: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        total_flux,
        num_rays_hit: int,
        data=None,
    ) -> None:
        """Initialize IrradianceMap.

        Args:
            irradiance: 2D irradiance array [W/mm^2], shape (ny, nx).
            x_coords: Bin centre x-coordinates [mm], shape (nx,).
            y_coords: Bin centre y-coordinates [mm], shape (ny,).
            total_flux: Total detected flux [W]. May be a plain float or a
                backend array/tensor; kept attached if the latter.
            num_rays_hit: Number of rays that contributed.
            data: Flat accumulated flux be-array (shape ny*nx), optional.
                When provided, this is the attached differentiable buffer
                from which irradiance was computed. Defaults to None.
        """
        self.data = data  # attached tensor or numpy array (flat, ny*nx)
        self.irradiance = irradiance
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.total_flux = total_flux
        self.num_rays_hit = int(num_rays_hit)

    @property
    def total_flux_float(self) -> float:
        """Total detected flux [W] as a plain, detached Python float.

        Use this for printing, formatting, or any code path that cannot
        accept a gradient-carrying tensor; :attr:`total_flux` stays attached
        for differentiation.
        """
        from optiland.backend.utils import to_numpy  # noqa: PLC0415

        return float(to_numpy(self.total_flux))

    def plot(self, ax=None, **kwargs):
        """Plot the irradiance map.

        Args:
            ax: Optional Matplotlib Axes. If None, a new figure is created.
            **kwargs: Additional arguments passed to imshow.

        Returns:
            The Matplotlib Figure object.
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415

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
            self.irradiance,
            origin="lower",
            extent=extent,
            aspect="equal",
            **kwargs,
        )
        plt.colorbar(im, ax=ax, label="Irradiance [W/mm^2]")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title(
            f"Irradiance Map -- {self.num_rays_hit} rays, {self.total_flux_float:.3g} W"
        )
        return fig

    def save(self, path: str | Path) -> None:
        """Save irradiance map to a .npz file.

        Args:
            path: Output file path.
        """
        np.savez(
            path,
            irradiance=self.irradiance,
            x_coords=self.x_coords,
            y_coords=self.y_coords,
            total_flux=self.total_flux_float,
            num_rays_hit=self.num_rays_hit,
        )

    def to_numpy(self) -> np.ndarray:
        """Return the irradiance array as a NumPy array.

        Returns:
            The irradiance array, shape (ny, nx).
        """
        return np.asarray(self.irradiance)
