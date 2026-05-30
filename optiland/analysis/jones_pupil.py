"""Jones Pupil Analysis

This module provides a Jones pupil analysis for optical systems.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

import optiland.backend as be
from optiland.analysis.base import BaseAnalysis
from optiland.rays import PolarizationState

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from optiland.optic import Optic


class JonesPupil(BaseAnalysis):
    """Generates and plots Jones pupil maps.

    This class computes the spatially resolved Jones matrix at the exit pupil
    (or image plane) as a function of normalized pupil coordinates. It visualizes
    the real and imaginary parts of the Jones matrix elements (Jxx, Jxy, Jyx, Jyy).

    Attributes:
        optic: Instance of the optic object to be assessed.
    Attributes:
        optic: Instance of the optic object to be assessed.
        field: Field at which data is generated (Hx, Hy).
        wavelengths: Wavelengths at which data is generated.
        grid_size: The side length of the square grid of rays (NxN).
        data: Contains Jones matrix data in a list, ordered by wavelength.
    """

    def __init__(
        self,
        optic: Optic,
        field: tuple[float, float] = (0, 0),
        wavelengths: str | list = "all",
        grid_size: int = 65,
    ):
        """Initializes the JonesPupil analysis.

        Args:
            optic: An instance of the optic object to be assessed.
            field: The normalized field coordinates (Hx, Hy) at which to
                generate data. Defaults to (0, 0).
            wavelengths: Wavelengths at which to generate data. If 'all', all
                defined wavelengths are used. Defaults to "all".
            grid_size: The number of points along one dimension of the pupil grid.
                Defaults to 65.
        """
        self.field = field
        self.grid_size = grid_size
        super().__init__(optic, wavelengths)

    def view(
        self,
        fig_to_plot_on: Figure | None = None,
        figsize: tuple[float, float] = (16, 8),
        *,
        show: bool = True,
    ) -> tuple[Figure, list[Axes]]:
        """Displays the Jones pupil plots.

        Args:
            fig_to_plot_on: An existing Matplotlib figure to plot on. If None,
                a new figure is created. Defaults to None.
            figsize: The figure size for the output window. Defaults to (16, 8).
            show (bool): If True (default), calls plt.show(). Set False for
                headless use.

        Returns:
            A tuple containing the Matplotlib figure and a list of its axes.
        """
        # Select primary wavelength index
        wl_idx = 0
        primary_wl = self.optic.primary_wavelength
        wl_values = [wp.value for wp in self.wavelengths]
        if primary_wl in wl_values:
            wl_idx = wl_values.index(primary_wl)

        data_fw = self.data[wl_idx]

        if fig_to_plot_on:
            fig = fig_to_plot_on
            fig.clear()
        else:
            fig = plt.figure(figsize=figsize)

        # 2 rows (Real, Imag), 4 columns (Jxx, Jxy, Jyx, Jyy)
        axs = fig.subplots(2, 4, sharex=True, sharey=True)

        # Elements to plot
        elements = [
            ("Jxx", data_fw["J"][:, 0, 0]),
            ("Jxy", data_fw["J"][:, 0, 1]),
            ("Jyx", data_fw["J"][:, 1, 0]),
            ("Jyy", data_fw["J"][:, 1, 1]),
        ]

        px = be.to_numpy(data_fw["Px"]).reshape(self.grid_size, self.grid_size)
        py = be.to_numpy(data_fw["Py"]).reshape(self.grid_size, self.grid_size)
        mask = px**2 + py**2 <= 1.0

        for col, (name, values) in enumerate(elements):
            val_np = be.to_numpy(values).reshape(self.grid_size, self.grid_size)
            val_np[~mask] = np.nan

            # Real part
            ax_real = axs[0, col]
            im_real = ax_real.pcolormesh(
                px, py, np.real(val_np), shading="nearest", cmap="viridis"
            )
            ax_real.set_title(f"Re({name})")
            ax_real.set_aspect("equal")
            fig.colorbar(im_real, ax=ax_real, fraction=0.046, pad=0.04)

            # Imag part
            ax_imag = axs[1, col]
            im_imag = ax_imag.pcolormesh(
                px, py, np.imag(val_np), shading="nearest", cmap="viridis"
            )
            ax_imag.set_title(f"Im({name})")
            ax_imag.set_aspect("equal")
            fig.colorbar(im_imag, ax=ax_imag, fraction=0.046, pad=0.04)

        # Labels
        for ax in axs[:, 0]:
            ax.set_ylabel("Py")
        for ax in axs[-1, :]:
            ax.set_xlabel("Px")

        field_val = self.field
        wl_val = self.wavelengths[wl_idx].value
        fig.suptitle(f"Jones Pupil - Field: {field_val}, Wavelength: {wl_val:.4f} µm")
        fig.tight_layout()

        is_gui_embedding = fig_to_plot_on is not None
        if show and not is_gui_embedding:
            plt.show()
        return fig, fig.get_axes()

    def _generate_data(self):
        """Generates Jones matrix data for all fields and wavelengths."""
        # Generate pupil grid
        x = be.linspace(-1.0, 1.0, self.grid_size)
        y = be.linspace(-1.0, 1.0, self.grid_size)
        Px_grid, Py_grid = be.meshgrid(x, y)
        Px = Px_grid.flatten()
        Py = Py_grid.flatten()

        data = []
        Hx, Hy = self.field
        for wp in self.wavelengths:
            data.append(self._generate_single_data(Hx, Hy, Px, Py, wp.value))

        return data

    def _generate_single_data(self, Hx, Hy, Px, Py, wavelength):
        """Generates data for a single field and wavelength configuration."""

        # Handle polarization state
        original_pol = self.optic.polarization
        if original_pol == "ignore":
            # Temporarily enable polarization to get PolarizedRays
            self.optic.updater.set_polarization(PolarizationState())

        try:
            rays = self.optic.trace_generic(
                Hx=Hx, Hy=Hy, Px=Px, Py=Py, wavelength=wavelength
            )
        finally:
            if original_pol == "ignore":
                self.optic.updater.set_polarization("ignore")

        if not hasattr(rays, "p"):
            # Fallback if rays are not polarized (should not happen w/ check above)
            raise RuntimeError("Ray tracing did not return polarized rays.")

        # Ray direction vectors (normalized)
        k = be.stack([rays.L, rays.M, rays.N], axis=1)
        # Normalize k (should be already, but to be safe)
        k_norm = be.linalg.norm(k, axis=1)
        k = k / be.unsqueeze_last(k_norm)

        # Construct local basis vectors (Standard Polar Projection / Dipole-like)
        # v ~ Y-axis: perpendicular to k and X=[1,0,0]
        x_axis = be.array([1.0, 0.0, 0.0])
        # Broadcast x_axis to match k shape
        x_axis = be.broadcast_to(x_axis, k.shape)

        v = be.cross(k, x_axis)
        v_norm = be.linalg.norm(v, axis=1)

        # Avoid division by zero
        v = v / be.unsqueeze_last(v_norm + 1e-15)

        # u ~ X-axis: perpendicular to v and k
        u = be.cross(v, k)
        u_norm = be.linalg.norm(u, axis=1)
        # Avoid division by zero
        u = u / be.unsqueeze_last(u_norm + 1e-15)

        # Project global P onto local basis (u, v)
        # Jxx = u . (P . x_in)
        # Jxy = u . (P . y_in)
        # Jyx = v . (P . x_in)
        # Jyy = v . (P . y_in)

        # p has shape (N, 3, 3)
        # P . x_in is simply the first column of p
        # P . y_in is simply the second column of p

        P_x_in = rays.p[:, :, 0]  # Shape (N, 3)
        P_y_in = rays.p[:, :, 1]  # Shape (N, 3)

        # Dot products
        Jxx = be.sum(u * P_x_in, axis=1)
        Jxy = be.sum(u * P_y_in, axis=1)
        Jyx = be.sum(v * P_x_in, axis=1)
        Jyy = be.sum(v * P_y_in, axis=1)

        # Stack into (N, 2, 2)
        row1 = be.stack([Jxx, Jxy], axis=1)
        row2 = be.stack([Jyx, Jyy], axis=1)
        J = be.stack([row1, row2], axis=1)

        return {"Px": Px, "Py": Py, "J": J}
