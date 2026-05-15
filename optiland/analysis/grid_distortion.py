"""Grid Distortion Analysis

This module provides a grid distortion analysis for optical systems.
This is module enables calculation of the distortion over a grid of points
for an optical system.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

import optiland.backend as be

from .base import BaseAnalysis

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class GridDistortion(BaseAnalysis):
    """Grid distortion analysis for an optical system.

    Args:
        optic (Optic): The optical system to analyze.
        wavelength (str | float | int, optional): Wavelength for analysis.
            Can be 'primary', 'all', or a numeric value. Defaults to 'primary'.
        num_points (int, optional): Number of grid points per axis. Defaults to 10.
        distortion_type (str, optional): Distortion model, either 'f-tan' or 'f-theta'.
            Defaults to 'f-tan'.

    Attributes:
        num_points (int): Number of grid points per axis.
        distortion_type (str): Distortion model used.
        data (dict): Computed distortion data (after running _generate_data()).

    Methods:
        view(fig_to_plot_on=None, figsize=(7, 7)):
            Visualizes the grid distortion analysis.
    """

    def __init__(
        self,
        optic,
        wavelength="primary",
        num_points=10,
        distortion_type="f-tan",
    ):
        # --- Start of Corrected Code ---
        if isinstance(wavelength, float | int):
            # Wrap the single wavelength number in a list for the base class.
            processed_wavelengths = [wavelength]
        elif isinstance(wavelength, str) and wavelength in ["primary", "all"]:
            # Pass the allowed strings directly to the base class.
            processed_wavelengths = wavelength
        else:
            raise TypeError(
                f"Unsupported wavelength: {wavelength}. "
                "Expected 'primary', 'all', or a number."
            )

        self.num_points = num_points
        self.distortion_type = distortion_type
        # Pass the formatted list or string to the base class constructor.
        super().__init__(optic, wavelengths=processed_wavelengths)

    def view(
        self,
        fig_to_plot_on: Figure | None = None,
        figsize: tuple[float, float] = (7, 7),
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """Visualizes the grid distortion analysis.

        Args:
            fig_to_plot_on (plt.Figure, optional): Existing figure to plot on.
                If None, a new figure is created. Defaults to None.
            figsize (tuple, optional): Size of the figure if a new one is created.
                Defaults to (7, 7) for a square plot.
            show (bool): If True (default), calls plt.show(). Set False for
                headless use.
        Returns:
            tuple: The figure and axes objects used for plotting.
        """
        is_gui_embedding = fig_to_plot_on is not None

        if is_gui_embedding:
            current_fig = fig_to_plot_on
            current_fig.clear()
            ax = current_fig.add_subplot(111)
        else:
            current_fig, ax = plt.subplots(figsize=figsize)

        ax.plot(
            be.to_numpy(self.data["xp"]),
            be.to_numpy(self.data["yp"]),
            "C1",
            linewidth=1,
            label="Ideal Grid",
        )
        ax.plot(
            be.to_numpy(self.data["xp"]).T,
            be.to_numpy(self.data["yp"]).T,
            "C1",
            linewidth=1,
        )
        ax.plot(
            be.to_numpy(self.data["xr"]),
            be.to_numpy(self.data["yr"]),
            "C0--",
            label="Distorted Grid",
        )
        ax.plot(be.to_numpy(self.data["xr"]).T, be.to_numpy(self.data["yr"]).T, "C0--")

        ax.set_xlabel("Image X (mm)")
        ax.set_ylabel("Image Y (mm)")
        ax.set_aspect("equal", adjustable="box")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
        ax.grid(True, linestyle=":", alpha=0.6)

        max_distortion = self.data["max_distortion"]
        ax.set_title(f"Grid Distortion (Max: {max_distortion:.2f}%)")
        current_fig.tight_layout()

        if is_gui_embedding and hasattr(current_fig, "canvas"):
            current_fig.canvas.draw_idle()
        if show and not is_gui_embedding:
            plt.show()
        return current_fig, ax

    def _generate_data(self):
        """Generates the data for the grid distortion analysis.

        Returns:
            dict: The generated data.

        Raises:
            ValueError: If the distortion type is not 'f-tan' or 'f-theta'.

        """
        # Trace chief ray to retrieve central (x, y) position
        current_wavelength = self.wavelengths[0].value
        self.optic.trace_generic(
            Hx=0,
            Hy=0,
            Px=0,
            Py=0,
            wavelength=current_wavelength,
        )
        x_chief = self.optic.surfaces.x[-1, 0]
        y_chief = self.optic.surfaces.y[-1, 0]

        # Trace single reference ray
        current_wavelength = self.wavelengths[0].value
        self.optic.trace_generic(
            Hx=0,
            Hy=1e-10,  # small field
            Px=0,
            Py=0,
            wavelength=current_wavelength,
        )
        y_ref = self.optic.surfaces.y[-1, 0]

        max_field = np.sqrt(2) / 2
        extent = be.linspace(-max_field, max_field, self.num_points)
        Hx, Hy = be.meshgrid(extent, extent)

        if self.distortion_type == "f-tan":
            const = (y_ref - y_chief) / (
                be.tan(1e-10 * be.radians(self.optic.fields.max_field))
            )
            xp = const * be.tan(Hx * be.radians(self.optic.fields.max_field))
            yp = const * be.tan(Hy * be.radians(self.optic.fields.max_field))
        elif self.distortion_type == "f-theta":
            const = (y_ref - y_chief) / (
                1e-10 * be.radians(self.optic.fields.max_field)
            )
            xp = const * Hx * be.radians(self.optic.fields.max_field)
            yp = const * Hy * be.radians(self.optic.fields.max_field)
        else:
            raise ValueError('''Distortion type must be "f-tan" or"f-theta"''')

        self.optic.trace_generic(
            Hx=Hx.flatten(),
            Hy=Hy.flatten(),
            Px=0,
            Py=0,
            wavelength=current_wavelength,
        )

        data = {}

        # make real grid square for ease of plotting
        data["xr"] = (
            be.reshape(
                self.optic.surfaces.x[-1, :],
                (self.num_points, self.num_points),
            )
            - x_chief
        )
        data["yr"] = (
            be.reshape(
                self.optic.surfaces.y[-1, :],
                (self.num_points, self.num_points),
            )
            - y_chief
        )

        data["xp"] = xp
        data["yp"] = yp

        # Find max distortion
        delta = be.sqrt((data["xp"] - data["xr"]) ** 2 + (data["yp"] - data["yr"]) ** 2)
        rp = be.sqrt(data["xp"] ** 2 + data["yp"] ** 2)

        data["max_distortion"] = be.max(100 * delta / rp)

        return data
