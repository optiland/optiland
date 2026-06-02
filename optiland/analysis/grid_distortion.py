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
from matplotlib.lines import Line2D

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
        if isinstance(wavelength, float | int):
            processed_wavelengths = [wavelength]
        elif isinstance(wavelength, str) and wavelength in ["primary", "all"]:
            processed_wavelengths = wavelength
        else:
            raise TypeError(
                f"Unsupported wavelength: {wavelength}. "
                "Expected 'primary', 'all', or a number."
            )

        self.num_points = num_points
        self.distortion_type = distortion_type
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
            fig = fig_to_plot_on
            fig.clear()
            ax = fig.add_subplot(111)
        else:
            fig, ax = plt.subplots(figsize=figsize)

        self._plot_grid(ax)
        self._style_axes(ax)

        fig.tight_layout()

        if is_gui_embedding and hasattr(fig, "canvas"):
            fig.canvas.draw_idle()
        if show and not is_gui_embedding:
            plt.show()

        return fig, ax

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _plot_grid(self, ax: Axes) -> None:
        """Draws ideal and distorted grid lines, each with a single legend entry."""
        xp = be.to_numpy(self.data["xp"])
        yp = be.to_numpy(self.data["yp"])
        xr = be.to_numpy(self.data["xr"])
        yr = be.to_numpy(self.data["yr"])

        # Plot rows and columns — suppress per-line labels so the legend
        # only shows one entry per grid type.
        ax.plot(xp, yp, "C1", linewidth=1)
        ax.plot(xp.T, yp.T, "C1", linewidth=1)
        ax.plot(xr, yr, "C0--", linewidth=1)
        ax.plot(xr.T, yr.T, "C0--", linewidth=1)

        legend_handles = [
            Line2D([0], [0], color="C1", linewidth=1, label="Ideal Grid"),
            Line2D(
                [0],
                [0],
                color="C0",
                linestyle="--",
                linewidth=1,
                label="Distorted Grid",
            ),
        ]
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 0.5), loc="center left")

    def _style_axes(self, ax: Axes) -> None:
        """Applies labels, title, and cosmetic styling to the axes."""
        max_distortion = self.data["max_distortion"]
        ax.set_title(f"Grid Distortion (Max: {max_distortion:.2f}%)")
        ax.set_xlabel("Image X (mm)")
        ax.set_ylabel("Image Y (mm)")
        ax.set_aspect("equal", adjustable="box")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.grid(True, linestyle=":", alpha=0.6)

    def _generate_data(self) -> dict:
        """Generates the data for the grid distortion analysis.

        Returns:
            dict: The generated data.

        Raises:
            ValueError: If the distortion type is not 'f-tan' or 'f-theta'.
        """
        current_wavelength = self.wavelengths[0].value
        x_chief, y_chief = self._trace_chief_ray(current_wavelength)
        y_ref = self._trace_reference_ray(current_wavelength, y_chief)

        Hx, Hy = self._build_field_grid()
        xp, yp = self._compute_ideal_positions(Hx, Hy, y_chief, y_ref)

        self.optic.trace_generic(
            Hx=Hx.flatten(),
            Hy=Hy.flatten(),
            Px=0,
            Py=0,
            wavelength=current_wavelength,
        )

        xr = (
            be.reshape(self.optic.surfaces.x[-1, :], (self.num_points, self.num_points))
            - x_chief
        )
        yr = (
            be.reshape(self.optic.surfaces.y[-1, :], (self.num_points, self.num_points))
            - y_chief
        )

        delta = be.sqrt((xp - xr) ** 2 + (yp - yr) ** 2)
        rp = be.sqrt(xp**2 + yp**2)
        max_distortion = be.max(100 * delta / rp)

        return {
            "xp": xp,
            "yp": yp,
            "xr": xr,
            "yr": yr,
            "max_distortion": max_distortion,
        }

    def _trace_chief_ray(self, wavelength: float) -> tuple:
        """Returns the image-plane (x, y) position of the on-axis chief ray."""
        self.optic.trace_generic(Hx=0, Hy=0, Px=0, Py=0, wavelength=wavelength)
        return self.optic.surfaces.x[-1, 0], self.optic.surfaces.y[-1, 0]

    def _trace_reference_ray(self, wavelength: float, y_chief) -> float:
        """Returns the image-plane y position of a near-axis reference ray."""
        self.optic.trace_generic(Hx=0, Hy=1e-10, Px=0, Py=0, wavelength=wavelength)
        return self.optic.surfaces.y[-1, 0]

    def _build_field_grid(self):
        """Returns (Hx, Hy) meshgrid spanning the normalised field square."""
        max_field = np.sqrt(2) / 2
        extent = be.linspace(-max_field, max_field, self.num_points)
        return be.meshgrid(extent, extent)

    def _compute_ideal_positions(self, Hx, Hy, y_chief, y_ref):
        """Returns (xp, yp) ideal grid positions for the chosen distortion model."""
        max_field_rad = be.radians(self.optic.fields.max_field)

        if self.distortion_type == "f-tan":
            const = (y_ref - y_chief) / be.tan(1e-10 * max_field_rad)
            xp = const * be.tan(Hx * max_field_rad)
            yp = const * be.tan(Hy * max_field_rad)
        elif self.distortion_type == "f-theta":
            const = (y_ref - y_chief) / (1e-10 * max_field_rad)
            xp = const * Hx * max_field_rad
            yp = const * Hy * max_field_rad
        else:
            raise ValueError('distortion_type must be "f-tan" or "f-theta"')

        return xp, yp
