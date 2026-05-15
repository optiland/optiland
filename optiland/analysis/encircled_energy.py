"""Encircled Energy Analysis

This module provides an encircled energy analysis for optical systems.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

import optiland.backend as be

from .spot_diagram import SpotData, SpotDiagram

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class EncircledEnergy(SpotDiagram):
    """Class representing the Encircled Energy analysis of a given optic.

    Args:
        optic (Optic): The optic for which the Encircled Energy analysis is
            performed.
        fields (str or tuple, optional): The fields for which the analysis is
            performed. Defaults to 'all'.
        wavelength (str or float, optional): The wavelength at which the
            analysis is performed. Defaults to 'primary'.
        num_rays (int, optional): The number of rays used for the analysis.
            Defaults to 100000.
        distribution (str, optional): The distribution of rays.
            Defaults to 'random'.
        num_points (int, optional): The number of points used for plotting the
            Encircled Energy curve. Defaults to 256.

    """

    def __init__(
        self,
        optic,
        fields="all",
        wavelength="primary",
        num_rays=100_000,
        distribution="random",
        num_points=256,
    ):
        self.num_points = num_points

        if isinstance(wavelength, float | int):
            # If a number is passed, wrap it in a list for the base classes.
            processed_wavelengths = [wavelength]
        elif isinstance(wavelength, str) and wavelength in ["primary", "all"]:
            # If 'primary' or 'all' is passed, let the base class handle it.
            processed_wavelengths = wavelength
        else:
            # Catch any other invalid input.
            raise TypeError(
                f"Unsupported wavelength: {wavelength}. "
                "Expected 'primary', 'all', or a number."
            )

        super().__init__(
            optic,
            fields=fields,
            wavelengths=processed_wavelengths,  # Pass the formatted value
            num_rings=num_rays,
            distribution=distribution,
        )

    def view(
        self,
        fig_to_plot_on: Figure | None = None,
        figsize: tuple[float, float] = (7, 4.5),
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """Plot the Encircled Energy curve.

        Args:
            fig_to_plot_on (plt.Figure, optional): The figure to plot on.
                If None, a new figure is created. Defaults to None.
            figsize (tuple, optional): The size of the figure if a new one is
                created. Defaults to (7, 4.5).
            show (bool): If True (default), calls plt.show(). Set False for
                headless use.

        Returns:
            tuple: A tuple containing the figure and axes objects.
        """
        is_gui_embedding = fig_to_plot_on is not None

        if is_gui_embedding:
            current_fig = fig_to_plot_on
            current_fig.clear()
            ax = current_fig.add_subplot(111)
        else:
            current_fig, ax = plt.subplots(figsize=figsize)

        data = self._center_spots(self.data)
        geometric_size = self.geometric_spot_radius()
        axis_lim = be.max(geometric_size)

        for k, field_data in enumerate(data):
            self._plot_field(ax, field_data, self.fields[k], axis_lim, self.num_points)

        ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
        ax.set_xlabel("Radius (mm)")
        ax.set_ylabel("Encircled Energy (-)")
        ax.set_title(f"Wavelength: {self.wavelengths[0].value:.4f} µm")
        ax.set_xlim((0, None))
        ax.set_ylim((0, None))
        ax.grid(True)
        current_fig.tight_layout()

        if is_gui_embedding and hasattr(current_fig, "canvas"):
            current_fig.canvas.draw_idle()
        if show and not is_gui_embedding:
            plt.show()
        return current_fig, ax

    def centroid(self):
        """Calculate the centroid of the Encircled Energy.

        Returns:
            list: A list of tuples representing the centroid coordinates for
                each field.

        """
        centroid = []
        for field_data in self.data:
            spot_data_item = field_data[0]
            centroid_x = be.mean(spot_data_item.x)
            centroid_y = be.mean(spot_data_item.y)
            centroid.append((centroid_x, centroid_y))
        return centroid

    def _plot_field(self, ax, field_data, field, axis_lim, num_points, buffer=1.2):
        """Plot the Encircled Energy curve for a specific field.

        Args:
            ax (matplotlib.axes.Axes): The axes on which to plot the curve.
            field_data (list): List of field data.
            field (tuple): Tuple representing the normalized field coordinates.
            axis_lim (float): Maximum axis limit.
            num_points (int): Number of points for plotting the curve.
            buffer (float, optional): Buffer factor for the axis limit.
                Defaults to 1.2.

        """
        r_max = axis_lim * buffer
        r_step = be.linspace(0, r_max, num_points)

        for points in field_data:
            x = points.x
            y = points.y
            # energy and intensity are used interchangeably here
            energy = points.intensity
            radii = be.sqrt(x**2 + y**2)

            def vectorized_ee(r):
                return be.nansum(energy[radii <= r])  # noqa: B023

            # element‑wise encircled energy (Tensor)
            ee = be.vectorize(vectorized_ee)(r_step)

            # convert both to plain numpy for plotting
            r_np = be.to_numpy(r_step)
            ee_np = be.to_numpy(ee)
            Hx, Hy = field.coord
            ax.plot(r_np, ee_np, label=f"Hx: {Hx:.3f}, Hy: {Hy:.3f}")

    def _generate_field_data(
        self,
        field,
        wavelength,
        num_rays=100,
        distribution="hexapolar",
        coordinates="local",
    ):
        """Generate the field data for a specific field and wavelength.

        Args:
            field (tuple): Tuple representing the field coordinates.
            wavelength (float): The wavelength.
            num_rays (int, optional): The number of rays. Defaults to 100.
            distribution (str, optional): The distribution of rays.
                Defaults to 'hexapolar'.
            coordinates (str): Coordinate system choice (ignored).

        Returns:
            SpotData: SpotData object containing x, y, and intensity arrays.

        """
        self.optic.trace(*field, wavelength, num_rays, distribution)
        x = self.optic.surfaces.x[-1, :]
        y = self.optic.surfaces.y[-1, :]
        intensity = self.optic.surfaces.intensity[-1, :]
        return SpotData(x=x, y=y, intensity=intensity)
