"""
SurgaceSagViewer

This module provides a viewer for visualizing the sag of an optical surface.
It generates a 2D sag map and 1D sag profiles along user-specified cross-sections.

Manuel Fragata Mendes, june 2025
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

import optiland.backend as be
from optiland.visualization.base import BaseViewer


class CustomScalarFormatter(ScalarFormatter):
    """Custom formatter to handle both small and large numbers automatically."""

    def __call__(self, x, pos=None):
        # For values close to zero, use scientific notation
        if abs(x) < 0.01 and x != 0.0:
            self.set_scientific(True)
            self.set_powerlimits((-3, 3))
        else:
            self.set_scientific(False)
        return ScalarFormatter.__call__(self, x, pos)


class SurfaceSagViewer(BaseViewer):
    """A viewer for visualizing the sag of an optical surface."""

    def view(
        self,
        surface_index: int,
        y_cross_section: float = 0.0,
        x_cross_section: float = 0.0,
        fig_to_plot_on: plt.Figure = None,
        max_extent: float = None,  # Renamed from max_extent_override for clarity
        num_points_grid: int = 50,
        buffer_factor: float = 1.1,  # Add 10% buffer around the aperture
    ):
        """
        Analyzes and visualizes the sag of a given lens surface.

        This method generates a 2D sag map and 1D sag profiles along
        user-specified x and y cross-sections.

        Args:
            surface_index (int): The index of the surface to analyze.
            y_cross_section (float, optional): The y-coordinate for the
                horizontal (X-axis) profile. Defaults to 0.0.
            x_cross_section (float, optional): The x-coordinate for the
                vertical (Y-axis) profile. Defaults to 0.0.
            fig_to_plot_on (plt.Figure, optional): Figure to plot on. If None,
                creates a new figure.
            max_extent (float, optional): Maximum extent of the plot in mm.
                If None, uses the surface's aperture with a buffer.
                This controls the viewing area size.
            num_points_grid (int, optional): Number of points in each dimension
                of the grid. Defaults to 50.
            buffer_factor (float, optional): Factor to multiply the aperture by
                to add a buffer around the plot. Defaults to 1.1.
        """
        is_gui_embedding = fig_to_plot_on is not None

        if is_gui_embedding:
            fig = fig_to_plot_on
            fig.clear()
        else:
            fig = plt.figure(figsize=(9, 9))

        # Get the surface to analyze
        surface = self.optic.surfaces[surface_index]

        # Determine the appropriate grid extent based on the surface aperture
        if max_extent is not None:
            max_extent_grid = max_extent
        else:
            # Use a default minimum value to ensure reasonable visualization
            # even for very small or undefined apertures
            min_grid_extent = 5.0

            # Check if semi_aperture is None or not defined
            if surface.semi_aperture is None:
                max_extent_grid = min_grid_extent
            else:
                # Convert the semi-aperture to a numpy scalar
                surface_semi_aperture = be.to_numpy(surface.semi_aperture)

                if surface_semi_aperture > 0:
                    max_extent_grid = surface_semi_aperture * buffer_factor
                else:
                    max_extent_grid = min_grid_extent

            # Ensure the grid extent is at least the minimum
            max_extent_grid = max(max_extent_grid, min_grid_extent)

        # For a 2D sag map, create grids for x and y
        x_grid_coords = be.linspace(-max_extent_grid, max_extent_grid, num_points_grid)
        y_grid_coords = be.linspace(-max_extent_grid, max_extent_grid, num_points_grid)
        X_grid_map, Y_grid_map = be.meshgrid(x_grid_coords, y_grid_coords)

        # Calculate the 2D sag map
        sag_map_2d = surface.geometry.sag(X_grid_map, Y_grid_map)

        # Create arrays for cross-sections
        y_cross_section_array = be.full_like(x_grid_coords, y_cross_section)
        x_cross_section_array = be.full_like(y_grid_coords, x_cross_section)

        # Calculate sag profiles at the specified cross-sections
        sag_profile_x = surface.geometry.sag(x_grid_coords, y_cross_section_array)
        sag_profile_y = surface.geometry.sag(x_cross_section_array, y_grid_coords)

        # --- Plotting using make_axes_locatable for robust alignment ---
        ax_map = fig.add_subplot(111)

        # Create a divider for the main axes
        divider = make_axes_locatable(ax_map)

        # Append axes for the profiles and colorbar
        ax_profile_x = divider.append_axes("bottom", size="25%", pad=0.2, sharex=ax_map)
        ax_profile_y = divider.append_axes("right", size="25%", pad=0.2, sharey=ax_map)
        cax = divider.append_axes("top", size="5%", pad=0.1)

        # Hide labels on the shared axes
        ax_map.tick_params(axis="x", labelbottom=False)
        ax_profile_y.tick_params(axis="y", labelleft=False)

        # -- Main Plot: 2D Sag Map --
        ax_map.set_aspect("equal")
        contour = ax_map.contourf(
            be.to_numpy(X_grid_map),
            be.to_numpy(Y_grid_map),
            be.to_numpy(sag_map_2d),
            levels=50,
        )

        x_formatter = CustomScalarFormatter()
        y_formatter = CustomScalarFormatter()
        ax_map.xaxis.set_major_formatter(x_formatter)
        ax_map.yaxis.set_major_formatter(y_formatter)
        ax_profile_x.xaxis.set_major_formatter(x_formatter)
        ax_profile_y.yaxis.set_major_formatter(y_formatter)
        ax_map.set_ylabel("Y-coordinate")

        # Add extent and aperture information to the title
        if surface.semi_aperture is None:
            aperture_info = "(No aperture defined)"
        else:
            aperture_info = (
                f"(Aperture: {be.to_numpy(surface.semi_aperture).item():.2f} mm)"
            )
        extent_info = f"View: ±{max_extent_grid:.2f} mm"

        ax_map.set_title(
            f"Surface S{surface_index} {aperture_info} | {extent_info}", pad=80
        )

        # Add profile indicator lines at specified cross-sections
        ax_map.axhline(
            y_cross_section,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"X-Profile (y={y_cross_section})",
        )
        ax_map.axvline(
            x_cross_section,
            color="blue",
            linestyle="--",
            linewidth=1.5,
            label=f"Y-Profile (x={x_cross_section})",
        )
        ax_map.legend(loc="upper right")

        # -- Top: Colorbar --
        cbar = fig.colorbar(contour, cax=cax, orientation="horizontal")
        cbar.set_label("Sag (z)", labelpad=15, y=0.5)

        # Get the sag value range for adaptive formatting
        sag_min = be.to_numpy(be.min(sag_map_2d))
        sag_max = be.to_numpy(be.max(sag_map_2d))
        sag_range = abs(sag_max - sag_min)

        # Set up the formatter based on the range of values
        formatter = CustomScalarFormatter()
        cax.xaxis.set_major_formatter(formatter)
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")

        # Add minor ticks for better readability
        cax.xaxis.set_minor_locator(AutoMinorLocator())

        # Apply scientific notation to profile plots if needed
        if sag_range < 0.01:
            sag_formatter = CustomScalarFormatter()
            ax_profile_x.yaxis.set_major_formatter(sag_formatter)
            ax_profile_y.xaxis.set_major_formatter(sag_formatter)

        # -- Bottom Plot: 1D X-Axis Sag Profile --
        ax_profile_x.plot(
            be.to_numpy(x_grid_coords),
            be.to_numpy(sag_profile_x),
            color="red",
            linestyle="--",
            linewidth=2,
        )
        ax_profile_x.set_xlabel("X-coordinate")
        ax_profile_x.set_ylabel("Sag (z)")
        ax_profile_x.grid(True)
        ax_profile_x.autoscale(enable=True, axis="y", tight=True)

        # -- Right Plot: 1D Y-Axis Sag Profile --
        ax_profile_y.plot(
            be.to_numpy(sag_profile_y),
            be.to_numpy(y_grid_coords),
            color="blue",
            linestyle="--",
            linewidth=2,
        )
        ax_profile_y.set_xlabel("Sag (z)", labelpad=10)
        ax_profile_y.grid(True)
        ax_profile_y.autoscale(enable=True, axis="x", tight=True)

        fig.tight_layout(pad=1.0)
        return fig, fig.get_axes()
