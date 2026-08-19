"""Spot Diagram Plotting Utilities

This module provides standalone plotting helper functions for spot diagrams.
These are extracted from the SpotDiagram class to keep the core analysis
class focused on data generation and metrics.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

import optiland.backend as be

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from .core import SpotData


def handle_no_fields(fig: Figure | None) -> tuple[None, None]:
    """Handles the case where there are no fields to plot.

    Args:
        fig: An optional existing figure to draw a message on.

    Returns:
        A tuple of (None, None).
    """
    print("Warning (SpotDiagram.view): No fields to plot.")
    if fig and hasattr(fig, "canvas") and fig.canvas:
        fig.text(0.5, 0.5, "No fields to plot Spot Diagram", ha="center", va="center")
        fig.canvas.draw_idle()
    return None, None


def setup_plot_layout(
    num_fields: int,
    fig_to_plot_on: Figure | None,
    figsize: tuple[float, float],
) -> tuple[Figure, NDArray[np.object_]]:
    """Sets up the Matplotlib figure and axes grid.

    Args:
        num_fields: Number of field points to plot.
        fig_to_plot_on: An existing figure to use, or None to create one.
        figsize: The size for the figure.

    Returns:
        A tuple of the figure and a flattened array of its axes.
    """
    num_cols = 3
    num_rows = (num_fields + num_cols - 1) // num_cols

    if fig_to_plot_on:
        fig = fig_to_plot_on
        fig.clear()
    else:
        fig = plt.figure(figsize=(figsize[0], num_rows * figsize[1]))

    axs = fig.subplots(num_rows, num_cols, sharex=True, sharey=True).flatten()
    return fig, axs


def calculate_axis_limits(
    centered_data: list[list[SpotData]],
    fields: list[tuple[float, float]],
    airy_disk_data: dict | None = None,
    buffer: float = 1.05,
) -> float:
    """Calculates the axis limits to encompass all spots and Airy disks.

    Args:
        centered_data: The centered spot data.
        fields: List of field coordinates.
        airy_disk_data: The prepared Airy disk data, if any.
        buffer: A multiplicative buffer to apply to the final limit.

    Returns:
        A float representing the symmetric axis limit (+/- limit).
    """
    max_radii = [
        be.to_numpy(be.max(be.sqrt(sd.x**2 + sd.y**2)))
        for field in centered_data
        for sd in field
    ]
    max_geom_radius = max(max_radii) if max_radii else 0.01

    if not airy_disk_data:
        return max_geom_radius * buffer

    rad_x = airy_disk_data["radii_x"]
    rad_y = airy_disk_data["radii_y"]
    centroids = airy_disk_data["airy_centers"]

    max_extent = 0
    for i in range(len(fields)):
        offset = np.max(np.abs(centroids[i]))
        radius = max(rad_x[i], rad_y[i], max_geom_radius)
        max_extent = max(max_extent, offset + radius)

    return max_extent * buffer if max_extent > 0 else max_geom_radius * buffer


def plot_field(
    ax: Axes,
    field_data: list[SpotData],
    wavelengths: list[float],
    field_coords: tuple[float, float],
    axis_lim: float,
    field_index: int,
    image_surface,
    airy_disk_data: dict | None = None,
) -> None:
    """Plots the data for a single field on a given axis.

    Args:
        ax: The Matplotlib axis to plot on.
        field_data: A list of SpotData for the current field.
        wavelengths: List of wavelength values.
        field_coords: The (Hx, Hy) coordinates of the field.
        axis_lim: The symmetric axis limit for x and y axes.
        field_index: The index of the current field.
        image_surface: The image surface of the optic.
        airy_disk_data: Optional dictionary with Airy disk data.
    """
    markers = ["o", "s", "^"]
    for i, points in enumerate(field_data):
        x, y, intensity = (
            be.to_numpy(points.x),
            be.to_numpy(points.y),
            be.to_numpy(points.intensity),
        )
        mask = intensity != 0
        # wavelengths[i] may be a WavelengthPoint or a float; extract value safely
        wl_entry = wavelengths[i]
        wl_val = wl_entry.value if hasattr(wl_entry, "value") else wl_entry
        ax.scatter(
            x[mask],
            y[mask],
            s=10,
            label=f"{wl_val:.4f} µm",
            marker=markers[i % 3],
            alpha=0.7,
        )

    if airy_disk_data:
        cx, cy = airy_disk_data["airy_centers"][field_index]
        width = 2 * airy_disk_data["radii_y"][field_index]
        height = 2 * airy_disk_data["radii_x"][field_index]
        ellipse = patches.Ellipse(
            (cx, cy),
            width,
            height,
            linestyle="--",
            edgecolor="black",
            fill=False,
            lw=2,
        )
        ax.add_patch(ellipse)

    # Determine axis labels based on image surface orientation
    cs = image_surface.geometry.cs
    if np.any(np.abs(cs.get_effective_rotation_euler())[:2] > 0.01):
        x_label, y_label = "U (mm)", "V (mm)"
    else:
        x_label, y_label = "X (mm)", "Y (mm)"

    ax.axis("square")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_title(f"Hx: {field_coords[0]:.3f}, Hy: {field_coords[1]:.3f}")
    ax.grid(True, alpha=0.25)


def finalize_plot(
    fig: Figure,
    axs: NDArray[np.object_],
    num_fields: int,
    wavelengths: list[float],
) -> None:
    """Applies final touches to the plot, including a dynamic shared legend.

    Args:
        fig: The Matplotlib figure.
        axs: The array of axes.
        num_fields: The number of fields that were plotted.
        wavelengths: The list of wavelengths used in the analysis.
    """
    for i in range(num_fields, len(axs)):
        fig.delaxes(axs[i])

    if num_fields == 0:
        if hasattr(fig, "canvas") and fig.canvas:
            fig.canvas.draw_idle()
        return

    handles, labels = axs[0].get_legend_handles_labels()
    if not handles:
        fig.tight_layout()
        if hasattr(fig, "canvas") and fig.canvas:
            fig.canvas.draw_idle()
        return

    fig.canvas.draw()
    pos_left = axs[0].get_position()
    num_cols = 3

    rightmost_ax_idx = min(num_fields - 1, num_cols - 1)
    pos_right = axs[rightmost_ax_idx].get_position()

    x_center = (pos_left.x0 + pos_right.x1) / 2.0

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(x_center, 0.05),
        ncol=len(wavelengths),
    )

    fig.subplots_adjust(bottom=0.2)
