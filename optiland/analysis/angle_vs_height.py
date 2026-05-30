"""Incident Angle vs. Height Plot Analysis

This module provides classes for analyzing the incident angle versus image height
for optical systems, across both pupil and field.

Original concept by BuergiR, 2025
Implemented in Optiland by Kramer Harrison, 2025
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

import optiland.backend as be
from optiland.utils import resolve_wavelength

from .base import BaseAnalysis

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure


def _plot_angle_vs_height(
    plot_data_list: list,
    axis: int,
    optic_name: str,
    plot_style: str,
    ax: Axes,
    title: str,
    color_label: str,
    cmap: str | Colormap,
) -> None:
    """Helper function to generate a consistent angle vs. image
    height plot on a given axis.

    Args:
        plot_data_list (list): A list of tuples, where each tuple contains:
            (height, angle_deg, scan_range, legend_label).
        axis (int): Specifies the axis for measurement (0 for x, 1 for y).
        optic_name (str): The name of the optic, used for the plot title.
        plot_style (str): Matplotlib plot style for the line.
        ax (matplotlib.axes.Axes): The axes object to plot on.
        title (str or None): An optional subtitle for the plot.
        color_label (str): The label for the colorbar.
        cmap (str): The name of the colormap to use.
    """
    norm = plt.Normalize(-1, 1)
    linewidth = 3

    base_title = ""
    if title and optic_name:
        base_title = f"{title} - {optic_name}"
    elif title:
        base_title = str(title)
    elif optic_name:
        base_title = str(optic_name)

    # Compose the full title
    full_title = base_title
    if full_title:
        full_title += "\n"
    full_title += ", ".join([p[3] for p in plot_data_list])

    for height, angle_deg, scan_range, _ in plot_data_list:
        # Create segments for the LineCollection
        points = np.array([height, angle_deg]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a LineCollection, color it with the scan_range, and add to plot
        lc = LineCollection(segments, cmap=cmap, norm=norm, linestyle=plot_style)
        lc.set_array(scan_range)
        lc.set_linewidth(linewidth)
        line = ax.add_collection(lc)

    fig = ax.get_figure()
    fig.suptitle("Incident Angle vs Image Height" + (" (x-axis)" if axis == 0 else ""))
    ax.set_title(full_title, fontsize=10)
    ax.set_xlabel("Image Height in Millimeters")
    ax.set_ylabel("Incident Angle in Degrees")
    cbar = fig.colorbar(line, ax=ax, label=color_label)
    cbar.set_label(color_label, labelpad=15)
    ax.grid(alpha=0.25)
    ax.autoscale_view()


class BaseAngleVsHeightAnalysis(BaseAnalysis, abc.ABC):
    """Abstract base class for Angle vs. Height analysis routines.

    This class provides the common framework for generating angle vs. height
    data using the optic's trace_generic method, and abstract methods for
    defining how the tracing coordinates vary.

    Args:
        optic (Optic): The optic object to analyze.
        surface_idx (int, optional): Index of the surface at which the angle and
            height are measured. Defaults to -1 (last surface).
        axis (int, optional): Specifies the axis for measurement. 0 for x-axis,
            1 for y-axis. Defaults to 1 (y-axis).
        wavelength (str or float, optional): A single wavelength in microns.
            Defaults to 'primary'.
        num_points (int, optional): The number of points used for the plot.
            Defaults to 128.

    Attributes:
        optic (Optic): The optic object being analyzed.
        surface_idx (int): Index of the surface for measurements.
        axis (int): Axis for measurement (0 for x, 1 for y).
        wavelengths (list): The wavelengths being analyzed (handled by BaseAnalysis).
        num_points (int): The number of points generated for the analysis.
        data (dict): The generated data for the analysis. Structure depends on
            subclass.
    """

    def __init__(
        self,
        optic,
        surface_idx: int = -1,
        axis: int = 1,
        wavelength: str | float = "primary",
        num_points: int = 128,
    ):
        self.surface_idx = surface_idx
        self.axis = axis
        self.num_points = num_points

        # The resolved wavelength is passed as a list to the parent constructor
        super().__init__(optic, wavelengths=[resolve_wavelength(optic, wavelength)])

    @abc.abstractmethod
    def _get_trace_coordinates(self, scan_range):
        """Abstract method to define the Hx, Hy, Px, Py coordinates for tracing.

        This method must be implemented by subclasses to specify how the rays
        are generated for the trace_generic method.

        Args:
            scan_range (numpy.ndarray): The linearly spaced array defining
                the range of varying coordinates.

        Returns:
            tuple: A tuple containing (Hx, Hy, Px, Py, coord_label).
                Hx, Hy, Px, Py should be backend arrays ready for trace_generic.
                coord_label is either "Pupil" or "Field" and represents which
                coordinates are fixed during the scan.
        """
        pass  # pragma: no cover

    def _generate_data(self):
        """Generates the incident angle vs. image height data using trace_generic.

        This method is common for all subclasses and orchestrates the ray tracing
        based on the coordinates provided by _get_trace_coordinates.

        Returns:
            dict: A dictionary containing the generated data.
                Keys are (fixed_param_1, fixed_param_2, wavelength_value) tuples,
                values are dictionaries with 'height' and 'angle' numpy arrays.
        """
        data = {}
        scan_range = be.linspace(start=-1, stop=1, num=self.num_points)

        Hx, Hy, Px, Py, coord_label = self._get_trace_coordinates(scan_range)

        Hx = be.atleast_1d(Hx)
        Hy = be.atleast_1d(Hy)
        Px = be.atleast_1d(Px)
        Py = be.atleast_1d(Py)

        # Use the first and only wavelength
        wavelength_value = self.wavelengths[0].value

        self.optic.trace_generic(
            Hx=Hx, Hy=Hy, Px=Px, Py=Py, wavelength=wavelength_value
        )

        if self.axis == 1:  # Y-direction measurement
            incident_dir_cosines = self.optic.surfaces.M[self.surface_idx, :]
            height = self.optic.surfaces.y[self.surface_idx, :]
        else:  # X-direction measurement
            incident_dir_cosines = self.optic.surfaces.L[self.surface_idx, :]
            height = self.optic.surfaces.x[self.surface_idx, :]

        angle_rad = be.arcsin(incident_dir_cosines)

        if coord_label == "Pupil":  # means pupil is fixed and field is scanned
            fixed_param_key = (
                Px[0].item() if be.size(Px) > 0 else 0,
                Py[0].item() if be.size(Py) > 0 else 0,
                float(wavelength_value),
            )
        elif coord_label == "Field":  # means field is fixed and pupil is scanned
            fixed_param_key = (
                Hx[0].item() if be.size(Hx) > 0 else 0,
                Hy[0].item() if be.size(Hy) > 0 else 0,
                float(wavelength_value),
            )
        else:
            raise ValueError("Coord. label must be 'Pupil' or 'Field'.")

        data[fixed_param_key] = {
            "height": be.to_numpy(height),
            "angle": be.to_numpy(angle_rad),
            "fixed_coordinates": coord_label,
            "scan_range": be.to_numpy(scan_range),
        }
        return data

    def view(
        self,
        fig_to_plot_on: Figure = None,
        figsize: tuple[float, float] = (8, 5.5),
        title: str = None,
        cmap: str | Colormap = "viridis",
        line_style: str = "-",
        *,
        show: bool = True,
    ) -> tuple[plt.Figure, Axes]:
        """Displays a plot of the incident angle vs. image height analysis.

        Args:
            fig_to_plot_on (matplotlib.figure.Figure, optional): A figure object
                to plot on. If None, a new figure is created. Defaults to None.
            figsize (tuple, optional): The size of the figure.
                Defaults to (8, 5.5).
            title (str, optional): An optional subtitle to be added to the plot.
                If None, lens name is used.
            cmap (str, optional): The colormap for the plot line.
                Defaults to 'viridis'.
            line_style (str, optional): Matplotlib plot style. Defaults to '-'.
            show (bool): If True (default), calls plt.show(). Set False for
                headless use.

        Returns:
            tuple: A tuple containing the figure and axes objects used for plotting.
        """
        is_gui_embedding = fig_to_plot_on is not None

        if is_gui_embedding:
            current_fig = fig_to_plot_on
            current_fig.clear()
            ax = current_fig.add_subplot(111)
        else:
            current_fig, ax = plt.subplots(figsize=figsize)

        if not self.data:
            ax.text(
                0.5,
                0.5,
                "Error: Data could not be generated.",
                ha="center",
                va="center",
                color="red",
            )
            if is_gui_embedding:
                current_fig.canvas.draw_idle()
            return ax

        plot_data_list = []

        # Determine the colorbar label based on what is being scanned.
        first_item_data = next(iter(self.data.values()))
        fixed_coords_type = first_item_data["fixed_coordinates"]
        if fixed_coords_type == "Pupil":  # Pupil is fixed, Field is scanned
            color_label = (
                f"Normalized Field Coordinate ({'Hx' if self.axis == 0 else 'Hy'})"
            )
        else:  # Field is fixed, Pupil is scanned
            color_label = (
                f"Normalized Pupil Coordinate ({'Px' if self.axis == 0 else 'Py'})"
            )

        # Iterate through the generated data items to prepare for plotting
        for (fixed_p1, fixed_p2, wavelength), plot_data in self.data.items():
            fixed_p1 = be.to_numpy(fixed_p1)
            fixed_p2 = be.to_numpy(fixed_p2)
            wavelength = be.to_numpy(wavelength)

            fixed_coords = plot_data["fixed_coordinates"]
            if fixed_coords == "Pupil":
                legend_label = (
                    f"Px={np.round(fixed_p1, 4).item()} "
                    f"Py={np.round(fixed_p2, 4).item()}, "
                    f"{np.round(wavelength, 4).item()} µm"
                )
            else:  # fixed_coords == 'Field'
                legend_label = (
                    f"Hx={np.round(fixed_p1, 4).item()} "
                    f"Hy={np.round(fixed_p2, 4).item()}, "
                    f"{np.round(wavelength, 4).item()} µm"
                )
            plot_data_list.append(
                (
                    plot_data["height"],
                    np.rad2deg(plot_data["angle"]),
                    plot_data["scan_range"],
                    legend_label,
                )
            )

        _plot_angle_vs_height(
            plot_data_list=plot_data_list,
            axis=self.axis,
            optic_name=self.optic.name,
            plot_style=line_style,
            ax=ax,
            title=title,
            color_label=color_label,
            cmap=cmap,
        )

        current_fig.tight_layout()

        if is_gui_embedding and hasattr(current_fig, "canvas"):
            current_fig.canvas.draw_idle()
        if show and not is_gui_embedding:
            plt.show()
        return current_fig, ax


class PupilIncidentAngleVsHeight(BaseAngleVsHeightAnalysis):
    """Represents an analysis of incident angle vs. image height by varying
    through all pupil coordinates (Px, Py) for a given image field point.

    This analysis is useful for testing the telecentricity of a lens after
    a point light source (object).

    Args:
        optic (Optic): The optic object to analyze.
        surface_idx (int, optional): Index of the surface at which the angle and
            height are measured. Defaults to -1 (last surface).
        axis (int, optional): Specifies the axis for measurement. 0 for x-axis,
            1 for y-axis. Defaults to 1 (y-axis).
        wavelength (str or float, optional): A single wavelength in microns.
            Defaults to 'primary'.
        field (tuple, optional): A single relative image field point (Hx, Hy).
            Defaults to (0, 0).
        num_points (int, optional): The number of points used for the plot.
            Defaults to 128.

    Attributes:
        optic (Optic): The optic object being analyzed.
        surface_idx (int): Index of the surface for measurements.
        axis (int): Axis for measurement (0 for x, 1 for y).
        field (tuple): The relative image field point (fixed for tracing).
        num_points (int): The number of points generated for the analysis.
        data (dict): The generated data for the analysis, structured as:
            {(Hx_fixed, Hy_fixed, wavelength_value):
            {'height': np.ndarray, 'angle': np.ndarray}}
    """

    def __init__(
        self,
        optic,
        surface_idx: int = -1,
        axis: int = 1,
        wavelength: str | float = "primary",
        field: tuple = (0, 0),
        num_points: int = 128,
    ):
        self.field = field
        super().__init__(optic, surface_idx, axis, wavelength, num_points)

    def _get_trace_coordinates(self, scan_range):
        """Defines how pupil coordinates (Px, Py) vary while field (Hx, Hy) is fixed.

        Args:
            scan_range (numpy.ndarray): The linearly spaced array for varying
                pupil coordinates.

        Returns:
            tuple: (Hx, Hy, Px, Py, label_prefix_str) for trace_generic.
        """
        hx_fixed, hy_fixed = self.field
        # Hx, Hy are constant for this analysis
        Hx = (
            be.full_like(scan_range, hx_fixed)
            if be.size(scan_range) > 0
            else be.array([hx_fixed])
        )
        Hy = (
            be.full_like(scan_range, hy_fixed)
            if be.size(scan_range) > 0
            else be.array([hy_fixed])
        )

        # Vary pupil coordinate along the specified axis
        coords = (scan_range, be.zeros_like(scan_range))
        Px, Py = coords if self.axis == 0 else coords[::-1]

        return (
            Hx,
            Hy,
            Px,
            Py,
            "Field",  # field coordinates are fixed
        )


class FieldIncidentAngleVsHeight(BaseAngleVsHeightAnalysis):
    """Represents an analysis of incident angle vs. image height by varying
    through all image field coordinates (Hx, Hy) for a given pupil field point.

    This analysis is useful for testing the telecentricity of a scan lens with
    a scan mirror at the entrance pupil. Note: Uses trace_generic(), which is
    slower than trace().

    Args:
        optic (Optic): The optic object to analyze.
        surface_idx (int, optional): Index of the surface at which the angle and
            height are measured. Defaults to -1 (last surface).
        axis (int, optional): Specifies the axis for measurement. 0 for x-axis,
            1 for y-axis. Defaults to 1 (y-axis).
        wavelength (str or float, optional): A single wavelength in microns.
            Defaults to 'primary'.
        pupil (tuple, optional): A single pupil field point (Px, Py).
            Defaults to (0, 0).
        num_points (int, optional): The number of points displayed on the plot.
            Defaults to 128.

    Attributes:
        optic (Optic): The optic object being analyzed.
        surface_idx (int): Index of the surface for measurements.
        axis (int): Axis for measurement (0 for x, 1 for y).
        pupil (tuple): The pupil field point (fixed for tracing).
        num_points (int): The number of points generated for the analysis.
        data (dict): The generated data for the analysis, structured as:
            {
                (Px_fixed, Py_fixed, wavelength_value):
                {'height': np.ndarray, 'angle': np.ndarray}
            }
    """

    def __init__(
        self,
        optic,
        surface_idx: int = -1,
        axis: int = 1,
        wavelength: str | float = "primary",
        pupil: tuple = (0, 0),
        num_points: int = 128,
    ):
        self.pupil = pupil
        super().__init__(optic, surface_idx, axis, wavelength, num_points)

    def _get_trace_coordinates(self, scan_range):
        """Defines how field coordinates (Hx, Hy) vary while pupil (Px, Py) is fixed.

        Args:
            scan_range (numpy.ndarray): The linearly spaced array for varying
                field coordinates.

        Returns:
            tuple: (Hx, Hy, Px, Py, label_prefix_str) for trace_generic.
        """
        px_fixed, py_fixed = self.pupil
        # Px, Py are constant for this analysis
        Px = (
            be.full_like(scan_range, px_fixed)
            if be.size(scan_range) > 0
            else be.array([px_fixed])
        )
        Py = (
            be.full_like(scan_range, py_fixed)
            if be.size(scan_range) > 0
            else be.array([py_fixed])
        )

        # Vary field coordinate along the specified axis
        coords = (scan_range, be.zeros_like(scan_range))
        Hx, Hy = coords if self.axis == 0 else coords[::-1]

        return (
            Hx,
            Hy,
            Px,
            Py,
            "Pupil",  # pupil coordinates are fixed
        )
