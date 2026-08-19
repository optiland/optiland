"""Through Focus Spot Diagram Analysis

This module provides a class for performing through-focus spot diagram
analysis, calculating the spot diagram at various focal planes.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np

import optiland.backend as be
from optiland.analysis.spot_diagram import SpotDiagram
from optiland.analysis.through_focus import ThroughFocusAnalysis

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from optiland._types import DistributionType
    from optiland.optic import Optic


class ThroughFocusSpotDiagram(ThroughFocusAnalysis):
    """Performs spot diagram analysis over a range of focal planes.

    This class extends `ThroughFocusAnalysis` to specifically calculate and
    report RMS spot radii from spot diagrams at various focal positions.
    It utilizes the `SpotDiagram` class for the core calculations at each
    focal plane.

    Attributes:
        optic (optiland.optic.Optic): The optical system being analyzed.
        delta_focus (float): The focal shift increment in mm.
        num_steps (int): Number of focal planes analyzed before and after
            the nominal focus.
        fields (list): Resolved list of field coordinates for analysis.
        wavelengths (list): Resolved list of wavelengths for analysis.
        num_rings (int): Number of rings for pupil sampling in the
            `SpotDiagram` calculation.
        distribution (str): Pupil sampling distribution type (e.g.,
            'hexapolar', 'random') for `SpotDiagram`.
        coordinates (Literal["global", "local"]): Coordinate system used for
            spot data generation within `SpotDiagram`.
        results (list[dict[float, list[float]]]): A list where each item is a
            dictionary. Each dictionary corresponds to a single focal plane
            and maps the delta focus (float, in mm) to a list of RMS spot
            radii (list of floats, in mm). Each RMS spot radius in the list
            corresponds to a field defined in `self.fields`, calculated at the
            primary wavelength.
    """

    def __init__(
        self,
        optic: Optic,
        delta_focus: float = 0.1,
        num_steps: int = 5,
        fields="all",
        wavelengths="all",
        num_rings: int = 6,
        distribution: DistributionType = "hexapolar",
        coordinates: Literal["global", "local"] = "local",
    ):
        """Initializes the ThroughFocusSpotDiagram analysis.

        Args:
            optic (optiland.optic.Optic): The optical system to analyze.
            delta_focus (float, optional): The increment of focal shift in mm.
                Defaults to 0.1.
            num_steps (int, optional): The number of focal planes to analyze
                on either side of the nominal focus. Defaults to 5. Must be in
                range [1, 7].
            fields (list[tuple[float,float]] | str, optional): Fields for
                analysis. If "all", uses all fields from `optic.fields`.
                Otherwise, expects a list of field coordinates.
                Defaults to "all".
            wavelengths (list[float] | str, optional): Wavelengths for
                analysis. If "all", uses all wavelengths from
                `optic.wavelengths`. Otherwise, expects a list of
                wavelength values. Defaults to "all".
            num_rings (int, optional): Number of rings for pupil sampling in
                the `SpotDiagram` calculation. Defaults to 6.
            distribution (str, optional): Pupil sampling distribution type for
                `SpotDiagram` (e.g., 'hexapolar', 'random').
                Defaults to "hexapolar".
            coordinates (Literal["global", "local"], optional): Coordinate
                system for spot data generation in `SpotDiagram`.
                Defaults to "local".
        """
        self.num_rings = num_rings
        self.distribution: DistributionType = distribution
        if coordinates not in ["global", "local"]:
            raise ValueError("Coordinates must be 'global' or 'local'.")
        self.coordinates = coordinates

        super().__init__(
            optic,
            delta_focus=delta_focus,
            num_steps=num_steps,
            fields=fields,
            wavelengths=wavelengths,
        )

    def _perform_analysis_at_focus(self):
        """Calculates RMS spot radii at the current focal plane.

        This method is called by the base class for each focal step. It
        instantiates a `SpotDiagram` object for the optic's current focal
        state, calculates the RMS spot radius for each specified field at the
        primary wavelength, and returns this data.

        Note:
            This implementation re-instantiates `SpotDiagram` for each focal
            step, which involves recalculating ray data. For high-performance
            needs, optimizing this by directly accessing or reusing ray tracing
            functionality might be considered.

        Returns:
            list: a list of spot diagram data, including intersection points and
                intensity
        """
        # Extract raw coords and wavelength values so SpotDiagram can consume them
        fields_raw = [fp.coord for fp in self.fields]
        wavelengths_raw = [wp.value for wp in self.wavelengths]
        spot_diagram_at_focus = SpotDiagram(
            self.optic,
            fields=fields_raw,
            wavelengths=wavelengths_raw,
            num_rings=self.num_rings,
            distribution=self.distribution,
            coordinates=self.coordinates,
        )
        return spot_diagram_at_focus.data

    def view(
        self,
        fig_to_plot_on: Figure | None = None,
        figsize_per_plot: tuple[float, float] = (3, 3),
        buffer: float = 1.05,
        *,
        show: bool = True,
    ) -> tuple[Figure, list[Axes]] | None:
        """
        Visualizes the through-focus spot diagrams, either in a new window or on a
        provided GUI figure.

        Args:
            fig_to_plot_on: A matplotlib figure to plot on.
                If None, a new figure will be created.
            figsize_per_plot: Size of each subplot
            in inches
                (width, height). Defaults to (3, 3).
            buffer: Scaling buffer applied to the maximum radius
                for axis limits. Defaults to 1.05.
            show (bool): If True (default), calls plt.show(). Set False for
                headless use.

        Returns:
            A tuple containing the figure and a list of axes used for plotting. Or None
                if updating the GUI.
        """
        is_gui_embedding = fig_to_plot_on is not None
        if not self._validate_view_prerequisites():
            if is_gui_embedding:
                fig_to_plot_on.text(
                    0.5, 0.5, "No data to display.", ha="center", va="center"
                )
                if hasattr(fig_to_plot_on, "canvas"):
                    fig_to_plot_on.canvas.draw_idle()
            return

        num_fields = len(self.fields)
        num_steps = self.num_steps

        if is_gui_embedding:
            current_fig = fig_to_plot_on
            current_fig.clear()
        else:
            current_fig = plt.figure(
                figsize=(
                    num_steps * figsize_per_plot[0],
                    num_fields * figsize_per_plot[1],
                )
            )

        axs = current_fig.subplots(
            num_fields, num_steps, sharex=True, sharey=True, squeeze=False
        )

        global_axis_limit = self._compute_global_axis_limit(buffer)
        x_label, y_label = self._get_plot_axis_labels()
        legend_handles, legend_labels = [], []

        for i, fp in enumerate(self.fields):
            field_coord = fp.coord
            for j, position in enumerate(self.positions):
                ax = axs[i, j]
                data = self.results[j][i]
                defocus = float(position) - be.to_numpy(self.nominal_focus).item()
                centroid_x, centroid_y = self._get_spot_centroid(data)
                self._plot_wavelengths(
                    ax,
                    data,
                    centroid_x,
                    centroid_y,
                    i,
                    j,
                    legend_handles,
                    legend_labels,
                )
                self._configure_subplot(
                    ax,
                    field_coord,
                    defocus,
                    i,
                    j,
                    num_fields,
                    x_label,
                    y_label,
                    global_axis_limit,
                )

        self._add_legend(
            current_fig, legend_handles, legend_labels, num_fields, figsize_per_plot
        )
        current_fig.tight_layout(rect=(0, 0.03, 1, 0.97))

        if is_gui_embedding and hasattr(current_fig, "canvas"):
            current_fig.canvas.draw_idle()
        if show and not is_gui_embedding:
            plt.show()
        return current_fig, current_fig.get_axes()

    def _validate_view_prerequisites(self) -> bool:
        """Validates prerequisites before plotting.

        Checks whether results, fields, and wavelengths are present
        and non-empty.

        Returns:
            True if plotting can proceed, False otherwise.
        """
        if not self.results:
            print("No data to display. Run analysis first.")
            return False
        if not self.fields or not self.wavelengths or self.num_steps == 0:
            print("No fields, defocus steps, or wavelengths to plot.")
            return False
        return True

    def _create_subplot_grid(
        self, num_fields: int, num_steps: int, figsize_per_plot: tuple[float, float]
    ) -> tuple[Figure, NDArray[np.object_]]:
        """Creates a 2D grid of subplots.

        Args:
            num_fields: Number of rows (fields).
            num_steps: Number of columns (defocus steps).
            figsize_per_plot: Size per subplot in inches.

        Returns:
            tuple: (matplotlib.figure.Figure, ndarray of Axes).
        """
        fig, axs = plt.subplots(
            num_fields,
            num_steps,
            figsize=(num_steps * figsize_per_plot[0], num_fields * figsize_per_plot[1]),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        return fig, axs

    def _get_plot_axis_labels(self) -> tuple[str, str]:
        """Determines axis labels based on image surface orientation.

        Returns:
            tuple[str, str]: Labels for the X and Y axes.
        """
        cs = self.optic.image_surface.geometry.cs
        orientation = np.abs(be.to_numpy(cs.get_effective_rotation_euler()))
        tol = 0.01
        if orientation[0] > tol or orientation[1] > tol:
            return "U (mm)", "V (mm)"
        return "X (mm)", "Y (mm)"

    def _compute_global_axis_limit(self, buffer: float) -> float:
        """Computes a global axis limit for consistent plot scaling.

        Considers the maximum geometric radius of spot positions
        (centered by centroid) across all defocus steps and fields.

        Args:
            buffer (float): Scaling buffer applied to max radius.

        Returns:
            float: Global axis limit after applying buffer.
        """
        max_r_sq = 0.0
        for data_at_step in self.results:
            for field_data in data_at_step:
                centroid_x, centroid_y = self._get_spot_centroid(field_data)
                for spot_data in field_data:
                    valid = spot_data.intensity != 0
                    if be.any(valid):
                        dx = spot_data.x - centroid_x
                        dy = spot_data.y - centroid_y
                        r_sq = dx[valid] ** 2 + dy[valid] ** 2
                        max_r_sq = max(max_r_sq, be.to_numpy(be.max(r_sq)).item())
        return np.sqrt(max_r_sq) * buffer if max_r_sq > 0 else 0.01

    def _get_spot_centroid(self, field_data: list) -> tuple[float, float]:
        """Computes the centroid of spot data for the primary wavelength.

        Uses intensity-weighted centroid unless all rays have zero intensity,
        in which case returns (0.0, 0.0).

        Args:
            field_data (list): List of spot data items across wavelengths.

        Returns:
            tuple[float, float]: (x, y) centroid in mm.
        """
        idx = self.optic.wavelengths.primary_index
        idx = min(idx, len(field_data) - 1)
        spot = field_data[idx]

        nonzero = spot.intensity != 0
        if be.any(nonzero):
            cx = be.to_numpy(be.mean(spot.x[nonzero])).item()
            cy = be.to_numpy(be.mean(spot.y[nonzero])).item()
        else:
            cx = cy = 0.0
        return cx, cy

    def _plot_wavelengths(
        self,
        ax: Axes,
        field_data: list,
        cx: float,
        cy: float,
        i: int,
        j: int,
        handles: list,
        labels: list,
    ):
        """Plots rays for all wavelengths, centered at the primary centroid.

        Args:
            ax (matplotlib.axes.Axes): Axis object to draw on.
            field_data (list): List of spot data for one field at one defocus step.
            cx (float): Centroid x-coordinate.
            cy (float): Centroid y-coordinate.
            i (int): Field index (row).
            j (int): Defocus step index (column).
            handles (list): List to store legend handle objects.
            labels (list): List to store corresponding legend labels.
        """
        markers = ["o", "s", "^"]
        for k, spot in enumerate(field_data):
            x = be.to_numpy(spot.x - cx)
            y = be.to_numpy(spot.y - cy)
            i_mask = be.to_numpy(spot.intensity) != 0

            if np.any(i_mask):
                scatter = ax.scatter(
                    x[i_mask],
                    y[i_mask],
                    s=10,
                    marker=markers[k % len(markers)],
                    alpha=0.7,
                )
                if i == 0 and j == 0:
                    wl = self.wavelengths[k].value
                    handles.append(scatter)
                    labels.append(f"{wl:.4f} µm")

    def _configure_subplot(
        self,
        ax: Axes,
        field: tuple,
        defocus: float,
        i: int,
        j: int,
        num_fields: int,
        x_label: str,
        y_label: str,
        limit: float,
    ):
        """Applies titles, labels, and axis limits to a subplot.

        Args:
            ax (matplotlib.axes.Axes): Axis to configure.
            field (tuple): Field coordinates (x, y).
            defocus (float): Defocus amount in mm.
            i (int): Field index.
            j (int): Defocus step index.
            num_fields (int): Total number of fields.
            x_label (str): Label for x-axis.
            y_label (str): Label for y-axis.
            limit (float): Axis limit for both x and y.
        """
        ax.axis("square")
        ax.grid(alpha=0.25)

        title = f"Field: ({field[0]:.2f},{field[1]:.2f})"
        if i == 0:
            title = f"Defocus: {defocus:+.3f} mm\n{title}"
        ax.set_title(title, fontsize=10)

        if i == num_fields - 1:
            ax.set_xlabel(x_label)
        if j == 0:
            ax.set_ylabel(y_label)

        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)

    def _add_legend(
        self,
        fig: Figure,
        handles: list,
        labels: list,
        num_fields: int,
        figsize_per_plot: tuple[float, float],
    ):
        """Adds a wavelength legend below the plot grid.

        Args:
            fig (matplotlib.figure.Figure): Figure object.
            handles (list): Legend handles for plotted wavelengths.
            labels (list): Corresponding labels.
            num_fields (int): Number of fields (rows).
            figsize_per_plot (tuple): Subplot size in inches.
        """
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=min(5, len(labels)),
                bbox_to_anchor=(0.5, -0.02 / (figsize_per_plot[1] * num_fields / 4)),
            )
