"""RMS versus Field Analysis

This module enables the calculation of both the RMS spot size and the RMS
wavefront error versus field coordinate of an optical system.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

import optiland.backend as be
from optiland.analysis.spot_diagram import SpotDiagram
from optiland.wavefront import Wavefront

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from optiland.optic import Optic


class RmsSpotSizeVsField(SpotDiagram):
    """RMS Spot Size versus Field Coordinate.

    This class is used to analyze the RMS spot size versus field coordinate
    of an optical system.

    Args:
        optic (Optic): the optical system.
        num_fields (int): the number of fields. Default is 64.
        wavelengths (list): the wavelengths to be analyzed. Default is 'all'.
        num_rings (int): the number of rings. Default is 6.
        distribution (str): the distribution of the fields.
            Default is 'hexapolar'.

    """

    def __init__(
        self,
        optic,
        num_fields: int = 64,
        wavelengths="all",
        num_rings: int = 6,
        distribution: str = "hexapolar",
    ):
        self.num_fields = num_fields
        fields = [(0, Hy) for Hy in be.linspace(0, 1, num_fields)]
        super().__init__(optic, fields, wavelengths, num_rings, distribution)

        self._field = be.array(fields)
        self._spot_size = be.array(self.rms_spot_radius())

    def view(
        self,
        fig_to_plot_on: Figure | None = None,
        figsize: tuple[float, float] = (7, 4.5),
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Plots the RMS spot size versus the normalized Y field coordinate for each
        analysis wavelength.

        Parameters
        ----------
        fig_to_plot_on : plt.Figure, optional
            An existing matplotlib Figure to plot on. If provided, the plot will be
            embedded in this figure.
            If None (default), a new figure will be created.
        figsize : tuple of float, optional
            Size of the figure to create if `fig_to_plot_on` is None.
            Default is (7, 4.5).
        show : bool, optional
            If True (default), calls plt.show(). Set False for headless use.

        Returns
        -------
        tuple[Figure, Axes]
            The matplotlib Figure and Axes objects containing the plot.

        Notes
        -----
        - Each wavelength's RMS spot size is plotted as a separate line for clarity
        and legend handling.
        - The legend is placed outside the plot area for better readability.
        - The method is suitable for both standalone plotting and GUI embedding.
        """
        is_gui_embedding = fig_to_plot_on is not None

        if is_gui_embedding:
            current_fig = fig_to_plot_on
            current_fig.clear()
            ax = current_fig.add_subplot(111)
        else:
            current_fig, ax = plt.subplots(figsize=figsize)

        analysis_wavelengths = self.wavelengths
        spot_size_data = be.to_numpy(self._spot_size)

        # Plot each wavelength's data as a separate line to handle legends correctly.
        for i, wp in enumerate(analysis_wavelengths):
            ax.plot(
                be.to_numpy(self._field[:, 1]),
                spot_size_data[:, i],
                label=f"{wp.value:.4f} µm",
            )

        ax.set_xlabel("Normalized Y Field Coordinate")
        ax.set_ylabel("RMS Spot Size (mm)")
        ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, None)
        ax.grid()
        current_fig.tight_layout()

        if is_gui_embedding and hasattr(current_fig, "canvas"):
            current_fig.canvas.draw_idle()
        if show and not is_gui_embedding:
            plt.show()
        return current_fig, ax


class RmsWavefrontErrorVsField(Wavefront):
    """RMS Wavefront Error versus Field Coordinate.

    This class is used to analyze the RMS wavefront error versus field
    coordinate of an optical system.

    Args:
        optic (Optic): the optical system.
        num_fields (int): the number of fields. Default is 32.
        wavelengths (list): the wavelengths to be analyzed. Default is 'all'.
        num_rays (int): the number of rays. Default is 12.
        distribution (str): the distribution of the fields.
            Default is 'hexapolar'.

    """

    def __init__(
        self,
        optic: Optic,
        num_fields: int = 32,
        wavelengths: str = "all",
        num_rays: int = 12,
        distribution: str = "hexapolar",
    ):
        self.num_fields = num_fields
        fields = [(0, Hy) for Hy in be.linspace(0, 1, num_fields)]
        super().__init__(optic, fields, wavelengths, num_rays, distribution)

        self._field = be.array(fields)
        self._wavefront_error = be.array(self._rms_wavefront_error())

    def view(
        self,
        fig_to_plot_on: Figure | None = None,
        figsize: tuple[float, float] = (7, 4.5),
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """View the RMS wavefront error versus field coordinate."""
        is_gui_embedding = fig_to_plot_on is not None

        if is_gui_embedding:
            current_fig = fig_to_plot_on
            current_fig.clear()
            ax = current_fig.add_subplot(111)
        else:
            current_fig, ax = plt.subplots(figsize=figsize)

        analysis_wavelengths = self.wavelengths
        wavefront_error_data = be.to_numpy(self._wavefront_error)

        # Plot each wavelength's data as a separate line.
        for i, wp in enumerate(analysis_wavelengths):
            ax.plot(
                be.to_numpy(self._field[:, 1]),
                wavefront_error_data[:, i],
                label=f"{wp.value:.4f} µm",
            )
        ax.set_xlabel("Normalized Y Field Coordinate")
        ax.set_ylabel("RMS Wavefront Error (waves)")
        ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, None)
        ax.grid()
        current_fig.tight_layout()

        if is_gui_embedding and hasattr(current_fig, "canvas"):
            current_fig.canvas.draw_idle()
        if show and not is_gui_embedding:
            plt.show()
        return current_fig, ax

    def _rms_wavefront_error(self):
        """Calculate the RMS wavefront error."""
        rows = []
        for fp in self.fields:
            field = fp.coord
            cols = []
            for wp in self.wavelengths:
                wavefront_data = self.get_data(field, wp.value)
                rms_ij = be.sqrt(be.mean(wavefront_data.opd**2))
                cols.append(rms_ij)
            # turn this row into a backend array/tensor
            rows.append(be.stack(cols, axis=0))
        # stack all rows into the final 2D result
        return be.stack(rows, axis=0)
