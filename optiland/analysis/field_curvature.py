"""Field Curvature Analysis

This module provides a field curvature analysis for optical systems.

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


class FieldCurvature(BaseAnalysis):
    """Represents a class for analyzing field curvature of an optic.

    Args:
        optic (Optic): The optic object to analyze.
        wavelengths (str or list, optional): The wavelengths to analyze.
            Defaults to 'all'.
        num_points (int, optional): The number of points to generate for the
            analysis. Defaults to 128.

    Attributes:
        optic (Optic): The optic object being analyzed.
        wavelengths (list): The wavelengths being analyzed.
        num_points (int): The number of points generated for the analysis.
        data (list): The generated data for the analysis.

    Methods:
        view(figsize=(8, 5.5)): Displays a plot of the field curvature
            analysis.

    """

    def __init__(self, optic, wavelengths="all", num_points=128):
        self.num_points = num_points
        super().__init__(optic, wavelengths)

    def view(
        self,
        fig_to_plot_on: Figure | None = None,
        figsize: tuple[float, float] = (8, 5.5),
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """Displays a plot of the field curvature analysis.

        Args:
            fig_to_plot_on (plt.Figure, optional): The figure to plot on.
                If None, a new figure will be created. Defaults to None.
            figsize (tuple[float, float], optional): The size of the figure.
                Defaults to (8, 5.5).
            show (bool): If True (default), calls plt.show(). Set False for
                headless use.
        Returns:
            tuple: The current figure and its axes.
        """
        is_gui_embedding = fig_to_plot_on is not None

        if is_gui_embedding:
            current_fig = fig_to_plot_on
            current_fig.clear()
            ax = current_fig.add_subplot(111)
        else:
            current_fig, ax = plt.subplots(figsize=figsize)

        field = be.linspace(0, self.optic.fields.max_field, self.num_points)
        field_np = be.to_numpy(field)

        for k, wp in enumerate(self.wavelengths):
            dk_np_tan = be.to_numpy(self.data[k][0])
            ax.plot(
                dk_np_tan,
                field_np,
                f"C{k}",
                zorder=10,
                label=f"{wp.value:.4f} µm, Tangential",
            )
            dk_np_sag = be.to_numpy(self.data[k][1])
            ax.plot(
                dk_np_sag,
                field_np,
                f"C{k}--",
                zorder=10,
                label=f"{wp.value:.4f} µm, Sagittal",
            )

        ax.set_xlabel("Image Plane Delta (mm)")
        ax.set_ylabel("Field")
        ax.set_ylim([0, self.optic.fields.max_field])
        current_xlim = ax.get_xlim()
        max_abs_lim = max(np.abs(current_xlim))
        if max_abs_lim > 1e-9:
            ax.set_xlim([-max_abs_lim, max_abs_lim])
        ax.set_title("Field Curvature")
        ax.axvline(x=0, color="k", linewidth=0.5)
        ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
        ax.grid(True)
        current_fig.tight_layout()

        if is_gui_embedding and hasattr(current_fig, "canvas"):
            current_fig.canvas.draw_idle()
        if show and not is_gui_embedding:
            plt.show()
        return current_fig, ax

    def _generate_data(self):
        """Generates field curvature data for each wavelength by calculating the
            tangential and sagittal intersections.

        Returns:
            list: A list of np.ndarry containing the tangential and sagittal
                intersection points for each wavelength.

        """
        data = []
        for wp in self.wavelengths:
            wavelength = wp.value
            tangential = self._intersection_parabasal_tangential(wavelength)
            sagittal = self._intersection_parabasal_sagittal(wavelength)

            data.append([tangential, sagittal])

        return data

    def _intersection_parabasal_tangential(self, wavelength, delta=1e-5):
        """Calculate the intersection of parabasal rays in tangential plane.

        Args:
            wavelength (float): The wavelength of the light.
            delta (float, optional): The delta value in normalized pupil y
                coordinates for pairs of parabasal rays. Defaults to 1e-5.

        Returns:
            numpy.ndarray: The calculated intersection values.

        """
        Hx = be.zeros(2 * self.num_points)
        Hy = be.repeat(be.linspace(0, 1, self.num_points), 2)

        Px = be.zeros(2 * self.num_points)
        Py = be.tile(be.array([-delta, delta]), self.num_points)

        self.optic.trace_generic(Hx, Hy, Px, Py, wavelength=wavelength)

        M1 = self.optic.surfaces.M[-1, ::2]
        N1 = self.optic.surfaces.N[-1, ::2]

        M2 = self.optic.surfaces.M[-1, 1::2]
        N2 = self.optic.surfaces.N[-1, 1::2]

        y01 = self.optic.surfaces.y[-1, ::2]
        z01 = self.optic.surfaces.z[-1, ::2]

        y02 = self.optic.surfaces.y[-1, 1::2]
        z02 = self.optic.surfaces.z[-1, 1::2]

        t1 = (M2 * z01 - M2 * z02 - N2 * y01 + N2 * y02) / (M1 * N2 - M2 * N1)

        return t1 * N1

    def _intersection_parabasal_sagittal(self, wavelength, delta=1e-5):
        """Calculate the intersection of parabasal rays in sagittal plane.

        Args:
            wavelength (float): The wavelength of the light.
            delta (float, optional): The delta value in normalized pupil y
                coordinates for pairs of parabasal rays. Defaults to 1e-5.

        Returns:
            numpy.ndarray: The calculated intersection values.

        """
        Hx = be.zeros(2 * self.num_points)
        Hy = be.repeat(be.linspace(0, 1, self.num_points), 2)

        Px = be.tile(be.array([-delta, delta]), self.num_points)
        Py = be.zeros(2 * self.num_points)

        self.optic.trace_generic(Hx, Hy, Px, Py, wavelength=wavelength)

        L1 = self.optic.surfaces.L[-1, ::2]
        N1 = self.optic.surfaces.N[-1, ::2]

        L2 = self.optic.surfaces.L[-1, 1::2]
        N2 = self.optic.surfaces.N[-1, 1::2]

        x01 = self.optic.surfaces.x[-1, ::2]
        z01 = self.optic.surfaces.z[-1, ::2]

        x02 = self.optic.surfaces.x[-1, 1::2]
        z02 = self.optic.surfaces.z[-1, 1::2]

        t2 = (L2 * z01 - L2 * z02 - N2 * x01 + N2 * x02) / (L1 * N2 - L2 * N1)

        return t2 * N1
