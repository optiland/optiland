"""Distortion Analysis

This module provides a distortion analysis for optical systems.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

import optiland.backend as be

from .base import BaseAnalysis
from .distortion_strategies import DistortionModel, create_distortion_model

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class Distortion(BaseAnalysis):
    """Represents a distortion analysis for an optic.

    Args:
        optic (Optic): The optic object to analyze.
        wavelengths (str or list, optional): The wavelengths to analyze.
            Defaults to 'all'.
        num_points (int, optional): The number of points to generate for the
            analysis. Defaults to 128.
        distortion_type (str, optional): The type of distortion analysis.
            Defaults to 'f-tan'.
        method (str or DistortionModel, optional): The distortion strategy to
            use. ``"paraxial"`` (default) traces a chief ray against a
            rotationally symmetric reference. ``"nonparaxial"`` uses the
            transmitted-energy centroid against a best-fit affine reference,
            enabling distortion for off-axis, freeform, or obscured systems
            where no chief ray can be traced. A custom :class:`DistortionModel`
            instance may also be supplied.

    Attributes:
        optic (Optic): The optic object being analyzed.
        wavelengths (list): The wavelengths being analyzed.
        num_points (int): The number of points generated for the analysis.
        distortion_type (str): The type of distortion analysis.
        method: The distortion strategy used.
        data (list): The generated distortion data.

    Methods:
        view(figsize=(7, 5.5)): Visualizes the distortion analysis.

    """

    def __init__(
        self,
        optic,
        wavelengths: str | list = "all",
        num_points: int = 128,
        distortion_type: str = "f-tan",
        method: str | DistortionModel = "paraxial",
    ):
        self.num_points = num_points
        self.distortion_type = distortion_type
        self.method = method
        super().__init__(optic, wavelengths)

    def view(
        self,
        fig_to_plot_on: Figure | None = None,
        figsize: tuple[float, float] = (7, 5.5),
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """Visualize the distortion analysis.

        Args:
            fig_to_plot_on (plt.Figure, optional): The figure to plot on.
                If None, a new figure will be created. Defaults to None.
            figsize (tuple, optional): The size of the figure to create.
                Defaults to (7, 5.5).
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

        ax.axvline(x=0, color="k", linewidth=1, linestyle="--")
        field = be.linspace(1e-10, self.optic.fields.max_field, self.num_points)
        field_np = be.to_numpy(field)

        for k, wp in enumerate(self.wavelengths):
            dist_k_np = be.to_numpy(self.data[k])
            ax.plot(dist_k_np, field_np, label=f"{wp.value:.4f} µm")

        ax.set_xlabel("Distortion (%)")
        ax.set_ylabel("Field")

        xlims = ax.get_xlim()
        max_abs_lim = max(np.abs(xlims))
        ax.set_xlim(-max_abs_lim, max_abs_lim)
        ax.set_ylim(0, None)
        ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
        ax.grid(True)
        current_fig.tight_layout()

        if is_gui_embedding and hasattr(current_fig, "canvas"):
            current_fig.canvas.draw_idle()
        if show and not is_gui_embedding:
            plt.show()
        return current_fig, ax

    def _generate_data(self):
        """Generate data for analysis.

        This method generates the distortion data to be used for plotting.

        Returns:
            list: A list of distortion data points.

        """
        model = create_distortion_model(
            self.method, distortion_type=self.distortion_type
        )

        Hx = be.zeros(self.num_points)
        Hy = be.linspace(1e-10, 1, self.num_points)

        data = []
        for wp in self.wavelengths:
            result = model.compute(self.optic, Hx, Hy, wp.value)
            data.append(model.percent(result, signed=True))

        return data
