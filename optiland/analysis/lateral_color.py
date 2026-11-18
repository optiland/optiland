"""Lateral Color Analysis

This module provides a lateral color (transverse chromatic aberration) analysis
for optical systems.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

import optiland.backend as be
from optiland.visualization.system.utils import transform

from .base import BaseAnalysis

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from optiland.optic import Optic


class LateralColor(BaseAnalysis):
    """Lateral color: change of real chief-ray image height with wavelength.

    For each field point ``Hy`` in ``linspace(0, 1, num_points)`` (``Hx = 0``)
    the real chief ray is traced at every wavelength. The lateral color of a
    wavelength is its chief-ray height on the image surface minus that of the
    primary wavelength, ``y(λ) - y(primary)``, in the image surface's local
    frame and in lens units (mm). A positive value means that wavelength lands
    at a larger local y than the primary wavelength, which is farther from the
    axis when the image height is positive.

    Args:
        optic (Optic): The optic object to analyze.
        wavelengths (str | list, optional): The wavelengths to analyze.
            Defaults to 'all'.
        num_points (int, optional): The number of field points. Defaults to 64.

    Attributes:
        optic (Optic): The optic object being analyzed.
        wavelengths (list): The wavelengths being analyzed.
        num_points (int): The number of field points.
        data (dict): ``"Hy"``, the normalized field coordinates, and
            ``"lateral_color"``, one array per wavelength (same order as
            ``wavelengths``) holding the lateral color in mm. Chief rays that
            do not reach the image surface give NaN.
    """

    def __init__(
        self, optic: Optic, wavelengths: str | list = "all", num_points: int = 64
    ):
        self.num_points = num_points
        super().__init__(optic, wavelengths)

    def view(
        self,
        fig_to_plot_on: Figure | None = None,
        figsize: tuple[float, float] = (8, 5.5),
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """Displays a plot of the lateral color versus field.

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

        field_np = be.to_numpy(self.data["Hy"]) * self.optic.fields.max_field

        for k, wp in enumerate(self.wavelengths):
            lateral_color_um = be.to_numpy(self.data["lateral_color"][k]) * 1e3
            ax.plot(
                lateral_color_um,
                field_np,
                f"C{k}",
                zorder=10,
                label=f"{wp.value:.4f} µm",
            )

        ax.set_xlabel("Lateral Color (µm)")
        ax.set_ylabel("Field")
        ax.set_ylim([0, self.optic.fields.max_field])
        current_xlim = ax.get_xlim()
        max_abs_lim = max(np.abs(current_xlim))
        if max_abs_lim > 1e-9:
            ax.set_xlim([-max_abs_lim, max_abs_lim])
        ax.set_title("Lateral Color")
        ax.axvline(x=0, color="k", linewidth=0.5)
        ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
        ax.grid(True)
        current_fig.tight_layout()

        if is_gui_embedding and hasattr(current_fig, "canvas"):
            current_fig.canvas.draw_idle()
        if show and not is_gui_embedding:
            plt.show()
        return current_fig, ax

    def _generate_data(self) -> dict:
        """Traces the chief ray at each field and wavelength.

        Returns:
            dict: ``"Hy"`` (normalized field coordinates) and
                ``"lateral_color"`` (one array per wavelength, in mm).
        """
        Hy = be.linspace(0, 1, self.num_points)
        primary = self.optic.primary_wavelength
        # Trace each distinct wavelength once, including the primary reference.
        values = dict.fromkeys([wp.value for wp in self.wavelengths] + [primary])
        heights = {value: self._chief_ray_height(Hy, value) for value in values}
        lateral_color = [
            heights[wp.value] - heights[primary] for wp in self.wavelengths
        ]
        return {"Hy": Hy, "lateral_color": lateral_color}

    def _chief_ray_height(self, Hy, wavelength: float):
        """Real chief-ray y on the image surface, in its local frame.

        Args:
            Hy (be.ndarray): Normalized y field coordinates.
            wavelength (float): Wavelength in µm.

        Returns:
            be.ndarray: Local y per field point; NaN where the ray is vignetted.
        """
        zeros = be.zeros_like(Hy)
        self.optic.trace_generic(zeros, Hy, zeros, zeros, wavelength)

        surfaces = self.optic.surfaces
        _, y_local, _ = transform(
            surfaces.x[-1, :],
            surfaces.y[-1, :],
            surfaces.z[-1, :],
            self.optic.image_surface,
            is_global=True,
        )
        return be.where(surfaces.intensity[-1, :] > 0, y_local, be.nan)
