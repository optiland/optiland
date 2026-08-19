"""Geometric Modulation Transfer Function (MTF) Module.

This module provides the GeometricMTF class for computing the MTF
of an optical system based on spot diagram data.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

import optiland.backend as be
from optiland.analysis import SpotDiagram
from optiland.utils import resolve_wavelength

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from optiland._types import BEArray, DistributionType, ScalarOrArray
    from optiland.optic import Optic


class GeometricMTF(SpotDiagram):
    """Smith, Modern Optical Engineering 3rd edition, Section 11.9

    This class represents the Geometric MTF (Modulation Transfer Function) of
    an optical system. It inherits from the SpotDiagram class.

    Args:
        optic (Optic): The optical system for which to calculate the MTF.
        fields (str or list, optional): The field points at which to calculate
            the MTF. Defaults to 'all'.
        wavelength (str or float, optional): The wavelength at which to
            calculate the MTF. Defaults to 'primary'.
        num_rays (int, optional): The number of rays to trace for each field
            point. Defaults to 100.
        distribution (str, optional): The distribution of rays within each
            field point. Defaults to 'uniform'.
        num_points (int, optional): The number of points to sample in the MTF
            curve. Defaults to 256.
        max_freq (str or float, optional): The maximum frequency to consider
            in the MTF curve. Defaults to 'cutoff'.
        scale (bool, optional): Whether to scale the MTF curve using the
            diffraction-limited curve. Defaults to True.

    Attributes:
        num_points (int): The number of points to sample in the MTF curve.
        scale (bool): Whether to scale the MTF curve.
        max_freq (float): The maximum frequency to consider in the MTF curve.
        freq (be.ndarray): The frequency values for the MTF curve.
        mtf (list): The MTF data for each field point. Each element is a list
            containing tangential and sagittal MTF data (`be.ndarray`) for a field.
        diff_limited_mtf (be.ndarray): The diffraction-limited MTF curve.

    Methods:
        view(figsize=(12, 4), add_reference=False): Plots the MTF curve.
        _generate_mtf_data(): Generates the MTF data for each field point.
        _compute_field_data(xi, v, scale_factor): Computes the MTF data for a
            given field point.
        _plot_field(ax, mtf_data, field, color): Plots the MTF data for a
            given field point.

    """

    def __init__(
        self,
        optic: Optic,
        fields: str | list = "all",
        wavelength: str | float = "primary",
        num_rays=100,
        distribution: DistributionType = "uniform",
        num_points=256,
        max_freq="cutoff",
        scale=True,
    ):
        self.num_points = num_points
        self.scale = scale

        resolved_wavelength = resolve_wavelength(optic, wavelength)
        # wavelength must be converted to mm for frequency units cycles/mm
        self.cutoff_freq = 1 / (resolved_wavelength * 1e-3 * optic.paraxial.FNO())
        if max_freq == "cutoff":
            self.max_freq = self.cutoff_freq
        else:
            # If a specific max_freq is provided, use it directly
            self.max_freq = max_freq

        super().__init__(optic, fields, [resolved_wavelength], num_rays, distribution)

        self.freq = be.linspace(0, self.max_freq, num_points)
        self.mtf, self.diff_limited_mtf = self._generate_mtf_data()

    def view(
        self,
        fig_to_plot_on: Figure | None = None,
        figsize: tuple[float, float] = (12, 4),
        add_reference: bool = False,
    ) -> tuple[Figure, Axes]:
        """Plots the MTF curve.

        Args:
            fig_to_plot_on (plt.Figure, optional): The figure to plot on.
                If provided, the existing figure is cleared and reused.
                Defaults to None, which creates a new figure.
            figsize (tuple, optional): The size of the figure.
                Defaults to (12, 4).
            add_reference (bool, optional): Whether to add the diffraction
                limit reference curve. Defaults to False.

        Returns:
            tuple: A tuple containing the matplotlib figure and axes objects.
                If `fig_to_plot_on` is provided, the existing figure is cleared
                and reused; otherwise, a new figure is created.

        """

        is_gui_embedding = fig_to_plot_on is not None

        if is_gui_embedding:
            current_fig = fig_to_plot_on
            current_fig.clear()
            ax = current_fig.add_subplot(111)
        else:
            current_fig, ax = plt.subplots(figsize=figsize)

        for k, data in enumerate(self.mtf):
            self._plot_field(ax, data, self.fields[k], color=f"C{k}")

        if add_reference:
            ax.plot(
                be.to_numpy(self.freq),
                be.to_numpy(self.diff_limited_mtf),
                "k--",
                label="Diffraction Limit",
            )

        ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
        ax.set_xlim([0, be.to_numpy(self.max_freq)])
        ax.set_ylim([0, 1])
        ax.set_xlabel("Frequency (cycles/mm)", labelpad=10)
        ax.set_ylabel("Modulation", labelpad=10)
        current_fig.tight_layout()
        ax.grid(alpha=0.25)
        if is_gui_embedding and hasattr(current_fig, "canvas"):
            current_fig.canvas.draw_idle()
        return current_fig, ax

    def _generate_mtf_data(self):
        """Generates the MTF data for each field point.

        Returns:
            tuple: A tuple containing the MTF data for each field point and
                the scale factor.

        """
        if self.scale:
            ratio = be.clip(self.freq / self.cutoff_freq, 0.0, 1.0)
            phi = be.arccos(ratio)
            scale_factor = 2 / be.pi * (phi - be.cos(phi) * be.sin(phi))
        else:
            scale_factor = 1

        mtf = []  # TODO: add option for polychromatic MTF
        for field_data in self.data:
            spot_data_item = field_data[0]
            xi, yi = spot_data_item.x, spot_data_item.y
            mtf.append(
                [
                    self._compute_field_data(yi, self.freq, scale_factor),
                    self._compute_field_data(xi, self.freq, scale_factor),
                ],
            )
        return mtf, scale_factor

    def _compute_field_data(
        self, xi: BEArray, v: BEArray, scale_factor: ScalarOrArray
    ) -> BEArray:
        """Computes the MTF data for a given field point.

        Args:
            xi (be.ndarray): The coordinate values (x or y) of the field point.
            v (be.ndarray): The frequency values for the MTF curve.
            scale_factor (float or be.ndarray): The scale factor for the MTF curve.

        Returns:
            be.ndarray: The MTF data for the field point.

        """
        A, edges = be.histogram(xi, bins=self.num_points + 1)
        x = (edges[1:] + edges[:-1]) / 2
        dx = x[1] - x[0]

        mtf = be.copy(be.zeros_like(v))  # copy required to maintain gradient
        for k in range(len(v)):
            Ac = be.sum(A * be.cos(2 * be.pi * v[k] * x) * dx) / be.sum(A * dx)
            As = be.sum(A * be.sin(2 * be.pi * v[k] * x) * dx) / be.sum(A * dx)

            mtf[k] = be.sqrt(Ac**2 + As**2)

        return mtf * scale_factor

    def _plot_field(
        self, ax: Axes, mtf_data: list[BEArray], field: tuple[float, float], color: str
    ):
        """Plots the MTF data for a given field point.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axes object.
            mtf_data (list[be.ndarray]): The MTF data for the field point,
                containing tangential and sagittal MTF arrays.
            field (tuple[float, float]): The field point coordinates (Hx, Hy).
            color (str): The color of the plotted lines.

        """
        ax.plot(
            be.to_numpy(self.freq),
            be.to_numpy(mtf_data[0]),
            label=f"Hx: {field.coord[0]:.1f}, Hy: {field.coord[1]:.1f}, Tangential",
            color=color,
            linestyle="-",
        )
        ax.plot(
            be.to_numpy(self.freq),
            be.to_numpy(mtf_data[1]),
            label=f"Hx: {field.coord[0]:.1f}, Hy: {field.coord[1]:.1f}, Sagittal",
            color=color,
            linestyle="--",
        )
