"""Through Focus MTF

This module provides a class for performing through-focus MTF
analysis, calculating the MTF at various focal planes for a given
spatial frequency, wavelength, and fields.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

import optiland.backend as be
from optiland.analysis.through_focus import ThroughFocusAnalysis
from optiland.mtf.sampled import SampledMTF

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class ThroughFocusMTF(ThroughFocusAnalysis):
    """
    Performs Modulation Transfer Function (MTF) analysis across a range of focal
    positions.

    This class calculates the MTF at a specified spatial frequency for both
    tangential and sagittal orientations at multiple focal planes around the
    nominal focus of an optical system.

    The results include tangential and sagittal MTF values for each analyzed
    field at each focal step.

    Args:
        optic: The optiland.optic.Optic object to analyze.
        spatial_frequency (float): The single spatial frequency (in cycles/mm)
            at which to calculate MTF. The calculation will be performed
            for both tangential (fx = spatial_frequency, fy = 0) and
            sagittal (fx = 0, fy = spatial_frequency) orientations.
        delta_focus (float, optional): The increment of focal shift in mm.
            Defaults to 0.1.
        num_steps (int, optional): The number of focal planes to analyze
            before and after the nominal focus. Must be an odd integer.
            Defaults to 5.
        fields (list[tuple[float, float]] | str, optional): Fields for
            analysis. If "all", uses all fields from `optic.fields`.
            Defaults to "all".
        wavelength (float | str, optional): The wavelength (in µm) for
            analysis. If "primary", uses the primary wavelength from
            `optic.primary_wavelength`. Defaults to "primary".
        num_rays (int, optional): The number of rays across the pupil in 1D
            for the SampledMTF calculation. Defaults to 64.
    """

    MAX_STEPS = 15
    MIN_STEPS = 1

    def __init__(
        self,
        optic,
        spatial_frequency: float,
        delta_focus: float = 0.1,
        num_steps: int = 5,
        fields: list[tuple[float, float]] | str = "all",
        wavelength: float | str = "primary",
        num_rays: int = 128,
    ):
        self.spatial_frequency = spatial_frequency
        self.num_rays = num_rays

        if wavelength == "primary":
            self.wavelength = optic.primary_wavelength
        else:
            self.wavelength = wavelength

        super().__init__(
            optic,
            delta_focus=delta_focus,
            num_steps=num_steps,
            fields=fields,
            wavelengths=[self.wavelength],
        )

    def _perform_analysis_at_focus(self):
        """
        Performs the MTF analysis at the current focal position for all fields.

        This method is called by the base class for each focal step. It
        calculates the tangential and sagittal MTF values for the specified
        spatial frequency for each field defined in `self.fields`.

        Returns:
            list[dict[str, float]]: A list of dictionaries, where each
            dictionary corresponds to a field and contains the tangential
            and sagittal MTF values, e.g.,
            [{'tangential': 0.5, 'sagittal': 0.45},  # Field 1
             {'tangential': 0.3, 'sagittal': 0.28}]  # Field 2
        """
        results_at_this_focus = []
        for fp in self.fields:
            field_coord = fp.coord
            sampled_mtf = SampledMTF(
                optic=self.optic,
                field=field_coord,
                wavelength=self.wavelength,
                num_rays=self.num_rays,
                distribution="uniform",
                zernike_terms=37,
                zernike_type="fringe",
            )

            freq_tan = (self.spatial_frequency, 0.0)
            freq_sag = (0.0, self.spatial_frequency)

            mtf_t = sampled_mtf.calculate_mtf([freq_tan])[0]
            mtf_s = sampled_mtf.calculate_mtf([freq_sag])[0]

            results_at_this_focus.append({"tangential": mtf_t, "sagittal": mtf_s})
        return results_at_this_focus

    def view(
        self,
        fig_to_plot_on: Figure | None = None,
        figsize: tuple[float, float] = (12, 4),
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Visualizes the through-focus Modulation Transfer Function (MTF) results for
        each analyzed field.

        This method generates a plot of tangential and sagittal MTF values as a function
        of defocus for each field position.
        If enough data points are available, spline smoothing is applied to the MTF
        data to produce smoother curves.
        Otherwise, raw data points are plotted. The plot displays MTF at the spatial
        frequency specified during initialization.

        Parameters
        ----------
        fig_to_plot_on : plt.Figure, optional
            An existing matplotlib Figure to plot on. If provided, the plot will be
            embedded in this figure.
            If None (default), a new figure will be created.
        figsize : tuple of float, optional
            Size of the figure to create if `fig_to_plot_on` is None.
            Default is (12, 4).
        show : bool, optional
            If True (default), calls plt.show(). Set False for headless use.

        Returns
        -------
        tuple[Figure, Axes]
            The matplotlib Figure and Axes objects containing the plot.

        Notes
        -----
        - Spline smoothing uses cubic splines if at least 4 data points are available,
        linear splines for 2-3 points, and raw data is plotted if fewer points
        are present.
        - The legend displays the field coordinates (Hx, Hy) for each curve.
        - The plot includes grid lines and is formatted for clarity.
        """
        is_gui_embedding = fig_to_plot_on is not None

        if is_gui_embedding:
            current_fig = fig_to_plot_on
            current_fig.clear()
            ax = current_fig.add_subplot(111)
        else:
            current_fig, ax = plt.subplots(figsize=figsize)

        np_positions = be.to_numpy(be.asarray(self.positions))
        np_nominal_focus = be.to_numpy(be.asarray(self.nominal_focus))
        defocus_values_np = np_positions - np_nominal_focus

        for i_field, fp in enumerate(self.fields):
            field_coord = fp.coord
            mtf_t_values = be.to_numpy(
                be.asarray(
                    [
                        self.results[i_pos][i_field]["tangential"]
                        for i_pos in range(self.num_steps)
                    ]
                )
            )
            mtf_s_values = be.to_numpy(
                be.asarray(
                    [
                        self.results[i_pos][i_field]["sagittal"]
                        for i_pos in range(self.num_steps)
                    ]
                )
            )

            num_data_points = len(defocus_values_np)
            Hx, Hy = field_coord

            # Determine spline order k based on number of points
            if num_data_points >= 4:  # Need at least k+1 points for spline of degree k
                k = 3  # Cubic spline
            elif num_data_points >= 2:
                k = 1  # Linear spline
            else:
                k = 0  # No spline, just plot points

            if k == 0:
                # Plot raw data if spline conditions not met or k was initially 0
                ax.plot(
                    defocus_values_np,
                    np.clip(mtf_t_values, 0, 1),
                    linestyle="-",
                    marker="o",
                    markersize=4,
                    color=f"C{i_field}",
                    label=f"Hx: {Hx:.1f}, Hy: {Hy:.1f}, Tangential (raw)",
                )
                ax.plot(
                    defocus_values_np,
                    np.clip(mtf_s_values, 0, 1),
                    linestyle="--",
                    marker="x",
                    markersize=4,
                    color=f"C{i_field}",
                    label=f"Hx: {Hx:.1f}, Hy: {Hy:.1f}, Sagittal (raw)",
                )
            else:
                defocus_smooth = np.linspace(
                    defocus_values_np.min(), defocus_values_np.max(), 256
                )

                spl_t = make_interp_spline(
                    defocus_values_np, mtf_t_values, k=k, check_finite=False
                )
                mtf_t_smooth = spl_t(defocus_smooth)

                spl_s = make_interp_spline(
                    defocus_values_np, mtf_s_values, k=k, check_finite=False
                )
                mtf_s_smooth = spl_s(defocus_smooth)

                ax.plot(
                    defocus_smooth,
                    np.clip(mtf_t_smooth, 0, 1),
                    linestyle="-",
                    color=f"C{i_field}",
                    label=f"Hx: {Hx:.1f}, Hy: {Hy:.1f}, Tangential",
                )
                ax.plot(
                    defocus_smooth,
                    np.clip(mtf_s_smooth, 0, 1),
                    linestyle="--",
                    color=f"C{i_field}",
                    label=f"Hx: {Hx:.1f}, Hy: {Hy:.1f}, Sagittal",
                )

        ax.set_title(
            f"Through-Focus MTF at {self.spatial_frequency} "
            f"cycles/mm, λ={self.wavelength:.3f} µm"
        )

        ax.set_xlabel("Defocus (mm)")
        ax.set_ylabel("MTF")
        ax.set_xlim([np.min(defocus_values_np), np.max(defocus_values_np)])
        ax.set_ylim([0, 1.05])
        ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
        ax.grid(True, linestyle=":", alpha=0.5)

        current_fig.tight_layout()

        if is_gui_embedding and hasattr(current_fig, "canvas"):
            current_fig.canvas.draw_idle()
        if show and not is_gui_embedding:
            plt.show()
        return current_fig, ax
