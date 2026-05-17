"""Ray Aberration Fan Analysis

This module provides a ray fan analysis for optical systems.

Kramer Harrison, 2024
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import optiland.backend as be
from optiland.distribution import create_distribution
from optiland.wavefront.strategy import BestFitSphereStrategy

from .base import BaseAnalysis


class RayFan(BaseAnalysis):
    """Represents a ray fan aberration analysis for an optic.

    Args:
        optic (Optic): The optic object to analyze.
        fields (str or list, optional): The fields to analyze.
            Defaults to 'all'.
        wavelengths (str or list, optional): The wavelengths to analyze.
            Defaults to 'all'.
        num_points (int, optional): The number of points in the ray fan.
            Defaults to 256.

    Attributes:
        optic (Optic): The optic object being analyzed.
        fields (list): The fields being analyzed.
        wavelengths (list): The wavelengths being analyzed.
        num_points (int): The number of points in the ray fan.
        data (dict): The generated ray fan data.

    Methods:
        view(figsize=(10, 3.33)): Displays the ray fan plot.

    """

    def __init__(self, optic, fields="all", wavelengths="all", num_points=256):
        from optiland.utils import resolve_fields

        _optic_ref = optic
        self.fields = resolve_fields(_optic_ref, fields)

        if num_points % 2 == 0:
            self.num_points = num_points + 1  # force to be odd so a point lies at P=0
        else:
            self.num_points = num_points

        super().__init__(optic, wavelengths)

    def view(
        self,
        fig_to_plot_on: plt.Figure = None,
        figsize: tuple[float, float] = (10, 3.33),
        *,
        show: bool = True,
    ):
        """
        Displays the ray fan plot, either in a new window or on a provided GUI figure.

        Args:
            fig_to_plot_on (plt.Figure, optional): The figure to plot on.
                If None, a new figure will be created. Defaults to None.
            figsize (tuple[float, float], optional): The size of the figure.
                Defaults to (10, 3.33).
            show (bool): If True (default), calls plt.show(). Set False for
                headless use.
        Returns:
            tuple: The current figure and its axes.
        """
        is_gui_embedding = fig_to_plot_on is not None
        num_fields = len(self.fields)

        if num_fields == 0:
            if is_gui_embedding:
                fig_to_plot_on.text(
                    0.5, 0.5, "No fields to plot.", ha="center", va="center"
                )
                if hasattr(fig_to_plot_on, "canvas"):
                    fig_to_plot_on.canvas.draw_idle()
            else:
                print("Warning (RayFan.view): No fields to plot.")
            return

        if is_gui_embedding:
            current_fig = fig_to_plot_on
            current_fig.clear()
            axs = current_fig.subplots(
                nrows=num_fields, ncols=2, sharex=True, sharey=True
            )
        else:
            current_fig, axs = plt.subplots(
                nrows=num_fields,
                ncols=2,
                figsize=(figsize[0], figsize[1] * num_fields),
                sharex=True,
                sharey=True,
            )

        axs = np.atleast_2d(axs)
        Px, Py = self.data["Px"], self.data["Py"]

        for k, fp in enumerate(self.fields):
            field = fp.coord
            ax_y, ax_x = axs[k, 0], axs[k, 1]
            for wp in self.wavelengths:
                wavelength = wp.value
                ex = self.data[f"{field}"][f"{wavelength}"]["x"]
                ey = self.data[f"{field}"][f"{wavelength}"]["y"]
                i_x = self.data[f"{field}"][f"{wavelength}"]["intensity_x"]
                i_y = self.data[f"{field}"][f"{wavelength}"]["intensity_y"]
                ex[i_x == 0], ey[i_y == 0] = be.nan, be.nan

                ax_y.plot(
                    be.to_numpy(Py),
                    be.to_numpy(ey),
                    zorder=3,
                    label=f"{wavelength:.4f} µm",
                )
                ax_x.plot(
                    be.to_numpy(Px),
                    be.to_numpy(ex),
                    zorder=3,
                    label=f"{wavelength:.4f} µm",
                )

            ax_y.grid()
            ax_y.axhline(0, lw=1, c="gray")
            ax_y.axvline(0, lw=1, c="gray")
            ax_y.set_xlabel("$P_y$")
            ax_y.set_ylabel("$\\epsilon_y$ (mm)")
            ax_y.set_xlim(-1, 1)
            ax_y.set_title(f"Hx: {field[0]:.3f}, Hy: {field[1]:.3f}")

            ax_x.grid()
            ax_x.axhline(0, lw=1, c="gray")
            ax_x.axvline(0, lw=1, c="gray")
            ax_x.set_xlabel("$P_x$")
            ax_x.set_ylabel("$\\epsilon_x$ (mm)")
            ax_x.set_xlim(-1, 1)
            ax_x.set_title(f"Hx: {field[0]:.3f}, Hy: {field[1]:.3f}")

        if num_fields > 0:
            handles, labels = axs[0, 0].get_legend_handles_labels()
            if handles:
                current_fig.legend(
                    handles,
                    labels,
                    loc="lower center",
                    bbox_to_anchor=(0.5, -0.05 / num_fields),
                    ncol=len(self.wavelengths),
                )

        current_fig.tight_layout()
        if is_gui_embedding and hasattr(current_fig, "canvas"):
            current_fig.canvas.draw_idle()
        if show and not is_gui_embedding:
            plt.show()
        return current_fig, current_fig.get_axes()

    def _remove_distortion(self, data):
        """Removes distortion from the ray fan data.

        Args:
            data (dict): The ray fan data.

        Returns:
            dict: The ray fan data with distortion removed.

        """
        wave_ref = self.optic.primary_wavelength
        center_idx = self.num_points // 2

        for fp in self.fields:
            field = fp.coord
            ref_data_x = data[f"{field}"][f"{wave_ref}"]["x"]
            ref_data_y = data[f"{field}"][f"{wave_ref}"]["y"]
            intensity_x = data[f"{field}"][f"{wave_ref}"]["intensity_x"]
            intensity_y = data[f"{field}"][f"{wave_ref}"]["intensity_y"]

            # Check if the central ray for the x-fan is valid
            if intensity_x[center_idx] > 0:
                x_offset = ref_data_x[center_idx]
            else:
                # If not, use the mean of all valid rays in the x-fan
                valid_x = ref_data_x[intensity_x > 0]
                x_offset = be.mean(valid_x) if be.size(valid_x) > 0 else 0.0

            # Check if the central ray for the y-fan is valid
            if intensity_y[center_idx] > 0:
                y_offset = ref_data_y[center_idx]
            else:
                # If not, use the mean of all valid rays in the y-fan
                valid_y = ref_data_y[intensity_y > 0]
                y_offset = be.mean(valid_y) if be.size(valid_y) > 0 else 0.0

            for wp in self.wavelengths:
                wavelength = wp.value
                orig_x = data[f"{field}"][f"{wavelength}"]["x"]
                orig_y = data[f"{field}"][f"{wavelength}"]["y"]
                data[f"{field}"][f"{wavelength}"]["x"] = orig_x - x_offset
                data[f"{field}"][f"{wavelength}"]["y"] = orig_y - y_offset
        return data

    def _generate_data(self):
        """Generates the ray fan data.

        Returns:
            dict: The generated ray fan data.

        """
        data = {}
        data["Px"] = be.linspace(-1, 1, self.num_points)
        data["Py"] = be.linspace(-1, 1, self.num_points)
        for fp in self.fields:
            field = fp.coord
            Hx, Hy = field[0], field[1]
            data[f"{field}"] = {}
            for wp in self.wavelengths:
                wavelength = wp.value
                data[f"{field}"][f"{wavelength}"] = {}

                rays_x = self.optic.trace(
                    Hx=Hx,
                    Hy=Hy,
                    wavelength=wavelength,
                    num_rays=self.num_points,
                    distribution="line_x",
                )
                data[f"{field}"][f"{wavelength}"]["x"] = rays_x.x
                data[f"{field}"][f"{wavelength}"]["intensity_x"] = rays_x.i

                rays_y = self.optic.trace(
                    Hx=Hx,
                    Hy=Hy,
                    wavelength=wavelength,
                    num_rays=self.num_points,
                    distribution="line_y",
                )
                data[f"{field}"][f"{wavelength}"]["y"] = rays_y.y
                data[f"{field}"][f"{wavelength}"]["intensity_y"] = rays_y.i

        data = self._remove_distortion(data)
        return data


class BestFitRayFan(RayFan):
    """Represents a ray fan analysis referenced to the best-fit sphere center.

    This class extends the standard `RayFan` analysis by changing the reference
    point for aberration calculations. Instead of using the chief ray's
    intersection with the image plane, it uses the lateral coordinates (x, y)
    of the center of the wavefront's best-fit sphere. This provides a measure
    of aberration relative to the point of optimal focus for the entire pupil.

    The analysis plane for determining the ray intersection points is the
    nominal image plane (the final surface in the optical model).

    Unlike the standard `RayFan`, this analysis does not recenter the plot on
    the chief ray. The origin (0,0) is the location of the best-fit sphere
    center. Therefore, the plot shows all aberrations, including distortion,
    relative to this optimal focal point.

    Args:
        optic (Optic): The optic object to analyze.
        fields (str or list, optional): The fields to analyze.
            Defaults to 'all'.
        wavelengths (str or list, optional): The wavelengths to analyze.
            Defaults to 'all'.
        num_points (int, optional): The number of points in each ray fan.
            Defaults to 256.
        num_rays_for_fit (int, optional): The number of rays (rings) in the
            hexapolar grid used to sample the pupil for the best-fit sphere
            calculation. A higher number provides a more accurate sphere center
            at the cost of computation time. Defaults to 15.

    Attributes:
        num_rays_for_fit (int): The number of rays used for the sphere fit.
    """

    def __init__(
        self,
        optic,
        fields: str | list = "all",
        wavelengths: str | list = "all",
        num_points: int = 256,
        num_rays_for_fit: int = 15,
    ):
        """Initializes the BestFitRayFan analysis."""
        self.num_rays_for_fit = num_rays_for_fit
        super().__init__(optic, fields, wavelengths, num_points)

    def _generate_data(self) -> dict:
        """Generates ray fan data using the best-fit sphere center.

        This method overrides the parent implementation. For each field and
        wavelength, it first performs a 2D ray trace across the pupil to
        calculate the wavefront. It then applies the `BestFitSphereStrategy`
        to find the 3D center of the sphere that best fits this wavefront.
        The (x, y) coordinates of this center are then used as the reference
        (origin) for the tangential and sagittal ray fan calculations.

        Returns:
            dict: A dictionary containing the computed ray fan data, structured
                  identically to the parent `RayFan` class's data.
        """
        data = {}
        data["Px"] = be.linspace(-1, 1, self.num_points)
        data["Py"] = be.linspace(-1, 1, self.num_points)

        dist_2d = create_distribution("hexapolar")
        dist_2d.generate_points(self.num_rays_for_fit)

        for fp in self.fields:
            field = fp.coord
            Hx, Hy = field
            data[f"{field}"] = {}

            # 1. Find the reference point by calculating the center of the
            #    best-fit sphere for the primary wavefront
            strategy = BestFitSphereStrategy(self.optic, dist_2d)
            strategy.compute_wavefront_data(field, self.optic.primary_wavelength)
            ref_x, ref_y, _ = strategy.center

            for wp in self.wavelengths:
                wavelength = wp.value
                data[f"{field}"][f"{wavelength}"] = {}

                # 2. Trace the tangential ray fan (along the x-axis of pupil)
                rays_x = self.optic.trace(
                    Hx=Hx,
                    Hy=Hy,
                    wavelength=wavelength,
                    num_rays=self.num_points,
                    distribution="line_x",
                )
                # 3. Calculate lateral error relative to the reference point
                data[f"{field}"][f"{wavelength}"]["x"] = rays_x.x - ref_x
                data[f"{field}"][f"{wavelength}"]["intensity_x"] = rays_x.i

                # 4. Trace the sagittal ray fan (along the y-axis of pupil)
                rays_y = self.optic.trace(
                    Hx=Hx,
                    Hy=Hy,
                    wavelength=wavelength,
                    num_rays=self.num_points,
                    distribution="line_y",
                )
                # 5. Calculate lateral error relative to the reference point
                data[f"{field}"][f"{wavelength}"]["y"] = rays_y.y - ref_y
                data[f"{field}"][f"{wavelength}"]["intensity_y"] = rays_y.i

        return data
