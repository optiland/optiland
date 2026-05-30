"""Pupil Aberration Analysis

The pupil abberration is defined as the difference between the paraxial
and real ray intersection point at the stop surface of the optic. This is
specified as a percentage of the on-axis paraxial stop radius at the
primary wavelength.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

import optiland.backend as be

from .base import BaseAnalysis

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


class PupilAberration(BaseAnalysis):
    """Represents the pupil aberrations of an optic.

    The pupil abberration is defined as the difference between the paraxial
    and real ray intersection point at the stop surface of the optic. This is
    specified as a percentage of the on-axis paraxial stop radius at the
    primary wavelength.

    Args:
        optic (Optic): The optic object to analyze.
        fields (str or list, optional): The fields to analyze.
            Defaults to 'all'.
        wavelengths (str or list, optional): The wavelengths to analyze.
            Defaults to 'all'.
        num_points (int, optional): The number of points in the pupil
            aberration. Defaults to 256.

    """

    def __init__(
        self,
        optic,
        fields: str | list = "all",
        wavelengths: str | list = "all",
        num_points: int = 256,
    ):
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
        fig_to_plot_on: Figure | None = None,
        figsize: tuple[float, float] = (10, 3.33),
        *,
        show: bool = True,
    ) -> tuple[Figure, NDArray[np.object_]]:
        """
        Displays the pupil aberration plots for each field and wavelength.

        Parameters
        ----------
        fig_to_plot_on : plt.Figure, optional
            An existing matplotlib Figure to plot on. If None, a new Figure is created.
        figsize : tuple of float, optional
            Size of the figure in inches as (width, height). Used only if a new
            Figure is created.
        show : bool, optional
            If True (default), calls plt.show(). Set False for headless use.

        Returns
        -------
        tuple[plt.Figure, list[Axes]]
            The matplotlib Figure and Axes array containing the plots.

        Notes
        -----
        - If `fig_to_plot_on` is provided, the plots are embedded in the given Figure,
        otherwise a new Figure is created.
        - For each field, two subplots are created: one for aberration vs $P_y$ and one
        for aberration vs $P_x$.
        - If there are no fields to plot, a warning is printed or a message is displayed
        on the Figure.
        - A legend is added if there are plotted wavelengths.
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
                print("Warning (PupilAberration.view): No fields to plot.")
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
            ax_y.set_ylabel("Pupil Aberration (%)")
            ax_y.set_xlim(-1, 1)
            ax_y.set_title(f"Hx: {field[0]:.3f}, Hy: {field[1]:.3f}")

            ax_x.grid()
            ax_x.axhline(0, lw=1, c="gray")
            ax_x.axvline(0, lw=1, c="gray")
            ax_x.set_xlabel("$P_x$")
            ax_x.set_ylabel("Pupil Aberration (%)")
            ax_x.set_xlim(-1, 1)
            ax_x.set_title(f"Hx: {field[0]:.3f}, Hy: {field[1]:.3f}")

        if num_fields > 0:
            handles, labels = axs[0, 0].get_legend_handles_labels()
            if handles:
                current_fig.legend(
                    handles,
                    labels,
                    loc="lower center",
                    bbox_to_anchor=(0.5, -0.1 / num_fields),
                    ncol=len(self.wavelengths),
                )

        current_fig.tight_layout()
        if is_gui_embedding and hasattr(current_fig, "canvas"):
            current_fig.canvas.draw_idle()
        if show and not is_gui_embedding:
            plt.show()
        return current_fig, axs

    def _generate_data(self) -> dict[str, Any]:
        """Generate the real pupil aberration data.

        Returns:
            dict: The pupil aberration data.

        """
        stop_idx = self.optic.surfaces.stop_index

        # Maybe use a data class for complex return values
        data: dict[str, Any] = {
            "Px": be.linspace(-1, 1, self.num_points),
            "Py": be.linspace(-1, 1, self.num_points),
        }

        # determine size of stop
        self.optic.paraxial.trace(0, 1, self.optic.primary_wavelength)
        d = self.optic.surfaces.y[stop_idx, 0]

        # Paraxial trace
        self.optic.paraxial.trace(0, data["Py"], self.optic.primary_wavelength)
        parax_ref = self.optic.surfaces.y[stop_idx, :]

        for fp in self.fields:
            field = fp.coord
            Hx = field[0]
            Hy = field[1]

            data[f"{field}"] = {}
            for wp in self.wavelengths:
                wavelength = wp.value
                data[f"{field}"][f"{wavelength}"] = {}

                # Trace along the x-axis
                self.optic.trace(
                    Hx=Hx,
                    Hy=Hy,
                    wavelength=wavelength,
                    num_rays=self.num_points,
                    distribution="line_x",
                )
                real_x = self.optic.surfaces.x[stop_idx, :]
                real_int_x = self.optic.surfaces.intensity[stop_idx, :]

                # Trace along the y-axis
                self.optic.trace(
                    Hx=Hx,
                    Hy=Hy,
                    wavelength=wavelength,
                    num_rays=self.num_points,
                    distribution="line_y",
                )
                real_y = self.optic.surfaces.y[stop_idx, :]
                real_int_y = self.optic.surfaces.intensity[stop_idx, :]

                # Compute error
                error_x = (parax_ref - real_x) / d * 100
                error_x[real_int_x == 0] = be.nan

                error_y = (parax_ref - real_y) / d * 100
                error_y[real_int_y == 0] = be.nan

                data[f"{field}"][f"{wavelength}"]["x"] = error_x
                data[f"{field}"][f"{wavelength}"]["y"] = error_y

        return data
