"""MTF versus Field Analysis

This module enables the calculation of the Modulation Transfer Function (MTF)
versus field coordinate of an optical system.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

import optiland.backend as be
from optiland.analysis.base import BaseAnalysis
from optiland.mtf.sampled import SampledMTF

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from optiland.optic import Optic


class MTFvsField(BaseAnalysis):
    """MTF versus Field Coordinate.

    This class is used to analyze the Modulation Transfer Function (MTF) versus
    the field coordinate of an optical system for specified spatial frequencies.

    Args:
        optic (Optic): the optical system.
        frequencies (list[float]): the spatial frequencies (in cycles/mm) to analyze.
        num_fields (int): the number of fields in the Y direction. Default is 32.
        wavelengths (str or list): the wavelengths to be analyzed. Default is 'all'.
        num_rays (int): the number of rays across the pupil in 1D for the SampledMTF
            calculation. Default is 128.
        override_limits (bool): If True, bypasses the limit on the number of frequencies
            and wavelengths to prevent cluttered plots. Default is False.

    """

    MAX_FREQUENCIES = 5
    MAX_WAVELENGTHS = 3

    def __init__(
        self,
        optic: Optic,
        frequencies: list[float],
        num_fields: int = 32,
        wavelengths: str | list[float] = "all",
        num_rays: int = 128,
        override_limits: bool = False,
    ):
        self.frequencies = frequencies
        self.num_fields = num_fields
        self.num_rays = num_rays

        self._check_limits(override_limits, wavelengths, optic)

        # Base Analysis will set self.wavelengths
        super().__init__(optic, wavelengths)

    def _check_limits(self, override_limits: bool, wavelengths, optic):
        """Check to ensure inputs won't produce an overly cluttered plot."""
        if override_limits:
            return

        if len(self.frequencies) > self.MAX_FREQUENCIES:
            raise ValueError(
                f"Number of frequencies ({len(self.frequencies)}) exceeds the "
                f"recommended limit of {self.MAX_FREQUENCIES} for clean plots. "
                "Set override_limits=True to bypass this check."
            )

        from optiland.utils import resolve_wavelengths as _resolve_wls

        resolved_wls = _resolve_wls(optic, wavelengths)
        num_wl = len(resolved_wls)

        if num_wl > self.MAX_WAVELENGTHS:
            raise ValueError(
                f"Number of wavelengths ({num_wl}) exceeds the recommended "
                f"limit of {self.MAX_WAVELENGTHS} for clean plots. "
                "Set override_limits=True to bypass this check."
            )

    def _generate_data(self):
        """Generate the MTF data across fields, wavelengths, and frequencies."""
        fields = [(0.0, float(Hy)) for Hy in be.linspace(0.0, 1.0, self.num_fields)]
        self._field_coords = be.array(fields)

        # Pre-build list of frequencies to calculate at once
        freqs_to_calc = []
        for freq in self.frequencies:
            freqs_to_calc.append((freq, 0.0))
            freqs_to_calc.append((0.0, freq))

        results = []
        for wp in self.wavelengths:
            wl = wp.value
            wl_results = [{"tangential": [], "sagittal": []} for _ in self.frequencies]

            for field in fields:
                sampled_mtf = SampledMTF(
                    optic=self.optic,
                    field=field,
                    wavelength=wl,
                    num_rays=self.num_rays,
                    distribution="uniform",
                    zernike_terms=37,
                    zernike_type="fringe",
                )

                mtfs = sampled_mtf.calculate_mtf(freqs_to_calc)

                for i_freq in range(len(self.frequencies)):
                    wl_results[i_freq]["tangential"].append(mtfs[2 * i_freq])
                    wl_results[i_freq]["sagittal"].append(mtfs[2 * i_freq + 1])

            for i_freq in range(len(self.frequencies)):
                wl_results[i_freq]["tangential"] = be.array(
                    wl_results[i_freq]["tangential"]
                )
                wl_results[i_freq]["sagittal"] = be.array(
                    wl_results[i_freq]["sagittal"]
                )

            results.append(wl_results)

        return results

    def view(
        self,
        fig_to_plot_on: Figure | None = None,
        figsize: tuple[float, float] = (8, 5),
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Plots the MTF versus the field coordinate for each frequency and wavelength.

        Args:
            fig_to_plot_on (Figure, optional): An existing matplotlib Figure to
                plot on. If provided, the plot will be embedded in this figure.
                If None (default), a new figure will be created.
            figsize (tuple[float, float], optional): Size of the figure to create
                if `fig_to_plot_on` is None. Defaults to (8, 5).
            show (bool): If True (default), calls plt.show(). Set False for
                headless use.

        Returns:
            tuple[Figure, Axes]: The matplotlib Figure and Axes objects
                containing the plot.
        """
        is_gui_embedding = fig_to_plot_on is not None

        if is_gui_embedding:
            current_fig = fig_to_plot_on
            current_fig.clear()
            ax = current_fig.add_subplot(111)
        else:
            current_fig, ax = plt.subplots(figsize=figsize)

        max_field = float(self.optic.fields.max_field)
        y_coords_normalized = be.to_numpy(self._field_coords[:, 1])
        x_plot = y_coords_normalized * max_field

        # Determine X-axis label
        field_def = self.optic.fields.field_definition
        x_label = "Field Coordinate"
        if field_def is not None:
            field_name = field_def.__class__.__name__
            if "Angle" in field_name:
                x_label = "Angle (deg)"
            elif "Height" in field_name:
                x_label = "Height (mm)"
        else:
            # Fallback if no specific type is set but fields exist
            x_label = "Field Coordinate"

        axes_color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for i_wl, wp in enumerate(self.wavelengths):
            for i_freq, freq in enumerate(self.frequencies):
                color_idx = (i_wl * len(self.frequencies) + i_freq) % len(
                    axes_color_cycle
                )
                color = axes_color_cycle[color_idx]

                tan_data = be.to_numpy(self.data[i_wl][i_freq]["tangential"])
                sag_data = be.to_numpy(self.data[i_wl][i_freq]["sagittal"])

                label_prefix = f"{freq} cyc/mm"
                if len(self.wavelengths) > 1:
                    label_prefix += f", {wp.value:.4f} µm"

                ax.plot(
                    x_plot,
                    tan_data,
                    linestyle="-",
                    color=color,
                    label=f"{label_prefix} (Tan)",
                )
                ax.plot(
                    x_plot,
                    sag_data,
                    linestyle="--",
                    color=color,
                    label=f"{label_prefix} (Sag)",
                )

        ax.set_xlabel(x_label)
        ax.set_ylabel("Modulus of the OTF")
        ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")

        if max_field > 0:
            ax.set_xlim(0, max_field)

        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle=":", alpha=0.5)
        current_fig.tight_layout()

        if is_gui_embedding and hasattr(current_fig, "canvas"):
            current_fig.canvas.draw_idle()
        if show and not is_gui_embedding:
            plt.show()
        return current_fig, ax
