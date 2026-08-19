"""Base Modulation Transfer Function (FFTMTF) Module.

This module contains the abstract base class for MTF calculations
based on the PSF. This includes, e.g., the FFT-based method
and the Huygen-Fresnel-based method.

Kramer Harrison, 2025
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

import optiland.backend as be
from optiland.utils import get_working_FNO, resolve_fields, resolve_wavelength

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class BaseMTF(abc.ABC):
    """Base class for MTF computations based on a PSF calculation.

    Attributes:
        optic: The optical system.
        fields: Original field point specification (e.g., "all" or list).
        wavelength: Original wavelength specification (e.g., "primary" or value).
        resolved_fields: List of actual field coordinates (Hx, Hy) to be used.
        resolved_wavelength: Actual wavelength value (in µm) to be used.
    """

    def __init__(
        self,
        optic,
        fields: str | list,
        wavelength: str | float,
        strategy="chief_ray",
        remove_tilt=False,
        **kwargs,
    ):
        """Initializes BaseMTF and resolves field/wavelength values.

        Args:
            optic: The optical system.
            fields: The field points for MTF calculation. Can be "all" to
                use all fields from the optic, or a list of field coordinates.
            wavelength: The wavelength for MTF calculation. Can be "primary"
                to use the optic's primary wavelength, or a specific
                wavelength value (typically in µm).
            strategy (str): The calculation strategy to use. Supported options are
                "chief_ray", "centroid_sphere", and "best_fit_sphere".
                Defaults to "chief_ray".
            remove_tilt (bool): If True, removes tilt and piston from the OPD data.
                Defaults to False.
            **kwargs: Additional keyword arguments passed to the strategy.
        """
        self.optic = optic
        self.fields = fields
        self.wavelength = wavelength
        self.strategy = strategy
        self.remove_tilt = remove_tilt
        self.strategy_kwargs = kwargs

        resolved = resolve_fields(optic, fields)
        # extract plain (x, y) coords; BaseMTF uses single wavelength (no weighting)
        self.resolved_fields = [fp.coord for fp in resolved]
        self.resolved_wavelength = resolve_wavelength(optic, wavelength)

        self._calculate_psf()
        self.mtf = self._generate_mtf_data()

    @abc.abstractmethod
    def _generate_mtf_data(self):
        """Generates and returns MTF data."""
        pass

    @abc.abstractmethod
    def _plot_field_mtf(self, ax, field_index, mtf_field_data, color):
        """Plots the MTF data for a single field on the given axes.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axes object.
            field_index (int): The index of the current field.
            mtf_field_data (any): The MTF data for this specific field.
                                Subclasses will define its structure.
            color (str): The color to use for plotting this field.
        """
        pass

    @abc.abstractmethod
    def _calculate_psf(self):
        """Calculates and potentially stores the Point Spread Function."""
        pass

    def view(
        self,
        fig_to_plot_on: Figure | None = None,
        figsize: tuple[float, float] = (12, 4),
        add_reference: bool = False,
    ) -> tuple[Figure, Axes]:
        """Visualizes the Modulation Transfer Function (MTF).

        This method sets up the plot and iterates through field data,
        calling `_plot_field_mtf` for each field's specific plotting.

        Subclasses must ensure `self.mtf`, `self.freq`, and `self.max_freq`
        are populated before calling this method. `self.resolved_fields`
        (from __init__) is also used.

        Args:
            fig_to_plot_on (plt.Figure, optional): The figure to plot on.
                If None, a new figure will be created. Defaults to None.
            figsize (tuple, optional): The size of the figure.
                Defaults to (12, 4).
            add_reference (bool, optional): Whether to overlay the theoretical
                diffraction-limited MTF curve for a clear circular aperture.
                The reference is computed using the *on-axis* working F/# and
                the resolved wavelength. Defaults to False.
        Returns:
            tuple: A tuple containing the figure and axes objects.
        """
        is_gui_embedding = fig_to_plot_on is not None

        if is_gui_embedding:
            current_fig = fig_to_plot_on
            current_fig.clear()
            ax = current_fig.add_subplot(111)
        else:
            current_fig, ax = plt.subplots(figsize=figsize)

        for k, field_mtf_item in enumerate(self.mtf):
            self._plot_field_mtf(ax, k, field_mtf_item, color=f"C{k}")

        if add_reference:
            # The reference curve shows the theoretical diffraction-limited OTF for
            # a clear circular aperture evaluated at the *on-axis* working F/# and
            # the resolved wavelength. For off-axis fields the working F/# may
            # differ; the on-axis reference provides a consistent, field-independent
            # benchmark.
            #
            # Formula (incoherent OTF for a circular aperture):
            #   MTF(u) = (2/π)(arccos(u) − u √(1 − u²))  for u ∈ [0, 1]
            # where u = f / f_c  and  f_c = 1 / (λ · F/#_on-axis).
            on_axis_fno = self._get_fno()
            cutoff_freq = 1 / (self.resolved_wavelength * 1e-3 * on_axis_fno)
            ref_freq = be.linspace(0, cutoff_freq, 256)

            ratio = be.clip(ref_freq / cutoff_freq, 0.0, 1.0)
            phi = be.arccos(ratio)
            diff_limited_mtf = (2 / be.pi) * (phi - be.cos(phi) * be.sin(phi))

            ax.plot(
                be.to_numpy(ref_freq),
                be.to_numpy(diff_limited_mtf),
                "k--",
                label="Diffraction Limit (on-axis)",
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

    def _get_fno(self):
        """Calculates the working F-number of the optical system for the
        single defined field point and given wavelength.

        Returns:
            float: The working F-number.
        """
        return get_working_FNO(
            optic=self.optic,
            field=(0, 0),  # always calculate on-axis F/#
            wavelength=self.resolved_wavelength,
        )
