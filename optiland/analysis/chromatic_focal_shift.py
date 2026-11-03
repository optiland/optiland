"""Chromatic Focal Shift Analysis

This module provides a chromatic focal shift analysis for optical systems:
the paraxial focus position as a function of wavelength, relative to the
focus at the primary wavelength.

Javier Ruiz, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

import optiland.backend as be

from .base import BaseAnalysis

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from optiland.optic import Optic


class ChromaticFocalShift(BaseAnalysis):
    """Chromatic focal shift: paraxial focus position versus wavelength.

    For each sampled wavelength, a paraxial marginal ray (Hy = 0, Py = 1) is
    traced. Its axial focus, measured from the image surface, is
    ``dz = -y / u`` at the image surface. The focal shift is ``dz`` minus its
    value at the primary wavelength.

    Args:
        optic (Optic): The optic object to analyze.
        wavelength_range (tuple[float, float], optional): The minimum and
            maximum wavelengths in µm. If None, the minimum and maximum of the
            optic's defined wavelengths are used. Defaults to None.
        num_points (int, optional): The number of wavelengths sampled across
            the range. Defaults to 128.

    Attributes:
        optic (Optic): The optic object being analyzed.
        wavelength_range (tuple[float, float]): The sampled wavelength range
            in µm.
        num_points (int): The number of sampled wavelengths.
        data (dict): The generated data, with keys ``"wavelengths"`` (µm),
            ``"focal_shift"`` (lens units), and the same two quantities at the
            optic's defined wavelengths, ``"defined_wavelengths"`` and
            ``"defined_focal_shift"``.

    Raises:
        ValueError: If the wavelength range is empty.
    """

    def __init__(
        self,
        optic: Optic,
        wavelength_range: tuple[float, float] | None = None,
        num_points: int = 128,
    ):
        if wavelength_range is None:
            values = [w.value for w in optic.wavelengths.wavelengths]
            wavelength_range = (min(values), max(values))
        if not wavelength_range[0] < wavelength_range[1]:
            raise ValueError(
                "The wavelength range is empty: its minimum must be smaller than "
                f"its maximum, got {wavelength_range}. With a single defined "
                "wavelength, pass wavelength_range."
            )
        self.wavelength_range = wavelength_range
        self.num_points = num_points
        super().__init__(optic, wavelengths="primary")

    def max_focal_shift(self) -> float:
        """Returns the range of the focal shift over the sampled wavelengths.

        Returns:
            float: The maximum minus the minimum focal shift, in lens units.
        """
        shift = self.data["focal_shift"]
        return float(be.to_numpy(be.max(shift) - be.min(shift)))

    def view(
        self,
        fig_to_plot_on: Figure | None = None,
        figsize: tuple[float, float] = (8, 5.5),
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """Displays a plot of the chromatic focal shift.

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

        # lens units (mm) to µm
        wavelengths = be.to_numpy(self.data["wavelengths"])
        shift = be.to_numpy(self.data["focal_shift"]) * 1e3
        defined = be.to_numpy(self.data["defined_wavelengths"])
        defined_shift = be.to_numpy(self.data["defined_focal_shift"]) * 1e3

        ax.plot(wavelengths, shift, "C0", zorder=10)
        ax.plot(
            defined,
            defined_shift,
            "o",
            color="C1",
            zorder=11,
            label="Defined wavelengths",
        )
        ax.axhline(y=0, color="k", linewidth=0.5)
        ax.set_xlabel("Wavelength (µm)")
        ax.set_ylabel("Focal Shift (µm)")
        ax.set_title(
            "Chromatic Focal Shift\n"
            f"Maximum focal shift: {self.max_focal_shift() * 1e3:.4f} µm"
        )
        ax.legend()
        ax.grid(True)
        current_fig.tight_layout()

        if is_gui_embedding and hasattr(current_fig, "canvas"):
            current_fig.canvas.draw_idle()
        if show and not is_gui_embedding:
            plt.show()
        return current_fig, ax

    def _generate_data(self):
        """Generates the focal shift on the wavelength grid and at the optic's
        defined wavelengths.

        Returns:
            dict: The wavelengths (µm) and focal shifts (lens units).
        """
        grid = be.linspace(
            self.wavelength_range[0], self.wavelength_range[1], self.num_points
        )
        defined = [w.value for w in self.optic.wavelengths.wavelengths]
        dz_primary = self._paraxial_focus(self.wavelengths[0].value)

        def shift(wavelengths):
            return be.stack([self._paraxial_focus(w) - dz_primary for w in wavelengths])

        return {
            "wavelengths": grid,
            "focal_shift": shift(be.to_numpy(grid).tolist()),
            "defined_wavelengths": be.array(defined),
            "defined_focal_shift": shift(defined),
        }

    def _paraxial_focus(self, wavelength: float):
        """Axial position of the paraxial marginal focus, from the image surface.

        The marginal ray starts on axis in object space, as in
        ``Paraxial.marginal_ray``, which traces the primary wavelength only.
        The focus does not depend on the ray's height or slope, only on its
        starting point.

        Args:
            wavelength (float): The wavelength in µm.

        Returns:
            be.ndarray: ``-y / u`` of the marginal ray at the image surface.
        """
        paraxial = self.optic.paraxial
        path = self.optic.surfaces.build_paraxial_path()
        epd = paraxial.EPD()
        if self.optic.object_surface.is_infinite:
            z0 = path.axial_positions.reshape(-1, 1)[1] - 10
            y0, u0 = epd / 2, 0.0
        else:
            z0 = self.optic.object_surface.geometry.cs.z
            epz = paraxial.entrance_pupil_axial_position(path=path)
            y0, u0 = 0.0, epd / (2 * (epz - z0))
        y, u = paraxial.trace_generic(y0, u0, z0, wavelength, path=path)
        return be.ravel(-y[-1] / u[-1])[0]
