"""
Y Y-bar Analysis

This module provides a y y-bar analysis for optical systems.
This is a plot of the marginal ray height versus the chief ray height
for each surface in the system.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

import optiland.backend as be

from .base import BaseAnalysis

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class YYbar(BaseAnalysis):
    """Performs and visualizes a Y Y-bar analysis of an optical system.

    This analysis plots marginal ray height versus chief ray height for
    each surface in the system.

    Args:
        optic (Optic): The optic object to analyze.
        wavelength (str | float | int, optional): Specific wavelength in µm or
            the string "primary" to use the optic's primary wavelength.
            Defaults to "primary". Primarily used for display in the plot title.

    Methods:
        view(fig_to_plot_on=None, figsize=(7, 5.5)):
            Generates and displays the Y Y-bar diagram.
    """

    def __init__(self, optic, wavelength: str | float = "primary") -> None:
        self.wavelength_value_for_display = self._resolve_wavelength(optic, wavelength)
        super().__init__(optic, wavelengths=[self.wavelength_value_for_display])

    @staticmethod
    def _resolve_wavelength(optic, wavelength: str | float) -> float:
        """Resolve the wavelength value to use for display."""
        if isinstance(wavelength, str) and wavelength.lower() == "primary":
            return optic.primary_wavelength
        if isinstance(wavelength, float | int):
            return float(wavelength)
        return optic.primary_wavelength

    def _generate_data(self) -> dict[str, list[float]] | None:
        """Generate marginal and chief ray heights for the analysis.

        Returns:
            dict: Dictionary containing "ya" (marginal ray heights) and
                "yb" (chief ray heights), or None if generation fails.
        """
        try:
            ya, _ = self.optic.paraxial.marginal_ray()
            yb, _ = self.optic.paraxial.chief_ray()
            return {"ya": ya.flatten(), "yb": yb.flatten()}
        except Exception as err:
            print(f"Error generating YYbar data: {err}")
            return None

    def view(
        self,
        fig_to_plot_on: Figure | None = None,
        figsize: tuple[float, float] = (7, 5.5),
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """Visualize the Y Y-bar diagram.

        Args:
            fig_to_plot_on (plt.Figure, optional): Existing figure to plot on.
                Creates a new figure if None.
            figsize (tuple, optional): Figure size if creating a new figure.
            show (bool): If True (default), calls plt.show(). Set False for
                headless use.

        Returns:
            tuple: Matplotlib Figure and Axes objects.
        """
        fig, ax, is_embedding = self._prepare_figure(fig_to_plot_on, figsize)

        if not self._has_valid_data():
            self._plot_error(ax, fig, is_embedding)
            return fig, ax

        self._plot_diagram(ax)
        self._finalize_plot(fig, ax, is_embedding)
        if show and not is_embedding:
            plt.show()
        return fig, ax

    def _has_valid_data(self) -> bool:
        """Check if required YYbar data exists."""
        return bool(self.data and "ya" in self.data and "yb" in self.data)

    @staticmethod
    def _prepare_figure(
        fig_to_plot_on: Figure | None, figsize: tuple[float, float]
    ) -> tuple[Figure, Axes, bool]:
        """Prepare a Matplotlib figure and axis."""
        is_embedding = fig_to_plot_on is not None
        if is_embedding:
            fig = fig_to_plot_on
            fig.clear()
            ax = fig.add_subplot(111)
        else:
            fig, ax = plt.subplots(figsize=figsize)
        return fig, ax, is_embedding

    @staticmethod
    def _plot_error(ax: Axes, fig: Figure, is_embedding: bool) -> None:
        """Plot error message when no data is available."""
        ax.text(
            0.5,
            0.5,
            "Error: YY-bar data could not be generated.",
            ha="center",
            va="center",
            color="red",
        )
        if is_embedding and hasattr(fig, "canvas"):
            fig.canvas.draw_idle()

    def _plot_diagram(self, ax: Axes) -> None:
        """Plot the main Y Y-bar diagram."""
        ya = self.data["ya"]
        yb = self.data["yb"]
        num_surfaces = self.optic.surfaces.num_surfaces

        for idx in range(1, num_surfaces):
            label = self._generate_surface_label(idx, num_surfaces)
            ax.plot(
                [be.to_numpy(yb[idx - 1]), be.to_numpy(yb[idx])],
                [be.to_numpy(ya[idx - 1]), be.to_numpy(ya[idx])],
                ".-",
                label=label,
                markersize=8,
            )

    def _generate_surface_label(self, idx: int, num_surfaces: int) -> str | None:
        """Generate label for a surface in the diagram."""
        sg = self.optic.surfaces

        if idx == num_surfaces - 1:
            return "Image"
        if idx == 1 or idx == sg.stop_index:
            return self._surface_label(sg.surfaces[idx], idx)
        return None

    def _surface_label(self, surface, idx: int) -> str:
        """Return a human-readable label for a surface."""
        label = surface.comment or (
            f"S{surface.id}" if hasattr(surface, "id") else f"S{idx}"
        )
        if idx == self.optic.surfaces.stop_index:
            label += " (Stop)"
        return label

    def _finalize_plot(self, fig: Figure, ax: Axes, is_embedding: bool) -> None:
        """Apply final touches to the plot."""
        ax.axhline(y=0, linewidth=0.5, color="k")
        ax.axvline(x=0, linewidth=0.5, color="k")
        ax.set_xlabel("Chief Ray Height (mm)")
        ax.set_ylabel("Marginal Ray Height (mm)")
        ax.set_title(f"Y Y-bar Diagram (λ={self.wavelength_value_for_display:.3f} µm)")
        ax.legend()
        fig.tight_layout()

        if is_embedding and hasattr(fig, "canvas"):
            fig.canvas.draw_idle()
