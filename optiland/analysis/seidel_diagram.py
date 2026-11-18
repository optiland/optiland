"""Seidel Diagram Analysis

This module provides a Seidel diagram for optical systems: a bar chart of the
Seidel aberration coefficients contributed by each surface, and their sum.

Rishi Raj, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

import optiland.backend as be

from .base import BaseAnalysis

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from optiland.optic import Optic

SEIDEL_LABELS = (
    "SI Spherical",
    "SII Coma",
    "SIII Astigmatism",
    "SIV Petzval",
    "SV Distortion",
)


class SeidelDiagram(BaseAnalysis):
    """Seidel diagram: the Seidel coefficients SI through SV of each surface.

    The coefficients are those of ``optic.aberrations.seidels_per_surface()``,
    computed at the primary wavelength. Their sum over the surfaces is
    ``optic.aberrations.seidels()``.

    Args:
        optic (Optic): The optic object to analyze.

    Attributes:
        optic (Optic): The optic object being analyzed.
        data (dict): The generated data, with keys ``"per_surface"``, an
            array of shape (5, num_surfaces - 2) whose columns are surfaces 1
            through num_surfaces - 2, and ``"sum"``, the five totals.
    """

    def __init__(self, optic: Optic):
        super().__init__(optic, wavelengths="primary")

    def view(
        self,
        fig_to_plot_on: Figure | None = None,
        figsize: tuple[float, float] = (10, 5),
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """Displays a grouped bar chart of the Seidel coefficients.

        Each group holds the five coefficients of one surface. The last group
        is their sum over all surfaces.

        Args:
            fig_to_plot_on (plt.Figure, optional): The figure to plot on.
                If None, a new figure will be created. Defaults to None.
            figsize (tuple[float, float], optional): The size of the figure.
                Defaults to (10, 5).
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

        per_surface = be.to_numpy(self.data["per_surface"])
        totals = be.to_numpy(self.data["sum"])
        values = np.column_stack([per_surface, totals])

        num_groups = values.shape[1]
        positions = np.arange(num_groups)
        width = 0.8 / len(SEIDEL_LABELS)
        for k, label in enumerate(SEIDEL_LABELS):
            offset = (k - (len(SEIDEL_LABELS) - 1) / 2) * width
            ax.bar(
                positions + offset,
                values[k],
                width,
                color=f"C{k}",
                zorder=10,
                label=label,
            )

        surface_labels = [str(i) for i in range(1, num_groups)]
        ax.set_xticks(positions)
        ax.set_xticklabels([*surface_labels, "Sum"])
        ax.set_xlabel("Surface")
        ax.set_ylabel("Seidel Coefficient (mm)")
        ax.set_title("Seidel Diagram")
        ax.axhline(y=0, color="k", linewidth=0.5, zorder=11)
        ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
        ax.grid(True, axis="y")
        current_fig.tight_layout()

        if is_gui_embedding and hasattr(current_fig, "canvas"):
            current_fig.canvas.draw_idle()
        if show and not is_gui_embedding:
            plt.show()
        return current_fig, ax

    def _generate_data(self):
        """Generates the Seidel coefficients of each surface and their sum.

        Returns:
            dict: The per-surface coefficients and their totals.
        """
        per_surface = self.optic.aberrations.seidels_per_surface()
        return {"per_surface": per_surface, "sum": be.sum(per_surface, axis=1)}
