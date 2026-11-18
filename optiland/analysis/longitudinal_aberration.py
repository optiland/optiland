"""Longitudinal Aberration Analysis

This module provides a longitudinal aberration analysis for optical systems.

Preetam Maske, 2026
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


class LongitudinalAberration(BaseAnalysis):
    """Longitudinal aberration: axial focus position of each pupil zone.

    On axis (``Hx = Hy = 0``), a real ray is traced through each pupil zone
    ``Py`` in ``linspace(1 / num_points, 1, num_points)`` (``Px = 0``) at every
    wavelength. ``Py = 0`` is skipped because the chief ray never crosses the
    axis. In the image surface's local frame, the ray's axis crossing measured
    from the image surface vertex is ``Δz = z - y * N / M``, in lens units (mm),
    along the local z axis. On a flat image surface ``z = 0`` and this is
    ``-y * N / M``; keeping ``z`` makes the result independent of the image
    surface's sag. When the final rays travel toward +z, as in a refractive
    system, a positive value means that zone focuses past the image surface.

    The analysis assumes a rotationally symmetric system, so the meridional
    (``Py``) zones describe the whole pupil.

    Raises:
        ValueError: If ``num_points`` is less than 1.

    Args:
        optic (Optic): The optic object to analyze.
        wavelengths (str | list, optional): The wavelengths to analyze.
            Defaults to 'all'.
        num_points (int, optional): The number of pupil zones. Defaults to 64.

    Attributes:
        optic (Optic): The optic object being analyzed.
        wavelengths (list): The wavelengths being analyzed.
        num_points (int): The number of pupil zones.
        data (dict): ``"Py"``, the normalized pupil zones, and ``"delta_z"``,
            one array per wavelength (same order as ``wavelengths``) holding the
            axis crossing in mm. Rays that do not reach the image surface, or
            that leave parallel to the axis, give NaN.
    """

    def __init__(
        self, optic: Optic, wavelengths: str | list = "all", num_points: int = 64
    ):
        if num_points < 1:
            raise ValueError(f"num_points must be at least 1, got {num_points}.")
        self.num_points = num_points
        super().__init__(optic, wavelengths)

    def view(
        self,
        fig_to_plot_on: Figure | None = None,
        figsize: tuple[float, float] = (6, 6),
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """Displays a plot of the longitudinal aberration versus pupil zone.

        Args:
            fig_to_plot_on (plt.Figure, optional): The figure to plot on.
                If None, a new figure will be created. Defaults to None.
            figsize (tuple[float, float], optional): The size of the figure.
                Defaults to (6, 6).
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

        pupil_np = be.to_numpy(self.data["Py"])

        for k, wp in enumerate(self.wavelengths):
            ax.plot(
                be.to_numpy(self.data["delta_z"][k]),
                pupil_np,
                f"C{k}",
                zorder=10,
                label=f"{wp.value:.4f} µm",
            )

        ax.set_xlabel("Longitudinal Aberration (mm)")
        ax.set_ylabel("Normalized Pupil Zone")
        ax.set_ylim([0, 1])
        current_xlim = ax.get_xlim()
        max_abs_lim = max(abs(limit) for limit in current_xlim)
        if max_abs_lim > 1e-9:
            ax.set_xlim([-max_abs_lim, max_abs_lim])
        ax.set_title("Longitudinal Aberration")
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
        """Traces one on-axis ray per pupil zone at each wavelength.

        Returns:
            dict: ``"Py"`` (normalized pupil zones) and ``"delta_z"`` (one
                array per wavelength, in mm).
        """
        Py = be.linspace(1 / self.num_points, 1, self.num_points)
        delta_z = [self._axis_crossing(Py, wp.value) for wp in self.wavelengths]
        return {"Py": Py, "delta_z": delta_z}

    def _axis_crossing(self, Py, wavelength: float):
        """Axial distance from the image surface to where each ray meets the axis.

        Args:
            Py (be.ndarray): Normalized pupil y coordinates.
            wavelength (float): Wavelength in µm.

        Returns:
            be.ndarray: ``z - y * N / M`` per zone, in the image surface's
            local frame; NaN where the ray is vignetted or parallel to the axis.
        """
        # The returned rays are a fresh object, so localizing them leaves the
        # recorded per-surface data in the global frame.
        rays = self.optic.trace_generic(0.0, 0.0, 0.0, Py, wavelength)
        self.optic.image_surface.geometry.localize(rays)

        # Rays leaving parallel to the axis (M = 0) never cross it. Dividing by
        # a placeholder 1 avoids an inf/NaN warning; the mask discards them.
        crosses = rays.M != 0
        safe_M = be.where(crosses, rays.M, 1.0)
        delta_z = rays.z - rays.y * rays.N / safe_M
        return be.where((rays.i > 0) & crosses, delta_z, be.nan)
