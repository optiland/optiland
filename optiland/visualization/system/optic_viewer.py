"""Optical System Visualization Module

This module provides tools for visualizing optical systems.
It utilizes Matplotlib to render optical components and ray tracing paths.
The `OpticViewer` class is the primary interface for generating these visualizations,
offering customization for ray properties, field of view, and display parameters.

Kramer Harrison, 2024

re-worked by Manuel Fragata Mendes, june 2025
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import optiland.backend as be
from optiland.visualization.base import BaseViewer
from optiland.visualization.system.interaction import InteractionManager
from optiland.visualization.system.rays import Rays2D
from optiland.visualization.system.system import OpticalSystem
from optiland.visualization.themes import get_active_theme


class OpticViewer(BaseViewer):
    """A class used to visualize optical systems.

    Args:
        optic: The optical system to be visualized.

    Attributes:
        optic: The optical system to be visualized.
        rays: An instance of Rays2D for ray tracing.
        system: An instance of OpticalSystem for system representation.

    Methods:
        view(fields='all', wavelengths='primary', num_rays=3,
             distribution='line_y', figsize=(10, 4), xlim=None, ylim=None):
            Visualizes the optical system with specified parameters.

    """

    def __init__(self, optic):
        self.optic = optic

        self.rays = Rays2D(optic)
        self.system = OpticalSystem(optic, self.rays, projection="2d")
        self.legend_artist_map = {}

    def view(
        self,
        fields="all",
        wavelengths="primary",
        num_rays=3,
        distribution=None,
        show_apertures=True,
        hide_vignetted=False,
        figsize=None,
        xlim=None,
        ylim=None,
        title=None,
        reference=None,
        tooltip_format=None,
        show_legend=True,
        projection="YZ",
        ax: BaseViewer | None = None,
    ):
        """Visualizes the optical system.

        Args:
            fields (str, optional): The fields to be visualized.
                Defaults to 'all'.
            wavelengths (str, optional): The wavelengths to be visualized.
                Defaults to 'primary'.
            num_rays (int, optional): The number of rays to be visualized.
                Defaults to 3.
            distribution (str | None, optional): The distribution of rays.
                Defaults to None, which selects a default based on projection.
            show_apertures (bool, optional): If True, overlays aperture graphics
                on the system view. Defaults to True.
            hide_vignetted (bool, optional): If True, rays that vignette at any
                surface are not shown. Defaults to False.
            figsize (tuple, optional): The size of the figure.
                Defaults to None, which uses the theme's default.
            xlim (tuple, optional): The x-axis limits. Defaults to None.
            ylim (tuple, optional): The y-axis limits. Defaults to None.
            reference (str, optional): The reference rays to plot. Options
                include "chief" and "marginal". Defaults to None.
            projection (str, optional): The projection plane. Must be 'XY',
                'XZ', or 'YZ'. Defaults to 'YZ'.
            ax (matplotlib.axes.Axes, optional): The axes to plot on.
                If None, a new figure and axes are created. Defaults to None.

        """
        if projection not in ["XY", "XZ", "YZ"]:
            raise ValueError("Invalid projection type. Must be 'XY', 'XZ', or 'YZ'.")

        if distribution is None:
            if projection == "XY":
                distribution = "hexapolar"
            elif projection == "XZ":
                distribution = "line_x"
            else:
                distribution = "line_y"

        theme = get_active_theme()
        params = theme.parameters
        if figsize is None:
            figsize = params["figure.figsize"]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            fig.set_facecolor(params["figure.facecolor"])
        else:
            fig = ax.get_figure()

        ax.set_facecolor(params["axes.facecolor"])

        interaction_manager = InteractionManager(fig, ax, self.optic, tooltip_format)

        ray_artists = self.rays.plot(
            ax,
            fields=fields,
            wavelengths=wavelengths,
            num_rays=num_rays,
            distribution=distribution,
            reference=reference,
            theme=theme,
            projection=projection,
            hide_vignetted=hide_vignetted,
        )
        for artist, ray_bundle in ray_artists.items():
            interaction_manager.register_artist(artist, ray_bundle)

        system_artists = self.system.plot(
            ax, theme=theme, projection=projection, show_apertures=show_apertures
        )
        for artist, surface in system_artists.items():
            interaction_manager.register_artist(artist, surface)

        ax.axis("image")
        if projection == "YZ":
            ax.set_xlabel("Z [mm]", color=params["axes.labelcolor"])
            ax.set_ylabel("Y [mm]", color=params["axes.labelcolor"])
        elif projection == "XZ":
            ax.set_xlabel("Z [mm]", color=params["axes.labelcolor"])
            ax.set_ylabel("X [mm]", color=params["axes.labelcolor"])
        else:  # XY
            ax.set_xlabel("X [mm]", color=params["axes.labelcolor"])
            ax.set_ylabel("Y [mm]", color=params["axes.labelcolor"])
        ax.tick_params(axis="x", colors=params["xtick.color"])
        ax.tick_params(axis="y", colors=params["ytick.color"])
        ax.spines["bottom"].set_color(params["axes.edgecolor"])
        ax.spines["top"].set_color(params["axes.edgecolor"])
        ax.spines["right"].set_color(params["axes.edgecolor"])
        ax.spines["left"].set_color(params["axes.edgecolor"])

        if title:
            ax.set_title(title, color=params["text.color"])

        if xlim is None or ylim is None:
            auto_xlim, auto_ylim = self._default_axis_limits(projection)
            xlim = xlim or auto_xlim
            ylim = ylim or auto_ylim
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        ax.grid(
            visible=True,
            color=params["grid.color"],
            alpha=params["grid.alpha"],
        )

        # Return the figure, axes and interaction_manager
        return fig, ax, interaction_manager

    def _default_axis_limits(self, projection):
        """Compute default axis limits sized to the lens system rather than
        the full ray extent.

        For an infinite-conjugate (angle field) system, the object-side ray
        segment is drawn from an arbitrary, often very distant, launch
        point -- fine for aiming, but if left to matplotlib's autoscale it
        dominates the view for wide-FOV systems, squeezing the actual lens
        system down to an unreadable sliver. Sizing instead from the real
        surfaces (z) and ``r_extent`` -- the same radially-symmetric,
        per-surface ray extent ``OpticalSystem`` already uses to size lens
        and mirror components (see ``system.py``) -- keeps the transverse
        limits centered on the optical axis and guaranteed to cover
        whatever radius the lens components are actually drawn at (an
        asymmetric min/max of the traced rays' signed coordinates isn't
        enough: a field that's only ever traced on one side of the axis
        would leave the *other*, still-drawn side of the lens clipped),
        while still leaving margin to see rays approaching and entering
        the first surface.

        Args:
            projection (str): The projection plane, 'XY', 'XZ', or 'YZ'.

        Returns:
            tuple: ``(xlim, ylim)``, each either a ``(min, max)`` tuple or
            ``None`` if there isn't enough information to size that axis
            (falls back to matplotlib's own autoscale).
        """
        if projection == "XY":
            # A single cross-section at one z -- no object-segment issue.
            return None, None

        positions = be.to_numpy(self.optic.surfaces.positions).reshape(-1)
        start_idx = 1 if self.optic.object_surface.is_infinite else 0
        if start_idx >= len(positions) - 1:
            return None, None

        z_min = float(positions[start_idx])
        z_max = float(positions[-1])
        z_margin = max(0.15 * (z_max - z_min), 1e-6)
        auto_xlim = (z_min - z_margin, z_max + z_margin)

        r_extent = be.to_numpy(self.rays.r_extent)[start_idx:]
        r_extent = r_extent[np.isfinite(r_extent)]
        if r_extent.size == 0:
            return auto_xlim, None

        r_max = float(r_extent.max())
        if r_max <= 0:
            return auto_xlim, None
        r_margin = max(0.15 * r_max, 1e-6)
        auto_ylim = (-(r_max + r_margin), r_max + r_margin)

        return auto_xlim, auto_ylim
