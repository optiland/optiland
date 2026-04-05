"""Optical System Visualization Module

This module provides tools for visualizing optical systems.
It utilizes Matplotlib to render optical components and ray tracing paths.
The `OpticViewer` class is the primary interface for generating these visualizations,
offering customization for ray properties, field of view, and display parameters.

Kramer Harrison, 2024

re-worked by Manuel Fragata Mendes, june 2025
"""

from __future__ import annotations

from optiland.visualization.base import BaseViewer2D
from optiland.visualization.system.interaction import InteractionManager
from optiland.visualization.system.rays import Rays2D
from optiland.visualization.system.system import OpticalSystem


class OpticViewer(BaseViewer2D):
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
        ax=None,
        theme=None,
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

        theme = self._resolve_theme(theme)
        fig, ax = self._make_figure(theme, figsize, ax)

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
        self._apply_axes_style(ax, projection, theme, title=title, xlim=xlim, ylim=ylim)

        # Return the figure, axes and interaction_manager
        return fig, ax, interaction_manager
