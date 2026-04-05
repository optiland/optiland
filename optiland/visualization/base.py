"""
Base Viewer Module
This module defines the `BaseViewer` abstract base class, which serves as
the foundation for all visualization viewers within the Optiland library.
It establishes a common interface for initializing viewers with an optical
system and for generating visualizations.

Manuel Fragata Mendes, June 2025
"""

from __future__ import annotations

import abc
from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class BaseViewer(abc.ABC):
    """
    Abstract base class for all visualization viewers in Optiland.

    This class defines the standard interface for viewers, ensuring that
    each viewer is initialized with an Optic object and has a `view`
    method to generate the visualization.
    """

    def __init__(self, optic):
        """
        Initializes the BaseViewer.

        Args:
            optic (Optic): The optical system to be visualized.
        """
        self.optic = optic

    @abc.abstractmethod
    def view(self, *args, **kwargs):
        """
        The main method to generate and display the visualization.

        This method must be implemented by all concrete viewer subclasses.
        """
        pass


class BaseViewer2D(BaseViewer, abc.ABC):
    """ABC for 2D (Matplotlib) optical system viewers.

    Provides shared helpers for theme resolution, figure creation, and
    axis styling so that sequential and NSQ 2D viewers share common
    setup, theming, and axis labelling.
    """

    # ------------------------------------------------------------------
    # Shared concrete helpers
    # ------------------------------------------------------------------

    def _resolve_theme(self, theme):
        """Return theme, falling back to the active theme if None.

        Args:
            theme: A Theme object or None.

        Returns:
            A Theme object.
        """
        from optiland.visualization.themes import get_active_theme  # noqa: PLC0415

        return theme if theme is not None else get_active_theme()

    def _make_figure(
        self,
        theme,
        figsize: tuple[int, int] | None = None,
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """Create or reuse a Matplotlib figure and axes.

        If *ax* is provided the existing figure is reused. Otherwise a new
        figure is created using *figsize* (or the theme's default).
        The figure and axes backgrounds are set from the theme.

        Args:
            theme: Active theme.
            figsize: Figure size override. None → theme default.
            ax: Optional existing axes to reuse.

        Returns:
            (fig, ax) tuple.
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415

        params = theme.parameters
        if figsize is None:
            figsize = params.get("figure.figsize", (10, 6))

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            fig.set_facecolor(params.get("figure.facecolor", "white"))
        else:
            fig = ax.get_figure()

        ax.set_facecolor(params.get("axes.facecolor", "white"))
        return fig, ax

    def _apply_axes_style(
        self,
        ax: Axes,
        projection: str,
        theme,
        title: str | None = None,
        xlim: tuple | None = None,
        ylim: tuple | None = None,
    ) -> None:
        """Apply labels, tick colours, grid, spine colours, and optional limits.

        Args:
            ax: Target axes.
            projection: ``'YZ'``, ``'XZ'``, or ``'XY'``.
            theme: Active theme.
            title: Optional axes title.
            xlim: Optional (xmin, xmax) limits.
            ylim: Optional (ymin, ymax) limits.
        """
        params = theme.parameters
        _label_map = {
            "YZ": ("Z [mm]", "Y [mm]"),
            "XZ": ("Z [mm]", "X [mm]"),
            "XY": ("X [mm]", "Y [mm]"),
        }
        xlabel, ylabel = _label_map.get(projection.upper(), ("Z [mm]", "Y [mm]"))

        label_color = params.get("axes.labelcolor", "black")
        ax.set_xlabel(xlabel, color=label_color)
        ax.set_ylabel(ylabel, color=label_color)

        ax.tick_params(axis="x", colors=params.get("xtick.color", "black"))
        ax.tick_params(axis="y", colors=params.get("ytick.color", "black"))

        edge_color = params.get("axes.edgecolor", "black")
        for spine in ax.spines.values():
            spine.set_color(edge_color)

        if title:
            ax.set_title(title, color=params.get("text.color", "black"))

        ax.grid(
            visible=True,
            color=params.get("grid.color", "gray"),
            alpha=params.get("grid.alpha", 0.25),
        )

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def view(
        self,
        *,
        theme=None,
        projection: str = "YZ",
        figsize: tuple[int, int] | None = None,
        title: str | None = None,
        xlim: tuple | None = None,
        ylim: tuple | None = None,
        ax: Axes | None = None,
        **kwargs,
    ) -> tuple[Figure, Axes]:
        """Render the 2D cross-section.

        Returns:
            (fig, ax) at minimum. Subclasses may return a longer tuple.
        """


class BaseViewer3D(BaseViewer, abc.ABC):
    """ABC for 3D (VTK) optical system viewers.

    Provides shared helpers for VTK renderer and window creation so that
    sequential and NSQ 3D viewers share common VTK boilerplate.
    """

    def _make_renderer(self, dark_mode: bool = False):
        """Create a VTK renderer with a gradient background.

        Args:
            dark_mode: Dark theme (dark grey gradient) if True;
                light blue gradient otherwise.

        Returns:
            Configured vtkRenderer.
        """
        import vtk  # type: ignore[import]  # noqa: PLC0415

        renderer = vtk.vtkRenderer()
        renderer.GradientBackgroundOn()
        renderer.SetGradientMode(vtk.vtkViewport.GradientModes.VTK_GRADIENT_VERTICAL)
        if dark_mode:
            renderer.SetBackground(0.13, 0.15, 0.19)
            renderer.SetBackground2(0.195, 0.21, 0.24)
        else:
            renderer.SetBackground(0.8, 0.9, 1.0)
            renderer.SetBackground2(0.4, 0.5, 0.6)
        return renderer

    def _make_window(
        self,
        renderer,
        figsize: tuple[int, int] = (1200, 800),
        title: str = "Optical System - 3D Viewer",
    ) -> tuple:
        """Create and configure VTK render window and interactor.

        Args:
            renderer: The renderer to attach to the window.
            figsize: (width, height) in pixels.
            title: Window title string.

        Returns:
            (render_window, interactor) tuple.
        """
        import vtk  # type: ignore[import]  # noqa: PLC0415

        window = vtk.vtkRenderWindow()
        window.AddRenderer(renderer)
        window.SetSize(*figsize)
        window.SetWindowName(title)

        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(window)
        interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        return window, interactor

    @abstractmethod
    def view(
        self,
        *,
        dark_mode: bool = False,
        figsize: tuple[int, int] = (1200, 800),
        **kwargs,
    ) -> None:
        """Open an interactive 3D VTK window."""
