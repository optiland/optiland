"""NSQViewer2D — 2D matplotlib visualization for NSQ scenes.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.visualization.base import BaseViewer2D

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from optiland.nonsequential.scene import NSQScene
    from optiland.nonsequential.tracer import SimulationResult


class NSQViewer2D(BaseViewer2D):
    """2D cross-section viewer for non-sequential scenes.

    Renders compound components, sources (as markers), and detectors onto a
    Matplotlib figure.  Optionally overlays a random sample of ray paths when
    a :class:`~SimulationResult` is provided.

    The renderer registry maps compound-component types to
    :class:`~ComponentRenderer2D` instances.  Default renderers are
    registered for :class:`~Lens`, :class:`~Mirror`, and detectors.

    Attributes:
        scene: The NSQScene to visualize.
        _renderer_registry: Mapping from component class to renderer.
    """

    def __init__(self, scene: NSQScene) -> None:
        """Initialize NSQViewer2D.

        Args:
            scene: The scene to visualize.
        """
        super().__init__(scene)
        self.scene = scene
        self._renderer_registry: dict = {}
        self._register_default_renderers()

    # ------------------------------------------------------------------
    # Renderer registration
    # ------------------------------------------------------------------

    def _register_default_renderers(self) -> None:
        """Register the built-in renderers for Lens and Mirror."""
        from optiland.nonsequential.components.lens import Lens  # noqa: PLC0415
        from optiland.nonsequential.components.mirror import Mirror  # noqa: PLC0415
        from optiland.nonsequential.visualization.renderers.lens import (  # noqa: PLC0415
            LensRenderer2D,
        )
        from optiland.nonsequential.visualization.renderers.mirror import (  # noqa: PLC0415
            MirrorRenderer2D,
        )

        self._renderer_registry[Lens] = LensRenderer2D()
        self._renderer_registry[Mirror] = MirrorRenderer2D()

    def register_renderer(self, component_type: type, renderer) -> None:
        """Register a custom 2D renderer for a compound-component type.

        Args:
            component_type: The class to bind the renderer to.
            renderer: ComponentRenderer2D instance.
        """
        self._renderer_registry[component_type] = renderer

    # ------------------------------------------------------------------
    # View
    # ------------------------------------------------------------------

    def view(
        self,
        result: SimulationResult | None = None,
        *,
        theme=None,
        projection: str = "YZ",
        n_rays_display: int = 100,
        figsize: tuple[int, int] | None = None,
        title: str | None = None,
        xlim: tuple | None = None,
        ylim: tuple | None = None,
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """Render the scene cross-section to a Matplotlib figure.

        Args:
            result: Optional SimulationResult; used to overlay ray paths
                when ``n_rays_display > 0``.
            theme: Optional theme object for colours and styles. If None,
                the active theme is used.
            projection: Projection plane — ``'YZ'`` (default), ``'XZ'``,
                or ``'XY'``.
            n_rays_display: Number of ray segments to sample and overlay.
                Defaults to 100.
            figsize: Figure size override (width, height). None → theme default.
            title: Optional axes title. Defaults to "NSQ Scene — 2D Cross-Section".
            xlim: Optional (xmin, xmax) axis limits.
            ylim: Optional (ymin, ymax) axis limits.
            ax: Optional existing axes to plot into.

        Returns:
            (fig, ax) tuple.
        """
        theme = self._resolve_theme(theme)
        fig, ax = self._make_figure(theme, figsize, ax)

        # Render each compound component
        for compound in self.scene.component_registry.compounds:
            renderer = self._renderer_registry.get(type(compound))
            if renderer is not None:
                renderer.render(compound, ax, theme=theme, projection=projection)

        # Render detectors
        from optiland.nonsequential.visualization.renderers.detector import (  # noqa: PLC0415
            DetectorRenderer2D,
        )

        det_renderer = DetectorRenderer2D()
        for det in self.scene.detectors:
            det_renderer.render(det, ax, theme=theme, projection=projection)

        if n_rays_display > 0:
            from optiland.nonsequential.visualization.rays import (
                NSQRays2D,  # noqa: PLC0415
            )

            rays = NSQRays2D(self.scene)
            rays.plot(ax, n_rays=n_rays_display, theme=theme, projection=projection)

        ax.set_aspect("equal")
        _title = title if title is not None else "NSQ Scene — 2D Cross-Section"
        self._apply_axes_style(
            ax, projection, theme, title=_title, xlim=xlim, ylim=ylim
        )

        import matplotlib.pyplot as plt  # noqa: PLC0415

        plt.tight_layout()
        return fig, ax
