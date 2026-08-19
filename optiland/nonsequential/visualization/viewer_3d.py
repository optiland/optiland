"""NSQViewer3D -- 3D VTK visualization for NSQ scenes.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.visualization.base import BaseViewer3D

if TYPE_CHECKING:
    from optiland.nonsequential.scene import NSQScene
    from optiland.nonsequential.tracer import SimulationResult


class NSQViewer3D(BaseViewer3D):
    """3D VTK viewer for non-sequential scenes.

    Renders compound components and detectors as VTK actors.  Optionally
    overlays ray-path polylines when a SimulationResult with a
    RayDatabase is provided.

    The renderer registry maps compound-component types to
    :class:`~ComponentRenderer3D` instances.  Default renderers are
    registered for Lens, Mirror, and detectors.

    Attributes:
        scene: The NSQScene to visualize.
        _renderer_registry: Mapping from component class to renderer.
    """

    def __init__(self, scene: NSQScene) -> None:
        """Initialize NSQViewer3D.

        Args:
            scene: The scene to visualize.
        """
        super().__init__(scene)
        self.scene = scene
        self._renderer_registry: dict = {}
        self._register_default_renderers()

    def _register_default_renderers(self) -> None:
        """Register the built-in 3D renderers."""
        from optiland.nonsequential.components.doublet import Doublet  # noqa: PLC0415
        from optiland.nonsequential.components.lens import Lens  # noqa: PLC0415
        from optiland.nonsequential.components.mirror import Mirror  # noqa: PLC0415
        from optiland.nonsequential.visualization.renderers.lens import (  # noqa: PLC0415
            DoubletRenderer3D,
            LensRenderer3D,
        )
        from optiland.nonsequential.visualization.renderers.mirror import (  # noqa: PLC0415
            MirrorRenderer3D,
        )

        self._renderer_registry[Lens] = LensRenderer3D()
        self._renderer_registry[Doublet] = DoubletRenderer3D()
        self._renderer_registry[Mirror] = MirrorRenderer3D()

    def register_renderer(self, component_type: type, renderer) -> None:
        """Register a custom 3D renderer for a compound-component type.

        Args:
            component_type: The class to bind the renderer to.
            renderer: ComponentRenderer3D instance.
        """
        self._renderer_registry[component_type] = renderer

    def view(
        self,
        result: SimulationResult | None = None,
        *,
        num_rays: int = 100,
        dark_mode: bool = False,
        figsize: tuple[int, int] = (1200, 800),
        color_by: str = "source",
    ) -> None:
        """Render the scene in a VTK interactive window.

        When *result* is provided and contains ``ray_paths``, those paths are
        used for the ray overlay without running a new trace.  If *result* is
        ``None`` (or its ``ray_paths`` is ``None``) and ``num_rays > 0``, a
        fresh trace is run internally.

        Args:
            result: Optional SimulationResult.  If its ``ray_paths`` dict is
                populated it is used for the ray overlay directly.
            num_rays: Number of ray segments to overlay. Pass 0 to skip ray
                drawing entirely.
            dark_mode: Use a dark background if True.
            figsize: (width, height) of the VTK window in pixels.
            color_by: Ray colouring strategy -- ``'source'`` (default),
                ``'bounce'``, or ``'segment'``.

        Raises:
            ImportError: If VTK is not installed.
        """
        import importlib.util  # noqa: PLC0415

        if importlib.util.find_spec("vtk") is None:
            raise ImportError(
                "VTK is required for 3D visualization. Install with: pip install vtk"
            )

        renderer = self._make_renderer(dark_mode)

        # Compound components
        for compound in self.scene.component_registry.compounds:
            r = self._renderer_registry.get(type(compound))
            if r is not None:
                r.render(compound, renderer)

        # Detectors
        from optiland.nonsequential.visualization.renderers.detector import (  # noqa: PLC0415
            DetectorRenderer3D,
        )

        det_r = DetectorRenderer3D()
        for det in self.scene.detectors:
            det_r.render(det, renderer)

        if num_rays > 0:
            from optiland.nonsequential.visualization.rays import (
                NSQRays3D,  # noqa: PLC0415
            )

            existing_paths = result.ray_paths if result is not None else None
            rays = NSQRays3D(self.scene)
            rays.plot(
                renderer,
                num_rays=num_rays,
                color_by=color_by,
                ray_paths=existing_paths,
            )

        window, interactor = self._make_window(
            renderer, figsize, "NSQ Scene -- 3D View"
        )

        window.Render()
        interactor.Start()
