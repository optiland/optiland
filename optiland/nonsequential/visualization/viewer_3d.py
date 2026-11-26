"""NSQViewer3D -- 3D VTK visualization for NSQ scenes.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.visualization.base import BaseViewer3D

if TYPE_CHECKING:
    from pathlib import Path

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

    def _plot_scene(
        self,
        renderer,
        result: SimulationResult | None = None,
        num_rays: int = 100,
        color_by: str = "source",
    ) -> None:
        """Add component, detector and ray actors to *renderer*."""
        for compound in self.scene.component_registry.compounds:
            r = self._renderer_registry.get(type(compound))
            if r is not None:
                r.render(compound, renderer)

        from optiland.nonsequential.visualization.renderers.detector import (  # noqa: PLC0415
            DetectorRenderer3D,
        )

        det_r = DetectorRenderer3D()
        for det in self.scene.detectors:
            det_r.render(det, renderer)

        if num_rays > 0:
            from optiland.nonsequential.visualization.rays import (  # noqa: PLC0415
                NSQRays3D,
            )

            existing_paths = result.ray_paths if result is not None else None
            rays = NSQRays3D(self.scene)
            rays.plot(
                renderer,
                num_rays=num_rays,
                color_by=color_by,
                ray_paths=existing_paths,
            )

    def _configure_camera(self, renderer, camera: str) -> None:
        """Apply *camera*; ``"default"`` keeps the renderer's auto view.

        Before camera presets existed, NSQViewer3D set no camera at all
        and let VTK's automatic first-render ResetCamera frame the scene
        (looking down -z with view-up +y). ``renderer.ResetCamera()``
        reproduces that view, so ``"default"`` must not go through
        :meth:`_apply_camera`, whose default applies the OpticViewer3D
        legacy look instead.

        Args:
            renderer: VTK renderer whose active camera is updated.
            camera: Preset name.

        Raises:
            ValueError: If *camera* is not a known preset.
        """
        if camera == "default":
            renderer.ResetCamera()
        else:
            self._apply_camera(renderer, camera)

    def view(
        self,
        result: SimulationResult | None = None,
        *,
        num_rays: int = 100,
        dark_mode: bool = False,
        figsize: tuple[int, int] = (1200, 800),
        color_by: str = "source",
        camera: str = "default",
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
            camera: Camera preset: ``"side"``, ``"top"``, ``"front"`` or
                ``"iso"``. Defaults to ``"default"``, which keeps the
                renderer's automatic view (no preset is applied).

        Raises:
            ImportError: If VTK is not installed.
        """
        import importlib.util  # noqa: PLC0415

        if importlib.util.find_spec("vtk") is None:
            raise ImportError(
                "VTK is required for 3D visualization. Install with: pip install vtk"
            )

        renderer = self._make_renderer(dark_mode)
        self._plot_scene(renderer, result=result, num_rays=num_rays, color_by=color_by)
        window, interactor = self._make_window(
            renderer, figsize, "NSQ Scene -- 3D View"
        )
        self.ren_win = window
        self.iren = interactor

        window.Render()
        self._configure_camera(renderer, camera)
        window.Render()
        interactor.Start()

    def save_image(
        self,
        path: str | Path,
        *,
        camera: str = "default",
        figsize: tuple[int, int] = (1200, 800),
        dark_mode: bool = False,
        scale: int = 1,
        transparent: bool = False,
        **view_kwargs,
    ) -> Path:
        """Render the scene off-screen and write a PNG.

        Builds the same scene as :meth:`view` but never starts the
        interactor, so it can run headlessly (docs, notebooks, CI).

        Args:
            path: Destination file path.
            camera: Camera preset applied before capture. ``"default"``
                (the default) keeps the renderer's automatic view, as in
                :meth:`view`.
            figsize: Window size in pixels before *scale* is applied.
            dark_mode: Use the dark gradient background if True.
            scale: Integer magnification of the output image.
            transparent: If True, write an RGBA PNG with a transparent
                background.
            **view_kwargs: Forwarded to the shared scene builder
                (``result``, ``num_rays``, ``color_by``).

        Returns:
            The path the image was written to.

        Raises:
            ImportError: If VTK is not installed.
        """
        import importlib.util  # noqa: PLC0415

        if importlib.util.find_spec("vtk") is None:
            raise ImportError(
                "VTK is required for 3D visualization. Install with: pip install vtk"
            )

        renderer = self._make_renderer(dark_mode)
        self._plot_scene(renderer, **view_kwargs)
        window, interactor = self._make_window(
            renderer, figsize, "NSQ Scene -- 3D View"
        )
        self.ren_win = window
        self.iren = interactor
        self._configure_camera(renderer, camera)
        return self._save_png(window, path, scale=scale, transparent=transparent)
