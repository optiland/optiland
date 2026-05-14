"""NSQScene -- single user-facing entry point for Non-Sequential Raytracing.

NSQScene owns three typed registries (components, sources, detectors) and
exposes builder methods, a .trace() shortcut, and .view()/.view3d() for
visualization.

Backward-compatible shims are provided so that code using the old flat-list
API (add_component, add_source(src_obj), add_detector(det_obj)) continues to
work without modification.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.nonsequential.components.registry import ComponentRegistry
from optiland.nonsequential.detectors.registry import DetectorRegistry
from optiland.nonsequential.sources.registry import SourceRegistry

if TYPE_CHECKING:
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.backends.base import TracerBackend
    from optiland.nonsequential.components.base import BaseComponent
    from optiland.nonsequential.components.configs import (
        DoubletConfig,
        LensConfig,
        MirrorConfig,
    )
    from optiland.nonsequential.detectors.base import BaseDetector
    from optiland.nonsequential.sources.base import BaseNSQSource
    from optiland.nonsequential.tracer import SimulationResult


class NSQScene:
    """Single user-facing entry point for Non-Sequential Raytracing.

    NSQScene owns three typed registries and exposes builder methods for
    common optical elements (lenses, mirrors, doublets), sources, and
    detectors.  The tracer works on flat surface/source/detector lists
    exposed via read-only properties.

    Usage (new API)::

        scene = NSQScene()
        scene.add_source('S1', cs, PointSourceConfig(...))
        scene.add_lens('L1', cs, LensConfig(...))
        scene.add_detector('D1', cs, IrradianceDetectorConfig(...))
        result = scene.trace(num_rays=1_000_000, seed=42)

    The old flat-list API is still supported for backward compatibility::

        scene.add_component(raw_component)
        scene.add_source(source_obj)      # auto-named
        scene.add_detector(detector_obj)  # auto-named

    Mutation note: Components are held by reference. Modifying a component's
    CoordinateSystem or geometry after scene construction is valid and takes
    effect on the next trace() call. If a BVH acceleration structure is later
    introduced, call scene.invalidate_cache() after any structural change.

    Attributes:
        component_registry: Registry of named compound components.
        source_registry: Registry of named sources.
        detector_registry: Registry of named detectors.
    """

    def __init__(self) -> None:
        """Initialize an empty NSQScene."""
        self.component_registry = ComponentRegistry()
        self.source_registry = SourceRegistry()
        self.detector_registry = DetectorRegistry()
        # Counters for auto-generated names (backward compat)
        self._auto_comp_count = 0
        self._auto_src_count = 0
        self._auto_det_count = 0

    @property
    def surfaces(self) -> list[BaseComponent]:
        """Flat list of all component sub-surfaces in registration order."""
        return self.component_registry.surfaces

    @property
    def sources(self) -> list[BaseNSQSource]:
        """Ordered list of all registered sources."""
        return self.source_registry.sources

    @property
    def detectors(self) -> list[BaseDetector]:
        """Ordered list of all registered detectors."""
        return self.detector_registry.detectors

    def add_lens(
        self,
        name: str,
        cs: CoordinateSystem,
        config: LensConfig,
    ) -> None:
        """Add a single refractive lens to the scene.

        Args:
            name: Unique name for the lens in the registry.
            cs: Front-vertex coordinate system.
            config: LensConfig describing the lens geometry.
        """
        from optiland.nonsequential.components.lens import Lens  # noqa: PLC0415

        self.component_registry.add(name, Lens(name, cs, config))

    def add_mirror(
        self,
        name: str,
        cs: CoordinateSystem,
        config: MirrorConfig,
    ) -> None:
        """Add a reflective mirror to the scene.

        Args:
            name: Unique name for the mirror in the registry.
            cs: Surface coordinate system.
            config: MirrorConfig describing the mirror geometry.
        """
        from optiland.nonsequential.components.mirror import Mirror  # noqa: PLC0415

        self.component_registry.add(name, Mirror(name, cs, config))

    def add_doublet(
        self,
        name: str,
        cs: CoordinateSystem,
        config: DoubletConfig,
    ) -> None:
        """Add a cemented achromatic doublet to the scene.

        Args:
            name: Unique name for the doublet in the registry.
            cs: Front-vertex coordinate system.
            config: DoubletConfig describing the doublet geometry.
        """
        from optiland.nonsequential.components.doublet import Doublet  # noqa: PLC0415

        self.component_registry.add(name, Doublet(name, cs, config))

    def add_raw_component(self, name: str, component: BaseComponent) -> None:
        """Add a raw BaseComponent (advanced use).

        Wraps the component in a thin single-surface CompoundComponent so it
        can be stored in the ComponentRegistry.

        Args:
            name: Unique name.
            component: Pre-built BaseComponent to register.
        """
        from optiland.nonsequential.components.compound import (  # noqa: PLC0415
            CompoundComponent,
        )

        class _SingleSurface(CompoundComponent):
            def __init__(self_, n: str, c: BaseComponent) -> None:
                self_._name = n
                self_._component = c

            @property
            def name(self_) -> str:
                return self_._name

            @property
            def surfaces(self_) -> list[BaseComponent]:
                return [self_._component]

            @property
            def coordinate_system(self_):
                return self_._component.cs

        self.component_registry.add(name, _SingleSurface(name, component))

    def add_source(
        self,
        name_or_source: str | BaseNSQSource,
        cs: CoordinateSystem | None = None,
        config=None,
    ) -> None:
        """Add a source to the scene.

        Supports two call signatures:

        **New API** (preferred)::

            scene.add_source('S1', cs, PointSourceConfig(...))

        **Legacy API** (backward compat)::

            scene.add_source(source_object)   # auto-named 'source_N'

        Args:
            name_or_source: Either a unique string name (new API) or a
                pre-built :class:`~BaseNSQSource` object (legacy API).
            cs: Coordinate system (new API only).
            config: Source config dataclass (new API only).
        """
        from optiland.nonsequential.sources.base import BaseNSQSource  # noqa: PLC0415

        if isinstance(name_or_source, BaseNSQSource):
            # Legacy path: auto-name from counter
            name = f"source_{self._auto_src_count}"
            self._auto_src_count += 1
            self.source_registry.add(name, name_or_source)
            return

        # New path
        name: str = name_or_source  # type: ignore[assignment]
        source = _build_source(cs, config)
        self.source_registry.add(name, source)

    def add_detector(
        self,
        name_or_detector: str | BaseDetector,
        cs: CoordinateSystem | None = None,
        config=None,
    ) -> None:
        """Add a detector to the scene.

        Supports two call signatures:

        **New API** (preferred)::

            scene.add_detector('D1', cs, IrradianceDetectorConfig(...))

        **Legacy API** (backward compat)::

            scene.add_detector(detector_object)   # auto-named 'detector_N'

        Args:
            name_or_detector: Either a unique string name (new API) or a
                pre-built :class:`~BaseDetector` object (legacy API).
            cs: Coordinate system (new API only).
            config: Detector config dataclass (new API only).
        """
        from optiland.nonsequential.detectors.base import BaseDetector  # noqa: PLC0415

        if isinstance(name_or_detector, BaseDetector):
            name = f"detector_{self._auto_det_count}"
            self._auto_det_count += 1
            self.detector_registry.add(name, name_or_detector)
            return

        name: str = name_or_detector  # type: ignore[assignment]
        detector = _build_detector(cs, config)
        self.detector_registry.add(name, detector)

    def remove_component(self, name: str) -> None:
        """Remove a compound component by name.

        Args:
            name: Registry name of the component to remove.
        """
        self.component_registry.remove(name)

    def remove_source(self, name: str) -> None:
        """Remove a source by name.

        Args:
            name: Registry name of the source to remove.
        """
        self.source_registry.remove(name)

    def remove_detector(self, name: str) -> None:
        """Remove a detector by name.

        Args:
            name: Registry name of the detector to remove.
        """
        self.detector_registry.remove(name)

    def add_component(self, component: BaseComponent) -> None:
        """Add a raw component (backward-compatible shim).

        Equivalent to ``add_raw_component('component_N', component)`` with
        an auto-generated name.

        Args:
            component: Pre-built BaseComponent to add.
        """
        name = f"component_{self._auto_comp_count}"
        self._auto_comp_count += 1
        self.add_raw_component(name, component)

    def trace(
        self,
        num_rays: int,
        max_bounces: int = 200,
        min_flux_fraction: float = 1e-6,
        batch_size: int = 1_000_000,
        seed: int | None = None,
        backend: TracerBackend | None = None,
    ) -> SimulationResult:
        """Run the Monte Carlo simulation and return results.

        This is a one-liner shortcut equivalent to::

            NSQTracer(self, backend=backend).trace(num_rays, ...)

        Args:
            num_rays: Total rays to launch.
            max_bounces: Maximum surface hits per ray.
            min_flux_fraction: Kill threshold relative to per-ray initial flux.
            batch_size: Rays per processing batch.
            seed: RNG seed.
            backend: TracerBackend to use. Defaults to NumpyBackend.

        Returns:
            SimulationResult with per-detector results and statistics.
        """
        from optiland.nonsequential.tracer import NSQTracer  # noqa: PLC0415

        self.validate()
        tracer = NSQTracer(self, backend=backend)
        return tracer.trace(
            num_rays,
            max_bounces=max_bounces,
            min_flux_fraction=min_flux_fraction,
            batch_size=batch_size,
            seed=seed,
        )

    def view(
        self,
        result: SimulationResult | None = None,
        **kwargs,
    ) -> None:
        """Render a 2D (matplotlib) cross-section of the scene.

        Args:
            result: Optional SimulationResult with ray paths to overlay.
            **kwargs: Forwarded to NSQViewer2D.view().
        """
        from optiland.nonsequential.visualization import NSQViewer2D  # noqa: PLC0415

        NSQViewer2D(self).view(result, **kwargs)

    def view3d(
        self,
        result: SimulationResult | None = None,
        **kwargs,
    ) -> None:
        """Render a 3D (VTK) scene visualization.

        Args:
            result: Optional SimulationResult with ray paths to overlay.
            **kwargs: Forwarded to NSQViewer3D.view().
        """
        from optiland.nonsequential.visualization import NSQViewer3D  # noqa: PLC0415

        NSQViewer3D(self).view(result, **kwargs)

    def validate(self) -> None:
        """Validate the scene for common configuration errors.

        Raises:
            ValueError: If no sources or no detectors are registered.
        """
        if not self.source_registry.sources:
            raise ValueError("Scene has no sources. Add at least one source.")
        if not self.detector_registry.detectors:
            raise ValueError("Scene has no detectors. Add at least one detector.")


def _build_source(cs: CoordinateSystem, config) -> BaseNSQSource:
    """Instantiate a BaseNSQSource from a config dataclass.

    Args:
        cs: Coordinate system for the source.
        config: One of PointSourceConfig, CollimatedSourceConfig,
            ExtendedSourceConfig.

    Returns:
        Constructed source object.

    Raises:
        TypeError: If the config type is not recognised.
    """
    from optiland.nonsequential.sources.collimated import (  # noqa: PLC0415
        CollimatedSource,
    )
    from optiland.nonsequential.sources.configs import (  # noqa: PLC0415
        CollimatedSourceConfig,
        ExtendedSourceConfig,
        PointSourceConfig,
    )
    from optiland.nonsequential.sources.extended import ExtendedSource  # noqa: PLC0415
    from optiland.nonsequential.sources.point import PointSource  # noqa: PLC0415

    if isinstance(config, PointSourceConfig):
        return PointSource(
            cs=cs,
            spectrum=config.spectrum,
            total_flux=config.total_flux,
            half_angle_deg=config.half_angle_deg,
            medium=getattr(config, "medium", None),
        )
    if isinstance(config, CollimatedSourceConfig):
        return CollimatedSource(
            cs=cs,
            spectrum=config.spectrum,
            total_flux=config.total_flux,
            aperture_radius=config.aperture_radius,
            medium=getattr(config, "medium", None),
        )
    if isinstance(config, ExtendedSourceConfig):
        return ExtendedSource(
            cs=cs,
            spectrum=config.spectrum,
            total_flux=config.total_flux,
            width=config.width,
            height=config.height,
            half_angle_deg=config.half_angle_deg,
            medium=getattr(config, "medium", None),
        )
    raise TypeError(
        f"Unrecognised source config type: {type(config).__name__}. "
        "Expected PointSourceConfig, CollimatedSourceConfig, or ExtendedSourceConfig."
    )


def _build_detector(cs: CoordinateSystem, config) -> BaseDetector:
    """Instantiate a BaseDetector from a config dataclass.

    Args:
        cs: Coordinate system for the detector.
        config: One of IrradianceDetectorConfig, SpectralDetectorConfig,
            FarFieldDetectorConfig, RayDatabaseConfig.

    Returns:
        Constructed detector object.

    Raises:
        TypeError: If the config type is not recognised.
    """
    from optiland.nonsequential.detectors.configs import (  # noqa: PLC0415
        FarFieldDetectorConfig,
        IrradianceDetectorConfig,
        RayDatabaseConfig,
        SpectralDetectorConfig,
    )
    from optiland.nonsequential.detectors.far_field import (
        FarFieldDetector,  # noqa: PLC0415
    )
    from optiland.nonsequential.detectors.irradiance import (
        IrradianceDetector,  # noqa: PLC0415
    )
    from optiland.nonsequential.detectors.ray_database import (  # noqa: PLC0415
        RayDatabaseDetector,
    )
    from optiland.nonsequential.detectors.spectral import (
        SpectralDetector,  # noqa: PLC0415
    )

    if isinstance(config, IrradianceDetectorConfig):
        return IrradianceDetector(
            cs=cs,
            width=config.width,
            height=config.height,
            num_pixels_x=config.num_pixels_x,
            num_pixels_y=config.num_pixels_y,
        )
    if isinstance(config, SpectralDetectorConfig):
        wl_bins = be.linspace(config.wl_min, config.wl_max, config.num_bins + 1)
        return SpectralDetector(
            cs=cs,
            width=config.width,
            height=config.height,
            num_pixels_x=config.num_pixels_x,
            num_pixels_y=config.num_pixels_y,
            wavelength_bins=wl_bins,
        )
    if isinstance(config, FarFieldDetectorConfig):
        return FarFieldDetector(
            cs=cs,
            theta_max_deg=90.0,  # Default to hemisphere
            num_bins_theta=config.num_theta,
            num_bins_phi=config.num_phi,
        )
    if isinstance(config, RayDatabaseConfig):
        return RayDatabaseDetector(
            cs=cs,
            width=config.width,
            height=config.height,
        )
    raise TypeError(f"Unrecognised detector config type: {type(config).__name__}.")
