"""NSQScene -- single user-facing entry point for Non-Sequential Raytracing.

NSQScene owns three typed registries (components, sources, detectors) and
exposes builder methods, a .trace() shortcut, and .view()/.view3d() for
visualization.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.nonsequential._utils import DEFAULT_BATCH_SIZE
from optiland.nonsequential.components.registry import ComponentRegistry
from optiland.nonsequential.detectors.registry import DetectorRegistry
from optiland.nonsequential.sources.registry import SourceRegistry

if TYPE_CHECKING:
    import os

    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.backends.base import TracerBackend
    from optiland.nonsequential.components.base import BaseComponent
    from optiland.nonsequential.components.configs import (
        DoubletConfig,
        LensConfig,
        MirrorConfig,
    )
    from optiland.nonsequential.tracer import SimulationResult


class NSQScene:
    """Single user-facing entry point for Non-Sequential Raytracing.

    NSQScene owns three typed registries and exposes builder methods for
    common optical elements (lenses, mirrors, doublets), sources, and
    detectors.  The tracer works on flat surface/source/detector lists
    exposed via read-only properties.

    Usage::

        scene = NSQScene()
        scene.add_source('S1', cs, PointSourceConfig(...))
        scene.add_lens('L1', cs, LensConfig(...))
        scene.add_detector('D1', cs, IrradianceDetectorConfig(...))
        result = scene.trace(num_rays=1_000_000, seed=42)

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
        # Rare-path sampling policy: importance biasing, bounded
        # splitting (NumPy forward engine only), and Russian roulette. The
        # default reproduces the engine's pre-PR11 behaviour exactly -- see
        # optiland.nonsequential.ir.scene_ir.SamplingPolicy.
        from optiland.nonsequential.ir.scene_ir import (  # noqa: PLC0415
            SamplingPolicy,
        )

        self.sampling_policy = SamplingPolicy()

    @property
    def surfaces(self) -> list[BaseComponent]:
        """Flat list of all component sub-surfaces in registration order."""
        return self.component_registry.surfaces

    @property
    def sources(self):
        """Ordered list of all registered sources."""
        return self.source_registry.sources

    @property
    def detectors(self):
        """Ordered list of all registered detectors."""
        return self.detector_registry.detectors

    @property
    def component_names(self) -> list[str]:
        """Names of the registered components, in registration order."""
        return list(self.component_registry._registry.keys())

    @property
    def source_names(self) -> list[str]:
        """Names of the registered sources, in registration order."""
        return list(self.source_registry._registry.keys())

    @property
    def detector_names(self) -> list[str]:
        """Names of the registered detectors, in registration order.

        These are the keys of ``SimulationResult.detectors``.
        """
        return list(self.detector_registry._registry.keys())

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

    def add_component(self, name: str, component: BaseComponent) -> None:
        """Add a raw BaseComponent (advanced use).

        Args:
            name: Unique name for this component in the registry.
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
        name: str,
        cs: CoordinateSystem,
        config,
    ) -> None:
        """Add a source to the scene.

        Args:
            name: Unique string name for the source.
            cs: Coordinate system for the source.
            config: Source config dataclass (PointSourceConfig,
                CollimatedSourceConfig, or ExtendedSourceConfig).
        """
        source = _build_source(cs, config)
        self.source_registry.add(name, source)

    def add_detector(
        self,
        name: str,
        cs: CoordinateSystem,
        config,
    ) -> None:
        """Add a detector to the scene.

        Args:
            name: Unique string name for the detector.
            cs: Coordinate system for the detector.
            config: Detector config dataclass (IrradianceDetectorConfig,
                SpectralDetectorConfig, FarFieldDetectorConfig, or
                RayDatabaseConfig).
        """
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

    def trace(
        self,
        num_rays: int,
        max_depth: int = 16,
        min_flux_fraction: float = 1e-6,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seed: int | None = None,
        backend: TracerBackend | None = None,
        record_paths: bool | int = False,
    ) -> SimulationResult:
        """Run the Monte Carlo simulation and return results.

        Args:
            num_rays: Total rays to launch.
            max_depth: Maximum surface hits per ray.
            min_flux_fraction: Russian-roulette threshold, relative to
                per-ray initial flux -- combined with the scene's
                ``sampling_policy.rr_start_flux`` (the larger of the two
                wins). Below threshold, rays are killed with an unbiased
                probability and survivors' flux is boosted accordingly,
                rather than truncated outright.
            batch_size: Rays per processing batch. Does not change the result,
                only the speed; see ``DEFAULT_BATCH_SIZE``.
            seed: RNG seed.
            backend: TracerBackend to use. Defaults to NumpyBackend or
                TorchBackend based on the active ``optiland.backend``.
            record_paths: ``False`` (default) records nothing. ``True``
                records every ray's full path. A positive ``int`` records
                an approximately that-many-ray subset, selected
                deterministically by ``ray_id`` hash, so a
                large trace stays cheap while still yielding a bounded
                sample for visualization/diagnosis, e.g.
                ``scene.trace(num_rays=10_000_000, record_paths=1_000)``.

        Returns:
            SimulationResult with per-detector results and statistics.
        """
        from optiland.nonsequential.tracer import NSQTracer  # noqa: PLC0415

        self.validate()
        tracer = NSQTracer(self, backend=backend)
        return tracer.trace(
            num_rays,
            max_depth=max_depth,
            min_flux_fraction=min_flux_fraction,
            batch_size=batch_size,
            seed=seed,
            record_paths=record_paths,
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

    def to_json(self, path: str | os.PathLike) -> None:
        """Serialize the scene to a versioned JSON file.

        Serializes all components, sources, and detectors.  Simulation
        results and accumulated detector data are **not** included.

        Tensor values (e.g. ``torch.Tensor`` parameters) are detached and
        written as plain floats.  ``requires_grad`` is not persisted; to
        differentiate a loaded scene, re-wrap the relevant parameters in
        ``torch.tensor(..., requires_grad=True)`` after loading.

        Args:
            path: Destination file path (created or overwritten).

        Raises:
            TypeError: If a component/source/detector type is not serializable.
            ValueError: If a material cannot be round-tripped (e.g. no
                catalog name is available).

        Example::

            scene.to_json("my_scene.json")
        """
        from optiland.nonsequential.serialization import (  # noqa: PLC0415
            scene_to_json,
        )

        scene_to_json(self, path)

    @classmethod
    def from_json(cls, path: str | os.PathLike) -> NSQScene:
        """Load a scene from a versioned JSON file.

        The loaded scene is plain-valued: all numeric parameters are Python
        floats, not tensors.  Re-wrap parameters in
        ``torch.tensor(..., requires_grad=True)`` if you need gradients.

        Args:
            path: Path to a JSON file previously written by
                :meth:`to_json`.

        Returns:
            Reconstructed :class:`NSQScene`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the ``nsq_schema_version`` is missing or does not
                match the current loader.

        Example::

            scene = NSQScene.from_json("my_scene.json")
        """
        from optiland.nonsequential.serialization import (  # noqa: PLC0415
            scene_from_json,
        )

        return scene_from_json(path)

    def validate(self) -> None:
        """Validate the scene for common configuration errors.

        Raises:
            ValueError: If no sources or no detectors are registered.
        """
        if not self.source_registry.sources:
            raise ValueError("Scene has no sources. Add at least one source.")
        if not self.detector_registry.detectors:
            raise ValueError("Scene has no detectors. Add at least one detector.")


def _resolve_total_flux(config) -> float:
    """Resolve a source config's radiometric flux [W].

    ``total_flux_lumens``, when set, takes precedence over ``total_flux``
    and is converted to watts via
    :func:`optiland.nonsequential.units.lumens_to_watts` using the config's
    own spectrum -- NSQ's trace loop is radiometric throughout, so this
    conversion happens once, here, at scene-construction time.

    Args:
        config: A source config with ``total_flux``, ``total_flux_lumens``,
            and ``spectrum`` attributes.

    Returns:
        Radiometric flux [W].

    Raises:
        ValueError: If ``total_flux_lumens`` is set and ``spectrum`` has
            negligible overlap with the visible band (guardrail).
    """
    lumens = getattr(config, "total_flux_lumens", None)
    if lumens is None:
        return config.total_flux

    if config.total_flux != 1.0:
        import warnings  # noqa: PLC0415

        warnings.warn(
            f"{type(config).__name__} was given both total_flux="
            f"{config.total_flux!r} and total_flux_lumens={lumens!r}; "
            "total_flux_lumens takes precedence and total_flux is ignored.",
            stacklevel=3,
        )

    from optiland.nonsequential.units import lumens_to_watts  # noqa: PLC0415

    return lumens_to_watts(lumens, config.spectrum)


def _build_source(cs: CoordinateSystem, config) -> object:
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
            total_flux=_resolve_total_flux(config),
            half_angle_deg=config.half_angle_deg,
            medium=getattr(config, "medium", None),
        )
    if isinstance(config, CollimatedSourceConfig):
        return CollimatedSource(
            cs=cs,
            spectrum=config.spectrum,
            total_flux=_resolve_total_flux(config),
            aperture_radius=config.aperture_radius,
            profile=config.profile,
            gaussian_sigma=config.gaussian_sigma,
            medium=getattr(config, "medium", None),
        )
    if isinstance(config, ExtendedSourceConfig):
        return ExtendedSource(
            cs=cs,
            spectrum=config.spectrum,
            total_flux=_resolve_total_flux(config),
            width=config.width,
            height=config.height,
            aperture_radius=config.aperture_radius,
            half_angle_deg=config.half_angle_deg,
            medium=getattr(config, "medium", None),
        )
    raise TypeError(
        f"Unrecognised source config type: {type(config).__name__}. "
        "Expected PointSourceConfig, CollimatedSourceConfig, or ExtendedSourceConfig."
    )


def _build_detector(cs: CoordinateSystem, config) -> object:
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
            splat=config.splat,
            splat_sigma=config.splat_sigma,
            absorb=config.absorb,
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
            splat=config.splat,
            splat_sigma=config.splat_sigma,
            absorb=config.absorb,
        )
    if isinstance(config, FarFieldDetectorConfig):
        return FarFieldDetector(
            cs=cs,
            theta_max_deg=90.0,
            num_bins_theta=config.num_theta,
            num_bins_phi=config.num_phi,
            absorb=config.absorb,
        )
    if isinstance(config, RayDatabaseConfig):
        from optiland.nonsequential.components.geometry.analytic.plane import (  # noqa: PLC0415
            FinitePlaneGeometry,
        )

        geometry = FinitePlaneGeometry(width=config.width, height=config.height)
        return RayDatabaseDetector(
            cs=cs,
            geometry=geometry,
            # 0 ("unlimited", the config default) maps to RayDatabaseDetector's
            # own None-means-unlimited convention (this was
            # previously accepted and silently dropped -- the circular-buffer
            # limit never took effect).
            max_rays=config.max_rays if config.max_rays > 0 else None,
            absorb=config.absorb,
        )
    raise TypeError(f"Unrecognised detector config type: {type(config).__name__}.")
