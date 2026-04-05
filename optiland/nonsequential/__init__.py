"""Non-Sequential Raytracing for Optiland.

Monte Carlo non-sequential ray tracer for illumination design, stray-light
analysis, and non-imaging optics.  Architecturally independent from the
sequential tracer.

Public API::

    from optiland.nonsequential import (
        # Scene & tracer
        NSQScene, NSQTracer, SimulationResult,
        # Sources
        CollimatedSource, ExtendedSource, PointSource, Spectrum,
        PointSourceConfig, CollimatedSourceConfig, ExtendedSourceConfig,
        # Components (raw)
        AbsorbingComponent, RefractiveComponent, ReflectiveComponent,
        # Compound components
        Lens, Mirror, Doublet,
        LensConfig, MirrorConfig, DoubletConfig, SurfaceConfig, InteractionType,
        # Geometry
        ConicGeometry, FinitePlaneGeometry, MeshGeometry,
        CylindricalFrustumGeometry, AnnularPlaneGeometry,
        # BSDF
        SpecularBRDF, LambertianBSDF, HarveyShackBSDF, TabulatedBSDF,
        # Detectors
        IrradianceDetector, FarFieldDetector, SpectralDetector,
        RayDatabaseDetector,
        IrradianceDetectorConfig, FarFieldDetectorConfig,
        SpectralDetectorConfig, RayDatabaseConfig,
        # Materials
        NSQMaterial, VACUUM,
        # Visualization
        NSQViewer2D, NSQViewer3D,
    )

Kramer Harrison, 2026
"""

from __future__ import annotations

# Backends
from optiland.nonsequential.backends import NumpyBackend, TracerBackend

# BSDF
from optiland.nonsequential.bsdf import (
    HarveyShackBSDF,
    LambertianBSDF,
    SpecularBRDF,
    TabulatedBSDF,
)

# Components — raw
# Components — compound
from optiland.nonsequential.components import (
    AbsorbingComponent,
    ComponentRegistry,
    CompoundComponent,
    Doublet,
    DoubletConfig,
    InteractionType,
    Lens,
    LensConfig,
    Mirror,
    MirrorConfig,
    ReflectiveComponent,
    RefractiveComponent,
    SurfaceConfig,
)

# Geometry
from optiland.nonsequential.components.geometry import (
    ConicGeometry,
    FinitePlaneGeometry,
    MeshGeometry,
    ParaboloidGeometry,
    PlaneGeometry,
    SphereGeometry,
)
from optiland.nonsequential.components.geometry.analytic import (
    AnnularPlaneGeometry,
    CylindricalFrustumGeometry,
)

# Converter
from optiland.nonsequential.convert import ConversionError, sequential_to_nonsequential

# Detectors
from optiland.nonsequential.detectors import (
    BaseDetector,
    DetectorRegistry,
    FarFieldDetector,
    FarFieldDetectorConfig,
    IrradianceDetector,
    IrradianceDetectorConfig,
    RayDatabaseConfig,
    RayDatabaseDetector,
    SpectralDetector,
    SpectralDetectorConfig,
)

# Materials
from optiland.nonsequential.materials import VACUUM, NSQMaterial

# Results
from optiland.nonsequential.results import (
    FarFieldPattern,
    IrradianceMap,
    RayDatabase,
    SpectralResult,
)

# Scene and tracer
from optiland.nonsequential.scene import NSQScene

# Sources
from optiland.nonsequential.sources import (
    BaseNSQSource,
    CollimatedSource,
    CollimatedSourceConfig,
    ExtendedSource,
    ExtendedSourceConfig,
    PointSource,
    PointSourceConfig,
    SourceRegistry,
    Spectrum,
)
from optiland.nonsequential.tracer import NSQTracer, SimulationResult

# Visualization
from optiland.nonsequential.visualization import NSQViewer2D, NSQViewer3D

__all__ = [
    # Scene & tracer
    "NSQScene",
    "NSQTracer",
    "SimulationResult",
    # Sources
    "BaseNSQSource",
    "CollimatedSource",
    "CollimatedSourceConfig",
    "ExtendedSource",
    "ExtendedSourceConfig",
    "PointSource",
    "PointSourceConfig",
    "SourceRegistry",
    "Spectrum",
    # Components — raw
    "AbsorbingComponent",
    "RefractiveComponent",
    "ReflectiveComponent",
    # Components — compound
    "CompoundComponent",
    "ComponentRegistry",
    "Doublet",
    "DoubletConfig",
    "InteractionType",
    "Lens",
    "LensConfig",
    "Mirror",
    "MirrorConfig",
    "SurfaceConfig",
    # Geometry
    "AnnularPlaneGeometry",
    "ConicGeometry",
    "CylindricalFrustumGeometry",
    "FinitePlaneGeometry",
    "MeshGeometry",
    "ParaboloidGeometry",
    "PlaneGeometry",
    "SphereGeometry",
    # BSDF
    "HarveyShackBSDF",
    "LambertianBSDF",
    "SpecularBRDF",
    "TabulatedBSDF",
    # Detectors
    "BaseDetector",
    "DetectorRegistry",
    "FarFieldDetector",
    "FarFieldDetectorConfig",
    "IrradianceDetector",
    "IrradianceDetectorConfig",
    "RayDatabaseDetector",
    "RayDatabaseConfig",
    "SpectralDetector",
    "SpectralDetectorConfig",
    # Materials
    "NSQMaterial",
    "VACUUM",
    # Results
    "FarFieldPattern",
    "IrradianceMap",
    "RayDatabase",
    "SpectralResult",
    # Backends
    "NumpyBackend",
    "TracerBackend",
    # Visualization
    "NSQViewer2D",
    "NSQViewer3D",
    # Converter
    "ConversionError",
    "sequential_to_nonsequential",
]

try:
    from optiland.nonsequential.backends import CupyBackend  # noqa: F401

    __all__ += ["CupyBackend"]
except ImportError:
    pass
