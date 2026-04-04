"""Non-Sequential Raytracing for Optiland.

Monte Carlo non-sequential ray tracer for illumination design, stray-light
analysis, and non-imaging optics. Architecturally independent from the
sequential tracer.

Public API::

    from optiland.nonsequential import (
        NSQScene, NSQTracer,
        PointSource, ExtendedSource, CollimatedSource, Spectrum,
        RefractiveComponent, ReflectiveComponent, AbsorbingComponent,
        PlaneGeometry, FinitePlaneGeometry, SphereGeometry,
        ConicGeometry, MeshGeometry,
        SpecularBRDF, LambertianBSDF, HarveyShackBSDF, TabulatedBSDF,
        IrradianceDetector, FarFieldDetector, SpectralDetector,
        RayDatabaseDetector,
        NSQMaterial, VACUUM,
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

# Components
from optiland.nonsequential.components import (
    AbsorbingComponent,
    ReflectiveComponent,
    RefractiveComponent,
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

# Detectors
from optiland.nonsequential.detectors import (
    FarFieldDetector,
    IrradianceDetector,
    RayDatabaseDetector,
    SpectralDetector,
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
    CollimatedSource,
    ExtendedSource,
    PointSource,
    Spectrum,
)
from optiland.nonsequential.tracer import NSQTracer, SimulationResult

__all__ = [
    # Scene & tracer
    "NSQScene",
    "NSQTracer",
    "SimulationResult",
    # Sources
    "CollimatedSource",
    "ExtendedSource",
    "PointSource",
    "Spectrum",
    # Components
    "AbsorbingComponent",
    "RefractiveComponent",
    "ReflectiveComponent",
    # Geometry
    "ConicGeometry",
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
    "FarFieldDetector",
    "IrradianceDetector",
    "RayDatabaseDetector",
    "SpectralDetector",
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
]

try:
    from optiland.nonsequential.backends import CupyBackend  # noqa: F401

    __all__ += ["CupyBackend"]
except ImportError:
    pass
