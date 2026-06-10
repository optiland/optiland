"""Non-Sequential Raytracing for Optiland.

.. warning:: **Beta release — API stabilizing toward a frozen 1.0.**

   The public symbols listed below are stable and will not be removed without a
   deprecation cycle.  Internal implementation details (backend classes,
   geometry helper methods) may still change.  Differentiability is
   interior-correct for refractive/reflective surfaces; *visibility gradients*
   are not yet supported (see Limitations and Roadmap below).

Overview
--------
Monte Carlo non-sequential ray tracer for illumination design, stray-light
analysis, and non-imaging optics.  Architecturally independent from the
sequential tracer.  Rays propagate freely through a scene and interact with
surfaces in any order, making it suitable for:

- Illumination uniformity and flux budget analysis
- Stray-light and ghost-image characterisation
- Scatter and diffuse-surface modelling
- Non-imaging optics (concentrators, light pipes)
- **Differentiable illumination design** — optimize scene parameters via
  ``loss.backward()`` using the PyTorch backend (see "Differentiability"
  below)

Quick-start::

    import optiland.backend as be
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential import (
        NSQScene, Spectrum,
        CollimatedSourceConfig, LensConfig, IrradianceDetectorConfig,
    )

    be.set_backend("torch")   # optional — enables gradient mode

    scene = NSQScene()
    spec  = Spectrum.monochromatic(0.55)
    scene.add_source("S1", CoordinateSystem(), CollimatedSourceConfig(
        spectrum=spec, total_flux=1.0, aperture_radius=5.0))
    scene.add_lens("L1", CoordinateSystem(z=50), LensConfig(
        r1=100.0, r2=-100.0, thickness=5.0, material="N-BK7",
        front_aperture_radius=12.5))
    scene.add_detector("D1", CoordinateSystem(z=150), IrradianceDetectorConfig(
        width=20, height=20, num_pixels_x=64, num_pixels_y=64))

    result = scene.trace(num_rays=50_000, seed=42)
    result.detectors["D1"].plot()

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

Differentiability
-----------------
When ``optiland.backend`` is configured to ``"torch"`` the tracer builds a
full PyTorch autograd graph through the Monte Carlo loop.  Scene parameters
stored as ``torch.Tensor`` leaf variables (e.g. ``geometry.radius``) receive
gradients via ``loss.backward()``, enabling gradient-based illumination
optimisation.

**v1 envelope:**

- *Gradient mode (autograd)*: ~1 × 10\ :sup:`5` rays × depth 16 on a single
  GPU.  Memory scales as O(num_rays × max_depth) because compaction is
  disabled to keep fixed tensor shapes for the autograd graph.
- *Forward mode (no-grad / NumPy backend)*: 1 × 10\ :sup:`7`\+ rays, depth
  16, fully batched.

Limitations (v1)
----------------
The following capabilities are **not** available in v1 and are tracked on the
roadmap below:

1. **Visibility gradients are zero.**  When a ray silhouette moves across a
   surface boundary (e.g. vignetting), the boundary does not contribute a
   gradient.  Reparameterization (Loubet et al. 2019; Bangaru et al. 2020)
   is roadmap item #1 and is the single biggest physics limitation of the beta.
   A dedicated test (``tests/nonsequential/test_nsq_gradients.py``) explicitly
   asserts this behaviour and links the roadmap.
2. **Mesh geometry is forward-only.**  Analytic conics
   (:class:`ConicGeometry`, :class:`SphereGeometry`,
   :class:`ParaboloidGeometry`) are differentiable; :class:`MeshGeometry` is
   not — ``mesh.backward()`` will raise.
3. **No polarisation.**  Stokes tracking was removed from v1; it is roadmap
   item #6.
4. **Memory cap.**  The ~1 × 10\ :sup:`5`-ray gradient-mode envelope is a
   hard constraint of the naive autograd strategy.  Path Replay
   Backpropagation (roadmap #3) lifts this cap.

Roadmap
-------
Contributions welcome — use the ``feat/nonsequential`` branch and open a PR
against ``master``.  Each item below has the spec section as a reference.

1. **Reparameterization for visibility gradients** — Loubet et al. 2019 /
   Bangaru et al. 2020 warped-area reparameterization so
   silhouette/vignetting/which-surface-hit gradients become correct.  Closes
   the single biggest physics limitation of the beta.
2. **Optimization integration** — wire NSQ into Optiland's
   ``Variable``/operand system and optimizers for end-to-end
   illumination/stray-light design.
3. **Path Replay Backpropagation (PRB)** — constant-memory, unbiased
   gradients; lifts the ~1 × 10\ :sup:`5`-ray cap via the
   ``gradient_mode`` seam already present in :class:`TorchBackend`.
4. **GUI integration** — NSQ scene building and visualization in
   ``optiland_gui``.
5. **Volumetric media** — Beer–Lambert bulk absorption and volume scattering.
6. **Polarisation** — reintroduce Stokes tracking through Fresnel coatings
   (the removed s0–s3, done properly this time).
7. **Dr.Jit / Mitsuba 3 backend** — high-performance
   :class:`~optiland.nonsequential.backends.TracerBackend` plugin via scene
   translation.

Call to Action
--------------
**Try it and tell us what you build.**  If you use the NSQ module for
illumination design, stray-light analysis, or differentiable optics, open a
GitHub issue and describe your use case — your feedback directly shapes the
roadmap.

**Contribute.**  The roadmap items above (especially reparameterization, PRB,
and GUI integration) are open for contributors.  The NSQ revamp spec
``SPEC_NSQ_Revamp_20260608.md`` in the repository root is the canonical
reference.  Join the discussion at
https://github.com/HarrisonKramer/optiland/issues.

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

# Components -- raw
# Components -- compound
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
    # Components -- raw
    "AbsorbingComponent",
    "RefractiveComponent",
    "ReflectiveComponent",
    # Components -- compound
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
