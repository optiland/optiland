r"""Non-Sequential Raytracing for Optiland.

.. warning:: **Pre-release — not yet part of a tagged Optiland version.**

   NSQ has never shipped in an official release, so the public API may still
   change without a deprecation cycle. Once it ships in a tagged release the
   usual Optiland API-stability guarantee applies. Differentiability is
   interior-correct for refractive/reflective surfaces; *visibility
   gradients* are not yet supported (see Limitations and Roadmap below).

Overview
--------
Monte Carlo non-sequential ray tracer for illumination design, stray-light
analysis, and non-imaging optics. Architecturally independent from the
sequential tracer. Rays propagate freely through a scene and interact with
surfaces (and, for AR coatings and bulk-absorbing glass, media) in any
order, making it suitable for:

- Illumination uniformity and flux budget analysis, in radiometric (W) or
  photometric (lm, lux) units
- Stray-light and ghost-image characterisation, with coatings and mirror
  reflectance so NSQ and the sequential engine agree on R
- Scatter and diffuse-surface modelling (Lambertian, Harvey-Shack, tabulated
  BRDF/BTDF)
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
    print(result.report())       # self-diagnosing summary -- read this first
    result.detectors["D1"].plot()

Public API (selected)::

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
        Volume, NonWatertightVolumeError,
        # Geometry
        ConicGeometry, FinitePlaneGeometry, MeshGeometry, SphereGeometry,
        ParaboloidGeometry, PlaneGeometry,
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
        # Conversion from a sequential Optic
        sequential_to_nonsequential, ConversionError,
        # Photometric units
        to_photometric, lumens_to_watts, v_lambda,
        PhotometricMap, PhotometricScalar,
        # Visualization
        NSQViewer2D, NSQViewer3D,
    )

See :mod:`optiland.nonsequential.scene`, :mod:`optiland.nonsequential.units`,
and :mod:`optiland.nonsequential.convert` for the rest, or the
:ref:`API reference <api_nonsequential>` for the complete, generated symbol
list.

Coatings, mirrors, and absorption
----------------------------------
Reflectance comes from the same ``optiland.coatings`` models the sequential
engine uses, so the two engines agree on R for a given coating::

    from optiland.coatings import SimpleCoating
    from optiland.nonsequential import SurfaceConfig

    ar_coat = SimpleCoating(reflectance=0.005, transmittance=0.995)
    scene.add_lens("L1", cs, LensConfig(
        r1=100.0, r2=-100.0, thickness=5.0, material="N-BK7",
        front_aperture_radius=12.5,
        front=SurfaceConfig(coating=ar_coat),
    ))

A mirror's ``reflectance`` is **required** — a constant, a
``callable(wavelength_um) -> reflectance``, or a coating. There is no
implicit perfect-mirror default, so a mirror built without one raises rather
than silently reflecting 100%. Bulk (Beer-Lambert) absorption through a
glass path is automatic whenever the material's extinction coefficient
``k`` is nonzero — no configuration needed.

Diagnostics
-----------
Every :class:`~optiland.nonsequential.tracer.SimulationResult` carries a
``diagnostics`` object (depth-truncated flux, Russian-roulette loss,
unreached geometry, per-detector undersampling, flux-conservation error)
with a threshold-based warning list. ``result.report()`` prints it;
``__repr__`` shows a one-line warning count so a problem is visible even in
a REPL. Read this before trusting a trace's numbers.

Rare-path sampling
-------------------
``scene.sampling_policy`` (a
:class:`~optiland.nonsequential.ir.scene_ir.SamplingPolicy`) controls how
rare paths (faint ghosts, high-order reflections) are sampled: importance
biasing (``reflect_prob``) works on both backends, bounded bounce-splitting
(``split_depth``) is NumPy-forward-only, and Russian roulette
(``rr_start_flux``) replaces flux truncation as an unbiased low-flux kill.
Defaults reproduce plain single-branch sampling.

Reproducibility
----------------
Random numbers are drawn from a counter-based PCG32 keyed by
``(seed, ray_id, bounce, event_slot)`` — see
:mod:`optiland.nonsequential.rng`. Results are bit-identical across
``batch_size``, ray-bundle compaction, and — for the *random decisions*
specifically — the NumPy and Torch backends. Final floating-point results
agree only to documented tolerance across backends (arithmetic order,
FMA, and transcendental implementations differ); same random decisions,
same code path, is the actual guarantee.

Differentiability
-----------------
When ``optiland.backend`` is configured to ``"torch"`` the tracer builds a
full PyTorch autograd graph through the Monte Carlo loop. Pass a
``torch.Tensor`` with ``requires_grad=True`` directly into the ordinary
config objects and call ``loss.backward()``::

    r1 = torch.tensor(120.0, dtype=torch.float64, requires_grad=True)
    scene.add_lens("L1", CoordinateSystem(z=100.0), LensConfig(
        r1=r1, r2=float("inf"), thickness=5.0, material="N-BK7",
        front_aperture_radius=12.0))
    loss = scene.trace(num_rays=20_000, seed=1).detectors["D1"].data.sum()
    loss.backward()        # -> r1.grad

**Differentiable parameters:**

- Component geometry — conic ``radius`` and ``conic``, ``aperture_radius``,
  sphere/plane/annulus/frustum extents
- ``IrradianceDetector`` ``width``, ``height``, and the (now attached)
  ``total_flux`` on every detector result
- Source ``total_flux``
- Material refractive index and extinction coefficient ``k`` (Beer-Lambert
  absorption), and BSDF reflectance/transmittance, including
  ``scatter_fraction`` (the detached-sample/attached-weight branch, not a
  silently dead variable)

**Not differentiable in this release** (these *raise* rather than silently
detaching, so a dead design variable is never mistaken for a live one):

- Source geometry — ``aperture_radius``, ``half_angle_deg``, emitter extent —
  because source sampling runs in NumPy
- ``SpectralDetector`` extents, which accumulate into a NumPy histogram
- Visibility: *which* surface a ray hits is a discrete choice and contributes
  no gradient (see Limitations below)

Use ``float64`` (``be.set_precision("float64")``) for gradient work; the
Monte Carlo trace is numerically delicate near surface edges.

**Current envelope:**

- *Gradient mode (autograd)*: ~1 × 10\ :sup:`5` rays × depth 16 on a single
  GPU. Memory scales as O(num_rays × max_depth) because compaction is
  disabled to keep fixed tensor shapes for the autograd graph.
- *Forward mode (no-grad / NumPy backend)*: 1 × 10\ :sup:`7`\+ rays, depth
  16, fully batched.

Limitations & Roadmap
---------------------
The biggest limitation is that **visibility gradients are zero** (silhouette,
vignetting, and which-surface-hit boundaries do not contribute gradients);
mesh geometry is forward-only, there is no polarisation, and gradient mode is
capped at ~1e5 rays. These gaps and the full roadmap
(reparameterization → optimization integration → PRB → GUI → volumetric
scattering → polarisation → Dr.Jit/Mitsuba) are tracked on the canonical
**NSQ Limitations & Roadmap** documentation page:
https://optiland.readthedocs.io/en/latest/gallery/nonsequential/limitations_and_roadmap.html

Call to Action
--------------
**Try it and tell us what you build.** If you use the NSQ module for
illumination design, stray-light analysis, or differentiable optics, open a
GitHub issue and describe your use case — your feedback directly shapes the
roadmap.

**Contribute.** The roadmap items above (especially reparameterization, PRB,
and GUI integration) are open for contributors. The Limitations and Roadmap
page linked above is the canonical reference. Join the discussion at
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
    NonWatertightVolumeError,
    ReflectiveComponent,
    RefractiveComponent,
    SurfaceConfig,
    Volume,
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

# RNG
from optiland.nonsequential.rng import EventSlot, NSQRng

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

# Photometric conversion layer
from optiland.nonsequential.units import (
    PhotometricMap,
    PhotometricScalar,
    lumens_to_watts,
    to_photometric,
    v_lambda,
)

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
    "NonWatertightVolumeError",
    "SurfaceConfig",
    "Volume",
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
    # RNG
    "EventSlot",
    "NSQRng",
    # Backends
    "NumpyBackend",
    "TracerBackend",
    # Visualization
    "NSQViewer2D",
    "NSQViewer3D",
    # Converter
    "ConversionError",
    "sequential_to_nonsequential",
    # Photometric conversion layer
    "PhotometricMap",
    "PhotometricScalar",
    "lumens_to_watts",
    "to_photometric",
    "v_lambda",
]
