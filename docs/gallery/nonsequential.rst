.. _gallery_nonsequential:

Non-Sequential Ray Tracing
===========================

The Optiland **non-sequential (NSQ) engine** is a Monte Carlo ray tracer for
illumination design, stray light analysis, and non-imaging optics. Unlike the
sequential tracer — where surfaces are numbered and rays always traverse them
in order — the NSQ engine lets rays propagate freely through a 3-D scene,
bouncing, refracting, and scattering on any surface they encounter.

.. rubric:: When to use the NSQ engine

* **Illumination design** — uniform lighting, LED arrays, light-pipe uniformity
* **Stray light / ghost analysis** — Fresnel reflections, inter-lens ghosts
* **Scatter modelling** — diffuse coatings, rough mirrors (Harvey–Shack ABg model)
* **Non-imaging optics** — solar concentrators, parabolic reflectors
* **Detector characterisation** — encircled energy, spot diagrams, far-field patterns

.. rubric:: Core concepts

**NSQScene**
    The central container. Holds sources, compound components (lenses, mirrors,
    doublets), and detectors. Call ``scene.trace(num_rays=N)`` to run the
    simulation and receive a :class:`~optiland.nonsequential.tracer.SimulationResult`.

**Coordinate system**
    Every object (source, component, detector) is placed with a
    :class:`~optiland.coordinate_system.CoordinateSystem`. Positions are in **mm**;
    rotation angles ``rx``, ``ry``, ``rz`` are in **radians**.

**Spectrum**
    All wavelengths in the NSQ engine are in **micrometres (µm)**.
    Use ``Spectrum.monochromatic(wl_um)`` for a single wavelength or
    ``Spectrum(wavelengths, weights)`` for a polychromatic distribution.

**Sources**

.. list-table::
   :widths: 20 25 55
   :header-rows: 1

   * - Class
     - Config
     - Description
   * - ``PointSource``
     - ``PointSourceConfig``
     - Single point; configurable emission cone (``half_angle_deg``)
   * - ``CollimatedSource``
     - ``CollimatedSourceConfig``
     - Parallel beam; ``tophat`` or ``gaussian`` spatial profile
   * - ``ExtendedSource``
     - ``ExtendedSourceConfig``
     - Rectangular Lambertian emitter; width × height

**Components**

.. list-table::
   :widths: 20 25 55
   :header-rows: 1

   * - Class
     - Config
     - Description
   * - ``Lens``
     - ``LensConfig``
     - Single refractive element; Fresnel splitting at each face
   * - ``Doublet``
     - ``DoubletConfig``
     - Cemented achromatic doublet; crown + flint elements
   * - ``Mirror``
     - ``MirrorConfig``
     - Conic reflective surface; conic=0 (sphere), -1 (paraboloid)

**Detectors**

.. list-table::
   :widths: 20 25 55
   :header-rows: 1

   * - Class
     - Config
     - Result class
   * - ``IrradianceDetector``
     - ``IrradianceDetectorConfig``
     - ``IrradianceMap`` — 2-D spatial flux map [W/mm²]
   * - ``FarFieldDetector``
     - ``FarFieldDetectorConfig``
     - ``FarFieldPattern`` — angular (θ, φ) intensity [W/sr]
   * - ``SpectralDetector``
     - *(construct directly)*
     - ``SpectralResult`` — per-wavelength irradiance map
   * - ``RayDatabaseDetector``
     - *(construct directly)*
     - ``RayDatabase`` — full phase-space record

**BSDFs (Surface Scatter)**

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Class
     - Description
   * - Default (``None``)
     - Fresnel refraction/reflection (probabilistic splitting)
   * - ``SpecularBRDF``
     - Perfect mirror; no transmission
   * - ``LambertianBSDF``
     - Cosine-weighted hemisphere; ``reflectance_value`` ∈ [0, 1]
   * - ``HarveyShackBSDF``
     - ABg roughness model; ``b0``, ``l0``, ``s`` parameters

**Sequential conversion**

``sequential_to_nonsequential(optic)`` converts an existing sequential
:class:`~optiland.optic.optic.Optic` design to an ``NSQScene`` automatically,
mapping singlets → ``Lens``, cemented doublets → ``Doublet``, and the image
surface → ``IrradianceDetector``.

.. rubric:: Quick-start example

.. code-block:: python

    import numpy as np
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential import (
        NSQScene, Spectrum,
        PointSourceConfig, LensConfig, IrradianceDetectorConfig,
    )

    scene = NSQScene()
    spec = Spectrum.monochromatic(0.55)   # 550 nm

    scene.add_source('S', CoordinateSystem(z=-100),
                     PointSourceConfig(spectrum=spec, total_flux=1.0, half_angle_deg=15))
    scene.add_lens('L', CoordinateSystem(z=0),
                   LensConfig(r1=50, r2=-50, thickness=5, material='N-BK7',
                               front_aperture_radius=12.5))
    scene.add_detector('D', CoordinateSystem(z=110),
                       IrradianceDetectorConfig(width=20, height=20))

    result = scene.trace(num_rays=100_000, seed=42)
    result.detectors['D'].plot()

Examples
--------

.. nbgallery::

   nonsequential/01_getting_started
   nonsequential/02_ray_sources
   nonsequential/03_optical_components
   nonsequential/04_detectors_and_results
   nonsequential/05_surface_scattering
   nonsequential/06_simulation_diagnostics
   nonsequential/07_multi_source_illumination
   nonsequential/08_stray_light_analysis
   nonsequential/09_reflective_systems
   nonsequential/10_advanced_topics
