.. _nonsequential_raytracing:

Nonsequential Ray Tracing
=========================

Optiland's nonsequential (NSQ) ray tracing engine allows for the simulation of complex optical systems where light can follow any path, interacting with surfaces in any order. This is essential for modeling illumination systems, stray light, ghost reflections, and systems with arbitrary geometry.

Core Concepts
-------------

The NSQ engine is built around several key concepts that differ from the sequential engine:

- **NSQScene**: The central container for a nonsequential model. Unlike the `Optic` class, which defines a sequence of surfaces, the `NSQScene` holds unordered collections of components, sources, and detectors.
- **Monte Carlo Tracing**: The NSQ engine uses a Monte Carlo approach, where rays are sampled from sources and traced through the scene until they escape, are absorbed, or reach a maximum number of bounces.
- **Russian Roulette**: To maintain physical accuracy while keeping computation bounded, Optiland uses Russian Roulette to decide whether rays are reflected or refracted at interfaces (e.g., Fresnel splitting).
- **Coordinate Systems**: Every component in an NSQ scene is placed using a :class:`~optiland.coordinate_system.CoordinateSystem` in the global frame.

Key Components
--------------

NSQScene
^^^^^^^^

The :class:`~optiland.nonsequential.scene.NSQScene` class is the primary entry point for nonsequential modeling. It provides methods for adding and managing components, sources, and detectors.

.. code-block:: python

    from optiland.nonsequential import NSQScene
    from optiland.coordinate_system import CoordinateSystem

    scene = NSQScene()
    cs = CoordinateSystem()  # Global origin
    
    # Add components, sources, and detectors
    # ...

Components
^^^^^^^^^^

Components are the physical objects in the scene. Optiland provides several built-in compound components:

- **Lens**: A refractive lens with two surfaces and an optional barrel.
- **Mirror**: A reflective surface.
- **Doublet**: A cemented achromatic doublet.

Low-level components (Refractive, Reflective, Absorbing) can also be added directly for custom geometries.

Sources
^^^^^^^

Sources generate the initial ray bundles for the simulation. Available source types include:

- **PointSource**: An infinitesimal point emitting light into a specified cone.
- **CollimatedSource**: A parallel beam of light with a circular cross-section.
- **ExtendedSource**: A spatially and angularly extended source (e.g., a rectangular LED).

Detectors
^^^^^^^^^

Detectors record ray interactions for analysis. Common detectors include:

- **IrradianceDetector**: Records the spatial distribution of light on a rectangular surface.
- **SpectralDetector**: Records irradiance as a function of wavelength.
- **FarFieldDetector**: Records the angular distribution of intensity (radiant intensity).
- **RayDatabaseDetector**: Stores detailed information for every ray that hits the detector surface.

The Tracing Process
-------------------

The tracing process is managed by a :class:`~optiland.nonsequential.tracer.NSQTracer`. You can trigger a trace directly from the scene:

.. code-block:: python

    result = scene.trace(num_rays=1_000_000, max_bounces=100)

Trace Parameters
^^^^^^^^^^^^^^^^

- `num_rays`: Total number of rays to launch.
- `max_bounces`: Maximum number of surface interactions per ray.
- `min_flux_fraction`: A threshold to kill rays that have lost most of their energy.
- `batch_size`: The number of rays processed in a single parallel batch (important for memory management).

Simulation Results
^^^^^^^^^^^^^^^^^^

The trace returns a :class:`~optiland.nonsequential.results.SimulationResult` object containing:

- **Detector Data**: Results from all registered detectors.
- **Statistics**: Ray counts (total, absorbed, escaped, etc.) and flux conservation metrics.
- **Ray Paths**: Optional full path information for visualization (when `record_paths=True`).

Sequential to Nonsequential Conversion
--------------------------------------

Optiland includes a powerful utility to convert a sequential `Optic` system into an `NSQScene`. This is useful for analyzing stray light or ghosting in imaging designs.

.. code-block:: python

    from optiland.nonsequential.convert import sequential_to_nonsequential

    nsq_scene = sequential_to_nonsequential(optic)

Advanced Features
-----------------

Hardware Acceleration
^^^^^^^^^^^^^^^^^^^^^

The NSQ engine supports multiple backends:

- **NumPy**: The default CPU backend.
- **CuPy**: A high-performance GPU backend for large-scale Monte Carlo simulations.

You can specify the backend during the trace:

.. code-block:: python

    from optiland.nonsequential.backends import CupyBackend
    result = scene.trace(num_rays=10_000_000, backend=CupyBackend())

BSDF and Scattering
^^^^^^^^^^^^^^^^^^^

Surfaces can be assigned BSDF (Bidirectional Scattering Distribution Function) models to simulate real-world scattering from rough surfaces or coatings.

Visualization
-------------

NSQ scenes and ray paths can be visualized in both 2D and 3D:

.. code-block:: python

    scene.view(result)    # 2D cross-section with ray overlay
    scene.view3d(result)  # 3D interactive visualization
