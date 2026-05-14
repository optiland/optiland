.. _gallery_nonsequential:

Nonsequential Ray Tracing
=========================

This section showcases examples of nonsequential ray tracing in Optiland, covering everything from basic scene setup to complex stray light analysis.

.. note:: These examples demonstrate the :class:`~optiland.nonsequential.scene.NSQScene` API.

.. carousel::

    ```python
    # Example: Simple Lens with Point Source
    from optiland.nonsequential import NSQScene
    from optiland.nonsequential.components.configs import LensConfig
    from optiland.nonsequential.sources.configs import PointSourceConfig
    from optiland.nonsequential.detectors.configs import IrradianceDetectorConfig
    from optiland.coordinate_system import CoordinateSystem

    scene = NSQScene()
    
    # Add a point source
    scene.add_source('S1', CoordinateSystem(z=-10), PointSourceConfig(half_angle_deg=10))
    
    # Add a lens
    scene.add_lens('L1', CoordinateSystem(z=0), LensConfig(r1=50, r2=-50, thickness=5))
    
    # Add a detector
    scene.add_detector('D1', CoordinateSystem(z=20), IrradianceDetectorConfig(width=10, height=10))
    
    # Trace and visualize
    result = scene.trace(num_rays=100_000)
    scene.view(result)
    ```

    <!-- slide -->

    ```python
    # Example: Stray Light Analysis (Ghosting)
    from optiland.nonsequential.convert import sequential_to_nonsequential
    from optiland.samples.objectives import CookeTriplet

    # Start with a sequential design
    optic = CookeTriplet()
    
    # Convert to NSQ
    scene = sequential_to_nonsequential(optic)
    
    # Trace with high bounce limit to capture ghosts
    result = scene.trace(num_rays=1_000_000, max_bounces=50)
    
    # Analyze detector for ghost images
    result.detectors['image_plane'].view()
    ```

    <!-- slide -->

    ```python
    # Example: Scattering from Rough Surfaces
    # (Placeholder for BSDF example)
    ```

Examples Collection
-------------------

.. toctree::
   :maxdepth: 1

   nonsequential/basic_scene
   nonsequential/stray_light_analysis
   nonsequential/scattering_models
   nonsequential/multi_source_illumination
   nonsequential/gpu_acceleration
