Visualization
=============

This section contains the visualization modules of Optiland, which enable 2D and 3D visualization of optical systems.

Visualization is performed in a modular manner to allow for easy customization and extension. Separate
modules for visualization of surfaces, lenses, and rays are provided, both for 2D and 3D. Visualization
of the full system, including rays, is orchestrated by the `optiland.visualization.OpticViewer` or
`optiland.visualization.OpticViewer3D` classes.

.. autosummary::
   :toctree: visualization/
   :caption: visualization Modules

   visualization.analysis.surface_sag
   visualization.component_renderer
   visualization.info.lens_info_viewer
   visualization.info.material_formatter

   visualization.info.providers
   visualization.palettes
   visualization.system.interaction
   visualization.system.lens
   visualization.system.mirror
   visualization.system.optic_viewer
   visualization.system.optic_viewer_3d
   visualization.system.rays
   visualization.system.surface
   visualization.system.system
   visualization.system.utils
   visualization.themes
