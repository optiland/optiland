Rays
====

This section gives an overview of Optiland modules related to paraxial and real rays, their definition, generation, polarization and ray aiming.
Optiland utilizes NumPy arrays to represent rays, which allows for efficient computation and manipulation of ray data.

Core Ray Modules
----------------

.. autosummary::
   :toctree: rays/
   :caption: Ray Modules

   optiland.rays.paraxial_rays
   optiland.rays.polarization_state
   optiland.rays.polarized_rays
   optiland.rays.ray_generator
   optiland.rays.real_rays

Ray Aiming
----------

This section provides an overview of Optiland modules related to ray aiming.

.. autosummary::
   :toctree: ray_aiming/
   :caption: Ray Aiming Modules

   optiland.rays.ray_aiming.base
   optiland.rays.ray_aiming.cached
   optiland.rays.ray_aiming.initialization
   optiland.rays.ray_aiming.iterative
   optiland.rays.ray_aiming.paraxial
   optiland.rays.ray_aiming.registry
   optiland.rays.ray_aiming.robust
