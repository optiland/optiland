Core
====

This section covers the core functionality of the Optiland package.
These modules are used to define the basic optical elements and properties of the optical system.
This includes defining the system aperture, fields, wavelength, etc. The core class of
the Optiland package is the :class:`optic.Optic` class, which manages the
properties of the optical system and provides interfaces to core functionalities, such as raytracing.

.. autosummary::
   :toctree: core/
   :caption: Core Modules

   aberrations
   apodization
   aperture
   coordinate_system
   distribution
   jones
   pickup
   plugins
   scatter
   utils
   wavelength
