Surfaces
========

This section describes the surface modules in Optiland. Surfaces are the building blocks of optical systems, 
and Optiland provides a variety of surface types and properties to support the design and analysis of complex optical systems.

Optiland builds individual surfaces into the so-called `SurfaceGroup`, which represents a collection of surfaces that form an optical system.
Most analyses are performed on the `SurfaceGroup` level, which abstracts the complexity of individual surfaces and provides a unified interface for optical system design and analysis.

.. autosummary::
   :toctree: surfaces/
   :caption: Surface Modules

   surfaces.converters
   surfaces.image_surface
   surfaces.object_surface
   surfaces.standard_surface
   surfaces.surface_group

Surface Factory
---------------

The `surfaces.factories` subpackage is used to build surface instances based on user-provided inputs. The primary class class in this subpackage
is the `SurfaceFactory`, which orchestrates generation of surfaces and delegates creation of surface subcomponents to several
submodules, which are listed here.

.. autosummary::
   :toctree: surfaces/factories/
   :caption: Factory Modules
   :recursive:

   surfaces.factories.coating_factory
   surfaces.factories.coordinate_system_factory
   surfaces.factories.geometry_factory
   surfaces.factories.interaction_model_factory
   surfaces.factories.material_factory
   surfaces.factories.surface_factory
