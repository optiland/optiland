Geometries
==========

This section describes the geometry modules in Optiland. Geometries define the shape of an optical surface
and its position and orientation in space. Optiland provides a variety of geometry types, as well as a means to
build new geometry types by subclassing either the `BaseGeometry` class or the `NewtonRaphsonGeometry` class.
The former is used for geometries that can be described by a closed-form equation, while the latter is used for geometries
that require a numerical solution to find the intersection point with a ray.

.. autosummary::
   :toctree: geometries/
   :caption: Geometry Modules

   geometries.base
   geometries.biconic
   geometries.chebyshev
   geometries.even_asphere
   geometries.forbes
   geometries.odd_asphere
   geometries.newton_raphson
   geometries.nurbs
   geometries.plane
   geometries.plane_grating
   geometries.polynomial
   geometries.standard
   geometries.standard_grating
   geometries.toroidal
   geometries.zernike
