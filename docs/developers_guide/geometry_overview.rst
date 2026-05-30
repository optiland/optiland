Geometry Overview
=================

Geometries define the shapes of optical surfaces and play a key role in ray tracing by handling two critical operations:

1. **Ray-Surface Intersection**: Determining the intersection point (or propagation distance) between a ray and the surface.
2. **Surface Normal Calculation**: Computing the surface normal at the intersection point, which is essential for applying Snell's law or the law of reflection.

Key Components
--------------

A geometry is defined by:

- **Coordinate System**: Each geometry relies on a `CoordinateSystem` object, which specifies:

  - Position: `x`, `y`, `z` (surface origin in the global coordinate space)
  - Rotation: `rx`, `ry`, `rz` (Euler angles or equivalent rotation representation)
  - Reference Coordinate System: Enables nested or relative transformations.

- **Geometry-Specific Parameters**: Additional attributes tailored to the surface type. For example:

- **StandardGeometry**: This class in `standard.py` handles spherical and conic surfaces.
- **Aspheric Geometries**: Implemented in `even_asphere.py` and `odd_asphere.py`.
  - **Custom Geometries**: User-defined parameters. This flexibility allows for a wide range of surface shapes, including freeforms.

.. note::
    Geometries are defined in their local coordinate systems. They are then transformed based on their `CoordinateSystem` argument, which specifies their position and orientation in the global coordinate space.

Intersection and Normal Computation
-----------------------------------

Geometries provide methods for:

1. **Finding the Intersection Point**:

   - For simple shapes, this is computed using closed-form equations.
   - For complex shapes, iterative methods like Newton-Raphson are used to find the intersection.
   - The `NewtonRaphsonGeometry` class serves as the base for geometries requiring iterative solutions.

2. **Computing Surface Normals**:

   - Normals are derived from the mathematical description of the geometry and are essential for determining the ray's direction after refraction or reflection.

Supported Geometry Types
------------------------

Optiland includes a wide range of built-in geometries:

- **Standard**: Handles spherical and conic surfaces.
- **Planes**: Flat surfaces with infinite or finite extent.
- **EvenAsphere**: Described by polynomial terms for deviations from a sphere.
- **OddAsphere**: Similar to even aspheres but with additional terms for odd powers.
- **Biconic**: A surface with different radii and conic constants in x and y.
- **Toroidal**: Defined by two radii of curvature, allowing for toroidal shapes.
- **PlaneGrating and StandardGrating**: Surfaces with diffraction gratings.
- **Polynomial and Chebyshev**: Useful for advanced freeform optical systems.
- **Zernike Surfaces**: Represented by Zernike polynomials. For a detailed mathematical description of the Zernike geometry, see the `Zernike Geometry Mathematics Reference <https://github.com/optiland/optiland/blob/master/docs/references/zernike_description.md>`_.
- **Forbes Surfaces**: As described in the corresponding `Forbes Surface` gallery example, these surfaces are defined following the convention by the papers: [1] G. W. Forbes, “Manufacturability estimates for optical aspheres,” Opt. Express 19(10), 9923–9941 (2011) and [2] G. W. Forbes, "Characterizing the shape of freeform optics," Opt. Express 20, 2483-2499 (2012). The `qpoly.py` module in `optiland\geometries\forbes` was adapted from the implementation in the `prysm <https://github.com/brandondube/prysm?tab=readme-ov-file>` package.
- **NURBS**: Non-Uniform Rational B-Splines for highly flexible freeform surfaces. See the NURBS Freeform Optics gallery example (:ref:`gallery_freeforms`) for usage.
- **Custom Geometries**: Users can easily extend the framework by subclassing the `BaseGeometry` (analytical geometries) or `NewtonRaphsonGeometry` (iterative geometries) classes.

Extensibility
-------------

Adding new geometries is straightforward:

1. Subclass the `BaseGeometry` base class or the `NewtonRaphsonGeometry` class.
2. For `BaseGeometry` implement the `distance(rays)`, `sag(x, y)` and `surface_normal(rays)` methods. For `NewtonRaphsonGeometry` implement the `sag(x, y)` and `_surface_normal(x, y)` methods.
3. Optionally, define additional parameters in the constructor for the geometry's specific shape.


.. tip::
   See the **Surface Overview** section for how geometries integrate with surfaces.

How to Extend This
------------------

**Scenario:** Add a new geometry class to Optiland.

**Step 1:** Create a new file in ``optiland/geometries/my_geometry.py``.
**Step 2:** Subclass ``BaseGeometry`` and implement ``distance(rays)``, ``sag(x, y)``, and
``surface_normal(rays)``. For iterative intersection, subclass ``NewtonRaphsonGeometry`` and
implement ``sag(x, y)`` and ``_surface_normal(x, y)`` only.
**Step 3:** Register in ``optiland/geometries/__init__.py``.
**Step 4:** Add tests in ``tests/test_geometries/test_my_geometry.py``.

For a complete worked example, see :doc:`../examples/Tutorial_8a_Custom_Surface_Types`.
For step-by-step guidance, see :ref:`extension_recipes`.
