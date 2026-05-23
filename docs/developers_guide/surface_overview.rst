.. _surface_overview:

Surface Overview
================

Surfaces are core elements of Optiland's optical system. Each surface represents an optical interface, defined by its geometry,
material properties, coatings, and optional apertures. Surfaces are organized into a **Surface Group** to manage operations on multiple surfaces.

Surface Components
------------------

A surface consists of several components that define its optical properties:

- **Geometry**: The shape of the surface (e.g., planar, spherical, aspheric, freeform). This includes the surface's coordinate system.
- **Materials**: The material type before and after the surface, which determines the refractive index and extinction coefficient.
- **Coatings**: Coatings (e.g., thin films) applied to the surface for modifying reflection, transmission, and/or polarization properties.
- **Stop Surface Flag**: Indicates if the surface is the aperture stop of the system.
- **Interaction Model**: Defines how a ray interacts with the surface (e.g., refraction, reflection). For more details, see the :ref:`interaction_models` section.
- **Physical Aperture (optional)**: A physical or virtual aperture defining the area where rays can interact with the surface.
- **BSDF** (optional): Bidirectional Scattering Distribution Function for modeling scattering behavior.

Paraxial Surfaces
-----------------

Optiland handles paraxial surfaces not as a distinct class, but through a `ThinLensInteractionModel` that can be applied to a standard `Surface`. This model simplifies the surface to an ideal thin lens with a given focal length. This approach allows for first-order layouts and analysis. A `Surface` with a `ThinLensInteractionModel` can be converted into a thick lens equivalent using the `convert_to_thick_lens` function in `optiland.surfaces.converters`.

Ray Interaction with Surfaces
-----------------------------

When a ray interacts with a surface, the following steps are typically performed:

1. **Intersection**: The ray's path is intersected with the surface's geometry.
2. **Aperture Check**: The ray's intersection point is checked against the surface's aperture to determine if the ray is blocked.
3. **Refraction/Reflection**: The ray's direction is updated based on Snell's law or the law of reflection, and the ray properties may be affected by the surface's material/coating properties.
4. **Scattering**: If the surface has a BSDF, the ray may be scattered based on the scattering distribution function.
- **Recording**: During a trace, each `Surface` temporarily stores the ray data at the intersection point.

Surface Group
-------------

Surfaces are combined into a **Surface Group**, which manages a collection of surfaces and facilitates operations like ray tracing. The Surface Group:

- Tracks the ordered list of surfaces in the optical system.
- Propagates rays through the system, invoking surface-specific logic at each step.
- Aggregates ray trace history from all surfaces, providing a complete picture of the ray paths.
- Exposes methods for adding, removing, and modifying surfaces in the system.

.. tip::
   The Surface Group allows efficient iteration over multiple surfaces, simplifying complex ray tracing operations.

Surface Factories
-----------------

To streamline surface creation, Optiland uses a multi-factory pattern, with distinct factories for different components:

- **SurfaceFactory**: The main factory that orchestrates the creation of `Surface` objects.
- **GeometryFactory**: Creates different geometry types (e.g., `StandardGeometry`, `ToroidalGeometry`).
- **MaterialFactory**: Handles the creation of materials.
- **CoatingFactory**: Manages the creation of coatings.

This granular approach makes the system more modular and extensible.

Extensibility
-------------

The surface framework is designed for extensibility:

- Custom geometries, coatings, or aperture definitions can be added by subclassing existing components. These may be added to any surface instance.
- The factories can be extended to handle new component types.

For more detailed information on surface geometry and coatings, see their dedicated sections in this guide.

How to Extend This
------------------

**Scenario:** Add a new surface geometry type to Optiland.

**Step 1:** Create a new file in ``optiland/geometries/my_geometry.py``.
**Step 2:** Subclass ``BaseGeometry`` (closed-form intersection) or ``NewtonRaphsonGeometry``
(iterative intersection) and implement ``distance(rays)``, ``sag(x, y)``, and ``surface_normal(rays)``.
**Step 3:** Register in ``optiland/geometries/__init__.py``.
**Step 4:** Add the geometry type string to ``optiland/surfaces/factories/geometry_factory.py``
for JSON serialisation support.
**Step 5:** Add tests in ``tests/test_geometries/test_my_geometry.py``.

For a complete worked example, see :doc:`../examples/Tutorial_8a_Custom_Surface_Types`.
For step-by-step guidance, see :ref:`extension_recipes`.
