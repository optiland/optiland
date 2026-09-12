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

Conic Intersection Numerics and Execution
-----------------------------------------

Geometry modules remain backend agnostic. They pass coordinate/direction arrays,
radius, conic constant, and an optional aperture-membership callable to
``be.conic_intersection``. The operation is part of ``AbstractBackend`` and the
normal NumPy/Torch backend implementations. Backend code does not import ray,
geometry, or aperture classes; the callable returns a boolean mask and only
selects a discrete branch, without gradients through the aperture's state.

``optiland.backend._conic`` holds the single shared mathematical root/selection
policy. NumPy compilation belongs to ``backend.numpy_backend.conic``; tensor
dispatch and custom autograd belong to ``backend.torch_backend.conic``. Keep
framework imports, device/dtype dispatch, and storage adapters in the backend
package rather than adding a parallel Torch implementation for each geometry.
Torch remains optional: importing geometry and tracing with NumPy must work
when Torch is unavailable.

``StandardGeometry`` and ``StandardGratingGeometry`` share the conic solver,
which also supplies the initial intersection for Newton-Raphson geometries.
It solves the factored implicit equation using a cancellation-resistant
quadratic formula. Each ray selects the nearest strictly positive root on
the sag sheet, preferring roots inside a supplied physical aperture. If
neither root is admissible, it returns the finite root nearest the vertex,
including negative distances needed by virtual propagation. An equation
with no finite solution returns NaN.

The solver classifies zero coefficients directly and preserves positive
discriminants. Exact self-crossings are excluded. Sag evaluation can also
leave a rounded origin slightly off the surface: when its implicit residual
is within ``4 * eps * (x*x + y*y + abs(z)*(abs((1+k)*z) + 2*abs(R)))``,
the smaller-magnitude root is a possible self-hit. It is excluded from forward
selection only if its displacement is also within
``4 * eps * sqrt(x*x + y*y + z*z)``. A small residual alone is insufficient
near tangency, where it can correspond to a resolved propagation distance.
The signed fallback remains available. Both bounds rescale with the geometry;
there is no absolute distance floor. As with other floating-point calculations,
roots near tangency remain ill-conditioned and large intermediate values can
overflow.

Torch execution preserves the input tensors' device and dtype and supports
autograd through regular selected roots. An exact double root has a singular
intersection derivative. Its forward value is preserved, but its gradient
contribution is explicitly zero. Root selection and aperture boundaries are
also discrete transitions; derivatives describe the selected branch away
from those boundaries.

Matching one-dimensional NumPy float64 arrays and scalar float64 geometry
parameters use cached Numba loops. The no-aperture loop returns the selected
distance directly. When an aperture is supplied, a second loop retains both
roots so the existing aperture code can choose between them. Both loops and
the general array path share their arithmetic and selection policy.

Ordinary Torch CPU float64 tensors with the same ray shapes and scalar or
single-element surface parameters reuse these loops through NumPy views of
their existing storage. The solver neither copies nor mutates the input ray
arrays. A custom autograd function supplies the implicit derivatives of the
selected root. For hit coordinates ``(u, v, w)``, define
``D = u*L + v*M + ((1+k)*w-R)*N``. Differentiating the implicit conic gives
``dt/dx = -u/D``, ``dt/dy = -v/D``, ``dt/dz = -((1+k)*w-R)/D``,
``dt/dR = w/D``, and ``dt/dk = -w*w/(2*D)``. Direction partials are the
corresponding position partials multiplied by ``t``. These operations remain
in the Torch graph for higher derivatives. Forward-mode differentiation and
batching are supported by the finite-conic kernel; the existing public
plane/conic wrapper still requires an unbatched radius for its plane check.

CUDA, float32, broadcasting, and tensor subclasses use native backend array
operations. CUDA inputs never enter the CPU loop. NumPy array subclasses also
keep the general path. Benchmark after a warmup: the first compiled call has
a compilation or cache-loading cost. CUDA benchmarks additionally require
synchronization around the timed operation.

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
