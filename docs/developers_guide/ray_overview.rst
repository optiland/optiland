.. _ray_overview:

Ray Tracing Framework
=====================

The ray tracing framework in Optiland simulates light propagation through an optical system. It handles various types of rays,
supports flexible ray generation, and traces rays across surfaces in sequential optical systems.

Ray Representation
------------------
Rays are the fundamental elements of the ray tracing process. A ray consists of the following attributes:

- **Origin (x, y, z)**: The starting point of the ray.
- **Direction (L, M, N)**: The ray's unit vector indicating its direction.
- **Wavelength**: The wavelength of the ray in the system.
- **Intensity**: The ray's relative intensity.
- **Optical Path Difference (OPD)**: The accumulated path difference of the ray relative to the chief ray.
- **Polarization Matrix (optional)**: A 3x3 matrix representing the transformation of a ray's initial electric field into its final state. This is used by the `PolarizedRays` class.

.. note::
  Ray attributes use arrays from the active backend (NumPy or optional Torch).

Ray Types
---------
Optiland supports three types of rays:

- **Real Rays**: Standard rays with full 3D attributes, used for detailed tracing.
- **Paraxial Rays**: Simplified rays using only height (`y`), angle (`u`), wavelength, and starting position (`z`). These are used for first-order system analysis.
- **Polarized Rays**: Rays with additional polarization information, allowing simulations involving birefringent materials or coatings.

Ray Generation
--------------
The ray tracing framework uses a **Ray Generator** to produce rays for tracing. The generator operates based on user-defined parameters, including:

- Field points
- Wavelengths
- Aperture sampling (e.g., uniform grid, random, etc.)
- Polarization state
- System properties (e.g., F/#, NA, telecentricity)
- Apodization (intensity distribution within the pupil)

Generated rays are passed to the **Surface Group** for tracing through the optical system. Each `Optic` instance has a `ray_tracer` attribute, which in turn contains the `RayGenerator`.

Ray Tracing
-----------
For real rays, the ray tracing process is managed by the **RealRayTracer**. The ray tracer is responsible for:

- Generating the appropriate rays via its `RayGenerator`.
- Ray tracing input validation.
- Propagating rays through the surface group.

For paraxial rays, there are two primary methods for tracing:
- The **ParaxialRayTracer** class provides a dedicated tracer for paraxial rays.
- The `Surface.trace` method can also directly handle `ParaxialRays`, providing an alternative way to trace them.

Tracing Process
---------------
Ray tracing is performed by sequentially propagating rays across the surfaces in the system:

1. Rays are converted into the local coordinate system of the current surface.
2. Rays intersections with the surface are identified and the ray propagates to the intersection point.
3. If the surface has a physical aperture, rays may be clipped (intensity set to zero) if they fall outside this aperture.
4. The rays interact with the surface, modifying their direction, intensity, polarization matrix, and other attributes.
5. Ray are transformed back to the global coordinate system.
6. Ray information (intersection points, intensities, etc.) is recorded on the surface for later analysis and visualization.
7. The process is repeated for each surface in the system.

Paraxial rays follow a similar process but use simplified equations for faster computations.

.. _polarization_transport:

Polarization frames and scalar power
----------------------------------------

``PolarizedRays.p`` maps the fixed incident electric-field basis to the current
output frame. Coordinate-system localization/globalization rotates its output
rows alongside ray positions and directions; it retains the original input
basis. This permits a local Cartesian Jones element on a tilted surface without
interpreting its coefficients as incidence-plane s/p coefficients.

Use ``rays.attenuate(factor)`` for polarization-independent **intensity** losses,
and ``rays.clip(mask)`` to extinguish rays. In polarized bundles these operations
update the retained scalar power as well as the currently evaluated intensity.
``update_intensity(state)`` can then evaluate another incident state without
restoring absorbed/clipped power or applying a previous loss twice.
``get_exit_fields(state)`` includes the corresponding square-root field scaling.
Unpolarized light uses two orthogonal incoherent fields whose powers add to the
reported intensity.

A zero intensity for one incident polarization is not the same as a clipped
ray: a polarizer that blocks vertical input may still transmit horizontal input.
Retained scalar power distinguishes these cases. Extinguished rays remain dark
even if a later missed surface produces invalid geometry or polarization
matrices. Invalid results on rays with nonzero retained power remain visible;
they are not silently converted to zero.

.. list-table:: Responsibilities and callers
   :header-rows: 1
   :widths: 35 65

   * - Component
     - Responsibility and callers
   * - ``RealRays.attenuate`` / ``PolarizedRays.attenuate``
     - Apply scalar intensity factors. Called by ``SimpleCoating``,
       ``HomogeneousPropagation`` and phase-profile efficiency handling.
   * - ``PolarizedRays.clip``
     - Extinguish both current intensity and retained scalar power. Called by
       physical-aperture clipping and other ray rejection paths.
   * - ``PolarizedRays.rotate_x/y/z``
     - Transport output fields with the ray frame. Called by coordinate-system
       transforms; supports scalar or per-ray rotation angles in radians.
   * - ``CartesianJonesCoating``
     - Apply local X/Y complex coefficients after geometrical interaction.
       Called by surface coating dispatch; available to file readers.
   * - ``update_intensity`` / ``get_exit_fields``
     - Evaluate an incident state using retained transport and scalar power.
       Used by the real-ray tracer and field/polarization analyses.
   * - ``PolarizationState`` / ``OpticSerializer``
     - Own source-state validation and backend-independent native JSON.
       File readers reuse that existing contract rather than adding another
       polarization representation.

Directly assigning ``rays.i`` changes only the current evaluation. Custom scalar
loss models should call ``attenuate`` so subsequent field/state evaluation keeps
their losses. This does not change Jones matrices into intensity coefficients.

Changes to existing results
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Coordinate transformations now rotate the output field with ray positions and
directions. The incident field basis remains fixed. Consequently a native
``PolarizerCoating`` with local X transmission and a surface Z rotation of
45 degrees transmits horizontal input with field ``(0.5, 0.5, 0)`` and power
0.5 for unit incident power. Previously the unrotated field matrix incorrectly
gave ``(1, 0, 0)`` and power 1. Coating axes are expressed in the surface's
local frame; a direct Jones calculation uses the frame of its supplied rays.

``SimpleCoating`` now transports the field through the geometrical bend before
applying its scalar power factor, just as an uncoated surface transports it.
Previously that field could cease to be transverse after refraction/reflection.
For example, a unit horizontal ray incident at 0.35 radians onto a 1-to-1.5
interface with transmittance 0.36 leaves with amplitude
``0.6 * (cos(beta), 0, -sin(beta))``, where
``beta = asin(sin(0.35) / 1.5)``. Its evaluated power remains 0.36 rather than
returning to 1.

Simple-coating losses, bulk absorption, phase efficiency and clipping now persist
through source-state re-evaluation and exit-field analyses. The private ``_i0``
therefore stores incident power multiplied by accumulated scalar intensity
factors, rather than immutable launch power. Jones amplitude losses remain in
``p``. Zero power stays zero after a later invalid intersection; invalid fields
on powered rays are still exposed. Blocking one incident polarization in ``p``
does not prevent evaluating another transmitted state.

Scalar ray geometry and optical path calculations are unchanged. Aligned,
lossless polarized systems retain their previous results. Tilted anisotropic
elements, scalar-coated bends and polarized power/field analyses with scalar
loss or clipping may change for the corrections above.

.. note::
   For details on how rays interact with surfaces, see the :ref:`surface_overview` section.

Extensibility
-------------

The framework is designed to be extensible:

- New ray types can be added by subclassing the `BaseRays` base class.
- Custom ray generators can be implemented by following the existing interface.
- Additional tracing logic can be integrated into the surface `trace` method for specialized applications.

For a practical example of ray tracing, see the :ref:`getting_started` section.

Ray Aiming
----------
Optiland implements a flexible **Ray Aiming** system to determine the correct launch coordinates for rays such that they fill the stop surface, even in systems with significant pupil aberrations.

This functionality is managed by the :class:`~optiland.rays.ray_aiming.base.BaseRayAimer` class and its subclasses. The system uses a **Registry Pattern**, allowing users to easily switch strategies or register custom ones.

Available Strategies
^^^^^^^^^^^^^^^^^^^^
- **Paraxial**: Standard aiming using paraxial entrance pupil approximation. Fast but less accurate for wide-angle/aberrated systems.
- **Iterative**: Uses a Newton-Raphson-like iterative solver to refine ray launch coordinates until they hit the physical stop.
- **Robust**: An extension of the iterative method using **Chief-Ray Calibration** (or Pupil Map). It determines the entrance-pupil center per field first (via field-marching continuation), fits a local 2D affine pupil map from cardinal edge probes, and uses this map to seed a fast and precise 2-DOF Newton-Broyden polish step. This guarantees convergence in highly stressed and wide-angle systems.
- **Cached**: A wrapper that caches intermediate results from any other strategy. Retained for backward compatibility (explicit ``cache=True``), but retired-by-default for the ``robust`` aimer, which implements its own intrinsic, always-on warm-start ``PupilMapCache``.

Configuration
^^^^^^^^^^^^^
Ray aiming is configured via the ray tracer on the `Optic` instance:

.. code-block:: python

    # Enable robust ray aiming
    optic.ray_tracer.set_aiming(mode="robust", max_iter=20, tol=1e-6)

Custom Aimers
^^^^^^^^^^^^^
Users can implement custom strategies by subclassing `BaseRayAimer` and registering them:

.. code-block:: python

    from optiland.rays.ray_aiming import BaseRayAimer, register_aimer

    @register_aimer("my_custom_aimer")
    class MyAimer(BaseRayAimer):
        def aim_rays(self, fields, wavelengths, pupil_coords):
            # Custom logic
            return x, y, z, L, M, N
