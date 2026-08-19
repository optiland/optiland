.. _glossary:

Glossary
========

This page defines the core concepts in Optiland. Want runnable examples? See the :ref:`cheat_sheet`.

Optic
-----

The central container for an entire optical system. An ``Optic`` instance holds all surfaces, aperture
definitions, field points, and wavelength information. It also exposes paraxial analysis, aberration
computations, the ray tracer, and polarization state.

Example: ``lens = optic.Optic(name="My System")``

SurfaceGroup
------------

Manages the ordered collection of ``Surface`` objects within an ``Optic``. It propagates rays through
the system by invoking surface-specific logic at each step, aggregates ray-trace history, and exposes
methods for adding, removing, and modifying surfaces.

Surface
-------

Represents a single optical interface — a lens element, mirror, image plane, grating, etc. Each
surface is composed of:

- **Geometry**: The mathematical shape (e.g., planar, spherical, aspheric, freeform).
- **Material**: Refractive index and extinction coefficient on each side.
- **Coating**: Optional thin-film stack modifying reflection, transmission, or polarization.
- **Interaction model**: How rays interact — refraction, reflection, diffraction, or a custom phase profile.
- **Physical aperture**: Optional mask defining the clear aperture.
- **BSDF**: Bidirectional scattering distribution function for surface scatter.

Special surface types include ``ObjectSurface`` (first surface, object plane), ``ImageSurface``
(final surface, image plane), and surfaces carrying a ``ThinLensInteractionModel`` (paraxial thin-lens
approximation).

Material
--------

Defines the optical medium between surfaces. Optiland supports:

- ``MaterialFile``: Loads dispersion data from the `refractiveindex.info <https://refractiveindex.info>`_
  database, which is bundled with the package.
- ``IdealMaterial``: A wavelength-independent medium specified by a single refractive index.
- ``AbbeMaterial``: A medium specified by its refractive index at the d-line (nd) and Abbe number (Vd),
  using a Buchdahl dispersion model (recommended) or a legacy polynomial model.
- User-registered materials via ``MaterialRegistry``.

Geometry
--------

Defines the mathematical shape of a surface and provides two critical operations: ray–surface intersection
and surface normal computation. Built-in geometries include:

- ``StandardGeometry``: Spherical and conic surfaces.
- ``EvenAsphere``, ``OddAsphere``: Polynomial aspheres.
- ``Biconic``, ``Toroidal``, ``Polynomial``, ``ChebyshevPolynomialGeometry``, ``ZernikePolynomialGeometry``: Freeforms.
- ``PlaneGrating``, ``StandardGrating``: Diffraction gratings.
- ``NURBSGeometry``: Non-Uniform Rational B-Splines.
- ``ForbesGeometry``: Q-polynomial freeform surfaces (Forbes convention).

Custom geometries can be added by subclassing ``BaseGeometry`` (closed-form) or
``NewtonRaphsonGeometry`` (iterative).

Aperture
--------

Defines the system's limiting aperture. The aperture type determines how the entrance-pupil size is
specified:

- ``EPD``: Entrance pupil diameter in mm.
- ``imageFNO``: Image-space f-number.
- ``objectNA``: Object-space numerical aperture.
- ``float_by_stop_size``: The aperture stop physical diameter drives the pupil size.

Fields
------

Define the points in the object plane (or angular directions) being imaged. Fields can be specified by
angle (degrees) or object height (mm). Vignetting factors can be applied per field. Each ``Field`` has an
optional ``weight`` used in weighted analysis aggregation.

Wavelengths
-----------

Specify the wavelengths of light used for analysis. All values are stored internally in microns (µm).
One wavelength is designated as the primary wavelength, used for paraxial calculations and single-wavelength
analyses. Each ``Wavelength`` has an optional ``weight`` used in weighted analysis aggregation.

Coordinate System
-----------------

Each surface has its own Local Coordinate System (LCS) defined by position (x, y, z) and rotation
(rx, ry, rz) relative to a reference. Key conventions:

- Light propagates from **left to right** along the **+z axis**.
- Surface 1 is typically at the global origin (z = 0).
- **Thickness** is the axial separation to the *next* surface; positive means to the right.
- **Radius of curvature**: positive means center of curvature to the right (convex to the incoming
  beam); negative means to the left.
- Tilts and decenters are applied as ``R = Rz @ Ry @ Rx``.

Apodization
-----------

Defines the intensity (amplitude) distribution across the entrance pupil. The default is
``UniformApodization`` (flat). ``GaussianApodization`` applies a Gaussian intensity profile across the
pupil, modelling a Gaussian input beam.

Backend
-------

Optiland routes all numerical operations through a unified backend abstraction (``optiland.backend``),
allowing transparent switching between **NumPy** (default, CPU) and **PyTorch** (GPU and autograd). All
Optiland code uses ``import optiland.backend as be`` instead of importing NumPy or PyTorch directly.
Switch backends with ``be.set_backend("torch")`` or ``be.set_backend("numpy")``.

See :ref:`configurable_backend` in the Developer's Guide for the full backend architecture.

Non-sequential ray tracing
--------------------------

A tracing mode where rays propagate freely through a 3-D scene and interact with surfaces in **any
order** (rather than a fixed numbered sequence). Used for illumination, stray-light, ghost, and
non-imaging analysis. See the :ref:`NSQ gallery <gallery_nonsequential>`.

Monte Carlo tracing
-------------------

Sampling many random rays from sources and accumulating their contributions statistically. NSQ uses
Monte Carlo sampling; results converge as the ray count increases.

Splatting
---------

How a detector deposits a ray's flux into its pixel grid. ``hard`` splatting bins to the nearest
pixel; ``bilinear`` splatting distributes flux across the four nearest pixels and is **differentiable**
with respect to the landing position.

Throughput (weight)
-------------------

A per-ray multiplicative factor on flux that accumulates as a ray reflects, refracts, or scatters.
NSQ's differentiable Fresnel split uses an *attached* throughput weight whose forward value is 1 but
which carries gradients (see *detached-sample / attached-weight*).

BSDF
----

Bidirectional Scattering Distribution Function — the model that determines how a surface scatters
incident light into outgoing directions (e.g. specular, Lambertian, Harvey–Shack roughness, or
tabulated measured data).

Irradiance map
--------------

A 2-D spatial distribution of flux per unit area [W/mm²] recorded on a detector surface.

Far-field pattern
-----------------

The angular distribution of radiant intensity [W/sr] emitted into a hemisphere, recorded by a
``FarFieldDetector``.

Étendue
-------

A conserved geometric quantity (area × projected solid angle) describing how "spread out" a bundle of
light is in space and angle. It bounds what any non-imaging concentrator can achieve.

Detached-sample / attached-weight
---------------------------------

The estimator NSQ uses to keep stochastic reflect/refract choices differentiable: the *branch* is
sampled from a detached (non-differentiable) probability, while the throughput *weight* stays attached
to the autograd graph so gradients flow through material and geometry parameters. See the
:ref:`NSQ developer guide <nonsequential_raytracing>`.

Visibility gradient
-------------------

The gradient contributed by a moving silhouette or occlusion boundary (e.g. vignetting at an aperture
edge). NSQ v1 does **not** capture these (they are zero); see :ref:`nsq_limitations_and_roadmap`.
